"""One client for three different model backends.

The pipeline talks to three kinds of model service, each with its own wire
protocol:

    provider     backend                      used for
    -----------  ---------------------------  --------------------------------
    "ollama"     local Ollama on :11434       generating samples
    "passau"     OpenAI-compatible vLLM API   judging samples, LLM detection,
                                              text embeddings for the ML arm
    "anthropic"  Anthropic Messages API       optional second-vendor detector
    "openai"     OpenAI Chat Completions      optional third-vendor detector
    "systemone"  TypeSafe /v1/systemone, and  judging samples with a typed,
                 the local `laya` package     non-generative model

THE LAST ONE IS NOT A CHAT BACKEND
    `chat_json` covers the first four, because all four take a system prompt
    and a user prompt and return text constrained to a JSON schema. System One
    models do not: they take a `state` and a set of named typed questions, and
    return a probability distribution per question. There is no prompt and no
    generated text to parse.

    So they get their own entry point, `decide_typed`, rather than being
    disguised as chat models. A caller that wants a verdict from Jev or Laya
    asks a different question in a different shape and gets calibrated
    probabilities back -- which is the point of using them at all, and would be
    thrown away by pretending the call was a chat completion.

WHY WRAP THEM AT ALL
    Every stage of the pipeline wants the same thing: send a system prompt and
    a user prompt, get back an object matching a JSON schema. Without a wrapper
    each stage would carry three branches and three sets of error handling, and
    adding a fourth backend would mean editing every stage. With it, a stage
    names a provider and a model and gets a `ChatResult` back.

STRUCTURED OUTPUT IS NOT OPTIONAL HERE
    Every call in this project asks for a verdict or a code pair, never prose.
    All three backends can constrain generation to a JSON schema -- Ollama and
    vLLM through grammar-guided decoding, Anthropic through its parse helper --
    so the pipeline never has to parse free-form text or strip markdown fences
    out of a model's answer.
"""

from __future__ import annotations

import importlib.util
import json
import re
import warnings
from dataclasses import dataclass, field

import requests

import config
import token_budgets
from common import (
    custom_endpoints,
    endpoint_credentials,
    load_anthropic_key,
    load_judgment_credentials,
    load_openai_key,
    load_typesafe_key,
)

# Providers understood by `chat_json`. Exposed as a constant so the web layer
# can validate a browser-supplied provider name against it rather than
# trusting the request body.
#
# This is the *shipped* set. An installation can add more by putting a url and
# a key in api-keys.txt under a new name -- see `all_providers()` -- which is
# why nothing should compare against this tuple directly when it means "is
# this provider callable".
PROVIDERS = ("ollama", "passau", "anthropic", "openai", "systemone")


def all_providers() -> tuple[str, ...]:
    """Every provider callable here: the shipped four plus added endpoints.

    Read at call time rather than cached, because a credential added on the
    Models page must take effect without restarting the server.
    """
    return PROVIDERS + tuple(sorted(custom_endpoints()))


def provider_for_model(model: str) -> str:
    """Work out which backend serves a given model id.

    Models from all three backends now appear together in the same dropdowns --
    a judge or an LLM detector may be a Passau-hosted Qwen, an Anthropic Claude,
    or a local Ollama tag. Rather than make every caller carry a provider
    alongside every model name, the mapping is derived from the model id once,
    here.

    Resolution order matters: the explicit registries in config.py win, and the
    shape-based guess is only a fallback for a model that has been pulled
    locally but not listed in config (Ollama tags carry a `name:tag` colon,
    which neither remote registry uses).

    Args:
        model: a model identifier as it appears in a dropdown.

    Returns:
        One of `PROVIDERS`.
    """
    # An installation's own registry wins, because it is an explicit
    # declaration by whoever set this up and beats inference from the shape of
    # an id. It is also the only way a self-hosted copy of a model whose name
    # looks like someone else's can be routed correctly.
    try:
        from model_registry import provider_of

        declared = provider_of(model)
        if declared:
            return declared
    except Exception:                                        # noqa: BLE001
        # A broken overlay must not break provider resolution for the
        # shipped models.
        pass

    if model in config.ANTHROPIC_MODELS:
        return "anthropic"
    if model in config.OPENAI_MODELS:
        return "openai"
    if model in config.SYSTEMONE_MODELS:
        return "systemone"
    if model in config.JUDGMENT_MODELS:
        return "passau"
    if ":" in model:
        return "ollama"
    # Unknown and unshaped: the remote judgment endpoint is the only backend
    # whose model list can grow without a code change, so it is the safer
    # default than guessing at a local tag that probably is not pulled.
    return "passau"


@dataclass
class ChatResult:
    """One model response, normalised across all three backends.

    Attributes:
        data: the parsed JSON object the model returned, already validated
            against the requested schema by the backend.
        raw_text: the unparsed response body. Kept for the run log, because
            when a call fails the raw text is the only thing that explains why.
        input_tokens: prompt tokens billed/consumed, 0 if the backend does not
            report them.
        output_tokens: completion tokens, 0 if not reported.
        model: the model id that actually answered.
    """

    data: dict
    raw_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    # Per-call metadata a caller may want to log (finish reason, timings).
    meta: dict = field(default_factory=dict)


class ProviderError(RuntimeError):
    """A backend call failed or returned something unusable.

    A dedicated exception type so callers can distinguish a *transport or
    model* failure -- worth retrying or reporting per sample -- from a bug in
    the pipeline, which should propagate.
    """


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
# Guided decoding usually returns exactly one JSON object. "Usually" is not
# "always": a model may still wrap it in a markdown fence, or prepend a
# sentence, particularly the smaller local ones. These helpers recover the
# object rather than failing the whole sample over a stray backtick.

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Parse the first JSON object in `text`.

    Tries three strategies in increasing order of desperation: parse the whole
    string, strip a markdown fence and parse that, then scan for the first
    balanced `{...}` span and parse it.

    Args:
        text: a model's raw response.

    Returns:
        The parsed object.

    Raises:
        ProviderError: if no strategy yields valid JSON. The message includes a
            truncated copy of the text, which is what makes the run log useful
            when a model misbehaves.
    """
    stripped = text.strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = _FENCE_RE.search(stripped)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Balanced-brace scan. A plain regex cannot match nested objects, so this
    # walks the string counting depth, ignoring braces inside string literals.
    start = stripped.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[start:index + 1])
                    except json.JSONDecodeError:
                        break

    # Last resort: the response is a valid JSON *prefix* that was cut off.
    #
    # A repaired object may be MISSING FIELDS the schema marked required -- the
    # truncation is precisely what removed them. That is acceptable for a short
    # verdict, where the lost field is usually the trailing `reasoning` and
    # nothing is scored on it. It is NOT acceptable for generation, where a
    # response truncated after `vulnerable_code` yields a pair with no safe
    # twin. Callers that need particular fields must check for them; the
    # `__repaired__` marker below is how they can tell a salvaged object from
    # a complete one.
    repaired = repair_truncated_json(stripped)
    if repaired is not None:
        repaired["__repaired__"] = True
        return repaired

    raise ProviderError(
        f"No parseable JSON object in model response: {stripped[:400]!r}"
    )


def repair_truncated_json(text: str) -> dict | None:
    """Recover an object from a response that was cut off mid-write.

    WHY THIS IS NEEDED
        Guided decoding constrains *which* tokens may follow, but it does not
        stop a model emitting its end-of-turn token early. Gemma does this
        routinely: it produces a perfectly good object up to the last field and
        then stops, leaving the closing brace unwritten. Without repair, seven
        of eight judgments in one run were discarded for want of a `}` -- and
        because a failed judgment drops the sample, that alone cost most of the
        dataset.

        The salvaged fields are the ones that matter: label, CWE, count and
        confidence all precede the truncation point. Only `reasoning` is
        typically lost, and nothing is scored on it. Every caller reads the
        object with `.get(..., default)`, so a missing field degrades to a
        default rather than raising.

    HOW
        Candidate cut points are the positions of `,` and `}` in the fragment,
        tried from the end backwards. At each one the text is truncated, any
        unclosed arrays and objects are closed, and the result is parsed. The
        first candidate that parses to a dict wins, so the repair keeps as much
        of the response as possible.

    Args:
        text: the model's raw response.

    Returns:
        The recovered object, or None if nothing parseable can be salvaged.
    """
    start = text.find("{")
    if start == -1:
        return None
    fragment = text[start:]

    # Cut only at structural punctuation, so the scan is linear in the number
    # of fields rather than in the number of characters.
    cuts = [i for i, char in enumerate(fragment) if char in ",}"]
    cuts.append(len(fragment))

    for cut in reversed(cuts):
        candidate = fragment[:cut].rstrip().rstrip(",")
        if not candidate:
            continue

        # Walk the candidate to learn what is still open. Braces inside string
        # literals must not count, so string state is tracked.
        object_depth = array_depth = 0
        in_string = escaped = False
        for char in candidate:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                object_depth += 1
            elif char == "}":
                object_depth -= 1
            elif char == "[":
                array_depth += 1
            elif char == "]":
                array_depth -= 1

        # A cut landing inside a string literal cannot be closed safely --
        # appending a quote would invent content. Try an earlier cut instead.
        if in_string or object_depth < 0 or array_depth < 0:
            continue

        closed = candidate + ("]" * array_depth) + ("}" * object_depth)
        try:
            parsed = json.loads(closed)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None


# ---------------------------------------------------------------------------
# Backend: Ollama (local)
# ---------------------------------------------------------------------------

def list_ollama_models() -> list[str]:
    """Return the model tags currently pulled into the local Ollama server.

    Returns an empty list if the server is not running, rather than raising:
    the web interface calls this to populate a dropdown, and a stopped Ollama
    should grey out the generator options, not break the page.
    """
    try:
        response = requests.get(
            f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5
        )
        response.raise_for_status()
        return sorted(m["name"] for m in response.json().get("models", []))
    except Exception:  # noqa: BLE001 - unreachable server is an expected state
        return []


def ollama_available() -> bool:
    """True if the local Ollama server answers. Used for UI status only."""
    try:
        return requests.get(
            f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5
        ).ok
    except Exception:  # noqa: BLE001
        return False


def _chat_ollama(model: str, system: str, user: str, schema: dict,
                 max_tokens: int) -> ChatResult:
    """Call the local Ollama chat endpoint with grammar-guided JSON output.

    Ollama takes the JSON schema directly in a top-level `format` field, which
    constrains token sampling so the response cannot be malformed. `stream` is
    off because the pipeline wants a whole object, not incremental text.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": schema,
        "options": {
            "num_predict": max_tokens,
            # Some determinism helps reproducibility without collapsing the
            # variety of programs the generator invents.
            "temperature": 0.7,
            # A generation call must fit two whole programs plus the prompt;
            # the 2048-token default context would silently truncate them.
            "num_ctx": 8192,
            # Same whitespace-loop guard as the vLLM path; `format` here is
            # grammar-guided too, so the same stall is possible.
            "stop": token_budgets.stop_sequences(),
        },
    }

    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=config.OLLAMA_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise ProviderError(
            f"Could not reach Ollama at {config.OLLAMA_BASE_URL}: {exc}\n"
            "Is the Ollama server running? Try: ollama serve"
        ) from exc

    if not response.ok:
        raise ProviderError(
            f"Ollama returned HTTP {response.status_code}: {response.text[:300]}"
        )

    body = response.json()
    text = body.get("message", {}).get("content", "")

    return ChatResult(
        data=extract_json(text),
        raw_text=text,
        # Ollama reports counts under these names; absent on older versions.
        input_tokens=body.get("prompt_eval_count", 0),
        output_tokens=body.get("eval_count", 0),
        model=model,
        # `done_reason` is Ollama's spelling; it is also mirrored under the
        # name the other backends use, so a caller reasoning about why
        # generation ended does not need a per-provider branch.
        meta={
            "done_reason": body.get("done_reason", ""),
            "finish_reason": body.get("done_reason", ""),
        },
    )


# ---------------------------------------------------------------------------
# Backend: Passau (OpenAI-compatible vLLM)
# ---------------------------------------------------------------------------

def _chat_passau(model: str, system: str, user: str, schema: dict,
                 max_tokens: int) -> ChatResult:
    """Call the shipped judgment endpoint with json_schema decoding.

    A thin alias for `_chat_openai_compatible` against the `judgment` entry,
    kept because "passau" is the provider name written into every existing
    run's `config_snapshot` and into models.local.json.
    """
    return _chat_openai_compatible(
        "judgment", model, system, user, schema, max_tokens
    )


def _chat_openai_compatible(endpoint: str, model: str, system: str, user: str,
                            schema: dict, max_tokens: int) -> ChatResult:
    """Call any OpenAI-compatible chat endpoint with json_schema decoding.

    The Passau endpoint is vLLM behind an OpenAI-compatible proxy, so it
    accepts the standard `response_format: {"type": "json_schema", ...}` and
    enforces the schema with guided decoding. Everything that speaks the same
    protocol can therefore use the same client, which is what makes a
    user-added endpoint callable without new code -- see
    `common.custom_endpoints()`.

    Args:
        endpoint: the credential entry name to call, e.g. "judgment".
        model: the model id as that endpoint names it.

    Raises:
        ProviderError: on an unreachable endpoint or a non-2xx reply. The
            endpoint name is in the message, because with several configured
            "the API returned 401" is not enough to act on.
    """
    creds = endpoint_credentials(endpoint)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        # Low but non-zero: verdicts should be stable across repeats, while
        # leaving the model enough freedom not to lock onto one phrasing.
        "temperature": 0.2,
        # Cuts a grammar-guided whitespace loop short. Guided decoding permits
        # unlimited whitespace between structural tokens, so a stalled model
        # can emit newlines until the budget is gone -- see token_budgets.
        "stop": token_budgets.stop_sequences(),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": schema,
            },
        },
    }

    try:
        response = requests.post(
            f"{creds['base_url']}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {creds['api_key']}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )
    except requests.RequestException as exc:
        raise ProviderError(
            f"Could not reach the {endpoint} API at {creds['base_url']}: "
            f"{exc}"
        ) from exc

    if not response.ok:
        raise ProviderError(
            f"{endpoint} API returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    body = response.json()
    choice = body["choices"][0]
    text = choice["message"]["content"] or ""
    usage = body.get("usage", {})

    return ChatResult(
        data=extract_json(text),
        raw_text=text,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        model=body.get("model", model),
        meta={"finish_reason": choice.get("finish_reason", "")},
    )


# ---------------------------------------------------------------------------
# Backend: OpenAI
# ---------------------------------------------------------------------------
# WHY THIS IS NOT JUST ANOTHER OPENAI-COMPATIBLE ENDPOINT
#   api.openai.com speaks the protocol `_chat_openai_compatible` implements,
#   and registering it as an endpoint in api-keys.txt very nearly works. It
#   fails on two parameters -- measured against the live API, not assumed:
#
#     max_tokens      Every GPT-5 and GPT-6 model refuses it outright:
#                     "Unsupported parameter: 'max_tokens' is not supported
#                     with this model. Use 'max_completion_tokens' instead."
#                     The vLLM endpoint and the GPT-4 generation accept both,
#                     so `max_completion_tokens` goes to all of them and there
#                     is no per-model branch to keep correct.
#
#     temperature     Refused by the same models unless it is the default:
#                     "Unsupported value: 'temperature' does not support 0.2
#                     with this model. Only the default (1) value is
#                     supported." OpenAI's own guidance is to drop
#                     temperature, top_p and the logprobs fields whenever
#                     reasoning effort is anything but `none`.
#
#   So this backend sends no temperature at all. That gives up the deliberate
#   0.2 the other backends use for verdict stability -- see
#   `_chat_openai_compatible` -- and it is not a free choice: a judge at
#   temperature 1 is a slightly noisier judge, and two runs of it agree a
#   little less often. The alternative is a hardcoded list of which ids still
#   accept a temperature, which would be wrong the next time OpenAI ships a
#   family, and wrong silently.
#
#   `strict: true` IS sent, unlike on the vLLM path, because OpenAI then
#   enforces the schema by construction. All of this project's schemas were
#   checked against it: VERDICT_SCHEMA in judge.py and in
#   detectors/llm_tools.py, including the `minimum`/`maximum` bounds on
#   `confidence_percent` and the nullable `predicted_cwe`.


def _chat_openai(model: str, system: str, user: str, schema: dict,
                 max_tokens: int) -> ChatResult:
    """Call the OpenAI Chat Completions API, constrained to `schema`.

    Args:
        model: an OpenAI model id, e.g. "gpt-6.1-sol".
        max_tokens: sent as `max_completion_tokens`. On a reasoning model this
            budget covers the thinking as well as the answer, and the thinking
            is charged and counted but never returned -- so a budget large
            enough for the JSON alone can be spent entirely on reasoning,
            leaving an empty reply. That case is reported, not parsed.

    Raises:
        ProviderError: with no key configured, on an unreachable endpoint, on
            a non-2xx reply, or when the budget ran out before any answer was
            written.
    """
    key = load_openai_key()
    if not key:
        raise ProviderError(
            "No OpenAI key found. Add it to api-keys.txt as:\n"
            'openai.key = "sk-..."'
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # See the note above for why this is not `max_tokens`, and why no
        # temperature is sent.
        "max_completion_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": schema,
                "strict": True,
            },
        },
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
        )
    except requests.RequestException as exc:
        raise ProviderError(f"Could not reach the OpenAI API: {exc}") from exc

    if not response.ok:
        raise ProviderError(
            f"OpenAI API returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    body = response.json()
    choice = body["choices"][0]
    text = choice["message"]["content"] or ""
    usage = body.get("usage", {})
    finish = choice.get("finish_reason", "")
    # Counted separately because it is charged separately, and because it
    # explains an otherwise baffling bill: a one-line verdict from a reasoning
    # model can cost several hundred tokens that never appear in the output.
    reasoning_tokens = (
        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    )

    if not text.strip():
        # Told apart from a parse failure, because the remedy differs: a
        # truncated answer needs a larger budget or a model that thinks less,
        # not a better prompt.
        detail = (
            f"spent its whole {max_tokens}-token budget on reasoning "
            f"({reasoning_tokens} reasoning tokens) and returned no answer"
            if finish == "length"
            else f"returned an empty reply (finish_reason={finish!r})"
        )
        raise ProviderError(f"{model} {detail}")

    return ChatResult(
        data=extract_json(text),
        raw_text=text,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        model=body.get("model", model),
        meta={"finish_reason": finish, "reasoning_tokens": reasoning_tokens},
    )


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed a batch of source files using the endpoint's embedding model.

    Used by the embedding-based ML detector. Batching matters: one request per
    sample would multiply latency by the dataset size for no benefit, since the
    endpoint accepts a list of inputs in a single call.

    Args:
        texts: the strings to embed, in order.
        model: embedding model id; defaults to `config.EMBEDDING_MODEL`.

    Returns:
        One vector per input, in the same order as `texts`.

    Raises:
        ProviderError: if the endpoint is unreachable or returns an error.
    """
    creds = load_judgment_credentials()
    model = model or config.EMBEDDING_MODEL

    try:
        response = requests.post(
            f"{creds['base_url']}/v1/embeddings",
            headers={
                "Authorization": f"Bearer {creds['api_key']}",
                "Content-Type": "application/json",
            },
            json={"model": model, "input": texts},
            timeout=180,
        )
    except requests.RequestException as exc:
        raise ProviderError(f"Embedding request failed: {exc}") from exc

    if not response.ok:
        raise ProviderError(
            f"Embedding API returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    body = response.json()
    # The response may come back out of order, so entries are sorted by their
    # `index` field rather than trusted to match the request order.
    entries = sorted(body["data"], key=lambda item: item.get("index", 0))
    return [entry["embedding"] for entry in entries]


# ---------------------------------------------------------------------------
# Backend: Anthropic
# ---------------------------------------------------------------------------

def _chat_anthropic(model: str, system: str, user: str, schema: dict,
                    max_tokens: int) -> ChatResult:
    """Call the Anthropic Messages API, asking for schema-shaped JSON.

    Imported lazily so that the whole pipeline still runs on a machine with no
    Anthropic key and no `anthropic` package installed -- this backend is an
    optional extra, not a dependency of the core experiment.
    """
    key = load_anthropic_key()
    if not key:
        raise ProviderError(
            "No Anthropic key found. Add it to api-key-llm.txt as:\n"
            'anthropic api key:\n"sk-ant-..."'
        )

    try:
        import anthropic
    except ImportError as exc:
        raise ProviderError(
            "The 'anthropic' package is not installed: pip install anthropic"
        ) from exc

    client = anthropic.Anthropic(api_key=key)

    # The schema is appended to the system prompt rather than passed as a tool.
    # It keeps this backend's call shape identical to the other two, which is
    # the whole point of the wrapper.
    system_with_schema = (
        f"{system}\n\n"
        "Reply with a single JSON object and nothing else. It must conform to "
        f"this JSON schema:\n{json.dumps(schema, indent=2)}"
    )

    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_with_schema,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as exc:  # noqa: BLE001 - SDK raises many concrete types
        raise ProviderError(f"Anthropic API call failed: {exc}") from exc

    text = "".join(
        block.text for block in message.content if block.type == "text"
    )

    return ChatResult(
        data=extract_json(text),
        raw_text=text,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        model=model,
        meta={"stop_reason": message.stop_reason},
    )


# ---------------------------------------------------------------------------
# System One: typed decisions instead of chat
# ---------------------------------------------------------------------------
# Wire format, identical for both models (TypeSafe document it; laya-serve
# implements the same shape deliberately):
#
#   POST /v1/systemone
#   {"model": ..., "state": <the text being judged>,
#    "questions": {"<id>": {"type": "choice"|"score"|"noul", ...}}}
#
#   -> {"model": ..., "usage": {"input_tokens": n, "output_tokens": n},
#       "answers": {"<id>": {"type": "choice", "choice": ...,
#                            "confidence": 0-1, "probabilities": {...}}}}
#
# A `choice` question carries `criteria`: a map of option name to description,
# up to 255 options. A `score` question carries `criteria` as an ordered list
# of up to 10 level descriptions, the answer's `score` being the
# probability-weighted expected level. A `noul` question needs no criteria and
# answers with a single probability that the statement is true.


@dataclass
class TypedResult:
    """One System One response, normalised across the hosted and local models.

    Attributes:
        answers: question id -> answer object, in the documented shape above.
            Left as returned rather than flattened, because which fields are
            present depends on the question type and the caller knows which it
            asked.
        input_tokens: prompt tokens reported by the backend, 0 if unreported.
        output_tokens: as above. Jev bills input only; Laya is local and free.
        model: the model version that actually answered, which for an alias
            like `jev-latest` is more specific than what was requested and is
            therefore what belongs in the run record.
    """

    answers: dict
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


def systemone_available(model: str) -> tuple[bool, str]:
    """Report whether a System One model can be called, and why not if it cannot.

    Returns a reason rather than raising, because this is called to decide what
    to show in a dropdown. An unavailable model is offered but disabled, with
    the reason beside it -- the alternative is a run that starts and then fails
    on the first judgement call.

    Args:
        model: one of `config.SYSTEMONE_MODELS`.

    Returns:
        `(available, reason)`. `reason` is empty when available.
    """
    if model not in config.SYSTEMONE_MODELS:
        return False, f"{model} is not a System One model"

    if model in config.LAYA_CHECKPOINTS:
        if importlib.util.find_spec("laya") is None:
            return False, "the 'laya' package is not installed (pip install laya)"
        return True, ""

    if load_typesafe_key() is None:
        return False, "no TypeSafe key in api-key-typesafe.txt"
    return True, ""


def _decide_jev(model: str, state: str, questions: dict,
                timeout: int) -> TypedResult:
    """Call the hosted TypeSafe endpoint."""
    key = load_typesafe_key()
    if not key:
        raise ProviderError(
            "No TypeSafe key found. Add it to api-key-typesafe.txt as:\n"
            '"<your key>"\n'
            "Jev is in early access; a key comes from the waitlist at "
            "typesafe.ai."
        )

    try:
        response = requests.post(
            f"{config.TYPESAFE_BASE_URL}/v1/systemone",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={"model": model, "state": state, "questions": questions},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise ProviderError(f"TypeSafe request failed: {exc}") from exc

    if response.status_code != 200:
        # The body carries the reason -- an unknown question type, a state over
        # the context limit, an expired key -- and is far more useful than the
        # status code alone.
        raise ProviderError(
            f"TypeSafe HTTP {response.status_code}: {response.text[:300]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise ProviderError(
            f"TypeSafe returned non-JSON: {response.text[:200]}"
        ) from exc

    answers = payload.get("answers")
    if not isinstance(answers, dict) or not answers:
        raise ProviderError(f"TypeSafe returned no answers: {str(payload)[:200]}")

    usage = payload.get("usage") or {}
    return TypedResult(
        answers=answers,
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        model=payload.get("model", model),
    )


# One loaded Laya checkpoint per model id, kept for the life of the process.
#
# The checkpoint is a 0.4B encoder: loading it takes seconds and a few hundred
# megabytes, and the judge calls it once per sample. Reloading per call would
# dominate the stage's runtime -- the same mistake that made the ML arm's
# availability check re-read an 8.6MB pickle on every page load.
_LAYA_CACHE: dict[str, object] = {}

# Warnings the checkpoint emitted while loading, per model id. Captured rather
# than left on stderr: these are statements about how far its confidence can
# be trusted, and a web run never shows stderr to anyone.
_LAYA_NOTES: dict[str, list[str]] = {}


def _laya_agent(model: str):
    """Load, and then reuse, the local Laya checkpoint for `model`."""
    if model in _LAYA_CACHE:
        return _LAYA_CACHE[model]

    try:
        import laya
    except ImportError as exc:
        raise ProviderError(
            "The 'laya' package is not installed: pip install laya"
        ) from exc

    checkpoint = config.LAYA_CHECKPOINTS[model]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            agent = laya.load(checkpoint)
    except Exception as exc:  # noqa: BLE001 - torch/HF raise many types
        raise ProviderError(
            f"Could not load Laya checkpoint {checkpoint}: {exc}"
        ) from exc

    _LAYA_NOTES[model] = [str(item.message) for item in caught]
    _LAYA_CACHE[model] = agent
    return agent


def systemone_load_notes(model: str) -> list[str]:
    """Warnings the model's checkpoint emitted when it was loaded.

    Empty before the first call, because the checkpoint is loaded lazily, and
    empty for the hosted model, which has no local checkpoint to complain.

    Returns:
        The warning texts, for a caller that will put them in the run log.
    """
    return list(_LAYA_NOTES.get(model, []))


def _decide_laya(model: str, state: str, questions: dict) -> TypedResult:
    """Run the local Laya checkpoint over one state.

    Laya's Python API takes the same `(state, questions)` pair as the HTTP
    protocol -- its own server implements TypeSafe's shape on top of this --
    so the answers come back in the documented form and need no translation.
    Only the container is normalised, since a local model reports no usage.
    """
    agent = _laya_agent(model)

    try:
        result = agent.predict(state, questions)
    except Exception as exc:  # noqa: BLE001 - torch raises many types
        raise ProviderError(f"Laya inference failed: {exc}") from exc

    # The package returns either a plain dict or an object exposing one; both
    # are accepted so a minor release that changes the wrapper does not break
    # the judge.
    if hasattr(result, "model_dump"):
        payload = result.model_dump()
    elif isinstance(result, dict):
        payload = result
    else:
        payload = {"answers": getattr(result, "answers", None)}

    answers = payload.get("answers", payload)
    if not isinstance(answers, dict) or not answers:
        raise ProviderError(f"Laya returned no answers: {str(payload)[:200]}")

    return TypedResult(answers=answers, model=payload.get("model", model))


def decide_typed(model: str, state: str, questions: dict,
                 timeout: int = 90) -> TypedResult:
    """Ask a System One model a set of typed questions about one state.

    Args:
        model: a key of `config.SYSTEMONE_MODELS`.
        state: the content every question refers to -- here, one program.
        questions: `{id: {"type": ..., ...}}` in the shape documented above.
            At least one is required by the protocol.
        timeout: seconds to wait on the hosted endpoint. Ignored for Laya,
            which runs locally and cannot hang on a network.

    Returns:
        A `TypedResult`.

    Raises:
        ProviderError: on an unknown model, a missing key or package, a
            transport failure, or a response carrying no answers.
    """
    if not questions:
        raise ProviderError("decide_typed needs at least one question")

    if model in config.LAYA_CHECKPOINTS:
        return _decide_laya(model, state, questions)
    if model in config.SYSTEMONE_MODELS:
        return _decide_jev(model, state, questions, timeout)
    raise ProviderError(f"Unknown System One model: {model}")


def answer_confidence(answer: dict) -> float:
    """How much probability the model put on the answer it gave, 0.0-1.0.

    WHICH FIELD, AND WHY IT MATTERS
        A System One answer can carry two different confidence numbers and
        they are not interchangeable:

            answer_confidence  max(p) -- the probability mass on the reported
                               answer. Laya's own documentation calls this
                               "the calibrated one, on every question type,
                               so a caller can gate across types on a single
                               number".
            confidence         normalised Shannon entropy, 1 - H(p)/log(k):
                               how concentrated the WHOLE distribution is.
                               Its own docstring says it is "not calibrated:
                               it is not what temperature scaling fits and
                               not what the reported ECE measures".

        The chat models report "how certain are you, 0-100", which is a
        statement about the answer given -- so max(p) is the comparable
        quantity and the entropy score is not. The difference decides runs:
        on a real sample the label answer was P(not_vulnerable)=0.69, which
        max(p) reports as 0.69 and the entropy score as 0.11. Against
        JUDGE_MIN_CONFIDENCE = 0.60 the first is admitted and the second is
        dropped as "low confidence" -- for a model that was not unsure at all.

        So the preference order is: the backend's own calibrated field, then
        max(p) computed from the distribution, and only then `confidence`.
        Which branch runs depends on the backend, and both were checked:
        Laya returns `answer_confidence` and takes the first; Jev's OpenAPI
        schema has no such field, so it takes the second, which for a choice
        answer is the same quantity. Neither reaches the third, and that is
        the point of the order -- Jev documents its `confidence` as based on
        how the probability is spread, which is the entropy-style measure
        again, not the mass on the answer.

    Used by both stages that consume a typed answer: judge.py, where it gates
    admission, and detectors/llm_tools.py, where it becomes the detector's
    reported confidence and is scored like any other tool's.

    Args:
        answer: one answer object from a System One response.

    Returns:
        A probability in 0.0-1.0, or 0.0 if the answer carries nothing usable.
    """
    value = answer.get("answer_confidence")
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))

    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict) and probabilities:
        numeric = [v for v in probabilities.values()
                   if isinstance(v, (int, float))]
        if numeric:
            return max(0.0, min(1.0, float(max(numeric))))

    value = answer.get("confidence")
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return 0.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chat_json(provider: str, model: str, system: str, user: str,
              schema: dict, max_tokens: int = 4000) -> ChatResult:
    """Send one prompt to one model and get a schema-conforming object back.

    This is the only function the rest of the pipeline calls. Every stage names
    a provider and a model; nothing outside this module knows how the three
    backends differ.

    Args:
        provider: one of `all_providers()` -- a shipped backend or an
            endpoint named in api-keys.txt.
        model: backend-specific model identifier.
        system: the system prompt (rules that hold for every request).
        user: the user prompt (what varies per request).
        schema: a JSON Schema object the response must conform to.
        max_tokens: cap on the response length. Generation needs a high value
            because a response carries two complete programs; a verdict needs
            far less.

    Returns:
        A `ChatResult` whose `.data` already matches `schema`.

    Raises:
        ProviderError: on an unknown provider, an unreachable backend, or a
            response that cannot be parsed as JSON.
    """
    # A registered per-model budget overrides the stage's default, because one
    # constant cannot fit both kinds of model: a verdict costs gemma4-31b-it
    # about 610 output tokens, while qwen-agentworld-35b-a3b spends 6,200-8,000
    # reasoning before it writes any JSON and returned an empty string on 52%
    # of its calls at the shared 4,000 cap.
    wire_name, budget = _registry_overrides(model)
    max_tokens = budget or max_tokens

    # Every backend goes through the budget guard, including user-added
    # endpoints, because a padding loop is a property of the model rather than
    # of the protocol -- see token_budgets. `sent` equals `max_tokens` unless
    # this model has already been caught padding at this call size.
    #
    # Keyed on the registry id, not the wire name, so the learned state follows
    # the model the user named and survives the endpoint renaming it.
    sent = token_budgets.budget_for(model, max_tokens)
    result = _dispatch(provider, wire_name, system, user, schema, sent)
    token_budgets.observe(model, max_tokens, sent, result)
    return result


def _registry_overrides(model: str) -> tuple[str, int | None]:
    """What the installation's registry says to send for this model.

    Returns `(wire_name, max_tokens)`. The id is a local handle -- it reaches
    filenames, report rows and the budget keys above -- and it may differ from
    what the endpoint calls the model: aggregators publish namespaced ids such
    as `meta/llama-4-scout-17b-16e-instruct`, which cannot be an id here
    because it would reach a path. Only the outgoing call is rewritten;
    everything the user sees stays the id.

    `max_tokens` is None unless the entry sets one.

    Defensive for the same reason as `provider_for_model`: a broken or
    hand-edited overlay must not stop the shipped models from being called.
    """
    try:
        import model_registry

        return (model_registry.api_model_for(model) or model,
                model_registry.max_tokens_for(model))
    except Exception:                                        # noqa: BLE001
        return model, None


def _dispatch(provider: str, model: str, system: str, user: str,
              schema: dict, max_tokens: int) -> ChatResult:
    """Route one call to its backend. Split out so `chat_json` can wrap it."""
    if provider == "ollama":
        return _chat_ollama(model, system, user, schema, max_tokens)
    if provider == "passau":
        return _chat_passau(model, system, user, schema, max_tokens)
    if provider == "anthropic":
        return _chat_anthropic(model, system, user, schema, max_tokens)
    if provider == "openai":
        return _chat_openai(model, system, user, schema, max_tokens)
    if provider == "systemone":
        # In PROVIDERS, but deliberately not reachable from here: these models
        # generate no text, so there is no completion to constrain to a schema.
        # Callers use decide_typed() and ask typed questions instead.
        raise ProviderError(
            f"{model} is a System One model and cannot answer a chat prompt. "
            "Use decide_typed() with typed questions."
        )
    # Anything else must be an endpoint the installation added to api-keys.txt.
    # Checked last so a user-added entry can never shadow a shipped provider,
    # whatever it is called.
    if provider in custom_endpoints():
        return _chat_openai_compatible(
            provider, model, system, user, schema, max_tokens
        )
    raise ProviderError(
        f"Unknown provider {provider!r}; expected one of "
        f"{', '.join(all_providers())}. A provider added to api-keys.txt "
        "needs both a url and a key before it can be called."
    )
