"""One client for three different model backends.

The pipeline talks to three kinds of model service, each with its own wire
protocol:

    provider     backend                      used for
    -----------  ---------------------------  --------------------------------
    "ollama"     local Ollama on :11434       generating samples
    "passau"     OpenAI-compatible vLLM API   judging samples, LLM detection,
                                              text embeddings for the ML arm
    "anthropic"  Anthropic Messages API       optional second-vendor detector

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

import json
import re
from dataclasses import dataclass, field

import requests

import config
from common import load_anthropic_key, load_judgment_credentials

# Providers understood by `chat_json`. Exposed as a constant so the web layer
# can validate a browser-supplied provider name against it rather than
# trusting the request body.
PROVIDERS = ("ollama", "passau", "anthropic")


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
    if model in config.ANTHROPIC_MODELS:
        return "anthropic"
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
        meta={"done_reason": body.get("done_reason", "")},
    )


# ---------------------------------------------------------------------------
# Backend: Passau (OpenAI-compatible vLLM)
# ---------------------------------------------------------------------------

def _chat_passau(model: str, system: str, user: str, schema: dict,
                 max_tokens: int) -> ChatResult:
    """Call the OpenAI-compatible chat endpoint with json_schema decoding.

    The endpoint is vLLM behind an OpenAI-compatible proxy, so it accepts the
    standard `response_format: {"type": "json_schema", ...}` and enforces the
    schema with guided decoding.
    """
    creds = load_judgment_credentials()

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
            f"Could not reach the judgment API at {creds['base_url']}: {exc}"
        ) from exc

    if not response.ok:
        raise ProviderError(
            f"Judgment API returned HTTP {response.status_code}: "
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
# Public entry point
# ---------------------------------------------------------------------------

def chat_json(provider: str, model: str, system: str, user: str,
              schema: dict, max_tokens: int = 4000) -> ChatResult:
    """Send one prompt to one model and get a schema-conforming object back.

    This is the only function the rest of the pipeline calls. Every stage names
    a provider and a model; nothing outside this module knows how the three
    backends differ.

    Args:
        provider: one of `PROVIDERS`.
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
    if provider == "ollama":
        return _chat_ollama(model, system, user, schema, max_tokens)
    if provider == "passau":
        return _chat_passau(model, system, user, schema, max_tokens)
    if provider == "anthropic":
        return _chat_anthropic(model, system, user, schema, max_tokens)
    raise ProviderError(
        f"Unknown provider {provider!r}; expected one of {PROVIDERS}."
    )
