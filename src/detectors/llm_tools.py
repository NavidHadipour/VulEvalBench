"""Arm 3 -- large-language-model detectors.

A model is shown one program and asked whether it contains a vulnerability.
Structurally this is the same operation the judging stage performs, but the
*role* is completely different and the distinction matters:

    judge.py      decides what goes INTO the benchmark. Its verdicts become
                  part of ground truth and are never scored.
    this module   is a tool UNDER TEST. Its verdicts are scored against that
                  ground truth like Bandit's or the ML models'.

Because of that, the default LLM detector in config.py is deliberately a
different model from the default judges. A model that helped certify a sample
must not also be graded on finding the vulnerability in it, or it is marking
its own homework and its score is meaningless.

TWO KINDS OF MODEL
    A CHAT detector is given the system prompt below and must return a JSON
    object matching VERDICT_SCHEMA. A TYPED detector -- Jev or Laya, the
    System One models -- generates no text at all: it reads the program once
    and answers a fixed set of questions with probability distributions. Both
    produce the same prediction shape, so `evaluate.py` scores them
    identically and a report can hold both without special cases.

    The typed path asks only TWO questions, label and CWE, where the judging
    stage asks three. The third is the vulnerability count, which exists for
    REQUIRE_SINGLE_VULNERABILITY -- an admission rule about what belongs in
    the dataset. A detector is not admitting anything; it is being asked
    "is this vulnerable, and with what". Asking a question whose answer is
    then discarded would only cost latency and, on the hosted model, tokens.

    This matters for interpreting results. The count question is what blocked
    Laya as a judge (it answered "two or more" for 47 of 59 programs and never
    "none", so every safe sample failed the rule). As a detector that
    bottleneck is simply absent, and Laya's measured label behaviour -- high
    precision, poor recall -- is what its score will reflect.

METHODOLOGICAL PRECAUTIONS -- identical to the judging stage
    1. One sample per request, no conversation history, so no verdict can
       influence another and the model cannot balance its answers across the
       set ("about half of these should be vulnerable").
    2. The prompt never states how many samples exist or what fraction are
       vulnerable.
    3. The model sees source only -- no filename, no CWE hint, no mention that
       a safe twin of this program exists elsewhere in the dataset.
    4. A model used as a JUDGE on a run must not also be scored as a detector
       on it. detect.py refuses that overlap rather than warning about it,
       because the resulting number looks valid and is not. This applies to
       the typed models exactly as it does to the chat ones.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import config
from detectors import build_result, make_prediction
from providers import (
    ProviderError,
    answer_confidence,
    chat_json,
    decide_typed,
    provider_for_model,
)

# Requests in flight at once. Kept modest: this is a shared university endpoint
# and the benefit of more parallelism falls off quickly at these dataset sizes.
LLM_CONCURRENCY = 4

# Response budget per verdict.
#
# WHY THIS IS NOT SMALL. A verdict is four short fields, so 1200 tokens looks
# generous -- and it silently broke the arm. Several of these models reason
# before answering, and that reasoning is billed against the same budget:
# qwen36-35b consumes around 1470 output tokens to produce a verdict, so at
# 1200 it was truncated mid-object on every single call. Measured, not guessed:
# the same prompt fails at 1200 and succeeds at 4000.
MAX_RESPONSE_TOKENS = 4000

# Response schema.
#
# `confidence_percent` is a bounded integer rather than a 0.0-1.0 float for the
# same reason as in judge.py: a JSON `number` grammar permits unlimited digits
# after the decimal point, and these models reliably run away with it --
# emitting `0.95000000...` until the token limit truncates the object and the
# call fails. An integer has no fractional part to run away with.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "predicted_label": {
            "type": "string",
            "enum": [config.LABEL_VULNERABLE, config.LABEL_SAFE],
            "description": "Whether this program contains a security vulnerability.",
        },
        "predicted_cwe": {
            "type": ["string", "null"],
            "description": "CWE identifier such as 'CWE-89', or null if not vulnerable.",
        },
        "confidence_percent": {
            "type": "integer", "minimum": 0, "maximum": 100,
            "description": "Confidence in the verdict, 0 to 100.",
        },
        "reasoning": {
            "type": "string",
            "description": "One or two sentences justifying the verdict.",
        },
        # LOCALISATION: A QUOTED SNIPPET, NOT A LINE NUMBER.
        #   evaluate.py scores `predicted_lines`, so a line number is what is
        #   ultimately wanted -- but asking the model for one directly is the
        #   wrong way to get it, for two reasons.
        #
        #   The program in USER_PROMPT carries no line numbers. To ask for a
        #   number the lines would have to be numbered in the prompt, and that
        #   changes the text every LLM detector sees: verdicts, and therefore
        #   every previously recorded LLM metric, would no longer be comparable
        #   with runs made before this field existed. Localisation is not worth
        #   invalidating the detection numbers for.
        #
        #   The alternative -- leaving the code unnumbered and asking the model
        #   to count lines -- measures arithmetic rather than security
        #   reasoning, and language models are poor at it.
        #
        #   So the model quotes the offending line, which is what it is good
        #   at, and `_locate_snippet` finds that text in the source to recover
        #   the number exactly. The code shown to the model is untouched.
        "vulnerable_snippet": {
            "type": ["string", "null"],
            "description": (
                "The single most important line of code where the "
                "vulnerability occurs, copied EXACTLY as it appears in the "
                "program, without the surrounding lines. Null if the program "
                "is not vulnerable."
            ),
        },
    },
    "required": [
        "predicted_label", "predicted_cwe", "confidence_percent", "reasoning",
        "vulnerable_snippet",
    ],
    "additionalProperties": False,
}

# CLOSED-SET CWE PREDICTION.
#   The model is shown the candidate identifiers so its category answer is
#   comparable with ground truth: without the list it might answer "CWE-943"
#   for an SQL injection and be scored wrong on a naming technicality rather
#   than a detection failure.
#
#   The cost is that this narrows the search space and makes the task easier
#   than an open-set setting. It also gives the LLM arm a small advantage over
#   the static tools, which map findings to categories through a fixed rule
#   table and cannot "choose" from a list. Worth stating in a write-up.
#
# The two "do not assume" sentences counter a well-known bias: a model told it
# is reviewing code for a security benchmark expects to find a vulnerability
# and duly reports one, which inflates the false positive rate on the safe half
# of the dataset.
SYSTEM_PROMPT = """\
You are a security code reviewer analysing a single Python program.

Decide whether the program contains a security vulnerability that is reachable
from untrusted input. Judge only the code in front of you: do not assume a
vulnerability is present, and do not assume the code is safe. Many of the
programs you will see are clean.

If it is vulnerable, identify the category using one of these identifiers:
{cwe_list}

Report `predicted_cwe` as null when the program is not vulnerable. Set
`confidence_percent` to a whole number between 0 and 100.

If it is vulnerable, set `vulnerable_snippet` to the single most important
line of code where the flaw occurs, copied exactly as it appears in the
program. Copy one line only, character for character, with no line number, no
quotation marks around it and no commentary. Report it as null when the
program is not vulnerable.
"""

USER_PROMPT = """\
Analyse the following Python program.

```python
{code}
```
"""


# Instructions for the typed models.
#
# These carry the same two anti-bias sentences as SYSTEM_PROMPT. A typed model
# never sees a system prompt -- each question's `instructions` field is all the
# framing it gets -- so the precaution has to be repeated per question rather
# than stated once.
TYPED_LABEL_INSTRUCTIONS = (
    "Decide whether this Python program contains a security vulnerability "
    "that is reachable from untrusted input. Judge only the code in front of "
    "you: do not assume a vulnerability is present, and do not assume the "
    "code is safe. Many of these programs are clean."
)

TYPED_CWE_INSTRUCTIONS = (
    "If this program contains a security vulnerability reachable from "
    "untrusted input, which category best describes it? Answer 'none' if the "
    "program has no such vulnerability."
)


def typed_questions() -> dict:
    """Build the question set a System One detector answers.

    Deliberately NOT imported from judge.py even though the label and CWE
    questions are worded almost identically. Two reasons, and both are about
    keeping the experiment honest rather than about code style:

        1. Nothing under detectors/ may import judge.py. The judge produces
           the ground truth these detectors are scored against; a shared
           module would let a change to the admission criteria silently move
           the thing being measured.
        2. The judge's set has a third question (the count) that serves an
           admission rule a detector has no part in. Copying the set and then
           ignoring an answer would misstate what the detector was asked.

    Returns:
        A `questions` object for `providers.decide_typed`.
    """
    # Option descriptions, not bare identifiers: "CWE-918" means nothing to a
    # model that was never trained on the catalogue, and the description is
    # what a choice question compares against the state.
    criteria = {
        cwe: config.CWE_NAMES.get(cwe, cwe) for cwe in config.CWE_CATEGORIES
    }
    criteria["none"] = (
        "No security vulnerability reachable from untrusted input."
    )

    return {
        "label": {
            "type": "choice",
            "instructions": TYPED_LABEL_INSTRUCTIONS,
            "criteria": {
                config.LABEL_VULNERABLE: (
                    "The program contains a security vulnerability that "
                    "untrusted input can reach."
                ),
                config.LABEL_SAFE: (
                    "The program has no such vulnerability."
                ),
            },
        },
        "cwe": {
            "type": "choice",
            "instructions": TYPED_CWE_INSTRUCTIONS,
            "criteria": criteria,
        },
    }


def _detect_one_typed(model: str, code: str, questions: dict,
                      sample_id: str) -> dict:
    """Ask one System One model about one program, returning a prediction.

    The same retry-then-mark policy as the chat path: a failed call is
    infrastructure failing, not the model judging the file clean, so it is
    recorded as "not vulnerable" to keep the sample in the confusion matrix
    but flagged with `call_failed` so the count reaches the report.

    The confidence is the probability mass on the LABEL answer, which is the
    quantity the chat detectors' `confidence_percent` is also meant to
    express. The CWE answer has its own, usually much lower, confidence; it is
    not mixed in, because the reported number has to mean one thing.
    """
    last_error = ""
    for _ in range(2):
        try:
            result = decide_typed(model, code, questions)
            break
        except ProviderError as exc:
            last_error = str(exc)
    else:
        prediction = make_prediction(
            sample_id, False, None, 0.0,
            f"detector call failed: {last_error}"[:400],
        )
        prediction["call_failed"] = True
        return prediction

    label_answer = result.answers.get("label") or {}
    cwe_answer = result.answers.get("cwe") or {}

    # A response with no label answer did not analyse this program, whatever
    # else it returned. Treated as a failed call for the same reason: scoring
    # it as "clean" would invent a verdict the model never gave.
    if not label_answer.get("choice"):
        prediction = make_prediction(
            sample_id, False, None, 0.0,
            f"{model} returned no label answer; "
            f"answers: {sorted(result.answers)}",
        )
        prediction["call_failed"] = True
        return prediction

    is_vulnerable = label_answer.get("choice") == config.LABEL_VULNERABLE

    # "none" is the sentinel option that lets the CWE question express "no
    # vulnerability"; as a prediction that is an absent category, not a
    # category named "none".
    cwe = cwe_answer.get("choice")
    if cwe == "none" or not is_vulnerable:
        cwe = None

    probabilities = label_answer.get("probabilities") or {}
    reasoning = (
        "[no prose: typed decision model] "
        + ", ".join(f"P({name})={float(value):.2f}"
                    for name, value in sorted(probabilities.items())
                    if isinstance(value, (int, float)))
    )
    if cwe:
        cwe_confidence = answer_confidence(cwe_answer)
        reasoning += f"; cwe={cwe} p={cwe_confidence:.2f}"

    return make_prediction(
        sample_id,
        is_vulnerable,
        cwe,
        answer_confidence(label_answer),
        reasoning,
    )


def _locate_snippet(code: str, snippet: str | None) -> list[int]:
    """Find the 1-based line number(s) a quoted snippet corresponds to.

    The model is asked to copy a line verbatim, and mostly does, but not
    always: it may re-indent it, drop a trailing comma, wrap it in backticks,
    or return two lines when asked for one. So the match is attempted from
    strictest to loosest, and gives up rather than guessing:

        1. the snippet appears verbatim in the source;
        2. a source line equals the snippet once both are stripped;
        3. a source line contains the stripped snippet;
        4. a source line starts with the snippet's first 24 significant
           characters -- enough to pin a statement, short enough to survive a
           truncated or lightly reworded tail.

    Returning [] is a real answer and is recorded as such: it means the model
    named a location that could not be tied to the file, which evaluate.py
    must not count as a correct localisation. The caller tallies these
    separately from "the model named nothing", because a model that quotes
    something unfindable and one that declines to quote at all are different
    failures.

    Args:
        code: the program the model was shown.
        snippet: whatever the model put in `vulnerable_snippet`.

    Returns:
        Matching line numbers, ascending. Several when the snippet spans or
        matches more than one line.
    """
    if not snippet or not snippet.strip():
        return []

    # Models occasionally fence the snippet or prefix it with a line number.
    cleaned = snippet.strip().strip("`").strip()
    if not cleaned:
        return []

    lines = code.splitlines()

    # The looser rules below are capped, and rule (1) is bounded the same way:
    # a snippet matching many lines has not localised anything, and returning
    # all of them would let a detector earn credit by luck -- evaluate.py
    # counts a hit if ANY predicted line is near a real one, so a long list is
    # close to a free pass. Three allows for genuinely repeated statements
    # without opening that door.
    max_ambiguous = 3

    # A floor on length, applied before any rule. "q" is a verbatim substring
    # of almost every program and localises nothing; neither does "x = 1" in a
    # file that repeats it. Four significant characters is the shortest thing
    # that can plausibly name a statement.
    if len(cleaned.replace(" ", "")) < 4:
        return []

    # (1) verbatim, possibly spanning lines. Every occurrence, not just the
    # first: if the text appears three times the model has not said which, and
    # silently taking the first would invent a precision it did not have.
    occurrences = []
    position = code.find(cleaned)
    while position != -1:
        first = code.count("\n", 0, position) + 1
        occurrences += list(range(first, first + cleaned.count("\n") + 1))
        position = code.find(cleaned, position + 1)
    if occurrences:
        unique = sorted(set(occurrences))
        return unique if len(unique) <= max_ambiguous else []

    # Multi-line snippet that did not match verbatim: match on its first
    # non-empty line instead, which is the line actually being pointed at.
    head = next((part for part in cleaned.splitlines() if part.strip()), "")
    target = head.strip()
    if not target:
        return []

    # (2) equal once stripped.
    hits = [i for i, line in enumerate(lines, start=1)
            if line.strip() == target]
    if hits:
        return hits if len(hits) <= max_ambiguous else []

    # (3) contained in a line. Needs a floor: without one, a one-character
    # snippet is "contained in" most lines of any program. Lines shorter than
    # this are matched exactly by rule (1) or (2) anyway, so nothing is lost.
    if len(target) >= 8:
        hits = [i for i, line in enumerate(lines, start=1)
                if target in line.strip()]
        if hits:
            return hits if len(hits) <= max_ambiguous else []

    # (4) prefix match on a distinctive head. Below this length a prefix
    # matches too much to mean anything, so no attempt is made.
    prefix = target[:24]
    if len(prefix) >= 12:
        hits = [i for i, line in enumerate(lines, start=1)
                if line.strip().startswith(prefix)]
        if hits:
            return hits if len(hits) <= max_ambiguous else []

    return []


def _detect_one(provider: str, model: str, code: str,
                system_prompt: str, sample_id: str) -> dict:
    """Ask one model about one program, returning a prediction.

    RETRIES, AND WHY A FAILURE IS MARKED RATHER THAN SCORED
        A failed call used to degrade silently into a "not vulnerable"
        prediction, which `evaluate.py` then scored as though the model had
        genuinely judged the file clean. That is defensible for a *static
        analyser* reporting no finding -- silence really does mean "I consider
        this clean" -- but it is wrong for an API error, which is infrastructure
        failing, not the tool deciding. It silently corrupted the LLM arm's
        metrics: in one run half the calls failed and every one was counted as
        a confident miss.

        So a failed call is now retried once, and if it still fails the
        prediction carries `call_failed: True`. The caller counts those, logs
        them, and records the count in the result file, where the report
        surfaces it as a note. The verdict is still recorded as "not
        vulnerable" so the sample is not silently dropped from the confusion
        matrix, but the failure is no longer invisible.
    """
    last_error = ""
    # Two attempts: transient truncation and timeouts on a shared endpoint are
    # common enough to be worth one retry, and rare enough that a second retry
    # would mostly add latency.
    for _ in range(2):
        try:
            result = chat_json(
                provider=provider,
                model=model,
                system=system_prompt,
                user=USER_PROMPT.format(code=code),
                schema=VERDICT_SCHEMA,
                max_tokens=MAX_RESPONSE_TOKENS,
            )
            break
        except ProviderError as exc:
            last_error = str(exc)
    else:
        prediction = make_prediction(
            sample_id, False, None, 0.0,
            f"detector call failed: {last_error}"[:400],
        )
        prediction["call_failed"] = True
        return prediction

    data = result.data
    is_vulnerable = data.get("predicted_label") == config.LABEL_VULNERABLE

    # Normalise the bounded integer back to the 0.0-1.0 scale, clamping in case
    # a model ignores the schema bounds.
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence_percent", 0)) / 100.0))
    except (TypeError, ValueError):
        confidence = 0.0

    snippet = data.get("vulnerable_snippet") if is_vulnerable else None
    lines = _locate_snippet(code, snippet)

    prediction = make_prediction(
        sample_id,
        is_vulnerable,
        data.get("predicted_cwe"),
        confidence,
        data.get("reasoning", ""),
        lines=lines,
    )
    # Kept whether or not it resolved, so a disagreement between the model and
    # the ground truth can be read rather than only counted.
    if snippet:
        prediction["predicted_snippet"] = snippet[:300]
        prediction["snippet_resolved"] = bool(lines)
    return prediction


def run_llm_detector(run, manifest: list[dict], log,
                     provider: str | None = None,
                     model: str | None = None) -> dict:
    """Run an LLM detector over a run's samples.

    Args:
        run: the `runstore.Run` being analysed.
        manifest: the samples to judge -- normally the verified manifest, so
            the model is tested only on samples whose labels survived judging.
        log: logger for progress.
        provider: which backend serves the model; see providers.PROVIDERS.
            A `systemone` model takes the typed path and never sees a prompt.
        model: model id. Defaults to `config.DEFAULT_LLM_DETECTOR`.

    Returns:
        A result dict in the common shape. The model id is included in the
        `detector` display name so two LLM detectors are told apart in reports.
    """
    model = model or config.DEFAULT_LLM_DETECTOR
    # Derived from the model id unless the caller forces one, so a single
    # selection list can mix Passau, Anthropic and local models.
    provider = provider or provider_for_model(model)

    typed = provider == "systemone"

    if typed:
        questions = typed_questions()

        def analyse(entry):
            return _detect_one_typed(
                model, run.read_sample_code(entry), questions, entry["id"],
            )
    else:
        cwe_list = "\n".join(
            f"- {cwe} ({config.CWE_NAMES.get(cwe, cwe)})"
            for cwe in config.CWE_CATEGORIES
        )
        system_prompt = SYSTEM_PROMPT.format(cwe_list=cwe_list)

        def analyse(entry):
            return _detect_one(
                provider, model, run.read_sample_code(entry),
                system_prompt, entry["id"],
            )

    kind = "typed, no prose" if typed else "chat"
    log.info(f"  running LLM detector: {model} (via {provider}; {kind})")

    # Laya runs locally on this machine's CPU, where four threads contend for
    # the same cores instead of overlapping network waits -- measured slower
    # than running one at a time. Everything else is a remote endpoint and
    # parallelises normally.
    workers = 1 if model in config.LAYA_CHECKPOINTS else LLM_CONCURRENCY
    if workers == 1 and len(manifest) > 1:
        log.info(f"    {model} is local: one sample at a time")

    # Samples are independent, so they parallelise safely.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        predictions = list(pool.map(analyse, manifest))

    flagged = sum(
        1 for p in predictions
        if p["predicted_label"] == config.LABEL_VULNERABLE
    )
    failed = sum(1 for p in predictions if p.get("call_failed"))

    # Localisation bookkeeping. A typed model cannot quote anything, so these
    # are zero for Jev and Laya by construction rather than by failure.
    quoted = sum(1 for p in predictions if p.get("predicted_snippet"))
    unresolved = sum(1 for p in predictions
                     if p.get("predicted_snippet")
                     and not p.get("snippet_resolved"))

    log.info(f"    flagged {flagged}/{len(predictions)} samples as vulnerable")
    if quoted:
        log.info(
            f"    quoted a vulnerable line for {quoted} of {flagged} flagged; "
            f"{quoted - unresolved} resolved to a line number"
        )
        if unresolved:
            # Not a warning: a snippet that cannot be found is usually the
            # model paraphrasing, which costs localisation credit but says
            # nothing about the verdict.
            log.info(
                f"    {unresolved} quoted snippet(s) could not be located in "
                "the source and score as unlocalised"
            )
    if failed:
        # Loud, because these predictions are scored: a run with many failures
        # is measuring the endpoint, not the model.
        log.warning(
            f"    {failed}/{len(predictions)} calls FAILED after a retry and "
            "were recorded as 'not vulnerable'. This detector's metrics are "
            "unreliable for this run."
        )

    result = build_result("llm_detector", predictions, {
        "model": model,
        "provider": provider,
        "failed_calls": failed,
        "snippets_quoted": quoted,
        "snippets_unresolved": unresolved,
        # Which path produced these predictions. The report prints it, because
        # "the model wrote a reason" and "the model returned a probability" are
        # different enough that a reader comparing rows should see which.
        "kind": "typed" if typed else "chat",
    })
    # Overwrite the generic registry name so a report containing two LLM
    # detectors distinguishes them by model rather than showing "LLM reviewer"
    # twice.
    result["detector"] = f"LLM ({model})"
    # The result filename derives from this key, so it must carry the model
    # too, or a second LLM detector would overwrite the first one's file.
    result["detector_key"] = f"llm_{model}"
    return result
