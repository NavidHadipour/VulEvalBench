"""Arm 3 -- large-language-model detectors.

A chat model is shown one program and asked whether it contains a
vulnerability. Structurally this is the same operation the judging stage
performs, but the *role* is completely different and the distinction matters:

    judge.py      decides what goes INTO the benchmark. Its verdicts become
                  part of ground truth and are never scored.
    this module   is a tool UNDER TEST. Its verdicts are scored against that
                  ground truth like Bandit's or the ML models'.

Because of that, the default LLM detector in config.py is deliberately a
different model from the default judges. A model that helped certify a sample
must not also be graded on finding the vulnerability in it, or it is marking
its own homework and its score is meaningless.

METHODOLOGICAL PRECAUTIONS -- identical to the judging stage
    1. One sample per request, no conversation history, so no verdict can
       influence another and the model cannot balance its answers across the
       set ("about half of these should be vulnerable").
    2. The prompt never states how many samples exist or what fraction are
       vulnerable.
    3. The model sees source only -- no filename, no CWE hint, no mention that
       a safe twin of this program exists elsewhere in the dataset.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import config
from detectors import build_result, make_prediction
from providers import ProviderError, chat_json, provider_for_model

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
    },
    "required": [
        "predicted_label", "predicted_cwe", "confidence_percent", "reasoning",
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
"""

USER_PROMPT = """\
Analyse the following Python program.

```python
{code}
```
"""


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

    return make_prediction(
        sample_id,
        is_vulnerable,
        data.get("predicted_cwe"),
        confidence,
        data.get("reasoning", ""),
    )


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
        model: model id. Defaults to `config.DEFAULT_LLM_DETECTOR`.

    Returns:
        A result dict in the common shape. The model id is included in the
        `detector` display name so two LLM detectors are told apart in reports.
    """
    model = model or config.DEFAULT_LLM_DETECTOR
    # Derived from the model id unless the caller forces one, so a single
    # selection list can mix Passau, Anthropic and local models.
    provider = provider or provider_for_model(model)

    cwe_list = "\n".join(
        f"- {cwe} ({config.CWE_NAMES.get(cwe, cwe)})"
        for cwe in config.CWE_CATEGORIES
    )
    system_prompt = SYSTEM_PROMPT.format(cwe_list=cwe_list)

    log.info(f"  running LLM detector: {model} (via {provider})")

    # Samples are independent, so they parallelise safely.
    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        predictions = list(pool.map(
            lambda entry: _detect_one(
                provider, model, run.read_sample_code(entry),
                system_prompt, entry["id"],
            ),
            manifest,
        ))

    flagged = sum(
        1 for p in predictions
        if p["predicted_label"] == config.LABEL_VULNERABLE
    )
    failed = sum(1 for p in predictions if p.get("call_failed"))

    log.info(f"    flagged {flagged}/{len(predictions)} samples as vulnerable")
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
    })
    # Overwrite the generic registry name so a report containing two LLM
    # detectors distinguishes them by model rather than showing "LLM reviewer"
    # twice.
    result["detector"] = f"LLM ({model})"
    # The result filename derives from this key, so it must carry the model
    # too, or a second LLM detector would overwrite the first one's file.
    result["detector_key"] = f"llm_{model}"
    return result
