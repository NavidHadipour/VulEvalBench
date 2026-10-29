"""Stage 2 of the pipeline -- independent verification of the generated labels.

WHAT THIS STAGE IS FOR
    The previous draft's biggest methodological weakness was that ground truth
    came from the generator: a sample was labelled "vulnerable" because the
    model had been *asked* to write a vulnerability, not because anyone
    confirmed it had. Every downstream metric inherited that assumption.

    This stage removes it. One or two judge models -- different models, on a
    different machine, from a different vendor than the generator -- each read
    every sample cold and say what they see. A sample is admitted to the
    benchmark only if the judges' independent reading matches what the
    generator intended. Everything else is dropped.

    The result is a smaller but defensible dataset: every surviving label has
    been agreed on by at least one party that did not write the code.

WHAT THE JUDGE IS NOT
    The judge is not a detector, and its verdicts are never scored as detection
    results. It is a *filter* applied before the tools under test ever see the
    data. Keeping the two roles separate matters: a model that certified a
    sample must not also be graded on finding the vulnerability in it, or it
    would be marking its own homework. config.py keeps the default judge models
    and the default LLM-detector model distinct for this reason.

WHY MORE THAN ONE JUDGE
    Agreement between independently-prompted models is much weaker evidence
    than a human review, but it is far stronger than trusting the generator
    alone. Where the judges disagree is exactly where the label is unreliable
    -- so with the default "all" consensus rule, disagreement drops the sample.
    The web interface shows every judge's verdict side by side so the
    disagreements can be read, which is often the most informative view of a
    run.

    Any number of judges can be selected and three slots are offered, because
    two judges cannot resolve a disagreement: under "all" every dispute is
    fatal, and there is no way to tell which of the two was wrong. A third
    judge makes `JUDGE_CONSENSUS = "majority"` meaningful, where two agreeing
    judges outvote one dissenter. Weaker verification, and reported as such --
    see the note on the rule in config.py.

TWO KINDS OF JUDGE
    Most judges here are chat models forced into a JSON schema by guided
    decoding. The System One models -- Jev and Laya -- are not: they read the
    program once and answer typed questions (pick an option, score on a
    rubric) with calibrated probabilities, and generate no text at all.

    They answer the same questions and are held to the same admission rules,
    so their verdicts are directly comparable. Two differences matter when
    reading a run: their `confidence` is a real probability rather than a
    number a chat model invented, and they produce no `reasoning` -- the field
    carries their answer probabilities instead, marked as derived. See
    `judge_one_typed` and the note above SYSTEMONE_MODELS in config.py.

PRECAUTIONS
    1. One sample per request, with no conversation history. Sending the whole
       set at once would let a judge balance its answers across it ("about half
       of these should be vulnerable") instead of judging each program alone.
    2. The judge is never told the intended label, the intended CWE, or that a
       safe twin of this program exists.
    3. Filenames the judge sees are neutral and carry no label information.

Inputs:
    runs/<run_id>/01_generation/generation_manifest.json
Outputs:
    runs/<run_id>/02_judgement/judgement_<model>.json    one file per judge
    runs/<run_id>/02_judgement/verified_manifest.json    the survivors
    logs/<run_id>.log

Run with:
    python src/judge.py <run_id>
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import config
import runstore
from common import get_logger, write_json
from providers import (
    ProviderError,
    answer_confidence,
    chat_json,
    decide_typed,
    provider_for_model,
    systemone_load_notes,
)

# How many judgment requests to keep in flight at once.
#
# Sequential judging of 2 judges over 40 samples is 80 round trips, which at a
# few seconds each is long enough that a user assumes the page has hung. A
# small pool cuts that substantially while staying polite to a shared
# university endpoint -- this is not a load test.
JUDGE_CONCURRENCY = 4

# Response budget per verdict. See the note in detectors/llm_tools.py: several
# of these models reason before answering and that reasoning is billed against
# the same budget, so a seemingly generous 1200 truncated every call from
# qwen36-35b mid-object. A failed judgment drops the sample, so an undersized
# budget here quietly destroys the dataset.
MAX_RESPONSE_TOKENS = 4000


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------
# Enforced by the endpoint's guided decoding, so a judge cannot answer "maybe"
# or "VULNERABLE" and break the string comparison below.

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "predicted_label": {
            "type": "string",
            "enum": [config.LABEL_VULNERABLE, config.LABEL_SAFE],
            "description": "Whether this program contains a security vulnerability.",
        },
        "predicted_cwe": {
            # Nullable: a program judged safe has no category to report.
            "type": ["string", "null"],
            "description": "CWE identifier such as 'CWE-89', or null if not vulnerable.",
        },
        "vulnerability_count": {
            "type": "integer",
            "description": (
                "How many DISTINCT security vulnerabilities are present. "
                "0 for a clean program."
            ),
        },
        # WHY AN INTEGER PERCENTAGE AND NOT A 0.0-1.0 FLOAT
        #   Guided decoding constrains the response to this schema by
        #   restricting which tokens may come next. A JSON `number` grammar
        #   permits unlimited digits after the decimal point, and these models
        #   reliably exploit that: asked for a float they emit
        #   `1.00000000000...` until they hit the token limit, truncating the
        #   object mid-write and failing the whole call. An integer with a
        #   bounded range has no fractional part to run away with.
        #   Converted back to a 0.0-1.0 float in `judge_one` so the rest of the
        #   pipeline still sees a conventional confidence.
        "confidence_percent": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Confidence in the verdict, 0 to 100.",
        },
        "reasoning": {
            "type": "string",
            "description": "One or two sentences justifying the verdict.",
        },
    },
    "required": [
        "predicted_label", "predicted_cwe",
        "vulnerability_count", "confidence_percent", "reasoning",
    ],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# CLOSED-SET CWE PREDICTION. The judge is shown the candidate identifiers so
# its category answer is directly comparable with the intended one: without the
# list it might answer "CWE-943" for an SQL injection and be recorded as a
# mismatch on a naming technicality rather than a real disagreement.
#
# The two "do not assume" sentences counter a known bias: a model told it is
# reviewing code for a security benchmark expects to find a vulnerability and
# duly reports one, which would wave through every safe twin as "vulnerable"
# and destroy the safe half of the dataset.

SYSTEM_PROMPT = """\
You are a security code reviewer analysing a single Python program.

Decide whether the program contains a security vulnerability that is reachable
from untrusted input. Judge only the code in front of you: do not assume a
vulnerability is present, and do not assume the code is safe. Many of the
programs you will see are clean.

If it is vulnerable, identify the category using one of these identifiers:
{cwe_list}

Also report `vulnerability_count`: the number of DISTINCT security
vulnerabilities you can actually justify, counting 0 for a clean program. Do
not inflate this with style issues, missing input validation that has no
security consequence, or defensive hardening you would merely prefer.

Report `predicted_cwe` as null when the program is not vulnerable. Set
`confidence_percent` to a whole number between 0 and 100, reflecting how
certain you are.
"""

# The program is fenced so the model can tell source from instructions. Note
# there is no filename, no CWE hint, no intended label and no mention that a
# twin of this program exists.
USER_PROMPT = """\
Analyse the following Python program.

```python
{code}
```
"""


# ---------------------------------------------------------------------------
# One judge over one sample
# ---------------------------------------------------------------------------

def judge_one(model: str, code: str, system_prompt) -> dict:
    """Ask one judge about one program, whichever kind of judge it is.

    Args:
        model: judgment model id, as served by the endpoint.
        code: the program's source.
        system_prompt: for a chat judge, the fully-formatted system prompt;
            for a System One judge, the question set from `typed_questions()`.
            Built once by the caller either way, so the CWE list is not
            re-rendered for every sample.

    Returns:
        A verdict dict. On a provider failure it returns a verdict carrying an
        `error` key rather than raising: one unreachable call should cost one
        sample, not the whole run. Such a verdict never agrees with anything,
        so the affected sample is dropped by the consensus rule below.
    """
    # The backend is derived from the model id, so Passau-hosted, Anthropic and
    # local models can all be selected as judges from the same dropdown.
    provider = provider_for_model(model)

    # A System One model answers typed questions instead of a prompt, so it
    # takes a different route to the same verdict shape. The caller does not
    # need to know which kind of judge it selected.
    if provider == "systemone":
        return judge_one_typed(model, code, system_prompt)

    # Two attempts: a failed judgment drops the sample entirely, so one retry
    # against a transient truncation or timeout is cheap insurance against
    # losing data to infrastructure rather than to disagreement.

    last_error = ""
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
        return {
            "predicted_label": None,
            "predicted_cwe": None,
            "vulnerability_count": -1,
            "confidence": 0.0,
            "reasoning": "",
            "error": last_error[:400],
            "input_tokens": 0,
            "output_tokens": 0,
        }

    data = result.data
    # Normalise the bounded integer back to the 0.0-1.0 scale the rest of the
    # pipeline uses, clamping in case a model ignores the schema's bounds.
    percent = data.get("confidence_percent", 0)
    try:
        confidence = max(0.0, min(1.0, float(percent) / 100.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "predicted_label": data.get("predicted_label"),
        "predicted_cwe": data.get("predicted_cwe"),
        "vulnerability_count": data.get("vulnerability_count", -1),
        "confidence": confidence,
        "reasoning": data.get("reasoning", ""),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }


# ---------------------------------------------------------------------------
# One typed judge over one sample
# ---------------------------------------------------------------------------
# The same three questions the chat judges answer, in the form a System One
# model takes them: a closed choice for the label, a closed choice over the
# CWE list, and an ordinal score for the count. The wording is kept as close
# to SYSTEM_PROMPT as the format allows -- including the anti-bias sentences,
# which matter more here, not less: these models were trained on prose and
# guardrail decisions, and a prompt that implies a vulnerability is expected
# would push them the same way it pushes a chat model.

TYPED_LABEL_INSTRUCTIONS = (
    "The state is a single Python program. Decide whether it contains a "
    "security vulnerability that is reachable from untrusted input. Judge only "
    "the code in front of you: do not assume a vulnerability is present, and "
    "do not assume the code is safe. Many of these programs are clean."
)

TYPED_CWE_INSTRUCTIONS = (
    "If this program contains a security vulnerability reachable from "
    "untrusted input, which category describes it? Choose 'none' if the "
    "program has no such vulnerability."
)

TYPED_COUNT_INSTRUCTIONS = (
    "How many DISTINCT security vulnerabilities does this program contain? Do "
    "not count style issues, missing input validation with no security "
    "consequence, or hardening you would merely prefer."
)

# Ordered, because a score question's level IS its position in this list.
TYPED_COUNT_LEVELS = [
    "No security vulnerability.",
    "Exactly one security vulnerability.",
    "Two or more distinct security vulnerabilities.",
]


def typed_questions() -> dict:
    """Build the question set a System One judge answers.

    Built once per run by the caller and reused for every sample, like the
    chat judges' formatted system prompt.

    Returns:
        A `questions` object for `providers.decide_typed`.
    """
    # The CWE options carry their names as descriptions, which is what a choice
    # question uses to tell the options apart -- the bare identifier "CWE-918"
    # means nothing to a model that was never trained on the catalogue.
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
        "count": {
            "type": "score",
            "instructions": TYPED_COUNT_INSTRUCTIONS,
            "criteria": TYPED_COUNT_LEVELS,
        },
    }


def _argmax_level(answer: dict) -> int:
    """The most probable level of a score answer, as an integer.

    A score answer's `score` is the probability-weighted EXPECTED level, so a
    model torn between "none" and "two or more" returns 1.0 -- the one value
    neither of its candidates. For a count that must be compared against an
    exact 0 or 1, the most probable level is the honest reading, so the
    distribution is used rather than its mean.

    Falls back to rounding `score` if `probabilities` is absent.
    """
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict) and probabilities:
        best = max(probabilities.items(), key=lambda kv: kv[1])[0]
        try:
            return int(str(best).strip())
        except (TypeError, ValueError):
            pass
    try:
        return int(round(float(answer.get("score", -1))))
    except (TypeError, ValueError):
        return -1


def _answer_confidence(answer: dict) -> float:
    """How much probability a typed model put on the answer it gave, 0.0-1.0.

    Delegates to `providers.answer_confidence`, which carries the reasoning
    for WHICH of a System One answer's two confidence numbers is the right one
    to read -- and the measurement showing that reading the wrong one flips
    every verdict against JUDGE_MIN_CONFIDENCE. It lives there because the
    detector arm needs the same reading, and nothing under detectors/ may
    import this module: judging produces the ground truth the detectors are
    scored against.
    """
    return answer_confidence(answer)


def judge_one_typed(model: str, code: str, questions: dict) -> dict:
    """Ask one System One judge about one program.

    Returns the same verdict shape as `judge_one`, so the admission rule,
    the stored files and the comparison page treat both kinds of judge
    identically.

    Args:
        model: a System One model id (`config.SYSTEMONE_MODELS`).
        code: the program's source. Passed as the `state` exactly as written:
            there is no prompt to fence it off from, which also means none of
            the "ignore previous instructions" exposure a chat judge has.
        questions: the set from `typed_questions()`.

    Returns:
        A verdict dict. On failure it carries an `error` key and the sample is
        treated as an abstention, exactly as for a chat judge.
    """
    last_error = ""
    for _ in range(2):
        try:
            result = decide_typed(model, code, questions)
            break
        except ProviderError as exc:
            last_error = str(exc)
    else:
        return {
            "predicted_label": None,
            "predicted_cwe": None,
            "vulnerability_count": -1,
            "confidence": 0.0,
            "reasoning": "",
            "error": last_error[:400],
            "input_tokens": 0,
            "output_tokens": 0,
        }

    answers = result.answers
    label_answer = answers.get("label") or {}
    cwe_answer = answers.get("cwe") or {}
    count_answer = answers.get("count") or {}

    label = label_answer.get("choice")
    cwe = cwe_answer.get("choice")

    # A response with no label answer did not judge this program, whatever
    # else it returned. Reported as an error so the consensus rule treats it
    # as an ABSTENTION: recording it as predicted_label=None instead would be
    # read as a disagreement the judge never expressed, and under the "all"
    # rule one malformed response would discard the sample.
    if not label:
        return {
            "predicted_label": None,
            "predicted_cwe": None,
            "vulnerability_count": -1,
            "confidence": 0.0,
            "reasoning": "",
            "error": (f"{model} returned no label answer; "
                      f"answers: {sorted(answers)}"),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }

    confidence = _answer_confidence(label_answer)

    # "none" is this format's way of saying null: a choice question must pick
    # one of its options, so the absence of a category has to be an option.
    if cwe == "none" or label == config.LABEL_SAFE:
        cwe = None

    # No prose exists to put here, so the field carries the numbers the model
    # actually produced. Marked, so nobody reads it as a model's own words.
    label_probabilities = label_answer.get("probabilities") or {}
    reasoning = (
        "[no prose: typed decision model] "
        + ", ".join(f"P({name})={float(value):.2f}"
                    for name, value in sorted(label_probabilities.items())
                    if isinstance(value, (int, float)))
    )
    if cwe_answer:
        reasoning += (
            f"; cwe={cwe_answer.get('choice')} "
            f"p={_answer_confidence(cwe_answer):.2f}"
        )

    return {
        "predicted_label": label,
        "predicted_cwe": cwe,
        "vulnerability_count": _argmax_level(count_answer),
        "confidence": confidence,
        "reasoning": reasoning,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }


# ---------------------------------------------------------------------------
# Admission rule
# ---------------------------------------------------------------------------

def verdict_agrees(entry: dict, verdict: dict) -> tuple[bool, list[str]]:
    """Decide whether one judge's verdict confirms one sample's intended label.

    This is the rule that decides what stays in the benchmark, so it is written
    as one explicit function rather than scattered through the loop below.

    Args:
        entry: the manifest entry, carrying the generator's intended `label`
            and `cwe`.
        verdict: one judge's verdict for that sample.

    Returns:
        `(agrees, reasons)`. `reasons` lists every rule that failed, so a
        dropped sample can be explained in the UI rather than merely vanishing.
    """
    reasons: list[str] = []

    if verdict.get("error"):
        return False, ["judge_error"]

    intended = entry["label"]
    predicted = verdict.get("predicted_label")

    # Rule 1 -- the label itself must match. This is the core of the stage.
    if predicted != intended:
        reasons.append(f"label_mismatch:judge_said_{predicted}")

    # Rule 2 -- an uncertain judge is not evidence. A verdict the judge itself
    # only half believes cannot serve as ground truth for scoring other tools.
    if verdict.get("confidence", 0.0) < config.JUDGE_MIN_CONFIDENCE:
        reasons.append(
            f"low_confidence:{verdict.get('confidence', 0.0):.2f}"
        )

    # Rule 3 -- exactly one vulnerability in a vulnerable sample, none in a
    # safe one. A program with two flaws breaks per-CWE scoring, because a
    # detector that finds the unintended flaw is marked right on the label and
    # wrong on the category.
    if config.REQUIRE_SINGLE_VULNERABILITY:
        count = verdict.get("vulnerability_count", -1)
        if intended == config.LABEL_VULNERABLE and count != 1:
            reasons.append(f"expected_1_vulnerability_got_{count}")
        if intended == config.LABEL_SAFE and count not in (0, -1):
            reasons.append(f"expected_0_vulnerabilities_got_{count}")

    return (not reasons), reasons


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------

def run_judgement(
    run_id: str,
    judge_models: list[str] | None = None,
    on_progress=None,
    should_stop=None,
) -> dict:
    """Judge every sample in a run and write the verified manifest.

    Args:
        run_id: the run to judge, as returned by `generate_dataset`.
        judge_models: any number of judgment model ids, from any of the
            registries in config -- the judgment endpoint, Anthropic, or the
            System One models. Defaults to the pair in config. Passing one
            model is supported and simply means no cross-model agreement check
            is performed; three make `JUDGE_CONSENSUS = "majority"` meaningful.
        on_progress: callable taking one string; receives every log line.
        should_stop: callable taking nothing, consulted before each judge
            model. Checked between models rather than between samples: one
            model's samples run in a thread pool, and the verified manifest is
            written from the models that completed, so a judge is either fully
            applied or not applied at all. A judge half-applied across the
            samples would make "the judges agreed" mean different things for
            different rows.

    Returns:
        A summary dict: how many samples were judged, how many survived, and
        the per-judge agreement rate with the generator.

    Raises:
        SystemExit: if the run has no generated samples.
    """
    run = runstore.load_run(run_id)
    log = get_logger(run_id, on_progress)

    manifest = run.read_generation_manifest()
    if not manifest:
        raise SystemExit(
            f"Run {run_id} has no generated samples. Run generation first."
        )

    judge_models = judge_models or [config.DEFAULT_JUDGE_A, config.DEFAULT_JUDGE_B]
    # De-duplicate while preserving order: selecting the same model twice in
    # the UI would otherwise double the cost and produce a meaningless
    # "agreement" between a model and itself.
    seen: set[str] = set()
    judge_models = [
        m for m in judge_models if m and not (m in seen or seen.add(m))
    ]
    if not judge_models:
        raise SystemExit("No judge models selected.")

    cwe_list = "\n".join(
        f"- {cwe} ({config.CWE_NAMES.get(cwe, cwe)})"
        for cwe in config.CWE_CATEGORIES
    )
    system_prompt = SYSTEM_PROMPT.format(cwe_list=cwe_list)

    # The System One judges take typed questions rather than a prompt. Built
    # here alongside the prompt, for the same reason: once per run, not once
    # per sample.
    questions = typed_questions()

    log.info("")
    log.info("=== Stage 2: judgement ===")
    log.info(
        f"Judging {len(manifest)} samples with {len(judge_models)} judge(s): "
        f"{', '.join(judge_models)}"
    )
    log.info(
        f"Consensus rule: {config.JUDGE_CONSENSUS}; "
        f"min confidence {config.JUDGE_MIN_CONFIDENCE}; "
        f"single-vulnerability check "
        f"{'on' if config.REQUIRE_SINGLE_VULNERABILITY else 'off'}"
    )

    # verdicts_by_judge[model][sample_id] -> verdict dict
    verdicts_by_judge: dict[str, dict[str, dict]] = {}

    for model in judge_models:
        if should_stop is not None and should_stop():
            log.warning(
                f"  Stop requested; skipping {model} and every judge after "
                "it. The verified manifest is written from the judges that "
                "finished."
            )
            break

        provider = provider_for_model(model)
        kind = "typed decisions" if provider == "systemone" else "chat"
        log.info(f"  judge: {model}  ({provider}, {kind})")

        task = questions if provider == "systemone" else system_prompt

        # Samples are independent, so they parallelise safely. The pool is
        # small deliberately -- see JUDGE_CONCURRENCY.
        #
        # A local Laya checkpoint is the exception: it runs in this process on
        # the CPU, so four threads do not make it faster -- they contend for
        # the same cores while each holds a copy of the activations. One at a
        # time is both faster and lighter there.
        workers = 1 if model in config.LAYA_CHECKPOINTS else JUDGE_CONCURRENCY

        with ThreadPoolExecutor(max_workers=workers) as pool:
            verdict_list = list(pool.map(
                lambda entry: judge_one(
                    model, run.read_sample_code(entry), task
                ),
                manifest,
            ))

        verdicts = dict(zip((e["id"] for e in manifest), verdict_list))
        verdicts_by_judge[model] = verdicts

        # What the checkpoint said about itself while loading. Reported here
        # rather than dropped on stderr, because these warnings are about how
        # far `confidence` can be trusted -- and JUDGE_MIN_CONFIDENCE admits
        # or drops every sample on exactly that number.
        for note in systemone_load_notes(model):
            log.warning(f"    {model} calibration: {note}")

        errors = sum(1 for v in verdicts.values() if v.get("error"))
        agreed = sum(
            1 for entry in manifest
            if verdict_agrees(entry, verdicts[entry["id"]])[0]
        )
        log.info(
            f"    agreed with generator on {agreed}/{len(manifest)} samples"
            + (f"; {errors} call(s) failed" if errors else "")
        )

        # A judge that answered nothing is a broken configuration, not a strict
        # reviewer, and the distinction has to be loud: the symptom otherwise
        # reaching the user is "0 samples admitted", which looks like a dataset
        # quality problem and sends them hunting in the wrong place.
        if errors == len(manifest) and manifest:
            first_error = next(
                (v.get("error", "") for v in verdicts.values() if v.get("error")),
                "",
            )
            log.warning(
                f"    {model} answered NONE of the {len(manifest)} samples -- "
                "it is abstaining, not disagreeing. Check the model id against "
                "the endpoint's /v1/models."
            )
            if first_error:
                log.warning(f"      first error: {first_error[:200]}")

        # One file per judge, named after the model, so two judges' outputs sit
        # side by side in the run directory and are told apart at a glance.
        write_json(run.judgement_file(model), {
            "run_id": run_id,
            "judge_model": model,
            "judge_display_name": config.model_display_name(model),
            "samples_judged": len(manifest),
            "agreed_with_generator": agreed,
            "failed_calls": errors,
            "tokens": {
                "input": sum(v.get("input_tokens", 0) for v in verdicts.values()),
                "output": sum(v.get("output_tokens", 0) for v in verdicts.values()),
            },
            # Stored as a list rather than a dict so the file reads in sample
            # order, which is how a human reviews it.
            "verdicts": [
                {"id": entry["id"], **verdicts[entry["id"]]}
                for entry in manifest
            ],
        })

    # --- apply the consensus rule ------------------------------------------
    verified: list[dict] = []
    dropped: list[dict] = []

    for entry in manifest:
        per_judge = {}
        agreements = []
        abstained = []

        for model in judge_models:
            verdict = verdicts_by_judge[model][entry["id"]]
            agrees, reasons = verdict_agrees(entry, verdict)

            # A judge whose call failed ABSTAINS rather than voting "no".
            #
            # It never read the program, so its silence is not evidence that
            # the label is wrong. Counting it as disagreement means one model
            # being unreachable silently discards an entire run under the
            # "all" rule -- which is exactly what happened when the endpoint
            # stopped serving a model that /v1/models still advertised.
            if reasons == ["judge_error"]:
                abstained.append(model)
            else:
                agreements.append(agrees)

            per_judge[model] = {
                "agrees": agrees,
                "reasons": reasons,
                "predicted_label": verdict.get("predicted_label"),
                "predicted_cwe": verdict.get("predicted_cwe"),
                "vulnerability_count": verdict.get("vulnerability_count"),
                "confidence": verdict.get("confidence"),
                "reasoning": verdict.get("reasoning", ""),
            }

        # With every judge abstaining there is no verification at all, so the
        # sample is dropped -- but for the honest reason, which the record
        # below preserves, rather than as a pretended disagreement.
        #
        # Every rule below counts only the judges that answered. An abstention
        # is not half a vote: a majority of three judges where one was
        # unreachable is two of two, not two of three.
        if not agreements:
            admitted = False
        elif config.JUDGE_CONSENSUS == "all":
            admitted = all(agreements)
        elif config.JUDGE_CONSENSUS == "majority":
            # Strictly more than half, so two of four is not a majority. With
            # two judges this is the same as "all", which is why the rule only
            # becomes a different rule at three.
            admitted = sum(agreements) * 2 > len(agreements)
        else:
            admitted = any(agreements)

        # Recorded but not enforced: whether the judges also named the same CWE
        # the generator intended. Kept out of the admission rule because with a
        # closed CWE list two categories can legitimately overlap (a path
        # traversal that is also an SSRF, say), and dropping on that would
        # prune good samples for a naming disagreement. It is surfaced in the
        # UI so a systematic mismatch is still visible.
        cwe_agreement = [
            per_judge[m]["predicted_cwe"] == entry["cwe"]
            for m in judge_models
            if per_judge[m]["predicted_label"] == config.LABEL_VULNERABLE
        ]

        record = {
            **entry,
            "judgement": {
                "admitted": admitted,
                "abstained": abstained,
                "verified_by": [m for m in judge_models if m not in abstained],
                "judges": per_judge,
                "cwe_agreement": cwe_agreement,
                "label_source": "generator+judge",
            },
        }

        if admitted:
            verified.append(record)
        else:
            dropped.append(record)

    write_json(run.verified_manifest_file, verified)
    write_json(run.judgement_dir / "judgement_dropped.json", dropped)

    # A consolidated side-by-side view, which is what the comparison page
    # renders. Building it here rather than in the web layer keeps the browser
    # and any CLI reader looking at identical data.
    write_json(run.judgement_dir / "judgement_comparison.json", {
        "run_id": run_id,
        "judges": [
            {"model": m, "display_name": config.model_display_name(m),
             # Which kind of judge produced this column. The comparison page
             # needs it to explain an empty reasoning cell, and a reader of
             # the file needs it to know whether `confidence` is a calibrated
             # probability or a number a chat model wrote.
             "kind": ("typed" if provider_for_model(m) == "systemone"
                      else "chat")}
            for m in judge_models
        ],
        "consensus_rule": config.JUDGE_CONSENSUS,
        "rows": [
            {
                "id": entry["id"],
                "intended_label": entry["label"],
                "intended_cwe": entry["cwe"],
                "difficulty": entry.get("difficulty", ""),
                "admitted": entry["judgement"]["admitted"],
                "judges": entry["judgement"]["judges"],
            }
            for entry in verified + dropped
        ],
    })

    survived_vulnerable = sum(
        1 for e in verified if e["label"] == config.LABEL_VULNERABLE
    )

    # Judging decides which samples the detectors are scored on, so re-judging
    # a run that has already been through detection invalidates those results:
    # they were produced against a different admitted set. Left in place they
    # are scored anyway, and every sample the old run never saw is counted as a
    # correct "not vulnerable" call -- which flatters the stale detector and
    # can hand it a false positive rate of zero.
    #
    # The files are not deleted, because a previous result is evidence worth
    # keeping. evaluate.py marks any detector with incomplete coverage as
    # STALE in the report; this warning is the earlier, louder half of that.
    existing = list(run.detection_dir.glob("detection_*.json"))
    if existing:
        log.warning(
            f"{len(existing)} detection result file(s) predate this judging "
            "pass and are now STALE -- they were scored against a different "
            "admitted set. Re-run detection for this run; until then the "
            "evaluation marks them."
        )

    run.update_metadata(
        status="judged",
        judges=judge_models,
        judged_samples=len(manifest),
        verified_samples=len(verified),
        dropped_samples=len(dropped),
        detection_stale_after_judging=bool(existing),
    )

    log.info(
        f"Admitted {len(verified)}/{len(manifest)} samples "
        f"({survived_vulnerable} vulnerable / "
        f"{len(verified) - survived_vulnerable} safe)"
    )
    log.info(f"Dropped {len(dropped)}; see judgement_dropped.json for reasons")

    if not verified:
        log.warning(
            "No samples survived judgement. The tools under test have nothing "
            "to run on. Consider a larger generator model, or relaxing "
            "JUDGE_CONSENSUS to 'any' in config.py."
        )

    return {
        "run_id": run_id,
        "judges": judge_models,
        "judged": len(manifest),
        "verified": len(verified),
        "dropped": len(dropped),
        "verified_vulnerable": survived_vulnerable,
        "verified_safe": len(verified) - survived_vulnerable,
    }


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Judge a run named on the command line, or the most recent one."""
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        latest = runstore.latest_run()
        if latest is None:
            raise SystemExit("No runs found. Run 'python src/generate.py' first.")
        run_id = latest.run_id
        print(f"No run id given; using the most recent: {run_id}")

    run_judgement(run_id, on_progress=print)


if __name__ == "__main__":
    main()
