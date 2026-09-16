# Architecture

Module-by-module design, the data formats that connect the stages, and how to
extend the framework.

For installation see [SETUP.md](SETUP.md); for the project overview see the
[README](../README.md).

---

## Layout

```
src/
    config.py           every knob: axes, models, thresholds, weights
    common.py           paths, credentials, logging, small utilities
    providers.py        one client for three model backends
    runstore.py         persistent, append-only run directories

    generate.py         stage 1  -- write sample pairs (local Ollama)
    judge.py            stage 2  -- verify labels, drop the unconfirmed
    detect.py           stage 3  -- orchestrate the tools under test
    evaluate.py         stage 4  -- metrics and cross-tool comparison
    ingest.py           alternative input: GitHub repo or local folder

    detectors/
        __init__.py     registry and the common result shape
        static_tools.py Bandit, Ruff, Semgrep
        ml_models.py    TF-IDF + LogReg, Embeddings + LogReg
        llm_tools.py    LLM reviewers

web/
    app.py              Flask routes; no benchmark logic
    templates/          base.html + one template per page
    static/style.css

runs/                   every run, never overwritten
logs/                   one log file per run
models/                 cached ML artefacts
ingested/               working copies of external code
```

### The dependency rule

Dependencies point one way only:

```
config  <-  common  <-  providers  <-  detectors  <-  detect
                    <-  runstore   <-  generate / judge / ingest / evaluate
                                                    <-  web/app.py
```

`web/app.py` imports from `src/` and never the reverse, which is what
guarantees the browser and the CLI execute the same code. No pipeline module
imports Flask.

---

## Module responsibilities

### `config.py`
Every value a supervisor might want changed, and nothing else. No logic. Split
into sections: model roles, benchmark axes, dataset size, label vocabulary,
judging thresholds, detector behaviour, comparison weights.

### `common.py`
Filesystem paths, credential parsing, logger construction, JSON helpers.

Credentials are identified **by value shape, not by label**: `http…` is a base
URL, `sk-ant-…` an Anthropic key, any other `sk-…` an OpenAI-compatible key.
The labels in the key files have already been reworded once during this
project; parsing by shape means that cannot break the loader again.

`get_logger(run_id, on_progress)` returns a logger writing to
`logs/<run_id>.log` *and* to an optional callback. That callback is how the web
app streams a run into the browser without any pipeline module knowing a web
app exists.

### `providers.py`
One function — `chat_json(provider, model, system, user, schema)` — over three
backends:

| provider | backend | structured output via |
|---|---|---|
| `ollama` | local, port 11434 | top-level `format` field |
| `passau` | OpenAI-compatible vLLM | `response_format: json_schema` |
| `anthropic` | Anthropic Messages | schema appended to the system prompt |

Two hard-won details live here:

**Bounded integers, never floats, in a schema.** Guided decoding constrains
which token may come next, and a JSON `number` grammar permits unlimited digits
after the decimal point. Asked for a `0.0–1.0` confidence, these models emit
`1.0000000000…` until they hit the token limit and the object is truncated.
Every confidence field is therefore an `integer` with `minimum`/`maximum`,
converted back to a float in the caller.

**Truncated-JSON repair.** Guided decoding does not stop a model emitting its
end-of-turn token early. Gemma does this routinely — a perfect object up to the
last field, then a stop, with no closing brace. In one run this discarded seven
of eight judgments, and because a failed judgment drops the sample, that alone
cost most of the dataset. `repair_truncated_json()` truncates at the last
structural boundary, closes what is open, and parses. The salvaged fields are
the ones that matter; only `reasoning` is typically lost, and nothing is scored
on it.

### `runstore.py`
A `Run` handle owns every path inside a run directory, so the layout is
enforced in one place. `RUN_ID_RE` validates an id **before** it is joined onto
a path — the web app takes run ids from URLs, and without that check a request
for `../../api-key-llm.txt` would be a path-traversal read, which is precisely
the CWE-22 class this project generates test cases for.

### `generate.py`
Prompts the local model for a pair, then **checks the answer rather than
trusting it**. A 3B model follows instructions unreliably, so `check_pair()`
returns flags for: syntax errors, identical twins, leaked keywords, gross
length mismatch, and implausible lengths. Syntax errors and identical twins are
fatal to a pair; the rest are recorded as quality warnings, because the judge
in stage 2 is a better arbiter of borderline cases than a line-count heuristic.

`find_leakage()` normalises identifiers before matching. `\bsafe\b` does **not**
match `safe_url` — `_` is a word character — and that is exactly the form the
leak takes in practice. Letter-only lookarounds plus a camelCase split catch
`safe_url`, `is_safe`, `isSafe` and `SafeLoader` alike.

`strip_fences()` handles opening and closing fences *independently*. The model
frequently emits valid Python followed by a bare ` ``` ` with no opening fence;
an anchored whole-string regex misses that entirely, leaving the fence in the
file and breaking every such sample.

### `judge.py`
The admission rule is one explicit function, `verdict_agrees()`, rather than
logic scattered through a loop. A sample is admitted when, for the configured
consensus rule:

1. the judge's label matches the generator's intent;
2. its confidence is at least `JUDGE_MIN_CONFIDENCE`; and
3. it counts exactly one vulnerability (or zero, for a safe sample).

CWE mismatch is **recorded but not enforced**: with a closed CWE list two
categories legitimately overlap — a path traversal that is also an SSRF — and
dropping on that would prune good samples over a naming disagreement.

### `detect.py`
Runs the selected detectors and writes one result file each. A detector that
fails is recorded and the rest continue: one broken arm must not cost the whole
comparison, and the ML arm legitimately cannot run on very small runs.

`RUNNERS` is kept separate from `DETECTOR_REGISTRY` so the web layer can render
the Tools panel without importing scikit-learn.

### `evaluate.py`
Knows nothing about how any tool works. It reads any file carrying a `detector`
name and a `predictions` list, and scores it. That is what lets the framework
claim a common evaluation protocol: there is literally one scorer.

A missing verdict counts as "not vulnerable" — silence is a miss, not an
exemption — which also matches how a static analyser reporting no finding means
"I consider this file clean".

---

## Data formats

### Manifest entry

```json
{
  "id": "sample_007",
  "file": "sample_007.py",
  "pair_id": "pair_004",
  "label": "vulnerable",
  "cwe": "CWE-89",
  "cwe_name": "SQL Injection",
  "context": "Flask web application",
  "difficulty": "Hard",
  "generator_model": "qwen2.5-coder:7b",
  "quality_flags": [],
  "note": "where the flaw is and how it is reachable"
}
```

`label` is `"unknown"` for ingested code, which makes `evaluate.py` skip it for
accuracy-style metrics.

### Detector result — the contract every tool implements

```json
{
  "detector":      "Bandit",
  "detector_key":  "static_bandit",
  "detector_type": "static",
  "predictions": [
    {"id": "sample_001",
     "predicted_label": "vulnerable",
     "predicted_cwe":   "CWE-89",
     "confidence":      0.82,
     "reasoning":       "why"}
  ]
}
```

Any extra top-level keys are carried through into the report as notes —
`findings_total`, `cv_folds`, `tool_version` and so on — so a result explains
itself rather than only stating a number.

---

## Adding a detector

Four steps; `evaluate.py` never changes.

1. Write a runner with the signature `run_x(run, manifest, log) -> dict`,
   returning the shape above. Use `detectors.build_result()` and
   `detectors.make_prediction()` so the envelope is correct by construction.
2. Add an entry to `DETECTOR_REGISTRY` in `src/detectors/__init__.py`, giving
   its `arm` (`static`, `ml` or `llm`) and a one-line `blurb` — the blurb is
   what the Tools page and the sidebar tooltip show.
3. Add it to `RUNNERS` in `src/detect.py`.
4. Add an availability probe to `available_tools()` so the interface disables
   it when its prerequisite is missing, rather than letting a job fail halfway.

---

## Concurrency and state

The judging and LLM detection stages parallelise across samples with a small
thread pool (4). Samples are independent, so this is safe; the pool is small
deliberately, because the judging endpoint is a shared university service.

The web app holds job state in a module-level dict behind a lock, which carries
a deployment constraint: **the server must run as one process.** With several
worker processes each would hold its own copy and `/api/status` would answer
from whichever one the request happened to reach. A real queue would be the
answer at scale; for a single-user research tool it would be pure overhead.

---

## Security notes

This is a local research tool with no authentication, and it must not be
exposed to an untrusted network. Within that scope:

- **Ingested code is never executed.** Every tool reads it as text — the static
  analysers parse it, the ML models tokenise or embed it, the LLMs read it.
- **Run and sample ids are validated against a whitelist pattern** before being
  joined onto a path.
- **Model names and tool keys from the browser are checked against allow-lists**
  before reaching a subprocess argument or an API call.
- **`subprocess` is always called with an argument list and never a shell**, and
  the GitHub URL is matched against a strict pattern so a string beginning
  `--upload-pack=` cannot be read by git as an option.
- **The browser is never handed HTML built from model output.** Every value is
  set with `textContent`; a sample viewer that rendered generated code as HTML
  would be an injection flaw in a vulnerability-detection tool.
- **Secrets are read from gitignored files at run time** and never copied into
  a Docker image layer, where they would persist in the image history even if a
  later layer deleted them.
