# VulEvalBench

A benchmarking framework for comparing vulnerability-detection tools on Python
code, under a single evaluation protocol.

An LLM generates a CWE-labelled benchmark; independent judge models verify the
labels and discard the ones they cannot confirm; three families of detector —
static analysers, machine-learning classifiers and LLM reviewers — are then run
over what survives and scored against each other.

> **Status.** The pipeline is complete and runs end to end. The datasets it
> currently produces are small, so the numbers are illustrative rather than
> findings. See [Known limitations](#known-limitations).

---

## The pipeline

```
  config.py
      |
      v
 1. generate.py     local Ollama model writes vulnerable/safe program pairs
      |                 runs/<run_id>/01_generation/
      v
 2. judge.py        1-2 independent LLMs verify each label;
      |             samples they cannot confirm are DROPPED
      |                 runs/<run_id>/02_judgement/
      v
 3. detect.py       the tools under test run on the survivors
      |                 runs/<run_id>/03_detection/
      v
 4. evaluate.py     metrics per tool, then a cross-tool comparison
                        runs/<run_id>/04_evaluation/
```

Each stage reads the previous stage's output from the run directory, so any
stage can be re-run on its own — for example, to try a different tool selection
without paying to regenerate the dataset.

### Why there is a judging stage

The single biggest weakness of the earlier draft was that ground truth came
from the generator: a sample was labelled "vulnerable" because the model had
been *asked* to write a vulnerability, not because anyone confirmed it had.
Every downstream metric silently inherited that assumption.

Stage 2 removes it. One or two judge models — different models, on different
hardware, from a different vendor than the generator — read each program cold,
without being told the intended label, and say what they see. A sample enters
the benchmark only if their independent reading matches the generator's intent.
Everything else is discarded.

The result is a smaller but defensible dataset in which every surviving label
has been agreed on by a party that did not write the code.

---

## The three detector arms

| Arm | Tools | What it actually does |
|---|---|---|
| **Static analysis** | Bandit, Ruff, Semgrep | Bandit matches AST patterns; Ruff reimplements the same ruleset in a separate engine; Semgrep adds taint analysis, tracking whether untrusted input actually reaches a sink. Fast, deterministic, no training. |
| **Machine learning** | TF-IDF + LogReg, Embeddings + LogReg | Learns from labelled examples. The two differ *only* in representation — lexical versus semantic — so any gap between them is attributable to that. |
| **LLM** | Any chat model, local or remote | Reads the program and reasons about it. |

Every tool returns the same verdict shape, so `evaluate.py` scores all of them
with one code path and none gets a bespoke advantage.

### Three things that must be said about the comparison

These are real methodological asymmetries, not footnotes. Any write-up has to
state them.

1. **The ML arm is measured under a different protocol.** It is the only arm
   that learns from the data it is scored on. Every ML prediction is therefore
   made *out of fold*: the model that labelled a sample was fitted without ever
   seeing it. Scoring on training data would report memorisation as detection.

2. **Static analysers are scored on the benchmark's categories only.** A linter
   reports everything it knows about. On the very first generated pair, both
   static tools flagged "requests call without timeout" (CWE-400) on the
   vulnerable *and* the safe program — counting that would push the false
   positive rate to 1.0 for a reason unconnected to the question being asked.
   `config.STATIC_SCOPE_TO_BENCHMARK_CWES` controls this, and both the scoped
   and unscoped finding counts are recorded either way.

   **Two CWE-matching policies are reported side by side**, and the reason is
   worth stating plainly. Exact-id matching is *not* a neutral choice: it
   rewards tools that label findings coarsely and penalises tools that label
   them precisely. Semgrep reported CWE-276 "Incorrect Default Permissions" on
   a CWE-732 sample — MITRE lists CWE-276 as a child of CWE-732, so the tool
   had found exactly the right weakness, yet exact matching discarded it.

   The hierarchy-aware policy accepts any *descendant* of a catalogue entry
   (never an ancestor — CWE-20 "Improper Input Validation" must not be credited
   for every injection flaw). Relationships are imported wholesale from MITRE's
   published catalogue via `src/cwe_taxonomy.py`, not hand-picked, and applied
   uniformly to every tool.

   **This rule was added after observing that it disadvantaged one tool**, so
   both scorings appear in every report rather than only the more generous one.
   On the reference run it gained Semgrep exactly one true positive
   (F1 0.500 → 0.667) and gained Bandit nothing — and it did not change the
   ranking. The hierarchy applies to *scoping* only; CWE-identification
   accuracy still requires an exact match.

3. **LLM detectors are shown the candidate CWE list.** This makes their
   category answers comparable with ground truth, but it is a small advantage
   over the static tools, which map findings through a fixed rule table and
   cannot choose from a list.

---

## The comparison mathematics

Four layers, from standard to project-specific.

**1 — Confusion matrix and rates.** Accuracy, precision, recall, F1, FPR, FNR.
The positive class is `vulnerable`, so a false positive is a nuisance alert and
a false negative is a missed vulnerability.

**2 — Two robust single-number summaries.**

- **MCC** (Matthews Correlation Coefficient), a correlation across the whole
  confusion matrix, ranging −1 to +1 with 0 at chance. Unlike accuracy it
  cannot be inflated by always answering with the majority class, which matters
  because judging tends to leave the two classes unbalanced.
- **Youden's J** = recall + specificity − 1. Reads as "how much better than
  guessing", weighting both error types symmetrically.

**3 — A composite score**, used to rank tools:

```
composite = 0.50 × F1  +  0.30 × (1 − FPR)  +  0.20 × CWE_accuracy
```

Three terms for three separate questions: is it any good at finding flaws; can
it be trusted not to cry wolf; did it understand *what* the flaw was. The
weights live in `config.COMPOSITE_WEIGHTS`, must sum to 1.0, and are an
explicit editorial choice rather than a hidden formula.

**4 — Pairwise agreement** between tools, as Cohen's kappa. This is the only
measure here that needs no ground truth, which makes it the one that still
works on ingested GitHub code. Two tools that agree strongly are largely
redundant; two that disagree are finding different things.

---

## Quick start

Full setup instructions, including on a fresh machine, are in
**[docs/SETUP.md](docs/SETUP.md)**.

```bash
# 1. Clone and install (Python 3.10+)
git clone https://github.com/NavidHadipour/VulEvalBench.git
cd VulEvalBench
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt        # Linux/macOS: .venv/bin/pip

# 2. Local generator
winget install Ollama.Ollama          # or https://ollama.com/download
ollama pull qwen2.5-coder:3b          # the default; 7b if you have a GPU

# 3. Credentials -- copy the template and fill in your URL and key
copy api-key-judgment.txt.example api-key-judgment.txt
#    (api-key-llm.txt.example is optional, for Claude models)

# 4. Verify everything, then run
.venv\Scripts\python src\check_setup.py   # prints a checklist; fixes each [FAIL]
.venv\Scripts\python web\app.py           # then open http://localhost:8000
```

Or from the command line, which runs the same code:

```bash
.venv\Scripts\python src\generate.py            # prints a run id
.venv\Scripts\python src\judge.py    <run_id>
.venv\Scripts\python src\detect.py   <run_id>
.venv\Scripts\python src\evaluate.py <run_id>
```

Every script defaults to the most recent run when given no id.

---

## The web interface

| Page | What it is for |
|---|---|
| **New run** | All configuration, in the left sidebar. Choose the generator, the judges, the tools, and the sample source. |
| **Previous runs** | Every run ever made. Nothing is overwritten. |
| **Run detail** | One run's samples, both judges' verdicts side by side, its results, and the source of any sample. |
| **Tools** | What each detector does and whether it can run in this environment. |
| **Comparison** | Mean scores per tool across every evaluated run. |

The interface is a thin layer over the same modules the CLI uses — it contains
no benchmark logic, so the browser and the terminal cannot produce different
numbers.

---

## Analysing real code instead of generated samples

A GitHub repository or a local folder can be fed through the same tools:

```bash
.venv\Scripts\python src\ingest.py https://github.com/owner/repo
.venv\Scripts\python src\ingest.py C:\path\to\project
```

**Ingested code has no ground truth.** Nobody has labelled that repository, so
those samples are stored with `label: "unknown"` and `evaluate.py` omits
accuracy, precision, recall, F1, FPR and MCC for them. What remains valid is
Cohen's kappa between tools, and each tool's raw flag rate. Ingested code is
never executed — every tool reads it as text.

---

## Storage layout

```
runs/run_2026-08-10_17-30-00/
    run_metadata.json                what was configured, and how it ended
    01_generation/
        samples/sample_001.py ...    the programs
        generation_manifest.json     intended label + CWE per sample
        generation_rejected.json     pairs that failed the structural checks
    02_judgement/
        judgement_<model>.json       one file per judge
        judgement_comparison.json    both judges side by side
        judgement_dropped.json       what was discarded, and why
        verified_manifest.json       the survivors
    03_detection/
        detection_<tool>.json        one file per detector
    04_evaluation/
        evaluation_report.json
        comparison_report.json
        comparison_report.txt        the same report as readable text

logs/run_2026-08-10_17-30-00.log     every line the run printed
```

### Why sample files are *not* descriptively named

Everything in a run is named to be self-describing **except the sample files**,
which stay neutral (`sample_007.py`). This is a correctness requirement.

A detector is handed these files. If a filename were
`sample_007_CWE-89_vulnerable.py`, every tool — and every LLM reading a path —
could score perfectly without analysing a line of code, and the benchmark would
measure nothing. The descriptive metadata lives in the manifest, which the web
interface joins against the sample id, so a human browsing the site still sees
rich labels.

---

## Design decisions worth discussing

**Paired generation.** Each vulnerable program is generated together with a
safe twin differing only in the minimal change needed to fix the flaw. Were the
two classes generated independently they would differ in style, length and
structure as well as in security, and a detector could score highly by
recognising which prompt produced a file rather than by understanding the
vulnerability.

**Exactly one vulnerability per sample**, enforced in three places: the
generation prompt states it; `generate.py` rejects structurally broken pairs;
and each judge is asked to *count* the vulnerabilities it finds, with any
sample not reporting exactly one being dropped. A program with two flaws breaks
per-CWE scoring — a detector that finds the unintended flaw is scored right on
the label and wrong on the category.

**Leakage controls.** The generator is forbidden from using words like
"vulnerable" or "sanitised" anywhere in either version, and `find_leakage()`
splits identifiers on underscores and case transitions before checking — so
`safe_url`, `is_safe` and `isSafe` are all caught, not just the bare word.
Samples are shuffled before being named, so neither filename nor file order
reveals anything. Each sample reaches a judge or detector as an independent
request, so no verdict can influence another.

**Local generation, remote judging.** Generation is by far the most token-hungry
stage — two complete programs per call — so it runs locally through Ollama at no
per-call cost. That also guarantees the model that *writes* a sample is never
the model that *certifies* it.

**Random CWE selection.** `config.CWE_SELECTION` defaults to `"random"`, as
requested. The trade-off is real: random draws leave some of the sixteen
categories empty and double others at small N, making the per-CWE table noisy.
`"cycle"` guarantees balance instead.

---

## Known limitations

- **Generator quality drives everything downstream, and CPU-only inference
  caps it.** Both models were benchmarked on the development machine (16 GB
  RAM, Intel Iris Xe graphics, no CUDA device):

  | model | time per pair | memory | stability |
  |---|---|---|---|
  | `qwen2.5-coder:3b` | ~45 s | 1.9 GB | stable |
  | `qwen2.5-coder:7b` | ~457 s | 5.5 GB | intermittent OOM |

  The 7B model is **ten times slower** here, not the two to three times a rough
  estimate suggests, and with ~3 GB of RAM free it failed to allocate its KV
  cache on several consecutive pairs. Its output was not visibly better either:
  still short programs, still a `# Vulnerable SQL query` comment. So 3B is the
  default, and leakage is handled deterministically instead (see below).
  On a CUDA machine, `GENERATOR_MODEL=qwen2.5-coder:7b` is the better choice
  and needs no code change.

- **Both generators annotate their own work.** They write comments naming the
  flaw despite an explicit instruction not to. This cannot be prompted away
  reliably, so `scrub_leaking_comments()` removes such comments before
  validation. Comments carry no semantics, so deleting one cannot change
  whether a program is vulnerable. Identifier leaks (`safe_url`) are *not*
  auto-fixed — renaming could change behaviour — and remain a reported flag.

- **Judging is model agreement, not human review.** Two models agreeing is much
  weaker evidence than a human check, though far stronger than trusting the
  generator alone. A human-verified subset is the natural next step.

- **Dataset sizes are small.** At tens of samples a single flipped verdict moves
  F1 by several points, and the ML arm is severely data-starved. Cross-run
  consistency on the Comparison page is more informative than any single run.

- **Semgrep needs network access on first run** to fetch its `p/python`
  ruleset from semgrep.dev; it is cached afterwards. Releases before ~1.173 had
  no Windows build at all, so pin at least that version.

- **The three static tools are not equally independent, and that is by design.**
  Measured over the same 15 samples: Bandit vs Ruff κ = 0.706, Bandit vs Semgrep
  κ = 0.545, Ruff vs Semgrep κ = 0.444. Ruff shares Bandit's ruleset, so where
  those two disagree the rule cannot be the explanation — which is what lets a
  rule's limits be told apart from a tool's.

- **The web app is a local research tool.** No authentication, no CSRF
  protection, one concurrent run. It binds `127.0.0.1` and must not be exposed
  to an untrusted network.

---

## Next steps

1. Scale to a few hundred samples and hand-verify a random subset, to measure
   how well the judges' agreement tracks human judgement.
2. Add Semgrep as a third static analyser inside the Docker image.
3. Add CodeBERT / GraphCodeBERT as a heavier ML arm alongside the two current
   models, keeping the same out-of-fold protocol.
4. Add the adversarial transformation layer (AST-based, so
   semantics-preservation is guaranteed by construction).

---

## Documentation

- **[docs/SETUP.md](docs/SETUP.md)** — installing and running on another machine.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — module-by-module design,
  data formats, and how to add a detector.
