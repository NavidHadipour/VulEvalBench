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
 1. generate.py     local Ollama model writes the samples -- in vulnerable/safe
      |             pairs by default, or singly (see Twins, below)
      |                 runs/<run_id>/01_generation/
      v
 2. judge.py        1-3 independent judges verify each label;
      |             samples they cannot confirm are DROPPED
      |                 02_judgement/verified_manifest.json
      v
2b. review          OPTIONAL, in the browser. The run pauses and a human
      |             keeps or restores samples with the verdicts in front of
      |             them        02_judgement/selected_manifest.json
      v
2c. balance.py      OPTIONAL, chosen in the same panel. Drops samples until
      |             the set is even by label and/or by CWE class
      |                 02_judgement/balanced_manifest.json
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

Stages 2b and 2c exist only in the browser; the four numbered modules are the
pipeline proper and run identically from the command line. Nothing is ever
deleted — each stage writes its own manifest beside the last one, so a final
count of "46 of 59" can always be decomposed into what the judges dropped,
what a human dropped, and what balancing dropped.

A run can be **stopped** at any point — see
[Starting and stopping a run](#starting-and-stopping-a-run).

### Why there is a judging stage

The single biggest weakness of the earlier draft was that ground truth came
from the generator: a sample was labelled "vulnerable" because the model had
been *asked* to write a vulnerability, not because anyone confirmed it had.
Every downstream metric silently inherited that assumption.

Stage 2 removes it. Up to three judge models — different models, on different
hardware, from a different vendor than the generator — read each program cold,
without being told the intended label, and say what they see. A sample enters
the benchmark only if their independent reading matches the generator's intent.
Everything else is discarded.

The result is a smaller but defensible dataset in which every surviving label
has been agreed on by a party that did not write the code.

#### Two judges, or three

Two judges cannot resolve a disagreement: there is no way to tell which of the
two was wrong, so the strict `all` rule has to treat every dispute as fatal and
the dataset pays for each one. A third slot makes `JUDGE_CONSENSUS =
"majority"` meaningful, where two agreeing judges outvote one dissenter. That is
weaker verification and has to be reported as such — a majority-admitted
sample has one judge on record calling the label wrong — but it stops a
single harsh or miscalibrated model from deciding the size of the dataset.

The third slot is empty by default, because under `all` a third judge can only
ever shrink the admitted set. Judges that abstain (a failed call) are not
counted in either rule: a majority of three where one was unreachable is two of
two, not two of three.

### Reviewing samples by hand

**Review samples after judging** in the sidebar (on by default) stops the
pipeline once the judges have finished and lists every sample on the page.
Each row shows its intended label, whether the judges admitted it, and opens
to show each judge's verdict and reasoning above the source. The detectors are
then scored on exactly what you keep.

It runs for **every source** — generated, ingested from GitHub or disk, or an
existing run being re-analysed — and whether or not judging ran. On an unjudged
run there are simply no verdicts to show and every box starts checked.

**After judging, not before**, and that ordering is the whole point. Reviewing
first would let a judge silently undo a human decision, and would force the
reviewer to decide with no verdict in front of them. Reviewing last means the
boxes start from the judges' own decision, so you only touch the cases you
disagree with — in **either** direction:

- unticking a sample the judges admitted drops it;
- ticking one they rejected **restores** it.

A restored sample carries a label no judge agreed with. That is a legitimate
override — judges are wrong often enough that the comparison page exists — but
it is written into the run log by name (`Restored against the judges'
decision: ...`) rather than passed over silently. **As judged** resets every
box to the judges' decision and discards the overrides.

Three more things worth knowing:

- **Nothing is deleted.** The kept set goes to
  `02_judgement/selected_manifest.json`, beside the judgement it followed, and
  the ids land in run metadata under `manual_selection`. Both
  `generation_manifest.json` and the judges' `verified_manifest.json` stay
  intact, so "46 of 59" can always be decomposed into what the judges dropped
  and what a human did afterwards.
- **The selection has the last word.** `read_verified_manifest()` returns the
  selection when one exists, then the judges' manifest, then the full
  generated set — so detection and evaluation pick it up with no further
  wiring.
- **Dropping one half of a pair is allowed, and flagged.** Each vulnerable
  sample has a safe twin differing only by the flaw; keeping one without the
  other removes its control. The page names the split pairs rather than
  preventing it, because there are legitimate reasons to do it — but it
  belongs in the write-up if you do.

A review left unanswered for 30 minutes stops the run rather than holding the
worker thread forever. The samples and the judgement are already on disk, so
the run can be picked up again from the **existing run** source.

Reviewing by hand is a defensible step for a benchmark, but it is also the
point where the author's expectations can enter the dataset. Dropping samples
because the generator produced something malformed is housekeeping; dropping
them because a detector you favour would miss them is not, and the stored
`manual_selection` record exists so that distinction can be audited later.

### Balancing the dataset — stage 2c

Two checkboxes in the same review panel, beside the kept set and a live
composition line reading `16 samples · 50.0% vulnerable · 8 complete pairs ·
0 orphans`.

**Why it is offered at all.** Two kinds of imbalance distort different numbers.
Label imbalance distorts *accuracy*: on a set that is 70% vulnerable, a
detector answering "vulnerable" to everything scores 0.70 and has detected
nothing. MCC and Youden's J resist that, which is why they are reported, but
accuracy is the number a reader reaches for first and it should not be quietly
flattering. Class imbalance distorts the comparison *between* detectors: if
SQL injection is half the dataset, the ranking is largely a ranking of
SQL-injection performance wearing a general-purpose label.

**It only ever drops, and it is off by default.** Up-sampling by duplication
would put the same program in the dataset twice and let a detector be scored
twice on one answer; synthesising replacements would mean the generator
deciding what the benchmark contains, which is the thing judging exists to
prevent. So balancing discards real, correctly-labelled samples — on a small
run, possibly most of them — and a report is printed *before* anything is
dropped, so the cost is visible in advance rather than discovered afterwards.

**Orphans go first.** An intact paired dataset is already 50/50 by
construction, so label imbalance only arises once something has broken pairs:
judging rejecting one half, a reviewer dropping one half, or the VUDENC
importer, whose filters admit halves unevenly. An *orphan* is a sample whose
twin is gone. Dropping those first repairs the pair structure while correcting
the count, which is strictly better than breaking a good pair to fix a number.
On a 12-pair VUDENC run that meant 20 samples at 60.0% vulnerable with 4
orphans became 16 samples at 50.0% with the same 8 complete pairs and no
orphans — the four orphans and nothing else.

**Why the choice is made here and not in the sidebar.** How much balancing
costs depends entirely on what survives the review, so a choice made before
the run would be made blind. A run that skips the review is therefore never
balanced, because it was never asked.

#### Judges that do not write prose

Two of the selectable judges are not chat models. **Jev** (hosted by TypeSafe)
and **Laya** (Apache-2.0, 0.4B, runs locally) are *System One* models: an
encoder reads the program once and answers typed questions — pick one of
these options, score on this rubric — returning calibrated probabilities. They
generate no text at all.

They are offered as judges because this stage's questions are exactly that
shape. The judge does not need to explain anything: it answers whether the
program is vulnerable, which category from a fixed list, and how many flaws it
can justify. The chat judges only produce that shape because guided decoding
forces them into a JSON schema; these models answer it natively.

Two differences matter when reading a run:

- their `confidence` is a probability the model actually assigned over its own
  options, where a chat judge's is a figure it wrote in prose. Since
  `JUDGE_MIN_CONFIDENCE` drops samples below a threshold, that threshold does
  not mean the same thing for the two kinds of judge. It is *not* validated
  calibration: the field read is `answer_confidence` (max p), whose own
  documentation says the calibration property holds only once temperatures have
  been fitted on held-out data for that checkpoint and option count — and
  loading Laya warns that this checkpoint ships invalid temperatures for the
  11-or-more-options bucket, which is where the 17-option CWE question lands.
  The warning is written into the run log rather than left on stderr;
- there is no `reasoning` to read. The judgement files carry the answer
  probabilities in that field instead, marked as derived, and the comparison
  table labels the column `[typed]`.

Neither model was trained on source code — Laya's card names English text,
guardrails and email triage — so treat a poor result as evidence about how
such a model transfers to code, not as a defect. That comparison is the reason
to run them.

Jev needs a key as `typesafe.key` in `api-keys.txt` (early access, from
their waitlist).
Laya needs `pip install laya`, which pulls in torch. Without either, the model
appears in the dropdown disabled, with the reason.

Both are also selectable as **detectors**, in the LLM arm — see
[Typed models as detectors](#typed-models-as-detectors). A model may not be
used in both roles on the same run, and the pipeline refuses the combination
rather than warning about it.

---

## The three detector arms

| Arm | Tools | What it actually does |
|---|---|---|
| **Static analysis** | Bandit, Ruff, Semgrep | Bandit matches AST patterns; Ruff reimplements the same ruleset in a separate engine; Semgrep adds taint analysis, tracking whether untrusted input actually reaches a sink. Fast, deterministic, no training. |
| **Machine learning** | *Pretrained:* BiLSTM (Farasat & Posegga 2024), LSTM (VUDENC 2022). *Trained here:* TF-IDF + LogReg, embeddings + LogReg | Two families, deliberately kept side by side. The pretrained pair was fitted on security-fix commits from thousands of real Python projects and has never seen this benchmark — same corpus and preprocessing, differing only in the network. The trained pair learns from the run itself, out of fold, and differs only in the representation. The two families are **not on the same protocol** — see below. |
| **LLM** | Any chat model, local or remote; plus the typed models Jev and Laya | A chat model reads the program and reasons about it in prose. A typed model reads it once and returns a probability distribution over the answers, writing nothing. |

Every tool returns the same verdict shape, so `evaluate.py` scores all of them
with one code path and none gets a bespoke advantage.

### Typed models as detectors

Jev and Laya can be selected in the LLM arm as well as in the judge slots, and
as detectors they are tools under test like any other: scored against the
verified labels, printed in the same comparison table.

The detector path asks **two** questions where the judging path asks three.
The third is the vulnerability count, which exists only to serve
`REQUIRE_SINGLE_VULNERABILITY` — an admission rule about what belongs in the
dataset. A detector admits nothing, so asking it would cost latency and tokens
for an answer that is then discarded.

That difference matters for reading the results. The count question is what
crippled Laya **as a judge**: it answered "two or more vulnerabilities" for 47
of 59 programs and never once answered "none", so every safe sample failed the
rule regardless of how well it had labelled it. As a detector that bottleneck
is simply absent, and the score reflects its label call alone.

The questions are deliberately *not* shared with `judge.py`, even though the
label and CWE wording is nearly identical. Nothing under `detectors/` may
import the module that produces the ground truth, or a change to the admission
criteria could silently move the thing being measured.

**A model scored on a run it judged is reported, not refused.** The number
looks entirely valid while being inflated: a model is unusually good at
finding the vulnerability it certified minutes earlier, and no metric in its
row shows a symptom. `detect.py` used to raise over it. It now logs the
overlap and records it on the round and in the run metadata
(`detectors_that_also_judged`), because the comparison is one a reader may
legitimately want — the same model as judge and as tool under test is a
question about that model — and because refusing at stage 3 discarded a
generation and a judging pass already paid for. `DEFAULT_LLM_DETECTOR` is
still deliberately not a default judge, so the overlap only happens on
purpose.

**The generator may not judge or be scored on its own run.** This one *is*
refused, at `/api/run`, before a call is paid for. It is a different thing in
kind: the generator did not read the program, it wrote it knowing where it
put the flaw, so asking it to confirm the label or find the vulnerability
measures recall of its own intent and nothing else.

Two practical notes. Laya runs locally and the arm drops to one sample at a
time for it: four threads on one CPU contend rather than overlap, measurably
slower than serial. And the LLM arm is now marked available when *any* of its
backends is reachable — Judgment API, Anthropic, a local Ollama tag, or a typed
model — because Laya needs no credential at all and gating the whole arm on the
Judgment API key hid it from users who could have run it.

### Three things that must be said about the comparison

These are real methodological asymmetries, not footnotes. Any write-up has to
state them.

1. **The ML arm contains two families measured under different protocols,
   and printing them in one ranking without saying so is misleading.**

   *Pretrained* (`ml_farasat`, `ml_vudenc`) are asked a question exactly as
   Bandit or an LLM is: no training, no knowledge of this dataset. Their
   numbers are directly comparable with the other arms.

   *Trained here* (`ml_tfidf`, `ml_embedding`) learn from the same tens of
   samples they are scored on, so every prediction is made **out of fold** —
   the model that labelled a sample was fitted without ever seeing it.
   Scoring on training data would report memorisation as detection.

   Both are kept because they answer different questions. The pretrained pair
   asks "does a published detector transfer to this benchmark?". The trained
   pair asks "how much signal is learnable from these samples at all?" — a
   lexical model scoring well would mean the generator leaves surface
   giveaways, and one scoring at chance is evidence the pairs are lexically
   balanced. That is a property of the *benchmark*, not of any tool, and no
   unsupervised detector can report it.

   **The pretrained pair additionally answers only five of the sixteen
   categories.** They have a head for CWE-89, 79, 78, 94 and 22 (plus CWE-601
   and CWE-352, which this benchmark does not generate). For the other eleven
   they have no detector at all and answer "safe" every time — not because the
   code is clean but because nobody trained them on the question. On the
   59-sample reference run only 18 samples fell inside their competence.

   Those result files carry a `coverage` block naming which samples were in
   scope, and it must be quoted next to the score. Note this is *not* the
   `coverage` field `evaluate.py` computes, which means "fraction of samples
   answered at all" and is 1.0 here. The trained pair has no such limit: it
   learns whatever categories the run happens to contain.

   **A result file that carries `substituted: true` is not a result for the
   model its row is named after.** The pretrained slots can be pointed at
   weights other than the published ones, and when that happens the detector
   renames itself and records the substitution and its reason in every file it
   writes. Read that field before quoting the row, and quote it alongside the
   score. See `config.py` for which substitutions are configured here.

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

Three layers, all standard.

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

**3 — CWE identification accuracy**, scored only on true positives: among the
samples a tool correctly flagged, how often did it also name the right
category. Asking whether a *missed* vulnerability was correctly categorised is
meaningless, and scoring the category on a false positive would reward a lucky
guess about a flaw that is not there.

> **There is deliberately no combined score.** An earlier version ranked tools
> by a weighted sum of F1, (1 − FPR) and CWE accuracy. It was removed:
> collapsing three different questions into one number hid which of them a tool
> actually failed, and the weights were an editorial choice that silently
> decided the ordering. Tables are sorted by F1 so they are stable and
> readable, but that is a sort order, not a score — F1 ignores true negatives
> entirely, so the tool at the top may still be the one crying wolf most often.
> Read the whole row.

**4 — Localisation**, where the ground truth records it. A sample may carry
`vulnerable_lines` — the 1-based lines known to hold the flaw — and a detector
may answer with `predicted_lines`. Where both exist, true positives are also
scored on whether the detector pointed within
`config.LOCALISATION_TOLERANCE_LINES` (2) of a real one.

This matters as soon as real code is scored. On a 40-line generated program
built around one planted flaw, "is this file vulnerable" and "where is the
flaw" are the same question. On a 150-line file from a real repository where
two lines are wrong they are not, and a file-level verdict cannot tell a tool
that found the bug from one that flagged an unrelated line and was counted
correct. Only runs built from the VUDENC dataset carry this ground truth today
— it comes from the fix commit's own diff — so every generated run reports
`localisable: 0` and the block is inert rather than scored as zero.

Three counts are reported rather than one ratio, because a single number
cannot tell the two ways of scoring zero apart:

| field | meaning |
|---|---|
| `localisable` | true positives whose ground truth records the lines — the denominator |
| `localisation_reported` | …of those, how many the detector named any line for |
| `localised` | …of those, how many named a line within tolerance |

Accuracy 0.0 with `localisation_reported` 0 means the detector never reports
positions at all — true of the ML arm, which answers per file, and of the LLM
arm as currently prompted. The same 0.0 with that count high means it does
report them and they are wrong. Those are opposite findings and a write-up
must not merge them.

`localisation_precision` is deliberately **not** computed. It would need every
flawed line in each sample to be known, and the ground truth records the lines
a fix commit touched — a lower bound, not a complete list. A precision scored
against an incomplete key would punish a tool for finding a real flaw nobody
had labelled.

**5 — Pairwise agreement** between tools, as Cohen's kappa. This is the only
measure here that needs no ground truth, which makes it the one that still
works on ingested GitHub code. Two tools that agree strongly are largely
redundant; two that disagree are finding different things.

---

## Installation

Roughly 15 minutes on a machine with none of this installed, most of it
waiting for a model download. Tested on Windows 11 with Python 3.12; on Linux
and macOS replace `.venv\Scripts\` with `.venv/bin/` throughout.

| You need | Why | Check |
|---|---|---|
| Python 3.10+ | runs everything | `python --version` |
| ~6 GB free disk | model weights plus dependencies | |
| Git *(optional)* | only to ingest GitHub repositories | `git --version` |
| An OpenAI-compatible LLM API | judging, LLM detection, embeddings | step 4 |

The judging API is the one external service this project cannot run without.
Generation, static analysis and the ML arm all run locally.

### 1. Dependencies

```bash
git clone https://github.com/NavidHadipour/VulEvalBench.git
cd VulEvalBench
python -m venv .venv
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
```

Verify the three static analysers really installed:

```bash
.venv\Scripts\bandit --version
.venv\Scripts\ruff --version
.venv\Scripts\semgrep --version
```

> **Semgrep's version floor is deliberate.** Releases before ~1.173 shipped no
> Windows executable at all — `pip install semgrep` reported success and
> installed nothing runnable, which is why an earlier draft ran Ruff in its
> place. Semgrep also fetches its `p/python` ruleset from semgrep.dev on first
> use and caches it, so the first run on a machine needs network access.

### 2. The local generator

```powershell
winget install Ollama.Ollama        # macOS/Linux: https://ollama.com/download
ollama pull qwen2.5-coder:3b        # 1.9 GB -- the default
```

**Pick the tag by your hardware, not by model size.** See
[Known limitations](#known-limitations) for the measurements: on a CPU-only
machine the 7B model is ten times slower, not two to three times, and runs out
of memory. Check with `ollama ps` while a model is loaded — a `PROCESSOR`
column reading `100% CPU` means no GPU offload, and 7B will be painful. With a
CUDA GPU, prefer `qwen2.5-coder:7b` and set it with no code change:

```bash
set GENERATOR_MODEL=qwen2.5-coder:7b        # Windows
export GENERATOR_MODEL=qwen2.5-coder:7b     # Linux/macOS
```

Confirm the server is up with `ollama list` and
`curl http://localhost:11434/api/tags`. If nothing answers, `ollama serve`
(on Windows the installer normally registers a background service).

### 3. Pretrained ML weights

The ML arm's pretrained pair runs published models rather than training
anything. Their weights (~30 MB) belong to their original authors and are not
redistributed here, so fetch them once into the gitignored `models/`:

```bash
.venv\Scripts\python src\fetch_models.py        # --force to re-download
```

It prints a line per file and says which detectors ended up usable.

### 4. Credentials

One file in the project root, `api-keys.txt`, gitignored and never copied
into a Docker image. A template is provided — copy and fill in rather than
writing from scratch:

```bash
copy api-keys.txt.example api-keys.txt   # Windows
cp   api-keys.txt.example api-keys.txt   # Linux/macOS
```

Each line names the endpoint it belongs to, the field, and the value:

```
judgment.url   = "https://llms.example.edu"
judgment.key   = "sk-..."
anthropic.key  = "sk-ant-..."
typesafe.key   = "..."
```

| entry | needed for | fields |
|---|---|---|
| `judgment` | **required** — judges, LLM detectors, embeddings | `url` *and* `key` |
| `anthropic` | optional — Anthropic models in either role | `key` |
| `typesafe` | optional — the Jev typed model in either role | `key` |

Those three names are reserved because the shipped providers read them.
**Any other name carrying both a `url` and a `key` becomes a provider you can
register models against**, so another API needs no code change — see
[Adding another API](#adding-another-api).

You never have to edit the file by hand. The **Models** page lists every
credential with its value masked, and adds, replaces or removes one; so does
`python src/manage_models.py keys`.

> **Upgrading from an earlier checkout?** The three files this replaces —
> `api-key-judgment.txt`, `api-key-llm.txt` and `api-key-typesafe.txt` — are
> still read wherever `api-keys.txt` has no entry, so nothing breaks if you
> change nothing. The Models page offers a one-click import that copies them
> across and leaves the originals in place.

### Adding another API

A credential entry with both a `url` and a `key`, under any name you like,
becomes a selectable provider:

```
openrouter.url = "https://openrouter.ai/api"
openrouter.key = "sk-or-..."
```

It is called with the same client as the judgment endpoint, so the API must
accept `POST <url>/v1/chat/completions` with a bearer token and honour
`response_format: {"type": "json_schema", …}` — vLLM, LiteLLM, Ollama's own
compatible API and most commercial providers all do. List what one offers
with `curl -H "Authorization: Bearer sk-..." https://your-endpoint/v1/models`,
then register the model ids against that provider on the **Models** page
rather than editing `config.py` — see
[Using your own models](#using-your-own-models).

The launch page shows a green dot per credential it found, so a misconfigured
file is visible before a run rather than after.

**Laya**, the local typed model, is the one optional *package* rather than a
key: `pip install laya` pulls in torch and transformers (~3 GB installed) and
fetches its ~800 MB checkpoint from Hugging Face on first use. Nothing else in
the pipeline needs torch, which is why it is commented out in
`requirements.txt`. Without it the model appears in both dropdowns disabled,
with the reason.

### 5. Verify, then run

```bash
.venv\Scripts\python src\check_setup.py
```

This checks every prerequisite — Python version, packages, the three static
analysers, the credential files, the Ollama server and model, the CWE
hierarchy cache, and that `runs/` and `logs/` are writable — and prints a
checklist in which every `[FAIL]` names the command that fixes it:

```
  [ ok ]  Python                      3.12.3
  [ ok ]  tool: semgrep               ...\.venv\Scripts\semgrep.exe
  [FAIL]  Ollama model                'qwen2.5-coder:3b' not pulled -- run: ollama pull qwen2.5-coder:3b
  1 required check(s) failed -- fix the [FAIL] lines above.
```

Its exit code is 0 only when everything required passes, so it also works as a
gate in a script. Then:

```bash
.venv\Scripts\python web\app.py           # open http://localhost:8000
```

Or from the command line, which runs the same code:

```bash
.venv\Scripts\python src\generate.py            # prints a run id
.venv\Scripts\python src\generate.py 200        # 200 samples; no cap here
.venv\Scripts\python src\judge.py    <run_id>
.venv\Scripts\python src\detect.py   <run_id>
.venv\Scripts\python src\evaluate.py <run_id>
```

Every script defaults to the most recent run when given no id.

### 6. Docker *(optional)*

```bash
docker compose up --build        # then open http://localhost:8000
```

The Docker files are kept out of the repository (see `.gitignore`), so this
applies only to an installation that has them. Two things to know:
**Ollama is not in the container** — it runs on the host and the container
reaches it through `host.docker.internal`, so set `OLLAMA_BASE_URL` and make
sure Ollama listens on all interfaces (`OLLAMA_HOST=0.0.0.0`) rather than only
on loopback. And `runs/` and `logs/` are bind-mounted so results survive
`docker compose down`, while credential files are mounted read-only at run
time and never baked into an image layer — a secret in a layer persists in the
image history even if a later layer deletes it.

### Troubleshooting

**"Could not reach Ollama"** — the server is not running. `ollama serve`, then
check `curl http://localhost:11434/api/tags`.

**"The 'judgment' endpoint is missing: url, key"** — `api-keys.txt` has no
`judgment.url` / `judgment.key`, or the file is not in the project root. Set
them on the **Models** page, or with
`python src/manage_models.py keys --set judgment.key --value sk-...`. If you
are upgrading from an older checkout, the same page imports the legacy
`api-key-*.txt` files.

**Generation produces no usable samples** — almost always the tag in
`config.GENERATOR_MODEL` is not one `ollama list` shows. The per-sample
rejection reasons are in `logs/<run_id>.log`.

**Every sample is dropped in judging** — expected with a small generator.
`runs/<run_id>/02_judgement/judgement_dropped.json` records the failed rule
per sample. Either move to a larger generator or relax `JUDGE_CONSENSUS` from
`"all"` to `"any"`.

**"The ML arm is skipped: only one class is present"** — judging admitted
samples of a single label, so there is nothing to classify.

**"A run is already in progress" with nothing running** — a worker thread
parked on a review nobody answered, or an older `web/app.py` still listening.
Werkzeug sets `SO_REUSEADDR`, so a second server can bind port 8000 and
*both* will answer, serving different code. Check with
`netstat -ano | findstr :8000` and kill the strays before restarting.

**"not enough space on the disk" during `ollama pull`** — a failed pull leaves
a `-partial` blob that still occupies the space. Delete
`~/.ollama/models/blobs/*-partial*`.

**Windows console `UnicodeEncodeError`** — the reports are deliberately plain
ASCII to avoid this. Keep added output ASCII, or set `PYTHONIOENCODING=utf-8`.

---

## The web interface

| Page | What it is for |
|---|---|
| **New run** | All configuration, in the left sidebar. Choose the generator, the judges, the tools, and the sample source. |
| **Previous runs** | Every run ever made. Nothing is overwritten. |
| **Run detail** | One run's samples, every judge's verdict side by side, its results, and the source of any sample. |
| **Models** | Register, hide or remove models per installation — see [Using your own models](#using-your-own-models). |
| **Tools** | What each detector does and whether it can run in this environment. |
| **Comparison** | Mean scores per tool across every evaluated run. |

The interface is a thin layer over the same modules the CLI uses — it contains
no benchmark logic, so the browser and the terminal cannot produce different
numbers.

### Three vendors answering the same prompts

The judge and LLM-detector pick lists offer four families: the Passau-hosted
open weights, Anthropic's Claude models, OpenAI's GPT models, and the typed
non-generative models. The third of those is there for a specific reason. The
ground truth in this benchmark is whatever the judges agreed on, so a
vulnerability that one vendor's models systematically misread becomes a
labelling error that no amount of cross-checking *within* that vendor can
find. With three vendors available, a disagreement between two of them can be
read against the third.

The four OpenAI entries are the tiers OpenAI's own guide tells callers to use,
and they span a hundredfold in price. That matters here more than it looks,
because judging is the expensive stage — one call per sample per judge, and a
full generation is around 750 samples:

| model | $/1M in | $/1M out | OpenAI's description | judging pass, 750 samples |
|---|---|---|---|---|
| `gpt-6-astra` | $10 | $50 | highest intelligence | ~$15 |
| `gpt-6.1-sol` | $2 | $10 | balanced speed, cost, intelligence | ~$3 |
| `gpt-6.1-terra` | $1 | $5 | fast and efficient | ~$1.50 |
| `gpt-6-luna` | $0.10 | $0.50 | fastest and most cost-effective | ~$0.15 |

The last column is the list price applied to **measured** token counts, not an
estimate: a `gpt-6-luna` judging pass over this project's own samples cost 602
input and 284 output tokens per verdict, so 750 samples is ~452K input and
~213K output. Astra as one of three judges on a full run is therefore a
deliberate purchase, which is why no default points at it.

For comparison, the Passau-hosted `gemma4-31b-it` spent 3,742 output tokens
per verdict on the same job — thirteen times as many as Luna — because it
writes its reasoning out in full. The endpoint is free to this installation,
so that costs nothing but wall-clock time; it is worth knowing before
assuming a paid model will be the slow one.

`gpt-6-sol` also exists and works, but OpenAI's guide omits it from the ids it
tells you to set and points existing callers at `gpt-6.1-sol`, so it is not
listed. Any id the API will accept can still be added on the Models page
without touching the code.

### Starting and stopping a run

**One button, two modes.** While a run holds the worker there is nothing else
`Run pipeline` could usefully do — it used to sit there disabled — and
stopping is wanted at exactly the moment the run is going, so it becomes
`Stop run`. The mode is driven by polling `/api/status` rather than by what
was last clicked, so it is correct after a page refresh, when a run ends by
itself, when a run was started from another tab, and when the server goes
away.

**Stopping is cooperative, not a kill.** The worker cannot be interrupted
mid-write without leaving a half-written run directory, so a stop is honoured
at the next point where the directory is consistent: generation stops after
the request already in flight, judging after the current judge, detection
after the current detector, and a run parked on the review panel stops
immediately. What already exists is **kept and written** — the metadata
records `status: "stopped"`, `stopped_early: true` and the
`requested_samples` originally asked for, so a 1-of-40 run is on disk and
explicable rather than lost. Evaluation still runs over whichever detectors
finished.

A stopped run ends in its own terminal state, `stopped`, which is neither
`done` nor `error`: nothing failed, so reporting it as an error would be
wrong, and it did not finish, so reporting it as done would be worse.

**A progress meter for the model in flight.** Under the live log sits a bar
reading, say, `gemma4-31b-it  412 / 753 samples -- 21m elapsed -- ~17m left
(55%)`. It exists because the log alone could not answer the question people
actually ask of a long stage: one judge over 750 samples is hours of work, and
the log could only announce which model had started and then fall silent until
it finished. Silence of that kind is indistinguishable from a hang.

Counting every sample into the log instead would cost 750 lines and bury
everything worth reading, so the count travels on its own channel and the
browser redraws it in place; the log keeps one milestone line per tenth of the
way, which is enough that `logs/<run_id>.log` still shows the shape of the run
after the process is gone, and enough to follow a command-line run.

The meter names the model, not the stage, so a three-judge pass is legible as
it moves from one judge to the next, and the same bar covers generation (per
request), the LLM detector arm (per model) and the pretrained ML detectors.
The static analysers have no bar: they run one subprocess over the whole
sample directory, so there is no per-sample position to report.

Two details are deliberate. The elapsed clock is computed when the figure is
*read*, not when the last sample finished — so a model stuck on its first
sample shows a count frozen at 0 beside a clock that keeps moving, which is
what distinguishes "slow" from "hung". And the estimate of what is left is
withheld until a few samples and a whole second are in, because extrapolating
from one sample of a four-thread pool produces a number worse than none. It is
an extrapolation from the rate so far, quoted in minutes, and it moves.

**Run size: up to 1000 from this page, uncapped from the command line.** An
over-size request is **refused with a message**, not quietly reduced:

```
Run size must be between 1 and 1000; you asked for 2000.
The command line has no cap: python src/generate.py 2000
```

Silently clamping 2000 to 50 is the worst of the options — the run looks
successful and the number in the write-up is wrong. (A bug worth recording:
the job used to be marked `starting` *before* the request was validated, so
every refused request left the worker permanently claimed and blocked all
later runs with "a run is already in progress" until the server was
restarted. Validation now happens first and the worker is claimed afterwards,
under the lock.)

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

### Real code that *is* labelled: the VUDENC dataset

The generated benchmark has complete labels and synthetic code; an ingested
repository has real code and no labels. VUDENC's dataset closes the gap. Each
record is a GitHub commit that fixed a vulnerability, carrying the file as it
stood *before* the fix together with the exact lines the fix replaced — so it
is real-world code, labelled from a human developer's own security fix, and
naturally **paired** in the same minimal-diff way:

```
vulnerable half = sourceWithComments               (the pre-fix file)
safe half       = that, with each badpart replaced
                  by its goodpart                  (the upstream fix)
```

Command line only, in two steps:

```bash
.venv\Scripts\python src\fetch_vudenc_dataset.py       # once, ~50 MB
.venv\Scripts\python src\ingest_vudenc.py --max-pairs 60
.venv\Scripts\python src\ingest_vudenc.py --classes sql,xss --max-pairs 40
```

This is also the only source that supplies **line-level ground truth**: the
lines the fix commit touched become `vulnerable_lines`, which is what makes
the localisation metric live.

Four things must be said about any number from such a run.

- **The labels are commit-level.** Every file touched by a security-fix commit
  inherits the label, including files changed incidentally. A real example
  from the XSS class is a `setup.py` whose only change is a version bump from
  `0.3.22` to `0.3.23b0` — the commit fixed an XSS somewhere, but this file is
  not the fix and is not vulnerable. So a label here means "a developer
  changed this file in a commit that fixed a *class* vulnerability", which is
  weaker than the generated benchmark's "this program was written to contain
  exactly this flaw". Two heuristic filters cut the obvious noise (an excluded
  path list and a relevance-token list); they are heuristics, not ground
  truth. **Running the judging stage over such a run is the recommended next
  step and has not yet been done.**
- **"Safe" means repaired, not clean.** The safe half is only known to have
  *that* flaw fixed. The file may contain others.
- **Requiring both halves to parse as Python 3 excludes most of the dataset**,
  which is largely Python 2 code from 2015–2020 repositories. That biases the
  sample toward more modern files.
- **`source` is not usable and `sourceWithComments` is.** Records carry two
  source fields and only one is code: `source` has been through VUDENC's own
  preprocessing, which strips comments *and* collapses every newline into a
  space, because their model reads 200-character windows and never needed the
  text to parse. Measured over the full dataset, 0 of 1889 files parse. This
  is also why VUDENC cannot be compared like-for-like against an AST-based
  analyser on its own representation — the representation discards the
  structure those tools need.

**Do not commit a run built this way.** The samples are real files from
hundreds of third-party repositories, each under its own licence. `runs/` is
gitignored for this reason among others.

---

## Using your own models

`config.py` ships the models this thesis was measured with, and that list is
deliberately specific: it names a university endpoint, an Anthropic account,
an early-access API and two local checkpoints. Nobody else has that
combination, and a reader who clones this framework has their own endpoint,
their own keys, and no reason to care which Qwen tag Passau was serving in
October 2026.

Editing `config.py` to fix that would be wrong twice over: it mixes one
installation's settings into the file that records what the published numbers
were produced with, and it makes every upgrade a merge conflict. So the
shipped registries stay in `config.py` as the **defaults**, and a separate
per-installation overlay lives in the gitignored `models.local.json`. Use the
**Models** page, or the CLI — both write the same file:

```bash
.venv\Scripts\python src\manage_models.py list
.venv\Scripts\python src\manage_models.py add --id qwen3-32b \
    --name "Qwen3 32B" --provider passau --roles judge,llm_detector
.venv\Scripts\python src\manage_models.py add --id llama4-scout \
    --name "Llama 4 Scout" --provider nvidia --roles judge \
    --api-model meta/llama-4-scout-17b-16e-instruct --max-tokens 4000
.venv\Scripts\python src\manage_models.py disable --id claude-haiku-4-5
.venv\Scripts\python src\manage_models.py remove --id qwen3-32b
```

The overlay can do two things: add a model the framework never heard of, and
hide a shipped one this installation has no access to. **Nothing is ever
deleted from `config.py`** — `remove` hides — so an old run's
`config_snapshot` still resolves and a published number stays explicable.

A model is registered for one or more of four roles, because the requirements
differ and a model good for one may be useless for another:

| role | what it must do |
|---|---|
| `generator` | write the benchmark samples. Needs to emit long, schema-valid JSON. Normally a local Ollama tag. |
| `judge` | vote on whether a sample carries its stated label. Needs guided JSON. |
| `llm_detector` | be scored as a tool under test. |
| `ml` | a pretrained word2vec + recurrent family, loaded from a directory under `models/`. Data only. |

Registering one model as both `judge` and `llm_detector` is normal and
allowed, and using it in both roles on the same run is allowed too — the
overlap is logged and recorded rather than refused. The generator is the
exception: a run that names it as a judge or a detector is refused outright.

Two optional fields exist for models an endpoint describes differently from
the way this framework has to name them.

**API model name.** The id may not contain a slash, because it reaches a
filename (`judgement_<id>.json`, `detection_<key>.json`) and a URL path. Most
aggregators -- NVIDIA NIM, OpenRouter -- publish namespaced ids such as
`meta/llama-4-scout-17b-16e-instruct`, which cannot be an id here. That name
goes in `api_model` instead: it is what the outgoing call carries, while the
id stays the short local handle shown in menus, reports and filenames. Blank
means the id is sent, which is the case for every shipped model.

**Response budget.** One global constant cannot fit two kinds of model.
`gemma4-31b-it` writes a verdict in about 610 output tokens; a reasoning model
spends its budget thinking first, and `qwen-agentworld-35b-a3b` needed
6,200-8,000 tokens before it emitted any JSON -- at the shared 4,000-token cap
52% of its calls returned an empty string, because the thinking arrives in a
separate `reasoning_content` field and `content` was never written. A
per-model `max_tokens` overrides the stage default wherever that model is
used. Blank means the default, which is right for almost every model.

Both are read by `providers.chat_json`, so they hold for judging, detection
and generation alike. The padding guard's learned state is keyed on the **id**
rather than on the wire name, so evidence about a model follows the name you
gave it even if the endpoint renames the model.

Two things the registry does **not** do. It does not store API keys — those
stay in `api-keys.txt`, managed by the **Keys** panel further down the same
page; a model whose credential is absent is listed as *unavailable* rather
than hidden, so a missing key reads as "not configured" instead of the model
silently not existing. And it does not make a
new *architecture* work: an `ml` entry describes where to find weights for the
pipeline `detectors/ml_models.py` already implements (word2vec embedding →
recurrent net → per-window score). A model that is not that shape needs a
detector module, not a registry row.

### When a model wastes its whole token budget

A model you point this at may do something no amount of configuration
anticipates. The one that cost the most here is worth describing, because the
framework now defends against it automatically and the defence is not
model-specific.

Grammar-guided decoding constrains *which* token may come next, but it cannot
make a model pick a useful one. A JSON grammar accepts unlimited whitespace
between a value and the next structural token, so a model that stalls there
can emit newlines until its budget is gone. Measured on `gemma4-31b-it`
judging with a 4000-token budget:

```
{
  "predicted_label": "not_vulnerable",
  "predicted_cwe": null,
  "vulnerability_count": 0,
  "confidence_percent": 100
  \n  \n  \n  \n  ...            <- 3960 more tokens of this
```

Four of five fields in about 40 tokens, then a whitespace loop instead of
starting `"reasoning"`. Because the object is never closed,
`repair_truncated_json` salvaged it and **the call succeeded** — which is why
this hid for so long. What it cost instead:

| | with the loop | guarded |
|---|---|---|
| per call | 48 s, 4000 output tokens | **1 s, 65 output tokens** |
| 753-sample judging pass | **12 h 27 m** | ~5 min |
| calls lost to the 180 s HTTP timeout | **391 of 753** | 0 |
| verdicts | — | identical on all 12 replayed |

The 391 lost calls are the real damage: a failed judgment abstains, so over
half that run's dataset was labelled by two judges instead of the three
selected.

`src/token_budgets.py` handles this without naming a model, because the next
model to do it will not be that one. Both defences key off the shape of the
response, which is observable for anything on any backend:

1. **Stop sequences, always on.** A long run of literal newlines cannot be
   legitimate output from a grammar-guided call — inside a JSON string a
   newline must be escaped as the two characters `\n`, so real whitespace
   only appears between structural tokens, where no formatter emits more than
   one. Six in a row means the model is looping. This is what protects the
   *first* run, the one that would otherwise lose half its samples before
   anything could be learned.
2. **A learned budget**, for loops that stop strings miss — a backend that
   ignores `stop`, or padding built from some other legal character. After two
   confirmed padding events a model's budget is capped at a multiple of the
   longest useful response ever seen from it *at that call size*, so a verdict
   ceiling can never truncate a generation.

The reduction is self-correcting, which it has to be: if a call exhausts a
*reduced* budget and ends on real content rather than whitespace, that is a
model being cut off rather than looping. The full budget is restored, the
model is never reduced again, and the run log says so. Guessing wrong costs
one sample and fixes itself.

What a model does is a property of that model and that endpoint, so it is
remembered in the gitignored `budgets.local.json` and a second run starts
already defended. To inspect or clear it:

```bash
.venv\Scripts\python src\manage_models.py budgets
.venv\Scripts\python src\manage_models.py budgets --reset gemma4-31b-it
```

Reset after upgrading a model behind the same id, so stale evidence does not
keep a budget capped for a model that no longer pads.

**One caveat worth stating plainly.** A cut-short response loses its trailing
field. For a verdict that is `reasoning`, and nothing is scored on it — but it
means an empty explanation column for those samples, which is a real loss when
auditing a judge by hand rather than a cosmetic one. The run log reports the
count at the end of the judging stage, so an empty column is never a mystery.
For generation a short response fails the existing incomplete-response check
and is rejected cleanly, which is what already happened to a runaway pair.

### The opposite failure: a model cut off mid-thought

A padding loop wastes the budget. A reasoning model *uses* it, and the symptom
looks nothing like the one above: the call returns `HTTP 200` with an **empty
response body** and the sample is dropped for a judgment that never arrived.

`qwen-agentworld-35b-a3b` failed this way on 52% of a 753-sample pass, with
`No parseable JSON object in model response: ''` and no other clue. The cause
is that it writes its thinking into a separate `reasoning_content` field while
`_chat_openai_compatible` reads only `message.content`. One raw call showed
`reasoning_tokens: 289` of `completion_tokens: 297` — 97% of the budget — so
when the reasoning runs past the cap, `content` is never written at all and
the entire answer is lost. Measured across the answered calls: a median of
2,273 output tokens for a verdict of about 50 characters.

The budget guard correctly refuses to shrink such a model — the tokens are
real work, not whitespace — so this is not something it can fix. The fix is
the per-model **response budget** described under the registry above. Four
samples that returned `''` at 4,000 tokens all answered correctly at 16,000,
spending 6,448–12,948 output tokens each.

**How to recognise it.** Empty responses with `finish_reason: length` and a
high `completion_tokens` count mean a reasoning model, not a broken one. Give
that model a budget of its own rather than raising the shared constant, which
would undo the learned reduction for every well-behaved model at once.

A related trap, worth knowing before blaming the framework: an endpoint's
`/v1/models` listing is a catalogue, not a guarantee. Passau advertised two
models that returned `HTTP 400 invalid model name` — its `/model/info` route
listed only the four actually deployed — and of ten models advertised by an
NVIDIA NIM free key, six returned `HTTP 404` and two were retired with
`HTTP 410 Gone`. A judging pass against an undeployed model fails on every
sample and admits nothing, because a failed call abstains and a single
abstaining judge leaves no verdict at all. Test one call before committing a
full pass.

---

## Re-analysing a run, and comparing the passes

Pick **Re-analyse a previous run** as the sample source to run a different tool
selection over samples that already exist — no regeneration, no judging cost.

Runs are listed by a short handle rather than their id, numbered from the
oldest so a handle never changes. Each line gives the generation of the samples
(`v2`, `noisy`), how many survived judging out of how many were generated, and
the run it was derived from, so a chain reads straight off the list:

```
#38  v3 perturbed  6/6 verified       <- #37
#37  v2 perturbed  6/6 verified       <- #28
#28  v1 original   6/8 verified
```

Whether the samples were rewritten is spelled out in both directions. A
perturbed run was always identifiable -- `source` says so and the id carries
`_noisy` -- but an unperturbed one said nothing at all, which made "clean"
something you inferred from the absence of a mark.

The full id of the selected run is shown beneath the list, since that is the
name on disk and in every URL.

Each pass is kept. The previous results are not overwritten: every pass is also
recorded under `03_detection/rounds/`, holding just the detectors that ran in
it. The results area then shows one collapsible block per earlier pass beneath
the current one, with the same standard metrics plus the change in F1, so two
configurations can be read against each other on identical samples. Only the
current pass is expanded; a run that has been analysed many times stays one
screen until a block is opened.

**"This pass" is the latest round, not the results directory.** The flat
`03_detection/` directory holds the newest file per *tool* across every pass
ever run on that run, so re-analysing with one detector left the previous
twelve files sitting beside it. Reading that directory as the current pass
reported thirteen detectors for a pass that ran one, twelve of them scored
against a manifest they had never been run on and distinguishable only by a
coverage figure. The evaluation therefore scores the newest round, which
records exactly what the last pass produced; the flat directory remains the
per-tool cache it always was. A run detected before rounds were recorded has
no archive to read and falls back to the directory, so those runs evaluate
unchanged.

The ranking, the agreement matrix and the per-sample verdict table are all
built from that same set, so they agree with each other.

**A pass with no detectors selected shows no detectors.** Judging a run without
re-running the tools archives no round, so the newest round belongs to an
*earlier* pass -- and reporting it as the current one put the previous
selection under a heading reading "this pass", every row marked `[STALE]`
because it had been scored against labels it never saw. Judging records
`detection_stale_after_judging` whenever results from before it are still on
disk, and detection clears it again, so the flag answers exactly the question:
is the newest round still the pass that just ran? When it is not, the current
pass is empty and says why, and every set of numbers the run has appears below
as an earlier pass. Select one tool and the current pass holds that one tool,
at full coverage, however many files the flat directory has accumulated.

Passes recorded before a re-judging are marked as not comparable, because they
were scored against a different admitted sample set.

Coverage is not reported for a run with no ground truth. An ingested run that
was never judged has no labelled samples to cover, so the fraction is 0/0; it
had been rendering as 0% coverage and marking every detector `[STALE]` on a
run where staleness cannot arise, advising a re-run that would change nothing.
What is missing there is labels, not verdicts, and the report says that
separately.

### Robustness: applying noise to the samples

When re-analysing a run, tick **Apply noise to the code samples**. Each program
is rewritten without changing what it does — identifier renaming, string
obfuscation, ASCII encoding, logic refactoring, no-op insertion, dummy
computations — and the whole pipeline runs on the rewritten copy as a new run.

The gap between the two is the robustness result: a detector that understood
the weakness should answer the same way, while one matching surface patterns
should not. The transformations follow Farasat, Bouzid & Posegga (CEUR
Vol-4075), which found they drop VUDENC from 97.8% to 59.6% accuracy and the
Passau BiLSTM from 98.8% to 40.4% — on the ML arm and XSS only. Running them
here extends that to every CWE category and all three detector arms.

Perturbed runs are named so they can never be mistaken for clean ones:
`run_2026-10-20_13-47-11_noisy_v2.0`. The `noisy` tag marks what was done and
`_v2.0` marks the generation of the samples — perturbing a v2.0 run gives a
v3.0, so the chain is readable straight off the id. An original run keeps its
plain `run_<timestamp>` name, since the markers only appear where they carry
information. The run page links back to the run each copy came from.

Rewrites are AST-based and validated: a file that will not parse, or that
pyflakes finds less runnable than the original, is written out unchanged and
recorded as skipped. Because the rewrite preserves semantics the label carries
over, and re-judging the perturbed run is what verifies that rather than
assuming it.

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
        judgement_comparison.json    every judge side by side
        judgement_dropped.json       what was discarded, and why
        verified_manifest.json       the survivors        (stage 2)
        selected_manifest.json       what a human kept    (stage 2b)
        balanced_manifest.json       what balancing kept  (stage 2c)
    03_detection/
        detection_<tool>.json        one file per detector, latest pass
        rounds/
            round_<timestamp>/       one analysis pass, kept for comparison
                detection_<tool>.json
                round_metadata.json  what ran, when, on how many samples
    04_evaluation/
        evaluation_report.json
        comparison_report.json
        comparison_report.txt        the same report as readable text

logs/run_2026-08-10_17-30-00.log     every line the run printed
```

`read_verified_manifest()` returns the **last** of those three manifests that
exists — balanced, then selected, then verified, then the full generated set
— so detection and evaluation pick up whatever the run actually decided with
no further wiring. `judge.py` reads the *generation* manifest instead, so a
re-judge is never narrowed by a previous review.

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

## Code map

```
src/
    config.py           every knob: axes, models, thresholds, tolerances
    common.py           paths, credentials, logging, JSON helpers
    providers.py        one client over three model backends
    token_budgets.py      ... and the guard against models that pad
                        their output to the token cap
    runstore.py         persistent, append-only run directories
    model_registry.py   the per-installation overlay (models.local.json)
    manage_models.py      ... its command-line face
    cwe_taxonomy.py     MITRE's hierarchy, for descendant-aware scoring

    generate.py         stage 1  -- write the samples (local Ollama)
    judge.py            stage 2  -- verify labels, drop the unconfirmed
    balance.py          stage 2c -- optional label / class balancing
    detect.py           stage 3  -- orchestrate the tools under test
    evaluate.py         stage 4  -- metrics and cross-tool comparison

    ingest.py           alternative input: GitHub repo or local folder
    ingest_vudenc.py    alternative input: VUDENC's labelled commits
    fetch_vudenc_dataset.py   downloads that dataset into datasets/
    fetch_models.py     downloads the pretrained weights into models/
    perturb.py          semantics-preserving rewrites, for robustness
    check_setup.py      the environment checklist

    detectors/
        __init__.py     registry and the common result shape
        static_tools.py Bandit, Ruff, Semgrep
        ml_models.py    the four ML detectors
        vudenc_pipeline.py
                        shared tokenisation / context windows / weights
        llm_tools.py    chat and typed LLM reviewers

web/
    app.py              Flask routes; no benchmark logic
    templates/          base.html + one per page
    static/style.css

runs/        every run, never overwritten          (gitignored)
logs/        one log file per run                  (gitignored)
models/      downloaded pretrained weights         (gitignored)
datasets/    downloaded third-party corpora        (gitignored)
ingested/    working copies of external code       (gitignored)
```

**Dependencies point one way only.**

```
config  <-  common  <-  providers  <-  detectors  <-  detect
                    <-  runstore   <-  generate / judge / balance /
                                       ingest / evaluate
                                                   <-  web/app.py
```

`web/app.py` imports from `src/` and never the reverse, which is what
guarantees the browser and the CLI execute the same code — no pipeline module
imports Flask. And **nothing under `detectors/` may import `judge.py`**:
judging produces the ground truth the detectors are scored against, so a
shared helper between the two could silently move the thing being measured.
That is why `llm_tools.typed_questions()` is its own function rather than an
import, even though its wording is nearly identical.

Two modules are worth singling out:

- **`providers.py`** holds one function, `chat_json(provider, model, system,
  user, schema)`, over three backends: Ollama's top-level `format` field, an
  OpenAI-compatible `response_format: json_schema`, and Anthropic's Messages
  API. Two hard-won details live there. Every confidence field is a **bounded
  integer, never a float** — guided decoding constrains which token may come
  next, and a JSON `number` grammar permits unlimited digits, so a model asked
  for a 0.0–1.0 confidence emits `1.0000000…` until the object truncates. And
  `repair_truncated_json()` salvages an object whose model emitted its
  end-of-turn token early: Gemma does this routinely, and in one run it
  discarded seven of eight judgments, which — since a failed judgment drops
  the sample — cost most of the dataset.
- **`runstore.py`** owns every path inside a run directory, and `RUN_ID_RE`
  validates an id **before** it is joined onto a path. The web app takes run
  ids from URLs, and without that check a request for
  `../../api-key-llm.txt` would be a path-traversal read — precisely the
  CWE-22 class this project generates test cases for.

### Adding a detector

Four steps; `evaluate.py` never changes, because it reads any file carrying a
`detector` name and a `predictions` list and scores it. There is literally one
scorer, which is what lets the framework claim a common evaluation protocol.

1. Write a runner with the signature `run_x(run, manifest, log) -> dict`,
   using `detectors.build_result()` and `detectors.make_prediction()` so the
   envelope is correct by construction.
2. Add an entry to `DETECTOR_REGISTRY` in `src/detectors/__init__.py` with its
   `arm` (`static`, `ml` or `llm`) and a one-line `blurb` — the blurb is what
   the Tools page and the sidebar tooltip show.
3. Add it to `RUNNERS` in `src/detect.py`.
4. Add an availability probe to `available_tools()`, so the interface disables
   it when a prerequisite is missing rather than letting a job fail halfway.

The result contract:

```json
{
  "detector":      "Bandit",
  "detector_key":  "static_bandit",
  "detector_type": "static",
  "predictions": [
    {"id": "sample_001",
     "predicted_label": "vulnerable",
     "predicted_cwe":   "CWE-89",
     "predicted_lines": [42],
     "confidence":      0.82,
     "reasoning":       "why"}
  ]
}
```

Any extra top-level keys are carried through into the report as notes —
`findings_total`, `cv_folds`, `tool_version`, `coverage` — so a result explains
itself rather than only stating a number. `predicted_lines` is optional; an
empty list is meaningful and is not the same as its absence.

A manifest entry, for comparison:

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
  "generator_model": "qwen2.5-coder:3b",
  "quality_flags": [],
  "vulnerable_lines": [42, 43],
  "note": "where the flaw is and how it is reachable"
}
```

`label` is `"unknown"` for ingested code, which is what makes `evaluate.py`
skip it for accuracy-style metrics. `pair_id` is absent on an unpaired sample
and `vulnerable_lines` is absent unless the source supplied line-level truth.

### Concurrency and state

Judging and LLM detection parallelise across samples with a small thread pool
(4). Samples are independent, so this is safe; the pool is small deliberately,
because the judging endpoint is a shared university service. Laya drops to one
sample at a time — it runs on this machine's CPU, where four threads contend
for the same cores instead of overlapping network waits.

The web app holds job state in a module-level dict behind a lock, which
carries a deployment constraint: **the server must run as one process.** With
several worker processes each would hold its own copy and `/api/status` would
answer from whichever one the request happened to reach. A real queue would be
the answer at scale; for a single-user research tool it would be pure
overhead.

### Security notes

A local research tool with no authentication, which must not be exposed to an
untrusted network. Within that scope:

- **Ingested code is never executed.** Every tool reads it as text.
- **Run and sample ids are validated against a whitelist pattern** before
  being joined onto a path.
- **Model names and tool keys from the browser are checked against
  allow-lists** before reaching a subprocess argument or an API call.
- **`subprocess` is always called with an argument list, never a shell**, and
  a GitHub URL is matched against a strict pattern so a string beginning
  `--upload-pack=` cannot be read by git as an option.
- **The browser is never handed HTML built from model output.** Every value is
  set with `textContent` — a sample viewer that rendered generated code as
  HTML would be an injection flaw in a vulnerability-detection tool.
- **Secrets are read from gitignored files at run time** and never copied into
  a Docker image layer, where they would persist in the image history even if
  a later layer deleted them.

---

## Design decisions worth discussing

**Independent samples, and the paired generation that was removed.** Every
sample is now generated on its own, asked for as vulnerable or as safe, with
no counterpart. This is how most published vulnerability-detection benchmarks
are built, so it is what a comparison against them has to be measured on.

The framework used to generate **twins** by default: each vulnerable program
came with a safe version differing only in the minimal change needed to fix
the flaw. It was not a cheap option to drop, and the cost should be stated
rather than glossed over. Measured over all 177 pairs of a VUDENC run, the
median pair differed by **1 line**, 103 of 177 by exactly one, 160 of 177 by
three or fewer, and none were identical; median character similarity 0.9975.
That bought seven things, all of which are now gone: a control for style,
length and structure; the negative class for free; a 50/50 split by
construction; the comparison that caught the generator annotating its own
work (`check_pair`, `scrub_leaking_comments`); line-level ground truth taken
from the diff; a difficulty floor, since a pair differing by one line cannot
be separated by surface statistics; and a reviewer seeing both halves at once.

**What this means for reading the numbers.** A detector can now score well by
reacting to surface properties of the two populations — prompt style, length,
imports — rather than to the vulnerability, and nothing in the dataset can
tell the difference. The "no vulnerability at all" claim on a safe sample has
no twin to check it against, so the stage-2 judges are its only check. The
structural tests that need both halves (`identical_versions`,
`length_mismatch`) cannot run, and there is no diff to take line-level truth
from, so localisation is not scored. The label mix is whatever the judges
admit rather than 50/50 by construction, which is what the balancing step in
stage 2c exists to even out.

`config.GENERATE_TWINS` is `False` and is no longer offered in the web form.
The paired code path survives in `generate.py` only so that runs recorded
before this change still resolve against their `config_snapshot`.

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

**Random CWE selection, and a blank seed.** Categories are always drawn at
random, as requested, and there is no second mode to choose. The first
implementation drew independently per sample with `rng.choice()`, and with a
fixed seed that produced the same sequence every run: ten pairs covered seven
of the sixteen categories and nine were never generated at all, across eight
consecutive runs. Selection now **shuffles the catalogue and walks it**,
reshuffling each time it is exhausted, so a run of N samples covers
`min(N, 16)` distinct categories and the counts differ by at most one.

A `"cycle"` mode — the catalogue in declared order, `categories[i % 16]` —
used to stand beside it. It gave the same coverage and differed only in being
identical at every seed, which made N short runs all target the same first few
categories. It was removed rather than left as a trap.

`config.SEED` is now `None` and the form's seed box is blank, so every run
draws its own categories, contexts and difficulties. A seed only ever made the
benchmark *configuration* reproducible, never the generated code — the model
is not deterministic, so two runs at one seed produce different programs
anyway, and holding the configuration fixed while the programs varied bought
reproducibility of the wrong half. A number can still be entered to repeat one
run's draw exactly.

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

- **The pretrained ML models score near chance on this benchmark, and that is
  a result rather than a defect.** They flag the safe twin about as often as
  the vulnerable one. The tokenisation port is verified byte-identical to the
  authors' own `myutils.py`, and on the authors' real-world example files the
  models behave as published, so the pipeline is faithful. Three things
  plausibly explain the gap, and a write-up should say so: the samples are
  short synthetic programs rather than the large real files the models were
  fitted on; vulnerable and safe twins differ by a handful of tokens, which the
  same group's adversarial study found these models cannot survive; and the
  published word2vec vocabulary is small enough (3,576 tokens for the Passau
  model) that `flask`, `request` and `sqlite3` are out-of-vocabulary, leaving
  the taint source invisible to the network.

- **The original VUDENC detector cannot normally run.** Its word2vec vector
  array was saved by gensim to a `.npy` sidecar that the authors never
  published — it is absent from the repository and from all four VUDENC Zenodo
  records. It cannot be reconstructed by retraining, because a fresh word2vec
  run produces a different vector space that the pretrained LSTM would read as
  noise. The detector therefore reports itself unavailable rather than
  returning confident nonsense. See `src/fetch_models.py` for how to enable it
  if you obtain the file.

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

- **Nothing is scored at the pair level yet.** `evaluate.py` scores samples
  independently and never reads `pair_id`, so the paired design pays for the
  dataset's construction and buys nothing in the metrics. Three things it
  should buy and does not: a **pair-consistency** rate (of the pairs where
  both halves survived, on how many did the tool answer *vulnerable* for one
  and *safe* for the other — the only measure that distinguishes understanding
  the flaw from reacting to the program), **McNemar's test** for whether two
  tools differ by more than chance on the same samples, and **paired
  bootstrap** confidence intervals. Until those exist, a difference of a few
  points between two tools in the comparison table has no stated significance.

- **The VUDENC importer pairs multi-line fixes by index, and for some hunks
  that is wrong.** The safe half is built by replacing each removed line with
  the added line at the same index, `zip(badparts, goodparts)`. Where a fix
  replaced *n* lines with *n* lines in order that is exactly right, and 113 of
  177 pairs substitute a single line. But 64 substitute more than one, and in
  7 of those the worst index-paired line barely resembles its replacement:
  `pair_036` and `pair_038` at similarity 0.0, then `pair_117` (0.12),
  `pair_145` (0.14), `pair_121` (0.15), `pair_098` (0.18), `pair_144` (0.35).
  `pair_036` is a confirmed failure — 12 removed lines mapped onto 12 added
  lines that do not positionally correspond, and a filter that skips blank
  removed lines silently discards three non-blank added ones. The result
  parses, so nothing rejects it, and it is semantically incoherent. The fix is
  to apply such a hunk as a block, or to drop hunks that are not 1:1; neither
  is implemented. Until then, a run built from this source should have its
  multi-line pairs reviewed, and the judging stage is the mechanism for that.

- **The VUDENC labels have never been vetted by the judges.** The 177-pair run
  on disk was imported and detected but not judged, so its commit-level labels
  stand unchecked. See the caveats under
  [the VUDENC dataset](#real-code-that-is-labelled-the-vudenc-dataset).

- **The web app is a local research tool.** No authentication, no CSRF
  protection, one concurrent run. It binds `127.0.0.1` and must not be exposed
  to an untrusted network. Note that Werkzeug sets `SO_REUSEADDR`: a second
  copy of `web/app.py` can bind port 8000 while the first is still listening,
  and both will answer, so a stale server can serve old code to a freshly
  reloaded page. Kill strays rather than assuming a reload was enough.

---

## Next steps

1. **Run the judging stage over the VUDENC import.** Its commit-level labels
   are the weakest ground truth in the project and the one place where model
   agreement plus a human review would buy the most.
2. Scale to a few hundred samples and hand-verify a random subset, to measure
   how well the judges' agreement tracks human judgement.
3. **Add the pair-level metrics** — pair consistency, McNemar, paired
   bootstrap CIs — so the paired design is used in the scoring and not only in
   the construction.
4. **Fix the multi-line hunk pairing** in `ingest_vudenc.py`, then re-import.
5. Run the ML and LLM arms over the full 177-pair VUDENC run; only the static
   arm has been scored on it.
6. Add CodeBERT / GraphCodeBERT as a heavier ML arm alongside the current
   models, keeping the same out-of-fold protocol.

---

## Documentation

**This file is the documentation.** Installation, the code map, the data
contracts, every methodological caveat and every known limitation are above,
because the long-form companion documents are deliberately *not* in the
repository — `.gitignore` excludes `docs/`, since those documents carry
measurements and prose belonging to the thesis text rather than to the
published code. On a working copy that has them, they are:

| file | what it adds beyond this README |
|---|---|
| `docs/SETUP.md` | the same installation, step by step, with more troubleshooting |
| `docs/ARCHITECTURE.md` | the full module-by-module rationale, far longer than the code map above |
| `docs/laya-jev-perf.md` | what the two typed models measured in both roles, across the run family descended from #19, and which is worth using where |

Beyond that, **the code is the documentation of last resort and is written to
be read**: every non-obvious decision is argued in a block comment at the
point it was made, usually with the measurement that forced it.
`src/config.py` in particular is a commented catalogue of every knob and why
it has the value it does.
