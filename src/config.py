"""Central configuration for the VulEvalBench pipeline.

Everything a reviewer or supervisor is likely to want changed lives in this one
file, so that adjusting the experiment never means editing pipeline code.

WHAT THIS FILE CONTAINS
    The *axes of the experiment*: which CWE categories may be generated, which
    programming contexts and difficulty levels exist, how many samples to
    produce, which models play which role, and how the final comparison score
    is weighted.

WHAT THIS FILE DOES NOT CONTAIN
    Any vulnerable code. Every program in the benchmark is written by the
    generator model from a prompt; the harness only decides what to ask for.

PIPELINE ROLES -- three distinct model jobs, deliberately kept separate
    1. GENERATOR  (local, via Ollama)   writes the candidate sample pairs.
    2. JUDGE      (remote, Passau API)  verifies each sample really contains
                                        the single intended vulnerability, and
                                        drops the ones that do not.
    3. DETECTOR   (the tools under test) tries to find the vulnerability.

    Using a *local* generator and a *remote* judge means the model that wrote
    a program is never the model that certifies it, which removes the
    self-preference bias that would otherwise inflate the dataset's quality.
"""

from __future__ import annotations

import os

# ===========================================================================
# SECTION 1 -- Model roles
# ===========================================================================

# --- Role 1: the generator -------------------------------------------------
# Runs locally through Ollama. Local generation is what makes scaling to a few
# hundred samples affordable: generation is by far the most token-hungry stage
# (two complete programs per call), and doing it locally costs nothing per call.
#
# The tag must already be pulled -- `ollama pull qwen2.5-coder:3b`. See
# providers.list_ollama_models() for what is actually available at run time.
#
# WHY 3B AND NOT 7B -- measured, not assumed.
#   Both were benchmarked on the development machine (16 GB RAM, Intel Iris Xe
#   integrated graphics, no CUDA device, so Ollama runs 100% on CPU):
#
#       qwen2.5-coder:3b    ~45 s per pair    1.9 GB    stable
#       qwen2.5-coder:7b    ~457 s per pair   5.5 GB    intermittent OOM
#
#   The 7B model is ten times slower here, not the two to three times a rough
#   estimate suggests, and with only ~3 GB of RAM free it failed to allocate
#   its KV cache on several consecutive pairs. A ten-pair run would spend over
#   an hour in generation alone with a real chance of dying partway.
#
#   The quality gain did not justify that. The 7B sample it did produce was
#   still 13 lines long and still carried a `# Vulnerable SQL query` comment --
#   the same failure modes as the 3B model. Leakage is now handled
#   deterministically by scrub_leaking_comments() in generate.py, which helps
#   both models equally and costs nothing.
#
#   On a machine with a CUDA GPU, or with substantially more free RAM, 7B is
#   the better choice: set GENERATOR_MODEL=qwen2.5-coder:7b in the environment,
#   no code change needed.
# Read from the environment when set, so the Docker image can point at the
# host's Ollama (http://host.docker.internal:11434) without editing this file.
# Defaults to the local server for a normal, non-containerised run.
# 127.0.0.1 rather than "localhost", deliberately and measurably. On Windows
# "localhost" resolves to the IPv6 address ::1 first; Ollama binds IPv4 only,
# so every request stalls on the ::1 attempt before falling back. Measured on
# this machine: 2.07 s via "localhost" against 0.037 s via 127.0.0.1 -- a 56x
# penalty paid on every generation call AND on every web page load, which made
# the whole interface feel hung.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "qwen2.5-coder:3b")

# How long to wait for one local generation call.
#
# Sized from measurement rather than caution: on this machine a normal pair
# completes in 41 seconds (76 s at the slowest), and generate.py caps output at
# 3000 tokens, which at the observed ~19 tokens/second bounds a full-budget
# generation to roughly three minutes.
#
# 300 s therefore leaves ample room for a legitimately slow pair while cutting
# the cost of a runaway. The previous value of 600 s was set before any of this
# was measured, and nine pairs in one 50-pair run each burned the full ten
# minutes before being abandoned.
#
# Raise this if you move to a larger generator model: 7B on CPU is roughly ten
# times slower per pair.
OLLAMA_TIMEOUT_SECONDS = 300

# --- Role 2: the judges ----------------------------------------------------
# Served by the University of Passau endpoint, whose URL and key both live in
# `api-key-judgment.txt`. These models never see the generator's intended
# label as a hint -- they are asked to analyse the program independently, and
# their verdict is then compared with the intended label.
#
# IDs below are the exact strings the endpoint's /v1/models returns; the
# human-readable names are what the web interface shows.
JUDGMENT_MODELS = {
    "qwen3-next-80b-a3b-instruct": "Qwen3 Next 80B A3B Instruct FP8",
    "qwen-agentworld-35b-a3b": "Qwen-AgentWorld 35B-A3B",
    "qwen36-35b": "Qwen3.6 35B-A3B-FP8",
    "gemma4-31b-it": "Gemma 4 31B-it",
}

# Two different judges by default, so that a sample only survives if two
# independently-prompted models agree with the generator's intent. Agreement
# between two models is much weaker evidence than a human check, but it is far
# stronger than trusting the generator alone -- which was the single biggest
# methodological weakness of the previous draft.
# JUDGE SELECTION IS A MEASURED CHOICE, NOT AN ARBITRARY ONE.
#   All four chat models on the endpoint were benchmarked as judges over the
#   same 20 samples (10 vulnerable / 10 safe twins):
#
#       model                          agreed   vuln/safe admitted   errors
#       qwen3-next-80b-a3b-instruct    10/20      10 / 0             0
#       gemma4-31b-it                  17/20       8 / 9             0
#       qwen36-35b                     15/20       6 / 9             0
#       qwen-agentworld-35b-a3b        11/20       4 / 7             6
#
#   qwen3-next-80b admitted ZERO safe twins -- not once, across every run. A
#   judge that never accepts a negative example is not verifying labels, it is
#   flagging everything, and under the "all" consensus rule it collapsed the
#   entire safe half of the dataset on its own. That is the security-review
#   expectation bias the system prompt tries to counter, and on this model the
#   counter-instruction simply does not work.
#
#   qwen-agentworld returns empty responses on roughly a third of calls.
#
#   The two below are the ones that behave: both balanced across the classes,
#   both error-free. Together under the strict "all" rule they admit 15/20
#   (6 vulnerable / 9 safe) -- a usable, balanced dataset with the strongest
#   verification rule left intact.
#
# OCTOBER 2026 -- THE ENDPOINT STOPPED SERVING TWO OF THESE MODELS.
#   qwen36-35b and qwen3-next-80b-a3b-instruct are still advertised by
#   /v1/models but /chat/completions now rejects both with
#   "HTTP 400: Invalid model name". Verified by calling both endpoints
#   directly; gemma4-31b-it and qwen-agentworld-35b-a3b still answer.
#
#   That broke the default judge pair: qwen36-35b failed on every sample, and
#   because a failed call used to count as disagreement under the "all" rule,
#   entire runs were discarded with "0 admitted". judge.py now treats a failed
#   call as an abstention and warns loudly, so a dead judge degrades the run
#   instead of silently voiding it.
#
#   Judge B therefore moves to Claude, which is reachable and reliable. If
#   Passau restores the Qwen models, change it back -- a second judge from a
#   different organisation than the detector is the arrangement this benchmark
#   prefers, and the measured comparison above still stands.
DEFAULT_JUDGE_A = "gemma4-31b-it"
DEFAULT_JUDGE_B = "claude-sonnet-5"

# A THIRD JUDGE SLOT, EMPTY BY DEFAULT.
#   judge.py has always taken a list, so a third judge costs no new logic --
#   but it does change what the consensus rule means, and that is why the slot
#   ships empty rather than filled.
#
#   Under the default "all" rule a third judge can only ever shrink the
#   admitted set: every sample it disputes is dropped, and with three models
#   the chance that at least one disputes any given sample is high. On the
#   20-sample benchmark above, the two default judges admitted 15; a third
#   strict judge would be expected to take that into single figures.
#
#   The reason to want three is the other rule. With two judges a
#   disagreement is unresolvable -- there is no way to tell which model is
#   wrong -- so "all" has to treat it as fatal. With three, JUDGE_CONSENSUS =
#   "majority" lets two agreeing judges outvote one dissenter, which keeps the
#   dataset usable while still refusing to take the generator's word for
#   anything. That is the arrangement a third judge is for; see the note on
#   JUDGE_CONSENSUS.
DEFAULT_JUDGE_C = ""

# --- System One judges: a different kind of verifier ----------------------
# Jev and Laya are not chat models. They are "System One" models: an encoder
# reads the program once and answers typed questions -- pick one of these
# options, score on this rubric, is this statement true -- returning calibrated
# probabilities instead of text. Neither can write a sentence; Laya's own
# documentation says it "never generates text".
#
# WHY THAT IS INTERESTING FOR THIS STAGE SPECIFICALLY
#   The judge's job is not to explain anything. It answers one closed question
#   -- does this program contain a vulnerability -- plus a category from a fixed
#   list and a count. That is exactly the shape of a typed decision, and the
#   current judges only produce it because guided decoding forces a chat model
#   into a JSON schema it was never trained to fill.
#
#   Two consequences worth measuring, which is why these are offered as judges
#   and not merely mentioned:
#     - `confidence` from these models is a probability the model actually
#       assigned over its own options, where a chat model's
#       `confidence_percent` is a figure it wrote in prose. Since
#       JUDGE_MIN_CONFIDENCE drops samples below a threshold, the meaning of
#       that threshold is not the same for the two kinds of judge.
#
#       NOT the same as calibrated, and the difference was measured rather
#       than assumed. The field to read is `answer_confidence` (max(p)), whose
#       own documentation says the calibration property "holds only after the
#       temperatures have been fitted and validated on held-out data for this
#       checkpoint and this option count" -- which nobody has done for Python
#       source. Worse, loading the English checkpoint warns that it ships
#       invalid temperatures for the `choice:11+` bucket and clamps them; the
#       CWE question has 17 options, so it lands in exactly that bucket.
#       judge.py writes that warning into the run log rather than letting it
#       disappear. Read these confidences as the model's own probabilities,
#       not as validated calibration.
#     - they are roughly two orders of magnitude smaller or cheaper than the
#       chat judges, so if they agree with them the stage gets much cheaper.
#
# WHAT THEY CANNOT DO
#   No `reasoning` string. The judgement files and the comparison page show a
#   judge's own justification for its verdict, and for these two there is none
#   to show -- judge.py puts the answer probabilities there instead, clearly
#   marked as derived rather than written. A disagreement between these judges
#   is therefore harder to adjudicate by reading.
#
#   Neither model is trained on source code. Laya's card names English text,
#   guardrails and email triage; Jev is a general structured-decision model.
#   Expect that to show, and treat a poor result as evidence about transfer to
#   code rather than as a bug.
# MEASURED, OCTOBER 2026 -- LAYA AS A JUDGE, 82 SAMPLES (41 vulnerable /
# 41 safe) FROM run_2026-08-27_23-10-16, THE SAME WAY THE CHAT JUDGES ABOVE
# WERE BENCHMARKED.
#
#       label call alone      tp=14  fp=3  fn=27  tn=38
#                             accuracy 0.634   MCC 0.331
#                             precision 0.824  recall 0.341  FPR 0.073
#       failed calls          0
#       confidence            mean 0.694, min 0.503, max 0.948
#       speed                 13.4 s per sample, CPU, one at a time
#
#   Read the label row first: MCC 0.331 is well above chance, but the model is
#   heavily conservative -- it called only 17 of 82 programs vulnerable when
#   41 were. High precision, poor recall. For a judge that is not the worst
#   failure mode (it waves through few false vulnerabilities) but it costs
#   most of the vulnerable half of the dataset.
#
#   THE COUNT RULE IS WHAT ACTUALLY BLOCKS IT. The funnel:
#
#       82  samples
#       52  label matched the generator
#       40  ... and confidence >= JUDGE_MIN_CONFIDENCE
#       12  ... and REQUIRE_SINGLE_VULNERABILITY satisfied
#
#   The count question returned "exactly one vulnerability" for 78 of 82
#   programs and "two or more" for 4. Level 0, "no security vulnerability",
#   was never the most probable level -- not once, for either class. So every
#   safe sample fails the count rule however well the label question did.
#
#   That is a property of the format, not a bug to tune away. Each question is
#   evaluated in parallel and in isolation against the same state, so nothing
#   stops a typed judge from answering "safe" and "exactly one vulnerability"
#   about the same program. A chat judge writes one JSON object and is at
#   least internally consistent. REQUIRE_SINGLE_VULNERABILITY was designed for
#   that kind of judge, and it is close to inapplicable to this one.
#
#   The rubric wording was deliberately NOT adjusted until the numbers
#   improved: that would be fitting the judging stage to the test labels, the
#   same mistake avoided when the VUDENC pooling rule was chosen. If a typed
#   judge is wanted in the write-up, the honest options are to report it as
#   measured, or to turn REQUIRE_SINGLE_VULNERABILITY off for the whole run
#   and say so -- not to reword the question until it agrees.
#
#   Jev is NOT in this table. It needs an early-access key, so it has never
#   been called. Its request and response shape is implemented against the
#   published OpenAPI schema and the mapping is unit-tested against a
#   synthetic response, but no measurement of it exists.
SYSTEMONE_MODELS = {
    # Hosted by TypeSafe. Needs api-key-typesafe.txt; the model is in early
    # access, so a key comes from their waitlist.
    "jev-latest": "Jev (TypeSafe System One)",
    # Local, Apache-2.0, 0.4B: a ModernBERT-large encoder with a decision head.
    # Runs in this process once `pip install laya` has fetched it; the
    # checkpoint itself comes from Hugging Face on first use.
    "laya": "Laya 0.4B (local)",
    "laya-multilingual": "Laya multilingual (local)",
}

# Both models speak the same wire protocol -- POST /v1/systemone, a `state`
# plus named typed `questions` -- because laya-serve deliberately implements
# TypeSafe's shape. Laya is called in-process through its Python package
# instead, which avoids running a second HTTP server on the port the web
# interface already uses.
TYPESAFE_BASE_URL = "https://api.typesafe.ai"

# Which Laya checkpoint each model id maps to. These are the package's own
# checkpoint names (laya.DEFAULT_MODELS), not Hugging Face repo ids: the
# multilingual weights are a SUBFOLDER of convaiinnovations/laya rather than a
# repo of their own, so naming the repo would silently load the English model.
LAYA_CHECKPOINTS = {
    "laya": "english",
    "laya-multilingual": "multilingual",
}

# Non-chat models on the same endpoint. No detector uses these since the ML
# arm moved to pretrained local models, but they remain part of what the
# judgment key unlocks and providers.embed_texts() still exposes them, so they
# are listed here to keep one place recording what the endpoint offers.
EMBEDDING_MODEL = "octen-embedding-8b"
RERANKER_MODEL = "qwen3-reranker-4b"

# --- Role 3: the LLM detector arm -----------------------------------------
# The LLM tools under test. These are scored like any other detector, so they
# must be kept separate from the judges above: a model that certified a sample
# must not also be graded on finding the vulnerability in it, or it would be
# marking its own homework.
# Deliberately qwen3-next-80b: it is the one model NOT used as a judge, so no
# model grades a sample it helped certify.
#
# Its strong bias towards reporting a vulnerability -- the very thing that
# disqualified it as a judge -- is not a defect here. As a detector it is a
# property to be measured: it should show up as high recall with a high false
# positive rate, which is exactly the trade-off the recall and FPR columns
# exist to expose.
# Was qwen3-next-80b-a3b-instruct, which the endpoint no longer serves (see
# the note above DEFAULT_JUDGE_A). claude-haiku-4-5 is the replacement: it is
# reachable, it was already the strongest detector in every scored run, and it
# is a different model from both judges, so nothing grades a sample it helped
# certify. Note the weaker separation this forces -- judge B and the detector
# now come from the same provider, where previously every role sat with a
# different organisation. Worth stating in the write-up.
DEFAULT_LLM_DETECTOR = "claude-haiku-4-5"

# A SECOND DETECTOR SLOT, EMPTY BY DEFAULT -- and the arm may also be switched
# off entirely.
#
# The LLM arm has always taken a list of models, so running two costs no new
# logic. What was missing was the two ends of the range: there was no second
# *default*, and selecting no models at all fell back to the single default
# rather than meaning "no LLM arm". Both are now possible:
#
#     none   deselect every model, or set DEFAULT_LLM_DETECTOR = "" with this
#            empty. The llm_detector tool is dropped with a log line instead
#            of running against an unasked-for model. Useful when the API
#            budget is spent, or when only the static and ML arms are wanted.
#     one    the default: slot A filled, slot B empty.
#     two+   fill this slot, or select several in the interface. Each model
#            produces its own row, named by model, so they are directly
#            comparable within one run.
#
# Empty rather than filled, for the reason DEFAULT_JUDGE_C is: a second
# detector doubles the cost of the arm and nothing about the benchmark needs
# it, so it is opt-in. Unlike the judges, though, adding one changes no
# consensus rule -- detectors do not vote, they are each scored separately --
# so there is no methodological cost to switching it on, only money and time.
DEFAULT_LLM_DETECTOR_B = ""

# MEASURED, OCTOBER 2026 -- THE TYPED MODELS AS DETECTORS.
#   Jev and Laya are selectable in this arm as well as as judges. Scored over
#   the 59 verified samples of run_2026-08-27_23-10-16 (36 vulnerable / 23
#   safe), which was judged by gemma4-31b-it and qwen36-35b -- so neither
#   typed model had any hand in the labels it was scored against.
#
#       detector             tp  fp  fn  tn   acc    prec   rec    fpr    mcc     cwe
#       claude-haiku-4-5     34   5   2  18  0.881  0.872  0.944  0.217  +0.749  30/34
#       jev-latest           32   3   4  20  0.881  0.914  0.889  0.130  +0.753  28/32
#       laya                 16   3  20  20  0.610  0.842  0.444  0.130  +0.328   2/16
#
#   Zero failed calls for all three. Speed: Jev 0.22 s/sample (hosted, four in
#   flight), Laya 4.08 s/sample (local CPU, one at a time, after a ~60 s
#   checkpoint load).
#
#   JEV MATCHES THE CHAT DETECTOR. Same accuracy, marginally higher MCC, and
#   it reaches it from the other side: fewer false positives (FPR 0.130 against
#   0.217) at the cost of two more misses. For a 0.4B-class encoder that
#   generates no text and was never trained on source code, matching a frontier
#   chat model on this task is the single most interesting number in this file.
#   Its CWE accuracy is also real: 28 of 32, against Haiku's 30 of 34.
#
#   LAYA IS NOT COMPETITIVE, and it is worth being precise about how. Its
#   precision is 0.842 -- when it says "vulnerable" it is usually right -- but
#   recall is 0.444, so it misses more than half the vulnerable half. And its
#   CWE head is the weak point: 2 correct out of 16, against Jev's 28 of 32.
#   It can tell that something is wrong far better than it can say what.
#
#   WHAT THIS DOES NOT SHOW. These are single-run numbers on one generator's
#   output at n=59, not a benchmark result with error bars. The ordering of
#   Jev and Haiku in particular is well inside what one run can produce by
#   chance; the honest claim is that they are comparable, not that Jev wins.
#
#   Note the asymmetry with the judging stage: the detector path asks only the
#   label and CWE questions, not the vulnerability count, because the count
#   exists for REQUIRE_SINGLE_VULNERABILITY and a detector admits nothing.
#   That is why Laya scores respectably here while being close to unusable as
#   a judge -- the count question was what blocked it, and it is not asked.

# Anthropic models, available when `api-key-llm.txt` holds an Anthropic key.
# Offered as an option so the local/remote comparison can be extended to a
# second vendor, which is the credibility gap named in the README.
ANTHROPIC_MODELS = {
    "claude-opus-5": "Claude Opus 5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
}


def model_display_name(model: str) -> str:
    """The human-readable name for any model id in this file.

    Four registries now hold model ids -- the judgment endpoint, Anthropic,
    the System One models and the Ollama tags -- and three different places
    were looking a name up by chaining `.get()` calls across them. One
    function, so adding a fifth registry does not mean hunting for them.

    Args:
        model: a model id as it appears in a dropdown or a run record.

    Returns:
        The display name, or the id itself when it is not a listed model
        (a local Ollama tag, for instance, which is its own best label).
    """
    for registry in (JUDGMENT_MODELS, ANTHROPIC_MODELS, SYSTEMONE_MODELS):
        if model in registry:
            return registry[model]
    return model


# ===========================================================================
# SECTION 2 -- Benchmark axes
# ===========================================================================

# COVERAGE AXIS. The catalogue a sample's vulnerability is drawn from.
#
# Expanded from the five of the first draft to sixteen. Every entry is:
#   - genuinely expressible in idiomatic Python,
#   - reachable from untrusted input (so it is a real defect, not a lint nit),
#   - and detectable in principle by at least one of the three tool arms, so
#     that a zero score means a tool missed it rather than that the category
#     was impossible for that paradigm.
CWE_CATEGORIES = [
    "CWE-89",    # SQL Injection
    "CWE-78",    # OS Command Injection
    "CWE-22",    # Path Traversal
    "CWE-502",   # Deserialization of Untrusted Data
    "CWE-798",   # Use of Hard-coded Credentials
    "CWE-79",    # Cross-Site Scripting
    "CWE-94",    # Code Injection (eval/exec on untrusted input)
    "CWE-327",   # Use of a Broken or Risky Cryptographic Algorithm
    "CWE-330",   # Use of Insufficiently Random Values
    "CWE-611",   # XML External Entity (XXE)
    "CWE-918",   # Server-Side Request Forgery
    "CWE-295",   # Improper Certificate Validation
    "CWE-377",   # Insecure Temporary File
    "CWE-732",   # Incorrect Permission Assignment for Critical Resource
    "CWE-1333",  # Inefficient Regular Expression Complexity (ReDoS)
    "CWE-209",   # Error Message Containing Sensitive Information
]

# Human-readable names, used in generation prompts, the web tables and the
# metrics report. Kept as a separate mapping so a category can be removed from
# the active list above without losing its name -- historical runs still need
# to render their own CWEs correctly.
CWE_NAMES = {
    "CWE-89": "SQL Injection",
    "CWE-78": "OS Command Injection",
    "CWE-22": "Path Traversal",
    "CWE-502": "Deserialization of Untrusted Data",
    "CWE-798": "Use of Hard-coded Credentials",
    "CWE-79": "Cross-Site Scripting",
    "CWE-94": "Code Injection",
    "CWE-327": "Use of a Broken or Risky Cryptographic Algorithm",
    "CWE-330": "Use of Insufficiently Random Values",
    "CWE-611": "XML External Entity (XXE)",
    "CWE-918": "Server-Side Request Forgery",
    "CWE-295": "Improper Certificate Validation",
    "CWE-377": "Insecure Temporary File",
    "CWE-732": "Incorrect Permission Assignment for Critical Resource",
    "CWE-1333": "Inefficient Regular Expression Complexity (ReDoS)",
    "CWE-209": "Error Message Containing Sensitive Information",
    # Not in CWE_CATEGORIES -- the generator does not produce these. They
    # arrive only with real-world data (see VUDENC_CWE_MAP and
    # ingest_vudenc.py), and need names so those runs render correctly.
    "CWE-601": "Open Redirect",
    "CWE-352": "Cross-Site Request Forgery (CSRF)",
}

# A one-line hint per category, injected into the generation prompt. Without
# these a 7B local model tends to reach for the same textbook example every
# time; naming the *mechanism* rather than the code pushes it to invent a
# fresh program while still hitting the requested category.
CWE_HINTS = {
    "CWE-89": "untrusted text reaches a SQL string that is concatenated or interpolated rather than parameterised",
    "CWE-78": "untrusted text reaches a shell command line, e.g. subprocess with shell=True or os.system",
    "CWE-22": "untrusted text is used to build a filesystem path without confining the result to a base directory",
    "CWE-502": "untrusted bytes are deserialised by pickle, marshal, PyYAML's unsafe loader or similar",
    "CWE-798": "a password, API key or token is written as a literal in the source and used for real authentication",
    "CWE-79": "untrusted text is written into an HTML response without escaping",
    "CWE-94": "untrusted text reaches eval, exec or compile",
    "CWE-327": "a broken primitive such as MD5, SHA1, DES or ECB mode is used where a secure one is required",
    "CWE-330": "the random module (or time) is used to make a security-relevant token, password or session id",
    "CWE-611": "XML from an untrusted source is parsed with external entity resolution left enabled",
    "CWE-918": "a URL derived from untrusted input is fetched server-side without host validation",
    "CWE-295": "TLS certificate verification is disabled or a permissive context is used",
    "CWE-377": "a predictable temporary filename is created, or a temp file is created insecurely",
    "CWE-732": "a file, directory or socket is created with permissions far wider than needed, e.g. chmod 0777",
    "CWE-1333": "a regular expression with catastrophic backtracking is run against untrusted input",
    "CWE-209": "an exception's internals, a stack trace or a query string is returned to the caller",
}

# The correct remedy for each category, injected into the generation prompt.
#
# WHY THIS EXISTS -- the single biggest quality problem measured so far.
#   Asked to write a vulnerable program and a fixed twin in one call, the local
#   generator reliably produced the flaw and then failed to repair it. Its
#   "fix" for CWE-89 was:
#
#       cursor.execute(query)        ->  cursor.execute(query, ())
#
#   which changes nothing at all: the whole query string still arrives from
#   argv. It had imitated the *shape* of parameterisation without understanding
#   it. Both judges caught this, so every safe twin in that run was dropped and
#   the dataset came out 100% vulnerable -- which makes the false positive rate
#   undefined, MCC degenerate, and the ML arm untrainable.
#
#   Injecting a flaw is pattern reproduction, which a small model does well.
#   Inventing a minimal correct repair requires understanding, which it does
#   not. So the harness supplies the remedy and the model only has to apply it.
#   This is the same principle the CWE selection follows: coverage and
#   correctness are the harness's job, invention is the model's.
CWE_FIXES = {
    "CWE-89": "pass every piece of user data as a bound query parameter; the SQL text itself must be a literal string in the source, never built from input",
    "CWE-78": "call the program with an argument list and shell=False, so no shell parses the input; never build a command string",
    "CWE-22": "resolve the joined path with os.path.realpath and reject it unless it is still inside the intended base directory",
    "CWE-502": "parse with a data-only format such as json.load, or yaml.safe_load; never unpickle untrusted bytes",
    "CWE-798": "read the credential from an environment variable or a config file at run time; no literal secret in the source",
    "CWE-79": "escape the untrusted text before inserting it into HTML, or use the template engine's automatic escaping",
    "CWE-94": "replace eval/exec with an explicit lookup, such as a dict of permitted operations, or a typed parser",
    "CWE-327": "use a current primitive: SHA-256 for hashing, and an authenticated cipher mode such as AES-GCM",
    "CWE-330": "generate the value with the secrets module, not random",
    "CWE-611": "disable entity resolution, e.g. use defusedxml, or set the parser to forbid external entities and DTDs",
    "CWE-918": "validate the resolved host against an explicit allow-list before making the request",
    "CWE-295": "leave certificate verification enabled; do not pass verify=False or an unverified SSL context",
    "CWE-377": "create the file with tempfile.NamedTemporaryFile or mkstemp, which choose an unpredictable name safely",
    "CWE-732": "create the resource with least-privilege permissions, e.g. 0o600 for a private file",
    "CWE-1333": "bound the input length, or rewrite the pattern so it has no nested or overlapping quantifiers",
    "CWE-209": "return a generic message to the caller and log the detail server-side",
}

# How the CWE for each sample (or pair) is chosen.
#
#   "random" -- the catalogue is shuffled and walked, and reshuffled each time
#               it is exhausted. A draw WITHOUT replacement, not an
#               independent draw per sample.
#   "cycle"  -- the catalogue in its declared order, categories[i % 16].
#
# BOTH MODES NOW GIVE THE SAME COVERAGE, which is worth saying because an
# earlier version of this comment claimed otherwise. A run of N samples covers
# min(N, len(CWE_CATEGORIES)) distinct categories either way, and the counts
# differ by at most one. The first implementation really did draw
# independently with rng.choice(), and with a fixed seed that produced the same
# sequence every run: ten pairs covered seven of the sixteen categories and
# nine were never generated at all, across eight consecutive runs. See
# generate._cwe_sequence for the measurement and the fix.
#
# WHAT STILL DIFFERS is only the order, and therefore reproducibility:
#   - "random" varies the order with the seed, so two runs at different seeds
#     are independent draws over the catalogue and a short run does not always
#     target the same categories.
#   - "cycle" is a deterministic prefix of the catalogue, identical at every
#     seed. Exactly balanced and exactly repeatable, but N runs of 4 samples
#     all hit the same first four categories.
#
# Random is the default because the project brief asked for random selection,
# and the coverage objection to it no longer applies.
CWE_SELECTION = "random"

# DIVERSITY AXIS 1 -- the kind of program the vulnerability is embedded in.
# Sampled per pair, so the same CWE appears in several different settings
# across a dataset. This tests whether a detector recognises the flaw itself
# or merely a familiar surrounding shape.
CONTEXTS = [
    "Flask web application",
    "FastAPI REST endpoint",
    "Django view",
    "command-line tool",
    "standalone data-processing script",
    "background worker / task queue consumer",
    "internal microservice client",
]

# The imports each context needs, injected into the generation prompt.
#
# WHY THIS EXISTS -- measured, not guessed.
#   Generated programs were being discarded for using names they never
#   imported. Counting the causes across every run, framework names dominated:
#   `HttpResponse` four times and `render` twice, against `os` twice and
#   `pickle` once. Every one of those six was a Django sample.
#
#   A 3B model reliably remembers `import os`; it does not reliably remember
#   that `HttpResponse` lives in `django.http`. Telling it outright costs
#   nothing and removes the single commonest reason a sample is unrunnable.
#   The general instruction to import everything stays in the system prompt --
#   this just supplies the framework-specific part it kept getting wrong.
CONTEXT_IMPORTS = {
    "Flask web application": [
        "from flask import Flask, request, jsonify",
    ],
    "FastAPI REST endpoint": [
        "from fastapi import FastAPI, Request, HTTPException",
    ],
    "Django view": [
        "from django.http import HttpResponse, JsonResponse",
        "from django.shortcuts import render",
    ],
    "command-line tool": [
        "import argparse",
        "import sys",
    ],
    "standalone data-processing script": [
        "import sys",
    ],
    "background worker / task queue consumer": [
        "import logging",
        "import time",
    ],
    "internal microservice client": [
        "import requests",
    ],
}

# The imports each CWE's *mechanism* needs -- covering both the flaw and the
# required fix, because the safe twin has to import whatever the remedy uses.
#
# WHY THIS EXISTS -- measured, like CONTEXT_IMPORTS above.
#   After the per-context imports were added, missing imports were still the
#   single largest cause of rejected pairs: 20 of 50 pairs in one run. But the
#   names had changed. They were no longer framework names -- they were the
#   standard-library modules the vulnerability itself requires:
#
#       subprocess x6   requests x4   sqlite3 x3
#       hashlib x2      os x2         re x2
#
#   Which module a sample needs depends on the CWE, not the context: command
#   injection needs `subprocess`, SQL injection needs a DB-API module, broken
#   crypto needs `hashlib`. Nothing in the prompt said so, and a 3B model does
#   not reliably infer it. Both halves of a pair are covered, so `random` and
#   `secrets` both appear for CWE-330 -- the flawed version needs the first,
#   the fixed version the second.
CWE_IMPORTS = {
    "CWE-89": ["import sqlite3"],
    "CWE-78": ["import subprocess", "import shlex"],
    "CWE-22": ["import os"],
    "CWE-502": ["import pickle", "import json"],
    "CWE-798": ["import os"],
    "CWE-79": ["import html"],
    "CWE-94": ["import ast"],
    "CWE-327": ["import hashlib"],
    "CWE-330": ["import random", "import secrets"],
    "CWE-611": ["import xml.etree.ElementTree as ET"],
    "CWE-918": ["import requests", "from urllib.parse import urlparse"],
    "CWE-295": ["import requests", "import ssl"],
    "CWE-377": ["import os", "import tempfile"],
    "CWE-732": ["import os", "import stat"],
    "CWE-1333": ["import re"],
    "CWE-209": ["import logging", "import traceback"],
}

# DIVERSITY AXIS 2 -- how far untrusted input travels before reaching the
# unsafe call, and whether partial validation stands in the way. The precise
# meaning of each level is spelled out in generate.py's system prompt.
#
# This is the main lever for making the benchmark hard enough to separate
# tools. A detector scoring perfectly means the difficulty distribution needs
# to shift, not that the detector is finished.
DIFFICULTIES = ["Easy", "Medium", "Hard"]


# ===========================================================================
# SECTION 3 -- Dataset size and reproducibility
# ===========================================================================

# Samples are generated in vulnerable/safe pairs, so this MUST be even --
# generate.py refuses to run otherwise. With GENERATE_TWINS = False the
# constraint does not apply and any count is accepted.
N_SAMPLES = 10

# WHETHER A GENERATED SAMPLE COMES WITH ITS TWIN.
#
#   True  -- the default and the benchmark proper. One request produces two
#            versions of ONE program: with the flaw and with it corrected,
#            differing as little as possible. The twin is the control for its
#            vulnerable half, so a verdict that differs between them is
#            attributable to the flaw and nothing else -- not to length, not
#            to imports, not to how tidy the code looks. The pairing is also
#            what detects the generator annotating its own work (see
#            generate.py's check_pair), what supplies the negative class, and
#            what makes the dataset 50/50 by construction.
#
#   False -- each sample is an independent program, asked for as vulnerable
#            or as safe, with no counterpart. This is how most published
#            vulnerability-detection benchmarks are built, so it is what a
#            comparison against them has to be measured on; it also halves
#            the tokens per sample and lets an odd sample count be requested.
#
#            Two things are genuinely lost, and both are stated rather than
#            worked around. First, there is no control: a detector can score
#            well by reacting to surface properties of the two populations
#            instead of to the vulnerability, and nothing in the dataset can
#            tell the difference. Second, the "no vulnerability at all" claim
#            on a safe sample has no twin to compare it against, so the judge
#            in stage 2 becomes its only check -- the structural checks that
#            need both halves (identical_versions, length_mismatch) cannot run.
#
# Topic is held constant either way: an unpaired safe sample is still asked
# for a program that performs the same KIND of operation as the vulnerable
# samples of that CWE. Without that, the two classes would differ in subject
# matter and the safe set would be separable from the vulnerable set by
# nothing more interesting than which modules it imports.
GENERATE_TWINS = True

# Seeds the CWE/context/difficulty sampling and the pre-naming shuffle, so a
# run is reproducible: the same seed regenerates the same benchmark
# *configuration*. It does not make the generated code identical -- the model
# is not deterministic -- so the programs still differ between runs.
# Set to None for fresh randomness each time.
SEED = 20260810


# ===========================================================================
# SECTION 4 -- Label vocabulary
# ===========================================================================
# Defined as constants rather than written as bare strings throughout the
# pipeline, so ground truth and every detector are guaranteed to agree. A typo
# in a literal would otherwise silently score every sample as a mismatch.

LABEL_VULNERABLE = "vulnerable"
LABEL_SAFE = "not_vulnerable"

# Used by ingested (GitHub / local) code, which arrives without ground truth.
# evaluate.py treats these samples separately: they can be used to measure
# tool *agreement*, but never accuracy, because there is nothing to be
# accurate against.
LABEL_UNKNOWN = "unknown"


# ===========================================================================
# SECTION 5 -- Judgment stage
# ===========================================================================

# How many judges must agree with the generator's intended label before a
# sample is admitted to the benchmark.
#   "all"       -- every selected judge must agree (strict; smaller, cleaner)
#   "majority"  -- more than half must agree (needs 3+ judges to differ
#                  from "all"; with 2 judges it is the same rule)
#   "any"       -- one agreeing judge is enough (permissive; larger, noisier)
#
# "all" is the default because the purpose of this stage is to remove the
# "ground truth comes from the generator" weakness. A sample that two models
# disagree about is exactly the sample whose label cannot be trusted.
#
# WHEN TO MOVE TO "majority"
#   Only with an odd number of judges, and for a specific reason: with two
#   judges a disagreement cannot be resolved, so "all" must treat it as fatal
#   and the dataset pays for every dispute. With three, two agreeing judges
#   outvote one dissenter, so a single harsh or miscalibrated model stops
#   deciding the size of the dataset on its own.
#
#   This is weaker verification than "all" and has to be reported as such: a
#   majority-admitted sample has one judge on record saying the label is
#   wrong. The dropped set still holds every dissent, so the cost is
#   measurable rather than hidden -- compare the two rules on the same run
#   before choosing one for the write-up.
#
#   Abstentions do not count towards the total. A judge whose call failed
#   never read the program, so a majority is computed over the judges that
#   actually answered (see the abstention note in judge.py).
JUDGE_CONSENSUS = "all"

# A judge must also confirm the sample contains EXACTLY ONE vulnerability.
# A program with two flaws breaks per-CWE scoring: a detector that finds the
# second flaw but not the intended one is marked correct on the label and
# wrong on the category, which is not a meaningful measurement.
REQUIRE_SINGLE_VULNERABILITY = True

# Verdicts below this confidence are treated as abstentions and the sample is
# dropped, on the same principle: an uncertain label is not ground truth.
JUDGE_MIN_CONFIDENCE = 0.60


# ===========================================================================
# SECTION 6 -- Detector behaviour
# ===========================================================================

# Whether a static analyser's finding counts towards a "vulnerable" verdict
# only when its CWE is one of this benchmark's categories.
#
# WHY THIS MATTERS -- it moves the headline numbers.
#   A security linter reports everything it knows about. On the first generated
#   pair, Bandit and Ruff both flagged "requests call without timeout"
#   (CWE-400) on the vulnerable AND the safe program. With scoping off, that
#   one rule marks every sample positive and drives the false positive rate to
#   1.0 for a reason unrelated to the question the benchmark asks.
#
#   True  -- score the tools on the categories they were asked about. Fairer
#            comparison against the LLM and ML arms, which are only ever asked
#            about these sixteen.
#   False -- score raw out-of-the-box behaviour, including unrelated findings.
#            Closer to what a developer experiences, and much noisier.
#
# Both counts are recorded in every static result file regardless of this
# setting, so the other number can always be reported alongside.
STATIC_SCOPE_TO_BENCHMARK_CWES = True

# --- Localisation scoring -------------------------------------------------
# How far a reported line may sit from a known vulnerable line and still count
# as having found it.
#
# Zero would be wrong. A static analyser points at the call it objects to, a
# fix commit often changes the line above or below it as well (a guard added,
# an argument rewritten across two lines), and an LLM asked for "the line"
# picks one line of a multi-line statement. Requiring an exact match would
# score correct findings as misses.
#
# Two lines is tight enough that a tool cannot earn credit by pointing
# somewhere unrelated -- at this tolerance a 150-line file has roughly 30
# distinct places to be wrong -- and loose enough to absorb the off-by-one
# that the cases above produce. Stated here rather than buried in evaluate.py
# because it is an editorial choice that moves the numbers.
LOCALISATION_TOLERANCE_LINES = 2

# Cross-validation folds for the TRAINED half of the ML arm (ml_tfidf,
# ml_embedding).
#
# Those two detectors learn from the same small dataset they are scored on, so
# scoring them on their training data would report memorisation as detection.
# Instead every prediction is made out of fold: the model that labels a sample
# was never fitted on it. See ml_models.py, "FAMILY B".
#
# Lowered automatically when a class has fewer members than the fold count,
# since a stratified split cannot produce more folds than the rarest class has
# members. The pretrained detectors ignore this entirely -- they do no fitting.
ML_CV_FOLDS = 5

# --- The pretrained ML arm -------------------------------------------------
# The ML arm runs two *published, frozen* models rather than classifiers
# trained here. Both were trained on real Python security fixes scraped from
# GitHub, both target web vulnerabilities, and neither has ever seen this
# benchmark's samples -- so for the first time the ML arm is scored under
# exactly the same protocol as the static analysers and the LLMs, with no
# cross-validation caveat attached to its numbers.
#
#   ml_farasat   BiLSTM + word2vec. Farasat & Posegga, ACM CODASPY 2024.
#   ml_vudenc    LSTM + word2vec.   Wartschinski et al., IST 144 (2022).
#
# The Passau model is a direct successor to VUDENC -- same dataset, same
# preprocessing, deeper recurrent stack -- so running both measures what the
# architecture change bought, on data neither author chose.

# The seven vulnerability types both model families cover, and the CWE each
# one corresponds to. Every type has its own binary model; a sample is run
# through all seven and the strongest responder names the category, which is
# how these detectors answer the benchmark's "which weakness?" question.
#
# The names are the authors' own file-naming scheme and must match the weight
# filenames exactly.
VUDENC_VULNERABILITIES = [
    "sql",
    "xss",
    "command_injection",
    "remote_code_execution",
    "path_disclosure",
    "open_redirect",
    "xsrf",
]

# Mapping onto CWE ids. The authors label by vulnerability name, not by CWE,
# so this mapping is ours and is stated here rather than buried in the
# detector -- it decides whether a correct detection is also scored as a
# correct *identification*.
#
# Five of the seven fall inside this benchmark's catalogue. The other two --
# open redirect (CWE-601) and CSRF (CWE-352) -- do not, because neither is a
# category the generator produces. They are still run and still recorded: a
# model firing CWE-601 on a sample whose planted flaw is CWE-89 is a false
# positive, and suppressing those predictions would flatter the models.
VUDENC_CWE_MAP = {
    "sql": "CWE-89",
    "xss": "CWE-79",
    "command_injection": "CWE-78",
    "remote_code_execution": "CWE-94",
    "path_disclosure": "CWE-22",
    "open_redirect": "CWE-601",
    "xsrf": "CWE-352",
}

# COVERAGE -- the honest limitation of this arm.
#
# Those five in-catalogue CWEs are five of the benchmark's sixteen. For the
# other eleven (502, 798, 327, 330, 611, 918, 295, 377, 732, 1333, 209) these
# models have no detector at all and will answer "safe" every time -- not
# because the code is clean but because nobody trained them on the question.
#
# Scoring that as a miss would understate the models by more than half, and
# hiding it would overstate them. So every ml_* result file records a
# `coverage` block naming which samples fell inside the models' competence,
# and the write-up must quote it next to the score.
#
# Note this is NOT the `coverage` field evaluate.py computes: that one means
# "fraction of samples this detector returned a verdict for", which is 1.0
# here because an untrained category still produces an (always-safe) answer.
# The two are different quantities and should not be conflated in the report.
# This is the same kind of disclosure the static arm makes with
# findings_in_scope.
VUDENC_COVERED_CWES = sorted(
    {cwe for cwe in VUDENC_CWE_MAP.values()}
)

# Decision threshold on the per-window probability. 0.5 is the authors' own
# operating point, used in both papers' reported precision/recall; moving it
# would make the numbers here incomparable with the published ones.
VUDENC_THRESHOLD = 0.5

# Preprocessing constants. These are NOT tunable: they describe the shape the
# published weights were trained on, so changing one silently feeds the model
# a representation it has never seen. Both papers use the same values.
VUDENC_STEP = 5             # tokens between successive window centres
VUDENC_CONTEXT_LENGTH = 200  # characters per context window
VUDENC_MAX_LENGTH = 200      # padded sequence length fed to the network

# How many windows to push through the network at once. Pure throughput knob,
# safe to change; the benchmark's samples are small so this is rarely binding.
VUDENC_BATCH_SIZE = 64

# Where the downloaded artefacts live, relative to models/. Separate
# directories because the two families use different word2vec dimensionalities
# (300 for Passau, 200 for VUDENC) and the filenames would otherwise collide.
VUDENC_FARASAT_DIR = "farasat-bilstm"
VUDENC_ORIGINAL_DIR = "vudenc-lstm"

# SUBSTITUTING FARASAT'S ARTEFACTS FOR VUDENC'S.
#   VUDENC's published word2vec carries its vocabulary (132,891 tokens) but
#   not its vector array: gensim wrote that to a `.model.wv.vectors.npy`
#   sidecar that the authors never uploaded. Confirmed against the Zenodo
#   record (3559480), which lists only the 9 MB header. Without it the
#   original LSTM cannot be run at all.
#
#   With this switch on, the `ml_vudenc` slot loads Farasat's 300-dimensional
#   word2vec and her BiLSTM weights instead, at her instruction.
#
#   WHAT THAT PRODUCES, STATED PLAINLY. Both the embedding and the network
#   then come from Farasat & Posegga, and the pipeline around them is the same
#   one `ml_farasat` already uses -- so this slot becomes a second copy of
#   `ml_farasat`, not a measurement of VUDENC. Nothing of Wartschinski et al.
#   remains in it except the name of the directory it no longer reads.
#
#   It is therefore wired up to be honest about itself rather than to look
#   like a VUDENC result: the detector renames itself, and every result file
#   it writes carries `substituted: true` with the reason, so no table can
#   report it as the original model by accident.
#
#   The alternative the advisor also raised -- keeping VUDENC's LSTM and
#   reshaping her 300-dim vectors down to 200 -- is NOT what this does, and
#   is not offered. A network's weights are fitted against one specific
#   embedding instance; word2vec coordinates are arbitrary up to rotation, so
#   vectors from a different training run are not a smaller version of the
#   right numbers, they are different numbers of the right shape. It would
#   run and produce scores that mean nothing.
VUDENC_USE_FARASAT_WEIGHTS = True


# ===========================================================================
# SECTION 7 -- Comparison scoring
# ===========================================================================
# There is deliberately no scoring configuration here any more.
#
# This section used to define COMPOSITE_WEIGHTS, a weighted sum of F1,
# (1 - FPR) and CWE accuracy that produced one number per detector and decided
# the ranking. It was removed: collapsing three different questions into one
# number hid which of them a tool actually failed, and the weights were an
# editorial choice that silently determined the ordering.
#
# evaluate.py now reports the standard confusion matrix and the measures
# derived from it -- accuracy, precision, recall, F1, FPR, FNR, MCC, Youden's
# J, and CWE identification accuracy -- for every detector, with no combined
# score. Those are what the vulnerability-detection literature reports, so the
# numbers can be compared against other work without first learning a formula
# private to this project.
#
# The summary table is ordered by F1 so it is stable and readable. That is a
# sort order, not a score.
