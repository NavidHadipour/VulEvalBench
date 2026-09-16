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
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
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
DEFAULT_JUDGE_A = "gemma4-31b-it"
DEFAULT_JUDGE_B = "qwen36-35b"

# Non-chat models on the same endpoint, used by the ML arm rather than for
# judgment. Listed here so there is one place that records what the key unlocks.
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
# positive rate, which is exactly the trade-off the composite score and the
# (1 - FPR) term exist to expose.
DEFAULT_LLM_DETECTOR = "qwen3-next-80b-a3b-instruct"

# Anthropic models, available when `api-key-llm.txt` holds an Anthropic key.
# Offered as an option so the local/remote comparison can be extended to a
# second vendor, which is the credibility gap named in the README.
ANTHROPIC_MODELS = {
    "claude-opus-5": "Claude Opus 5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
}


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

# How the CWE for each pair is chosen.
#   "random"  -- drawn uniformly at random from CWE_CATEGORIES.
#   "cycle"   -- stepped through in order, giving an exactly balanced dataset.
#
# TRADE-OFF worth knowing: random selection is what makes each run an
# independent draw, but with a small N it will leave some categories empty and
# others doubled, so the per-CWE rows of the metrics table become noisy.
# Cycling guarantees balance at the cost of making runs non-independent.
# Random is the default because the supervisor asked for random selection.
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
# generate.py refuses to run otherwise.
N_SAMPLES = 10

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
#   "all"  -- every selected judge must agree (strict; smaller, cleaner set)
#   "any"  -- one agreeing judge is enough (permissive; larger, noisier set)
#
# "all" is the default because the purpose of this stage is to remove the
# "ground truth comes from the generator" weakness. A sample that two models
# disagree about is exactly the sample whose label cannot be trusted.
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

# Cross-validation folds for the ML arm.
#
# The ML detectors learn from the same small dataset they are scored on, so
# scoring them on their training data would report near-perfect numbers that
# mean nothing. Instead every prediction is made out-of-fold: the model that
# labels a sample was never trained on it. See ml_models.py.
#
# Lowered automatically when a class has fewer members than the fold count,
# since a stratified split cannot produce more folds than the rarest class has
# members.
ML_CV_FOLDS = 5


# ===========================================================================
# SECTION 7 -- Comparison scoring
# ===========================================================================
# The composite score that ranks tool sets against each other. Kept as an
# explicit weighted sum rather than a learned or hidden formula, because a
# thesis has to be able to defend why one tool ranked above another.
#
#   composite = W_F1 * F1  +  W_TNR * (1 - FPR)  +  W_CWE * CWE_accuracy
#
# The three terms answer three different questions:
#   F1              -- overall detection quality on the vulnerable class
#   (1 - FPR)       -- how much a tool can be trusted not to cry wolf
#   CWE_accuracy    -- whether it understood *what* the flaw was, not just
#                      that something was wrong
#
# Weights must sum to 1.0; evaluate.py asserts this so a mistyped weight
# cannot silently produce scores above 1.
COMPOSITE_WEIGHTS = {
    "f1": 0.50,
    "tnr": 0.30,
    "cwe": 0.20,
}
