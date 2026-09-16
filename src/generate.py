"""Stage 1 of the pipeline -- test case generation with a local Ollama model.

Asks the generator for PAIRS of programs: one containing exactly one specified
vulnerability, and a safe twin that differs only in the minimal change needed
to fix it.

WHY PAIRS
    If vulnerable and safe programs are generated independently they end up
    differing in style, length, structure and phrasing as well as in security.
    A detector can then score highly by recognising which prompt produced a
    file -- effectively classifying writing style, not vulnerabilities. Holding
    everything constant except the flaw removes that shortcut, so a correct
    verdict has to come from analysing the logic.

WHY A LOCAL MODEL
    Generation is the most token-hungry stage in the project: every call
    returns two complete programs. Running it locally through Ollama makes the
    cost of scaling to several hundred samples zero, and it separates the
    model that *writes* a sample from the models that *judge* and *detect* it,
    which removes self-preference bias by construction.

EXACTLY ONE VULNERABILITY PER SAMPLE
    Enforced in three places, because a prompt alone is not enough:
      1. the system prompt states it as a hard requirement,
      2. this module rejects pairs that fail its structural checks, and
      3. stage 2 (judge.py) asks each judge to count the vulnerabilities it
         finds and drops any sample where the count is not one.
    A program with two flaws breaks per-CWE scoring: a detector that finds the
    second flaw but not the intended one is scored correct on the label and
    wrong on the category, which measures nothing meaningful.

WHAT THIS FILE DOES NOT DO
    It contains no vulnerable code. The model invents every program from a
    prompt; this module only decides what to ask for, checks the answer is
    usable, and files it.

Inputs:
    src/config.py               benchmark axes and model choice
Outputs:
    runs/<run_id>/01_generation/samples/sample_NNN.py
    runs/<run_id>/01_generation/generation_manifest.json
    logs/<run_id>.log

Run with:
    python src/generate.py
"""

from __future__ import annotations

import ast
import random
import re

import config
import runstore
from common import get_logger, write_json
from providers import ProviderError, chat_json

# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------
# Passed to Ollama as a grammar constraint, so the model physically cannot
# return anything but an object with these four string fields. That removes an
# entire class of "the model wrapped it in markdown" failures.
#
# `additionalProperties: false` stops the model padding the object with extra
# commentary fields, which smaller models are prone to do.

# Output budget for one pair, and the single biggest lever on run time.
#
# MEASURED, on a 50-pair run: a normal pair completes in 41 seconds (14 s
# fastest, 76 s slowest) and produces about 762 output tokens. The budget was
# 6000 -- roughly seven times what a pair actually needs.
#
# That headroom was not free. Nine of 49 pairs hit the 600-second timeout, and
# those are pairs where the model stopped converging and generated until it
# exhausted the budget. Every one cost ten minutes and produced nothing: about
# 90 minutes of a single run spent on runaway generations.
#
# Sized from the pairs that actually succeeded. Reconstructing the JSON payload
# for 20 known-good pairs gives a median of 276 tokens and a MAXIMUM of 546 --
# an order of magnitude below the old budget. A cap of 1500 would have
# truncated none of them.
#
# 2500 is therefore about 4.5x the largest pair ever produced, with room for
# the fuller 30-70 line programs the prompt asks for (a pair at that size is
# roughly 1600 tokens). Anything exceeding it is not a long program, it is a
# model that has stopped converging: the runaway pairs generate five times more
# than any successful pair ever has.
#
# Bounding it matters because a runaway used to cost the full 600-second
# timeout and produce nothing -- nine such pairs consumed about 90 minutes of a
# single 50-pair run. A truncated pair is now rejected cleanly by the
# incomplete-response check instead of crashing the run.
MAX_GENERATION_TOKENS = 2500

PAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "vulnerable_code": {
            "type": "string",
            "description": "Complete Python program containing the requested vulnerability.",
        },
        "safe_code": {
            "type": "string",
            "description": "The same program with the vulnerability fixed, minimal diff.",
        },
        "vulnerability_explanation": {
            "type": "string",
            "description": "One or two sentences: where the flaw is and how untrusted input reaches it.",
        },
        "fix_summary": {
            "type": "string",
            "description": "One sentence describing exactly what changed in the safe version.",
        },
    },
    "required": [
        "vulnerable_code", "safe_code",
        "vulnerability_explanation", "fix_summary",
    ],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# The system prompt carries the rules that hold for every request. Most of it
# is leakage prevention: if either program hints at which one it is, the
# benchmark measures nothing.

SYSTEM_PROMPT = """\
You generate Python programs for an academic vulnerability-detection benchmark.

For each request you produce TWO versions of the SAME program:

1. `vulnerable_code` -- contains EXACTLY ONE security vulnerability, of the
   requested CWE category and no other. The flaw must be genuinely reachable
   from untrusted input (a request parameter, a CLI argument, an environment
   value, a file read from disk). It must be a real security defect, not a
   stylistic issue.

2. `safe_code` -- the identical program with that one flaw corrected, changing
   as little as possible. Same imports, same function and variable names, same
   control flow, same comments, same overall length. Only the specific lines
   that constitute the vulnerability may differ.

THINK OF IT AS ONE PROGRAM, WRITTEN TWICE

Write the complete program first, in your head, including all the real work it
does -- opening the database, reading the file, making the request. Then emit
it twice: once with the flawed line or lines, once with the corrected ones.
Everything else must be character-for-character the same.

- BOTH versions must be equally complete and realistic. Never write a stub,
  a placeholder, a `print()` standing in for real work, or a comment such as
  "placeholder for actual database interaction" in either one.
- If one version connects to a database, opens a file or makes a request, the
  other must do exactly the same in exactly the same way.
- The two versions must be within a couple of lines of each other in length.
  A large difference means you rewrote the program instead of fixing one flaw,
  and the pair is unusable.

HARD REQUIREMENTS

- EXACTLY ONE vulnerability in `vulnerable_code`. Do not add a second flaw of
  any kind, not even a minor one. Every other part of the program must be
  written securely. `safe_code` must contain no vulnerabilities at all.
- Neither version may hint at which one it is. Never use words such as
  vulnerable, insecure, unsafe, safe, secure, fixed, patched, sanitised,
  sanitized, validated, exploit, injection, attack, malicious or dangerous in
  ANY comment, docstring, string literal, function name or variable name. A
  reader must be unable to tell the two apart without analysing the logic.
- Both versions must carry the SAME comments and docstrings. Do not add a
  comment to one version that is absent from the other.
- Write realistic, idiomatic code that would pass casual review: real imports,
  plausible surrounding logic, roughly 30 to 70 lines.
- IMPORT EVERY MODULE AND NAME YOU USE, at the top of both versions. If the
  program calls `pickle.load`, it must `import pickle`. If it calls
  `requests.get`, it must `import requests`. If it returns an `HttpResponse`,
  it must import it from django.http. A missing import makes the program crash
  on its first call, and a program that cannot run is discarded.
- Call your own functions with exactly the arguments they declare. Do not pass
  a keyword the function signature does not accept.
- Both versions must be syntactically valid Python that could actually run.
- Emit raw Python source only in the code fields. No markdown fences, no prose.

DIFFICULTY controls how obvious the flaw is:
- Easy    -- the unsafe call sits plainly in a short, direct path.
- Medium  -- untrusted input passes through a helper or two before reaching it.
- Hard    -- the input crosses several functions, and there is partial or
             incomplete validation along the way that does not actually stop
             the attack.
"""

# The user prompt carries only what varies between requests. The mechanism hint
# and the "do not reuse a textbook example" instruction both fight the same
# tendency: a small local model will otherwise emit the same canonical snippet
# for a given CWE every single time, making the dataset repetitive and trivially
# memorisable.
# `Required fix` is the important line. Without it the generator injects the
# flaw correctly and then fails to repair it -- see config.CWE_FIXES for the
# measured failure this addresses. Supplying the remedy turns the safe half
# from an invention task into an application task, which is what a small model
# can actually do.
USER_PROMPT = """\
Generate one benchmark pair.

CWE:          {cwe} ({cwe_name})
Mechanism:    {hint}
Required fix: {fix}
Context:      {context}
Difficulty:   {difficulty}

Both versions must begin with at least these imports, plus any others the
program uses:

{context_imports}

`vulnerable_code` must contain the flaw described under Mechanism.
`safe_code` must be the same program with the Required fix applied, and must
contain NO vulnerability of any kind. Applying the fix must genuinely close the
hole: a change that only looks like the fix, without removing the attacker's
control, is wrong.

Invent the program yourself: choose the feature it implements, the function
names and the surrounding logic. Do not reuse a textbook example.
"""


# ---------------------------------------------------------------------------
# Post-generation checks
# ---------------------------------------------------------------------------
# A local 7B model follows instructions less reliably than a frontier model, so
# its output is checked rather than trusted. Each check returns a flag string
# instead of raising: one bad pair should be recorded and skipped, not abort a
# forty-pair run.

# Words that would reveal a sample's label to a reader or a detector.
# Checked case-insensitively against whole words, so "safely" does not trip
# the "safe" rule but "# safe version" does.
LEAKAGE_WORDS = [
    "vulnerable", "vulnerability", "insecure", "unsafe", "safe", "secure",
    "fixed", "patched", "sanitise", "sanitize", "sanitised", "sanitized",
    "exploit", "injection", "attack", "malicious", "dangerous", "cwe",
    # Added after observing real generator output: the model wrote the comment
    # "# Incorrectly creating a directory with permissions 0777" into the
    # vulnerable half. None of the words above catch that, yet it hands the
    # label to any reader. Judgemental adjectives are as much of a leak as the
    # word "vulnerable" itself.
    "incorrect", "incorrectly", "improper", "improperly", "flaw", "flawed",
    "weakness", "hardcoded",
]

# NOTE ON THE BOUNDARIES -- do not replace these with `\b`.
#   `\b` treats the underscore as a word character, so `\bsafe\b` does NOT
#   match the identifier `safe_url`. That is exactly the form the leak takes in
#   practice: a model told not to write "safe" in a comment will happily name a
#   variable `safe_url` or `is_safe`, handing the label to any detector that
#   reads identifiers. Using explicit letter-only lookarounds makes `_` count
#   as a boundary, so `safe_url`, `url_safe` and `isSafe` are all caught, while
#   genuine words that merely contain a keyword (`safely`, `unsafely`) are not
#   double-counted beyond their own entry in the list.
LEAKAGE_RE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(LEAKAGE_WORDS) + r")(?![A-Za-z])",
    re.IGNORECASE,
)

# Inserts a separator at every lowercase-to-uppercase transition, so camelCase
# and PascalCase identifiers are broken into their constituent words before the
# leakage scan runs. Without this, `isSafe` and `SafeLoader` would pass a check
# that correctly rejects `is_safe` -- the same leak wearing different casing.
CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def find_leakage(source: str) -> set[str]:
    """Return the set of label-revealing words appearing in `source`.

    Identifiers are split on case transitions first so that snake_case,
    camelCase and PascalCase are all covered by the one word list.

    Args:
        source: a generated program's text.

    Returns:
        The lowercased keywords found, empty when the program is clean.
    """
    normalised = CAMEL_BOUNDARY_RE.sub("_", source)
    return {m.group(1).lower() for m in LEAKAGE_RE.finditer(normalised)}

def strip_fences(code: str) -> str:
    """Remove markdown code fences the model added despite instructions.

    WHY THIS IS NOT A REGEX ANY MORE
        The obvious implementation matches a *complete* fence -- an opening
        ```` ```python ```` line, a body, and a closing ```` ``` ```` -- and
        returns the body. That misses the failure that actually happens.

        The local generator frequently emits a *trailing* fence with no opening
        one: valid Python, then a bare ```` ``` ```` on the final line. An
        anchored whole-string pattern does not match that at all, so the stray
        fence survived into the sample file and every such program failed to
        parse. Two of six pairs in one run were rejected for exactly this,
        which looked like model incompetence and was in fact this function's
        fault.

        Handling the opening and closing fences independently covers all three
        observed cases: both present, only the opening, only the closing.

    Also normalises the file to end in exactly one newline.
    """
    lines = code.strip().splitlines()

    # A leading fence, with or without a language tag.
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]

    # Trailing fences, however many the model produced.
    while lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip() + "\n"


def scrub_leaking_comments(code: str) -> tuple[str, int]:
    """Delete comments that name the label, from either half of a pair.

    WHY THIS EXISTS
        The commonest leak by far is not a variable name -- it is the model
        annotating its own work: `# Vulnerable SQL query`,
        `# Incorrectly creating a directory with permissions 0777`. Both the 3B
        and the 7B generator do this regularly despite an explicit instruction
        not to, so it cannot be prompted away reliably.

        A leaked comment is also the one kind of leak that can be removed
        safely and deterministically. Comments carry no semantics: deleting one
        cannot change whether the program is vulnerable, so the sample stays
        valid while the giveaway disappears. Identifier leaks (`safe_url`)
        cannot be auto-fixed the same way -- renaming could collide or change
        behaviour -- so those remain a reported flag for the judges to weigh.

    WHY IT IS APPLIED TO BOTH HALVES BY THE SAME RULE
        The pair design requires both versions to carry identical comments. The
        rule here is purely local -- "drop this line if it names the label" --
        and is applied independently to each half, so a comment that leaks in
        only one version is removed from that version alone. That is precisely
        the asymmetry we want gone: it existed only because one half was
        annotated and the other was not.

    Args:
        code: one program's source.

    Returns:
        `(cleaned_source, number_of_comments_removed)`.
    """
    cleaned: list[str] = []
    removed = 0

    for line in code.splitlines():
        stripped = line.lstrip()

        # A whole-line comment that names the label: drop the line entirely.
        if stripped.startswith("#"):
            if find_leakage(stripped):
                removed += 1
                continue
            cleaned.append(line)
            continue

        # A trailing comment on a code line: keep the code, drop the comment.
        # Only split on a '#' that is outside a string literal, so a hash in
        # a URL or a format string is not mistaken for a comment.
        if "#" in line:
            in_string: str | None = None
            escaped = False
            for index, char in enumerate(line):
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == in_string:
                        in_string = None
                    continue
                if char in ("'", '"'):
                    in_string = char
                elif char == "#":
                    if find_leakage(line[index:]):
                        cleaned.append(line[:index].rstrip())
                        removed += 1
                    else:
                        cleaned.append(line)
                    break
            else:
                cleaned.append(line)
            continue

        cleaned.append(line)

    return "\n".join(cleaned).rstrip() + "\n", removed


def check_runtime_soundness(source: str) -> list[str]:
    """Catch code that parses but would crash on the first call.

    WHY `ast.parse` IS NOT ENOUGH
        Syntactic validity is a very low bar. Real generator output has
        repeatedly been valid Python that cannot run:

          - a Flask handler calling `requests.get(...)` with no `import
            requests` -- NameError on the first request;
          - `create_file(path, content)` invoked as
            `create_file(p, c, mode=0o600)` -- TypeError, because the model
            bolted the remedy from the prompt onto the call without adding the
            parameter.

        Both parse cleanly, so both reached the judges, who correctly refused
        to certify them and dropped the sample. Catching them here is cheaper
        than spending two judgment calls to discover the same thing, and it
        keeps a broken program out of the dataset entirely.

    Two checks:
      1. undefined names, via pyflakes;
      2. calls to functions defined in the same file with an argument count or
         keyword the signature cannot accept.

    Returns:
        A list of flag strings, empty when the program looks runnable.
    """
    flags: list[str] = []

    # --- 1. undefined names ------------------------------------------------
    try:
        import io

        from pyflakes.api import check as pyflakes_check
        from pyflakes.reporter import Reporter

        out, err = io.StringIO(), io.StringIO()
        pyflakes_check(source, "sample.py", Reporter(out, err))
        undefined = sorted({
            line.split("undefined name ")[1].strip().strip("'")
            for line in out.getvalue().splitlines()
            if "undefined name " in line
        })
        if undefined:
            flags.append(f"undefined_names:{','.join(undefined)}")
    except ImportError:
        # pyflakes is an optional convenience; without it this check is simply
        # skipped rather than failing the run.
        pass

    # --- 2. calls that do not match a local signature ----------------------
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return flags  # the caller's own syntax check already reports this

    signatures: dict[str, ast.arguments] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signatures[node.name] = node.args

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        args = signatures.get(node.func.id)
        if args is None:
            continue

        # A decorated function may be called through a wrapper with a
        # different signature, so decorated definitions are skipped.
        positional = len(args.posonlyargs) + len(args.args)
        accepts_extra_positional = args.vararg is not None
        accepts_extra_keyword = args.kwarg is not None

        given_positional = len(node.args)
        if not accepts_extra_positional and given_positional > positional:
            flags.append(
                f"call_arity:{node.func.id}_takes_{positional}_got_"
                f"{given_positional}"
            )
            continue

        if not accepts_extra_keyword:
            known = (
                {a.arg for a in args.posonlyargs}
                | {a.arg for a in args.args}
                | {a.arg for a in args.kwonlyargs}
            )
            unexpected = sorted(
                kw.arg for kw in node.keywords
                if kw.arg is not None and kw.arg not in known
            )
            if unexpected:
                flags.append(
                    f"call_keyword:{node.func.id}_rejects_{','.join(unexpected)}"
                )

    return flags


def check_pair(vulnerable: str, safe: str) -> list[str]:
    """Structurally validate one generated pair.

    These checks catch the failure modes a local model actually exhibits. They
    cannot confirm the vulnerability is real -- that is the judge's job in
    stage 2 -- only that the pair is well-formed enough to be worth judging.

    Args:
        vulnerable: the flawed program's source.
        safe: the corrected twin's source.

    Returns:
        A list of flag strings, empty when the pair passed everything. The
        flags are stored in the manifest so a rejected pair can be diagnosed
        from the run directory without re-running generation.
    """
    flags: list[str] = []

    # 1. Both halves must be valid Python. A file that does not parse cannot be
    #    analysed by the static tools at all, so it would silently distort
    #    their scores.
    for name, source in (("vulnerable", vulnerable), ("safe", safe)):
        try:
            ast.parse(source)
        except SyntaxError as exc:
            flags.append(f"syntax_error_{name}:line_{exc.lineno}")

    # 2. The two halves must actually differ. Small models sometimes return the
    #    same text twice, which would put two identically-coded samples into
    #    the dataset under opposite labels.
    if vulnerable.strip() == safe.strip():
        flags.append("identical_versions")

    # 2b. Neither half may be code that parses but cannot run. See
    #     check_runtime_soundness() for the failure modes this catches.
    for name, source in (("vulnerable", vulnerable), ("safe", safe)):
        for flag in check_runtime_soundness(source):
            flags.append(f"{flag}_{name}")

    # 3. Neither half may contain a label-revealing word, in any identifier
    #    casing -- see find_leakage().
    for name, source in (("vulnerable", vulnerable), ("safe", safe)):
        leaked = find_leakage(source)
        if leaked:
            flags.append(f"leakage_{name}:{','.join(sorted(leaked))}")

    # 4. The minimal-diff requirement. A safe twin that is a completely
    #    different program reintroduces exactly the style shortcut that pairing
    #    exists to remove. Compared by line count because it is robust to
    #    reformatting; the threshold is deliberately loose, flagging only
    #    gross violations rather than enforcing a strict diff size.
    vulnerable_lines = len(vulnerable.splitlines())
    safe_lines = len(safe.splitlines())
    longer = max(vulnerable_lines, safe_lines)
    if longer and abs(vulnerable_lines - safe_lines) / longer > 0.40:
        flags.append(
            f"length_mismatch:{vulnerable_lines}_vs_{safe_lines}_lines"
        )

    # 5. Length sanity. A ten-line program is too trivial to be a meaningful
    #    test case, and a 400-line one usually means the model lost the plot.
    for name, source in (("vulnerable", vulnerable), ("safe", safe)):
        line_count = len(source.splitlines())
        if line_count < 12:
            flags.append(f"too_short_{name}:{line_count}_lines")
        elif line_count > 250:
            flags.append(f"too_long_{name}:{line_count}_lines")

    return flags


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------
# The work lives in a function rather than in `main()` so both entry points can
# share it: the command line calls it below, and the web app calls it with a
# progress callback that streams into the browser. There is exactly one
# implementation of the generation logic.

def required_imports(context: str, cwe: str) -> str:
    """The import lines a sample for this context and CWE will need.

    Two sources are merged, because a sample needs both:
      - the framework its *context* implies (Django's HttpResponse), and
      - the module its *CWE mechanism* implies (subprocess, sqlite3, hashlib).

    Missing imports were the largest single cause of discarded pairs in two
    successive runs, and the two rounds of measurement pointed at different
    culprits each time -- framework names first, standard-library modules
    second. Supplying both removes the whole class.

    Order is preserved and duplicates removed, so a module named by both
    sources (`os` for a path-traversal CLI tool, say) appears once.

    Args:
        context: one of `config.CONTEXTS`.
        cwe: one of `config.CWE_CATEGORIES`.

    Returns:
        Newline-joined import statements, ready to drop into the prompt.
    """
    lines: list[str] = []
    for source in (config.CONTEXT_IMPORTS.get(context, ()),
                   config.CWE_IMPORTS.get(cwe, ())):
        for line in source:
            if line not in lines:
                lines.append(line)
    return "\n".join(lines)


def _cwe_sequence(categories: list[str], n_pairs: int, selection: str,
                  rng: random.Random) -> list[str]:
    """Decide which CWE each pair will target.

    WHY RANDOM SELECTION DRAWS WITHOUT REPLACEMENT
        The first implementation called `rng.choice()` per pair -- an
        independent uniform draw. Combined with a fixed seed, that produced a
        genuinely bad outcome: every run drew the *same* sequence, so ten pairs
        covered only seven of the sixteen categories and nine were never
        generated at all, across eight consecutive runs. The catalogue was
        expanded to sixteen categories and more than half of it was dead code.

        Shuffling the catalogue and walking it instead keeps selection random
        -- the order differs per seed, and which categories appear in a short
        run differs too -- while guaranteeing that a run of N pairs covers
        min(N, len(catalogue)) distinct categories. The catalogue is reshuffled
        each time it is exhausted, so longer runs stay varied rather than
        repeating one fixed order.

    Args:
        categories: the CWE ids to draw from.
        n_pairs: how many pairs the run will generate.
        selection: "cycle" for the catalogue in order, anything else for the
            shuffled draw described above.
        rng: the run's seeded Random instance.

    Returns:
        A list of `n_pairs` CWE ids.
    """
    if selection == "cycle":
        return [categories[i % len(categories)] for i in range(n_pairs)]

    sequence: list[str] = []
    while len(sequence) < n_pairs:
        deck = list(categories)
        rng.shuffle(deck)
        sequence.extend(deck)
    return sequence[:n_pairs]


def generate_dataset(
    n_samples: int | None = None,
    seed: int | None = None,
    generator_model: str | None = None,
    cwe_selection: str | None = None,
    cwe_subset: list[str] | None = None,
    label: str = "",
    on_progress=None,
) -> dict:
    """Generate a benchmark dataset into a fresh run directory.

    Args:
        n_samples: total samples to produce; must be even, since samples come
            in pairs. Defaults to `config.N_SAMPLES`.
        seed: seeds axis sampling and the pre-naming shuffle, making the run
            *configuration* reproducible. Defaults to `config.SEED`; pass None
            explicitly for fresh randomness.
        generator_model: Ollama model tag. Defaults to
            `config.GENERATOR_MODEL`.
        cwe_selection: "random" or "cycle"; see config.CWE_SELECTION for the
            trade-off between independent draws and guaranteed balance.
        cwe_subset: restrict generation to these CWE ids. Defaults to the full
            catalogue. Used by the web interface when a user ticks a subset.
        label: free-text note stored in the run metadata.
        on_progress: callable taking one string; receives every log line.

    Returns:
        A summary dict including the new `run_id`, which the caller needs in
        order to run any later stage against this dataset.

    Raises:
        SystemExit: on an odd `n_samples` or an empty CWE subset -- both are
            configuration errors, reported readably rather than as tracebacks.
    """
    # `or` is deliberately not used for these defaults: n_samples=0 and seed=0
    # are falsy but legitimate, and would be silently replaced by the config
    # value. An explicit `is None` test treats them correctly.
    n_samples = config.N_SAMPLES if n_samples is None else n_samples
    seed = config.SEED if seed is None else seed
    generator_model = (
        config.GENERATOR_MODEL if generator_model is None else generator_model
    )
    cwe_selection = (
        config.CWE_SELECTION if cwe_selection is None else cwe_selection
    )
    categories = list(cwe_subset) if cwe_subset else list(config.CWE_CATEGORIES)

    if n_samples % 2 != 0:
        raise SystemExit("n_samples must be even: samples are generated in pairs.")
    if not categories:
        raise SystemExit("No CWE categories selected; nothing to generate.")

    # The run directory and its log exist before the first API call, so a run
    # that fails halfway still leaves a diagnosable record on disk.
    run = runstore.create_run(
        source="generated",
        label=label,
        config_snapshot={
            "n_samples": n_samples,
            "seed": seed,
            "generator_model": generator_model,
            "cwe_selection": cwe_selection,
            "cwe_categories": categories,
        },
    )
    log = get_logger(run.run_id, on_progress)

    # A dedicated Random instance rather than the global `random` module: the
    # seed then controls this pipeline's choices only, without perturbing any
    # other library that relies on global random state.
    rng = random.Random(seed)

    n_pairs = n_samples // 2
    cwe_sequence = _cwe_sequence(categories, n_pairs, cwe_selection, rng)
    records: list[dict] = []
    rejected: list[dict] = []
    in_tokens = out_tokens = 0

    log.info(f"Run {run.run_id}")
    log.info(
        f"Generating {n_pairs} pairs ({n_samples} samples) with "
        f"{generator_model} via Ollama"
    )
    log.info(
        f"CWE selection: {cwe_selection} over {len(categories)} categories"
    )

    for index in range(n_pairs):
        # --- choose this pair's axes ---------------------------------------
        cwe = cwe_sequence[index]

        cwe_name = config.CWE_NAMES.get(cwe, cwe)
        hint = config.CWE_HINTS.get(cwe, "")
        context = rng.choice(config.CONTEXTS)
        difficulty = rng.choice(config.DIFFICULTIES)

        log.info(
            f"  [{index + 1}/{n_pairs}] {cwe:<9} {difficulty:<7} {context}"
        )

        # --- ask the model -------------------------------------------------
        # max_tokens is generous because one response carries two complete
        # programs; running out mid-answer would truncate the second one.
        try:
            result = chat_json(
                provider="ollama",
                model=generator_model,
                system=SYSTEM_PROMPT,
                user=USER_PROMPT.format(
                    cwe=cwe, cwe_name=cwe_name, hint=hint,
                    fix=config.CWE_FIXES.get(cwe, "remove the flaw at its root"),
                    context=context, difficulty=difficulty,
                    context_imports=required_imports(context, cwe),
                ),
                schema=PAIR_SCHEMA,
                max_tokens=MAX_GENERATION_TOKENS,
            )
        except ProviderError as exc:
            # One failed pair must not destroy a long run: record it and move
            # on, so a forty-pair generation survives a single bad response.
            log.warning(f"      pair {index + 1} failed: {exc}")
            rejected.append({
                "pair_index": index + 1, "cwe": cwe,
                "reason": "provider_error", "detail": str(exc)[:500],
            })
            continue

        in_tokens += result.input_tokens
        out_tokens += result.output_tokens

        # A response truncated mid-object is salvaged by the provider's JSON
        # repair, which can return a pair missing `safe_code` entirely. Reading
        # that key directly raised KeyError and aborted a fifty-pair run at
        # pair seven, losing every pair generated before it. A pair without
        # both halves is simply unusable, so it is rejected like any other bad
        # pair and the run continues.
        missing = [
            field for field in ("vulnerable_code", "safe_code")
            if not (result.data.get(field) or "").strip()
        ]
        if missing:
            truncated = result.data.get("__repaired__")
            log.warning(
                f"      rejected: incomplete response, missing "
                f"{', '.join(missing)}"
                + (" (response was truncated)" if truncated else "")
            )
            rejected.append({
                "pair_index": index + 1, "cwe": cwe,
                "reason": "incomplete_response",
                "missing_fields": missing,
                "truncated": bool(truncated),
            })
            continue

        vulnerable = strip_fences(result.data["vulnerable_code"])
        safe = strip_fences(result.data["safe_code"])

        # --- repair what can be repaired -----------------------------------
        # Comments naming the label are removed before validation, since they
        # carry no semantics and are the single commonest leak. Identifier
        # leaks survive into the flags below for the judges to weigh.
        vulnerable, scrubbed_v = scrub_leaking_comments(vulnerable)
        safe, scrubbed_s = scrub_leaking_comments(safe)
        if scrubbed_v or scrubbed_s:
            log.info(
                f"      scrubbed {scrubbed_v + scrubbed_s} label-revealing "
                "comment(s)"
            )

        # --- validate ------------------------------------------------------
        flags = check_pair(vulnerable, safe)

        # Syntax errors and identical twins make a pair unusable; the other
        # flags are recorded as quality warnings but the pair is kept, because
        # the judge in stage 2 is a better arbiter of borderline cases than a
        # line-count heuristic.
        # Fatal flags make a pair unusable, so it is discarded rather than
        # sent to the judges. Code that cannot run is fatal for the same reason
        # a syntax error is: the static analysers would parse a program that
        # crashes on its first call, and the judges would (correctly) refuse to
        # certify it -- costing two judgment calls to reach the same verdict
        # this check reaches for free.
        fatal = [
            f for f in flags
            if f.startswith(("syntax_error", "undefined_names", "call_arity",
                             "call_keyword"))
            or f == "identical_versions"
        ]
        if fatal:
            log.warning(f"      rejected: {', '.join(fatal)}")
            rejected.append({
                "pair_index": index + 1, "cwe": cwe,
                "reason": "failed_checks", "flags": flags,
            })
            continue

        if flags:
            log.info(f"      quality flags: {', '.join(flags)}")

        # --- record both halves --------------------------------------------
        # `pair_id` links the two halves so later analysis can ask pair-level
        # questions, e.g. "did the detector get BOTH versions right, or did it
        # just label everything vulnerable?"
        pair_id = f"pair_{index + 1:03d}"
        shared = {
            "pair_id": pair_id,
            "cwe": cwe,
            "cwe_name": cwe_name,
            "context": context,
            "difficulty": difficulty,
            "generator_model": generator_model,
            "quality_flags": flags,
        }
        records.append({
            **shared,
            "label": config.LABEL_VULNERABLE,
            "code": vulnerable,
            "note": result.data["vulnerability_explanation"],
        })
        records.append({
            **shared,
            "label": config.LABEL_SAFE,
            "code": safe,
            "note": result.data["fix_summary"],
        })

    if not records:
        run.update_metadata(status="failed", error="No usable pairs generated.")
        raise SystemExit(
            "Generation produced no usable pairs. Check logs/"
            f"{run.run_id}.log -- the most common causes are the Ollama server "
            "not running and the model tag not being pulled."
        )

    # --- shuffle, then name -------------------------------------------------
    # Shuffling BEFORE assigning filenames is a leakage control. Without it,
    # sample_001/sample_002 would always be a vulnerable/safe pair, so the file
    # number and the directory order would both reveal the label to anyone --
    # or anything -- listing the directory.
    rng.shuffle(records)

    manifest: list[dict] = []
    for position, record in enumerate(records, start=1):
        # Neutral, zero-padded ids. Three digits so a 500-sample run still
        # sorts correctly in a file listing.
        #
        # These filenames stay deliberately uninformative -- see the note in
        # runstore.py. Everything else in a run is descriptively named; sample
        # files cannot be, because detectors read them.
        sample_id = f"sample_{position:03d}"
        filename = f"{sample_id}.py"

        # .pop() removes the source as it is written to disk, so the manifest
        # holds metadata only rather than duplicating every program.
        (run.samples_dir / filename).write_text(
            record.pop("code"), encoding="utf-8"
        )
        manifest.append({"id": sample_id, "file": filename, **record})

    write_json(run.generation_manifest_file, manifest)

    vulnerable_count = sum(
        1 for e in manifest if e["label"] == config.LABEL_VULNERABLE
    )
    cwes_present = sorted({e["cwe"] for e in manifest})

    if rejected:
        write_json(
            run.generation_dir / "generation_rejected.json", rejected
        )

    run.update_metadata(
        status="generated",
        generator_model=generator_model,
        generated_samples=len(manifest),
        generated_pairs=len(manifest) // 2,
        rejected_pairs=len(rejected),
        cwes_present=cwes_present,
        tokens={"input": in_tokens, "output": out_tokens},
    )

    log.info(f"Wrote {len(manifest)} samples to {run.samples_dir}")
    log.info(
        f"Composition: {vulnerable_count} vulnerable / "
        f"{len(manifest) - vulnerable_count} safe, "
        f"{len(cwes_present)} CWE categories present"
    )
    if rejected:
        log.info(f"Rejected {len(rejected)} pair(s); see generation_rejected.json")
    log.info(f"Tokens: {in_tokens} in, {out_tokens} out")

    return {
        "run_id": run.run_id,
        "samples": len(manifest),
        "pairs": len(manifest) // 2,
        "rejected": len(rejected),
        "generator_model": generator_model,
        "cwes_present": cwes_present,
        "tokens": {"input": in_tokens, "output": out_tokens},
    }


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run generation with the settings in config.py, printing progress."""
    summary = generate_dataset(on_progress=print)
    print(f"\nRun id: {summary['run_id']}")


if __name__ == "__main__":
    main()
