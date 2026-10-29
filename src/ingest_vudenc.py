"""Build a labelled, paired benchmark run from VUDENC's dataset.

WHY THIS EXISTS, AND WHY IT IS NOT ingest.py
    `ingest.py` points the tool sets at real Python -- a GitHub repository or
    a local directory -- and that is valuable, but nobody has labelled that
    code, so every sample is stored `label: "unknown"` and the
    accuracy-style metrics are undefined. The generated benchmark has the
    opposite problem: complete labels, but every program was written to order
    by a model, so a tool could do well on it and still be useless on real
    software.

    VUDENC's dataset closes that gap. Each record is a GitHub commit that
    fixed a vulnerability, and it carries the file as it stood *before* the
    fix together with the exact lines the fix replaced. That is real-world
    code with a label derived from a human developer's own security fix, and
    it is naturally PAIRED in the same minimal-diff way the generated
    benchmark is:

        vulnerable half = sourceWithComments            (the pre-fix file)
        safe half       = that, with each badpart
                          replaced by its goodpart      (the upstream fix)

    Only the flawed lines differ between the halves, so the pair isolates the
    flaw exactly as a generated pair does -- except that a person wrote both
    versions, and the fix is the one that actually shipped.

WHICH SOURCE FIELD, AND WHY IT MATTERS
    Records carry two source fields and only one of them is code.
    `source` has been through VUDENC's own preprocessing, which strips
    comments AND collapses every newline into a space -- their model reads
    200-character windows, so it never needed the text to be parseable. It
    is a single enormous line and no Python parser accepts it; measured on
    the full dataset, 0 of 1889 files parse. `sourceWithComments` is the
    original file. This module uses `sourceWithComments` throughout.

    This is also why VUDENC cannot be compared like-for-like against an AST
    based analyser on its own representation -- the representation discards
    the structure those tools need.

THE LABEL CAVEAT -- READ BEFORE QUOTING ANY NUMBER FROM A RUN BUILT HERE
    VUDENC labels at COMMIT level: every file touched by a security-fix
    commit inherits the vulnerability label, including files changed
    incidentally. A real example from the XSS class is a `setup.py` whose
    only change is a version bump from '0.3.22' to '0.3.23b0' -- the commit
    fixed an XSS somewhere, but this file is not the fix and is not
    vulnerable.

    So a label here means "a developer changed this file in a commit that
    fixed a <class> vulnerability", which is weaker than the generated
    benchmark's "this program was written to contain exactly this flaw".
    Two filters below cut the obvious noise (`_EXCLUDED_PATHS` and
    `_RELEVANT_TOKENS`), and they are heuristics, not ground truth.

    Two further limits that no filter removes:
      - the safe half is only known to have THAT flaw fixed. The file may
        contain other vulnerabilities, so `not_vulnerable` means "this site
        was repaired", not "this file is clean".
      - requiring both halves to parse as Python 3 excludes most of the
        dataset, which is largely Python 2 code from 2015-2020 repositories.
        That biases the sample toward more modern files.

    The judging stage exists for exactly this, and is the recommended next
    step: run it over a run built here and the judge models vote on whether
    each sample really carries its stated label, with the review page for a
    human pass over the verdicts.

LICENSING
    The samples are real files from hundreds of third-party repositories,
    each under its own licence. Runs live in `runs/`, which .gitignore
    excludes for this reason among others. Do not commit a run built here.

Run with:
    python src/fetch_vudenc_dataset.py        # once, downloads the data
    python src/ingest_vudenc.py               # all usable pairs
    python src/ingest_vudenc.py --max-pairs 60
    python src/ingest_vudenc.py --classes sql,xss --max-pairs 40
"""

from __future__ import annotations

import ast
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import runstore
from common import PROJECT_ROOT, ensure_dirs, get_logger, write_json

DATASET_DIR = PROJECT_ROOT / "datasets" / "vudenc"

# Same bounds as ingest.py: below the floor there is nothing to analyse, and
# above the ceiling the embedding arm's context window is exceeded.
MIN_FILE_BYTES = 200
MAX_FILE_BYTES = 100_000

# The class -> CWE mapping is config.VUDENC_CWE_MAP, not a copy of it: the
# ml_vudenc detector already scores identification against that mapping, and
# two different mappings for the same seven classes would eventually disagree.
# Five of the seven are in config.CWE_CATEGORIES; open redirect (CWE-601) and
# XSRF (CWE-352) are not, because the generator does not produce them -- which
# is precisely the kind of gap real-world data is useful for filling.
CWE_FOR_CLASS = config.VUDENC_CWE_MAP

# Files whose changes in a security-fix commit are almost never the fix
# itself. See THE LABEL CAVEAT above -- this is the cheap half of the noise
# filter, and it is matched on the path, case-insensitively.
_EXCLUDED_PATHS = (
    "setup.py", "version.py", "_version.py", "conftest.py", "__init__.py",
    "/test", "test_", "_test.py", "/tests/", "/doc", "/docs/", "/example",
    "/migrations/", "/locale/", "changelog", "changes.py", "conf.py",
)

# The expensive half: the replaced lines must mention something plausibly
# related to the vulnerability class. A version bump mentions none of these,
# a parameterised-query fix mentions several. Deliberately generous -- the
# aim is to drop obvious noise, not to pre-judge what the detectors should
# find, which would bias the benchmark toward flaws we already thought of.
_RELEVANT_TOKENS = {
    "sql": ("execute", "cursor", "select", "insert into", "update ", "delete",
            "where", "format", "%s", "text(", "query", "sql", "session.",
            "raw(", "executemany", "commit()"),
    "command_injection": ("subprocess", "os.system", "popen", "shell=true",
                          "check_output", "check_call", "call(", "commands.",
                          "exec", "spawn", "getoutput", "cmd"),
    "xss": ("escape", "markup", "safe", "render", "autoescape", "mark_safe",
            "html", "quote", "sanitiz", "bleach", "template", "innerhtml",
            "striptags", "|safe"),
    "remote_code_execution": ("eval", "exec", "pickle", "loads", "yaml.load",
                              "marshal", "__import__", "compile(", "literal_",
                              "subprocess", "input(", "deserial"),
    "path_disclosure": ("open(", "os.path", "join(", "abspath", "realpath",
                        "basename", "send_file", "static", "..", "normpath",
                        "safe_join", "filename", "path"),
    "open_redirect": ("redirect", "url", "next", "location", "urlparse",
                      "netloc", "is_safe_url", "allowed_hosts", "referer",
                      "host", "scheme"),
    "xsrf": ("csrf", "xsrf", "token", "referer", "origin", "samesite",
             "csrf_exempt", "state", "nonce", "verify"),
}


def _parses(text: str) -> bool:
    """True if the text is syntactically valid Python 3."""
    try:
        ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return False
    return True


def _lines_of(text: str, fragment: str) -> list[int]:
    """The 1-based line numbers that `fragment` occupies inside `text`.

    Used to turn a badpart -- the lines the fixing commit replaced -- into
    line-level ground truth. The caller has already checked that the fragment
    occurs exactly once, so there is no ambiguity about which occurrence this
    is. A fragment spanning several lines yields all of them.
    """
    start = text.find(fragment)
    if start == -1:
        return []
    first = text.count("\n", 0, start) + 1
    return list(range(first, first + fragment.count("\n") + 1))


def _excluded_path(path: str) -> bool:
    """True if this file's path marks it as unlikely to hold the fix."""
    lowered = path.lower()
    return any(marker in lowered for marker in _EXCLUDED_PATHS)


def _is_relevant(vulnerability: str, badparts: list[str],
                 goodparts: list[str]) -> bool:
    """True if the replaced lines mention anything related to the class."""
    blob = " ".join(badparts + goodparts).lower()
    return any(token in blob
               for token in _RELEVANT_TOKENS.get(vulnerability, ()))


def build_pairs(vulnerability: str, data: dict) -> tuple[list[dict], dict]:
    """Construct every usable vulnerable/safe pair for one class.

    Returns:
        (pairs, rejected) -- the pairs, and a count per rejection reason so
        the caller can report the yield honestly rather than only the
        successes.
    """
    pairs: list[dict] = []
    rejected: dict[str, int] = {}

    def drop(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for repo, commits in data.items():
        for sha, record in commits.items():
            for path, meta in record.get("files", {}).items():
                vulnerable = meta.get("sourceWithComments")
                if not vulnerable:
                    drop("no source")
                    continue
                if _excluded_path(path):
                    drop("path excluded (not where a fix lives)")
                    continue

                badparts, goodparts = [], []
                for change in meta.get("changes") or []:
                    badparts += change.get("badparts") or []
                    goodparts += change.get("goodparts") or []

                if not badparts:
                    drop("no badparts")
                    continue
                # An unequal count means the fix added or removed lines rather
                # than replacing them, so there is no 1:1 site to substitute.
                if len(badparts) != len(goodparts):
                    drop("badparts/goodparts not 1:1")
                    continue
                if not _is_relevant(vulnerability, badparts, goodparts):
                    drop("change unrelated to the class")
                    continue

                # Every site must appear exactly once, or the substitution
                # would be guessing which occurrence the developer fixed.
                sites = [(b.strip(), g.strip()) for b, g in
                         zip(badparts, goodparts) if b.strip()]
                if not sites:
                    drop("empty badparts")
                    continue
                if any(vulnerable.count(b) != 1 for b, _ in sites):
                    drop("site missing or ambiguous in source")
                    continue

                safe = vulnerable
                for bad, good in sites:
                    safe = safe.replace(bad, good, 1)
                if safe == vulnerable:
                    drop("fix changes nothing")
                    continue

                size = len(vulnerable.encode("utf-8"))
                if not (MIN_FILE_BYTES <= size <= MAX_FILE_BYTES):
                    drop("size out of range")
                    continue
                # Both halves, because a substitution that breaks the syntax
                # has not reproduced the upstream fix.
                if not (_parses(vulnerable) and _parses(safe)):
                    drop("not valid Python 3 (dataset is largely Python 2)")
                    continue

                # Line-level ground truth: the lines the developer's fix
                # replaced. A LOWER BOUND on what is wrong with the file, not
                # a complete list -- the commit fixed this flaw and may have
                # left others -- which is why evaluate.py scores localisation
                # recall against it but deliberately not precision.
                vulnerable_lines = sorted({
                    n for bad, _ in sites for n in _lines_of(vulnerable, bad)
                })

                message = (record.get("message") or record.get("msg")
                           or "").strip().splitlines()
                pairs.append({
                    "vulnerability": vulnerability,
                    "vulnerable": vulnerable,
                    "safe": safe,
                    "sites": len(sites),
                    "vulnerable_lines": vulnerable_lines,
                    "origin_repo": repo,
                    "origin_commit": sha,
                    "origin_path": path,
                    "origin_url": record.get("html_url") or record.get("url"),
                    "commit_message": (message[0][:300] if message else ""),
                    "keyword": record.get("keyword", ""),
                })

    return pairs, rejected


def ingest(classes: list[str] | None = None, max_pairs: int = 0,
           seed: int = 0, label: str = "", on_progress=None) -> dict:
    """Build a run from the VUDENC dataset.

    Args:
        classes: which vulnerability classes to include; all seven if empty.
        max_pairs: cap on total pairs, applied after a shuffle so the cap is
            a representative sample rather than the first N. 0 means no cap.
        seed: shuffle seed, so the same arguments give the same run.
        label: free-text note stored in the run metadata.
        on_progress: callable taking one string; receives every log line.

    Returns:
        A summary dict including the new `run_id`.

    Raises:
        SystemExit: if the dataset is absent or yields no usable pairs.
    """
    ensure_dirs()
    wanted = classes or list(CWE_FOR_CLASS)

    missing = [c for c in wanted if not (DATASET_DIR / f"plain_{c}").exists()]
    if missing:
        raise SystemExit(
            f"Dataset files not found for: {', '.join(missing)}\n"
            f"Expected under {DATASET_DIR}\n"
            "Run: python src/fetch_vudenc_dataset.py"
        )

    run = runstore.create_run(
        source="vudenc",
        label=label or f"VUDENC dataset ({', '.join(wanted)})",
        config_snapshot={
            "source": "vudenc", "classes": wanted, "max_pairs": max_pairs,
            "seed": seed, "dataset_dir": str(DATASET_DIR),
        },
    )
    log = get_logger(run.run_id, on_progress)

    log.info(f"Run {run.run_id}")
    log.info("Building labelled pairs from the VUDENC dataset")
    log.info("  vulnerable half = pre-fix file; safe half = upstream fix")

    all_pairs: list[dict] = []
    totals: dict[str, int] = {}
    for vulnerability in wanted:
        path = DATASET_DIR / f"plain_{vulnerability}"
        data = json.loads(path.read_text(encoding="utf-8"))
        pairs, rejected = build_pairs(vulnerability, data)
        all_pairs += pairs
        log.info(f"  {vulnerability:24} {len(pairs):4} usable pair(s)")
        for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
            log.info(f"      rejected {count:5}  {reason}")
        for reason, count in rejected.items():
            totals[reason] = totals.get(reason, 0) + count

    if not all_pairs:
        run.update_metadata(status="failed", error="No usable pairs.")
        raise SystemExit("No usable pairs could be built from the dataset.")

    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    if max_pairs and len(all_pairs) > max_pairs:
        log.info(f"  capping at {max_pairs} of {len(all_pairs)} pairs")
        all_pairs = all_pairs[:max_pairs]

    # --- expand pairs into samples -----------------------------------------
    records: list[dict] = []
    for index, pair in enumerate(all_pairs, start=1):
        cwe = CWE_FOR_CLASS[pair["vulnerability"]]
        shared = {
            "pair_id": f"pair_{index:03d}",
            "cwe": cwe,
            "cwe_name": config.CWE_NAMES.get(cwe, pair["vulnerability"]),
            "context": f"vudenc:{pair['vulnerability']}",
            "difficulty": "real-world",
            "generator_model": "vudenc-dataset",
            "quality_flags": [],
            "origin_repo": pair["origin_repo"],
            "origin_commit": pair["origin_commit"],
            "origin_path": pair["origin_path"],
            "origin_url": pair["origin_url"],
            "commit_message": pair["commit_message"],
            "fix_sites": pair["sites"],
        }
        records.append({
            **shared,
            "label": config.LABEL_VULNERABLE,
            "code": pair["vulnerable"],
            # Only on this half. The safe half has no vulnerable line to find,
            # and an empty list there is the correct ground truth.
            "vulnerable_lines": pair["vulnerable_lines"],
            "note": (
                f"File as it stood before the fixing commit. "
                f"{pair['sites']} line(s) were replaced by the fix."
            ),
        })
        records.append({
            **shared,
            "label": config.LABEL_SAFE,
            "code": pair["safe"],
            "note": (
                "Same file with the upstream fix applied. Only the flawed "
                "line(s) differ from the vulnerable half; other flaws in the "
                "file, if any, remain."
            ),
        })

    # Shuffled before naming, for the same reason as in generate.py: without
    # it sample_001/sample_002 would always be a vulnerable/safe pair, and the
    # filename would leak the label to anything listing the directory.
    rng.shuffle(records)

    manifest: list[dict] = []
    for position, record in enumerate(records, start=1):
        sample_id = f"sample_{position:03d}"
        filename = f"{sample_id}.py"
        code = record.pop("code")
        (run.samples_dir / filename).write_text(code, encoding="utf-8")
        manifest.append({
            "id": sample_id, "file": filename,
            "lines": len(code.splitlines()), **record,
        })

    write_json(run.generation_manifest_file, manifest)

    vulnerable_count = sum(
        1 for e in manifest if e["label"] == config.LABEL_VULNERABLE
    )
    run.update_metadata(
        status="generated",
        generator_model="vudenc-dataset",
        generated_samples=len(manifest),
        generated_pairs=len(manifest) // 2,
        ingested_from=str(DATASET_DIR),
        has_ground_truth=True,
        label_provenance="developer security-fix commits (commit-level)",
    )

    log.info(f"Built {len(manifest)} samples "
             f"({len(manifest) // 2} pairs, {vulnerable_count} vulnerable)")
    log.info("Labels come from developer fix commits, assigned at COMMIT "
             "level -- run the judging stage to vet them before quoting "
             "any metric. See this module's docstring.")

    return {
        "run_id": run.run_id,
        "source": "vudenc",
        "samples": len(manifest),
        "pairs": len(manifest) // 2,
        "classes": wanted,
        "rejected": totals,
    }


def main() -> None:
    """Usage: python src/ingest_vudenc.py [--classes a,b] [--max-pairs N]"""
    args = sys.argv[1:]
    classes: list[str] | None = None
    max_pairs = 0

    if "--classes" in args:
        raw = args[args.index("--classes") + 1]
        classes = [c.strip() for c in raw.split(",") if c.strip()]
        unknown = [c for c in classes if c not in CWE_FOR_CLASS]
        if unknown:
            raise SystemExit(
                f"Unknown class(es): {', '.join(unknown)}\n"
                f"Choose from: {', '.join(CWE_FOR_CLASS)}"
            )
    if "--max-pairs" in args:
        max_pairs = int(args[args.index("--max-pairs") + 1])

    summary = ingest(classes=classes, max_pairs=max_pairs, on_progress=print)
    print(f"\nRun id: {summary['run_id']}")
    print(f"{summary['samples']} samples / {summary['pairs']} pairs")


if __name__ == "__main__":
    main()
