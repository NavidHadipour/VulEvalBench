"""Alternative input source -- analyse real code instead of generated samples.

The generated benchmark is synthetic by construction: every program was written
to order by a model. That is what makes it balanced and label-complete, and it
is also its central weakness -- a tool could do well on it and still be useless
on real software.

This module lets the same tool sets run over real Python: a public GitHub
repository, or a directory already on this machine. An ingested run flows
through exactly the same detection and comparison stages as a generated one.

THE THING TO UNDERSTAND ABOUT INGESTED RUNS -- there is no ground truth
    Nobody has labelled that repository. So every ingested sample is stored
    with `label: "unknown"`, and `evaluate.py` excludes unlabelled samples from
    accuracy, precision, recall, F1, FPR, FNR and MCC. Computing those against
    guessed labels would produce numbers that look authoritative and mean
    nothing.

    What still works, and is the point of doing this at all:
      - the tools all run and produce verdicts;
      - Cohen's kappa between every pair of tools is fully valid, because
        agreement does not need a reference answer;
      - the raw flag rate per tool is comparable -- a tool that flags 60% of a
        mature codebase is telling you something about itself.

    If labels are wanted, the judging stage can be run over an ingested run to
    have the judge models assign them. Those labels are model opinions, not
    ground truth, and anything derived from them must say so.

SAFETY
    Ingested code is never executed. Every tool in the framework reads source
    as text: the static analysers parse it, the ML models tokenise or embed it,
    the LLM detectors read it. Nothing imports or runs it, which is what makes
    it safe to point this at an arbitrary repository.

Run with:
    python src/ingest.py https://github.com/owner/repo
    python src/ingest.py C:\\path\\to\\local\\project
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import config
import runstore
from common import INGEST_DIR, ensure_dirs, get_logger, slugify, write_json

# Directories never worth analysing: dependencies, build output, caches and
# version-control internals. Including them would swamp a run with thousands of
# third-party files that say nothing about the project being studied.
SKIP_DIRECTORIES = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "build", "dist", "site-packages", ".eggs", ".idea", ".vscode",
}

# Files below this are usually `__init__.py` stubs with nothing to analyse;
# files above it are usually generated code or vendored bundles, and they blow
# past the embedding model's context window.
MIN_FILE_BYTES = 200
MAX_FILE_BYTES = 100_000

# Only a well-formed https GitHub URL is accepted. The URL is passed to
# `git clone` as an argument, so it is validated against a strict pattern
# rather than sanitised: without this, a string beginning with `--upload-pack=`
# would be read by git as an option rather than a URL and could run a command.
# `subprocess` is also called with a list and no shell, so there is no shell
# metacharacter exposure either way.
GITHUB_URL_RE = re.compile(
    r"^https://(?:www\.)?github\.com/"
    r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
    r"(?:\.git)?/?$"
)


def _is_skipped(path: Path, root: Path) -> bool:
    """True if any directory between `root` and `path` is on the skip list."""
    return any(
        part in SKIP_DIRECTORIES
        for part in path.relative_to(root).parts[:-1]
    )


def collect_python_files(root: Path, limit: int) -> list[Path]:
    """Find analysable Python files under `root`.

    Args:
        root: directory to search.
        limit: stop after this many files. Ingestion is bounded because the
            LLM and embedding arms make one API call per sample -- pointing
            this at a large repository without a cap would be slow and
            expensive.

    Returns:
        Up to `limit` paths, sorted for a deterministic selection so the same
        repository ingests the same files each time.
    """
    candidates: list[Path] = []

    for path in sorted(root.rglob("*.py")):
        if not path.is_file() or _is_skipped(path, root):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if not (MIN_FILE_BYTES <= size <= MAX_FILE_BYTES):
            continue
        candidates.append(path)
        if len(candidates) >= limit:
            break

    return candidates


def _clone_repository(url: str, destination: Path, log) -> Path:
    """Shallow-clone a GitHub repository.

    `--depth 1` fetches only the current commit: history is irrelevant here and
    a full clone of a large project would be far slower for no benefit.

    Raises:
        SystemExit: if the URL is not an accepted GitHub URL, or git fails.
    """
    if not GITHUB_URL_RE.match(url):
        raise SystemExit(
            f"Not an accepted GitHub URL: {url!r}\n"
            "Expected the form https://github.com/owner/repository"
        )

    if shutil.which("git") is None:
        raise SystemExit(
            "git is not installed or not on PATH, so a repository cannot be "
            "cloned. Install git, or ingest a local directory instead."
        )

    if destination.exists():
        # A previous ingest of the same repository. Removed so the clone is of
        # the current head rather than a stale copy silently reused.
        shutil.rmtree(destination, ignore_errors=True)

    log.info(f"  cloning {url}")
    process = subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", url, str(destination)],
        capture_output=True, text=True, timeout=600,
    )
    if process.returncode != 0:
        raise SystemExit(
            f"git clone failed: {process.stderr.strip()[:400]}"
        )
    return destination


def ingest(
    source: str,
    max_files: int = 40,
    label: str = "",
    on_progress=None,
) -> dict:
    """Ingest real Python code into a new run.

    Args:
        source: a GitHub https URL, or a path to a local directory or .py file.
        max_files: cap on how many files to take; see `collect_python_files`.
        label: free-text note stored in the run metadata.
        on_progress: callable taking one string; receives every log line.

    Returns:
        A summary dict including the new `run_id`.

    Raises:
        SystemExit: if the source cannot be read or contains no usable files.
    """
    ensure_dirs()

    is_github = source.startswith("http://") or source.startswith("https://")
    kind = "github" if is_github else "local"

    run = runstore.create_run(
        source=kind,
        label=label or source,
        config_snapshot={"source": source, "max_files": max_files},
    )
    log = get_logger(run.run_id, on_progress)

    log.info(f"Run {run.run_id}")
    log.info(f"Ingesting {kind} source: {source}")

    # --- resolve the source to a directory of files ------------------------
    if is_github:
        workdir = _clone_repository(
            source, INGEST_DIR / slugify(source), log
        )
        files = collect_python_files(workdir, max_files)
    else:
        path = Path(source).expanduser()
        if not path.exists():
            raise SystemExit(f"No such path: {path}")
        if path.is_file():
            if path.suffix != ".py":
                raise SystemExit(f"Not a Python file: {path}")
            files = [path]
            workdir = path.parent
        else:
            workdir = path
            files = collect_python_files(path, max_files)

    if not files:
        run.update_metadata(status="failed", error="No usable Python files.")
        raise SystemExit(
            f"No analysable Python files found under {workdir}.\n"
            f"Files must be between {MIN_FILE_BYTES} and {MAX_FILE_BYTES} "
            "bytes and outside dependency or cache directories."
        )

    log.info(f"  found {len(files)} analysable Python file(s)")

    # --- copy into the run, renaming neutrally -----------------------------
    # The originals stay where they are; the run holds its own immutable copy,
    # so a later re-analysis reads exactly the bytes that were scored.
    #
    # Sample filenames are neutral for the same reason as in a generated run:
    # a detector must not be able to infer anything from the path. The original
    # path is preserved in the manifest for display.
    manifest: list[dict] = []
    for position, path in enumerate(files, start=1):
        sample_id = f"sample_{position:03d}"
        filename = f"{sample_id}.py"

        try:
            # errors="replace" rather than failing: real repositories contain
            # files with declared encodings that do not match their bytes, and
            # one such file should not abort the ingest.
            code = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log.warning(f"    skipped {path.name}: {exc}")
            continue

        (run.samples_dir / filename).write_text(code, encoding="utf-8")

        try:
            origin = str(path.relative_to(workdir))
        except ValueError:
            origin = path.name

        manifest.append({
            "id": sample_id,
            "file": filename,
            # No ground truth exists for real code -- see the module docstring.
            "label": config.LABEL_UNKNOWN,
            "cwe": None,
            "cwe_name": "",
            "context": kind,
            "difficulty": "",
            "origin_path": origin,
            "origin_source": source,
            "lines": len(code.splitlines()),
        })

    write_json(run.generation_manifest_file, manifest)

    run.update_metadata(
        status="ingested",
        generator_model=f"{kind}:{source}",
        generated_samples=len(manifest),
        ingested_from=source,
        has_ground_truth=False,
    )

    log.info(f"Ingested {len(manifest)} file(s) into {run.samples_dir}")
    log.info(
        "These samples have no ground-truth labels, so accuracy-style metrics "
        "will be omitted. Tool agreement (Cohen's kappa) remains valid."
    )

    return {
        "run_id": run.run_id,
        "source": source,
        "kind": kind,
        "samples": len(manifest),
    }


def main() -> None:
    """Ingest a source named on the command line."""
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage:\n"
            "  python src/ingest.py https://github.com/owner/repo\n"
            "  python src/ingest.py C:\\path\\to\\project"
        )

    summary = ingest(sys.argv[1], on_progress=print)
    print(f"\nRun id: {summary['run_id']}")


if __name__ == "__main__":
    main()
