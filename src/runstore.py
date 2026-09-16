"""Persistent, append-only storage for benchmark runs.

THE PROBLEM THIS SOLVES
    The first version of the pipeline wrote every dataset to a single
    `dataset/` directory and deleted the previous one first. A run that
    produced an interesting result was therefore destroyed by the next run, and
    the numbers in the README could not be traced back to the samples that
    produced them. That is not acceptable for a thesis, where a reported figure
    has to remain reproducible and inspectable months later.

    Every run now gets its own immutable directory under `runs/`, named by the
    moment it started. Nothing is ever overwritten or deleted.

DIRECTORY LAYOUT
    runs/run_2026-08-10_17-30-00/
        run_metadata.json               what was configured, and how it ended
        01_generation/
            samples/sample_001.py ...   the generated programs
            generation_manifest.json    intended label + CWE for each sample
        02_judgement/
            judgement_<model>.json      one file per judge, side-by-side
            verified_manifest.json      only the samples that survived judging
        03_detection/
            detection_<tool>.json       one file per detector
        04_evaluation/
            evaluation_report.json      metrics per detector
            comparison_report.json      the cross-tool comparison

    The numeric prefixes make the pipeline order obvious in a file listing,
    which matters when a supervisor opens the folder without reading the code.

A NOTE ON FILENAMES -- why samples are NOT descriptively named
    Everything in a run is named to be self-describing except the sample files
    themselves, which stay neutral (`sample_007.py`). This is deliberate and it
    is a correctness requirement, not an oversight.

    A detector is handed these files. If a filename were
    `sample_007_CWE-89_vulnerable.py`, then every tool -- and every LLM reading
    a path -- could score perfectly without analysing a single line of code,
    and the entire benchmark would measure nothing. The descriptive metadata
    lives in the manifest, which the web interface joins against the sample id
    for display, so a human browsing the site still sees rich labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from common import RUNS_DIR, ensure_dirs, new_run_id, read_json, slugify, write_json

# A run id must match this exactly before it is ever joined onto a path.
#
# The web app takes run ids straight from the URL, so without this check a
# request for `../../api-key-llm.txt` would read a credential file off disk --
# precisely the CWE-22 path traversal this project generates test cases for.
# Validating against a whitelist pattern is the fix; rejecting `..` alone is
# not, because encodings and alternate separators slip through such checks.
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,80}$")

# Subdirectory names, defined once so a rename cannot desynchronise writers
# from readers.
DIR_GENERATION = "01_generation"
DIR_JUDGEMENT = "02_judgement"
DIR_DETECTION = "03_detection"
DIR_EVALUATION = "04_evaluation"


def is_valid_run_id(run_id: str) -> bool:
    """True if `run_id` is safe to join onto a filesystem path."""
    return bool(RUN_ID_RE.match(run_id or ""))


@dataclass
class Run:
    """Handle to one run's directory, with typed accessors for its files.

    Every path in a run is derived here rather than assembled by callers, so
    the layout documented above is enforced in one place.
    """

    run_id: str
    root: Path

    # --- directories -------------------------------------------------------

    @property
    def generation_dir(self) -> Path:
        return self.root / DIR_GENERATION

    @property
    def samples_dir(self) -> Path:
        return self.generation_dir / "samples"

    @property
    def judgement_dir(self) -> Path:
        return self.root / DIR_JUDGEMENT

    @property
    def detection_dir(self) -> Path:
        return self.root / DIR_DETECTION

    @property
    def evaluation_dir(self) -> Path:
        return self.root / DIR_EVALUATION

    # --- individual files --------------------------------------------------

    @property
    def metadata_file(self) -> Path:
        return self.root / "run_metadata.json"

    @property
    def generation_manifest_file(self) -> Path:
        """Ground truth as *intended by the generator*, before judging."""
        return self.generation_dir / "generation_manifest.json"

    @property
    def verified_manifest_file(self) -> Path:
        """Ground truth *after* judging -- the samples that survived."""
        return self.judgement_dir / "verified_manifest.json"

    def judgement_file(self, model: str) -> Path:
        """Path for one judge's verdicts. The model id is in the filename so
        two judges' outputs sit side by side and are told apart at a glance."""
        return self.judgement_dir / f"judgement_{slugify(model)}.json"

    def detection_file(self, tool_key: str) -> Path:
        """Path for one detector's predictions, named after the tool."""
        return self.detection_dir / f"detection_{slugify(tool_key)}.json"

    @property
    def evaluation_report_file(self) -> Path:
        return self.evaluation_dir / "evaluation_report.json"

    @property
    def comparison_report_file(self) -> Path:
        return self.evaluation_dir / "comparison_report.json"

    @property
    def comparison_text_file(self) -> Path:
        """Human-readable twin of the comparison report, for pasting into a
        thesis draft or attaching to an email without a JSON viewer."""
        return self.evaluation_dir / "comparison_report.txt"

    # --- metadata ----------------------------------------------------------

    def metadata(self) -> dict:
        """Read this run's metadata, or an empty dict if it is missing."""
        if not self.metadata_file.exists():
            return {}
        return read_json(self.metadata_file)

    def update_metadata(self, **fields) -> dict:
        """Merge `fields` into the run metadata and write it back.

        Read-modify-write rather than overwrite, so a later stage recording its
        own results cannot erase what an earlier stage wrote.
        """
        data = self.metadata()
        data.update(fields)
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(self.metadata_file, data)
        return data

    # --- manifests ---------------------------------------------------------

    def read_generation_manifest(self) -> list[dict]:
        """All generated samples with their intended labels."""
        if not self.generation_manifest_file.exists():
            return []
        return read_json(self.generation_manifest_file)

    def read_verified_manifest(self) -> list[dict]:
        """The judged-correct samples, which are what detectors are scored on.

        Falls back to the generation manifest when judging has not been run,
        so the detection and evaluation stages still work on an unjudged run --
        useful when comparing "before and after judging" numbers.
        """
        if self.verified_manifest_file.exists():
            return read_json(self.verified_manifest_file)
        return self.read_generation_manifest()

    def read_sample_code(self, entry: dict) -> str:
        """Read one sample's Python source.

        `encoding` is explicit because Windows would otherwise fall back to the
        system code page and corrupt any non-ASCII character in the file.
        """
        return (self.samples_dir / entry["file"]).read_text(encoding="utf-8")

    # --- detector results --------------------------------------------------

    def detection_results(self) -> list[dict]:
        """Load every detector result file in this run, sorted by filename.

        Sorting gives a stable column order across reports, so two evaluations
        of the same run can be compared line by line.
        """
        if not self.detection_dir.exists():
            return []
        return [
            read_json(path)
            for path in sorted(self.detection_dir.glob("detection_*.json"))
        ]

    # Files in the judgement directory that match the `judgement_*.json` glob
    # but are NOT one judge's verdicts. Without excluding them, a caller
    # iterating "every judge file" also picks up the dropped-sample list and
    # the side-by-side comparison, neither of which has a `judge_model` key --
    # which is a KeyError the moment anything reads one.
    _NON_JUDGE_FILES = {"judgement_dropped.json", "judgement_comparison.json"}

    def judgement_results(self) -> list[dict]:
        """Load every judge's verdict file in this run, sorted by filename.

        Auxiliary files written into the same directory are filtered out both
        by name and by shape, so an older run or a future auxiliary file cannot
        reintroduce the problem.
        """
        if not self.judgement_dir.exists():
            return []

        records = []
        for path in sorted(self.judgement_dir.glob("judgement_*.json")):
            if path.name in self._NON_JUDGE_FILES:
                continue
            record = read_json(path)
            # Shape check as well as name check: a per-judge file always
            # carries the model it came from.
            if isinstance(record, dict) and "judge_model" in record:
                records.append(record)
        return records

    # --- summary for listings ---------------------------------------------

    def summary(self) -> dict:
        """A compact description of this run, for the "Previous runs" page.

        Reads only the metadata and counts files; it never loads sample source,
        so listing a hundred runs stays fast.
        """
        meta = self.metadata()
        generated = meta.get("generated_samples")
        if generated is None:
            generated = len(self.read_generation_manifest())

        verified = meta.get("verified_samples")
        if verified is None:
            verified = (
                len(read_json(self.verified_manifest_file))
                if self.verified_manifest_file.exists()
                else None
            )

        return {
            "run_id": self.run_id,
            "created_at": meta.get("created_at", ""),
            "source": meta.get("source", "generated"),
            "status": meta.get("status", "unknown"),
            "generator_model": meta.get("generator_model", ""),
            "judges": meta.get("judges", []),
            "generated_samples": generated,
            "verified_samples": verified,
            "detectors": len(list(self.detection_dir.glob("detection_*.json")))
            if self.detection_dir.exists() else 0,
            "has_evaluation": self.comparison_report_file.exists(),
            "label": meta.get("label", ""),
        }


# ---------------------------------------------------------------------------
# Module-level operations
# ---------------------------------------------------------------------------

def create_run(source: str = "generated", label: str = "",
               config_snapshot: dict | None = None) -> Run:
    """Start a new run directory and write its initial metadata.

    Args:
        source: where this run's samples come from -- "generated", "github" or
            "local". Recorded because evaluation treats ingested code
            differently: it has no ground truth, so accuracy is undefined for it.
        label: optional free-text note from the user, shown in the run list.
            Purely descriptive; nothing branches on it.
        config_snapshot: the settings this run was launched with. Stored so a
            result can be reproduced later even after config.py has moved on --
            without this, an old run's numbers could not be explained.

    Returns:
        A `Run` handle whose directories already exist.
    """
    ensure_dirs()

    run_id = new_run_id()

    # Guard against two runs starting inside the same second, which would
    # otherwise share a directory and interleave their samples.
    candidate = RUNS_DIR / run_id
    suffix = 1
    while candidate.exists():
        suffix += 1
        run_id = f"{new_run_id()}-{suffix}"
        candidate = RUNS_DIR / run_id

    run = Run(run_id=run_id, root=candidate)
    for directory in (
        run.root, run.generation_dir, run.samples_dir,
        run.judgement_dir, run.detection_dir, run.evaluation_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    run.update_metadata(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        source=source,
        label=label,
        status="created",
        config=config_snapshot or {},
    )
    return run


def load_run(run_id: str) -> Run:
    """Open an existing run by id.

    Raises:
        ValueError: if `run_id` is not a safe identifier. Raised *before* the
            path is built, so a traversal attempt never touches the filesystem.
        FileNotFoundError: if no such run exists.
    """
    if not is_valid_run_id(run_id):
        raise ValueError(f"Invalid run id: {run_id!r}")

    root = RUNS_DIR / run_id
    if not root.is_dir():
        raise FileNotFoundError(f"No such run: {run_id}")

    return Run(run_id=run_id, root=root)


def list_runs(newest_first: bool = True) -> list[Run]:
    """Return every run on disk.

    Run ids embed a big-endian timestamp, so sorting them as strings sorts them
    chronologically -- no need to read metadata just to order the list.
    """
    if not RUNS_DIR.exists():
        return []

    directories = [
        path for path in RUNS_DIR.iterdir()
        if path.is_dir() and is_valid_run_id(path.name)
    ]
    directories.sort(key=lambda p: p.name, reverse=newest_first)
    return [Run(run_id=path.name, root=path) for path in directories]


def latest_run() -> Run | None:
    """The most recent run, or None if none exist yet."""
    runs = list_runs()
    return runs[0] if runs else None
