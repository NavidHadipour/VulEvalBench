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

    The same promise extends to re-analysis. Re-running the detectors over an
    existing run used to overwrite `detection_<tool>.json`, so comparing "this
    run with Bandit" against "this run with Bandit and an LLM" meant losing the
    first answer to get the second. Each detection pass is now snapshotted into
    `03_detection/rounds/` before the next one starts, so every pass over a run
    remains readable and the passes can be compared against each other.

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
            detection_<tool>.json       one file per detector, latest pass
            rounds/
                round_<timestamp>/      a previous pass, kept verbatim
                    detection_<tool>.json
                    round_metadata.json what ran, when, with which models
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
import shutil
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
# Snapshots of previous detection passes, one subdirectory per round.
DIR_ROUNDS = "rounds"
ROUND_METADATA = "round_metadata.json"
DIR_EVALUATION = "04_evaluation"


def is_valid_run_id(run_id: str) -> bool:
    """True if `run_id` is safe to join onto a filesystem path."""
    return bool(RUN_ID_RE.match(run_id or ""))


# A round id is a run-style timestamp, optionally with a "-2" disambiguator
# appended when two passes land in the same second. Validated for the same
# reason run ids are: the value reaches the filesystem from a URL.
ROUND_ID_RE = re.compile(r"^round_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(-\d+)?$")


def is_valid_round_id(round_id: str) -> bool:
    """True if `round_id` is safe to join onto a filesystem path."""
    return bool(ROUND_ID_RE.match(round_id or ""))


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
    def rounds_dir(self) -> Path:
        """Where previous detection passes are archived."""
        return self.detection_dir / DIR_ROUNDS

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
    def selected_manifest_file(self) -> Path:
        """The samples a human kept, when the run was reviewed by hand.

        Written after judging, so it is the LAST word on what the detectors
        see -- see `read_verified_manifest`. Absent on a run that was never
        reviewed, which is why every reader falls back rather than treating a
        missing file as an empty selection.

        Stored beside the judgement it followed rather than in the generation
        directory: it is a decision about the judged set, not about what the
        generator produced.
        """
        return self.judgement_dir / "selected_manifest.json"

    @property
    def balanced_manifest_file(self) -> Path:
        """The samples left after optional balancing -- stage 2c.

        Balancing runs AFTER the manual review and drops samples so the
        dataset is equal by label, by vulnerability class, or both. It is
        therefore the last word of all, above even the selection: a reviewer
        decides what is admissible, balancing decides what is comparable, and
        the second operates on the output of the first.

        Absent on a run that was not balanced, which is the default.
        """
        return self.judgement_dir / "balanced_manifest.json"

    @property
    def balance_report_file(self) -> Path:
        """Balance measured before the stage and after each enabled step.

        Kept separate from the manifest so the *history* survives: the
        manifest says what is left, this says what it looked like at each
        step and what each step cost. A dataset reported as balanced should
        be traceable to how it got that way.
        """
        return self.judgement_dir / "balance_report.json"

    @property
    def verified_manifest_file(self) -> Path:
        """Ground truth *after* judging -- the samples that survived."""
        return self.judgement_dir / "verified_manifest.json"

    def judgement_file(self, model: str) -> Path:
        """Path for one judge's verdicts. The model id is in the filename so
        every judge's output sits side by side and is told apart at a glance."""
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

    def write_selected_manifest(self, kept_ids) -> list[dict]:
        """Record which samples a human kept, reviewing after judging.

        Stores the kept ENTRIES, not just their ids, so the file is a manifest
        in the same shape as every other one and any reader can consume it
        without joining back to another file.

        Chosen from the GENERATION manifest, not the verified one, because the
        review can go both ways: a reviewer may restore a sample the judges
        rejected, and that sample exists only in the generation manifest.

        The order follows the generation manifest rather than the order the
        ids arrived in, so a selection made by clicking around a page still
        produces a stably ordered dataset.

        Args:
            kept_ids: the sample ids to keep. Ids not in the generation
                manifest are ignored rather than raising -- the caller is a
                web form, and a stale page should not be able to invent a
                sample.

        Returns:
            The kept entries, in manifest order.
        """
        wanted = set(kept_ids)
        kept = [e for e in self.read_generation_manifest() if e["id"] in wanted]
        write_json(self.selected_manifest_file, kept)
        return kept

    def read_verified_manifest(self) -> list[dict]:
        """What the detectors are scored on, in order of authority.

            1. the balanced manifest, if balancing ran. It operates on the
               output of the review, so it sits above it: the reviewer
               decides what is admissible, balancing decides what is
               comparable.
            2. the manual selection, if a human reviewed this run. The review
               happens after judging and may both drop admitted samples and
               restore rejected ones, so it overrides the judges rather than
               being filtered by them.
            3. the judges' verified manifest.
            4. the generation manifest, so an unjudged, unreviewed run can
               still be analysed -- useful for "before and after judging"
               comparisons.
        """
        if self.balanced_manifest_file.exists():
            return read_json(self.balanced_manifest_file)
        if self.selected_manifest_file.exists():
            return read_json(self.selected_manifest_file)
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

    # --- detection rounds ---------------------------------------------------
    # A "round" is one detection pass over this run. The newest pass lives in
    # the flat detection directory, where every reader already expects it;
    # older passes are copied into rounds/ before being superseded.
    #
    # Copied rather than moved, deliberately. A pass that re-runs only one tool
    # leaves the other tools' results in place, so the flat directory always
    # holds the complete current picture. Moving would strand the tools that
    # were not re-run.

    def record_detection_round(self, tool_keys: list[str], label: str = "",
                               extra: dict | None = None) -> str | None:
        """Archive the results of ONE detection pass as a round.

        Called after a pass finishes, with the detector keys that pass actually
        produced. Only those files are copied, which is what makes a round
        meaningful: re-analysing a run with Bandit alone and then with Bandit
        plus an LLM gives two rounds holding two different tool sets, rather
        than two identical snapshots of everything ever run.

        The flat detection directory is left alone and keeps accumulating the
        latest result per tool, so every existing reader -- and the evaluation
        of "the current state of this run" -- behaves exactly as before.

        Args:
            tool_keys: detector keys written by this pass.
            label: optional note describing the pass.
            extra: additional metadata to record, e.g. the models used.

        Returns:
            The round id, or None if the pass produced nothing to record.
        """
        paths = [
            self.detection_file(key) for key in tool_keys
            if self.detection_file(key).exists()
        ]
        if not paths:
            return None

        round_id = f"round_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        target = self.rounds_dir / round_id
        # Two passes inside the same second would otherwise merge into one
        # directory and silently misreport which tools belonged to which pass.
        suffix = 2
        while target.exists():
            target = self.rounds_dir / f"{round_id}-{suffix}"
            suffix += 1
        target.mkdir(parents=True, exist_ok=True)

        detectors: list[str] = []
        for path in paths:
            shutil.copy2(path, target / path.name)
            try:
                detectors.append(read_json(path).get("detector", path.stem))
            except (OSError, ValueError):
                detectors.append(path.stem)

        meta = self.metadata()
        write_json(target / ROUND_METADATA, {
            "round_id": target.name,
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "tool_keys": list(tool_keys),
            "detectors": detectors,
            "detector_files": [path.name for path in paths],
            # The sample set this pass was scored against. Recorded because a
            # re-judging pass changes it, and two rounds are only comparable
            # when it did not.
            "sample_count": len(self.read_verified_manifest()),
            "judged": self.verified_manifest_file.exists(),
            "generator_model": meta.get("generator_model"),
            **(extra or {}),
        })
        return target.name

    def list_detection_rounds(self) -> list[dict]:
        """Every archived round, newest first, with its metadata."""
        if not self.rounds_dir.exists():
            return []
        rounds = []
        for directory in sorted(self.rounds_dir.iterdir(), reverse=True):
            if not directory.is_dir():
                continue
            meta_path = directory / ROUND_METADATA
            meta = (read_json(meta_path) if meta_path.exists()
                    else {"round_id": directory.name})
            meta.setdefault("round_id", directory.name)
            rounds.append(meta)
        return rounds

    def detection_results_for_round(self, round_id: str) -> list[dict]:
        """Load one archived round's detector results.

        `round_id` is validated as a single path segment before use, so a
        crafted value from the browser cannot escape the run directory.
        """
        if not is_valid_round_id(round_id):
            return []
        directory = self.rounds_dir / round_id
        if not directory.is_dir():
            return []
        return [
            read_json(path)
            for path in sorted(directory.glob("detection_*.json"))
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
            # Set only on a perturbed run: the run its samples were rewritten
            # from. Surfaced so the clean/perturbed pair can always be found
            # again from either side.
            "derived_from": meta.get("derived_from", ""),
            "perturbations": meta.get("perturbations", []),
            # Generation of the samples: 1 for original, 2+ for a run whose
            # samples were processed again. Mirrors the _vN.0 id suffix.
            "version": meta.get("version", 1),
            "tag": meta.get("tag", ""),
        }


# ---------------------------------------------------------------------------
# Module-level operations
# ---------------------------------------------------------------------------

def create_run(source: str = "generated", label: str = "",
               config_snapshot: dict | None = None,
               version: int = 1, tag: str = "") -> Run:
    """Start a new run directory and write its initial metadata.

    Args:
        source: where this run's samples come from -- "generated", "github",
            "local", or "perturbed" for a semantics-preserving rewrite of
            another run. Recorded because evaluation treats ingested code
            differently: it has no ground truth, so accuracy is undefined for
            it. A perturbed run keeps the ground truth of the run it came
            from, since the rewrite does not change what the code does.
        label: optional free-text note from the user, shown in the run list.
            Purely descriptive; nothing branches on it.
        config_snapshot: the settings this run was launched with. Stored so a
            result can be reproduced later even after config.py has moved on --
            without this, an old run's numbers could not be explained.
        version: generation of this run's SAMPLES, not of the software. A run
            holding freshly produced samples is version 1 and gets no suffix.
            A run whose samples were processed again -- currently, perturbed
            from an earlier run -- is version 2, named `..._v2.0`, and
            processing that one again gives `..._v3.0`.

            The point is that a run id alone says whether its samples are
            original or derived, without opening the metadata. The timestamp
            still leads, so runs keep sorting chronologically.
        tag: short descriptor of what was done to the samples, inserted before
            the version -- "noisy" for a perturbed run. Together these give
            `run_2026-10-20_13-40-06_noisy_v2.0`, which says what it is and
            which generation it belongs to without opening anything.

    Returns:
        A `Run` handle whose directories already exist.
    """
    ensure_dirs()

    # Both parts are omitted when they carry no information: an ordinary
    # freshly generated run stays `run_<timestamp>` exactly as before.
    # Restricted to the characters RUN_ID_RE permits, since this reaches the
    # filesystem and a URL.
    tag_text = f"_{re.sub(r'[^a-z0-9]+', '-', tag.lower()).strip('-')}" if tag else ""
    version_text = "" if version <= 1 else f"_v{version}.0"
    suffix_text = f"{tag_text}{version_text}"
    run_id = f"{new_run_id()}{suffix_text}"

    # Guard against two runs starting inside the same second, which would
    # otherwise share a directory and interleave their samples.
    candidate = RUNS_DIR / run_id
    suffix = 1
    while candidate.exists():
        suffix += 1
        run_id = f"{new_run_id()}{suffix_text}-{suffix}"
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
        version=version,
        tag=tag,
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
