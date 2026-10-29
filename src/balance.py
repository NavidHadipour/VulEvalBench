"""Stage 2c -- optional dataset balancing, by label and by vulnerability class.

WHY BALANCE AT ALL
    Every rate this benchmark reports is read against the composition of the
    dataset it was measured on, and two kinds of imbalance distort different
    numbers.

    Label imbalance distorts accuracy. On a set that is 70% vulnerable, a
    detector that answers "vulnerable" to everything scores 0.70 accuracy and
    has detected nothing. MCC and Youden's J are robust to this, which is why
    evaluate.py reports them, but accuracy is the number readers reach for
    first and it should not be quietly flattering.

    Class imbalance distorts the comparison *between* detectors. If SQL
    injection is half the dataset, the ranking is largely a ranking of
    SQL-injection performance wearing a general-purpose label. The per-CWE
    table in evaluate.py exposes this, but the headline row still averages
    over whatever mix happens to be present.

WHY IT IS OPTIONAL, AND WHY IT ONLY EVER DROPS
    Balancing is not free and is not always right. It throws away real,
    correctly-labelled samples, and on a small run it can throw away most of
    them -- so it is off by default, each step is chosen separately, and a
    report is printed before anything is dropped so the cost is visible in
    advance rather than discovered afterwards.

    It only ever removes. Up-sampling by duplication would put the same
    program in the dataset twice and let a detector be scored twice on one
    answer; synthesising replacements would mean the generator choosing what
    the benchmark contains, which is the thing judging exists to prevent.

WHERE IT SITS
    After the manual review (stage 2b) and before detection. That ordering is
    deliberate: the reviewer decides what is *admissible* and balancing
    decides what is *comparable*, and the second has to operate on the output
    of the first -- balancing a set a human then edits would leave it
    unbalanced again.

    It is therefore CHOSEN there too. The web interface asks for these two
    steps in the review panel, beside the kept set and its composition, not
    in the run configuration: how many samples balancing would cost depends
    entirely on what survives the review, so a choice made beforehand would
    be made blind. A run that skips the review is never balanced, because it
    was never asked.

    Output goes to `02_judgement/balanced_manifest.json`, which
    `read_verified_manifest()` places above the selection. Nothing is deleted:
    the generation, verified and selected manifests all stay intact, so
    "46 of 59" can still be decomposed into what the judges dropped, what a
    human dropped, and what balancing dropped.

PAIRS, AND THE HONEST COST
    The benchmark is paired: each vulnerable sample has a safe twin differing
    only by the flaw. A perfectly intact paired dataset is ALREADY balanced by
    label, so label imbalance only arises once something has broken pairs --
    judging rejecting one half, a reviewer dropping one half, or a source like
    the VUDENC importer whose filters admit halves unevenly.

    So balancing drops ORPHANS FIRST -- samples whose twin is already gone.
    That repairs pair structure while correcting the count, and is strictly
    better than breaking an intact pair to fix a count. Only when orphans run
    out does it break complete pairs, and it says so in the report when it
    does.

Run with:
    python src/balance.py <run_id>                  report only, drops nothing
    python src/balance.py <run_id> --labels
    python src/balance.py <run_id> --classes
    python src/balance.py <run_id> --labels --classes
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import runstore
from common import get_logger, write_json

# A label/class split closer than this to perfect is left alone rather than
# "corrected". Dropping a sample to turn 50.4% into 50.0% costs real data to
# buy a difference no metric can see.
#
# 0.02 is two percentage points of the total, so on a 354-sample run a gap of
# up to 7 samples is tolerated. Chosen to be smaller than the smallest
# difference that moves a third decimal place in any reported rate.
LABEL_TOLERANCE = 0.02


def _cell(entry: dict) -> tuple[str, str]:
    """The (class, label) cell a sample belongs to.

    Unlabelled samples -- ingested real code -- have no cell and are handled
    by the caller, which refuses to balance a run it cannot measure.
    """
    return (entry.get("cwe") or "(none)", entry.get("label") or "(none)")


def _pair_index(manifest: list[dict]) -> dict[str, list[dict]]:
    """Samples grouped by pair id. Entries without one get a group of their own."""
    groups: dict[str, list[dict]] = {}
    for entry in manifest:
        key = entry.get("pair_id") or f"__unpaired__{entry['id']}"
        groups.setdefault(key, []).append(entry)
    return groups


def _orphan_ids(manifest: list[dict]) -> set[str]:
    """Ids whose pair is incomplete, i.e. whose twin is already missing.

    These are dropped first: removing one corrects a count AND repairs the
    pair structure, where removing a sample from an intact pair corrects the
    count by breaking a pair. A sample with no pair_id at all counts as an
    orphan, which is correct for the unpaired sources.
    """
    orphans = set()
    for key, group in _pair_index(manifest).items():
        if key.startswith("__unpaired__") or len(group) < 2:
            orphans.update(entry["id"] for entry in group)
    return orphans


def report(manifest: list[dict]) -> dict:
    """Measure how balanced a manifest is. Pure; drops nothing."""
    total = len(manifest)
    labels = Counter(entry.get("label") or "(none)" for entry in manifest)
    classes = Counter(entry.get("cwe") or "(none)" for entry in manifest)
    cells = Counter(_cell(entry) for entry in manifest)

    vulnerable = labels.get(config.LABEL_VULNERABLE, 0)
    safe = labels.get(config.LABEL_SAFE, 0)
    unknown = labels.get(config.LABEL_UNKNOWN, 0)

    groups = _pair_index(manifest)
    complete = sum(1 for key, g in groups.items()
                   if not key.startswith("__unpaired__") and len(g) == 2)
    orphans = _orphan_ids(manifest)

    # max/min over the classes present. 1.0 is perfectly even; absent classes
    # are not counted, because a class with no samples is a coverage question
    # rather than a balance one and conflating them would report infinity.
    counts = sorted(classes.values())
    class_ratio = (counts[-1] / counts[0]) if counts and counts[0] else 0.0

    return {
        "total": total,
        "labels": dict(labels),
        "vulnerable": vulnerable,
        "safe": safe,
        "unknown": unknown,
        # Share of the labelled half that is vulnerable. 0.5 is balanced.
        "vulnerable_share": (vulnerable / (vulnerable + safe)
                             if (vulnerable + safe) else 0.0),
        "label_gap": abs(vulnerable - safe),
        "label_balanced": (
            abs(vulnerable - safe) <= LABEL_TOLERANCE * max(total, 1)
        ),
        "classes": dict(classes),
        "class_count": len(classes),
        "class_ratio": class_ratio,
        "class_balanced": len(counts) <= 1 or class_ratio <= 1.0 + 1e-9,
        "cells": {f"{cwe}|{label}": n for (cwe, label), n in cells.items()},
        "complete_pairs": complete,
        "orphans": len(orphans),
    }


def format_report(data: dict, title: str) -> list[str]:
    """The short human-readable form, for the log and the terminal."""
    lines = [f"  {title}"]
    lines.append(
        f"    {data['total']} sample(s): "
        f"{data['vulnerable']} vulnerable / {data['safe']} safe"
        + (f" / {data['unknown']} unlabelled" if data["unknown"] else "")
    )
    if data["vulnerable"] or data["safe"]:
        lines.append(
            f"    label split {data['vulnerable_share']:.1%} vulnerable"
            f"  (gap {data['label_gap']})"
            f"  {'balanced' if data['label_balanced'] else 'IMBALANCED'}"
        )
    if data["class_count"]:
        spread = ", ".join(
            f"{cwe} {n}" for cwe, n in sorted(
                data["classes"].items(), key=lambda kv: (-kv[1], kv[0])
            )
        )
        lines.append(
            f"    {data['class_count']} class(es), "
            f"max/min {data['class_ratio']:.2f}  "
            f"{'balanced' if data['class_balanced'] else 'IMBALANCED'}"
        )
        lines.append(f"      {spread}")
    lines.append(
        f"    {data['complete_pairs']} complete pair(s), "
        f"{data['orphans']} orphan(s)"
    )
    return lines


def _drop_to(candidates: list[dict], keep: int, orphans: set[str],
             rng: random.Random) -> list[dict]:
    """Choose `keep` of `candidates`, dropping orphans first.

    Within each tier the order is shuffled by a seeded RNG rather than taken
    from the manifest, because manifest order is generation order: taking the
    first N would bias the kept set towards whatever the generator produced
    early, and for the VUDENC importer towards whichever repository sorted
    first.
    """
    if keep >= len(candidates):
        return list(candidates)

    paired = [e for e in candidates if e["id"] not in orphans]
    unpaired = [e for e in candidates if e["id"] in orphans]
    rng.shuffle(paired)
    rng.shuffle(unpaired)

    # Keep complete-pair members in preference to orphans, which is the same
    # thing as dropping orphans first.
    ordered = paired + unpaired
    return ordered[:keep]


def _fill_class_pairwise(vulnerable: list[dict], safe: list[dict],
                         target: int, rng: random.Random) -> set[str]:
    """Pick `target` vulnerable and `target` safe samples from one class,
    taking complete pairs before single samples.

    Complete pairs are consumed first, each contributing one sample to both
    cells at once, so the kept set keeps as many twin relationships as the
    target allows. Only once pairs run out -- or once the target is odd
    relative to the pairs available -- are the remaining slots filled from
    whatever is left over.

    Returns the ids to keep, which is `2 * target` ids when the class has
    enough of both, and fewer only if it does not.
    """
    by_pair: dict[str, dict[str, dict]] = {}
    for entry in vulnerable + safe:
        key = entry.get("pair_id")
        if key:
            by_pair.setdefault(key, {})[entry["label"]] = entry

    complete = [
        group for group in by_pair.values()
        if config.LABEL_VULNERABLE in group and config.LABEL_SAFE in group
    ]
    rng.shuffle(complete)

    kept: set[str] = set()
    taken_v = taken_s = 0
    for group in complete:
        if taken_v >= target or taken_s >= target:
            break
        kept.add(group[config.LABEL_VULNERABLE]["id"])
        kept.add(group[config.LABEL_SAFE]["id"])
        taken_v += 1
        taken_s += 1

    # Top up each cell separately from what the pair pass did not take.
    for pool, taken in ((vulnerable, taken_v), (safe, taken_s)):
        remaining = [e for e in pool if e["id"] not in kept]
        rng.shuffle(remaining)
        kept |= {e["id"] for e in remaining[:max(0, target - taken)]}

    return kept


def balance_labels(manifest: list[dict],
                   rng: random.Random) -> tuple[list[dict], dict]:
    """Make the vulnerable and safe counts equal.

    Returns the kept manifest and a note describing what happened.
    """
    before = report(manifest)
    if before["label_balanced"]:
        return list(manifest), {
            "step": "labels", "applied": False,
            "reason": (
                f"already within tolerance: gap {before['label_gap']} of "
                f"{before['total']} (<= {LABEL_TOLERANCE:.0%})"
            ),
            "dropped": 0,
        }

    orphans = _orphan_ids(manifest)
    target = min(before["vulnerable"], before["safe"])

    vulnerable = [e for e in manifest
                  if e.get("label") == config.LABEL_VULNERABLE]
    safe = [e for e in manifest if e.get("label") == config.LABEL_SAFE]
    other = [e for e in manifest
             if e.get("label") not in (config.LABEL_VULNERABLE,
                                       config.LABEL_SAFE)]

    kept_ids = {e["id"] for e in _drop_to(vulnerable, target, orphans, rng)}
    kept_ids |= {e["id"] for e in _drop_to(safe, target, orphans, rng)}
    kept_ids |= {e["id"] for e in other}

    # Filtered against the original order so the manifest stays stably
    # ordered -- the shuffle above decides *which* samples survive, not what
    # order they are written in.
    kept = [e for e in manifest if e["id"] in kept_ids]
    return kept, {
        "step": "labels", "applied": True,
        "reason": f"equalised to {target} per label",
        "dropped": len(manifest) - len(kept),
    }


def balance_classes(manifest: list[dict], rng: random.Random,
                    keep_labels_even: bool) -> tuple[list[dict], dict]:
    """Make every vulnerability class the same size.

    Args:
        manifest: the samples to balance.
        rng: seeded, for a reproducible choice of which samples survive.
        keep_labels_even: when the label step also ran, equalise per
            (class, label) cell rather than per class. Balancing per class
            alone can drop more of one label than the other and undo the
            label step, so the two steps together target a grid: every class
            the same size AND even within itself. Honouring one step while
            silently breaking the other would be worse than refusing either.

    Returns the kept manifest and a note.
    """
    labelled = [e for e in manifest
                if e.get("label") in (config.LABEL_VULNERABLE,
                                      config.LABEL_SAFE)]
    other = [e for e in manifest if e not in labelled]

    if not labelled:
        return list(manifest), {
            "step": "classes", "applied": False,
            "reason": "no labelled samples to balance",
            "dropped": 0,
        }

    orphans = _orphan_ids(manifest)
    by_class: dict[str, list[dict]] = {}
    for entry in labelled:
        by_class.setdefault(entry.get("cwe") or "(none)", []).append(entry)

    if len(by_class) <= 1:
        return list(manifest), {
            "step": "classes", "applied": False,
            "reason": f"only {len(by_class)} class present; nothing to even out",
            "dropped": 0,
        }

    kept_ids: set[str] = {e["id"] for e in other}
    excluded: list[str] = []

    if keep_labels_even:
        # Grid target: the smallest (class, label) cell across all classes.
        cells: dict[tuple[str, str], list[dict]] = {}
        for entry in labelled:
            cells.setdefault(_cell(entry), []).append(entry)

        # A class missing one of the two labels entirely cannot be made even
        # within itself at any size above zero, so it is dropped as a whole
        # and named, rather than collapsing every other class to zero.
        usable = []
        for cwe in by_class:
            v = len(cells.get((cwe, config.LABEL_VULNERABLE), []))
            s = len(cells.get((cwe, config.LABEL_SAFE), []))
            if v and s:
                usable.append(cwe)
            else:
                excluded.append(cwe)

        if not usable:
            return list(manifest), {
                "step": "classes", "applied": False,
                "reason": "no class has both a vulnerable and a safe sample",
                "dropped": 0,
            }

        target = min(
            min(len(cells.get((cwe, config.LABEL_VULNERABLE), [])),
                len(cells.get((cwe, config.LABEL_SAFE), [])))
            for cwe in usable
        )
        # Filled PAIR-FIRST, not cell-by-cell.
        #
        # Choosing each cell independently hits the target but shreds the pair
        # structure: on the 177-pair VUDENC run it left 43 complete pairs and
        # 54 orphans, because the vulnerable sample kept for a class and the
        # safe sample kept for it were rarely twins. The safe twin IS the
        # control for its vulnerable half, so that is real information thrown
        # away for nothing -- taking whole pairs reaches exactly the same
        # counts with the pairs intact.
        for cwe in usable:
            kept_ids |= _fill_class_pairwise(
                cells.get((cwe, config.LABEL_VULNERABLE), []),
                cells.get((cwe, config.LABEL_SAFE), []),
                target, rng,
            )
        detail = f"equalised to {target} per class per label, pairs first"
    else:
        target = min(len(group) for group in by_class.values())
        for group in by_class.values():
            kept_ids |= {
                e["id"] for e in _drop_to(group, target, orphans, rng)
            }
        detail = f"equalised to {target} per class"

    kept = [e for e in manifest if e["id"] in kept_ids]
    note = {
        "step": "classes", "applied": True,
        "reason": detail,
        "dropped": len(manifest) - len(kept),
    }
    if excluded:
        note["excluded_classes"] = sorted(excluded)
        note["reason"] += (
            f"; dropped {len(excluded)} class(es) with only one label present"
        )
    return kept, note


def balance_run(run_id: str, by_label: bool = False, by_class: bool = False,
                seed: int | None = None, on_progress=None) -> dict:
    """Run stage 2c over one run.

    With neither step enabled this reports and writes nothing, which is the
    useful "how balanced is it?" query and the default.

    Args:
        run_id: the run to balance.
        by_label: equalise vulnerable and safe counts.
        by_class: equalise the vulnerability classes.
        seed: for the reproducible choice of which samples survive. Taken
            from the run's own config snapshot when not given, so re-running
            this on the same run twice keeps the same set.
        on_progress: callable taking one string; receives every log line.

    Returns:
        A summary including the report at each step.
    """
    run = runstore.load_run(run_id)
    log = get_logger(run_id, on_progress)
    manifest = run.read_verified_manifest()

    if seed is None:
        snapshot = (run.metadata().get("config") or {})
        seed = snapshot.get("seed")
    rng = random.Random(seed if seed is not None else 0)

    log.info("=== Stage 2c: balancing ===")
    before = report(manifest)
    for line in format_report(before, "Before balancing:"):
        log.info(line)

    if not (by_label or by_class):
        log.info("  no balancing step selected; nothing dropped")
        return {
            "run_id": run_id, "applied": False,
            "before": before, "after": before, "steps": [],
            "kept": len(manifest), "dropped": 0,
        }

    unlabelled = before["unknown"]
    if unlabelled and unlabelled == before["total"]:
        # Ingested code carries label "unknown" by design. There is no label
        # or class composition to even out, and silently writing an unchanged
        # balanced manifest would claim a guarantee that does not hold.
        log.warning(
            "  every sample is unlabelled, so there is nothing to balance. "
            "Skipping; the detectors will see the unbalanced set."
        )
        return {
            "run_id": run_id, "applied": False,
            "before": before, "after": before, "steps": [],
            "kept": len(manifest), "dropped": 0,
            "skipped_reason": "no ground truth",
        }

    steps = []
    current = manifest

    if by_label:
        current, note = balance_labels(current, rng)
        steps.append(note)
        log.info(f"  step 1, labels: {note['reason']}"
                 f" ({note['dropped']} dropped)")
        for line in format_report(report(current), "After the label step:"):
            log.info(line)

    if by_class:
        current, note = balance_classes(current, rng, keep_labels_even=by_label)
        steps.append(note)
        log.info(f"  step 2, classes: {note['reason']}"
                 f" ({note['dropped']} dropped)")
        if note.get("excluded_classes"):
            log.warning(
                "    classes dropped entirely for having only one label: "
                + ", ".join(note["excluded_classes"])
            )
        for line in format_report(report(current), "After the class step:"):
            log.info(line)

    after = report(current)
    dropped = len(manifest) - len(current)

    if not current:
        # Reachable on a small or lopsided run: the grid target can be zero.
        # Writing an empty manifest would make every later stage look broken
        # for a reason that belongs here.
        log.warning(
            "  balancing would leave NO samples, so it has been abandoned "
            "and nothing was written. The run keeps its unbalanced set; see "
            "the report above for which class forced this."
        )
        return {
            "run_id": run_id, "applied": False,
            "before": before, "after": before, "steps": steps,
            "kept": len(manifest), "dropped": 0,
            "skipped_reason": "would empty the dataset",
        }

    write_json(run.balanced_manifest_file, current)
    write_json(run.balance_report_file, {
        "run_id": run_id,
        "by_label": by_label,
        "by_class": by_class,
        "seed": seed,
        "tolerance": LABEL_TOLERANCE,
        "before": before,
        "steps": steps,
        "after": after,
        "kept": len(current),
        "dropped": dropped,
    })

    run.update_metadata(
        balanced=True,
        balanced_by_label=by_label,
        balanced_by_class=by_class,
        balanced_samples=len(current),
        balanced_dropped=dropped,
    )

    log.info(
        f"Balancing kept {len(current)} of {len(manifest)} sample(s); "
        f"{dropped} dropped."
    )
    log.info(
        "  The generation, verified and selected manifests are unchanged, so "
        "the drop is auditable."
    )
    return {
        "run_id": run_id, "applied": True,
        "before": before, "after": after, "steps": steps,
        "kept": len(current), "dropped": dropped,
    }


def main() -> None:
    """Usage: python src/balance.py <run_id> [--labels] [--classes]"""
    args = sys.argv[1:]
    if not args:
        latest = runstore.latest_run()
        if latest is None:
            raise SystemExit("No runs found.")
        run_id = latest.run_id
        print(f"No run id given; using the most recent: {run_id}")
        rest: list[str] = []
    else:
        run_id, rest = args[0], args[1:]

    summary = balance_run(
        run_id,
        by_label="--labels" in rest,
        by_class="--classes" in rest,
        on_progress=print,
    )
    if not (("--labels" in rest) or ("--classes" in rest)):
        print("\nNo step selected, so nothing was dropped. "
              "Add --labels and/or --classes to balance.")
    else:
        print(f"\nKept {summary['kept']}, dropped {summary['dropped']}.")


if __name__ == "__main__":
    main()
