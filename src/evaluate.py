"""Stage 4 of the pipeline -- the metrics engine and cross-tool comparison.

Scores every detector in a run against the verified ground truth, then ranks
the tool sets against one another.

DETECTOR-AGNOSTIC BY DESIGN
    This module knows nothing about how any tool works. It reads any file in a
    run's detection directory that carries a `detector` name and a list of
    `predictions`, and scores it. A new detector joins the comparison by
    writing that shape -- no change is needed here. That is what lets the
    framework claim to compare paradigms "under a common evaluation protocol":
    there is literally one scorer.

CONVENTION
    The positive class is `vulnerable`. So a false positive is a safe program
    flagged as vulnerable (a nuisance alert), and a false negative is a missed
    vulnerability (a security failure).

THE COMPARISON MATHEMATICS
    LOCALISATION -- only where the ground truth records it
        A sample may carry `vulnerable_lines`, the 1-based lines known to hold
        the flaw, and a detector may answer with `predicted_lines`. Where both
        exist, true positives are also scored on whether the detector pointed
        within config.LOCALISATION_TOLERANCE_LINES of a real one.

        This matters as soon as real code is scored. On a 40-line generated
        program built around one planted flaw, "is this file vulnerable" and
        "where is the flaw" are the same question. On a 150-line file from a
        real repository where two lines are wrong, they are not, and a
        file-level verdict alone cannot tell a tool that found the bug from
        one that flagged an unrelated line and was counted correct.

        Runs without line-level truth -- every generated run -- report
        `localisable: 0` and the block is skipped rather than scored as zero.

    Four layers, from most standard to most specific to this project:

    1. Confusion matrix and the usual rates -- accuracy, precision, recall,
       F1, FPR, FNR. Universally understood, and what the project brief asks
       for.

    2. Two single-number summaries that behave better than accuracy on an
       unbalanced set, which is what judging tends to leave behind:
         - Matthews Correlation Coefficient (MCC), a correlation between
           prediction and truth over the whole confusion matrix. Ranges -1 to
           +1; 0 is chance. It is the honest headline number, because unlike
           accuracy it cannot be inflated by always answering with the
           majority class.
         - Youden's J = recall + specificity - 1. Ranges -1 to +1, 0 is
           chance. Reads as "how much better than guessing", balancing the two
           error types symmetrically.

    3. NO SINGLE COMBINED SCORE.

       An earlier version ranked detectors by a weighted sum of F1, (1-FPR)
       and CWE accuracy. That has been removed deliberately. Collapsing three
       different questions into one number hides which of them a tool actually
       failed, and the weights were an editorial choice that silently decided
       the ordering: a reader could not tell whether a tool led because it
       detected better or because the weighting happened to suit it.

       Every detector is now reported with the standard confusion matrix and
       the measures derived from it, which is what the vulnerability-detection
       literature reports and what a reader can compare against other work
       without first learning this project's private formula.

       The summary table is ordered by F1 so that it is stable and readable.
       That ordering is a presentation convenience, NOT a score: F1 ignores
       true negatives entirely, so a tool leading on F1 may still be the one
       crying wolf most often. Read the full row, not the position.

    4. Per-round metrics. Re-analysing a run archives the previous detection
       pass rather than discarding it, so one run can hold several passes --
       "with Bandit alone" and "with Bandit plus an LLM", say. Each archived
       round is scored with the same function as the current one, against the
       same manifest, so the rows are directly comparable.

       Comparability has one precondition, and it is checked rather than
       assumed: a round scored before a re-judging pass was measured on a
       different admitted sample set. Those rounds are flagged, because
       comparing them against the current numbers would be comparing two
       different datasets.

    5. Pairwise agreement between tools, as Cohen's kappa. This is the only
       measure here that does not need ground truth, which makes it the one
       that still works on ingested GitHub code. Two tools that agree strongly
       are largely redundant; two that disagree are finding different things,
       which is the more interesting result for a comparison study.

Run with:
    python src/evaluate.py <run_id>
"""

from __future__ import annotations

import math
import sys

import config
import runstore
from common import get_logger, read_json, write_json

LINE = "-" * 78


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def safe_div(numerator: float, denominator: float) -> float:
    """Divide, returning 0.0 rather than raising when the denominator is zero.

    Zero denominators are normal here, not bugs: a tool that flags nothing has
    `tp + fp == 0`, which leaves precision undefined. Reporting 0.0 keeps the
    table printable on small or degenerate runs.
    """
    return numerator / denominator if denominator else 0.0


def score(manifest: list[dict], predictions: list[dict]) -> dict:
    """Compute the full metric block for one detector.

    Args:
        manifest: ground truth, one entry per sample.
        predictions: that detector's verdicts.

    Returns:
        A dict with the confusion-matrix counts and every metric described in
        the module docstring.
    """
    tolerance = config.LOCALISATION_TOLERANCE_LINES

    truth = {entry["id"]: entry for entry in manifest}
    # Indexed by id so lookup does not depend on a detector having preserved
    # the manifest's ordering.
    by_id = {pred["id"]: pred for pred in predictions}

    tp = fp = tn = fn = 0
    cwe_correct = 0
    # LOCALISATION -- see the block comment below the loop.
    localisable = located = localised = 0

    for sample_id, entry in truth.items():
        # Samples with no ground truth (ingested code) cannot be scored for
        # accuracy; they are handled by the agreement analysis instead.
        if entry.get("label") == config.LABEL_UNKNOWN:
            continue

        pred = by_id.get(sample_id)

        # A detector that returns nothing for a sample counts as having said
        # "not vulnerable". Silence is a miss, not an exemption -- otherwise a
        # tool could improve its score by declining to answer on hard cases.
        # This also matters for the static analysers, which report findings
        # rather than per-file verdicts: no finding legitimately means "I
        # consider this file clean".
        predicted = pred["predicted_label"] if pred else config.LABEL_SAFE
        actual = entry["label"]

        if actual == config.LABEL_VULNERABLE:
            if predicted == config.LABEL_VULNERABLE:
                tp += 1
                # CWE identification is scored ONLY on true positives. Asking
                # whether a missed vulnerability was correctly categorised is
                # meaningless, and scoring the category on a false positive
                # would reward a lucky guess about a flaw that is not there.
                if pred and pred.get("predicted_cwe") == entry["cwe"]:
                    cwe_correct += 1

                # Scored only on true positives, for the same reason as the
                # CWE: there is no site to find in a sample the detector
                # missed, and crediting a position inside a false positive
                # would reward pointing precisely at a flaw that is not there.
                truth_lines = entry.get("vulnerable_lines") or []
                if truth_lines:
                    localisable += 1
                    predicted_lines = (pred.get("predicted_lines") or []
                                       if pred else [])
                    if predicted_lines:
                        located += 1
                        if any(abs(p - t) <= tolerance
                               for p in predicted_lines
                               for t in truth_lines):
                            localised += 1
            else:
                fn += 1
        else:
            if predicted == config.LABEL_VULNERABLE:
                fp += 1
            else:
                tn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    cwe_accuracy = safe_div(cwe_correct, tp)

    # LOCALISATION -- did the detector point at the right place?
    #
    # Three counts rather than one ratio, because a single number cannot tell
    # the two ways of scoring zero apart:
    #
    #   localisable  true positives whose ground truth records the vulnerable
    #                lines. The denominator. Zero for a run whose samples have
    #                no line-level ground truth (every generated run today),
    #                which makes the whole block inert rather than wrong.
    #   located      ... of those, how many the detector named any line for.
    #   localised    ... of those, how many named a line within
    #                config.LOCALISATION_TOLERANCE_LINES of a real one.
    #
    # So `localisation_accuracy` of 0.0 with `located` 0 means the detector
    # never reports positions -- true of the LLM arm as currently prompted and
    # of the ML arm, which answers per file. The same 0.0 with `located` high
    # means it does report them and they are wrong. Those are opposite
    # findings and the write-up must not merge them.
    #
    # `localisation_precision` is deliberately NOT computed. It would need
    # every flawed line in each sample to be known, and the ground truth here
    # records the lines a fix commit touched -- which is a lower bound, not a
    # complete list. Reporting a precision against an incomplete key would
    # punish a tool for finding a real flaw nobody had labelled.
    localisation_accuracy = safe_div(localised, localisable)

    # Matthews Correlation Coefficient. The denominator is the square root of
    # the product of all four marginals; when any marginal is zero the
    # coefficient is conventionally defined as 0, which is what safe_div gives.
    mcc_denominator = math.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    mcc = safe_div(float(tp * tn - fp * fn), mcc_denominator)

    metrics = {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "scored_samples": tp + fp + tn + fn,
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,       # of everything flagged, how much was real
        "recall": recall,             # of everything real, how much was caught
        "specificity": specificity,   # of everything safe, how much was cleared
        "f1": safe_div(2 * precision * recall, precision + recall),
        "fpr": safe_div(fp, fp + tn),  # false alarm rate on safe code
        "fnr": safe_div(fn, fn + tp),  # miss rate on vulnerable code
        "mcc": mcc,
        "youden_j": recall + specificity - 1.0,
        "cwe_accuracy": cwe_accuracy,
        "cwe_correct": cwe_correct,
        "localisable": localisable,
        "localisation_reported": located,
        "localised": localised,
        "localisation_accuracy": localisation_accuracy,
    }
    return metrics


def per_cwe(manifest: list[dict], predictions: list[dict]) -> dict[str, dict]:
    """Break detection performance down by vulnerability category.

    This is where the interesting differences between paradigms appear:
    rule-based tools are strong on categories with a clear syntactic signature
    and weak on ones needing data-flow reasoning.

    Only vulnerable samples are counted -- a safe program has no category to
    detect. Returns, per CWE: how many exist, how many were flagged, and how
    many were additionally given the right category label.
    """
    by_id = {pred["id"]: pred for pred in predictions}
    stats: dict[str, dict] = {}

    for entry in manifest:
        if entry.get("label") != config.LABEL_VULNERABLE:
            continue

        # setdefault creates the row the first time a CWE is seen, so the
        # categories present are discovered from the data rather than assumed
        # from config -- the table stays correct when the CWE list changes.
        row = stats.setdefault(
            entry["cwe"], {"total": 0, "detected": 0, "named": 0}
        )
        row["total"] += 1

        pred = by_id.get(entry["id"])
        if pred and pred["predicted_label"] == config.LABEL_VULNERABLE:
            row["detected"] += 1
            if pred.get("predicted_cwe") == entry["cwe"]:
                row["named"] += 1

    return stats


# ---------------------------------------------------------------------------
# Pairwise agreement
# ---------------------------------------------------------------------------

def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Chance-corrected agreement between two detectors' verdicts.

    Raw percentage agreement is misleading when both tools mostly answer the
    same way: two detectors that call everything safe agree 100% of the time
    while sharing no actual judgement. Kappa subtracts the agreement expected
    from their individual answer rates alone.

        kappa = (p_observed - p_expected) / (1 - p_expected)

    Returns 1.0 for perfect agreement, 0.0 for chance, negative for systematic
    disagreement. Returns 1.0 when both label everything identically and no
    disagreement is possible, which is the conventional degenerate case.
    """
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0

    total = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / total

    # Expected agreement from the marginal rates of each label.
    expected = 0.0
    for label in (config.LABEL_VULNERABLE, config.LABEL_SAFE):
        rate_a = labels_a.count(label) / total
        rate_b = labels_b.count(label) / total
        expected += rate_a * rate_b

    if abs(1.0 - expected) < 1e-12:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def agreement_matrix(manifest: list[dict],
                     results: list[dict]) -> list[dict]:
    """Cohen's kappa for every pair of detectors.

    Ground truth is not used, so this analysis works on ingested code with no
    labels -- the one part of the evaluation that does.
    """
    sample_ids = [entry["id"] for entry in manifest]

    # Align every detector's verdicts to the same sample order, filling a
    # missing verdict with "not vulnerable" exactly as `score` does, so kappa
    # and the confusion matrix describe the same behaviour.
    aligned: dict[str, list[str]] = {}
    for result in results:
        by_id = {p["id"]: p["predicted_label"] for p in result["predictions"]}
        aligned[result["detector"]] = [
            by_id.get(sample_id, config.LABEL_SAFE) for sample_id in sample_ids
        ]

    names = list(aligned)
    pairs = []
    for index, name_a in enumerate(names):
        for name_b in names[index + 1:]:
            labels_a, labels_b = aligned[name_a], aligned[name_b]
            pairs.append({
                "detector_a": name_a,
                "detector_b": name_b,
                "kappa": cohens_kappa(labels_a, labels_b),
                "raw_agreement": safe_div(
                    sum(1 for a, b in zip(labels_a, labels_b) if a == b),
                    len(labels_a),
                ),
            })
    return pairs


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def score_round(manifest: list[dict], results: list[dict]) -> list[dict]:
    """Score one detection pass: the standard metrics for each detector in it.

    Shared by the current pass and every archived round, so a round can never
    be scored by a different code path than the one it is compared against.
    """
    rows = []
    for result in results:
        metrics = score(manifest, result["predictions"])
        rows.append({
            "name": result.get("detector", result.get("detector_key", "?")),
            "key": result.get("detector_key", ""),
            "arm": result.get("detector_type", ""),
            "tp": metrics["tp"], "fp": metrics["fp"],
            "fn": metrics["fn"], "tn": metrics["tn"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "fpr": metrics["fpr"],
            "fnr": metrics["fnr"],
            "mcc": metrics["mcc"],
            "youden_j": metrics["youden_j"],
            "cwe_accuracy": metrics["cwe_accuracy"],
        })
    return sorted(rows, key=lambda r: r["f1"], reverse=True)


def build_rounds(run, manifest: list[dict], current: list[dict]) -> list[dict]:
    """Score the current pass and every archived one, newest first.

    Args:
        run: the Run being evaluated.
        manifest: the verified manifest everything is scored against.
        current: the detector results of the live (unarchived) pass.

    Returns:
        One entry per pass. The first is always the current one, labelled as
        such so a reader can tell it from the archive.
    """
    sample_count = len(manifest)
    rounds = [{
        "round_id": "current",
        "is_current": True,
        "label": "latest pass",
        "archived_at": None,
        "sample_count": sample_count,
        "comparable": True,
        "detectors": score_round(manifest, current),
    }]

    for meta in run.list_detection_rounds():
        results = run.detection_results_for_round(meta["round_id"])
        if not results:
            continue
        # A round scored against a different number of samples was measured on
        # a different dataset -- almost always because the run was re-judged
        # between passes. Its numbers are kept but marked, because putting
        # them beside the current ones without a warning compares two
        # different things.
        archived_count = meta.get("sample_count")
        comparable = archived_count in (None, sample_count)
        rounds.append({
            "round_id": meta["round_id"],
            "is_current": False,
            "label": meta.get("label", ""),
            "archived_at": meta.get("archived_at"),
            "sample_count": archived_count,
            "comparable": comparable,
            "detectors": score_round(manifest, results),
        })
    return rounds


def build_baseline(run) -> dict | None:
    """The scores of the run this one was derived from, for a direct contrast.

    A perturbed run holds rewritten copies of another run's samples, so the
    interesting question is not "how did this score" but "how much did it move
    from the clean original". That needs the parent's numbers next to this
    run's, which nothing else in the report provides.

    The parent's STORED evaluation is read rather than recomputed. Its samples
    are different files -- that is the whole point of a perturbed run -- so
    scoring them against this run's manifest would be meaningless. Both sides
    were scored against their own manifest by the same code, and because the
    rewrite preserves labels the two manifests agree on every label, which is
    what makes the comparison valid.

    Returns:
        `{run_id, label, detectors: [...]}` using the same row shape as a
        round, or None when this run was not derived from another, the parent
        is missing, or the parent has not been evaluated.
    """
    parent_id = run.metadata().get("derived_from")
    if not parent_id or not runstore.is_valid_run_id(parent_id):
        return None

    try:
        parent = runstore.load_run(parent_id)
    except (SystemExit, OSError, ValueError):
        return None

    report_file = parent.evaluation_report_file
    if not report_file.exists():
        return None

    try:
        stored = read_json(report_file)
    except (OSError, ValueError):
        return None

    rows = stored.get("ranking") or []
    if not rows:
        return None

    # Older reports predate some columns. Fill them rather than letting the
    # template render "undefined" in a table a supervisor is reading.
    normalised = []
    for row in rows:
        normalised.append({
            "name": row.get("name", "?"),
            "arm": row.get("arm", ""),
            **{key: row.get(key, 0) for key in (
                "tp", "fp", "fn", "tn", "accuracy", "precision", "recall",
                "f1", "fpr", "fnr", "mcc", "youden_j", "cwe_accuracy",
            )},
        })

    return {
        "run_id": parent_id,
        "label": parent.metadata().get("label", ""),
        "perturbations": run.metadata().get("perturbations", []),
        "detectors": normalised,
    }


def build_report(run_id: str) -> dict:
    """Assemble the complete evaluation for one run as serialisable data.

    Keeping scoring separate from presentation means the terminal report, the
    JSON file and the browser can never disagree -- they render one structure.

    Returns:
        A dict with `run`, `dataset`, `detectors`, `ranking`, `agreement` and
        `samples` keys.

    Raises:
        SystemExit: if the run has no samples or no detector results.
    """
    run = runstore.load_run(run_id)
    manifest = run.read_verified_manifest()
    results = run.detection_results()

    if not manifest:
        raise SystemExit(f"Run {run_id} has no samples to score.")
    if not results:
        raise SystemExit(
            f"Run {run_id} has no detector results. Run 'python src/detect.py "
            f"{run_id}' first."
        )

    labelled = [
        e for e in manifest if e.get("label") != config.LABEL_UNKNOWN
    ]
    vulnerable = sum(
        1 for e in labelled if e["label"] == config.LABEL_VULNERABLE
    )
    meta = run.metadata()

    dataset = {
        "run_id": run_id,
        "source": meta.get("source", "generated"),
        "total_samples": len(manifest),
        "labelled_samples": len(labelled),
        "vulnerable": vulnerable,
        "safe": len(labelled) - vulnerable,
        "cwes": sorted({e["cwe"] for e in labelled if e.get("cwe")}),
        "contexts": sorted({e.get("context", "") for e in labelled} - {""}),
        "generator_model": meta.get("generator_model", "unknown"),
        "judges": meta.get("judges", []),
        "generated_samples": meta.get("generated_samples"),
        "dropped_by_judging": meta.get("dropped_samples"),
    }

    labelled_ids = {entry["id"] for entry in labelled}

    detectors = []
    for result in results:
        metrics = score(labelled, result["predictions"])

        # COVERAGE -- what fraction of the scored samples this detector
        # actually returned a verdict for.
        #
        # `score` deliberately treats a missing verdict as "not vulnerable",
        # because for a static analyser silence genuinely means "I consider
        # this file clean". But that same rule silently rewards a STALE result
        # file: re-judging a run changes which samples are admitted, and an old
        # detection file then covers only part of the new manifest while being
        # credited with a correct "safe" call on every sample it never saw.
        # One such file scored a false positive rate of 0.000 and ranked third
        # on 6 of 15 samples.
        #
        # Coverage makes that visible instead of plausible.
        covered = sum(
            1 for pred in result["predictions"] if pred["id"] in labelled_ids
        )
        coverage = safe_div(covered, len(labelled))

        # A detector may carry a second prediction set scored under the strict
        # exact-CWE policy (the static tools do). Scoring both means the
        # hierarchy rule -- which was added after it was seen to disadvantage a
        # tool -- can never be mistaken for the only number available.
        metrics_strict = None
        if result.get("predictions_strict"):
            metrics_strict = score(labelled, result["predictions_strict"])

        detectors.append({
            "coverage": coverage,
            "covered_samples": covered,
            "is_stale": coverage < 1.0,
            "metrics_strict": metrics_strict,
            "name": result["detector"],
            "key": result.get("detector_key", ""),
            "arm": result.get("detector_type", "unknown"),
            "model": result.get("model", ""),
            "runtime_seconds": result.get("runtime_seconds"),
            "metrics": metrics,
            "per_cwe": per_cwe(labelled, result["predictions"]),
            # Detector-specific extras, surfaced so the report explains a
            # result rather than only stating it.
            "notes": {
                key: result[key] for key in (
                    "findings_total", "findings_in_scope",
                    "scoped_to_benchmark_cwes", "cv_folds",
                    "evaluation_protocol", "embedding_dim", "tool_version",
                ) if key in result
            },
        })

    # Ordered by F1 purely so the summary table is stable and readable --
    # this is a sort order, not a score. Ties keep their input order, which is
    # alphabetical by filename, so the table is deterministic across runs.
    ranking = sorted(
        detectors, key=lambda d: d["metrics"]["f1"], reverse=True
    )

    # Per-sample detail, with every detector's verdict aligned to one row.
    indexed = [
        {p["id"]: p for p in result["predictions"]} for result in results
    ]
    samples = []
    for entry in manifest:
        verdicts = []
        for by_id in indexed:
            pred = by_id.get(entry["id"])
            if pred is None:
                verdicts.append(None)
            else:
                verdicts.append({
                    **pred,
                    # Precomputed so templates do not have to know how
                    # correctness is defined.
                    "correct": (
                        None if entry.get("label") == config.LABEL_UNKNOWN
                        else pred["predicted_label"] == entry["label"]
                    ),
                })
        samples.append({**entry, "verdicts": verdicts})

    return {
        "run": meta,
        "dataset": dataset,
        "detectors": detectors,
        "ranking": [
            {
                "rank": position,
                "name": d["name"] + (" [STALE]" if d.get("is_stale") else ""),
                "arm": d["arm"],
                "coverage": d["coverage"],
                "tp": d["metrics"]["tp"],
                "fp": d["metrics"]["fp"],
                "fn": d["metrics"]["fn"],
                "tn": d["metrics"]["tn"],
                "accuracy": d["metrics"]["accuracy"],
                "precision": d["metrics"]["precision"],
                "recall": d["metrics"]["recall"],
                "f1": d["metrics"]["f1"],
                "fpr": d["metrics"]["fpr"],
                "fnr": d["metrics"]["fnr"],
                "mcc": d["metrics"]["mcc"],
                "youden_j": d["metrics"]["youden_j"],
                "cwe_accuracy": d["metrics"]["cwe_accuracy"],
            }
            for position, d in enumerate(ranking, start=1)
        ],
        "rounds": build_rounds(run, manifest, results),
        # Present only on a derived run: the scores of the run it came from,
        # so the change is readable without opening two pages.
        "baseline": build_baseline(run),
        "agreement": agreement_matrix(manifest, results),
        "samples": samples,
    }


def evaluate_run(run_id: str, on_progress=None) -> dict:
    """Build the report, write it into the run directory, and return it."""
    run = runstore.load_run(run_id)
    log = get_logger(run_id, on_progress)

    log.info("")
    log.info("=== Stage 4: evaluation ===")

    report = build_report(run_id)

    write_json(run.evaluation_report_file, report)
    write_json(run.comparison_report_file, {
        "run_id": run_id,
        "dataset": report["dataset"],
        "ranking": report["ranking"],
        "agreement": report["agreement"],
    })
    run.comparison_text_file.write_text(
        render_report(report), encoding="utf-8"
    )

    run.update_metadata(status="evaluated")
    log.info(
        f"Scored {len(report['detectors'])} detector(s) over "
        f"{report['dataset']['labelled_samples']} labelled samples"
    )
    log.info(f"Wrote {run.comparison_report_file.name} and "
             f"{run.comparison_text_file.name}")

    return report


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------
# Plain ASCII on purpose: the Windows console defaults to the cp1252 code page
# and raises UnicodeEncodeError on box-drawing characters.

def render_report(report: dict) -> str:
    """Render the report as plain text, for the terminal and the .txt file."""
    out: list[str] = []
    dataset = report["dataset"]

    out.append(LINE)
    out.append(f"BENCHMARK RUN: {dataset['run_id']}")
    out.append(LINE)
    out.append(f"Source           : {dataset['source']}")
    out.append(
        f"Samples          : {dataset['labelled_samples']} labelled "
        f"({dataset['vulnerable']} vulnerable / {dataset['safe']} safe)"
    )
    # Only meaningful when judges actually ran. An ingested run has a
    # generated_samples count but was never judged, and printing "0 admitted"
    # for it would suggest every sample had been rejected.
    if dataset.get("judges") and dataset.get("generated_samples"):
        out.append(
            f"Judging          : {dataset['generated_samples']} generated, "
            f"{dataset.get('dropped_by_judging') or 0} dropped, "
            f"{dataset['labelled_samples']} admitted"
        )
    out.append(
        f"CWE categories   : {len(dataset['cwes'])}  "
        f"({', '.join(dataset['cwes']) if dataset['cwes'] else 'none'})"
    )
    out.append(f"Generator        : {dataset['generator_model']}")
    if dataset.get("judges"):
        out.append(f"Judges           : {', '.join(dataset['judges'])}")

    # An unlabelled run (ingested code) has nothing to be accurate against, so
    # every rate below is computed over zero samples and comes out as 0.000.
    # Said plainly here, because a column of zeroes reads like "this tool
    # scored nothing" rather than "there was nothing to score".
    unlabelled = dataset["labelled_samples"] == 0
    if unlabelled:
        out.append("")
        out.append("  NOTE: this run has no ground-truth labels, so accuracy,")
        out.append("        precision, recall, F1, FPR, FNR and MCC are NOT")
        out.append("        meaningful and are shown as 0.000 only because the")
        out.append("        formulas divide by zero. The pairwise agreement")
        out.append("        section below IS valid -- it needs no reference")
        out.append("        answer -- as is each tool's raw flag rate.")

    # --- per-detector blocks ------------------------------------------------
    for detector in report["detectors"]:
        m = detector["metrics"]
        out.append("")
        out.append(LINE)
        out.append(f"DETECTOR: {detector['name']}   [{detector['arm']} arm]")
        if detector.get("is_stale"):
            out.append(
                f"  *** STALE: only {detector['covered_samples']}/"
                f"{dataset['labelled_samples']} samples have a verdict "
                f"({detector['coverage']:.0%} coverage). The rest are counted "
                "as 'not vulnerable', so these metrics FLATTER this detector. "
                "Re-run detection for this run."
            )
        if unlabelled:
            flagged = sum(
                1 for sample in report["samples"]
                for verdict in [sample["verdicts"][
                    report["detectors"].index(detector)]]
                if verdict and verdict["predicted_label"] == config.LABEL_VULNERABLE
            )
            out.append(
                f"  Flagged {flagged}/{len(report['samples'])} files. "
                "No ground truth, so the rates below are not meaningful."
            )
        out.append(LINE)
        out.append("  Confusion matrix            Predicted")
        out.append("                        vulnerable      safe")
        out.append(f"    Actual vulnerable {m['tp']:>10} {m['fn']:>9}")
        out.append(f"    Actual safe       {m['fp']:>10} {m['tn']:>9}")
        out.append("")
        out.append(f"    Accuracy               {m['accuracy']:.3f}")
        out.append(f"    Precision              {m['precision']:.3f}")
        out.append(f"    Recall                 {m['recall']:.3f}")
        out.append(f"    F1-score               {m['f1']:.3f}")
        out.append(f"    False Positive Rate    {m['fpr']:.3f}")
        out.append(f"    False Negative Rate    {m['fnr']:.3f}")
        out.append(f"    MCC                    {m['mcc']:+.3f}")
        out.append(f"    Youden's J             {m['youden_j']:+.3f}")
        out.append(
            f"    CWE identification     {m['cwe_accuracy']:.3f}  "
            f"({m['cwe_correct']}/{m['tp']} correctly categorised)"
        )
        # Printed only for runs carrying line-level ground truth; a generated
        # run has none, and a row of zeros there would read as a failure
        # rather than as a question that was never asked.
        if m.get("localisable"):
            out.append(
                f"    Localisation           "
                f"{m['localisation_accuracy']:.3f}  "
                f"({m['localised']}/{m['localisable']} pointed within "
                f"{config.LOCALISATION_TOLERANCE_LINES} line(s))"
            )
            if not m["localisation_reported"]:
                out.append(
                    "      this detector reports no line numbers, so the "
                    "0.000 above means 'did not say', not 'said wrongly'"
                )
            elif m["localisation_reported"] < m["localisable"]:
                out.append(
                    f"      named a line for {m['localisation_reported']} of "
                    f"{m['localisable']}; the rest gave a verdict only"
                )

        # Side-by-side with the strict exact-CWE policy, where available.
        strict = detector.get("metrics_strict")
        if strict:
            same = (strict["tp"] == m["tp"] and strict["fp"] == m["fp"])
            out.append("")
            out.append(
                "    CWE scope policy       hierarchy (shown above) vs strict"
            )
            out.append(
                f"      strict exact-match   TP={strict['tp']} FP={strict['fp']} "
                f"TN={strict['tn']} FN={strict['fn']}  "
                f"F1={strict['f1']:.3f}  MCC={strict['mcc']:+.3f}"
            )
            if same:
                out.append(
                    "      (identical -- the hierarchy rule changed nothing "
                    "for this tool on this run)"
                )

        if detector["per_cwe"]:
            out.append("")
            out.append("  Per-CWE detection")
            out.append(f"    {'CWE':<10} {'name':<40} {'detected':>9} {'named':>7}")
            for cwe in sorted(detector["per_cwe"]):
                row = detector["per_cwe"][cwe]
                name = config.CWE_NAMES.get(cwe, "")[:39]
                out.append(
                    f"    {cwe:<10} {name:<40} "
                    f"{row['detected']}/{row['total']:<7} {row['named']}/{row['total']}"
                )

    # --- summary table ------------------------------------------------------
    # The standard confusion matrix and the measures derived from it, one row
    # per detector. No combined score: see the module docstring for why.
    out.append("")
    out.append(LINE)
    out.append("RESULTS -- all detectors, this round")
    out.append(LINE)
    out.append(
        f"  {'detector':<32}{'arm':<8}"
        f"{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}"
        f"{'Acc':>8}{'Prec':>7}{'Rec':>7}{'F1':>7}{'FPR':>7}{'MCC':>8}{'CWE':>7}"
    )
    for row in report["ranking"]:
        out.append(
            f"  {row['name'][:31]:<32}{row['arm']:<8}"
            f"{row['tp']:>4}{row['fp']:>4}{row['fn']:>4}{row['tn']:>4}"
            f"{row['accuracy']:>8.3f}{row['precision']:>7.3f}"
            f"{row['recall']:>7.3f}{row['f1']:>7.3f}{row['fpr']:>7.3f}"
            f"{row['mcc']:>+8.3f}{row['cwe_accuracy']:>7.3f}"
        )
    out.append("")
    out.append(
        "  Acc=accuracy  Prec=precision  Rec=recall  FPR=false positive rate"
    )
    out.append(
        "  MCC=Matthews correlation (-1..+1, 0=chance)  "
        "CWE=correct category among true positives"
    )
    out.append(
        "  Rows are ordered by F1 for readability only -- that is a sort "
        "order, not a score."
    )

    # --- baseline -----------------------------------------------------------
    # Only on a derived run, where the question is how far the scores moved.
    baseline = report.get("baseline")
    if baseline:
        current = {row["name"]: row for row in report["ranking"]}
        out.append("")
        out.append(LINE)
        out.append(f"CHANGE FROM {baseline['run_id']}")
        out.append(LINE)
        if baseline.get("perturbations"):
            out.append(f"  applied: {', '.join(baseline['perturbations'])}")
        out.append(
            f"  {'detector':<32}{'F1 before':>10}{'F1 now':>8}{'change':>8}"
            f"{'MCC before':>12}{'MCC now':>9}{'change':>8}"
        )
        for row in baseline["detectors"]:
            now = current.get(row["name"])
            if not now:
                continue
            out.append(
                f"  {row['name'][:31]:<32}"
                f"{row['f1']:>10.3f}{now['f1']:>8.3f}"
                f"{now['f1'] - row['f1']:>+8.3f}"
                f"{row['mcc']:>+12.3f}{now['mcc']:>+9.3f}"
                f"{now['mcc'] - row['mcc']:>+8.3f}"
            )

    # --- rounds -------------------------------------------------------------
    # Only worth printing when there is something to compare against.
    rounds = report.get("rounds", [])
    if len(rounds) > 1:
        out.append("")
        out.append(LINE)
        out.append(f"ROUND COMPARISON -- {len(rounds)} passes over this run")
        out.append(LINE)
        out.append(
            "  Each pass is scored against the same verified samples, so the "
            "rows are directly"
        )
        out.append("  comparable unless marked otherwise.")
        for entry in rounds:
            out.append("")
            heading = (
                "  current pass" if entry["is_current"]
                else f"  {entry['round_id']}"
            )
            if entry.get("archived_at"):
                heading += f"   archived {entry['archived_at'][:19].replace('T', ' ')}"
            out.append(heading)
            if entry.get("label") and not entry["is_current"]:
                out.append(f"      note: {entry['label']}")
            if not entry.get("comparable", True):
                out.append(
                    f"      [NOT COMPARABLE] scored on {entry['sample_count']} "
                    f"samples, this report uses {len(report['samples'])} -- "
                    "the run was re-judged in between."
                )
            out.append(
                f"      {'detector':<32}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}"
                f"{'Acc':>8}{'Prec':>7}{'Rec':>7}{'F1':>7}{'FPR':>7}{'MCC':>8}"
            )
            for row in entry["detectors"]:
                out.append(
                    f"      {row['name'][:31]:<32}"
                    f"{row['tp']:>4}{row['fp']:>4}{row['fn']:>4}{row['tn']:>4}"
                    f"{row['accuracy']:>8.3f}{row['precision']:>7.3f}"
                    f"{row['recall']:>7.3f}{row['f1']:>7.3f}"
                    f"{row['fpr']:>7.3f}{row['mcc']:>+8.3f}"
                )

    # --- agreement ----------------------------------------------------------
    if report["agreement"]:
        out.append("")
        out.append("  Pairwise agreement (Cohen's kappa; 1.0 identical, 0.0 chance)")
        for pair in report["agreement"]:
            out.append(
                f"    {pair['detector_a'][:28]:<30} vs "
                f"{pair['detector_b'][:28]:<30} "
                f"kappa={pair['kappa']:+.3f}  raw={pair['raw_agreement']:.2f}"
            )

    # --- per-sample detail --------------------------------------------------
    out.append("")
    out.append(LINE)
    out.append("PER-SAMPLE DETAIL")
    out.append(LINE)
    header = f"{'sample':<12}{'CWE':<10}{'diff':<8}{'truth':<7}"
    for detector in report["detectors"]:
        header += f"{detector['name'][:11]:<13}"
    out.append(header)

    for sample in report["samples"]:
        # Upper case for vulnerable, lower for safe: the eye picks out the
        # positive class without reading the word.
        truth = (
            "VULN" if sample.get("label") == config.LABEL_VULNERABLE
            else "safe" if sample.get("label") == config.LABEL_SAFE
            else "?"
        )
        # `or "-"` rather than a dict default: ingested samples carry the keys
        # with an explicit None value, so `.get(key, "-")` returns None and
        # format() then raises. This is the difference between "key absent" and
        # "key present but empty", and only the former is covered by a default.
        row = (
            f"{sample['id']:<12}{(sample.get('cwe') or '-'):<10}"
            f"{str(sample.get('difficulty') or '-')[:6]:<8}{truth:<7}"
        )
        for verdict in sample["verdicts"]:
            if verdict is None:
                cell = "-"
            else:
                guess = (
                    "VULN" if verdict["predicted_label"] == config.LABEL_VULNERABLE
                    else "safe"
                )
                mark = (
                    "" if verdict["correct"] is None
                    else " ok" if verdict["correct"] else " XX"
                )
                cell = f"{guess}{mark}"
            row += f"{cell:<13}"
        out.append(row)

    out.append("")
    out.append(LINE)
    out.append(
        f"Scored {len(report['detectors'])} detector(s) over "
        f"{dataset['labelled_samples']} labelled samples."
    )
    out.append(LINE)
    return "\n".join(out)


def main() -> None:
    """Evaluate a run named on the command line, or the most recent one."""
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        latest = runstore.latest_run()
        if latest is None:
            raise SystemExit("No runs found. Run 'python src/generate.py' first.")
        run_id = latest.run_id
        print(f"No run id given; using the most recent: {run_id}")

    report = evaluate_run(run_id)
    print(render_report(report))


if __name__ == "__main__":
    main()
