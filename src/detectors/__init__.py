"""The detector arms -- the tools actually under test.

THREE PARADIGMS, ONE INTERFACE
    The point of the framework is to compare fundamentally different kinds of
    vulnerability detector under a single evaluation protocol:

        arm       module            tools
        --------  ----------------  ------------------------------------------
        static    static_tools.py   Bandit, Ruff, Semgrep
        ml        ml_models.py      pretrained: BiLSTM (Farasat & Posegga
                                    2024), LSTM (VUDENC 2022);
                                    trained here: TF-IDF + LogReg,
                                    embeddings + LogReg
        llm       llm_tools.py      any chat model, local or remote

    Every detector -- however it works internally -- returns the same result
    shape, so `evaluate.py` scores all of them with one code path and none of
    them gets a bespoke advantage.

THE COMMON RESULT SHAPE
    {
      "detector":      "Bandit",              human-readable, used as a heading
      "detector_key":  "static_bandit",       stable id, used in filenames
      "detector_type": "static",              which arm: static | ml | llm
      "predictions": [
        {"id": "sample_001",
         "predicted_label": "vulnerable" | "not_vulnerable",
         "predicted_cwe":   "CWE-89" | null,
         "confidence":      0.0 - 1.0,
         "reasoning":       "why"}
      ]
    }

    A detector that produces no verdict for a sample may simply omit it;
    evaluate.py treats a missing verdict as "not vulnerable", on the principle
    that silence is a miss rather than an exemption.
"""

from __future__ import annotations

import config

# Registry of every available detector.
#
# `key`      -- stable identifier; becomes part of the result filename.
# `name`     -- what the web interface and the report call it.
# `arm`      -- which paradigm it belongs to, for grouping in the UI.
# `needs`    -- runtime prerequisite, so the interface can grey out a tool that
#               cannot run here rather than failing halfway through a job.
# `blurb`    -- one line explaining what the tool actually does, shown as help
#               text in the Tools panel.
DETECTOR_REGISTRY: dict[str, dict] = {
    # --- Arm 1: static analysis --------------------------------------------
    "static_bandit": {
        "key": "static_bandit",
        "name": "Bandit",
        "arm": "static",
        "needs": "bandit",
        "blurb": (
            "The reference Python security linter. Walks the AST looking for "
            "known-bad calls and patterns; reports a CWE id with each finding."
        ),
    },
    "static_ruff": {
        "key": "static_ruff",
        "name": "Ruff",
        "arm": "static",
        "needs": "ruff",
        "blurb": (
            "An independent Rust reimplementation of Bandit's flake8-bandit "
            "ruleset. The least independent of the three static tools by "
            "design -- where it and Bandit differ, the shared ruleset is not "
            "the explanation, which separates a rule's limits from a tool's."
        ),
    },
    "static_semgrep": {
        "key": "static_semgrep",
        "name": "Semgrep",
        "arm": "static",
        "needs": "semgrep",
        "blurb": (
            "Pattern matching plus taint analysis: it tracks whether untrusted "
            "input actually reaches a dangerous sink, so it can tell a "
            "hardcoded path from a request parameter at the same call. That "
            "dataflow question is what the Medium and Hard difficulty levels "
            "are built around."
        ),
    },

    # --- Arm 2: machine learning -------------------------------------------
    # Both are published, frozen models that have never seen this benchmark's
    # data, so unlike the classifiers they replaced they are scored on exactly
    # the same protocol as the static and LLM arms -- asked, not trained.
    "ml_farasat": {
        "key": "ml_farasat",
        "name": "BiLSTM (Farasat & Posegga 2024)",
        "arm": "ml",
        "needs": "pretrained-weights",
        "blurb": (
            "A bidirectional LSTM over word2vec embeddings, trained at the "
            "Chair of IT Security in Passau on security-fix commits from "
            "thousands of real Python projects. Slides a context window across "
            "the file and scores each position, so it localises a weakness "
            "rather than labelling the file as a whole."
        ),
    },
    "ml_vudenc": {
        "key": "ml_vudenc",
        "name": "LSTM (VUDENC 2022)",
        "arm": "ml",
        "needs": "pretrained-weights",
        "blurb": (
            "The model the Passau BiLSTM builds on: same corpus, same "
            "preprocessing, a single LSTM layer instead of a deeper "
            "bidirectional stack. Running both measures what the architecture "
            "change bought, on data neither author chose."
        ),
    },

    # The trained pair. Unlike everything else in this registry these learn
    # from the run they are scored on, so their predictions are made out of
    # fold and their numbers are NOT on the same protocol as the rest. They
    # are kept because they measure a property of the benchmark -- how much
    # signal is learnable from the samples themselves -- which no pretrained
    # or unsupervised tool can report.
    "ml_tfidf": {
        "key": "ml_tfidf",
        "name": "TF-IDF + Logistic Regression",
        "arm": "ml",
        "needs": "sklearn",
        "blurb": (
            "Lexical baseline, trained on this run. Character and token "
            "n-grams over the raw source, classified by logistic regression "
            "with every prediction made out of fold. Scoring well here means "
            "the generator left surface giveaways; scoring at chance means "
            "the samples are lexically balanced."
        ),
    },
    "ml_embedding": {
        "key": "ml_embedding",
        "name": "Code embeddings + Logistic Regression",
        "arm": "ml",
        "needs": "embedding-api",
        "blurb": (
            "Semantic counterpart to the lexical baseline, also trained on "
            "this run. Each program becomes one dense vector from a neural "
            "code-embedding model, then goes to the same logistic regression "
            "-- so the only thing that differs between the two is the "
            "representation."
        ),
    },

    # --- Arm 3: large language models --------------------------------------
    "llm_detector": {
        "key": "llm_detector",
        "name": "LLM reviewer",
        "arm": "llm",
        "needs": "judgment-api",
        "blurb": (
            "A chat model asked to review the program directly. The model is "
            "chosen at run time; by default it is a different model from the "
            "judges, so no model grades a sample it helped certify."
        ),
    },
}

# Detectors registered in models.local.json, so a model added on this
# installation appears in the Tools panel and in the tool checkboxes.
#
# Reads the overlay JSON only. This module is imported by the web layer purely
# to render that panel, and the comment on detect.py's RUNNERS explains why it
# must not pull in tensorflow or scikit-learn -- so the ML *families* are read
# from the registry here, and the runner that actually loads weights is built
# in detect.py instead.
try:
    import model_registry as _registry

    for _key, _spec in _registry.ml_families().items():
        if _key in DETECTOR_REGISTRY:
            continue
        DETECTOR_REGISTRY[_key] = {
            "key": _key,
            "name": _spec["label"],
            "arm": "ml",
            "needs": "models-dir",
            "blurb": (
                "Registered on this installation, not shipped with the "
                "framework. A pretrained word2vec + recurrent family loaded "
                f"from models/{_spec['directory']} with "
                f"{_spec['dimensions']}-dimensional embeddings."
                + (f" {_spec['citation']}" if _spec.get("citation") else "")
            ),
            "user_registered": True,
        }
except Exception as _exc:                                    # noqa: BLE001
    print(f"WARNING: registered detectors not listed ({_exc})")

# Display names for the three arms, used as section headings in the UI.
ARM_NAMES = {
    "static": "Static analysis",
    "ml": "Machine learning",
    "llm": "Large language models",
}


def detectors_by_arm() -> dict[str, list[dict]]:
    """Group the registry by arm, preserving registry order within each arm.

    Used by the web interface to render the Tools panel as three labelled
    groups rather than one flat list.
    """
    grouped: dict[str, list[dict]] = {arm: [] for arm in ARM_NAMES}
    for entry in DETECTOR_REGISTRY.values():
        grouped.setdefault(entry["arm"], []).append(entry)
    return grouped


def build_result(detector_key: str, predictions: list[dict],
                 extra: dict | None = None) -> dict:
    """Assemble the common result envelope around a list of predictions.

    Centralised so every detector writes an identical shape; a typo in one
    module's key would otherwise make its results silently unscoreable.

    Args:
        detector_key: a key present in `DETECTOR_REGISTRY`.
        predictions: one entry per sample the detector judged.
        extra: optional detector-specific metadata (model name, timings,
            training details) recorded alongside but never used for scoring.

    Returns:
        The result dict, ready to be written to the run's detection directory.
    """
    meta = DETECTOR_REGISTRY[detector_key]
    result = {
        "detector": meta["name"],
        "detector_key": detector_key,
        "detector_type": meta["arm"],
        "predictions": predictions,
    }
    if extra:
        result.update(extra)
    return result


def make_prediction(sample_id: str, is_vulnerable: bool,
                    cwe: str | None = None, confidence: float = 0.5,
                    reasoning: str = "",
                    lines: list[int] | None = None) -> dict:
    """Build one prediction entry in the common shape.

    Args:
        sample_id: the sample this verdict is about.
        is_vulnerable: the detector's binary verdict.
        cwe: predicted category, or None. Ignored when `is_vulnerable` is
            False, since a program judged clean has no category.
        confidence: 0.0-1.0. Tools without a native confidence report a fixed
            value; see each module for what it uses and why.
        reasoning: short human-readable justification, recorded for inspection
            but never scored.
        lines: 1-based line numbers the detector is pointing at, if it can say.
            Scored against the manifest's `vulnerable_lines` where both exist;
            see "LOCALISATION" in evaluate.py. An empty list is meaningful and
            is not the same as being wrong: it records that the detector gave
            a verdict without a location, which evaluate.py reports separately
            so a tool is never marked as mislocating something it never
            claimed to locate.
    """
    return {
        "id": sample_id,
        "predicted_label": (
            config.LABEL_VULNERABLE if is_vulnerable else config.LABEL_SAFE
        ),
        "predicted_cwe": cwe if is_vulnerable else None,
        "confidence": round(float(confidence), 4),
        "reasoning": reasoning,
        # Only on a positive verdict: a file called clean has no site.
        "predicted_lines": (
            sorted({int(n) for n in lines if n})
            if is_vulnerable and lines else []
        ),
    }
