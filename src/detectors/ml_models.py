"""Arm 2 -- machine-learning detectors: four models in two families.

The arm deliberately holds two kinds of model side by side, because they
answer different questions and are measured under different protocols. Keeping
both is the point: the contrast is itself a result.

    FAMILY A -- PRETRAINED, FROZEN            (never sees this benchmark)
        ml_farasat    BiLSTM + word2vec. Farasat & Posegga, ACM CODASPY 2024.
        ml_vudenc     LSTM + word2vec.   Wartschinski et al., IST 144 (2022).

    FAMILY B -- TRAINED HERE, ON THIS RUN     (learns from the samples)
        ml_tfidf      lexical.  Token and character n-grams.
        ml_embedding  semantic. Neural code embeddings of each program.

WITHIN EACH FAMILY, EXACTLY ONE THING DIFFERS
    Family A shares a corpus, a tokenisation and seven per-vulnerability
    heads, and differs only in the network -- so any gap between its two
    members is attributable to the architecture. Family B shares a classifier
    (logistic regression), the same samples and the same folds, and differs
    only in the representation -- so any gap there is attributable to lexical
    versus semantic features.

    Adding a second variable in either case would make the comparison
    uninterpretable. This mirrors the paired-generation logic used for the
    dataset itself: change one thing at a time, or you cannot attribute the
    result.

THE TWO FAMILIES ARE NOT MEASURED THE SAME WAY, AND THAT MUST BE STATED
    This is the single most important caveat in this arm.

    Family A is asked a question exactly as Bandit or an LLM is: one question,
    one answer, no training, no knowledge of this dataset. Its numbers are
    directly comparable with the static and LLM arms.

    Family B learns from the same tens of samples it is scored on. Every one
    of its predictions is therefore made OUT OF FOLD -- the model that labels
    a sample was fitted without ever seeing it -- because scoring on training
    data would report memorisation as detection. That is the honest way to put
    a supervised learner next to unsupervised tools, but it remains a
    different protocol, and a write-up that prints both families in one
    ranking without saying so is misleading.

    Why keep Family B at all, once Family A removed the need for it? Because
    it measures something Family A cannot: how much signal is learnable from
    this dataset itself. A lexical model that scores well is evidence the
    generator leaves surface giveaways; one that scores at chance is evidence
    the samples are lexically balanced. That is a property of the BENCHMARK
    rather than of any tool, and it is worth reporting for its own sake.

HOW FAMILY A REACHES A FILE-LEVEL VERDICT
    Neither pretrained model classifies files. Both slide a ~200-character
    window across the source and score each window, because the papers' unit
    of prediction is a token in its context. `vudenc_pipeline` owns that.

    A sample is vulnerable if ANY window reaches the authors' own 0.5
    operating point, and the responding head names the CWE. That rule matches
    the static arm, where a file is vulnerable if the linter reports any
    in-scope finding. It was not tuned: choosing a pooling rule by which one
    maximises MCC against these labels would be fitting the test set. For the
    record, the alternatives were measured and none separated the classes
    either -- see docs/ARCHITECTURE.md.

WHAT FAMILY A CAN AND CANNOT BE ASKED
    Seven vulnerability types, five of which are benchmark categories
    (CWE-89, 79, 78, 94, 22). For the other eleven the models have no head at
    all and answer "safe" every time -- not because the code is clean but
    because nobody trained them on the question. Every pretrained result file
    records a `coverage` block naming which samples were in scope, and the
    write-up must quote it next to the score.

    Family B has no such limit: it learns whatever categories the run happens
    to contain. That is another reason the two are not interchangeable.

HONEST LIMITATIONS, BOTH FAMILIES
    Family A scores near chance here, flagging the safe twin about as often as
    the vulnerable one. That is a measurement, not a bug: the tokeniser port is
    verified identical to the authors' own, and on their real-world example
    files the models behave as published. The samples are short and synthetic
    rather than the large real files the models were fitted on; the paired
    design differs by a handful of tokens, which the same group's adversarial
    study found these models cannot survive; and the published word2vec
    vocabulary is small enough (3,576 tokens) that `flask`, `request` and
    `sqlite3` are out-of-vocabulary, leaving the taint source invisible.

    Family B is severely data-starved at the sizes this project produces --
    tens of samples -- so its numbers are poor and noisy. That is a real
    finding about dataset size, not a defect; it becomes meaningful at the
    few-hundred-sample scale the framework is built for. The CWE multiclass
    head in particular needs far more data than the binary head, and declines
    to predict when it has too little rather than guessing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion

import config
from common import MODELS_DIR
from detectors import build_result, make_prediction
from detectors.vudenc_pipeline import ModelUnavailable, PretrainedDetector
from providers import ProviderError, embed_texts


class InsufficientData(RuntimeError):
    """The dataset is too small or too one-sided to train on.

    Raised by the trained family only; the pretrained models do no fitting and
    cannot run short of data. A dedicated type so the runner can report "this
    arm could not run on this run" as a normal outcome rather than a crash --
    which is what happens when judging admits only vulnerable samples, or only
    three samples in total.
    """


# ---------------------------------------------------------------------------
# The two published model families
# ---------------------------------------------------------------------------
# Built once per process and reused: loading TensorFlow plus seven networks
# takes several seconds, and a run scores every sample against both families.

_FAMILIES = {
    "ml_farasat": {
        "detector_key": "ml_farasat",
        "label": "BiLSTM (Farasat & Posegga 2024)",
        "directory": config.VUDENC_FARASAT_DIR,
        "word2vec": "word2vec_withString10-200-300.model",
        "dimensions": 300,
        "citation": (
            "T. Farasat and J. Posegga, 'Machine Learning Techniques for "
            "Python Source Code Vulnerability Detection', ACM CODASPY 2024"
        ),
    },
    "ml_vudenc": {
        "detector_key": "ml_vudenc",
        "label": "LSTM (VUDENC, Wartschinski et al. 2022)",
        "directory": config.VUDENC_ORIGINAL_DIR,
        "word2vec": "word2vec_withString10-100-200.model",
        "dimensions": 200,
        "citation": (
            "L. Wartschinski et al., 'VUDENC: Vulnerability Detection with "
            "Deep Learning on a Natural Codebase for Python', Information and "
            "Software Technology 144 (2022)"
        ),
    },
}

# The original VUDENC artefacts are incomplete upstream, so the slot can be
# pointed at Farasat's instead. See VUDENC_USE_FARASAT_WEIGHTS in config.py
# for what that does and does not measure. The rewrite happens here, once, at
# import, so every reader of _FAMILIES sees the same thing -- a detector whose
# label and citation describe what it will actually load.
if config.VUDENC_USE_FARASAT_WEIGHTS:
    _FAMILIES["ml_vudenc"].update({
        "directory": config.VUDENC_FARASAT_DIR,
        "word2vec": _FAMILIES["ml_farasat"]["word2vec"],
        "dimensions": _FAMILIES["ml_farasat"]["dimensions"],
        "label": "BiLSTM (Farasat weights in the VUDENC slot)",
        "citation": _FAMILIES["ml_farasat"]["citation"],
        "substituted": True,
        "substitution_note": (
            "VUDENC's own word2vec vector array was never published, so this "
            "detector loads Farasat & Posegga's 300-dimensional embedding and "
            "BiLSTM weights instead. It therefore duplicates ml_farasat and "
            "measures nothing of Wartschinski et al. (2022). Do not report it "
            "as a VUDENC result."
        ),
    })

# Models registered by whoever installed this, from models.local.json. Merged
# after the substitution above so a user entry is never rewritten by it, and
# imported lazily-ish: model_registry imports config and nothing heavy, so
# this costs nothing at import time.
#
# A registry entry describes WHERE weights are, not a new architecture: it
# feeds this same word2vec -> recurrent -> per-window-score pipeline. See the
# note in model_registry.py about what a genuinely different model needs.
try:
    from model_registry import ml_families as _user_ml_families

    for _key, _spec in _user_ml_families().items():
        if _key in _FAMILIES:
            # A shipped key is not silently replaced -- that would make a
            # published number irreproducible from the same run directory.
            continue
        _FAMILIES[_key] = _spec
except Exception as _exc:                                    # noqa: BLE001
    # A broken overlay must not stop the shipped detectors from loading.
    print(f"WARNING: user ML model registry ignored ({_exc})")


def runner_for(detector_key: str):
    """A `run_*`-shaped callable for any family in _FAMILIES.

    The shipped families have named wrappers (`run_farasat`, `run_vudenc`)
    because detect.py's RUNNERS table is written out explicitly and a named
    function reads better there. A family registered in models.local.json has
    no such wrapper to write, so detect.py builds one through this -- without
    it a registered ML model would appear in every menu and then have nothing
    to execute.
    """
    def run(run_, manifest: list[dict], log) -> dict:
        return _run_family(detector_key, run_, manifest, log)

    run.__name__ = f"run_{detector_key}"
    return run


def user_family_keys() -> list[str]:
    """Detector keys that came from the user registry, not from config.py."""
    return [key for key, spec in _FAMILIES.items()
            if spec.get("user_registered")]


def family_labels() -> dict[str, str]:
    """Every pretrained family's detector key and display label.

    Exposed for model_registry.builtin_models("ml"), so it does not have to
    reach into the private _FAMILIES dict.
    """
    return {key: spec["label"] for key, spec in _FAMILIES.items()}


_CACHE: dict[str, PretrainedDetector] = {}


def _family(detector_key: str) -> PretrainedDetector:
    """Return the loaded model family for a detector key, building it once."""
    if detector_key not in _CACHE:
        spec = _FAMILIES[detector_key]
        _CACHE[detector_key] = PretrainedDetector(
            family=detector_key,
            model_dir=MODELS_DIR / spec["directory"],
            word2vec_file=spec["word2vec"],
            weight_pattern="model_{vulnerability}.h5",
            dimensions=spec["dimensions"],
        )
    return _CACHE[detector_key]


# Availability is cached per process. Every page of the web interface renders
# the sidebar, which asks every detector whether it can run; without this the
# unavailable family re-read and re-failed on its 8.6 MB word2vec pickle on
# each page load, for an answer that cannot change while the process lives.
_AVAILABILITY: dict[str, bool] = {}


def family_available(detector_key: str) -> bool:
    """True if this family's weights and embedding table are usable.

    Called by `detect.available_tools()` to grey the detector out rather than
    let a user start a run that cannot finish. Never raises: a missing file is
    an expected state on a fresh checkout, before `fetch_models.py` has run.

    The answer is cached, so dropping a newly obtained weight file into
    `models/` needs a restart of the web app to be noticed.
    """
    if detector_key not in _AVAILABILITY:
        try:
            _AVAILABILITY[detector_key] = _family(detector_key).available()
        except Exception:  # noqa: BLE001 - any failure means "cannot run here"
            _AVAILABILITY[detector_key] = False
    return _AVAILABILITY[detector_key]


# ---------------------------------------------------------------------------
# Scoring one run
# ---------------------------------------------------------------------------

def _covered(cwe: str | None) -> bool:
    """True if this sample's planted CWE is one the models have a head for."""
    return cwe in config.VUDENC_COVERED_CWES


def _run_family(detector_key: str, run, manifest: list[dict], log) -> dict:
    """Score every sample in `manifest` with one pretrained model family.

    Args:
        detector_key: `ml_farasat` or `ml_vudenc`.
        run: the Run whose samples are being scored.
        manifest: the verified manifest entries to judge.
        log: the run logger.

    Returns:
        A result in the common shape, with `coverage` and `model_details`
        recorded alongside the predictions.

    Raises:
        ModelUnavailable: if the family's artefacts are missing. Deliberately
            not caught here -- a model that never loaded has not found the
            files clean, and silently emitting "safe" for every sample would
            award it perfect specificity for doing nothing.
    """
    spec = _FAMILIES[detector_key]
    detector = _family(detector_key)

    log.info(f"  loading {spec['label']}")
    log.info(f"    {len(detector.lookup)} word2vec tokens, "
             f"{len(config.VUDENC_VULNERABILITIES)} per-vulnerability models")

    predictions: list[dict] = []
    scores_by_sample: dict[str, dict[str, float]] = {}
    in_coverage: list[str] = []
    out_of_coverage: list[str] = []

    for entry in manifest:
        sample_id = entry["id"]
        (in_coverage if _covered(entry.get("cwe")) else
         out_of_coverage).append(sample_id)

        source = run.read_sample_code(entry)
        scores = detector.scan(source)
        scores_by_sample[sample_id] = {k: round(v, 4) for k, v in scores.items()}

        if not scores:
            # Too short to fill a single context window. The model has not
            # looked at this file, so record that rather than inventing a
            # verdict; evaluate.py treats a missing verdict as a miss.
            predictions.append(make_prediction(
                sample_id, False, None, 0.0,
                f"{spec['label']}: file too short to form a context window",
            ))
            continue

        strongest = max(scores, key=scores.get)
        probability = scores[strongest]
        is_vulnerable = probability >= config.VUDENC_THRESHOLD

        if is_vulnerable:
            reasoning = (
                f"{spec['label']}: '{strongest}' head fired on at least one "
                f"context window (p={probability:.3f})"
            )
        else:
            reasoning = (
                f"{spec['label']}: no head reached {config.VUDENC_THRESHOLD} "
                f"on any window (strongest was '{strongest}' at "
                f"{probability:.3f})"
            )

        predictions.append(make_prediction(
            sample_id,
            is_vulnerable,
            config.VUDENC_CWE_MAP.get(strongest),
            probability,
            reasoning,
        ))

    flagged = sum(
        1 for p in predictions
        if p["predicted_label"] == config.LABEL_VULNERABLE
    )
    log.info(f"    flagged {flagged}/{len(predictions)} samples")
    log.info(f"    {len(in_coverage)}/{len(manifest)} samples fall in a "
             f"category these models were trained for")

    if spec.get("substituted"):
        # Loud, every run. The registry's display name for this slot still
        # says VUDENC, and a reader skimming a report would take the number
        # at face value.
        log.warning(f"    NOTE: {spec['substitution_note']}")

    result = build_result(detector_key, predictions, {
        "model_family": spec["label"],
        "citation": spec["citation"],
        "pretrained": True,
        # Carried into every result file when this slot is running on
        # borrowed artefacts, so the caveat travels with the numbers instead
        # of living only in config.py. Absent on an unsubstituted family.
        **({"substituted": True,
            "substitution_note": spec["substitution_note"]}
           if spec.get("substituted") else {}),
        "threshold": config.VUDENC_THRESHOLD,
        "pooling": "max over context windows (any window >= threshold)",
        # Recorded so a reader can see how much of the benchmark these models
        # were ever in a position to answer. Quoted next to the score in the
        # report; see the coverage note in this module's docstring.
        "coverage": {
            "covered_cwes": config.VUDENC_COVERED_CWES,
            "samples_in_coverage": len(in_coverage),
            "samples_out_of_coverage": len(out_of_coverage),
            "out_of_coverage_ids": out_of_coverage,
        },
        "window_scores": scores_by_sample,
    })

    # The registry supplies the name every report prints, and for this slot it
    # says "LSTM (VUDENC 2022)" -- which is no longer what ran. Overridden
    # here so a comparison table cannot present borrowed weights as the
    # original model.
    if spec.get("substituted"):
        result["detector"] = spec["label"]

    return result


# ---------------------------------------------------------------------------
# The two runner entry points
# ---------------------------------------------------------------------------

def run_farasat(run, manifest: list[dict], log) -> dict:
    """Score a run with the Passau BiLSTM (Farasat & Posegga, CODASPY 2024).

    A bidirectional LSTM over a stacked recurrent network, trained on the
    VUDENC corpus with 300-dimensional word2vec embeddings. The published
    averages are precision 96.2%, recall 93.3%, F-score 94.7% on the authors'
    own held-out split -- measured per context window, not per file.
    """
    return _run_family("ml_farasat", run, manifest, log)


def run_vudenc(run, manifest: list[dict], log) -> dict:
    """Score a run with the original VUDENC LSTM (Wartschinski et al., 2022).

    The model the Passau work builds on: a single LSTM layer over
    200-dimensional word2vec embeddings, reporting 82-96% precision and 78-87%
    recall across the seven vulnerability types.

    Note that this detector is usually unavailable. Its word2vec vector array
    was saved by gensim to a `.npy` sidecar that the authors never published,
    and it cannot be reconstructed by retraining -- a fresh word2vec run
    produces a different vector space that the pretrained LSTM would read as
    noise. `fetch_models.py` explains how to enable it if you obtain the file.
    """
    return _run_family("ml_vudenc", run, manifest, log)


# =========================================================================
# FAMILY B -- trained on this run's own samples, scored out of fold
# =========================================================================
# Everything below learns from the dataset it is scored on. See the protocol
# note in the module docstring: these numbers are NOT directly comparable
# with the static, LLM or pretrained-ML arms, and must be labelled as such
# wherever they are printed beside them.


def _usable_folds(labels: list[int], requested: int) -> int:
    """Choose a fold count this dataset can actually support.

    A stratified split cannot make more folds than the rarest class has
    members: with three vulnerable samples, five folds is impossible. Rather
    than let scikit-learn raise, the count is reduced to what the data allows,
    and the caller records the number actually used.

    Returns:
        The usable fold count, at least 2.

    Raises:
        InsufficientData: if even two folds are impossible -- fewer than two
            members in a class, or only one class present.
    """
    counts = np.bincount(np.asarray(labels, dtype=int))
    present = counts[counts > 0]

    if len(present) < 2:
        raise InsufficientData(
            "Only one class is present, so a classifier cannot be trained. "
            "This usually means the judging stage dropped every sample of one "
            "label."
        )

    rarest = int(present.min())
    if rarest < 2:
        raise InsufficientData(
            f"The rarest class has only {rarest} sample(s); at least 2 are "
            "needed for cross-validation. Generate a larger dataset."
        )

    return max(2, min(requested, rarest))


def _binary_predictions(features, labels: list[int], folds: int):
    """Out-of-fold binary predictions and their probabilities.

    Args:
        features: the design matrix, however it was built.
        labels: 1 for vulnerable, 0 for safe.
        folds: number of stratified folds.

    Returns:
        `(predicted, probability)` arrays, aligned with the input rows.
    """
    # `class_weight="balanced"` because judging can leave the two classes
    # lopsided -- judges drop unconvincing vulnerable samples more often than
    # safe ones. Without it the model would learn to answer "safe" by default
    # and score well on accuracy while detecting nothing.
    # `max_iter` is raised well above the default because sparse n-gram
    # features converge slowly and the default 100 stops early with a warning.
    classifier = LogisticRegression(
        max_iter=2000, class_weight="balanced",
    )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.SEED or 0)

    predicted = cross_val_predict(
        classifier, features, labels, cv=splitter, method="predict"
    )
    probabilities = cross_val_predict(
        classifier, features, labels, cv=splitter, method="predict_proba"
    )
    # Column 1 is P(vulnerable); reported as the detector's confidence, which
    # for these models is a genuine calibrated-ish number rather than a
    # constant, unlike the static tools.
    return predicted, probabilities[:, 1]


def _cwe_predictions(features, labels: list[int], cwes: list[str],
                     folds: int) -> list[str | None]:
    """Out-of-fold CWE predictions for the vulnerable samples.

    Trained only on the vulnerable rows, since a safe program has no category.
    Returns None for every row when there is too little data to attempt it --
    which is the common case at small dataset sizes, and is reported honestly
    rather than filled with a guess.

    Returns:
        A list aligned with the full input, holding a CWE string for rows the
        model could predict and None elsewhere.
    """
    vulnerable_rows = [i for i, label in enumerate(labels) if label == 1]
    result: list[str | None] = [None] * len(labels)

    if len(vulnerable_rows) < 4:
        return result

    subset_cwes = [cwes[i] for i in vulnerable_rows]
    counts: dict[str, int] = {}
    for cwe in subset_cwes:
        counts[cwe] = counts.get(cwe, 0) + 1

    # Every class needs at least two members for a stratified split, and at
    # least two distinct classes must remain, or there is nothing to learn.
    if len(counts) < 2 or min(counts.values()) < 2:
        return result

    subset_folds = max(2, min(folds, min(counts.values())))

    # Slicing works for both a dense array and a scipy sparse matrix, so this
    # function does not need to know which representation it was handed.
    subset_features = features[vulnerable_rows]

    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    splitter = StratifiedKFold(
        n_splits=subset_folds, shuffle=True, random_state=config.SEED or 0
    )

    try:
        predicted = cross_val_predict(
            classifier, subset_features, subset_cwes, cv=splitter
        )
    except ValueError:
        # Stratification can still fail on pathological class distributions;
        # declining to predict a category is better than failing the arm.
        return result

    for row, cwe in zip(vulnerable_rows, predicted):
        result[row] = str(cwe)
    return result


def _assemble(detector_key: str, manifest: list[dict], features,
              labels: list[int], cwes: list[str], folds: int,
              log, extra: dict) -> dict:
    """Run both heads and package the results. Shared by both ML detectors."""
    used_folds = _usable_folds(labels, folds)
    log.info(
        f"    {len(labels)} samples, {sum(labels)} vulnerable, "
        f"{used_folds}-fold cross-validation"
    )

    predicted, probability = _binary_predictions(features, labels, used_folds)
    predicted_cwes = _cwe_predictions(features, labels, cwes, used_folds)

    predictions = []
    for index, entry in enumerate(manifest):
        is_vulnerable = bool(predicted[index])
        predictions.append(make_prediction(
            entry["id"],
            is_vulnerable,
            predicted_cwes[index] if is_vulnerable else None,
            float(probability[index]) if is_vulnerable
            else float(1.0 - probability[index]),
            "Out-of-fold prediction; this sample was excluded from the fold "
            "the model was fitted on.",
        ))

    return build_result(detector_key, predictions, {
        "cv_folds": used_folds,
        "training_samples": len(labels),
        "evaluation_protocol": "stratified k-fold cross_val_predict",
        **extra,
    })


def _dataset(run, manifest: list[dict]) -> tuple[list[str], list[int], list[str]]:
    """Read a run's samples into parallel source / label / CWE lists."""
    sources = [run.read_sample_code(entry) for entry in manifest]
    labels = [
        1 if entry["label"] == config.LABEL_VULNERABLE else 0
        for entry in manifest
    ]
    cwes = [entry.get("cwe", "unknown") for entry in manifest]
    return sources, labels, cwes


# ---------------------------------------------------------------------------
# Detector 1 -- lexical (TF-IDF)
# ---------------------------------------------------------------------------

def run_tfidf(run, manifest: list[dict], log) -> dict:
    """Train and score the lexical ML detector.

    Two feature blocks are combined:
      - word n-grams, which capture API names such as `os.system` or `pickle`;
      - character n-grams, which survive identifier renaming and catch
        fragments like `shell=` or `% (` that word tokenisation splits apart.

    `min_df=1` because at these dataset sizes a feature appearing in a single
    program may still be the informative one; the usual reason to raise it --
    controlling vocabulary size -- does not apply to tens of files.
    """
    log.info("  training TF-IDF + Logistic Regression")

    sources, labels, cwes = _dataset(run, manifest)

    vectoriser = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            # Python identifiers, dotted calls and keyword arguments, kept as
            # single tokens so `shell=True` is not shattered into punctuation.
            token_pattern=r"[A-Za-z_][A-Za-z0-9_.]+|[=(){}\[\]%]",
            ngram_range=(1, 2), min_df=1, sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True,
        )),
    ])
    features = vectoriser.fit_transform(sources)

    return _assemble(
        "ml_tfidf", manifest, features, labels, cwes,
        config.ML_CV_FOLDS, log,
        {"n_features": features.shape[1], "representation": "tf-idf n-grams"},
    )


# ---------------------------------------------------------------------------
# Detector 2 -- semantic (code embeddings)
# ---------------------------------------------------------------------------

def run_embedding(run, manifest: list[dict], log) -> dict:
    """Train and score the embedding-based ML detector.

    Programs are embedded by the endpoint's code embedding model, then fed to
    the same logistic regression as the lexical detector.

    Long files are truncated before embedding: the model has a fixed context
    window, and a request exceeding it is rejected outright. Truncation is
    recorded in the result so a run where it happened often can be spotted --
    it would mean the tail of those programs was never seen.
    """
    log.info("  embedding samples via the judgment endpoint")

    sources, labels, cwes = _dataset(run, manifest)

    # Roughly 8k characters is a safe budget for an 8B embedding model's
    # context. The benchmark's own samples are 30-70 lines and never come
    # close; ingested GitHub files can be far larger.
    max_chars = 8000
    truncated = sum(1 for s in sources if len(s) > max_chars)
    prepared = [s[:max_chars] for s in sources]

    try:
        vectors = embed_texts(prepared)
    except ProviderError as exc:
        raise InsufficientData(
            f"Could not embed samples: {exc}"
        ) from exc

    features = np.asarray(vectors, dtype=float)
    log.info(f"    embedded {features.shape[0]} samples, dim {features.shape[1]}")

    return _assemble(
        "ml_embedding", manifest, features, labels, cwes,
        config.ML_CV_FOLDS, log,
        {
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_dim": int(features.shape[1]),
            "truncated_samples": truncated,
            "representation": "neural code embeddings",
        },
    )
