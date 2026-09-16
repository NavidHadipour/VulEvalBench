"""Arm 2 -- machine-learning detectors.

TWO MODELS THAT DIFFER IN EXACTLY ONE THING
    Both detectors here are the same classifier -- logistic regression --
    trained on the same samples with the same folds. The only thing that
    differs is how a program is turned into numbers:

        ml_tfidf      lexical.   Token and character n-grams. Sees which
                                 *symbols* co-occur with vulnerability, with
                                 no notion of data flow or meaning.
        ml_embedding  semantic.  A neural code-embedding model turns each
                                 program into a single dense vector.

    Holding the classifier and the data fixed is deliberate: any difference in
    their scores is attributable to the representation, which is the only
    interesting question here. Adding a second variable -- a different
    classifier for each -- would make the comparison uninterpretable.

    This mirrors the paired-generation logic used for the dataset itself:
    change one thing at a time, or you cannot attribute the result.

WHY EVERY PREDICTION IS OUT-OF-FOLD
    These models learn from the same small dataset they are scored on. A model
    scored on its own training data would report near-perfect numbers that
    describe memorisation, not detection -- and it would be silently, wildly
    unfair to the static and LLM arms, which never see a label at all.

    So predictions come from `cross_val_predict` with stratified folds: the
    model that labels a given sample was fitted on the other folds only, and
    never saw that sample. This is the honest way to put a supervised learner
    in a benchmark against unsupervised tools, and it must be stated in any
    write-up -- the ML arm is measured under a genuinely different protocol
    from the other two.

A HONEST LIMITATION
    At the dataset sizes this project currently produces (tens of samples),
    these models are severely data-starved and their numbers will be poor and
    noisy. That is a real finding about the *dataset size*, not a bug. They
    become meaningful at the few-hundred-sample scale the framework is built
    for. The CWE multiclass head in particular needs far more data than the
    binary head, and gracefully declines to predict when it has too little.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion

import config
from detectors import build_result, make_prediction
from providers import ProviderError, embed_texts


class InsufficientData(RuntimeError):
    """The dataset is too small or too one-sided to train on.

    A dedicated type so the runner can report "this arm could not run on this
    run" as a normal outcome rather than a crash -- which is exactly what
    happens when a judging stage admits only vulnerable samples, or only three
    samples in total.
    """


# ---------------------------------------------------------------------------
# Shared training and scoring
# ---------------------------------------------------------------------------

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
