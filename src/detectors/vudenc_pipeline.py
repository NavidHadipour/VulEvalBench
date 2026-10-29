"""Shared inference pipeline for the two pretrained Python vulnerability models.

Both detectors in the ML arm -- VUDENC's LSTM and the Passau BiLSTM that
builds on it -- are *pretrained* sequence models that share one preprocessing
pipeline. This module owns that pipeline so neither detector reimplements it
and the two stay strictly comparable: any difference in their scores comes
from the weights, not from how the code was cut up and embedded.

PROVENANCE
    The preprocessing below is a port of `myutils.py` from

        L. Wartschinski et al., "VUDENC: Vulnerability Detection with Deep
        Learning on a Natural Codebase for Python", Information and Software
        Technology 144 (2022).
        https://github.com/LauraWartschinski/VulnerabilityDetection

    and the identical copy carried in

        T. Farasat and J. Posegga, "Machine Learning Techniques for Python
        Source Code Vulnerability Detection", ACM CODASPY 2024.
        https://github.com/Tf-arch/Python-Source-Code-Vulnerability-Detection
        (released under the Unlicense)

    It is a port rather than an import: the originals are scripts that print,
    draw PNGs and read hard-coded paths, and they call gensim 3.x and Keras 1.x
    APIs that no longer exist. The *algorithm* is unchanged, because changing
    it would invalidate the pretrained weights -- a model can only be fed the
    representation it was trained on. Where behaviour had to be preserved
    exactly, the original is quoted in a comment.

HOW THESE MODELS SEE CODE
    Neither model classifies a file. Both classify a sliding *context window*
    of roughly 200 characters, and the paper's unit of prediction is a token
    within its context. So one benchmark sample produces many windows:

        source -> sliding windows -> tokens -> word2vec vectors
               -> pad/truncate to 200 -> LSTM -> P(vulnerable) per window

    The benchmark asks a per-sample question, so the window probabilities are
    aggregated: a sample counts as vulnerable if *any* window crosses the
    threshold, and the reported confidence is the strongest window. That is
    the same "anything suspicious anywhere in the file" convention the static
    analysers already follow here, which keeps the arms comparable.

WHY THE WEIGHTS ARE NOT IN THE REPOSITORY
    The seven models plus their word2vec table are about 30 MB and belong to
    their original authors. `fetch_models.py` downloads them on demand into
    `models/`, which is gitignored. Nothing here trains anything: these are
    frozen published artefacts, which is exactly why they can be scored under
    the same protocol as Bandit or an LLM.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

import config
from common import MODELS_DIR

# Keras is noisy on import and the chatter would bury the run log.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


class ModelUnavailable(RuntimeError):
    """A pretrained artefact this detector needs is missing or unreadable.

    Raised rather than returning "everything is safe": a model that never
    loaded has not found the files clean, and scoring its silence as a verdict
    would quietly credit it with perfect specificity.
    """


# ---------------------------------------------------------------------------
# Tokenisation -- ported verbatim in behaviour from VUDENC's myutils.py
# ---------------------------------------------------------------------------
# The exact split character set the word2vec vocabulary was built with.
# Changing a single entry would produce tokens the embedding has never seen,
# so this list is frozen by the pretrained weights, not a design choice.
SPLIT_CHARS = [
    " ", "\t", "\n", ".", ":", "(", ")", "[", "]", "<", ">", "+", "-", "=",
    '"', "'", "*", "/", "\\", "~", "{", "}", "!", "?", "*", ";", ",", "%", "&",
]
_SPLIT_SET = frozenset(SPLIT_CHARS)


def get_tokens(change: str) -> list[str]:
    """Split source text into the token stream the word2vec model was built on.

    Mirrors `myutils.getTokens`. The leading `replace` calls glue punctuation
    back onto the preceding token -- the original's way of normalising the
    spacing variations found in diffs -- and are kept because the vocabulary
    was built from their output.
    """
    for old, new in ((" .", "."), (" ,", ","), (" )", ")"), (" (", "("),
                     (" ]", "]"), (" [", "["), (" {", "{"), (" }", "}"),
                     (" :", ":"), ("- ", "-"), ("+ ", "+"), (" =", "="),
                     ("= ", "=")):
        change = change.replace(old, new)

    tokens: list[str] = []
    start = 0
    for i, char in enumerate(change):
        if char in _SPLIT_SET:
            if i > start:
                start += 1
                token = change[:i] if start == 1 else change[start:i]
                if token:
                    tokens.append(token)
                tokens.append(char)
                start = i
    return tokens


def _next_split(source: str, focus: int) -> int:
    """Index of the next split character after `focus`, or -1."""
    for pos in range(focus + 1, len(source)):
        if source[pos] in _SPLIT_SET:
            return pos
    return -1


def _previous_split(source: str, focus: int) -> int:
    """Index of the previous split character before `focus`, or -1."""
    for pos in range(focus - 1, -1, -1):
        if source[pos] in _SPLIT_SET:
            return pos
    return -1


def get_context_pos(source: str, focus: int, full_length: int) -> tuple[int, int] | None:
    """Grow a window outwards from `focus` until it exceeds `full_length` chars.

    Mirrors `myutils.getContextPos`. The window grows alternately left and
    right -- one token at a time, never mid-token -- so the focus stays roughly
    centred. Returns None when the file is too short to fill a window, which is
    the original's way of saying "no prediction here".
    """
    if focus > len(source) - 1:
        return None

    start_context = end_context = focus
    grow_left = True
    while not len(source[start_context:end_context]) > full_length:
        if (_previous_split(source, start_context) == -1
                and _next_split(source, end_context) == -1):
            return None
        if grow_left:
            prev = _previous_split(source, start_context)
            if prev > -1:
                start_context = prev
        else:
            nxt = _next_split(source, end_context)
            if nxt > -1:
                end_context = nxt
        grow_left = not grow_left
    return start_context, end_context


def context_windows(source: str, step: int, full_length: int) -> list[str]:
    """Slide over `source` and return every distinct context window.

    Mirrors the traversal in `myutils.getblocks`: advance to the next line
    break when one is close, otherwise to the next split character `step`
    positions ahead. Duplicate windows are dropped -- the original does the
    same, and it matters here because a short benchmark sample produces many
    identical windows that would otherwise be predicted on repeatedly.
    """
    windows: list[str] = []
    seen: set[str] = set()
    focus = last_focus = 0

    while focus <= len(source):
        if source[last_focus:focus] != "\n":
            middle = last_focus + round(0.5 * (focus - last_focus))
            context = get_context_pos(source, middle, full_length)
            if context is not None:
                text = source[context[0]:context[1]]
                if text not in seen:
                    seen.add(text)
                    windows.append(text)

        # Advance. A nearby newline wins so windows align to lines where
        # possible; otherwise step forward to the next split character.
        lookahead = source[focus + 1:focus + 7]
        if "\n" in lookahead:
            last_focus, focus = focus, focus + lookahead.find("\n") + 1
        else:
            nxt = _next_split(source, focus + step)
            if nxt > -1:
                last_focus, focus = focus, nxt
            elif focus < len(source):
                last_focus, focus = focus, len(source)
            else:
                break
    return windows


# ---------------------------------------------------------------------------
# Loading the published artefacts
# ---------------------------------------------------------------------------

def _load_word_vectors(path: Path) -> dict[str, np.ndarray]:
    """Read a gensim 3.x word2vec model into a plain {token: vector} dict.

    gensim 4's backward-compatibility shim cannot load these 2020-era files,
    and depending on gensim at run time only to read a frozen lookup table
    would be a heavy dependency for no benefit. So the pickle is read directly
    and the two fields that matter are extracted.

    Raises:
        ModelUnavailable: if the file is missing, or if its vector array was
            saved to a `.npy` sidecar that was never published -- see the
            VUDENC note in `fetch_models.py`. Both are configuration problems
            with different fixes, so they get different messages.
    """
    if not path.exists():
        raise ModelUnavailable(
            f"word2vec model not found at {path}. Run: python src/fetch_models.py"
        )

    with path.open("rb") as handle:
        model = pickle.load(handle)

    fields = vars(model.wv)
    vectors = fields.get("vectors", fields.get("syn0"))
    vocab = fields.get("vocab", {})

    if vectors is None:
        raise ModelUnavailable(
            f"{path.name} contains its vocabulary ({len(vocab)} tokens) but not "
            "its vector array: gensim saved that to a '.model.wv.vectors.npy' "
            "sidecar which the authors did not publish. The embedding cannot be "
            "reconstructed by retraining -- a fresh word2vec run produces a "
            "different vector space, which the pretrained LSTM would read as "
            "noise. See docs/ARCHITECTURE.md, 'The VUDENC embedding gap'."
        )

    # gensim 3 stored a Vocab object per token whose `.index` addresses the
    # matrix row; gensim 4 stores the row number directly. Accept both.
    lookup: dict[str, np.ndarray] = {}
    for token, entry in vocab.items():
        index = getattr(entry, "index", entry)
        if isinstance(index, (int, np.integer)) and 0 <= index < len(vectors):
            lookup[token] = vectors[index]
    return lookup


def _keras_load(path: Path):
    """Load a Keras 2-era `.h5` model for inference only.

    `tf_keras` is the maintained Keras 2 API; Keras 3 cannot read these files.
    `compile=False` skips the optimiser and the custom `f1`/`f1_loss` metrics
    the originals were trained with -- those call TensorFlow 1 functions that
    no longer exist, and inference never needs them.
    """
    if not path.exists():
        raise ModelUnavailable(
            f"model weights not found at {path}. Run: python src/fetch_models.py"
        )
    try:
        import tf_keras
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ModelUnavailable(
            "tf-keras is not installed; the pretrained ML detectors need it. "
            "Run: pip install tensorflow-cpu tf-keras"
        ) from exc
    return tf_keras.models.load_model(str(path), compile=False)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _vectorise(windows: list[str], lookup: dict[str, np.ndarray],
               dimensions: int, max_length: int) -> tuple[np.ndarray, list[int]]:
    """Turn context windows into one padded batch ready for the model.

    Tokens outside the word2vec vocabulary are skipped rather than mapped to a
    zero vector -- that is what the original does, and the models were trained
    on sequences built that way.

    Returns:
        (batch, kept) where `batch` has shape (len(kept), max_length,
        dimensions) and `kept` indexes the windows that produced at least one
        known token. Windows with no known tokens yield no prediction at all,
        rather than a prediction on an all-zero sequence.
    """
    sequences: list[np.ndarray] = []
    kept: list[int] = []

    for index, window in enumerate(windows):
        tokens = get_tokens(window.replace("\n", " "))
        if len(tokens) <= 1:
            continue
        vectors = [lookup[t] for t in tokens if t != " " and t in lookup]
        if not vectors:
            continue
        sequences.append(np.asarray(vectors, dtype="float32"))
        kept.append(index)

    if not sequences:
        return np.empty((0, max_length, dimensions), dtype="float32"), []

    # Left-pad / left-truncate, matching keras.preprocessing.sequence.
    # pad_sequences defaults, which is how the training data was shaped.
    batch = np.zeros((len(sequences), max_length, dimensions), dtype="float32")
    for row, sequence in enumerate(sequences):
        clipped = sequence[-max_length:]
        batch[row, max_length - len(clipped):] = clipped
    return batch, kept


class PretrainedDetector:
    """One published model family: seven per-vulnerability binary classifiers.

    Neither VUDENC nor the Passau work trains a single multi-class model. Each
    vulnerability type has its own binary model, so a sample is run through all
    seven and the strongest responder names the CWE. That is why these
    detectors can answer the benchmark's second question -- *which* weakness --
    without a separate classifier: the model that fired is the answer.
    """

    def __init__(self, family: str, model_dir: Path, word2vec_file: str,
                 weight_pattern: str, dimensions: int):
        self.family = family
        self.model_dir = model_dir
        self.word2vec_file = word2vec_file
        self.weight_pattern = weight_pattern
        self.dimensions = dimensions
        self._lookup: dict[str, np.ndarray] | None = None
        self._models: dict[str, object] = {}

    # -- lazy loading -------------------------------------------------------
    # Loading TensorFlow and seven models takes several seconds and tens of
    # megabytes. Doing it on construction would pay that cost to render the
    # Tools page; doing it on first use pays it only in a run that uses them.

    @property
    def lookup(self) -> dict[str, np.ndarray]:
        """The token -> vector table, loaded once per process."""
        if self._lookup is None:
            self._lookup = _load_word_vectors(self.model_dir / self.word2vec_file)
        return self._lookup

    def model_for(self, vulnerability: str):
        """The binary model for one vulnerability type, loaded once per process."""
        if vulnerability not in self._models:
            filename = self.weight_pattern.format(vulnerability=vulnerability)
            self._models[vulnerability] = _keras_load(self.model_dir / filename)
        return self._models[vulnerability]

    def available(self) -> bool:
        """True if every artefact this family needs is present and readable.

        Used to grey the detector out in the interface instead of letting a
        user start a run that cannot finish.
        """
        try:
            self.lookup
        except ModelUnavailable:
            return False
        return all(
            (self.model_dir / self.weight_pattern.format(vulnerability=v)).exists()
            for v in config.VUDENC_VULNERABILITIES
        )

    # -- prediction ---------------------------------------------------------

    def scan(self, source: str) -> dict[str, float]:
        """Score one program against all seven models.

        Returns:
            {vulnerability_type: highest window probability}. The maximum over
            windows is the file-level score: these models localise, so a single
            confident window is the model saying "the weakness is here", and
            averaging it away with the file's clean majority would discard
            exactly the signal they were trained to produce.
        """
        windows = context_windows(
            source, config.VUDENC_STEP, config.VUDENC_CONTEXT_LENGTH
        )
        if not windows:
            return {}

        batch, kept = _vectorise(
            windows, self.lookup, self.dimensions, config.VUDENC_MAX_LENGTH
        )
        if not kept:
            return {}

        scores: dict[str, float] = {}
        for vulnerability in config.VUDENC_VULNERABILITIES:
            probabilities = self.model_for(vulnerability).predict(
                batch, verbose=0, batch_size=config.VUDENC_BATCH_SIZE
            )
            scores[vulnerability] = float(np.max(probabilities))
        return scores
