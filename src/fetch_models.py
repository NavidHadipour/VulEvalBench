"""Download the pretrained ML detector weights into models/.

    python src/fetch_models.py            # fetch what is missing
    python src/fetch_models.py --force    # re-download everything

The ML arm runs two published model families. Their weights are roughly 30 MB,
belong to their original authors, and are not redistributed in this
repository -- this script fetches them from the authors' own sources on
demand. `models/` is gitignored, so nothing downloaded here is ever committed.

WHAT IS FETCHED

    models/farasat-bilstm/     7 x BiLSTM .h5  + word2vec (300-dim)
        Farasat & Posegga, "Machine Learning Techniques for Python Source Code
        Vulnerability Detection", ACM CODASPY 2024.
        github.com/Tf-arch/Python-Source-Code-Vulnerability-Detection
        Released under the Unlicense.

    models/vudenc-lstm/        7 x LSTM .h5    + word2vec (200-dim)
        Wartschinski et al., "VUDENC", Information and Software Technology
        144 (2022).
        github.com/LauraWartschinski/VulnerabilityDetection

THE VUDENC EMBEDDING GAP -- read this before reporting VUDENC as broken

    VUDENC's seven LSTM models download and load cleanly. Its word2vec model
    does not work, and cannot be made to work from published files alone.

    gensim 3.x wrote any array over 10 MB to a separate `.npy` sidecar. The
    published `word2vec_withString10-100-200.model` is therefore only the
    vocabulary -- 132,891 tokens, 8.6 MB -- while the 106 MB
    `...model.wv.vectors.npy` holding the actual vectors is absent from the
    GitHub repository (all branches, no releases) and from all four VUDENC
    Zenodo records (3559841, 3559480, 3559203, 10903584), each of which was
    checked.

    The repository's `Code/w2v/othermodels.rar` is the last place it could
    have been, and it is not there either: the archive holds 31 further
    word2vec models -- every vocabulary size and window the paper swept -- and
    every one is a bare header of the same ~9 MB, with no `.npy` among the 31
    entries. The sidecar was never published for any variant, so this is not
    an oversight on one file.

    WHY it is missing is almost certainly mechanical rather than deliberate:
    132,891 tokens x 200 dimensions x 4 bytes is 101.4 MiB, which is 1.4 MiB
    over GitHub's hard 100 MiB per-file push limit. The file could not have
    been committed even if the authors had tried, and none of the 50 forks
    carries it either (every one is the same size as the original or smaller).

    This is not only our problem. Issue #14 on the repository, "Cannot load
    pretrained Word2vec model", reports the identical FileNotFoundError; it
    was filed in October 2023, has had no reply from the authors, and a second
    user asked in March 2025 whether anyone had found a solution. The
    repository's last commit is from March 2020. So the benchmark's
    "unavailable" verdict for VUDENC is the reproducible state of the
    artefact, independently observed, and not a local setup fault.

    Retraining word2vec on the published corpus does NOT fix this. Word2Vec is
    initialised randomly and has no canonical orientation, so a fresh run on
    identical data yields a different vector space. The pretrained LSTM was
    fitted against one specific space; feeding it another is feeding it noise,
    and it would produce confident nonsense rather than an honest failure.

    So the VUDENC detector reports itself unavailable until the file exists.
    To enable it, obtain the sidecar from the authors and drop it in beside
    the model:

        models/vudenc-lstm/word2vec_withString10-100-200.model.wv.vectors.npy

    The Passau models are unaffected: their word2vec is 300-dimensional over a
    3,576-token vocabulary, small enough that gensim stored it inline.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from common import MODELS_DIR, ensure_dirs

FARASAT_RAW = (
    "https://raw.githubusercontent.com/Tf-arch/"
    "Python-Source-Code-Vulnerability-Detection/main"
)
VUDENC_RAW = (
    "https://raw.githubusercontent.com/LauraWartschinski/"
    "VulnerabilityDetection/master/Code"
)
# VUDENC's word2vec is published on Zenodo rather than in the repository. The
# repository carries a different hyperparameter setting (10-300-200) than the
# one its training script actually uses (10-100-200), so the Zenodo copy is
# the correct partner for the published LSTM weights.
VUDENC_WORD2VEC_URL = (
    "https://zenodo.org/records/3559480/files/"
    "word2vec_withString10-100-200.model?download=1"
)

# The authors' own weight filenames. The Passau repository capitalises
# inconsistently -- five files start with an upper-case B and two do not --
# so the exact spellings are listed rather than generated from a pattern.
FARASAT_WEIGHTS = {
    "sql": "bidirectional_LSTM_model_sql.h5",
    "xss": "Bidirectional_LSTM_model_xss.h5",
    "command_injection": "Bidirectional_LSTM_model_command_injection.h5",
    "remote_code_execution": "Bidirectional_LSTM_model_remote_code_execution.h5",
    "path_disclosure": "Bidirectional_LSTM_model_path_disclosure.h5",
    "open_redirect": "Bidirectional_LSTM_model_open_redirect.h5",
    "xsrf": "bidirectional_LSTM_model_xsrf.h5",
}

# word2vec filenames encode their hyperparameters as mincount-iterations-size.
# Each family's LSTM was trained against one specific table and only works
# with that one, so these are fixed by the weights.
FARASAT_WORD2VEC = "word2vec_withString10-200-300.model"   # 300-dim, complete
VUDENC_WORD2VEC = "word2vec_withString10-100-200.model"    # 200-dim, no vectors

# Local names the pipeline expects. Normalised to one pattern so
# vudenc_pipeline can address both families with a single format string.
LOCAL_PATTERN = "model_{vulnerability}.h5"


def _download(url: str, target: Path, force: bool) -> str:
    """Fetch one file, returning a short status word for the report."""
    if target.exists() and target.stat().st_size > 0 and not force:
        return "present"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Written to a temporary name first so an interrupted download cannot
        # leave a truncated file that later looks "present" and fails to load.
        staging = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response:
            staging.write_bytes(response.read())
        if staging.stat().st_size == 0:
            staging.unlink()
            return "empty response"
        staging.replace(target)
        return f"{target.stat().st_size // 1024} KB"
    except (urllib.error.URLError, OSError) as exc:
        return f"FAILED ({exc})"


def fetch_family(name: str, base_url: str, subdir: str,
                 weights: dict[str, str], word2vec: str,
                 force: bool, weight_prefix: str) -> list[tuple[str, str]]:
    """Download one model family's weights and embedding table."""
    target_dir = MODELS_DIR / subdir
    results: list[tuple[str, str]] = []

    for vulnerability, remote_name in weights.items():
        local = target_dir / LOCAL_PATTERN.format(vulnerability=vulnerability)
        status = _download(f"{base_url}/{weight_prefix}/{remote_name}", local, force)
        results.append((f"{name}: {vulnerability}", status))

    url = (f"{base_url}/wordtovec_models/{word2vec}"
           if name == "Passau BiLSTM" else VUDENC_WORD2VEC_URL)
    results.append((f"{name}: word2vec",
                    _download(url, target_dir / word2vec, force)))
    return results


def main() -> int:
    """Download both families and print a status table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="re-download files that are already present")
    args = parser.parse_args()

    ensure_dirs()
    print("\nFetching pretrained ML detector weights into", MODELS_DIR)
    print("-" * 72)

    results = fetch_family(
        "Passau BiLSTM", FARASAT_RAW, config.VUDENC_FARASAT_DIR,
        FARASAT_WEIGHTS, FARASAT_WORD2VEC, args.force, "Model-BiLSTM",
    )
    results += fetch_family(
        "VUDENC LSTM", VUDENC_RAW, config.VUDENC_ORIGINAL_DIR,
        {v: f"LSTM_model_{v}.h5" for v in config.VUDENC_VULNERABILITIES},
        VUDENC_WORD2VEC, args.force, "model",
    )

    width = max(len(label) for label, _ in results) + 2
    for label, status in results:
        print(f"  {label:<{width}} {status}")
    print("-" * 72)

    failures = [label for label, status in results if status.startswith("FAILED")]
    if failures:
        print(f"  {len(failures)} download(s) failed -- check the network and retry.")
        return 1

    sidecar = (MODELS_DIR / config.VUDENC_ORIGINAL_DIR
               / f"{VUDENC_WORD2VEC}.wv.vectors.npy")
    print("  Passau BiLSTM detector: ready.")
    if sidecar.exists():
        print("  VUDENC LSTM detector:   ready.")
    else:
        print("  VUDENC LSTM detector:   weights downloaded, but UNAVAILABLE --")
        print("     its word2vec vector array was never published. See the")
        print("     'VUDENC embedding gap' note at the top of this file.")
        print(f"     Expected: {sidecar}")
    print("  Next: python src/check_setup.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
