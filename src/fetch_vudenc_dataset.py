"""Download VUDENC's labelled training data.

WHAT THIS FETCHES, AND WHY IT IS SEPARATE FROM fetch_models.py
    `fetch_models.py` downloads the trained artefacts a detector needs to make
    a prediction -- the LSTM weights and the word2vec table. This script
    downloads the *training data* those artefacts were fitted on, which the
    benchmark does not need to run and which is two orders of magnitude
    larger. Keeping them apart means a normal setup never pulls 50 MB of other
    people's commit history.

    Needed only if VUDENC is to be retrained. The published LSTM weights
    cannot be used as they stand, because the word2vec vector array they were
    fitted against was never uploaded -- see `_load_word_vectors` in
    detectors/vudenc_pipeline.py.

WHAT THE FILES ARE
    One per vulnerability class, named `plain_<vulnerability>`. Each is a JSON
    object keyed by GitHub repository URL, then by commit hash, holding the
    commit's metadata and its diff. These are the security-fix commits VUDENC
    mined from GitHub: the code *before* each fix is the vulnerable half of
    its training set and the code *after* it is the clean half.

    They are plain JSON despite having no extension, and they are large: the
    open-redirect file alone is 17 MB.

PROVENANCE
    github.com/LauraWartschinski/VulnerabilityDetection, MIT licensed, the
    repository accompanying Wartschinski et al., "VUDENC: Vulnerability
    Detection with Deep Learning on a Natural Codebase for Python",
    Information and Software Technology 144 (2022).

    Downloaded into datasets/, which .gitignore excludes: this is third-party
    data and not ours to redistribute.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT  # noqa: E402

RAW = (
    "https://raw.githubusercontent.com/LauraWartschinski/"
    "VulnerabilityDetection/master/Code"
)

DATASET_DIR = PROJECT_ROOT / "datasets" / "vudenc"

# The seven classes VUDENC trained a model for. The names are the authors'
# own and are used verbatim in the filenames, the model names and the
# config.VUDENC_VULNERABILITIES list, so they are not normalised here.
VULNERABILITIES = [
    "command_injection",
    "open_redirect",
    "path_disclosure",
    "remote_code_execution",
    "sql",
    "xsrf",
    "xss",
]

# Approximate sizes, for the progress line. Not checked against the download:
# a mismatch would mean the authors updated the file, which is not an error.
APPROX_MB = {
    "command_injection": 5.4, "open_redirect": 17.3, "path_disclosure": 5.3,
    "remote_code_execution": 4.0, "sql": 11.5, "xsrf": 8.4, "xss": 1.8,
}


def _download(url: str, target: Path, force: bool) -> str:
    """Fetch one file, skipping it if it is already there.

    Returns a one-word status for the caller to print.
    """
    if target.exists() and not force:
        return "present"

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=300) as response:
            # Written to a temporary name and moved into place, so an
            # interrupted download cannot leave a half-file that the next run
            # would see as "present" and skip.
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(response.read())
            temporary.replace(target)
        return "downloaded"
    except (urllib.error.URLError, OSError) as exc:
        return f"FAILED: {exc}"


def fetch(force: bool = False) -> int:
    """Download every labelled dataset file. Returns the number that failed."""
    print(f"VUDENC labelled training data -> {DATASET_DIR}")
    print("About 54 MB in total, from github.com/LauraWartschinski/"
          "VulnerabilityDetection (MIT).\n")

    failures = 0
    for vulnerability in VULNERABILITIES:
        name = f"plain_{vulnerability}"
        status = _download(f"{RAW}/data/{name}", DATASET_DIR / name, force)
        size = APPROX_MB.get(vulnerability, 0)
        print(f"  {name:32} ~{size:5.1f} MB  {status}")
        if status.startswith("FAILED"):
            failures += 1

    print()
    if failures:
        print(f"{failures} file(s) failed. Re-run to retry; "
              "files already present are skipped.")
    else:
        total = sum(f.stat().st_size for f in DATASET_DIR.glob("plain_*"))
        print(f"All seven classes present, {total / 1e6:.1f} MB.")
        print("Each file is JSON: {repo_url: {commit_hash: {...diff...}}}.")
    return failures


def main() -> None:
    """Usage: python src/fetch_vudenc_dataset.py [--force]"""
    sys.exit(fetch(force="--force" in sys.argv[1:]))


if __name__ == "__main__":
    main()
