"""Shared infrastructure: filesystem layout, credentials, logging, small utils.

Every pipeline stage imports from here, so the location of a run, the format of
the credential files and the way logs are written are each defined in exactly
one place. If the project layout changes, this is the only file that changes
with it.

Contains no benchmark logic -- only plumbing.

CREDENTIAL FILE FORMAT
    Both key files use the same shape: a label line, then the value on its own
    line in double quotes.

        api-key-judgment.txt          api-key-llm.txt (optional)
        ------------------------      ---------------------------
        URL:                          anthropic api key:
        "https://host.example"        "sk-ant-..."

        api key:
        "sk-..."

    Templates ship as `api-key-judgment.txt.example` and
    `api-key-llm.txt.example` in the project root: copy, rename, fill in.

    Rather than depend on the exact labels -- which have already been renamed
    once during this project -- the parser extracts every double-quoted value
    and classifies it by its own shape: something starting with `http` is a
    base URL, `sk-ant-` is an Anthropic key, any other `sk-` is an
    OpenAI-compatible key. That keeps the loader working when a label is
    reworded or the two entries swap order.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
# Paths derive from this file's own location rather than the current working
# directory, so the scripts behave identically whether launched from the
# project root, from inside src/, from the web app, or from an IDE.
#
#   __file__            -> <project>/src/common.py
#   .resolve().parent   -> <project>/src
#   .parent             -> <project>          (the project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Persistent, append-only store of every run ever made. This is the change the
# supervisor asked for: a new run never overwrites an old one, so a dataset
# that produced an interesting result can always be revisited.
RUNS_DIR = PROJECT_ROOT / "runs"

# One log file per run, plus a rolling application log for the web server.
LOGS_DIR = PROJECT_ROOT / "logs"

# Where trained ML detector artefacts (vectorisers, fitted classifiers) are
# cached so a model is not retrained on every evaluation.
MODELS_DIR = PROJECT_ROOT / "models"

# Working copies of code pulled in from GitHub or a local directory.
INGEST_DIR = PROJECT_ROOT / "ingested"

# Credential files. Both spellings of "judgment"/"judgement" are accepted
# because the file on disk and the project brief disagree, and a missing key
# is a confusing failure to debug.
JUDGMENT_KEY_FILES = [
    PROJECT_ROOT / "api-key-judgment.txt",
    PROJECT_ROOT / "api-key-judgement.txt",
]
# Likewise for the casing of the LLM key file, which matters on case-sensitive
# filesystems (a Linux machine running the container) but not on Windows.
LLM_KEY_FILES = [
    PROJECT_ROOT / "api-key-llm.txt",
    PROJECT_ROOT / "api-key-LLM.txt",
]


def ensure_dirs() -> None:
    """Create every top-level output directory if it does not already exist.

    `parents=True` also creates intermediates; `exist_ok=True` makes the call
    safe to repeat, so every entry point can call it unconditionally.
    """
    for directory in (RUNS_DIR, LOGS_DIR, MODELS_DIR, INGEST_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

# Matches a double-quoted value on its own. Non-greedy so two quoted values on
# one line are captured separately rather than swallowed as one.
_QUOTED_RE = re.compile(r'"([^"\n]+)"')


def _parse_credential_file(path: Path) -> dict[str, str]:
    """Extract the credentials from one key file.

    Values are classified by shape rather than by the label above them -- see
    the module docstring for why.

    Args:
        path: the credential file to read.

    Returns:
        A dict with any of the keys `base_url`, `api_key`, `anthropic_key`.
        Missing entries are simply absent rather than empty, so a caller can
        distinguish "not supplied" from "supplied but blank".
    """
    if not path.exists():
        return {}

    found: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")

    for raw in _QUOTED_RE.findall(text):
        value = raw.strip()
        if not value:
            continue
        if value.startswith("http://") or value.startswith("https://"):
            # rstrip("/") normalises the base URL so callers can always append
            # "/v1/..." without risking a double slash.
            found["base_url"] = value.rstrip("/")
        elif value.startswith("sk-ant-"):
            found["anthropic_key"] = value
        elif value.startswith("sk-"):
            found["api_key"] = value

    # Fall back to an unquoted single-line file, which is how the very first
    # version of this project stored its key. Cheap to support, and it means
    # an older checkout keeps working.
    if not found:
        bare = text.strip()
        if bare.startswith("sk-ant-"):
            found["anthropic_key"] = bare
        elif bare.startswith("sk-"):
            found["api_key"] = bare

    return found


def _first_existing(paths: list[Path]) -> Path | None:
    """Return the first path in `paths` that exists, or None."""
    return next((p for p in paths if p.exists()), None)


def load_judgment_credentials() -> dict[str, str]:
    """Return `{base_url, api_key}` for the judgment / OpenAI-compatible API.

    Raises:
        SystemExit: if the file is missing or does not contain both a URL and
            a key. This is a configuration error rather than a bug, so it stops
            with a readable message instead of raising a traceback at the user.
    """
    path = _first_existing(JUDGMENT_KEY_FILES)
    if path is None:
        raise SystemExit(
            "No judgment key file found. Create "
            f"{JUDGMENT_KEY_FILES[0].name} in {PROJECT_ROOT} containing the\n"
            'API base URL and key, each in double quotes on its own line:\n\n'
            "URL:\n"
            '"https://llms.example.edu"\n\n'
            "api key:\n"
            '"sk-..."\n'
        )

    creds = _parse_credential_file(path)
    missing = [k for k in ("base_url", "api_key") if k not in creds]
    if missing:
        raise SystemExit(
            f"{path.name} is missing: {', '.join(missing)}.\n"
            "It must contain both the base URL and the API key, each in "
            "double quotes."
        )
    return creds


def load_anthropic_key() -> str | None:
    """Return the Anthropic key from the LLM key file, or None if absent.

    Returns None rather than raising: Anthropic models are an *option* in this
    pipeline, not a requirement, so their absence must not stop a run that uses
    only the local generator and the Passau judges.
    """
    path = _first_existing(LLM_KEY_FILES)
    if path is None:
        return None
    return _parse_credential_file(path).get("anthropic_key")


def credential_status() -> dict[str, bool]:
    """Report which credentials are available, for display in the web UI.

    Never returns the secrets themselves -- only whether each one was found --
    so the status can be rendered in a browser page without leaking a key.
    """
    judgment_path = _first_existing(JUDGMENT_KEY_FILES)
    judgment = _parse_credential_file(judgment_path) if judgment_path else {}
    return {
        "judgment_url": "base_url" in judgment,
        "judgment_key": "api_key" in judgment,
        "anthropic_key": load_anthropic_key() is not None,
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# Two destinations, always both:
#   - a per-run file under logs/, which is the durable record, and
#   - an in-memory callback, which is what the web UI streams to the browser.
#
# Routing every stage's progress through one helper is what guarantees the
# terminal, the log file and the browser show the same lines.

def new_run_id(prefix: str = "run") -> str:
    """Build a sortable, human-readable identifier for a new run.

    Format: `run_2026-08-10_17-30-00`. Sorting these as strings sorts them
    chronologically, which is why the timestamp is written big-endian and
    zero-padded rather than in a local format.
    """
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{stamp}"


def get_logger(run_id: str, on_progress=None) -> logging.Logger:
    """Create a logger that writes to `logs/<run_id>.log` and to a callback.

    Args:
        run_id: identifies the run; also names the log file.
        on_progress: optional callable taking one string. Receives the same
            lines that reach the file, which is how the browser follows a run.

    Returns:
        A configured `logging.Logger`. Handlers are cleared first so that
        calling this twice for the same run -- which happens when a stage is
        re-run from the web UI -- does not duplicate every line.
    """
    ensure_dirs()

    logger = logging.getLogger(f"vulevalbench.{run_id}")
    logger.setLevel(logging.INFO)
    # Stop records propagating to the root logger, which Flask also writes to;
    # without this every pipeline line would appear twice in the console.
    logger.propagate = False

    # Remove handlers left over from a previous call for this same run id.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(
        LOGS_DIR / f"{run_id}.log", encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s")
    )
    logger.addHandler(file_handler)

    if on_progress is not None:
        logger.addHandler(_CallbackHandler(on_progress))

    return logger


class _CallbackHandler(logging.Handler):
    """Logging handler that forwards each formatted line to a callable.

    Used by the web app to stream a run's progress into the job log without
    the pipeline modules needing to know a web app exists.
    """

    def __init__(self, callback) -> None:
        super().__init__()
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        """Forward one formatted log line to the callback."""
        # A failing progress callback (a closed browser connection, say) must
        # never take down the pipeline thread that is doing the real work.
        try:
            self.callback(record.getMessage())
        except Exception:  # noqa: BLE001 - deliberately swallowed, see above
            pass


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Reduce arbitrary text to a safe filename fragment.

    Model identifiers contain dots, slashes and colons (`qwen2.5-coder:7b`),
    all of which are either illegal in a Windows filename or would silently
    create a subdirectory. This maps them to hyphens so a model name can be
    embedded in a result filename -- which is what makes those filenames
    self-describing.
    """
    normalised = unicodedata.normalize("NFKD", text)
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", ascii_only)
    return cleaned.strip("-").lower() or "unnamed"


def read_json(path: Path) -> dict | list:
    """Read a UTF-8 JSON file.

    `encoding` is always explicit because Windows would otherwise use the
    system code page (cp1252) and corrupt any non-ASCII character a model
    happened to emit.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict | list) -> None:
    """Write `payload` as indented UTF-8 JSON, creating parent directories.

    `indent=2` keeps files readable in a diff and in an editor, which matters
    because manifests and judgment records are reviewed by hand. The trailing
    newline is the usual text-file convention.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
