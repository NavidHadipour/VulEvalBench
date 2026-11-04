"""Shared infrastructure: filesystem layout, credentials, logging, small utils.

Every pipeline stage imports from here, so the location of a run, the format of
the credential files and the way logs are written are each defined in exactly
one place. If the project layout changes, this is the only file that changes
with it.

Contains no benchmark logic -- only plumbing.

CREDENTIAL FILE FORMAT
    One file holds every credential: `api-keys.txt` in the project root. Each
    line names the endpoint it belongs to, the field, and the value:

        # api-keys.txt
        judgment.url   = "https://host.example"
        judgment.key   = "sk-..."
        anthropic.key  = "sk-ant-..."
        typesafe.key   = "..."
        openrouter.url = "https://openrouter.ai/api"
        openrouter.key = "sk-or-..."

    An entry name is free text. Three names are reserved because the shipped
    providers read them -- `judgment`, `anthropic`, `typesafe` -- and any
    other name carrying both a `url` and a `key` becomes a selectable
    OpenAI-compatible backend, so a new API can be added without a code
    change. A template ships as `api-keys.txt.example`.

    WHY NAMED RATHER THAN SNIFFED
        The three files this replaces -- api-key-judgment.txt, api-key-llm.txt
        and api-key-typesafe.txt -- classified values by *shape*: `http...`
        was a base URL, `sk-ant-` an Anthropic key, any other `sk-` an
        OpenAI-compatible one. That survived the labels being reworded, but it
        cannot represent two OpenAI-compatible endpoints at once, which is
        exactly what "add any API" needs. Names are explicit, so two keys of
        the same shape no longer collide.

    THOSE THREE FILES ARE STILL READ
        Where `api-keys.txt` has no entry, the legacy file is consulted with
        the old shape-based parser, field by field. An existing checkout keeps
        working untouched, and `import_legacy_keys()` folds them into the
        unified file when asked. Nothing here deletes them.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
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

# The unified credential store: every API this installation can reach, in one
# hand-editable file. See the module docstring for the format.
KEYS_FILE = PROJECT_ROOT / "api-keys.txt"

# Superseded credential files, still read so an existing installation keeps
# working. Both spellings of "judgment"/"judgement" are accepted
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
# The TypeSafe endpoint, which serves the Jev System One model. A file of its
# own rather than another entry in the LLM file: the key's prefix is not
# documented, so it cannot be told apart from an OpenAI-style key by shape the
# way _parse_credential_file() separates the others.
TYPESAFE_KEY_FILES = [
    PROJECT_ROOT / "api-key-typesafe.txt",
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


# ---------------------------------------------------------------------------
# The unified store
# ---------------------------------------------------------------------------

# Entry names the shipped providers read. Everything else in the file is a
# user-added endpoint, and the Models page says which is which: deleting
# `judgment` breaks the Passau judges, while deleting `openrouter` breaks only
# the models registered against it.
RESERVED_ENTRIES = {
    "judgment": "the OpenAI-compatible judgment endpoint (url + key)",
    "anthropic": "Anthropic, for Claude models (key)",
    "openai": "OpenAI, for the GPT models (key)",
    "typesafe": "TypeSafe, which serves Jev (key)",
}

# The fields an entry may carry. Anything else is refused rather than stored:
# a mistyped field would otherwise sit in the file looking configured while no
# provider ever reads it.
KEY_FIELDS = ("url", "key")

# `name.field = "value"`. The value is captured inside quotes so a trailing
# comment can never be swallowed into the secret.
_ENTRY_RE = re.compile(
    r'^\s*([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)\s*=\s*"([^"]*)"\s*$'
)
# The same unquoted, for a hand-edited file; a trailing `#` comment is cut.
_BARE_ENTRY_RE = re.compile(
    r"^\s*([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)\s*=\s*([^#\n]+?)\s*$"
)

_KEYS_HEADER = """\
# VulEvalBench credentials -- every API this installation can reach.
#
# One entry per line:
#
#     <name>.<field> = "<value>"
#
# `judgment`, `anthropic`, `openai` and `typesafe` are read by the shipped
# providers.
# Any other name carrying both a url and a key becomes a selectable
# OpenAI-compatible backend; register a model against it on the Models page.
#
# Excluded from version control. Edit it here or on the Models page.
"""


def _parse_keys_file(path: Path) -> dict[str, dict[str, str]]:
    """Parse the unified store into `{entry name: {field: value}}`.

    A malformed line is skipped rather than raising. The file is meant to be
    hand-editable, and one bad line must not take every other credential down
    with it -- a missing key is a legible failure, an exception on import is
    not.
    """
    if not path.exists():
        return {}

    entries: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _ENTRY_RE.match(line) or _BARE_ENTRY_RE.match(line)
        if not match:
            continue
        name, field, value = match.groups()
        value = value.strip().strip('"')
        if not value:
            continue
        field = field.strip().lower()
        if field == "url":
            # rstrip("/") normalises the base URL so callers can always append
            # "/v1/..." without risking a double slash.
            value = value.rstrip("/")
        entries.setdefault(name.strip().lower(), {})[field] = value
    return entries


def _legacy_keys() -> dict[str, dict[str, str]]:
    """The three superseded files, read with the old shape-based parser."""
    entries: dict[str, dict[str, str]] = {}

    judgment_path = _first_existing(JUDGMENT_KEY_FILES)
    if judgment_path:
        parsed = _parse_credential_file(judgment_path)
        entry = {}
        if "base_url" in parsed:
            entry["url"] = parsed["base_url"]
        if "api_key" in parsed:
            entry["key"] = parsed["api_key"]
        if entry:
            entries["judgment"] = entry

    llm_path = _first_existing(LLM_KEY_FILES)
    if llm_path:
        parsed = _parse_credential_file(llm_path)
        if "anthropic_key" in parsed:
            entries["anthropic"] = {"key": parsed["anthropic_key"]}

    typesafe_path = _first_existing(TYPESAFE_KEY_FILES)
    if typesafe_path:
        text = typesafe_path.read_text(encoding="utf-8").strip()
        quoted = _QUOTED_RE.findall(text)
        key = (quoted[0] if quoted else text).strip()
        if key:
            entries["typesafe"] = {"key": key}

    return entries


def load_keys() -> dict[str, dict[str, str]]:
    """Every credential this installation has, as `{name: {field: value}}`.

    The unified file wins field by field rather than entry by entry, so an
    installation that has moved only its Anthropic key across keeps reading
    the legacy file for the rest. A half-migrated state behaves exactly like a
    finished one, which is what makes the migration optional.
    """
    merged = _legacy_keys()
    for name, fields in _parse_keys_file(KEYS_FILE).items():
        merged.setdefault(name, {}).update(fields)
    return merged


def save_keys(entries: dict[str, dict[str, str]]) -> None:
    """Write the unified store, reserved entries first, then the rest sorted.

    Written whole rather than patched line by line: the file is small, and
    rewriting it is the only way to guarantee a deleted entry is gone rather
    than shadowed by a duplicate further down.
    """
    lines = [_KEYS_HEADER.rstrip()]
    ordered = [n for n in RESERVED_ENTRIES if n in entries]
    ordered += sorted(n for n in entries if n not in RESERVED_ENTRIES)
    for name in ordered:
        fields = entries[name]
        if not fields:
            continue
        lines.append("")
        for field in sorted(fields, key=lambda f: (f != "url", f)):
            lines.append(f'{name}.{field} = "{fields[field]}"')
    KEYS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_key_entry(name: str, field: str, value: str) -> list[str]:
    """Check one proposed credential. Returns problems, empty if acceptable.

    Returned rather than raised so a web form can report every problem at once
    instead of only the first.
    """
    problems: list[str] = []
    name = (name or "").strip()
    field = (field or "").strip().lower()
    value = (value or "").strip()

    if not name:
        problems.append("a name is required")
    elif not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        # The name becomes a provider id in models.local.json and a value in a
        # form, so it is refused rather than quietly sanitised.
        problems.append("the name may contain only letters, digits, hyphens "
                        "and underscores")

    if field not in KEY_FIELDS:
        problems.append(f"field must be one of: {', '.join(KEY_FIELDS)}")

    if not value:
        problems.append("a value is required")
    elif '"' in value:
        problems.append("the value may not contain a double quote")
    elif field == "url" and not value.startswith(("http://", "https://")):
        problems.append("a url must start with http:// or https://")

    return problems


def set_key(name: str, field: str, value: str) -> list[str]:
    """Add or replace one credential. Returns problems, empty on success."""
    problems = validate_key_entry(name, field, value)
    if problems:
        return problems

    name = name.strip().lower()
    field = field.strip().lower()
    value = value.strip()
    if field == "url":
        value = value.rstrip("/")

    # Seeded from the merged view, so the first write through the UI carries
    # the legacy credentials across with it. Writing only the new entry would
    # leave api-keys.txt holding one field while the rest still came from the
    # old files -- correct, but confusing to read.
    entries = load_keys()
    entries.setdefault(name, {})[field] = value
    save_keys(entries)
    return []


def delete_key(name: str, field: str | None = None) -> list[str]:
    """Remove one field, or the whole entry when `field` is None.

    Only the unified store is rewritten. This never edits or deletes the
    legacy api-key-*.txt files -- it did not create them, and silently
    truncating a file the user wrote by hand is not a deletion they asked for
    -- so a credential that still lives in one is reported, not removed.
    """
    name = (name or "").strip().lower()
    entries = load_keys()
    if name not in entries:
        return [f"there is no credential named {name!r}"]

    if field:
        field = field.strip().lower()
        if field not in entries[name]:
            return [f"{name} has no {field}"]
        del entries[name][field]
        if not entries[name]:
            del entries[name]
    else:
        del entries[name]

    save_keys(entries)

    remaining = _legacy_keys().get(name, {})
    if remaining and (field is None or field in remaining):
        return [(f"note: {name} is also set in a legacy api-key-*.txt file, "
                 "which was left alone. Delete that file to remove the "
                 "credential completely.")]
    return []


def import_legacy_keys() -> list[str]:
    """Fold the three superseded files into the unified store.

    Returns a human-readable report of what moved. The old files are left in
    place: this copies rather than moves, so a mistake here costs nothing and
    can be undone by deleting api-keys.txt.
    """
    legacy = _legacy_keys()
    if not legacy:
        return ["No legacy api-key-*.txt files found; nothing to import."]

    entries = _parse_keys_file(KEYS_FILE)
    moved = []
    for name, fields in legacy.items():
        for field, value in fields.items():
            if entries.get(name, {}).get(field):
                continue
            entries.setdefault(name, {})[field] = value
            moved.append(f"{name}.{field}")

    if not moved:
        return ["Every legacy credential is already in api-keys.txt."]

    save_keys(entries)
    return [(f"Imported {', '.join(moved)} into api-keys.txt. The old files "
             "were left in place; delete them once you are satisfied the "
             "import is correct.")]


def mask_secret(value: str, field: str = "key") -> str:
    """Render a credential for display without disclosing it.

    A url is shown in full: it is not a secret, and the whole point of showing
    it is to confirm which endpoint is configured. A key shows its first eight
    and last four characters -- enough to tell two keys apart and to match one
    against a provider's console, and nothing more. A short value is replaced
    outright rather than mostly revealed.
    """
    if field == "url":
        return value
    if len(value) <= 16:
        return "*" * 8
    return f"{value[:8]}...{value[-4:]}"


def custom_endpoints() -> dict[str, dict[str, str]]:
    """User-added OpenAI-compatible endpoints, as `{name: {url, key}}`.

    An entry qualifies when it is not one of the reserved names and carries
    both a url and a key. Anything half-configured is left out, so a backend
    becomes selectable only once it could actually be called.
    """
    return {
        name: fields
        for name, fields in load_keys().items()
        if name not in RESERVED_ENTRIES
        and fields.get("url") and fields.get("key")
    }


def keys_summary() -> list[dict]:
    """Every credential, masked, for the Models page.

    Never returns a secret. `reserved` tells the page whether removing the row
    would break a shipped provider, and `source` says which file the value
    came from, so one that cannot be removed here is not presented as if it
    could be.
    """
    unified = _parse_keys_file(KEYS_FILE)
    rows = []
    for name, fields in sorted(load_keys().items()):
        for field in sorted(fields, key=lambda f: (f != "url", f)):
            in_unified = field in unified.get(name, {})
            rows.append({
                "name": name,
                "field": field,
                "masked": mask_secret(fields[field], field),
                "reserved": name in RESERVED_ENTRIES,
                "purpose": RESERVED_ENTRIES.get(name, ""),
                "source": "api-keys.txt" if in_unified else "legacy file",
            })
    return rows


def load_judgment_credentials() -> dict[str, str]:
    """Return `{base_url, api_key}` for the judgment / OpenAI-compatible API.

    Raises:
        SystemExit: if the file is missing or does not contain both a URL and
            a key. This is a configuration error rather than a bug, so it stops
            with a readable message instead of raising a traceback at the user.
    """
    return endpoint_credentials("judgment")


def endpoint_credentials(name: str) -> dict[str, str]:
    """Return `{base_url, api_key}` for one OpenAI-compatible endpoint.

    The same lookup serves the shipped `judgment` endpoint and every endpoint
    the user adds, which is what lets one client implementation call all of
    them.

    Args:
        name: the credential entry name, e.g. "judgment" or "openrouter".

    Raises:
        SystemExit: if the entry is missing a url or a key. This is a
            configuration error rather than a bug, so it stops with a readable
            message instead of raising a traceback at the user.
    """
    fields = load_keys().get(name, {})
    missing = [f for f in ("url", "key") if not fields.get(f)]
    if missing:
        raise SystemExit(
            f"The {name!r} endpoint is missing: {', '.join(missing)}.\n"
            f"Add it to {KEYS_FILE.name} in {PROJECT_ROOT}, or on the Models "
            "page of the web UI:\n\n"
            f'{name}.url = "https://llms.example.edu"\n'
            f'{name}.key = "sk-..."\n'
        )
    # Returned under the historical names because every caller and every
    # error message in providers.py already speaks them.
    return {"base_url": fields["url"], "api_key": fields["key"]}


def load_anthropic_key() -> str | None:
    """Return the Anthropic key from the LLM key file, or None if absent.

    Returns None rather than raising: Anthropic models are an *option* in this
    pipeline, not a requirement, so their absence must not stop a run that uses
    only the local generator and the Passau judges.
    """
    return load_keys().get("anthropic", {}).get("key")


def load_openai_key() -> str | None:
    """Return the OpenAI API key, or None if absent.

    Returns None rather than raising, for the same reason as the Anthropic
    loader: the GPT models are a third-vendor *option*, and a run using only
    the local generator and the Passau judges must not be stopped by their
    absence.

    Note that this is NOT the `judgment` entry, although that endpoint speaks
    the same protocol. They are separate credentials for separate services:
    `judgment` is the university's self-hosted vLLM proxy, and this is
    api.openai.com. Conflating them would route Passau model ids at OpenAI
    prices, and bill the wrong account for a 753-sample judging pass.
    """
    return load_keys().get("openai", {}).get("key")


def load_typesafe_key() -> str | None:
    """Return the TypeSafe API key for Jev, or None if absent.

    The whole file is the key, whether quoted or bare. Nothing is inferred
    from a prefix, because TypeSafe does not document one -- so a key that
    does not start with "sk-" is accepted exactly as given.

    Returns None rather than raising: Jev is one optional judge among several,
    and a run using the other judges must not be stopped by its absence.
    """
    return load_keys().get("typesafe", {}).get("key")


def credential_status() -> dict[str, bool]:
    """Report which credentials are available, for display in the web UI.

    Never returns the secrets themselves -- only whether each one was found --
    so the status can be rendered in a browser page without leaking a key.
    """
    keys = load_keys()
    judgment = keys.get("judgment", {})
    return {
        "judgment_url": bool(judgment.get("url")),
        "judgment_key": bool(judgment.get("key")),
        "anthropic_key": load_anthropic_key() is not None,
        "openai_key": load_openai_key() is not None,
        "typesafe_key": load_typesafe_key() is not None,
        # How many user-added endpoints are fully configured, so the status
        # panel can report them without naming or revealing any of them.
        "custom_endpoints": len(custom_endpoints()),
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


# The most recently created run logger. Low-level modules -- a provider, a
# token-budget guard -- occasionally have something a run's operator needs to
# read, but they sit below the pipeline and are handed no logger: threading one
# through every backend signature to carry the rare message would be worse than
# this. Safe because the web layer runs one job at a time on a single worker
# thread (see web/app.py), so there is only ever one run to report into.
_ACTIVE_LOGGER: logging.Logger | None = None


def active_logger() -> logging.Logger | None:
    """The current run's logger, or None outside a run.

    A caller must tolerate None: `check_setup.py` and the module tests call
    into providers with no run in progress.
    """
    return _ACTIVE_LOGGER


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

    global _ACTIVE_LOGGER
    _ACTIVE_LOGGER = logger
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
# Progress reporting
# ---------------------------------------------------------------------------
#
# WHY A SECOND CHANNEL ALONGSIDE THE LOG
#   A judge, or an LLM detector, works through every sample in the run -- some
#   hundreds of them -- and the log used to say only which model had started.
#   A stage that takes forty minutes and prints nothing in between is
#   indistinguishable from a stage that has hung, and the only way to tell the
#   two apart was to look at the process's open sockets.
#
#   Counting every completion into the log is not the answer either: it would
#   cost one line per sample and bury the lines that carry meaning. So
#   completions go to a separate callback, which the web layer keeps as a
#   single "done of total" figure that the browser redraws in place, and only
#   every tenth of the way through is ALSO written to the log -- enough that
#   logs/<run_id>.log still shows the shape of the run once the process is
#   gone, and enough to follow a command-line run in a terminal.


def progress_snapshot(info: dict | None) -> dict | None:
    """A progress figure with its clock-derived fields brought up to date.

    A counter reports only facts that cannot go stale -- which model, how many
    done, how many there are, and when it started. Elapsed time and the
    estimate of what is left are derived here instead, every time the figure
    is read, because the read and the last completion can be far apart: a
    model stuck on its first sample would otherwise report a frozen count
    beside a frozen clock, and "how long has it been stuck" is the whole
    question being asked of it.

    Args:
        info: a dict from `Progress`, or None.

    Returns:
        None for None, otherwise a copy with `elapsed_seconds` and
        `eta_seconds` set. `eta_seconds` is None until there is a rate worth
        extrapolating from -- see the threshold below.
    """
    if not info:
        return None

    started = info.get("monotonic_start")
    if started is None:
        return dict(info)

    elapsed = max(0.0, time.monotonic() - started)
    done = info.get("done", 0)
    total = info.get("total", 0)

    # Withheld until a few items and a whole second are in: an estimate drawn
    # from one sample of a parallel pool, or from a stage whose first items
    # landed inside the clock's resolution, is wild enough to be worse than no
    # estimate at all. Also withheld at the end, where it would read "0s left"
    # beside a stage that is already writing its results.
    eta = None
    if 3 <= done < total and elapsed >= 1:
        eta = round((total - done) * elapsed / done)

    return {**info, "elapsed_seconds": round(elapsed), "eta_seconds": eta}


class Progress:
    """Thread-safe completion counter for one stage's per-sample work.

    One instance covers one unit of work that has a known size: a single judge
    over the manifest, or one detector over it. A stage that runs several in
    sequence builds one per model, so the label the browser shows is always the
    model actually in flight.

    Args:
        label: what is doing the work -- normally a model id, which is what
            makes a three-judge pass legible as it moves from one to the next.
        total: how many items there are. Zero is tolerated (an empty run) and
            simply means nothing is reported.
        log: optional logger for the every-tenth milestone lines. Pass None
            for a stage that already prints a line per item, so the two do not
            interleave.
        on_count: optional callable taking one dict, or None to clear. Called
            on every completion, so it must be cheap -- the web layer stores
            the dict and returns. Exceptions from it are swallowed: a closed
            browser connection must not take down the thread doing the work.
            The dict carries no elapsed time or estimate; pass it through
            `progress_snapshot` when displaying it, and see that function for
            why those are derived at the point of reading.
        unit: plural noun for the items, used in the reported text.
    """

    def __init__(self, label: str, total: int, log=None, on_count=None,
                 unit: str = "samples") -> None:
        self.label = label
        self.total = max(0, int(total))
        self.unit = unit
        self.done = 0
        self._log = log
        self._on_count = on_count
        self._started = time.monotonic()
        # Milestone spacing for the log. One tenth of the way, with a floor of
        # 1 so a run of three samples still reports rather than dividing to 0.
        self._step = max(1, self.total // 10)
        self._next_log = self._step
        # Guards the counter. The thread pools call tick() concurrently, and
        # two threads incrementing at once would lose a completion.
        self._lock = threading.Lock()
        self._report()

    def _report(self) -> None:
        """Push the current count to the callback. Caller holds the lock."""
        if self._on_count is None:
            return
        try:
            self._on_count({
                "label": self.label,
                "unit": self.unit,
                "done": self.done,
                "total": self.total,
                "monotonic_start": self._started,
            })
        except Exception:  # noqa: BLE001 - deliberately swallowed, see Args
            pass

    def tick(self, n: int = 1) -> int:
        """Record `n` completed items. Returns the new total done."""
        with self._lock:
            self.done += n
            # Not at the very end: the stage logs its own summary line there,
            # and "753/753 (100%)" immediately above it says nothing new.
            due = self._next_log <= self.done < self.total
            if due:
                self._next_log = ((self.done // self._step) + 1) * self._step
            self._report()
            if due and self._log is not None:
                percent = round(100 * self.done / self.total)
                self._log.info(
                    f"    {self.label}: {self.done}/{self.total} "
                    f"{self.unit} ({percent}%)"
                )
            return self.done

    def wrap(self, function):
        """Wrap a one-argument function so each call ticks when it returns.

        For `ThreadPoolExecutor.map`, which is how the judges and the LLM
        detectors work through the manifest. The tick is in a `finally` so a
        failed item still counts: it is finished either way, and `map`
        re-raises when the result is consumed, so the count would otherwise
        stall at the first failure.
        """
        def counted(item):
            try:
                return function(item)
            finally:
                self.tick()
        return counted

    def close(self) -> None:
        """Clear the live counter, leaving the stage's own summary to speak.

        Called when the unit of work is done. Without it the browser would
        keep showing a full bar for the previous model while the next one
        loads, which reads as progress that has stalled.
        """
        if self._on_count is None:
            return
        try:
            self._on_count(None)
        except Exception:  # noqa: BLE001 - see Args
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
