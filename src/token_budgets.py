"""Defends the pipeline against models that pad their output to the cap.

THE FAILURE THIS EXISTS FOR
    Grammar-guided decoding constrains *which* token may come next, but it
    cannot force a model to pick a productive one. Between a value and the
    next structural token a JSON grammar accepts unlimited whitespace, so a
    model that is unsure what to write next can sit in that position emitting
    newlines until its token budget runs out. The grammar never objects,
    because whitespace is legal there.

    Measured on `gemma4-31b-it` against the judgment endpoint, with the
    verdict schema and a 4000-token budget:

        {
          "predicted_label": "not_vulnerable",
          "predicted_cwe": null,
          "vulnerability_count": 0,
          "confidence_percent": 100
          \\n  \\n  \\n  \\n  ...            <- 3960 tokens of this

    Four of five fields in about 40 tokens, then a whitespace loop instead of
    starting `"reasoning"`. The object is never closed, so
    `providers.repair_truncated_json` salvages it and the call *succeeds* --
    which is exactly why this hid for so long. The symptoms it produced
    instead:

      - 48 seconds per call where 0.5 would do;
      - calls that cross the 180-second HTTP read timeout and are lost
        entirely. On one 753-sample run, 391 judgments failed this way and
        over half the dataset was labelled by two judges instead of three;
      - an empty `reasoning` on every salvaged verdict, because the field the
        loop replaced is the one the repair cannot invent.

    A 753-sample judging pass took 12 h 27 m. The useful work in it was
    minutes.

WHY THIS IS NOT A PER-MODEL WORKAROUND
    This framework is meant to be pointed at whatever models an installation
    has, including ones that did not exist when it was written. A hardcoded
    "gemma pads, give it 600 tokens" entry would be wrong for the next model
    that does this and useless to anyone with a different set. So nothing here
    names a model. Both defences key off the *shape of the response*, which is
    observable for any model on any backend:

    1. STOP SEQUENCES, always on (`stop_sequences`). A long run of literal
       newlines cannot be legitimate output from a grammar-guided call: inside
       a JSON string a newline must be escaped as the two characters `\\n`, so
       real whitespace only ever appears between structural tokens, where no
       formatter emits more than one. Six in a row means the model is looping.
       Cutting generation there costs nothing and, crucially, protects the
       FIRST run -- the one that would otherwise lose half its samples to
       timeouts before anything could be learned.

    2. A LEARNED BUDGET (`budget_for`), for the loops stop strings miss --
       a backend that ignores `stop`, or padding built from some other legal
       character. After `PADDING_CONFIRM` confirmed padding events a model's
       budget is capped at a multiple of the longest useful response ever seen
       from it at that call size, so the loop burns seconds instead of
       minutes.

    Both are safe under repair: a cut-short object loses only its trailing
    fields, which for a verdict is `reasoning` and nothing that is scored. For
    generation a short pair fails the incomplete-response check and is
    rejected cleanly, which is the existing behaviour for a runaway.

THE SHRINK IS SELF-CORRECTING, WHICH IT HAS TO BE
    Capping a model's budget on evidence risks truncating a model that
    genuinely needed the room. So `observe` watches for the opposite
    signature: a call that exhausted a *reduced* budget and ended on real
    content rather than whitespace. That is a model being cut off, not a model
    looping, and it permanently disables shrinking for that call shape and
    says so in the run log. Guessing wrong therefore costs one sample and
    fixes itself, rather than quietly damaging a whole run.

STATE LIVES WITH THE INSTALLATION
    What a given model does is a property of that model and that endpoint, not
    of this project, so it is remembered in `budgets.local.json` beside
    `models.local.json` and excluded from version control for the same reason.
    A second run of a known-padding model starts already defended.
"""

from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, active_logger, read_json, write_json

STATE_FILE = PROJECT_ROOT / "budgets.local.json"

# ---------------------------------------------------------------------------
# Stop sequences
# ---------------------------------------------------------------------------
# Literal strings, because that is all a `stop` parameter accepts -- there is
# no regex on either the vLLM or the Ollama side, so the shapes a loop is
# actually built from have to be spelled out. These three cover every padding
# run observed: bare newlines, and newlines followed by two- or four-space
# indentation, which is what a model emits when it is pretty-printing and then
# stalls at the indent.
#
# The repeat counts are the smallest that cannot occur in real output. A JSON
# formatter puts at most one newline between tokens, so six consecutive ones
# are already impossible; the indented forms get eight, since each repetition
# carries fewer newlines and the extra length still only costs a dozen tokens
# before the stop fires.
#
# Deliberately absent: a run of plain spaces or tabs with no newline. Deep
# indentation in a nested schema could reach one legitimately, and this list
# must be safe for schemas this project has never seen.
PADDING_STOPS = (
    "\n" * 6,
    "\n  " * 8,
    "\n    " * 8,
)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# Trailing whitespace this long, on a response that exhausted its budget, is a
# loop rather than formatting. Set well above anything an indent could produce
# so the classification never fires on a merely verbose model.
PADDING_MIN_FILLER_CHARS = 200

# Confirmed padding events before the budget is reduced. Two rather than one
# because a single stalled response could be a fluke of one awkward sample,
# and the cost of waiting is two slow calls, once, ever.
PADDING_CONFIRM = 2

# The learned cap is this multiple of the longest useful response seen. Three
# leaves room for a sample that needs noticeably more than the longest so far
# without giving the loop back the minutes it was wasting.
BUDGET_MARGIN = 3

# Floor for a learned cap, so a model whose useful responses have all been
# tiny still gets room for a complete object with a real explanation in it.
MIN_LEARNED_BUDGET = 384

# Used only when a backend reports no token usage, to turn characters into an
# estimated token count. Deliberately low: it overestimates tokens, which
# makes the resulting cap generous rather than tight.
CHARS_PER_TOKEN = 3.0


_LOCK = threading.Lock()
_STATE: dict | None = None

# Observations recorded since the last write. A judging pass makes one call
# per sample per judge, so saving on every one would mean hundreds of writes
# of the same few numbers; saving on none would leave the file's counters
# stale enough to mislead anyone reading it. Transitions that change a
# decision are always written immediately, regardless of this.
_UNSAVED = 0
SAVE_EVERY = 10

# Observations that raised inside the accounting. Kept rather than discarded so
# `report()` can show that the guard stopped measuring -- otherwise a broken
# detector is indistinguishable from a well-behaved set of models. A list of
# one, trimmed in place, because a module global cannot be rebound from the
# function that writes it without more ceremony than the message is worth.
_ACCOUNTING_ERRORS = 0
_LAST_ACCOUNTING_ERROR: list[str] = []


def _key(model: str, requested: int) -> str:
    """Identify one call shape: a model at a particular budget.

    The budget is part of the key because the same model is asked for a
    two-sentence verdict at one size and two complete programs at another, and
    the longest legitimate response differs by an order of magnitude between
    them. Keying on the model alone would let a verdict's modest ceiling
    truncate a generation.
    """
    return f"{model}@{requested}"


def _load() -> dict:
    """The persisted state, read once per process."""
    global _STATE
    if _STATE is None:
        try:
            loaded = read_json(STATE_FILE)
            _STATE = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            # Missing is the normal first-run case; corrupt is recoverable by
            # relearning. Neither is worth failing a run over.
            _STATE = {}
    return _STATE


_DEFAULTS = {
    "calls": 0,
    "padded": 0,
    "recovered": 0,
    "peak_useful_tokens": 0,
    "filler_tokens": 0,
    "shrink_disabled": False,
}


def _entry(state: dict, key: str) -> dict:
    """The record for one call shape, with any missing counter filled in.

    Defaults are merged rather than only applied to a new record, so a state
    file written by an older copy of this module -- which may be missing a
    counter added since -- is upgraded on read instead of raising.
    """
    entry = state.setdefault(key, {})
    for name, default in _DEFAULTS.items():
        entry.setdefault(name, default)
    return entry


def _save(state: dict) -> None:
    """Persist state, ignoring write failures.

    A read-only checkout or a locked file must not break a run: the defence
    still works in-process, it just will not be remembered for next time.
    """
    global _UNSAVED
    _UNSAVED = 0
    try:
        write_json(STATE_FILE, state)
    except OSError:
        pass


def flush() -> None:
    """Write out any observations the interval has not yet persisted.

    Worth calling at the end of a stage so the file's counters match what the
    run log reports. Not required for correctness: every change that affects a
    future decision is written when it happens.
    """
    with _LOCK:
        if _UNSAVED:
            _save(_load())


def _log(message: str) -> None:
    """Write to the active run log if there is one, else stay quiet."""
    logger = active_logger()
    if logger is not None:
        logger.warning(message)


def stop_sequences() -> list[str]:
    """Stop strings for a grammar-guided backend.

    Returned as a list because that is what both wire protocols want, and
    freshly each call so a backend cannot mutate the shared constant.
    """
    return list(PADDING_STOPS)


def split_filler(text: str) -> tuple[str, str]:
    """Split a response into its useful prefix and its trailing whitespace.

    A padding loop always sits at the end of the text, because the grammar is
    waiting for the next structural token and whitespace is the only thing it
    will accept until the model produces one. So the trailing run is the whole
    of the waste.
    """
    useful = text.rstrip()
    return useful, text[len(useful):]


def _useful_tokens(result, useful: str, text: str) -> int:
    """Estimate how many of a response's tokens carried content.

    Prorated by character count rather than re-tokenised, because the
    tokeniser that produced these counts lives on the server and this only
    needs to be right to within a factor that `BUDGET_MARGIN` already covers.
    """
    reported = getattr(result, "output_tokens", 0) or 0
    if reported and text:
        return max(1, math.ceil(reported * len(useful) / len(text)))
    return max(1, math.ceil(len(useful) / CHARS_PER_TOKEN))


def budget_for(model: str, requested: int) -> int:
    """The budget to actually send for this call.

    Returns `requested` unchanged for every model that has not been caught
    padding, which is the overwhelming majority: this function is a no-op
    until evidence says otherwise.
    """
    with _LOCK:
        state = _load()
        entry = state.get(_key(model, requested))
        if (entry is None
                or entry.get("shrink_disabled")
                or entry.get("padded", 0) < PADDING_CONFIRM):
            return requested
        cap = max(
            MIN_LEARNED_BUDGET,
            entry.get("peak_useful_tokens", 0) * BUDGET_MARGIN,
        )
        return min(requested, cap)


def observe(model: str, requested: int, sent: int, result) -> None:
    """Record what one response looked like and adapt.

    Args:
        model: the model that answered.
        requested: the budget the caller asked for -- the key's identity, and
            what is restored if shrinking turns out to be wrong.
        sent: the budget actually sent, which `budget_for` may have reduced.
        result: the `ChatResult`. Read defensively, because a user-added
            backend may not populate `meta` or the usage fields.

    Never raises. A defect in accounting must not fail a sample that the model
    answered perfectly well -- but it is counted rather than discarded, so a
    guard that has quietly stopped working is visible in `report()` instead of
    looking like a model that never pads.
    """
    global _ACCOUNTING_ERRORS
    try:
        _observe(model, requested, sent, result)
    except Exception as exc:  # noqa: BLE001 - see docstring
        _ACCOUNTING_ERRORS += 1
        _LAST_ACCOUNTING_ERROR.append(f"{type(exc).__name__}: {exc}")
        del _LAST_ACCOUNTING_ERROR[:-1]


def _observe(model: str, requested: int, sent: int, result) -> None:
    text = getattr(result, "raw_text", "") or ""
    meta = getattr(result, "meta", None) or {}
    finish = str(meta.get("finish_reason", ""))
    output_tokens = getattr(result, "output_tokens", 0) or 0
    repaired = bool(getattr(result, "data", None)
                    and result.data.get("__repaired__"))

    useful, filler = split_filler(text)
    # Some backends report no finish reason, so exhaustion is also inferred
    # from the token count reaching the budget that was sent.
    hit_cap = finish in ("length", "max_tokens") or (
        bool(sent) and output_tokens >= sent
    )
    is_padding = hit_cap and len(filler) >= PADDING_MIN_FILLER_CHARS
    useful_tokens = _useful_tokens(result, useful, text)

    with _LOCK:
        state = _load()
        entry = _entry(state, _key(model, requested))
        entry["calls"] += 1

        was_confirmed = entry["padded"] >= PADDING_CONFIRM
        peak_before = entry["peak_useful_tokens"]
        entry["peak_useful_tokens"] = max(peak_before, useful_tokens)
        notable = False

        if is_padding:
            entry["padded"] += 1
            entry["filler_tokens"] += max(0, output_tokens - useful_tokens)
            if entry["padded"] == PADDING_CONFIRM and not was_confirmed:
                notable = True
                cap = max(MIN_LEARNED_BUDGET,
                          entry["peak_useful_tokens"] * BUDGET_MARGIN)
                _log(
                    f"{model} pads its output to the token cap: "
                    f"{PADDING_CONFIRM} responses used their whole "
                    f"{requested}-token budget with only ~{useful_tokens} "
                    f"tokens of content, the rest trailing whitespace. "
                    f"Capping its budget at {min(requested, cap)} tokens for "
                    f"the rest of this run and remembering it in "
                    f"{STATE_FILE.name}. Verdicts are unaffected -- the "
                    f"object is recovered as before -- but a padded response "
                    f"loses its last field, so expect an empty reasoning on "
                    f"those samples."
                )
        elif hit_cap and sent < requested:
            # The opposite signature: a reduced budget ran out on real
            # content. This model was not looping, it was cut off.
            if not entry["shrink_disabled"]:
                entry["shrink_disabled"] = True
                notable = True
                _log(
                    f"{model} exhausted the reduced {sent}-token budget on "
                    f"real output, not whitespace, so the padding cap was "
                    f"wrong for it. Restoring the full {requested} tokens and "
                    f"not reducing this model again. One response was cut "
                    f"short; later ones are unaffected."
                )
        elif finish == "stop" and repaired:
            # Generation ended without closing the object and repair salvaged
            # it. Either a padding stop fired mid-loop or the model emitted
            # its end-of-turn token early -- the two are indistinguishable
            # from the response alone, since a stop string is excluded from
            # the output it halted. Counted together, because the consequence
            # is the same one an operator needs to know about: the response's
            # last field is missing. Not a failure, and not worth a log line
            # per sample; the run summary reports the total.
            entry["recovered"] += 1

        # Always persist a change that affects a later decision -- a verdict
        # on this model, or a materially higher ceiling. Everything else is
        # counters, written on an interval.
        # A decision change is written at once; plain counters wait for the
        # interval, so a judging pass does not rewrite the file per sample.
        global _UNSAVED
        _UNSAVED += 1
        decision_changed = (
            notable or entry["peak_useful_tokens"] > peak_before * 1.2
        )
        if decision_changed or _UNSAVED >= SAVE_EVERY:
            _save(state)


def report() -> dict:
    """Per-call-shape totals, for the run log and the run detail page.

    Carries a `__accounting_errors__` entry when any observation raised, so a
    guard that has stopped measuring cannot be mistaken for a clean result.
    """
    with _LOCK:
        out = {key: dict(entry) for key, entry in _load().items()}
    if _ACCOUNTING_ERRORS:
        out["__accounting_errors__"] = {
            "count": _ACCOUNTING_ERRORS,
            "last": _LAST_ACCOUNTING_ERROR[-1] if _LAST_ACCOUNTING_ERROR
            else "",
        }
    return out


def baseline() -> dict:
    """A counter snapshot, for reporting one stage's share of the totals.

    The state is process-wide and cumulative, so a stage that logs the raw
    totals claims its predecessor's responses as its own -- the detector
    reporting the judge's recovered verdicts, against a model that behaved
    perfectly for it. A stage takes this before its work and passes it to
    `summary_lines` after.
    """
    with _LOCK:
        return {key: dict(entry) for key, entry in _load().items()}


def summary_lines(since: dict | None = None) -> list[str]:
    """Human-readable lines describing every model caught padding.

    Args:
        since: a `baseline()` snapshot. Given one, only what happened after it
            is reported, which is what a per-stage log line should say.

    Returns:
        Empty when there is nothing to report, so a caller can log the list
        unconditionally without printing a heading for no content.
    """
    lines = []
    snapshot = report()
    prior = since or {}
    broken = snapshot.pop("__accounting_errors__", None)
    if broken:
        lines.append(
            f"the padding guard failed to measure {broken['count']} "
            f"response(s) and may not have adapted: {broken['last']}"
        )
    for key, entry in sorted(snapshot.items()):
        was = prior.get(key, {})
        padded = entry.get("padded", 0) - was.get("padded", 0)
        recovered = entry.get("recovered", 0) - was.get("recovered", 0)
        if not padded and not recovered:
            continue
        model, _, requested = key.rpartition("@")
        parts = []
        if padded:
            parts.append(f"{padded} response(s) ran to the cap")
        if recovered:
            parts.append(
                f"{recovered} response(s) ended early and were "
                f"recovered without their last field"
            )
        wasted = entry.get("filler_tokens", 0) - was.get("filler_tokens", 0)
        if wasted:
            parts.append(f"~{wasted:,} tokens of whitespace")
        # Reported only when this stage is the one that disabled it, so the
        # notice appears once rather than in every stage from then on.
        if entry.get("shrink_disabled") and not was.get("shrink_disabled"):
            parts.append("budget reduction disabled after a real truncation")
        lines.append(
            f"{model} at a {requested}-token budget: " + "; ".join(parts)
        )
    return lines


def reset(model: str | None = None) -> None:
    """Forget what was learned, for one model or for everything.

    Here because a model served behind the same id can be swapped or upgraded,
    and stale evidence would then cap a model that no longer pads.
    """
    with _LOCK:
        state = _load()
        if model is None:
            state.clear()
        else:
            for key in [k for k in state if k.rpartition("@")[0] == model]:
                del state[key]
        _save(state)
