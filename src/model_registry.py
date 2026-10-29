"""User-editable model registry: add or remove models without touching code.

WHY THIS EXISTS
    config.py ships the models this thesis was measured with, and that list is
    deliberately specific: it names a university endpoint, an Anthropic
    account, an early-access API and two local checkpoints. Nobody else has
    that combination. A reader who clones this framework has their own
    endpoint, their own keys, and no reason to care which Qwen tag Passau was
    serving in October 2026.

    Editing config.py to fix that is the wrong answer twice over. It mixes a
    local installation's settings into the file that records what the
    published numbers were produced with, and it means every upgrade is a
    merge conflict.

    So the shipped registries stay in config.py as the *defaults*, and this
    module holds a separate, per-installation overlay:

        models.local.json       gitignored; names local endpoints and tags

    The overlay can do two things: add a model the framework never heard of,
    and hide a shipped one the installation has no access to. Nothing is ever
    deleted from config.py, so a run's `config_snapshot` still resolves and
    old results remain explicable.

ROLES
    A model is registered for one or more of the four roles the pipeline has,
    because the requirements differ and a model good for one may be useless
    for another:

        generator      writes the benchmark pairs. Needs to emit long,
                       schema-valid JSON. Normally a local Ollama tag.
        judge          votes on whether a sample carries its stated label.
                       Needs guided JSON; see config.JUDGE_CONSENSUS.
        llm_detector   is scored as a tool under test.
        ml             a pretrained word2vec + recurrent model family, loaded
                       from a directory under models/. Data only -- see
                       detectors/ml_models.py `_FAMILIES`.

    A model registered as both `judge` and `llm_detector` is allowed and is
    normal; detect.py refuses the combination per *run*, not per registry, so
    a model cannot grade a sample it certified.

WHAT THIS DOES NOT DO
    It does not store API keys. Keys stay in the `api-key-*.txt` files that
    .gitignore excludes, and a registry entry whose key file is absent is
    listed as unavailable rather than hidden -- so a missing key reads as
    "not configured" instead of the model silently not existing.

    It does not make a new *architecture* work. An `ml` entry describes where
    to find weights for the pipeline that detectors/ml_models.py already
    implements (word2vec embedding -> recurrent net -> per-window score). A
    model that is not that shape needs a detector module, not a registry row.

Run with:
    python src/manage_models.py list
    python src/manage_models.py add --id mymodel --name "My Model" \\
        --provider passau --roles judge,llm_detector
    python src/manage_models.py remove --id mymodel
    python src/manage_models.py disable --id claude-haiku-4-5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from common import PROJECT_ROOT, read_json, write_json

REGISTRY_FILE = PROJECT_ROOT / "models.local.json"

# The four stages a model can be registered for. Order is pipeline order, and
# the web page and CLI both render them in it.
ROLES = ("generator", "judge", "llm_detector", "ml")

# Backends a registered API model can name. These are providers.PROVIDERS plus
# nothing: a registry entry may not invent a backend, because something has to
# know how to make the call.
API_PROVIDERS = ("passau", "anthropic", "systemone", "ollama")

# An `ml` entry is local weights, not an API call, so it carries no provider
# and instead must say where the weights are and what shape they have.
ML_REQUIRED_FIELDS = ("directory", "word2vec", "dimensions")

_EMPTY: dict = {"version": 1, "models": [], "disabled": []}


# --------------------------------------------------------------------------
# storage

def load() -> dict:
    """Read the overlay, returning an empty one if it does not exist.

    A corrupt file is reported and treated as empty rather than raising: the
    registry is a convenience layer, and a bad edit to it should not stop the
    benchmark from running with its shipped defaults.
    """
    if not REGISTRY_FILE.exists():
        return dict(_EMPTY)
    try:
        data = read_json(REGISTRY_FILE)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"WARNING: {REGISTRY_FILE.name} could not be read ({exc}); "
              "falling back to the shipped model list.", file=sys.stderr)
        return dict(_EMPTY)

    if not isinstance(data, dict):
        return dict(_EMPTY)
    data.setdefault("version", 1)
    data.setdefault("models", [])
    data.setdefault("disabled", [])
    if not isinstance(data["models"], list):
        data["models"] = []
    if not isinstance(data["disabled"], list):
        data["disabled"] = []
    return data


def save(data: dict) -> None:
    """Write the overlay back."""
    write_json(REGISTRY_FILE, data)


# --------------------------------------------------------------------------
# the shipped defaults, per role

def builtin_models(role: str) -> dict[str, str]:
    """The models config.py ships for one role, as {id: display name}.

    Ollama tags are deliberately absent. They are whatever happens to be
    pulled on this machine, so they are discovered at request time by
    web/app.py rather than being a fixed list -- and a tag that is not pulled
    should not appear in a menu.
    """
    if role == "generator":
        # The configured generator is the only one named in config; everything
        # else for this role is a locally pulled tag.
        return {config.GENERATOR_MODEL: config.GENERATOR_MODEL}
    if role in ("judge", "llm_detector"):
        return {
            **config.JUDGMENT_MODELS,
            **config.ANTHROPIC_MODELS,
            **config.SYSTEMONE_MODELS,
        }
    if role == "ml":
        # Imported lazily: ml_models imports tensorflow, which is slow and
        # unnecessary for anyone only listing or editing the registry.
        from detectors.ml_models import family_labels
        return family_labels()
    return {}


# --------------------------------------------------------------------------
# the merged view

def _entries(role: str | None = None) -> list[dict]:
    """Overlay entries, optionally filtered to one role."""
    out = []
    for entry in load()["models"]:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        if role and role not in (entry.get("roles") or []):
            continue
        out.append(entry)
    return out


def disabled_ids() -> set[str]:
    """Model ids the installation has hidden."""
    return {str(i) for i in load()["disabled"]}


def models_for(role: str) -> dict[str, str]:
    """Every model available for a role, as {id: display name}.

    Shipped defaults first, then the overlay's additions, then anything
    disabled is removed. Overlay entries win on name collision, so an
    installation can relabel a shipped model -- useful when the same id is
    served by a different endpoint locally.
    """
    merged = dict(builtin_models(role))
    for entry in _entries(role):
        merged[entry["id"]] = entry.get("name") or entry["id"]
    for model_id in disabled_ids():
        merged.pop(model_id, None)
    return merged


def provider_of(model_id: str) -> str | None:
    """The backend an overlay entry declares, or None if it is not registered.

    Consulted by providers.provider_for_model BEFORE the shipped registries,
    because an explicit declaration by whoever installed this beats inference
    from the shape of an id.
    """
    for entry in _entries():
        if entry["id"] == model_id:
            return entry.get("provider")
    return None


def ml_families() -> dict[str, dict]:
    """Overlay `ml` entries in the shape detectors/ml_models.py `_FAMILIES` wants.

    Returned keyed by detector key. The key is prefixed `ml_` if the
    registered id does not already start with it, so a registry row called
    `mymodel` becomes the detector `ml_mymodel` and cannot collide with a
    static or LLM tool key.
    """
    families: dict[str, dict] = {}
    for entry in _entries("ml"):
        key = entry["id"]
        if not key.startswith("ml_"):
            key = f"ml_{key}"
        families[key] = {
            "detector_key": key,
            "label": entry.get("name") or key,
            "directory": entry["directory"],
            "word2vec": entry["word2vec"],
            "dimensions": int(entry["dimensions"]),
            "citation": entry.get("citation", entry.get("note", "")),
            "user_registered": True,
        }
    return families


# --------------------------------------------------------------------------
# mutation

def validate(entry: dict) -> list[str]:
    """Check one proposed entry. Returns a list of problems, empty if valid.

    Returned rather than raised so a web form can show every problem at once
    instead of only the first.
    """
    problems: list[str] = []
    model_id = (entry.get("id") or "").strip()

    if not model_id:
        problems.append("id is required")
    elif any(c in model_id for c in ' \t"\'\\/'):
        # The id reaches a filename (detection_<key>.json) and a URL path, so
        # whitespace and separators are refused rather than sanitised.
        problems.append(
            "id must not contain spaces, quotes, slashes or backslashes"
        )

    roles = entry.get("roles") or []
    if not roles:
        problems.append(f"at least one role is required: {', '.join(ROLES)}")
    for role in roles:
        if role not in ROLES:
            problems.append(
                f"unknown role {role!r}; choose from {', '.join(ROLES)}"
            )

    if "ml" in roles:
        for field in ML_REQUIRED_FIELDS:
            if not entry.get(field):
                problems.append(f"an ml model needs {field}")
        if entry.get("dimensions"):
            try:
                int(entry["dimensions"])
            except (TypeError, ValueError):
                problems.append("dimensions must be a whole number")
        # Checked but not required to exist: weights are often downloaded
        # after registering, and refusing here would force an awkward order.
        directory = entry.get("directory")
        if directory and not (PROJECT_ROOT / "models" / directory).exists():
            problems.append(
                f"note: models/{directory} does not exist yet, so this "
                "detector will report itself unavailable until it is there"
            )

    # Known non-ml roles only: an unrecognised role has already been reported,
    # and demanding a provider for it as well is noise.
    api_roles = [r for r in roles if r != "ml" and r in ROLES]
    if api_roles:
        provider = entry.get("provider")
        if not provider:
            problems.append(
                f"a {'/'.join(api_roles)} model needs a provider: "
                f"{', '.join(API_PROVIDERS)}"
            )
        elif provider not in API_PROVIDERS:
            problems.append(
                f"unknown provider {provider!r}; choose from "
                f"{', '.join(API_PROVIDERS)}"
            )

    return problems


def add(entry: dict, replace: bool = False) -> list[str]:
    """Add or update one overlay entry.

    Args:
        entry: the model to register.
        replace: overwrite an existing entry with the same id. Without it, a
            duplicate id is an error, so a typo cannot silently rewrite a
            working entry.

    Returns:
        A list of problems; empty means it was written. A "note:" line is
        advisory and does not block the write.
    """
    problems = [p for p in validate(entry) if not p.startswith("note:")]
    notes = [p for p in validate(entry) if p.startswith("note:")]
    if problems:
        return problems

    data = load()
    existing = [e for e in data["models"]
                if isinstance(e, dict) and e.get("id") == entry["id"]]
    if existing and not replace:
        message = (f"{entry['id']} is already registered; "
                   "pass --replace to overwrite it")
        return [message]

    data["models"] = [e for e in data["models"]
                      if not (isinstance(e, dict)
                              and e.get("id") == entry["id"])]
    data["models"].append(entry)
    # Registering a model re-enables it: hiding and then adding the same id is
    # otherwise a silent no-op.
    data["disabled"] = [i for i in data["disabled"] if i != entry["id"]]
    save(data)
    return notes


def remove(model_id: str) -> list[str]:
    """Remove an overlay entry, or hide a shipped model.

    These are different operations with one name on purpose, because from the
    outside they have the same effect -- the model stops appearing in menus --
    and which one applies depends on something the caller should not need to
    know. A shipped model cannot be deleted; it is added to `disabled`.
    """
    data = load()
    before = len(data["models"])
    data["models"] = [e for e in data["models"]
                      if not (isinstance(e, dict)
                              and e.get("id") == model_id)]
    if len(data["models"]) < before:
        save(data)
        return []

    shipped = any(model_id in builtin_models(role)
                  for role in ROLES if role != "ml")
    if not shipped:
        try:
            shipped = model_id in builtin_models("ml")
        except Exception:                                    # noqa: BLE001
            # ml_models needs tensorflow; if it cannot load, fall back to
            # hiding the id anyway rather than refusing the request.
            shipped = True
    if shipped:
        if model_id not in data["disabled"]:
            data["disabled"].append(model_id)
            save(data)
        return []

    return [f"{model_id} is not registered and is not a shipped model"]


def enable(model_id: str) -> list[str]:
    """Un-hide a model that `remove` disabled."""
    data = load()
    if model_id not in data["disabled"]:
        return [f"{model_id} is not disabled"]
    data["disabled"] = [i for i in data["disabled"] if i != model_id]
    save(data)
    return []


def summary() -> dict:
    """Everything the CLI and the web page need to render the registry."""
    overlay = load()
    per_role = {}
    for role in ROLES:
        try:
            shipped = builtin_models(role)
        except Exception as exc:                             # noqa: BLE001
            # Only reachable for "ml", whose import pulls tensorflow.
            shipped = {}
            per_role.setdefault("_errors", {})[role] = str(exc)
        added = {e["id"]: (e.get("name") or e["id"])
                 for e in _entries(role)}
        per_role[role] = {
            "shipped": shipped,
            "added": added,
            "available": models_for(role),
        }
    return {
        "file": str(REGISTRY_FILE),
        "exists": REGISTRY_FILE.exists(),
        "roles": per_role,
        "disabled": sorted(disabled_ids()),
        "entries": overlay["models"],
    }
