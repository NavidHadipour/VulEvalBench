"""Add, remove, hide or list models, per installation.

The framework ships the models this thesis was measured with. It does not ship
anyone else's endpoint or keys, so every installation needs its own list. This
is the command-line face of `model_registry`; the web interface has the same
thing on its /models page, and both write the same gitignored
`models.local.json`.

Nothing here edits config.py. A shipped model is never deleted -- `remove`
hides it -- so an old run's `config_snapshot` still resolves and a published
number stays explicable.

Usage:
    python src/manage_models.py list
    python src/manage_models.py list --role judge

    # a chat model on the university endpoint, usable as judge and detector
    python src/manage_models.py add --id qwen3-32b --name "Qwen3 32B" \\
        --provider passau --roles judge,llm_detector

    # an aggregator model whose id is namespaced, with a budget of its own.
    # The id stays filename-safe; --api-model is what goes on the wire.
    python src/manage_models.py add --id llama4-scout --name "Llama 4 Scout" \
        --provider nvidia --roles judge \
        --api-model meta/llama-4-scout-17b-16e-instruct --max-tokens 4000

    # a local Ollama tag for writing the benchmark pairs
    python src/manage_models.py add --id "codellama:13b" \\
        --name "CodeLlama 13B" --provider ollama --roles generator

    # a pretrained word2vec + recurrent family under models/
    python src/manage_models.py add --id mylstm --name "My LSTM" --roles ml \\
        --directory my-lstm --word2vec word2vec_withString10-100-200.model \\
        --dimensions 200

    python src/manage_models.py remove --id qwen3-32b
    python src/manage_models.py disable --id claude-haiku-4-5
    python src/manage_models.py enable --id claude-haiku-4-5

Credentials live in one file, api-keys.txt, and are managed with the same
script -- or on the Models page, which is the better route because a value
typed here lands in the shell history:

    python src/manage_models.py keys
    python src/manage_models.py keys --set anthropic.key --value sk-ant-...
    python src/manage_models.py keys --set openrouter.url \
        --value https://openrouter.ai/api
    python src/manage_models.py keys --delete openrouter
    python src/manage_models.py keys --import

What each model's output actually looks like is measured at run time rather
than configured -- which models pad their responses to the token cap, and how
long their useful answers are. See src/token_budgets.py:

    python src/manage_models.py budgets
    python src/manage_models.py budgets --reset gemma4-31b-it
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import model_registry


def _flag(args: list[str], name: str, default: str = "") -> str:
    """Read `--name value`, returning `default` when absent."""
    if name in args:
        index = args.index(name)
        if index + 1 < len(args):
            return args[index + 1]
    return default


def _print_list(role_filter: str = "") -> None:
    """Show what is available for each role, and where it came from."""
    data = model_registry.summary()
    print(f"Registry file: {data['file']}"
          f"{'' if data['exists'] else '  (not created yet)'}\n")

    roles = [role_filter] if role_filter else list(model_registry.ROLES)
    for role in roles:
        if role not in data["roles"]:
            print(f"Unknown role {role!r}; "
                  f"choose from {', '.join(model_registry.ROLES)}")
            continue
        block = data["roles"][role]
        print(f"{role}")
        if not block["available"]:
            print("    (none available)")
        for model_id, name in sorted(block["available"].items()):
            origin = "added" if model_id in block["added"] else "shipped"
            label = "" if name == model_id else f"  {name}"
            print(f"    {model_id:34} [{origin}]{label}")
        print()

    if data["disabled"]:
        print("hidden on this installation (re-show with `enable`):")
        for model_id in data["disabled"]:
            print(f"    {model_id}")
        print()

    errors = data["roles"].get("_errors") or {}
    for role, message in errors.items():
        print(f"NOTE: the {role} list could not be read ({message}).")


def _do_add(args: list[str]) -> int:
    """Register one model."""
    roles = [r.strip() for r in _flag(args, "--roles").split(",") if r.strip()]
    entry = {
        "id": _flag(args, "--id").strip(),
        "name": _flag(args, "--name").strip(),
        "roles": roles,
        "note": _flag(args, "--note").strip(),
    }

    if any(r != "ml" for r in roles):
        entry["provider"] = _flag(args, "--provider").strip()
        # Both optional, and set only when given: an absent flag must not be
        # stored as a deliberate empty setting.
        if _flag(args, "--api-model"):
            entry["api_model"] = _flag(args, "--api-model").strip()
        if _flag(args, "--max-tokens"):
            entry["max_tokens"] = _flag(args, "--max-tokens").strip()
    if "ml" in roles:
        entry["directory"] = _flag(args, "--directory").strip()
        entry["word2vec"] = _flag(args, "--word2vec").strip()
        entry["dimensions"] = _flag(args, "--dimensions").strip()
        if _flag(args, "--citation"):
            entry["citation"] = _flag(args, "--citation").strip()

    problems = model_registry.add(entry, replace="--replace" in args)
    blocking = [p for p in problems if not p.startswith("note:")]
    for problem in problems:
        print(("  " if problem.startswith("note:") else "ERROR: ") + problem)
    if blocking:
        return 1

    print(f"Registered {entry['id']} for: {', '.join(roles)}")
    print("It will appear in the relevant menus on the next page load.")
    return 0


def _do_budgets(args: list[str]) -> int:
    """`budgets` -- inspect or clear what the padding guard has learned.

        budgets                     show every model's measured output shape
        budgets --reset             forget everything
        budgets --reset <model>     forget one model

    Resetting matters when a model is upgraded behind the same id: the old
    evidence would otherwise keep a budget capped for a model that no longer
    pads, or keep reduction disabled for one that now does.
    """
    import token_budgets

    if "--reset" in args:
        index = args.index("--reset")
        model = args[index + 1] if len(args) > index + 1 else None
        token_budgets.reset(model)
        print(f"Forgot {model or 'every model'}.")
        return 0

    state = token_budgets.report()
    if not state:
        print("Nothing measured yet. The guard records a model's output shape "
              "the first time a run calls it.")
        return 0

    print(f"{'model':30s} {'budget':>7s} {'calls':>6s} {'padded':>7s} "
          f"{'peak':>6s}  note")
    for key, entry in sorted(state.items()):
        if key == "__accounting_errors__":
            continue
        model, _, requested = key.rpartition("@")
        note = []
        if entry.get("padded"):
            note.append(f"capped at {token_budgets.budget_for(model, int(requested))}")
        if entry.get("shrink_disabled"):
            note.append("reduction disabled")
        if entry.get("recovered"):
            note.append(f"{entry['recovered']} recovered")
        print(f"{model:30s} {requested:>7s} {entry.get('calls', 0):>6d} "
              f"{entry.get('padded', 0):>7d} "
              f"{entry.get('peak_useful_tokens', 0):>6d}  "
              f"{', '.join(note)}")
    for line in token_budgets.summary_lines():
        print(f"\n{line}")
    return 0


def _do_keys(args: list[str]) -> int:
    """`keys` -- inspect and edit the unified credential store.

    The same operations the Models page offers, for a headless install:

        keys                                   list, masked
        keys --set <name>.<field> --value <v>  add or replace
        keys --delete <name>[.<field>]         remove
        keys --import                          fold in the legacy files

    A value is taken from the command line, so on a shared machine it lands
    in the shell history. The page is the better route where there is one.
    """
    if "--import" in args:
        for line in common.import_legacy_keys():
            print(line)
        return 0

    target = _flag(args, "--set").strip()
    if target:
        name, _, field = target.partition(".")
        value = _flag(args, "--value").strip()
        problems = common.set_key(name, field, value)
        for problem in problems:
            print(f"ERROR: {problem}")
        if problems:
            return 1
        print(f"Saved {name.lower()}.{field.lower()} to "
              f"{common.KEYS_FILE.name}")
        return 0

    target = _flag(args, "--delete").strip()
    if target:
        name, _, field = target.partition(".")
        problems = common.delete_key(name, field or None)
        blocking = [p for p in problems if not p.startswith("note:")]
        for problem in problems:
            print(("  " if problem.startswith("note:") else "ERROR: ")
                  + problem)
        if blocking:
            return 1
        print(f"Removed {target}.")
        return 0

    rows = common.keys_summary()
    if not rows:
        print("No credentials configured.")
        print("Add one with: keys --set anthropic.key --value sk-ant-...")
        return 0

    print(f"{'NAME':<16} {'FIELD':<6} {'VALUE':<44} SOURCE")
    for row in rows:
        flag = "  (shipped)" if row["reserved"] else ""
        print(f"{row['name']:<16} {row['field']:<6} {row['masked']:<44} "
              f"{row['source']}{flag}")

    extra = common.custom_endpoints()
    if extra:
        print("\nSelectable as providers: " + ", ".join(sorted(extra)))
    return 0


def main() -> None:
    """Dispatch one subcommand."""
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__)
        raise SystemExit(0)

    command, rest = args[0], args[1:]

    if command == "list":
        _print_list(_flag(rest, "--role").strip())
        raise SystemExit(0)

    if command == "add":
        raise SystemExit(_do_add(rest))

    if command == "keys":
        raise SystemExit(_do_keys(rest))

    if command == "budgets":
        raise SystemExit(_do_budgets(rest))

    if command in ("remove", "disable", "enable"):
        model_id = _flag(rest, "--id").strip()
        if not model_id:
            raise SystemExit(f"{command} needs --id <model>")
        action = {
            "remove": model_registry.remove,
            "disable": model_registry.remove,
            "enable": model_registry.enable,
        }[command]
        problems = action(model_id)
        for problem in problems:
            print(f"ERROR: {problem}")
        if problems:
            raise SystemExit(1)
        verb = "re-enabled" if command == "enable" else "removed or hidden"
        print(f"{model_id} {verb}.")
        raise SystemExit(0)

    raise SystemExit(
        f"Unknown command {command!r}.\n"
        "Use: list | add | remove | disable | enable | keys | budgets  "
        "(or --help for examples)"
    )


if __name__ == "__main__":
    main()
