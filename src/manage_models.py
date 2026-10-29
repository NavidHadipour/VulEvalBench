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
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
        "Use: list | add | remove | disable | enable  "
        "(or --help for examples)"
    )


if __name__ == "__main__":
    main()
