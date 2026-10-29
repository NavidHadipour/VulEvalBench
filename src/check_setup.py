"""Verify that a machine is ready to run VulEvalBench.

Run this first on any new machine. It checks every prerequisite the pipeline
depends on and, for each one that is missing, says exactly what to do -- so a
failed run is diagnosed before it starts rather than halfway through.

    python src/check_setup.py

Exit code is 0 when every required check passes, 1 otherwise, so it can also
gate a CI job or a container health check.

WHAT IS CHECKED, AND WHY EACH ONE MATTERS
    Python version      the code uses `match`-free 3.10+ syntax (X | Y unions).
    Python packages     each detector arm needs its own libraries; a missing
                        one disables that arm silently at run time.
    Pretrained weights  the ML arm runs published models that are downloaded
                        rather than installed; without them that arm is dead.
    Static analysers    Bandit, Ruff and Semgrep are executables installed into
                        the virtualenv; pip reports success even when semgrep
                        installs no runnable binary on old versions.
    Credential files    the judgment API is the one external service the
                        pipeline cannot run without.
    Ollama              generation needs the local server up AND the configured
                        model tag already pulled.
    CWE hierarchy       the cached MITRE data; regenerable but needed offline.
    Writable dirs       runs/ and logs/ must be creatable where the code lives.
"""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

# Make src/ importable when run as `python src/check_setup.py` from the root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

OK, WARN, FAIL = "ok", "warn", "fail"
_results: list[tuple[str, str, str]] = []


def report(status: str, name: str, detail: str = "") -> None:
    """Record one check. Rendered as a table at the end."""
    _results.append((status, name, detail))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_python() -> None:
    """Python 3.10+ is required for the `X | Y` union syntax used throughout."""
    v = sys.version_info
    if v >= (3, 10):
        report(OK, "Python", f"{v.major}.{v.minor}.{v.micro}")
    else:
        report(FAIL, "Python",
               f"{v.major}.{v.minor} found; 3.10 or newer is required")


def check_packages() -> None:
    """Import each library an arm depends on; a missing one disables that arm."""
    required = {
        "requests": "pip install requests",
        "flask": "pip install flask",
        "sklearn": "pip install scikit-learn",
        "numpy": "pip install numpy",
        "pyflakes": "pip install pyflakes",
        "tf_keras": "pip install tensorflow-cpu tf-keras   (pretrained ML arm)",
    }
    optional = {
        "anthropic": "only needed for Claude models: pip install anthropic",
        "laya": "only needed for the Laya typed model, as judge or "
                "detector: pip install laya",
    }
    for module, fix in required.items():
        try:
            importlib.import_module(module)
            report(OK, f"package: {module}")
        except ImportError:
            report(FAIL, f"package: {module}", fix)
    for module, note in optional.items():
        try:
            importlib.import_module(module)
            report(OK, f"package: {module}", "(optional)")
        except ImportError:
            report(WARN, f"package: {module}", note)


def check_static_tools() -> None:
    """Locate bandit, ruff and semgrep next to the interpreter, as the pipeline does."""
    # Resolve next to the running interpreter, exactly as static_tools.py does,
    # so this check agrees with what the pipeline will actually find.
    bin_dir = Path(sys.executable).parent
    for tool in ("bandit", "ruff", "semgrep"):
        found = next(
            (p for p in (bin_dir / f"{tool}.exe", bin_dir / tool) if p.exists()),
            None,
        ) or shutil.which(tool)
        if found:
            report(OK, f"tool: {tool}", str(found))
        else:
            hint = f"pip install {tool}"
            if tool == "semgrep":
                hint += "   (needs >= 1.173 for a Windows binary)"
            report(FAIL, f"tool: {tool}", hint)


def check_credentials() -> None:
    """Confirm the judgment API URL and key were found; the rest are optional."""
    import common
    status = common.credential_status()
    if status["judgment_url"] and status["judgment_key"]:
        report(OK, "credentials: judgment API", "URL and key found")
    else:
        missing = [k for k in ("judgment_url", "judgment_key") if not status[k]]
        report(FAIL, "credentials: judgment API",
               f"missing {', '.join(missing)} -- copy api-key-judgment.txt.example "
               "to api-key-judgment.txt and fill it in")
    if status["anthropic_key"]:
        report(OK, "credentials: Anthropic", "(optional)")
    else:
        report(WARN, "credentials: Anthropic",
               "not set; Claude models unavailable (optional)")
    if status["typesafe_key"]:
        report(OK, "credentials: TypeSafe (Jev)", "(optional)")
    else:
        report(WARN, "credentials: TypeSafe (Jev)",
               "not set; Jev is unavailable as a judge and as a detector. "
               "Jev is in early access -- put the key in "
               "api-key-typesafe.txt (optional)")


def check_ollama() -> None:
    """Confirm the local server answers and the configured generator tag is pulled."""
    import config
    import providers
    if not providers.ollama_available():
        report(FAIL, "Ollama server",
               f"nothing answering at {config.OLLAMA_BASE_URL} -- "
               "install Ollama and run: ollama serve")
        report(FAIL, "Ollama model", "(server down, cannot check)")
        return
    report(OK, "Ollama server", config.OLLAMA_BASE_URL)
    models = providers.list_ollama_models()
    if config.GENERATOR_MODEL in models:
        report(OK, "Ollama model", config.GENERATOR_MODEL)
    else:
        report(FAIL, "Ollama model",
               f"'{config.GENERATOR_MODEL}' not pulled -- run: "
               f"ollama pull {config.GENERATOR_MODEL}"
               + (f"   (pulled: {', '.join(models)})" if models else ""))


def check_pretrained_models() -> None:
    """Confirm the downloaded ML weights are present and loadable.

    Reported per family. `ml_vudenc` is expected to fail on most machines --
    its embedding array was never published -- so it is a warning rather than
    a failure: the arm still runs with the Passau model alone.
    """
    import config
    from common import MODELS_DIR

    try:
        from detectors.ml_models import family_available
    except Exception as exc:  # noqa: BLE001
        report(FAIL, "pretrained ML models", f"cannot import detector: {exc}")
        return

    if family_available("ml_farasat"):
        report(OK, "pretrained: BiLSTM (Passau)",
               str(MODELS_DIR / config.VUDENC_FARASAT_DIR))
    else:
        report(FAIL, "pretrained: BiLSTM (Passau)",
               "weights missing -- run: python src/fetch_models.py")

    if config.VUDENC_USE_FARASAT_WEIGHTS:
        # Reporting "ok" against the VUDENC directory would be a lie: the slot
        # loads Farasat's files and duplicates the detector above it.
        report(WARN, "pretrained: LSTM (VUDENC)",
               "SUBSTITUTED -- running on Farasat's weights and word2vec "
               f"from {MODELS_DIR / config.VUDENC_FARASAT_DIR}. Duplicates "
               "ml_farasat; not a VUDENC measurement. Set "
               "VUDENC_USE_FARASAT_WEIGHTS = False to restore the real check")
    elif family_available("ml_vudenc"):
        report(OK, "pretrained: LSTM (VUDENC)",
               str(MODELS_DIR / config.VUDENC_ORIGINAL_DIR))
    else:
        report(WARN, "pretrained: LSTM (VUDENC)",
               "unavailable -- its word2vec vector array was never published "
               "by the authors; see src/fetch_models.py (expected)")


def check_cwe_hierarchy() -> None:
    """Confirm the cached MITRE hierarchy exists; without it matching is exact-only."""
    import cwe_taxonomy
    h = cwe_taxonomy.load_hierarchy()
    if h.get("children"):
        report(OK, "CWE hierarchy", f"MITRE catalogue v{h.get('version')}")
    else:
        report(WARN, "CWE hierarchy",
               "not built; exact CWE matching only. Run: python src/cwe_taxonomy.py")


def check_writable() -> None:
    """Confirm runs/ and logs/ can be created and written where the code lives."""
    import common
    for d in (common.RUNS_DIR, common.LOGS_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            report(OK, f"writable: {d.name}/")
        except OSError as exc:
            report(FAIL, f"writable: {d.name}/", str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run every check, print the checklist, and return 0 only if none failed."""
    for check in (check_python, check_packages, check_static_tools,
                  check_credentials, check_ollama, check_pretrained_models,
                  check_cwe_hierarchy, check_writable):
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - one broken check must not hide the rest
            report(FAIL, check.__name__, f"check itself crashed: {exc}")

    symbol = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]"}
    width = max(len(name) for _, name, _ in _results) + 2

    print()
    print("VulEvalBench setup check")
    print("-" * 72)
    for status, name, detail in _results:
        print(f"  {symbol[status]}  {name:<{width}} {detail}")
    print("-" * 72)

    failures = sum(1 for s, _, _ in _results if s == FAIL)
    warnings = sum(1 for s, _, _ in _results if s == WARN)
    if failures:
        print(f"  {failures} required check(s) failed -- fix the [FAIL] lines above.")
        return 1
    print(f"  All required checks passed"
          + (f" ({warnings} optional item(s) missing)." if warnings else "."))
    print("  Next: python web/app.py   then open http://localhost:8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
