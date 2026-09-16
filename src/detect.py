"""Stage 3 of the pipeline -- run the tools under test.

Takes the samples that survived judging and runs any selection of detectors
over them, writing one result file per detector into the run directory.

WHY THIS RUNS ON THE VERIFIED MANIFEST
    Stage 2 dropped every sample whose label the judges could not confirm. The
    tools are therefore scored only against labels that a model which did not
    write the code has agreed with. Running them on the raw generated manifest
    instead would reintroduce exactly the "ground truth comes from the
    generator" weakness the judging stage exists to remove.

    `runstore.Run.read_verified_manifest()` falls back to the generation
    manifest when judging has not been run, so an unjudged run can still be
    analysed -- useful for measuring how much the judging stage changes the
    numbers, but it should not be the headline configuration.

FAILURE POLICY
    A detector that cannot run -- a missing executable, an unreachable API, a
    dataset too small to train on -- is recorded as failed and the remaining
    detectors still run. One broken arm must not cost the whole comparison,
    especially since the ML arm legitimately cannot run on very small runs.

Inputs:
    runs/<run_id>/02_judgement/verified_manifest.json
Outputs:
    runs/<run_id>/03_detection/detection_<tool>.json
    logs/<run_id>.log

Run with:
    python src/detect.py <run_id> [tool_key ...]
"""

from __future__ import annotations

import sys
import time
import traceback

import runstore
from common import get_logger, write_json
from detectors import DETECTOR_REGISTRY
from detectors.llm_tools import run_llm_detector
from detectors.ml_models import InsufficientData, run_embedding, run_tfidf
from detectors.static_tools import run_bandit, run_ruff, run_semgrep

# Maps a registry key to the function that runs it.
#
# Kept separate from DETECTOR_REGISTRY so that module -- which the web layer
# imports just to render the Tools panel -- does not have to import scikit-learn
# and the HTTP clients. Importing this module is what pulls in the heavy
# dependencies, and only the worker thread does that.
RUNNERS = {
    "static_bandit": run_bandit,
    "static_ruff": run_ruff,
    "static_semgrep": run_semgrep,
    "ml_tfidf": run_tfidf,
    "ml_embedding": run_embedding,
    "llm_detector": run_llm_detector,
}

# What runs when the caller does not choose. All three static analysers plus
# one tool from each other arm, so a default run always produces the full
# cross-paradigm comparison -- and, within the static arm, lets a rule's limits
# be told apart from a tool's.
DEFAULT_TOOLS = [
    "static_bandit", "static_ruff", "static_semgrep",
    "ml_tfidf", "llm_detector",
]


def run_detection(
    run_id: str,
    tools: list[str] | None = None,
    llm_provider: str | None = None,
    llm_models: list[str] | None = None,
    on_progress=None,
) -> dict:
    """Run the selected detectors over a run's verified samples.

    Args:
        run_id: which run to analyse.
        tools: registry keys to run. Defaults to `DEFAULT_TOOLS`.
        llm_provider: backend for the LLM arm. None (the default) resolves
            it per model via providers.provider_for_model(), which is what
            allows one selection list to mix remote and local models.
        llm_models: one or more model ids for the LLM arm. Each produces its
            own result file, so several LLMs can be compared in one table.
        on_progress: callable taking one string; receives every log line.

    Returns:
        A summary dict listing which detectors succeeded and which failed.

    Raises:
        SystemExit: if the run has no samples to analyse.
    """
    run = runstore.load_run(run_id)
    log = get_logger(run_id, on_progress)

    manifest = run.read_verified_manifest()
    if not manifest:
        raise SystemExit(
            f"Run {run_id} has no samples to analyse. Generate and judge first."
        )

    tools = tools or list(DEFAULT_TOOLS)
    unknown = [t for t in tools if t not in RUNNERS]
    if unknown:
        raise SystemExit(f"Unknown detector(s): {', '.join(unknown)}")

    judged = run.verified_manifest_file.exists()

    log.info("")
    log.info("=== Stage 3: detection ===")
    log.info(
        f"Running {len(tools)} detector(s) over {len(manifest)} "
        f"{'verified' if judged else 'unverified'} samples"
    )
    if not judged:
        log.warning(
            "This run has not been judged, so detectors are being scored "
            "against unverified generator labels."
        )

    succeeded: list[str] = []
    failed: list[dict] = []

    for tool_key in tools:
        started = time.time()
        try:
            if tool_key == "llm_detector":
                # The LLM arm expands into one detector per selected model, so
                # several models can appear side by side in one comparison.
                for model in (llm_models or [None]):
                    result = run_llm_detector(
                        run, manifest, log,
                        provider=llm_provider, model=model,
                    )
                    result["runtime_seconds"] = round(time.time() - started, 2)
                    write_json(
                        run.detection_file(result["detector_key"]), result
                    )
                    succeeded.append(result["detector_key"])
                    started = time.time()
            else:
                result = RUNNERS[tool_key](run, manifest, log)
                result["runtime_seconds"] = round(time.time() - started, 2)
                write_json(run.detection_file(tool_key), result)
                succeeded.append(tool_key)

        except InsufficientData as exc:
            # An expected outcome on small runs, not a defect -- logged at
            # warning level rather than as an error, with the reason preserved.
            log.warning(f"  {tool_key} skipped: {exc}")
            failed.append({
                "tool": tool_key, "reason": "insufficient_data",
                "detail": str(exc),
            })

        except Exception as exc:  # noqa: BLE001 - one arm must not kill the run
            log.warning(f"  {tool_key} failed: {exc}")
            # Full traceback to the run log only; the browser sees the message.
            log.info(traceback.format_exc())
            failed.append({
                "tool": tool_key, "reason": "error", "detail": str(exc)[:500],
            })

    run.update_metadata(
        status="detected",
        detectors_run=succeeded,
        detectors_failed=[f["tool"] for f in failed],
        detection_sample_count=len(manifest),
    )

    log.info(
        f"Detection complete: {len(succeeded)} succeeded, {len(failed)} failed"
    )

    return {
        "run_id": run_id,
        "samples": len(manifest),
        "succeeded": succeeded,
        "failed": failed,
    }


def available_tools() -> dict[str, bool]:
    """Report which detectors can actually run in this environment.

    Used by the web interface to disable a tool rather than let a user start a
    job that is certain to fail. Imports are deliberately local so that a
    missing optional dependency degrades one entry instead of breaking the
    whole page.
    """
    status: dict[str, bool] = {}

    try:
        from detectors.static_tools import tool_available
        status["static_bandit"] = tool_available("bandit")
        status["static_ruff"] = tool_available("ruff")
        status["static_semgrep"] = tool_available("semgrep")
    except Exception:  # noqa: BLE001
        status["static_bandit"] = status["static_ruff"] = False
        status["static_semgrep"] = False

    try:
        import sklearn  # noqa: F401
        status["ml_tfidf"] = True
    except ImportError:
        status["ml_tfidf"] = False

    # The embedding detector needs both scikit-learn and a reachable endpoint;
    # only the credential presence is checked here, because probing the network
    # on every page load would make the interface sluggish.
    try:
        from common import credential_status
        creds = credential_status()
        status["ml_embedding"] = (
            status["ml_tfidf"] and creds["judgment_url"] and creds["judgment_key"]
        )
        status["llm_detector"] = creds["judgment_url"] and creds["judgment_key"]
    except Exception:  # noqa: BLE001
        status["ml_embedding"] = status["llm_detector"] = False

    return status


def main() -> None:
    """Run detection from the command line.

    Usage:
        python src/detect.py                       most recent run, default tools
        python src/detect.py <run_id>              that run, default tools
        python src/detect.py <run_id> static_bandit ml_tfidf
    """
    args = sys.argv[1:]

    if args:
        run_id, tools = args[0], (args[1:] or None)
    else:
        latest = runstore.latest_run()
        if latest is None:
            raise SystemExit("No runs found. Run 'python src/generate.py' first.")
        run_id, tools = latest.run_id, None
        print(f"No run id given; using the most recent: {run_id}")

    summary = run_detection(run_id, tools=tools, on_progress=print)
    if summary["failed"]:
        for failure in summary["failed"]:
            print(f"  ! {failure['tool']}: {failure['detail'][:200]}")


if __name__ == "__main__":
    main()
