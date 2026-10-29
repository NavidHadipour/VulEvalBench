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
from detectors.ml_models import (InsufficientData, run_embedding,
                                 run_farasat, run_tfidf, run_vudenc)
from detectors.vudenc_pipeline import ModelUnavailable
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
    "ml_farasat": run_farasat,
    "ml_vudenc": run_vudenc,
    "ml_tfidf": run_tfidf,
    "ml_embedding": run_embedding,
    "llm_detector": run_llm_detector,
}

# Families registered in models.local.json get a runner built for them, so a
# model added through the /models page or manage_models.py is actually
# executable rather than merely listed. Shipped keys are never overwritten.
try:
    from detectors.ml_models import runner_for, user_family_keys

    for _key in user_family_keys():
        RUNNERS.setdefault(_key, runner_for(_key))
except Exception as _exc:                                    # noqa: BLE001
    # A broken overlay must not stop the shipped detectors from running.
    print(f"WARNING: registered ML detectors unavailable ({_exc})")

# What runs when the caller does not choose. All three static analysers, one
# tool from EACH ML family, and one LLM -- so a default run always produces the
# full cross-paradigm comparison. Within the static arm it lets a rule's limits
# be told apart from a tool's; within the ML arm it gives both a transfer test
# (pretrained) and a learnability probe (trained here) in the same table.
#
# ml_embedding is deliberately left out of the default: it is the only ML
# detector that costs an API call per sample, and ml_tfidf answers the same
# "what is learnable from these samples" question for free. Tick it explicitly
# when the lexical-versus-semantic comparison is the point.
DEFAULT_TOOLS = [
    "static_bandit", "static_ruff", "static_semgrep",
    "ml_farasat", "ml_tfidf", "llm_detector",
]


def run_detection(
    run_id: str,
    tools: list[str] | None = None,
    llm_provider: str | None = None,
    llm_models: list[str] | None = None,
    on_progress=None,
    should_stop=None,
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
        SystemExit: if the run has no samples to analyse, or if a model is
            selected as a detector on a run it also judged.
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

    # A MODEL MAY NOT BE GRADED ON A RUN IT HELPED CERTIFY.
    #   The judges decided which samples carry which label; this stage scores
    #   tools against those labels. A model on both sides is marking its own
    #   homework -- it is unusually good at finding the vulnerability it just
    #   confirmed was there, and every metric in its row is inflated with no
    #   visible symptom. config.py has always said so in a comment, but the
    #   typed models now appear in both selection lists, so one click can
    #   produce it; refused here rather than warned about, because the number
    #   it produces looks valid.
    run_judges = set(run.metadata().get("judges") or [])
    overlap = sorted(run_judges.intersection(llm_models or []))
    if overlap:
        raise SystemExit(
            f"{', '.join(overlap)} judged run {run_id}, so it cannot also be "
            "scored as a detector on it -- it would be graded on labels it "
            "certified itself. Pick a different detector model, or re-judge "
            "this run with other judges."
        )

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
        # Between detectors, not between samples: each detector writes one
        # result file covering every sample, so stopping inside one would
        # leave a detector scored on a subset of the dataset while its
        # neighbours were scored on all of it -- and the comparison table
        # gives no sign of that.
        if should_stop is not None and should_stop():
            log.warning(
                f"  Stop requested; skipping {tool_key} and every detector "
                "after it. The detectors that finished are kept and can be "
                "evaluated."
            )
            break

        started = time.time()
        try:
            if tool_key == "llm_detector":
                # None and [] mean different things, and the distinction is
                # what lets the arm be switched off.
                #
                #   None  the caller did not say -- the command line, which
                #         takes no model argument. Falls through to [None],
                #         which run_llm_detector resolves to
                #         config.DEFAULT_LLM_DETECTOR.
                #   []    the caller explicitly selected no models. The arm is
                #         skipped. Running the default here would spend API
                #         budget the user just declined.
                if llm_models is not None and not llm_models:
                    log.info(
                        "  skipping the LLM arm: no model selected"
                    )
                    continue

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

        except (InsufficientData, ModelUnavailable) as exc:
            # Expected outcomes, not defects: a run too small to train on, or a
            # pretrained model whose weights have not been downloaded. Logged
            # at warning level with no traceback, because nothing is broken --
            # the fix is to supply data or run fetch_models.py.
            #
            # Skipping is deliberate rather than emitting "safe" for every
            # sample: a model that never loaded has not found the files clean,
            # and scoring its silence would award it perfect specificity for
            # doing nothing.
            reason = ("insufficient_data" if isinstance(exc, InsufficientData)
                      else "model_unavailable")
            log.warning(f"  {tool_key} skipped: {exc}")
            failed.append({
                "tool": tool_key, "reason": reason, "detail": str(exc),
            })

        except Exception as exc:  # noqa: BLE001 - one arm must not kill the run
            log.warning(f"  {tool_key} failed: {exc}")
            # Full traceback to the run log only; the browser sees the message.
            log.info(traceback.format_exc())
            failed.append({
                "tool": tool_key, "reason": "error", "detail": str(exc)[:500],
            })

    # Keep this pass as a round of its own. Re-analysing a run is how one
    # tool selection is compared against another, so each pass has to remain
    # readable after the next one has written over the flat result files.
    round_id = run.record_detection_round(
        succeeded,
        label=", ".join(tools),
        extra={
            "llm_models": list(llm_models or []),
            "failed": [f["tool"] for f in failed],
        },
    )
    if round_id:
        log.info(f"Recorded this pass as {round_id}")

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

    # The pretrained detectors need their downloaded weights. This only stats
    # files -- it does not load TensorFlow -- so the Tools page stays fast.
    try:
        from detectors.ml_models import family_available
        status["ml_farasat"] = family_available("ml_farasat")
        status["ml_vudenc"] = family_available("ml_vudenc")
    except Exception:  # noqa: BLE001
        status["ml_farasat"] = status["ml_vudenc"] = False

    # The lexical detector needs only scikit-learn, which is a hard dependency
    # of the evaluation code anyway -- so in practice it is always available.
    try:
        import sklearn  # noqa: F401
        status["ml_tfidf"] = True
    except ImportError:
        status["ml_tfidf"] = False

    # Only credential presence is checked here, because probing the network on
    # every page load would make the interface sluggish. The embedding
    # detector needs scikit-learn AND a reachable endpoint to vectorise with.
    try:
        from common import credential_status
        creds = credential_status()
        has_api = bool(creds["judgment_url"] and creds["judgment_key"])
        status["ml_embedding"] = status["ml_tfidf"] and has_api

        # The LLM arm has four possible backends and needs only one of them:
        # the Judgment API, Anthropic, a local Ollama tag, or a typed System
        # One model. Laya in particular is a local checkpoint and needs no
        # credential at all, so gating the arm on the Judgment API key alone
        # marked it unavailable for a user who had selected a model that
        # would have worked. WHICH model is reachable is still checked at
        # selection time, by the dropdown, and again on the first call.
        import config
        from providers import systemone_available
        has_typed = any(
            systemone_available(model)[0] for model in config.SYSTEMONE_MODELS
        )
        status["llm_detector"] = bool(
            has_api or creds["anthropic_key"] or has_typed
        )
    except Exception:  # noqa: BLE001
        status["ml_embedding"] = status["llm_detector"] = False

    return status


def main() -> None:
    """Run detection from the command line.

    Usage:
        python src/detect.py                       most recent run, default tools
        python src/detect.py <run_id>              that run, default tools
        python src/detect.py <run_id> static_bandit ml_farasat
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
