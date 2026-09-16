"""Web interface for VulEvalBench.

A thin Flask layer over the pipeline modules in `src/`. It contains NO
benchmark logic of its own: generation, judging, detection and scoring are all
imported, so the browser and the command line run byte-for-byte the same code
and cannot produce different numbers.

PAGE STRUCTURE
    Every page shares one left-hand sidebar (templates/base.html) holding the
    navigation and, on the run page, the whole configuration panel. The main
    area is then free for output -- logs, tables and source code -- which is
    what actually needs the width.

        /                run configuration and live progress
        /runs            every previous run, newest first
        /runs/<run_id>   one run: samples, judgements, results
        /tools           the detector catalogue, grouped by arm
        /compare         cross-run comparison of tool performance

WHY A BACKGROUND THREAD
    A full pipeline makes one local generation call per pair, then two judgment
    calls per sample, then a detection pass per tool. That is minutes, not
    seconds. Doing it inside the HTTP request would leave the browser on a
    blank page and hit proxy and browser timeouts. Instead POST /api/run starts
    a worker thread and returns immediately; the page polls /api/status and
    streams the log as it arrives.

STATE MODEL
    Job state is a single module-level dict guarded by a lock. That is
    deliberate, and it carries a deployment constraint: the server must run as
    ONE process (see the waitress command in the Dockerfile). With multiple
    worker processes each would hold a different copy of this dict and the
    status endpoint would answer from whichever one the request landed on. A
    queue such as Redis would be the answer at real scale; for a single-user
    research tool it would be pure overhead.

SCOPE AND SECURITY
    This is a local research tool. There is no authentication, no CSRF
    protection and no multi-user isolation, and it must not be exposed to an
    untrusted network. Inside the container it binds 0.0.0.0 because that is
    the only way Docker can forward a port to it -- the safety boundary is the
    published port mapping in docker-compose.yml (127.0.0.1:8000), not the bind
    address.

Run with:
    python web/app.py            # local development
    docker compose up            # containerised (see Dockerfile)
"""

from __future__ import annotations

import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request

# The pipeline modules live in src/, which is not a package and is not on the
# import path when this file runs directly. Adding it here keeps src/ free of
# packaging boilerplate the command-line scripts do not need.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config                                              # noqa: E402
import runstore                                            # noqa: E402
from common import credential_status                       # noqa: E402
from detect import DEFAULT_TOOLS, available_tools, run_detection          # noqa: E402
from detectors import ARM_NAMES, DETECTOR_REGISTRY, detectors_by_arm  # noqa: E402
from evaluate import build_report, evaluate_run            # noqa: E402
from generate import generate_dataset                      # noqa: E402
from ingest import ingest                                  # noqa: E402
from judge import run_judgement                            # noqa: E402
from providers import list_ollama_models, ollama_available  # noqa: E402

app = Flask(__name__)

# Bounds on the run size. The upper bound is a guard rail, not a technical
# limit: a mistyped "500" in the browser should not launch an hours-long job by
# accident. The command line has no such cap.
MIN_PAIRS = 1
MAX_PAIRS = 50

# Cap on files taken from an ingested repository, for the same reason: each
# file costs one embedding call and one LLM call per LLM detector.
MAX_INGEST_FILES = 100


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------
# `state` drives the UI:
#   idle        nothing has run yet in this process
#   generating  stage 1
#   judging     stage 2
#   detecting   stage 3
#   evaluating  stage 4
#   done        finished; results ready
#   error       failed; `error` holds the message

JOB: dict = {
    "state": "idle",
    "log": [],
    "error": None,
    "run_id": None,
    "started_at": None,
    "finished_at": None,
}

# Guards every read and write of JOB. The worker and the request threads touch
# it concurrently, and a partially-updated dict served to the browser would
# show a finished job with a half-written log.
JOB_LOCK = threading.Lock()


def log_line(message: str) -> None:
    """Append one line to the job log, from either thread."""
    with JOB_LOCK:
        JOB["log"].append(message)


def set_state(state: str, error: str | None = None, run_id: str | None = None) -> None:
    """Move the job to a new state, stamping the finish time when terminal."""
    with JOB_LOCK:
        JOB["state"] = state
        JOB["error"] = error
        if run_id:
            JOB["run_id"] = run_id
        if state in ("done", "error"):
            JOB["finished_at"] = datetime.now(timezone.utc).isoformat()


def job_is_running() -> bool:
    """True while a pipeline job occupies the worker thread."""
    return JOB["state"] in (
        "starting", "generating", "judging", "detecting", "evaluating",
        "ingesting",
    )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_pipeline(options: dict) -> None:
    """Execute the selected pipeline stages. Runs on a background thread.

    Progress from every stage is routed into the job log through the
    `on_progress` callback, which is why the browser sees exactly the lines the
    terminal would print, and why they also land in `logs/<run_id>.log`.

    Args:
        options: the validated request body -- see `api_run` for the fields and
            how each is bounded.
    """
    try:
        run_id = options.get("run_id")

        # --- stage 0/1: obtain a dataset ---------------------------------
        if options["source"] == "generated":
            set_state("generating")
            log_line("=== Stage 1: generating benchmark ===")
            summary = generate_dataset(
                n_samples=options["pairs"] * 2,
                seed=options["seed"],
                generator_model=options["generator_model"],
                cwe_selection=options["cwe_selection"],
                cwe_subset=options["cwe_subset"],
                label=options["label"],
                on_progress=log_line,
            )
            run_id = summary["run_id"]

        elif options["source"] in ("github", "local"):
            set_state("ingesting")
            log_line("=== Stage 1: ingesting external code ===")
            summary = ingest(
                options["ingest_source"],
                max_files=options["max_files"],
                label=options["label"],
                on_progress=log_line,
            )
            run_id = summary["run_id"]

        # `existing` reuses a previous run's samples, which is how a run is
        # re-analysed with a different tool selection without paying to
        # regenerate it.
        elif not run_id:
            raise ValueError("No run selected to analyse.")

        # Record which run this job produced, so the browser can fetch its
        # report when the job finishes. The state itself is set by whichever
        # stage runs next.
        with JOB_LOCK:
            JOB["run_id"] = run_id
        log_line(f"Run id: {run_id}")

        # --- stage 2: judging ---------------------------------------------
        if options["do_judge"] and options["judge_models"]:
            set_state("judging", run_id=run_id)
            run_judgement(
                run_id,
                judge_models=options["judge_models"],
                on_progress=log_line,
            )

        # --- stage 3: detection -------------------------------------------
        if options["tools"]:
            set_state("detecting", run_id=run_id)
            run_detection(
                run_id,
                tools=options["tools"],
                llm_provider=options["llm_provider"],
                llm_models=options["llm_models"],
                on_progress=log_line,
            )

            # --- stage 4: evaluation --------------------------------------
            set_state("evaluating", run_id=run_id)
            evaluate_run(run_id, on_progress=log_line)

        log_line("")
        log_line("=== Done. Results ready. ===")
        set_state("done", run_id=run_id)

    # SystemExit is caught alongside Exception on purpose: the pipeline modules
    # raise it for configuration errors (missing key, no usable samples) and it
    # does NOT inherit from Exception. Without this the worker would die
    # silently and the UI would poll a running job forever.
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        log_line("")
        log_line(f"ERROR: {exc}")
        # Full traceback to the server console; only the message to the browser.
        traceback.print_exc()
        set_state("error", str(exc))


# ---------------------------------------------------------------------------
# Shared template context
# ---------------------------------------------------------------------------

def sidebar_context() -> dict:
    """Everything the sidebar needs, on every page.

    Gathered in one place so the navigation, the credential badges and the tool
    availability indicators cannot drift between pages.
    """
    tools_status = available_tools()
    return {
        "arms": detectors_by_arm(),
        "arm_names": ARM_NAMES,
        "tool_status": tools_status,
        "credentials": credential_status(),
        "ollama_up": ollama_available(),
        "run_count": len(runstore.list_runs()),
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def page_index():
    """The run page: configuration in the sidebar, progress in the main area."""
    ollama_models = list_ollama_models()

    return render_template(
        "index.html",
        active_page="run",
        # Generator options come from what Ollama actually has pulled, so the
        # dropdown can never offer a model that will fail at run time.
        ollama_models=ollama_models,
        default_generator=(
            config.GENERATOR_MODEL if config.GENERATOR_MODEL in ollama_models
            else (ollama_models[0] if ollama_models else "")
        ),
        judgment_models=config.JUDGMENT_MODELS,
        anthropic_models=config.ANTHROPIC_MODELS,
        default_judge_a=config.DEFAULT_JUDGE_A,
        default_judge_b=config.DEFAULT_JUDGE_B,
        default_llm_detector=config.DEFAULT_LLM_DETECTOR,
        cwes=[(c, config.CWE_NAMES.get(c, c)) for c in config.CWE_CATEGORIES],
        cwe_selection=config.CWE_SELECTION,
        min_pairs=MIN_PAIRS,
        max_pairs=MAX_PAIRS,
        default_pairs=config.N_SAMPLES // 2,
        default_seed=config.SEED,
        max_ingest_files=MAX_INGEST_FILES,
        # Sourced from detect.DEFAULT_TOOLS rather than repeated here, so the
        # boxes ticked in the browser can never drift from what the command
        # line runs. They already had: this list still named the removed
        # `static_ruff`, which silently left Semgrep unticked on page load.
        default_tools=DEFAULT_TOOLS,
        **sidebar_context(),
    )


@app.route("/runs")
def page_runs():
    """Every previous run, newest first.

    This is the page the supervisor asked for: nothing is overwritten any more,
    so every dataset ever generated remains reachable and inspectable.
    """
    return render_template(
        "runs.html",
        active_page="runs",
        runs=[run.summary() for run in runstore.list_runs()],
        **sidebar_context(),
    )


@app.route("/runs/<run_id>")
def page_run_detail(run_id: str):
    """One run in full: its samples, both judges' verdicts, and its results."""
    if not runstore.is_valid_run_id(run_id):
        abort(404)
    try:
        run = runstore.load_run(run_id)
    except (FileNotFoundError, ValueError):
        abort(404)

    return render_template(
        "run_detail.html",
        active_page="runs",
        run_id=run_id,
        summary=run.summary(),
        metadata=run.metadata(),
        **sidebar_context(),
    )


@app.route("/tools")
def page_tools():
    """The detector catalogue: what each tool is and whether it can run here."""
    return render_template(
        "tools.html",
        active_page="tools",
        registry=DETECTOR_REGISTRY,
        **sidebar_context(),
    )


@app.route("/compare")
def page_compare():
    """Cross-run comparison of tool performance."""
    return render_template(
        "compare.html",
        active_page="compare",
        weights=config.COMPOSITE_WEIGHTS,
        **sidebar_context(),
    )


# ---------------------------------------------------------------------------
# API -- running the pipeline
# ---------------------------------------------------------------------------

def _clean_string_list(value, allowed: dict | list) -> list[str]:
    """Filter a browser-supplied list down to values on an allow-list.

    Everything in a request body is untrusted, even in a local tool. Model
    names and tool keys reach subprocess arguments and API calls, so they are
    checked against a known set rather than passed through. De-duplicated with
    order preserved, since selecting the same model twice would double the cost
    for no benefit.
    """
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    return [
        item for item in value
        if isinstance(item, str) and item in allowed
        and not (item in seen or seen.add(item))
    ]


@app.route("/api/run", methods=["POST"])
def api_run():
    """Start a pipeline job. Returns immediately; poll /api/status."""
    with JOB_LOCK:
        # One job at a time. Two concurrent runs would interleave their writes
        # and their log lines.
        if job_is_running():
            return jsonify({"error": "A run is already in progress."}), 409

        JOB.update({
            "state": "starting",
            "log": [],
            "error": None,
            "run_id": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
        })

    payload = request.get_json(silent=True) or {}

    # --- validate every field ----------------------------------------------
    source = payload.get("source", "generated")
    if source not in ("generated", "github", "local", "existing"):
        source = "generated"

    try:
        pairs = int(payload.get("pairs", config.N_SAMPLES // 2))
    except (TypeError, ValueError):
        pairs = config.N_SAMPLES // 2
    pairs = max(MIN_PAIRS, min(MAX_PAIRS, pairs))

    try:
        max_files = int(payload.get("max_files", 40))
    except (TypeError, ValueError):
        max_files = 40
    max_files = max(1, min(MAX_INGEST_FILES, max_files))

    # An empty seed box means "fresh randomness", which the pipeline expresses
    # as None.
    raw_seed = payload.get("seed")
    try:
        seed = int(raw_seed) if raw_seed not in (None, "") else None
    except (TypeError, ValueError):
        seed = None

    ollama_models = list_ollama_models()
    generator_model = payload.get("generator_model")
    if generator_model not in ollama_models:
        generator_model = (
            config.GENERATOR_MODEL if config.GENERATOR_MODEL in ollama_models
            else (ollama_models[0] if ollama_models else config.GENERATOR_MODEL)
        )

    # Judges and LLM detectors may come from either remote family. The backend
    # is derived per model by providers.provider_for_model(), so the request
    # carries model names only and there is no provider field to keep in sync
    # with them.
    selectable_models = {**config.JUDGMENT_MODELS, **config.ANTHROPIC_MODELS}

    judge_models = _clean_string_list(
        payload.get("judge_models"), selectable_models
    )[:2]  # at most two, which is what the side-by-side view renders

    # Local tags are additionally allowed for the detector arm, so a locally
    # served model can be compared against the remote ones.
    allowed_llm = {**selectable_models, **{m: m for m in ollama_models}}
    llm_models = _clean_string_list(payload.get("llm_models"), allowed_llm)
    if not llm_models:
        llm_models = [config.DEFAULT_LLM_DETECTOR]

    # None means "resolve from each model id"; an explicit value is honoured
    # only if the caller sends a valid one.
    llm_provider = payload.get("llm_provider")
    if llm_provider not in ("passau", "anthropic", "ollama"):
        llm_provider = None

    tools = _clean_string_list(payload.get("tools"), DETECTOR_REGISTRY)

    cwe_subset = _clean_string_list(
        payload.get("cwe_subset"), config.CWE_CATEGORIES
    )

    cwe_selection = payload.get("cwe_selection", config.CWE_SELECTION)
    if cwe_selection not in ("random", "cycle"):
        cwe_selection = config.CWE_SELECTION

    run_id = payload.get("run_id")
    if run_id is not None and not runstore.is_valid_run_id(str(run_id)):
        run_id = None

    ingest_source = str(payload.get("ingest_source", "")).strip()
    if source in ("github", "local") and not ingest_source:
        set_state("error", "No source path or URL was given.")
        return jsonify({"error": "No source path or URL was given."}), 400

    options = {
        "source": source,
        "run_id": run_id,
        "pairs": pairs,
        "seed": seed,
        "generator_model": generator_model,
        "cwe_selection": cwe_selection,
        "cwe_subset": cwe_subset or None,
        "label": str(payload.get("label", ""))[:200],
        "do_judge": bool(payload.get("do_judge", True)),
        "judge_models": judge_models,
        "tools": tools,
        "llm_provider": llm_provider,
        "llm_models": llm_models,
        "ingest_source": ingest_source,
        "max_files": max_files,
    }

    # daemon=True so the thread cannot keep the process alive on shutdown --
    # important in a container, where a lingering thread delays SIGTERM.
    threading.Thread(
        target=run_pipeline, args=(options,), daemon=True
    ).start()

    return jsonify({"started": True, **options})


@app.route("/api/status")
def api_status():
    """Current job state and the log so far.

    The whole log is returned each time rather than only new lines. At a few
    hundred lines that is far cheaper than the bookkeeping an incremental
    protocol needs, and it keeps the client stateless: a refresh mid-run picks
    up the complete history.
    """
    with JOB_LOCK:
        return jsonify({
            "state": JOB["state"],
            "log": list(JOB["log"]),      # a copy, so callers cannot mutate it
            "error": JOB["error"],
            "run_id": JOB["run_id"],
            "started_at": JOB["started_at"],
            "finished_at": JOB["finished_at"],
        })


# ---------------------------------------------------------------------------
# API -- reading runs
# ---------------------------------------------------------------------------

def _load_run_or_404(run_id: str):
    """Resolve a run id from the URL, refusing anything malformed.

    `run_id` is used to build a filesystem path, so it is validated rather than
    trusted. Without this a request for `../../api-key-llm.txt` would be a
    path-traversal read -- the exact CWE-22 class this project generates test
    cases for. `runstore.load_run` performs the check; this wrapper turns its
    exceptions into a 404.
    """
    try:
        return runstore.load_run(run_id)
    except (FileNotFoundError, ValueError):
        abort(404)


@app.route("/api/runs")
def api_runs():
    """Summaries of every run, newest first."""
    return jsonify({"runs": [run.summary() for run in runstore.list_runs()]})


@app.route("/api/runs/<run_id>/report")
def api_run_report(run_id: str):
    """The full evaluation report for one run."""
    run = _load_run_or_404(run_id)
    if not run.detection_results():
        return jsonify({"error": "This run has no detector results yet."}), 404
    return jsonify(build_report(run_id))


@app.route("/api/runs/<run_id>/samples")
def api_run_samples(run_id: str):
    """Every sample in a run, with its judging outcome.

    Labels and judge verdicts are included because this feeds the *inspection*
    page, which a human reads. Detectors never call this endpoint -- they are
    handed source through the pipeline, not through HTTP.
    """
    run = _load_run_or_404(run_id)

    generated = run.read_generation_manifest()
    verified_ids = {entry["id"] for entry in run.read_verified_manifest()}

    # Judge verdicts, indexed per model, so the page can render them side by
    # side without another round trip.
    judgements: dict[str, dict] = {}
    for record in run.judgement_results():
        judgements[record["judge_model"]] = {
            verdict["id"]: verdict for verdict in record["verdicts"]
        }

    return jsonify({
        "run_id": run_id,
        "judges": list(judgements),
        "samples": [
            {
                **entry,
                "admitted": entry["id"] in verified_ids,
                "judge_verdicts": {
                    model: verdicts.get(entry["id"])
                    for model, verdicts in judgements.items()
                },
            }
            for entry in generated
        ],
    })


@app.route("/api/runs/<run_id>/sample/<sample_id>")
def api_run_sample(run_id: str, sample_id: str):
    """The source of one sample, plus its metadata.

    `sample_id` is validated against the manifest rather than joined onto a
    path directly -- the same traversal concern as `_load_run_or_404`.
    """
    run = _load_run_or_404(run_id)

    entry = next(
        (e for e in run.read_generation_manifest() if e["id"] == sample_id),
        None,
    )
    if entry is None:
        return jsonify({"error": "Unknown sample id."}), 404

    return jsonify({
        **entry,
        "code": run.read_sample_code(entry),
    })


@app.route("/api/runs/<run_id>/judgement")
def api_run_judgement(run_id: str):
    """Both judges' verdicts side by side, with the admission decision."""
    run = _load_run_or_404(run_id)

    comparison = run.judgement_dir / "judgement_comparison.json"
    if not comparison.exists():
        return jsonify({"error": "This run has not been judged."}), 404

    from common import read_json
    return jsonify(read_json(comparison))


@app.route("/api/compare")
def api_compare():
    """Every evaluated run's ranking, for cross-run comparison.

    Only runs that have been through evaluation appear; the rest have no
    numbers to contribute.
    """
    from common import read_json

    rows = []
    for run in runstore.list_runs():
        if not run.comparison_report_file.exists():
            continue
        try:
            report = read_json(run.comparison_report_file)
        except (OSError, ValueError):
            # A truncated report from an interrupted run should not break the
            # whole comparison page.
            continue
        rows.append({
            "run_id": run.run_id,
            "created_at": run.metadata().get("created_at", ""),
            "source": report.get("dataset", {}).get("source", ""),
            "samples": report.get("dataset", {}).get("labelled_samples", 0),
            "ranking": report.get("ranking", []),
            "agreement": report.get("agreement", []),
        })

    return jsonify({"runs": rows, "weights": config.COMPOSITE_WEIGHTS})


# ---------------------------------------------------------------------------
# Local development entry point
# ---------------------------------------------------------------------------
# Only used by `python web/app.py`. In the container the app is served by
# waitress (see Dockerfile), which is a real WSGI server and does not print a
# development-server warning.

if __name__ == "__main__":
    # 127.0.0.1 for local runs: no reason to accept connections from the
    # network when the browser is on the same machine.
    # debug=False because debug mode exposes an interactive debugger that
    # executes arbitrary code -- unacceptable in a tool holding API keys.
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
