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
from balance import balance_run                            # noqa: E402
import model_registry                                      # noqa: E402
import runstore                                            # noqa: E402
from common import credential_status, read_json            # noqa: E402
from detect import DEFAULT_TOOLS, available_tools, run_detection          # noqa: E402
from detectors import ARM_NAMES, DETECTOR_REGISTRY, detectors_by_arm  # noqa: E402
from evaluate import build_report, evaluate_run            # noqa: E402
from generate import generate_dataset                      # noqa: E402
from ingest import ingest
from perturb import (DEFAULT_PERTURBATIONS, PERTURBATIONS,
                     PERTURBATION_LABELS, perturb_run)                                  # noqa: E402
from judge import run_judgement                            # noqa: E402
from providers import (  # noqa: E402
    list_ollama_models,
    ollama_available,
    systemone_available,
)

app = Flask(__name__)

# Bounds on the run size. The upper bound is a guard rail, not a technical
# limit, and the command line has no cap at all.
#
# A request above it is REFUSED, naming the limit and the number asked for.
# It used to be clamped silently -- `min(MAX_PAIRS, pairs)` -- so asking for
# 1000 started a 50-request run that looked, in the log and in the metadata,
# exactly like a 50-request run that had been asked for. Quietly substituting
# a different number is worse than refusing: the run that comes back is not
# the run that was requested, and nothing says so.
MIN_PAIRS = 1
MAX_PAIRS = 1000

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
#   selecting   stage 2b -- PAUSED, waiting for the user to pick samples
#   balancing   stage 2c -- optional; always reports, drops only if asked
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

# MANUAL SAMPLE SELECTION.
#   When the user asks to review the dataset, the worker stops after stage 1
#   and waits here while the browser shows the generated samples. The handshake
#   is one Event and one result slot:
#
#       worker                          request thread
#       ------                          --------------
#       SELECTION_READY.clear()
#       state -> "selecting"
#       SELECTION_READY.wait(timeout) ..
#                                       POST /api/selection
#                                       SELECTION["kept"] = [...]
#                                       SELECTION["balance_*"] = bool
#                                       SELECTION_READY.set()
#       reads SELECTION, continues
#
#   The balancing choice (stage 2c) travels on the same POST rather than with
#   the run configuration, because it is a decision about the set the reviewer
#   has just settled on: how many samples it would cost cannot be known before
#   the review, and a choice made beforehand would be made blind.
#
#   The timeout is what keeps a closed browser tab from pinning the worker
#   thread for the life of the process. One job runs at a time, so a thread
#   stuck here would block every later run with "a run is already in progress"
#   and the only cure would be restarting the server.
SELECTION_TIMEOUT_SECONDS = 30 * 60

SELECTION_READY = threading.Event()
SELECTION: dict = {
    "kept": None, "cancelled": False,
    "balance_labels": False, "balance_classes": False,
}

# STOPPING A RUNNING JOB.
#   Cooperative, not forced. The worker thread cannot be killed safely -- it
#   holds open files and half-written manifests -- so stopping is a flag the
#   stages themselves consult, and each one stops at the next point where the
#   run is in a consistent state:
#
#     generating  -- after the request in flight; what was generated is kept,
#                    written and usable as a smaller dataset
#     judging     -- after the judge model in flight finishes its samples
#     detecting   -- after the detector in flight finishes
#     selecting   -- at once, through the existing selection cancel path
#
#   So the button is not instant, and the log says what it is waiting for.
#   The alternative -- abandoning the thread mid-write -- would leave a run
#   directory that later stages read as valid.
STOP_REQUESTED = threading.Event()

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
        if state in ("done", "error", "stopped"):
            JOB["finished_at"] = datetime.now(timezone.utc).isoformat()


def job_is_running() -> bool:
    """True while a pipeline job occupies the worker thread.

    "stopped" is deliberately absent: a stopped job has left the worker, so
    the next run may start immediately.
    """
    return JOB["state"] in (
        "starting", "generating", "selecting", "balancing", "judging",
        "detecting", "evaluating", "ingesting", "perturbing",
    )


def stop_requested() -> bool:
    """Whether the user has asked for the running job to stop.

    Passed into the long-running stages as `should_stop`, so they can leave
    their loops at a safe point instead of being interrupted.
    """
    return STOP_REQUESTED.is_set()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def await_selection(run_id: str) -> dict | None:
    """Pause the pipeline while the user keeps or drops each sample.

    Writes the kept set to the run's selected manifest, which is what the
    judging stage reads, and records the decision in the run metadata so the
    dataset can be explained later: "46 of 59" means nothing without knowing
    whether a human or a judge removed the other 13.

    Returns:
        `{"kept": [...], "balance_labels": bool, "balance_classes": bool}`,
        or None if the user cancelled or never answered. The balancing flags
        are chosen in the same panel and acted on by stage 2c immediately
        afterwards, on exactly the set this function just wrote.
    """
    run = runstore.load_run(run_id)
    manifest = run.read_generation_manifest()
    # Counted from the judgement file rather than read_verified_manifest(),
    # which would return a selection from an EARLIER review of this run and
    # report the previous decision as though the judges had made it.
    admitted = (len(read_json(run.verified_manifest_file))
                if run.verified_manifest_file.exists() else None)

    SELECTION.update({
        "kept": None, "cancelled": False,
        "balance_labels": False, "balance_classes": False,
    })
    SELECTION_READY.clear()
    set_state("selecting", run_id=run_id)
    log_line("")
    log_line("=== Stage 2b: review samples ===")
    if admitted is None:
        log_line(f"{len(manifest)} samples, not judged. "
                 "Waiting for your selection.")
    else:
        log_line(f"{len(manifest)} samples, {admitted} admitted by the judges. "
                 "Waiting for your selection.")

    if not SELECTION_READY.wait(timeout=SELECTION_TIMEOUT_SECONDS):
        # Not an error in the run -- the samples are on disk and the run can be
        # re-analysed later -- so it ends quietly rather than as a failure.
        log_line(
            f"No selection after {SELECTION_TIMEOUT_SECONDS // 60} minutes. "
            "Stopping; the generated samples are kept and this run can be "
            "re-analysed from the 'existing run' source."
        )
        return None

    if SELECTION["cancelled"]:
        return None

    kept = run.write_selected_manifest(SELECTION["kept"] or [])
    dropped = [e["id"] for e in manifest if e["id"] not in {k["id"] for k in kept}]
    run.update_metadata(manual_selection={
        "kept": [e["id"] for e in kept],
        "dropped": dropped,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    })

    log_line(f"Kept {len(kept)} of {len(manifest)} samples.")
    if dropped:
        log_line(f"Dropped by hand: {', '.join(dropped)}")

    # Restoring a judge-rejected sample is legitimate -- a judge can be wrong,
    # and overriding one is the point of reviewing after judging -- but it puts
    # a label into the dataset that no judge agreed with, so it is recorded
    # rather than passed over in silence.
    if run.verified_manifest_file.exists():
        admitted_ids = {e["id"] for e in read_json(run.verified_manifest_file)}
        restored = [e["id"] for e in kept if e["id"] not in admitted_ids]
        if restored:
            log_line(
                f"Restored against the judges' decision: {', '.join(restored)}"
            )
    return {
        "kept": kept,
        "balance_labels": bool(SELECTION["balance_labels"]),
        "balance_classes": bool(SELECTION["balance_classes"]),
    }


def run_pipeline(options: dict) -> None:
    """Execute the selected pipeline stages. Runs on a background thread.

    Progress from every stage is routed into the job log through the
    `on_progress` callback, which is why the browser sees exactly the lines the
    terminal would print, and why they also land in `logs/<run_id>.log`.

    Args:
        options: the validated request body -- see `api_run` for the fields and
            how each is bounded.
    """
    # Cleared here, not in the endpoint: a stop asked for while the previous
    # job was finishing must not cancel this one before it starts.
    STOP_REQUESTED.clear()

    try:
        run_id = options.get("run_id")

        # --- stage 0/1: obtain a dataset ---------------------------------
        if options["source"] == "generated":
            set_state("generating")
            log_line("=== Stage 1: generating benchmark ===")
            summary = generate_dataset(
                # The count in the form is a count of REQUESTS: pairs when
                # twinned, programs when not. Doubling it either way would
                # silently generate twice what was asked for on an unpaired
                # run.
                n_samples=(options["pairs"] * 2 if options["twins"]
                           else options["pairs"]),
                seed=options["seed"],
                generator_model=options["generator_model"],
                cwe_selection=options["cwe_selection"],
                cwe_subset=options["cwe_subset"],
                twins=options["twins"],
                label=options["label"],
                on_progress=log_line,
                should_stop=stop_requested,
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

        # Optional robustness step, available only on the re-analysis path.
        # The perturbed samples become a NEW run linked back to the original,
        # so the clean results are never overwritten and the two appear side
        # by side in the comparison views like any other pair of runs.
        if options["source"] == "existing" and options.get("perturb"):
            set_state("perturbing", run_id=run_id)
            log_line("=== Stage 1b: perturbing samples ===")
            summary = perturb_run(
                run_id,
                kinds=options.get("perturb_kinds") or None,
                seed=options["seed"],
                label=options["label"],
                on_progress=log_line,
            )
            # Everything downstream -- judging, detection, evaluation -- now
            # operates on the perturbed copy.
            run_id = summary["run_id"]

        # Record which run this job produced, so the browser can fetch its
        # report when the job finishes. The state itself is set by whichever
        # stage runs next.
        with JOB_LOCK:
            JOB["run_id"] = run_id
        log_line(f"Run id: {run_id}")

        # A stop during stage 1 ends the pipeline here. The samples are on
        # disk and the run can be judged or analysed later from the
        # "re-analyse a previous run" source, so nothing is lost -- but
        # judging and detection cost API calls, and the user has just said
        # stop.
        if stop_requested():
            log_line("")
            log_line(
                "Stopped after stage 1. The samples are kept; judge or "
                "analyse them later with the 'existing run' source."
            )
            set_state("stopped", run_id=run_id)
            return

        # --- stage 2: judging ---------------------------------------------
        if options["do_judge"] and options["judge_models"]:
            set_state("judging", run_id=run_id)
            run_judgement(
                run_id,
                judge_models=options["judge_models"],
                on_progress=log_line,
                should_stop=stop_requested,
            )

        # --- stage 2b: manual selection ------------------------------------
        # After judging, deliberately. The reviewer sees each sample together
        # with the verdicts it collected, and their decision is the last word:
        # a sample dropped here stays dropped, and one restored here comes back
        # even though the judges rejected it. Runs for every source -- a run
        # whose samples were ingested or reused is reviewed exactly like a
        # generated one, and judging having been skipped only means there are
        # no verdicts to show.
        # Both balancing steps default to off for a run that is not reviewed:
        # they are chosen in the review panel, so a run that skips the review
        # has not been asked the question and nothing is dropped on a guess.
        balance_labels = balance_classes = False

        if stop_requested():
            log_line("")
            log_line("Stopped after judging. The run is kept as it stands.")
            set_state("stopped", run_id=run_id)
            return

        if options.get("manual_selection"):
            review = await_selection(run_id)
            if review is None:
                # Three ways to get here: Cancel, Stop, or the review timing
                # out. They are reported differently because they mean
                # different things to someone reading the log later.
                if stop_requested():
                    log_line("Stopped during sample review.")
                    set_state("stopped", run_id=run_id)
                else:
                    log_line("Run cancelled during sample review.")
                    set_state("idle", run_id=run_id)
                return
            balance_labels = review["balance_labels"]
            balance_classes = review["balance_classes"]

        # --- stage 2c: balancing -------------------------------------------
        # Immediately after the review, and on the set the review produced.
        # The reviewer decides what is admissible and balancing decides what is
        # comparable, so the second has to act on the output of the first:
        # balancing a set a human then edited would leave it unbalanced again,
        # and asking for balancing before the review would mean choosing
        # without knowing what it would cost.
        #
        # Called even when neither step is selected, because the report it
        # prints is worth having on every run -- it is the only place the live
        # log says what the dataset's composition actually is. With no step
        # enabled it writes nothing and drops nothing.
        set_state("balancing", run_id=run_id)
        balance_run(
            run_id,
            by_label=balance_labels,
            by_class=balance_classes,
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
                should_stop=stop_requested,
            )

            # --- stage 4: evaluation --------------------------------------
            # Runs even after a stop, on whichever detectors finished: a
            # detector that completed has a result worth reading, and
            # evaluation is local and free.
            set_state("evaluating", run_id=run_id)
            evaluate_run(run_id, on_progress=log_line)

        log_line("")
        if stop_requested():
            log_line("=== Stopped. Results for the stages that ran. ===")
            set_state("stopped", run_id=run_id)
        else:
            log_line("=== Done. Results ready. ===")
            set_state("done", run_id=run_id)

    # SystemExit is caught alongside Exception on purpose: the pipeline modules
    # raise it for configuration errors (missing key, no usable samples) and it
    # does NOT inherit from Exception. Without this the worker would die
    # silently and the UI would poll a running job forever.
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        # A stage can exit by raising when a stop lands early -- generation
        # stopped before its first usable sample, for instance. That is the
        # user's own decision carried out, not a failure, and reporting it in
        # red as an ERROR would misdescribe it.
        if stop_requested():
            log_line("")
            log_line(f"Stopped: {exc}")
            set_state("stopped", run_id=options.get("run_id"))
            return
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

    # The overlay can hide a shipped model and add new ones, so the menus are
    # built from it rather than straight from config. `hidden` is subtracted
    # from each shipped group; `registered` becomes its own optgroup, so it is
    # always visible whether a model was shipped or added here.
    hidden = model_registry.disabled_ids()
    shipped_ids = set(config.JUDGMENT_MODELS) | set(config.ANTHROPIC_MODELS)         | set(config.SYSTEMONE_MODELS)
    registered_models = {
        model: name
        for model, name in model_registry.models_for("llm_detector").items()
        if model not in shipped_ids
    }
    registered_judges = {
        model: name
        for model, name in model_registry.models_for("judge").items()
        if model not in shipped_ids
    }
    # A generator must actually be pulled, so added Ollama tags only appear if
    # the server reports them; a tag in the overlay that is not pulled would
    # fail on the first sample.
    generator_choices = [m for m in ollama_models if m not in hidden]

    return render_template(
        "index.html",
        active_page="run",
        registered_models=registered_models,
        registered_judges=registered_judges,
        # Generator options come from what Ollama actually has pulled, so the
        # dropdown can never offer a model that will fail at run time.
        ollama_models=generator_choices,
        default_generator=(
            config.GENERATOR_MODEL
            if config.GENERATOR_MODEL in generator_choices
            else (generator_choices[0] if generator_choices else "")
        ),
        judgment_models={k: v for k, v in config.JUDGMENT_MODELS.items()
                         if k not in hidden},
        anthropic_models={k: v for k, v in config.ANTHROPIC_MODELS.items()
                          if k not in hidden},
        # The typed, non-generative judges. Offered with the reason they
        # cannot be used, if they cannot: an unavailable judge is shown
        # disabled and explained, rather than accepted and then failing on the
        # first sample of a run the user has already paid to generate.
        systemone_models={k: v for k, v in config.SYSTEMONE_MODELS.items()
                          if k not in hidden},
        systemone_status={
            model: systemone_available(model)[1]
            for model in config.SYSTEMONE_MODELS
        },
        # The six semantics-preserving transformations, for the robustness
        # checkbox. Sourced from perturb.py so the form can never offer one the
        # backend does not implement.
        perturbations=PERTURBATION_LABELS,
        default_perturbations=DEFAULT_PERTURBATIONS,
        default_judge_a=config.DEFAULT_JUDGE_A,
        default_judge_b=config.DEFAULT_JUDGE_B,
        default_judge_c=config.DEFAULT_JUDGE_C,
        judge_consensus=config.JUDGE_CONSENSUS,
        # Both detector slots, empties dropped. A list because the arm takes
        # any number of models -- including none, which is why this may be
        # empty and the template must not assume a selection exists.
        default_llm_detectors=[
            m for m in (config.DEFAULT_LLM_DETECTOR,
                        config.DEFAULT_LLM_DETECTOR_B) if m
        ],
        cwes=[(c, config.CWE_NAMES.get(c, c)) for c in config.CWE_CATEGORIES],
        cwe_selection=config.CWE_SELECTION,
        min_pairs=MIN_PAIRS,
        max_pairs=MAX_PAIRS,
        # The field counts requests, so what N_SAMPLES means to it depends on
        # whether those requests carry twins. Halving it either way would
        # default an unpaired run to half the configured size.
        default_pairs=(config.N_SAMPLES // 2 if config.GENERATE_TWINS
                       else config.N_SAMPLES),
        default_twins=config.GENERATE_TWINS,
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


ROLE_HELP = {
    "generator": "writes the benchmark pairs; needs long schema-valid JSON, "
                 "normally a local Ollama tag",
    "judge": "votes on whether a sample really carries its stated label",
    "llm_detector": "is scored as a tool under test",
    "ml": "a pretrained word2vec + recurrent family loaded from models/",
}


@app.route("/models", methods=["GET", "POST"])
def page_models():
    """Add, delete or hide models, per installation.

    POSTs and then re-renders rather than redirecting: the result of an edit
    is a list of problems or a confirmation, and both belong on the page that
    caused them. There is no session to carry a flash message in -- the app
    is deliberately single-process and stateless between requests.
    """
    messages: list[str] = []
    errors = False

    if request.method == "POST":
        form = request.form
        action = form.get("action", "")

        if action == "add":
            roles = form.getlist("roles")
            entry = {
                "id": (form.get("id") or "").strip(),
                "name": (form.get("name") or "").strip(),
                "roles": roles,
                "note": (form.get("note") or "").strip(),
            }
            # Only the fields the chosen roles actually need are stored, so an
            # ml directory typed into an api-only entry is not kept and later
            # mistaken for meaningful configuration.
            if any(role != "ml" for role in roles):
                entry["provider"] = (form.get("provider") or "").strip()
            if "ml" in roles:
                entry["directory"] = (form.get("directory") or "").strip()
                entry["word2vec"] = (form.get("word2vec") or "").strip()
                entry["dimensions"] = (form.get("dimensions") or "").strip()

            problems = model_registry.add(
                entry, replace=bool(form.get("replace"))
            )
            blocking = [p for p in problems if not p.startswith("note:")]
            if blocking:
                errors = True
                messages = blocking
            else:
                messages = [
                    f"Registered {entry['id']} for {', '.join(roles)}."
                ] + problems

        elif action == "remove":
            model_id = (form.get("id") or "").strip()
            problems = model_registry.remove(model_id)
            errors = bool(problems)
            messages = problems or [f"{model_id} removed or hidden."]

        elif action == "enable":
            model_id = (form.get("id") or "").strip()
            problems = model_registry.enable(model_id)
            errors = bool(problems)
            messages = problems or [f"{model_id} is visible again."]

        else:
            errors = True
            messages = [f"Unknown action {action!r}."]

    return render_template(
        "models.html",
        active_page="models",
        registry=model_registry.summary(),
        roles=model_registry.ROLES,
        api_providers=model_registry.API_PROVIDERS,
        role_help=ROLE_HELP,
        messages=messages,
        errors=errors,
        key_status=[
            ("Judgment endpoint URL", credential_status()["judgment_url"]),
            ("Judgment endpoint key", credential_status()["judgment_key"]),
            ("Anthropic key (api-key-LLM.txt)",
             credential_status()["anthropic_key"]),
            ("TypeSafe key (api-key-typesafe.txt)",
             credential_status()["typesafe_key"]),
        ],
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
    """Start a pipeline job. Returns immediately; poll /api/status.

    VALIDATE FIRST, THEN CLAIM THE WORKER.
        The job used to be marked "starting" at the top, before the body was
        read. Any request refused after that point -- a missing ingest path, a
        run size out of bounds -- returned its 400 and left JOB sitting in
        "starting" forever. `job_is_running()` reads that as a live job, so
        every later run was refused with "A run is already in progress" and
        the only cure was restarting the server.

        So nothing touches JOB until the request is known to be runnable, and
        the "one job at a time" check and the claim happen together under the
        lock, which is what makes two simultaneous requests safe.
    """
    with JOB_LOCK:
        if job_is_running():
            return jsonify({"error": "A run is already in progress."}), 409

    payload = request.get_json(silent=True) or {}

    # --- validate every field ----------------------------------------------
    source = payload.get("source", "generated")
    if source not in ("generated", "github", "local", "existing"):
        source = "generated"

    try:
        pairs = int(payload.get("pairs", config.N_SAMPLES // 2))
    except (TypeError, ValueError):
        pairs = config.N_SAMPLES // 2
    if source == "generated" and not MIN_PAIRS <= pairs <= MAX_PAIRS:
        return jsonify({"error": (
            f"Run size must be between {MIN_PAIRS} and {MAX_PAIRS}; "
            f"you asked for {pairs}. The command line has no cap: "
            f"python src/generate.py {pairs}"
        )}), 400

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
    #
    # Both lists come from the overlay, so a model added on this installation
    # is accepted and one hidden here is refused -- the menus and the
    # validation must agree, or a hidden model could still be submitted by
    # hand to an endpoint the user had removed.

    # The System One models are available to BOTH stages. They cannot answer a
    # chat prompt, so each stage reaches them through its own typed path --
    # judge.judge_one_typed() and llm_tools._detect_one_typed() -- and the
    # model id is all either stage needs to route correctly.
    #
    # A model selected in both lists at once is refused by detect.py rather
    # than silently accepted: it would be scored on labels it certified.
    judge_selectable = model_registry.models_for("judge")

    judge_models = _clean_string_list(
        payload.get("judge_models"), judge_selectable
    )[:3]  # three slots; the consensus rule decides what agreement means

    # Local tags are additionally allowed for the detector arm, so a locally
    # served model can be compared against the remote ones. The typed models
    # join them: as detectors they are tools under test like any other, and
    # the count question that makes them awkward as judges is not asked here.
    allowed_llm = {
        **model_registry.models_for("llm_detector"),
        **{m: m for m in ollama_models},
    }
    # An EMPTY selection means "no LLM arm", and is honoured.
    #
    # It used to fall back to DEFAULT_LLM_DETECTOR, which made the arm
    # impossible to switch off from the interface: deselecting every model
    # silently ran the default one anyway, spending API budget the user had
    # just declined. The caller distinguishes the two cases by whether
    # `llm_detector` is in the tool list, so an empty list here is
    # unambiguous rather than a missing value.
    llm_models = _clean_string_list(payload.get("llm_models"), allowed_llm)

    # None means "resolve from each model id"; an explicit value is honoured
    # only if the caller sends a valid one.
    #
    # "systemone" is NOT accepted as a forced provider: it is not one backend
    # but two (a hosted API and a local checkpoint), and only the model id
    # says which. Leaving it unset lets provider_for_model() resolve it.
    llm_provider = payload.get("llm_provider")
    if llm_provider not in ("passau", "anthropic", "ollama"):
        llm_provider = None

    tools = _clean_string_list(payload.get("tools"), DETECTOR_REGISTRY)

    # Pausing for a human to review the samples only makes sense when there is
    # a browser waiting on the other end, which there always is here -- this is
    # the only entry point that can show the list. The CLI stages are
    # unaffected and keep running straight through.
    manual_selection = bool(payload.get("manual_selection"))

    # Perturbation is offered only on the re-analysis path: it rewrites
    # samples that already exist, so there is nothing to perturb on a fresh
    # generation or an ingest.
    perturb = bool(payload.get("perturb")) and source == "existing"
    perturb_kinds = _clean_string_list(
        payload.get("perturb_kinds"), PERTURBATIONS
    ) or list(DEFAULT_PERTURBATIONS)

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
        return jsonify({"error": "No source path or URL was given."}), 400

    # Pairing is a property of the dataset, not of one stage, so it is part
    # of the run configuration. Balancing is NOT: it travels on
    # POST /api/selection instead, because it is a decision about the reviewed
    # set rather than about the run.
    twins = bool(payload.get("twins", config.GENERATE_TWINS))

    options = {
        "source": source,
        "run_id": run_id,
        "pairs": pairs,
        "twins": twins,
        "seed": seed,
        "generator_model": generator_model,
        "cwe_selection": cwe_selection,
        "cwe_subset": cwe_subset or None,
        "label": str(payload.get("label", ""))[:200],
        "do_judge": bool(payload.get("do_judge", True)),
        "manual_selection": manual_selection,
        "judge_models": judge_models,
        "tools": tools,
        "llm_provider": llm_provider,
        "llm_models": llm_models,
        "ingest_source": ingest_source,
        "max_files": max_files,
        "perturb": perturb,
        "perturb_kinds": perturb_kinds,
    }

    # Re-checked under the lock and claimed in the same breath: the check at
    # the top of this function was only a cheap early refusal, and between it
    # and here another request could have claimed the worker.
    with JOB_LOCK:
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

    # daemon=True so the thread cannot keep the process alive on shutdown --
    # important in a container, where a lingering thread delays SIGTERM.
    threading.Thread(
        target=run_pipeline, args=(options,), daemon=True
    ).start()

    return jsonify({"started": True, **options})


@app.route("/api/selection", methods=["POST"])
def api_selection():
    """Hand the worker the samples to keep, releasing the paused pipeline.

    Accepts `{"keep": [sample_id, ...]}` with the optional booleans
    `balance_labels` and `balance_classes`, or `{"cancel": true}` to stop the
    run and leave the generated samples on disk.

    Rejected unless a job is actually waiting: without that check a stray POST
    would set the Event, and the NEXT run to reach the pause would sail
    straight through it with a stale selection.
    """
    with JOB_LOCK:
        if JOB["state"] != "selecting":
            return jsonify({"error": "No run is waiting for a selection."}), 409
        run_id = JOB["run_id"]

    payload = request.get_json(silent=True) or {}

    if payload.get("cancel"):
        SELECTION.update({"kept": [], "cancelled": True})
        SELECTION_READY.set()
        return jsonify({"cancelled": True})

    run = _load_run_or_404(run_id)
    known = {entry["id"] for entry in run.read_generation_manifest()}

    # Intersected with the manifest rather than trusted: the ids come from a
    # page that may have been open since before this run started.
    keep = [sid for sid in (payload.get("keep") or []) if sid in known]
    if not keep:
        return jsonify({
            "error": "Keep at least one sample, or cancel the run."
        }), 400

    # Stage 2c's two steps, chosen here because this is the first point at
    # which their cost is knowable: they act on the set being submitted.
    SELECTION.update({
        "kept": keep, "cancelled": False,
        "balance_labels": bool(payload.get("balance_labels")),
        "balance_classes": bool(payload.get("balance_classes")),
    })
    SELECTION_READY.set()
    return jsonify({"kept": len(keep), "dropped": len(known) - len(keep)})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Ask the running job to stop at its next safe point.

    Returns immediately -- this sets a flag, it does not wait. How long the
    job takes to notice depends on the stage: see STOP_REQUESTED for what each
    one finishes before leaving.

    A job parked for manual review is released at once through the selection
    handshake, since there is nothing in flight to finish.
    """
    with JOB_LOCK:
        if not job_is_running():
            return jsonify({"error": "No run is in progress."}), 409
        state = JOB["state"]

    STOP_REQUESTED.set()
    log_line("")
    log_line("Stop requested.")

    if state == "selecting":
        # Parked, waiting on a POST that is not coming. Release it through the
        # same path the Cancel button uses, or the worker would sit here until
        # the review timeout.
        SELECTION.update({"kept": [], "cancelled": True})
        SELECTION_READY.set()

    return jsonify({"stopping": True, "state": state})


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

    # Read the two files directly rather than through read_verified_manifest(),
    # which merges them: the review page has to distinguish "the judges let
    # this through" from "I kept this last time", and show both.
    admitted_ids = {
        entry["id"] for entry in
        (read_json(run.verified_manifest_file)
         if run.verified_manifest_file.exists() else [])
    }
    judged = run.verified_manifest_file.exists()

    selected_ids = {
        entry["id"] for entry in
        (read_json(run.selected_manifest_file)
         if run.selected_manifest_file.exists() else [])
    }
    reviewed = run.selected_manifest_file.exists()

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
        # Whether either stage has run at all. Without these the client cannot
        # tell "judged, and rejected this one" from "never judged", and both
        # arrive as admitted=False.
        "judged": judged,
        "reviewed": reviewed,
        "samples": [
            {
                **entry,
                "admitted": entry["id"] in admitted_ids,
                "selected": entry["id"] in selected_ids,
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
    """Every judge's verdict side by side, with the admission decision."""
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

    return jsonify({"runs": rows})


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
