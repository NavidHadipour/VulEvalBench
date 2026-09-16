"""Arm 1 -- static analysis detectors: Bandit, Ruff and Semgrep.

Both tools scan Python source for known-bad patterns and report findings with
line numbers. Neither produces a per-file "is this vulnerable" verdict, so this
module converts findings into the benchmark's verdict shape.

WHY THREE
    They span two axes -- analysis strategy, and implementation of a shared
    ruleset -- which is what makes the comparison worth drawing:

    Bandit walks the AST looking for known-bad calls -- `subprocess` with
    `shell=True`, `pickle.load`, a weak hash. It is syntactic: it asks whether
    a dangerous call appears anywhere in the file.

    Semgrep additionally performs taint analysis, tracking whether untrusted
    input actually *reaches* a dangerous sink. It can therefore separate a
    hardcoded path passed to `open()` from a request parameter passed to the
    same call. That is a dataflow question rather than a syntactic one, and it
    is precisely the distinction the Medium and Hard difficulty levels of this
    benchmark are built around.

    So where the two disagree, the disagreement is informative rather than
    noise: it usually marks a sample where reachability, not the presence of a
    dangerous call, decides the answer.

    Ruff reimplements Bandit's flake8-bandit ruleset in a separate Rust
    engine. It is deliberately the least independent of the three -- measured
    over the same 15 samples, Bandit agreed with Ruff at kappa 0.706 but with
    Semgrep at only 0.375. That redundancy is the point: where Bandit and Ruff
    disagree, the shared ruleset cannot be the explanation, so a rule's limits
    can be told apart from a tool's.

A NOTE ON SEMGREP AND WINDOWS
    Earlier semgrep releases had no Windows distribution -- `pip install
    semgrep` succeeded while installing no runnable binary, which is why an
    earlier draft of this project ran Ruff in its place. That is no longer
    true: semgrep 1.173 ships a working Windows executable, so all three now
    run side by side rather than one substituting for another.

RULESET AND NETWORK ACCESS
    Semgrep runs with `--config=p/python`, a curated registry ruleset. Those
    rules are fetched from semgrep.dev on first use and cached afterwards, so
    the first run on a machine needs network access. `run_semgrep` raises a
    clear error if the fetch fails, rather than returning zero findings --
    silently reporting "nothing found" would be indistinguishable from a tool
    that ran and found the files clean.

THE SCOPING DECISION -- which findings count
    A security linter reports everything it knows about, not only the sixteen
    categories this benchmark asks about. On the very first generated pair,
    both tools flagged "requests call without timeout" (CWE-400) on the
    vulnerable *and* the safe program. Counting that as "this file is
    vulnerable" would mark every sample positive and drive the false positive
    rate to 1.0 for a reason that has nothing to do with the question asked.

    So by default a finding only counts if its CWE is one of the benchmark's
    categories (`config.STATIC_SCOPE_TO_BENCHMARK_CWES`). Both numbers are
    recorded either way -- `findings_total` and `findings_in_scope` -- so the
    unscoped behaviour can be reported alongside, and the choice is visible
    rather than buried.

    This is a real methodological choice with a real effect on the headline
    numbers, and it should be stated explicitly in any write-up.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import config
import cwe_taxonomy
from detectors import build_result, make_prediction

# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------
# Executables are looked up next to the running interpreter rather than on
# PATH. Both tools are installed into the project virtualenv, and the web app
# may be launched by a process whose PATH does not include that venv's
# Scripts/bin directory -- resolving from sys.executable is what makes the
# tools findable from the browser as well as from a terminal.

def _tool_path(tool: str) -> Path | None:
    """Locate a console script installed alongside the current interpreter."""
    bin_dir = Path(sys.executable).parent
    for candidate in (bin_dir / f"{tool}.exe", bin_dir / tool):
        if candidate.exists():
            return candidate
    return None


def tool_available(tool: str) -> bool:
    """True if `tool` can be executed. Used to grey out unavailable tools."""
    return _tool_path(tool) is not None


# The registry ruleset semgrep runs with. `p/python` is the curated Python
# pack: broader than the security-only packs, but its findings carry CWE
# metadata, and the scoping rule below discards anything outside the
# benchmark's categories anyway.
SEMGREP_CONFIG = "p/python"

# Both tools report their own confidence per finding; these map onto the
# 0.0-1.0 scale the framework uses. The two happen to share vocabulary, so one
# table serves both.
TOOL_CONFIDENCE = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.35}


# ---------------------------------------------------------------------------
# Shared finding -> verdict logic
# ---------------------------------------------------------------------------

# The benchmark catalogue expanded down MITRE's ChildOf tree, computed once at
# import. Descendants only -- see cwe_taxonomy for why ancestors are excluded.
_CATALOGUE_WITH_DESCENDANTS = cwe_taxonomy.descendants_of(config.CWE_CATEGORIES)

# The two scoring policies, computed side by side on every run so neither is
# hidden. "strict" is exact-id matching; "hierarchy" also accepts a more
# specific child of a catalogue entry.
POLICY_STRICT = "strict"
POLICY_HIERARCHY = "hierarchy"


def _in_scope(cwe: str | None, policy: str = POLICY_HIERARCHY) -> bool:
    """True if a finding with this CWE should count towards a verdict.

    Args:
        cwe: the finding's category, or None.
        policy: `POLICY_STRICT` requires the id to be a catalogue entry
            exactly. `POLICY_HIERARCHY` also accepts any descendant of one, so
            a tool reporting CWE-276 for a CWE-732 sample is credited rather
            than penalised for being more specific.

    Disabling `config.STATIC_SCOPE_TO_BENCHMARK_CWES` makes every finding count
    under either policy, which is the raw out-of-the-box tool behaviour.
    """
    if not config.STATIC_SCOPE_TO_BENCHMARK_CWES:
        return True
    if policy == POLICY_STRICT:
        return cwe in config.CWE_CATEGORIES
    return cwe in _CATALOGUE_WITH_DESCENDANTS


def _verdict_from_findings(sample_id: str, findings: list[dict],
                           tool_name: str,
                           policy: str = POLICY_HIERARCHY) -> dict:
    """Turn one file's findings into one prediction.

    Args:
        sample_id: the sample being judged.
        findings: that file's findings, each with `cwe`, `code` and `message`.
        tool_name: used only in the reasoning text.

    Returns:
        A prediction in the common shape. When several in-scope findings exist,
        the one with the highest confidence is reported as *the* predicted CWE,
        because the benchmark asks for a single category per sample.
    """
    in_scope = [f for f in findings if _in_scope(f.get("cwe"), policy)]

    if not in_scope:
        # No relevant finding is a positive claim that the file is clean --
        # that is what a linter reporting nothing means.
        note = (
            f"{tool_name} reported no findings in the benchmark's categories"
            + (f" ({len(findings)} out-of-scope finding(s) ignored)"
               if findings else "")
        )
        return make_prediction(sample_id, False, None, 0.6, note)

    best = max(in_scope, key=lambda f: f.get("confidence", 0.0))
    detail = "; ".join(
        f"{f.get('code', '?')} {f.get('message', '')}".strip()
        for f in in_scope[:3]
    )
    return make_prediction(
        sample_id,
        True,
        best.get("cwe"),
        best.get("confidence", 0.6),
        f"{tool_name}: {detail}",
    )


def _empty_findings(manifest: list[dict]) -> dict[str, list[dict]]:
    """A findings map with an entry for every sample, so files with no findings
    still receive an explicit 'clean' verdict rather than being omitted."""
    return {entry["file"]: [] for entry in manifest}


def _package(tool_key: str, tool_name: str, manifest: list[dict],
             findings_by_file: dict[str, list[dict]], total: int,
             log, extra: dict) -> dict:
    """Build the result envelope, scoring under BOTH scoping policies.

    `predictions` uses the hierarchy-aware policy and is what evaluate.py
    scores by default. `predictions_strict` uses exact CWE matching and is
    carried alongside so the report can show both.

    Reporting both is deliberate. The hierarchy rule was introduced after
    noticing that exact matching penalised a tool for labelling findings more
    precisely than the benchmark catalogue does -- a change made in response to
    an observed result. Publishing only the more generous number would hide
    that; publishing both lets a reader see exactly what the policy is worth
    and decide for themselves.
    """
    predictions = [
        _verdict_from_findings(
            entry["id"], findings_by_file[entry["file"]], tool_name,
            POLICY_HIERARCHY,
        )
        for entry in manifest
    ]
    predictions_strict = [
        _verdict_from_findings(
            entry["id"], findings_by_file[entry["file"]], tool_name,
            POLICY_STRICT,
        )
        for entry in manifest
    ]

    all_findings = [f for fs in findings_by_file.values() for f in fs]
    in_hierarchy = sum(
        1 for f in all_findings if _in_scope(f["cwe"], POLICY_HIERARCHY)
    )
    in_strict = sum(
        1 for f in all_findings if _in_scope(f["cwe"], POLICY_STRICT)
    )

    log.info(
        f"    {total} finding(s); in benchmark categories: "
        f"{in_strict} exact, {in_hierarchy} including CWE descendants"
    )
    if in_hierarchy > in_strict:
        # Worth surfacing: these are findings the strict policy throws away
        # even though the tool named a child of a category we asked about.
        log.info(
            f"    {in_hierarchy - in_strict} finding(s) matched only via the "
            "MITRE CWE hierarchy"
        )

    result = build_result(tool_key, predictions, {
        "tool_version": _tool_version(extra.pop("executable")),
        "findings_total": total,
        "findings_in_scope": in_hierarchy,
        "findings_in_scope_strict": in_strict,
        "scoped_to_benchmark_cwes": config.STATIC_SCOPE_TO_BENCHMARK_CWES,
        "cwe_scope_policy": POLICY_HIERARCHY,
        "cwe_hierarchy_version": cwe_taxonomy.load_hierarchy().get("version"),
        **extra,
    })
    result["predictions_strict"] = predictions_strict
    return result


# ---------------------------------------------------------------------------
# Bandit
# ---------------------------------------------------------------------------

def run_bandit(run, manifest: list[dict], log) -> dict:
    """Run Bandit over a run's samples and convert its findings to verdicts.

    Bandit is invoked once over the whole samples directory rather than once
    per file: process start-up dominates its runtime on small files, so this is
    an order of magnitude faster on a few hundred samples.

    Args:
        run: the `runstore.Run` whose samples should be analysed.
        manifest: the samples to judge (the verified manifest, normally).
        log: logger for progress.

    Returns:
        A result dict in the common shape.
    """
    executable = _tool_path("bandit")
    if executable is None:
        raise RuntimeError(
            "bandit is not installed in this environment: pip install bandit"
        )

    log.info("  running Bandit")

    # `-f json` for machine-readable output, `-q` to suppress the progress
    # banner, `-r` to recurse into the samples directory.
    process = subprocess.run(
        [str(executable), "-f", "json", "-q", "-r", str(run.samples_dir)],
        capture_output=True, text=True, timeout=600,
    )

    # Bandit exits 1 when it finds issues, which is not an error condition
    # here -- findings are the entire point. Only an unparseable stdout is a
    # real failure.
    try:
        report = json.loads(process.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not parse Bandit output: {exc}. stderr: {process.stderr[:300]}"
        ) from exc

    findings_by_file = _empty_findings(manifest)
    total = 0

    for issue in report.get("results", []):
        # Bandit emits paths with mixed separators on Windows
        # ("dir/subdir\\sample_001.py"), so matching is done on the basename
        # rather than on the path as given.
        filename = Path(issue.get("filename", "").replace("\\", "/")).name
        if filename not in findings_by_file:
            continue

        cwe_id = (issue.get("issue_cwe") or {}).get("id")
        total += 1
        findings_by_file[filename].append({
            "cwe": f"CWE-{cwe_id}" if cwe_id else None,
            "code": issue.get("test_id", ""),
            "message": issue.get("issue_text", ""),
            "line": issue.get("line_number", 0),
            "confidence": TOOL_CONFIDENCE.get(
                issue.get("issue_confidence", "MEDIUM"), 0.6
            ),
        })

    return _package("static_bandit", "Bandit", manifest, findings_by_file,
                    total, log, {"executable": executable})


# ---------------------------------------------------------------------------
# Semgrep
# ---------------------------------------------------------------------------

# Semgrep reports CWEs as a list of descriptive strings, e.g.
# "CWE-918: Server-Side Request Forgery (SSRF)". Only the identifier before the
# colon is comparable with the benchmark's categories.
_CWE_ID_RE = re.compile(r"(CWE-\d+)")


def run_semgrep(run, manifest: list[dict], log) -> dict:
    """Run Semgrep over a run's samples and convert its findings to verdicts.

    Invoked once over the whole samples directory: semgrep's start-up cost
    (loading and compiling the ruleset) dominates its runtime on small files,
    so per-file invocation would be dramatically slower.

    Unlike Bandit, semgrep performs taint analysis, so a finding here generally
    means untrusted input was traced to a dangerous sink rather than merely
    that a dangerous call exists.
    """
    executable = _tool_path("semgrep")
    if executable is None:
        raise RuntimeError(
            "semgrep is not installed in this environment: pip install semgrep\n"
            "On older releases semgrep had no Windows build; 1.173 or newer does."
        )

    log.info(f"  running Semgrep ({SEMGREP_CONFIG})")

    process = subprocess.run(
        [
            str(executable), "scan",
            f"--config={SEMGREP_CONFIG}",
            "--json",
            "--quiet",
            # Sample directories are inside the project, which is gitignored;
            # without this semgrep would skip every file it was asked to scan.
            "--no-git-ignore",
            # Samples are standalone programs, so no project config should
            # influence their analysis.
            "--metrics=off",
            str(run.samples_dir),
        ],
        capture_output=True, text=True, timeout=900,
    )

    try:
        report = json.loads(process.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not parse Semgrep output: {exc}. "
            f"stderr: {process.stderr[:400]}"
        ) from exc

    # A ruleset that failed to download yields zero findings, which is
    # indistinguishable from "every file is clean" once converted to verdicts.
    # Fail loudly instead of quietly scoring a tool that never ran.
    fatal = [
        e for e in report.get("errors", [])
        if e.get("level") in ("error", "fatal")
    ]
    if fatal and not report.get("results"):
        raise RuntimeError(
            "Semgrep produced no results and reported errors -- the ruleset "
            f"may have failed to download. First error: {str(fatal[0])[:300]}"
        )

    findings_by_file = _empty_findings(manifest)
    total = 0

    for issue in report.get("results", []):
        # Semgrep emits native separators, so match on the basename.
        filename = Path(issue.get("path", "").replace("\\", "/")).name
        if filename not in findings_by_file:
            continue

        extra = issue.get("extra", {}) or {}
        metadata = extra.get("metadata", {}) or {}

        # Take the first parseable CWE; a rule occasionally lists several, and
        # the benchmark asks for a single category per sample.
        cwe = None
        for entry in (metadata.get("cwe") or []):
            match = _CWE_ID_RE.search(str(entry))
            if match:
                cwe = match.group(1)
                break

        total += 1
        findings_by_file[filename].append({
            "cwe": cwe,
            "code": issue.get("check_id", "").split(".")[-1],
            "message": extra.get("message", "").strip().split("\n")[0][:160],
            "line": (issue.get("start") or {}).get("line", 0),
            "confidence": TOOL_CONFIDENCE.get(
                str(metadata.get("confidence", "MEDIUM")).upper(), 0.6
            ),
        })

    return _package("static_semgrep", "Semgrep", manifest, findings_by_file,
                    total, log,
                    {"executable": executable, "ruleset": SEMGREP_CONFIG})


# ---------------------------------------------------------------------------
# Ruff
# ---------------------------------------------------------------------------
# Bandit and Semgrep both report a CWE with each finding. Ruff reports only its
# own rule code (S608, S602, ...), so the codes relevant to this benchmark's
# categories are mapped here. Codes absent from this table still count towards
# `findings_total` but carry no CWE, so under the default scoping rule they do
# not by themselves mark a file vulnerable.
RUFF_CWE_MAP = {
    # Command injection
    "S601": "CWE-78", "S602": "CWE-78", "S603": "CWE-78", "S604": "CWE-78",
    "S605": "CWE-78", "S606": "CWE-78", "S607": "CWE-78", "S609": "CWE-78",
    # SQL injection
    "S608": "CWE-89", "S610": "CWE-89", "S611": "CWE-89",
    # Code injection
    "S102": "CWE-94", "S307": "CWE-94",
    # Deserialization
    "S301": "CWE-502", "S302": "CWE-502", "S506": "CWE-502",
    # Hard-coded credentials
    "S105": "CWE-798", "S106": "CWE-798", "S107": "CWE-798",
    # Broken crypto
    "S303": "CWE-327", "S304": "CWE-327", "S305": "CWE-327",
    "S324": "CWE-327", "S502": "CWE-327", "S503": "CWE-327",
    # Weak randomness
    "S311": "CWE-330",
    # XML external entities
    "S313": "CWE-611", "S314": "CWE-611", "S315": "CWE-611",
    "S316": "CWE-611", "S317": "CWE-611", "S318": "CWE-611",
    "S319": "CWE-611", "S320": "CWE-611",
    # Certificate validation
    "S323": "CWE-295", "S501": "CWE-295", "S507": "CWE-295",
    # Insecure temporary files
    "S108": "CWE-377", "S306": "CWE-377",
    # Overly broad permissions
    "S103": "CWE-732",
    # Cross-site scripting
    "S308": "CWE-79", "S701": "CWE-79", "S702": "CWE-79",
    # Server-side request forgery / unvalidated URL open
    "S310": "CWE-918",
}


def run_ruff(run, manifest: list[dict], log) -> dict:
    """Run Ruff's flake8-bandit ruleset over a run's samples.

    WHY RUFF IS HERE ALONGSIDE BANDIT
        Ruff reimplements the flake8-bandit ruleset in a separate Rust engine.
        It is therefore the *least* independent of the three: measured over the
        same 15 samples, Bandit and Ruff agreed at kappa 0.706, while Bandit and
        Semgrep agreed at only 0.375.

        That redundancy is exactly why it earns a place. Two implementations of
        one ruleset let you separate a limitation of the *rule* from a
        limitation of the *tool*: where Bandit and Ruff differ, the ruleset is
        not the explanation. Ruff also scored well on its own -- it beat Semgrep
        on F1 (0.800 vs 0.500) on that run -- so it is a strong detector, not
        merely a control.

    `--select S` restricts Ruff to the security ruleset; without it Ruff would
    also report style and import-ordering issues, which are not vulnerabilities
    and would swamp the findings list.
    """
    executable = _tool_path("ruff")
    if executable is None:
        raise RuntimeError(
            "ruff is not installed in this environment: pip install ruff"
        )

    log.info("  running Ruff (S ruleset)")

    process = subprocess.run(
        [
            str(executable), "check",
            "--select", "S",
            "--output-format", "json",
            # The cache keys on path and mtime; disabling it removes any chance
            # of a stale verdict being served for a rewritten sample.
            "--no-cache",
            # Samples are standalone programs, not part of this project's
            # package, so project config must not influence their analysis.
            "--isolated",
            str(run.samples_dir),
        ],
        capture_output=True, text=True, timeout=600,
    )

    try:
        issues = json.loads(process.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not parse Ruff output: {exc}. stderr: {process.stderr[:300]}"
        ) from exc

    findings_by_file = _empty_findings(manifest)
    total = 0

    for issue in issues:
        filename = Path(issue.get("filename", "").replace("\\", "/")).name
        if filename not in findings_by_file:
            continue

        code = issue.get("code", "")
        total += 1
        findings_by_file[filename].append({
            "cwe": RUFF_CWE_MAP.get(code),
            "code": code,
            "message": issue.get("message", ""),
            "line": (issue.get("location") or {}).get("row", 0),
            # Ruff has no confidence model -- a rule either matches or it does
            # not. A single mid-range constant is reported rather than inventing
            # a spread, so the number is not mistaken for a real signal.
            "confidence": 0.7,
        })

    return _package("static_ruff", "Ruff", manifest, findings_by_file,
                    total, log, {"executable": executable,
                                 "ruleset": "flake8-bandit S"})


def _tool_version(executable: Path) -> str:
    """Best-effort version string, recorded so a run stays reproducible.

    Returns "unknown" rather than raising: a missing version string is not a
    reason to fail a detection run.
    """
    try:
        out = subprocess.run(
            [str(executable), "--version"],
            capture_output=True, text=True, timeout=30,
        )
        return (out.stdout or out.stderr).strip().splitlines()[0][:80]
    except Exception:  # noqa: BLE001 - version reporting must never be fatal
        return "unknown"
