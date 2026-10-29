"""Semantics-preserving perturbation of benchmark samples -- robustness testing.

WHAT THIS IS FOR
    A detector should answer the same way about two programs that do the same
    thing. If renaming a variable or splitting a string literal flips its
    verdict, the detector was keying on surface form rather than on the
    weakness itself -- and its headline accuracy is measuring the wrong thing.

    This module rewrites a run's samples without changing what they do, so the
    tools can be re-measured on the altered copies. The gap between the two
    scores is the robustness result.

PROVENANCE -- the six transformations are not invented here
    They are the ones applied in

        T. Farasat, A. Bouzid and J. Posegga, "Adversarial Evaluation of
        Machine Learning-Based Python Source Code Vulnerability Detectors",
        CEUR Workshop Proceedings Vol-4075.

    which found that these alone collapse two strong Python detectors:

        VUDENC LSTM                 accuracy 97.8%  ->  59.6%
        Farasat & Posegga BiLSTM    accuracy 98.8%  ->  40.4%

    That paper evaluates XSS only, on the ML arm only. Running the same
    transformations through this framework extends it on both axes at once:
    every CWE category the generator produces, and all three detector arms
    rather than just the learned models. The static analysers and the LLMs have
    never been measured against it.

    The wider argument for label-preserving evaluation is Risse & Bohme,
    "Uncovering the Limits of Machine Learning for Automatic Vulnerability
    Detection", USENIX Security 2024: if a change does not alter the
    vulnerability, the prediction should not change either.

WHY A NEW RUN RATHER THAN AN EDIT IN PLACE
    Perturbing a run writes a NEW run containing the altered samples, linked
    back to the original by `derived_from`. The original is never touched.

    That keeps the append-only promise the whole store is built on, and it
    means the clean and perturbed results are two ordinary runs -- so the
    existing comparison machinery shows them side by side with no special
    case anywhere in evaluate.py or the web layer.

HOW "SEMANTICS-PRESERVING" IS ENFORCED
    Claiming a rewrite is harmless is not the same as it being harmless, and a
    transformation that silently breaks a program would quietly destroy the
    dataset it was meant to test. So every perturbed file must clear three
    gates before it is accepted:

      1. it parses (`ast.parse`);
      2. `generate.check_runtime_soundness` reports nothing new -- the same
         pyflakes and call-signature check the generator's own output faces;
      3. at least one transformation actually applied.

    A file failing any gate is written out UNCHANGED and recorded as skipped.
    Degrading to the original is always safe; shipping a broken program is not.

    The judging stage is the second line of defence, and it is why the
    supervisor placed this option before it: the transformations are supposed
    to preserve the label, so re-judging the perturbed samples *tests* that
    claim instead of assuming it. A sample the judges no longer agree on is
    dropped, exactly as any other unreliable label would be.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path

import config
import runstore
from common import get_logger, write_json

# Python builtins and common module names must never be renamed: rebinding
# `len` or `os` changes behaviour, and renaming an imported module breaks the
# import that defines it.
_PROTECTED_NAMES = frozenset(dir(__builtins__) if isinstance(__builtins__, dict)
                             else dir(__builtins__)) | {
    "self", "cls", "__name__", "__main__", "__file__", "__doc__",
}


# ---------------------------------------------------------------------------
# Transformation 1 -- identifier renaming
# ---------------------------------------------------------------------------

class _LocalNameCollector(ast.NodeVisitor):
    """Collect names that are safe to rename: locals bound inside functions.

    Module-level names are excluded because anything else in the file -- or an
    importing module -- may refer to them. Parameters are excluded because a
    caller may pass them by keyword. Attributes are never touched: `x.send`
    and `y.send` are unrelated, and renaming one would rename both.
    """

    def __init__(self) -> None:
        self.candidates: set[str] = set()
        self._depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Name(self, node: ast.Name) -> None:
        if (self._depth > 0 and isinstance(node.ctx, ast.Store)
                and node.id not in _PROTECTED_NAMES
                and not node.id.startswith("__")):
            self.candidates.add(node.id)
        self.generic_visit(node)


class _Renamer(ast.NodeTransformer):
    """Apply a name mapping to every Name node, leaving attributes alone."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node


def rename_identifiers(tree: ast.AST, rng: random.Random) -> bool:
    """Rename local variables to neutral names. Returns True if anything changed.

    The replacement names are deliberately uninformative (`var_a`, `var_b`).
    That is the point of the transformation: a detector keying on a variable
    called `user_input` or `unsafe_query` loses that cue, while the dataflow
    the vulnerability actually depends on is untouched.
    """
    collector = _LocalNameCollector()
    collector.visit(tree)

    # Parameters are bound by the caller, so renaming them can break keyword
    # calls. Collect and exclude them.
    params: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in (args.posonlyargs + args.args + args.kwonlyargs
                        + ([args.vararg] if args.vararg else [])
                        + ([args.kwarg] if args.kwarg else [])):
                params.add(arg.arg)

    # Names bound at module level, or by an import, are visible outside any
    # single function and are left alone for the same reason as parameters.
    module_level: set[str] = set()
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                module_level.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    module_level.add(target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            module_level.add(node.name)

    renameable = sorted(collector.candidates - params - module_level)
    if not renameable:
        return False

    rng.shuffle(renameable)
    mapping = {
        name: f"var_{chr(ord('a') + index % 26)}{index // 26 or ''}"
        for index, name in enumerate(renameable)
    }
    # A generated name that already exists in the file would merge two
    # distinct variables into one, which is the one way this transformation
    # could change behaviour.
    existing = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    mapping = {k: v for k, v in mapping.items() if v not in existing}
    if not mapping:
        return False

    _Renamer(mapping).visit(tree)
    return True


# ---------------------------------------------------------------------------
# Transformations 2 and 3 -- string literal rewriting
# ---------------------------------------------------------------------------

def _protected_strings(tree: ast.AST) -> set[int]:
    """Ids of Constant nodes that must stay literal string constants.

    Two kinds, for different reasons:

      - DOCSTRINGS. Rewriting the first statement of a module, class or
        function into an expression stops it being a docstring.

      - F-STRING PARTS. The literal segments of an f-string live inside a
        JoinedStr, whose children may only be Constant or FormattedValue.
        Replacing one with a concatenation produces an AST that `ast.unparse`
        refuses with "Unexpected node inside JoinedStr" -- which silently cost
        a third of the samples their perturbation before this was caught.
    """
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                found.add(id(body[0].value))
        elif isinstance(node, ast.JoinedStr):
            for part in ast.walk(node):
                if isinstance(part, ast.Constant):
                    found.add(id(part))
    return found


class _StringSplitter(ast.NodeTransformer):
    """Split string literals into concatenations of two halves.

    `"<script>"` becomes `"<scr" + "ipt>"`. The value is identical at run time,
    but a detector matching the literal `<script>` in the source text no longer
    sees it. This is the transformation that should separate the pattern-based
    static analysers from the taint-based ones.
    """

    def __init__(self, skip: set[int], rng: random.Random) -> None:
        self.skip = skip
        self.rng = rng
        self.changed = False

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if (not isinstance(node.value, str) or id(node) in self.skip
                or len(node.value) < 4):
            return node
        cut = self.rng.randint(2, len(node.value) - 2)
        self.changed = True
        return ast.BinOp(
            left=ast.Constant(value=node.value[:cut]),
            op=ast.Add(),
            right=ast.Constant(value=node.value[cut:]),
        )


class _AsciiEncoder(ast.NodeTransformer):
    """Rewrite a short string literal as a join of chr() calls.

    `"id"` becomes `"".join([chr(105), chr(100)])`. Semantically identical and
    far harder to pattern-match. Restricted to short literals because the
    expansion is verbose and would otherwise dominate the file.
    """

    def __init__(self, skip: set[int], rng: random.Random,
                 max_length: int = 12) -> None:
        self.skip = skip
        self.rng = rng
        self.max_length = max_length
        self.changed = False

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if (not isinstance(node.value, str) or id(node) in self.skip
                or not node.value or len(node.value) > self.max_length
                or not node.value.isascii()):
            return node
        # Only some literals, or every string in the file becomes unreadable
        # and the result stops resembling plausible source code.
        if self.rng.random() > 0.5:
            return node
        self.changed = True
        return ast.Call(
            func=ast.Attribute(
                value=ast.Constant(value=""), attr="join", ctx=ast.Load()
            ),
            args=[ast.List(
                elts=[
                    ast.Call(
                        func=ast.Name(id="chr", ctx=ast.Load()),
                        args=[ast.Constant(value=ord(ch))],
                        keywords=[],
                    )
                    for ch in node.value
                ],
                ctx=ast.Load(),
            )],
            keywords=[],
        )


def obfuscate_strings(tree: ast.AST, rng: random.Random) -> bool:
    """Split string literals into concatenated halves."""
    transformer = _StringSplitter(_protected_strings(tree), rng)
    transformer.visit(tree)
    return transformer.changed


def encode_ascii(tree: ast.AST, rng: random.Random) -> bool:
    """Rewrite short string literals as chr() sequences."""
    transformer = _AsciiEncoder(_protected_strings(tree), rng)
    transformer.visit(tree)
    return transformer.changed


# ---------------------------------------------------------------------------
# Transformation 4 -- logic refactoring
# ---------------------------------------------------------------------------

class _LogicRefactorer(ast.NodeTransformer):
    """Rewrite conditions into equivalent forms.

    Only rewrites whose equivalence is unconditional are included:
      `x.find(y) != -1`  ->  `y in x`
      `x.find(y) == -1`  ->  `y not in x`
      `not (a == b)`     ->  `a != b`

    `.find(...) >= 0` is deliberately absent: it is equivalent for `str` but
    not for every object defining `find`, and the transformation must hold for
    any program the generator writes.
    """

    def __init__(self) -> None:
        self.changed = False

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if (len(node.ops) == 1 and isinstance(node.left, ast.Call)
                and isinstance(node.left.func, ast.Attribute)
                and node.left.func.attr == "find"
                and len(node.left.args) == 1):
            comparator = node.comparators[0]
            is_minus_one = (
                isinstance(comparator, ast.UnaryOp)
                and isinstance(comparator.op, ast.USub)
                and isinstance(comparator.operand, ast.Constant)
                and comparator.operand.value == 1
            )
            if is_minus_one and isinstance(node.ops[0], (ast.NotEq, ast.Eq)):
                self.changed = True
                operator = (ast.In() if isinstance(node.ops[0], ast.NotEq)
                            else ast.NotIn())
                return ast.Compare(
                    left=node.left.args[0],
                    ops=[operator],
                    comparators=[node.left.func.value],
                )
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if (isinstance(node.op, ast.Not)
                and isinstance(node.operand, ast.Compare)
                and len(node.operand.ops) == 1
                and isinstance(node.operand.ops[0], ast.Eq)):
            self.changed = True
            return ast.Compare(
                left=node.operand.left,
                ops=[ast.NotEq()],
                comparators=node.operand.comparators,
            )
        return node


def refactor_logic(tree: ast.AST, rng: random.Random) -> bool:
    """Rewrite conditions into equivalent forms."""
    transformer = _LogicRefactorer()
    transformer.visit(tree)
    return transformer.changed


# ---------------------------------------------------------------------------
# Transformations 5 and 6 -- inserted statements
# ---------------------------------------------------------------------------

def _insert_into_functions(tree: ast.AST, rng: random.Random,
                           make_statement) -> bool:
    """Insert a generated statement at a random point in each function body.

    Never inserted before a docstring, which would stop it being one, and
    never into an empty body.
    """
    changed = False
    for index, node in enumerate(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        body = node.body
        if not body:
            continue
        first = 1 if (isinstance(body[0], ast.Expr)
                      and isinstance(body[0].value, ast.Constant)
                      and isinstance(body[0].value.value, str)) else 0
        # Insert before a statement rather than after the last one, so the
        # insertion can never land after a `return` where it is unreachable
        # and pyflakes would flag it.
        position = rng.randint(first, max(first, len(body) - 1))
        body.insert(position, make_statement(index))
        changed = True
    return changed


def insert_noops(tree: ast.AST, rng: random.Random) -> bool:
    """Insert semantically neutral assignments, e.g. `noop_0 = None`."""
    return _insert_into_functions(
        tree, rng,
        lambda i: ast.Assign(
            targets=[ast.Name(id=f"noop_{i}", ctx=ast.Store())],
            value=ast.Constant(value=None),
        ),
    )


def insert_dummy_computations(tree: ast.AST, rng: random.Random) -> bool:
    """Insert irrelevant arithmetic, e.g. `calc_0 = 7 * 3 + 1`."""
    return _insert_into_functions(
        tree, rng,
        lambda i: ast.Assign(
            targets=[ast.Name(id=f"calc_{i}", ctx=ast.Store())],
            value=ast.BinOp(
                left=ast.Constant(value=rng.randint(2, 20)),
                op=ast.Mult(),
                right=ast.Constant(value=rng.randint(2, 20)),
            ),
        ),
    )


# The registry, in the order the paper lists them. The keys are what the web
# interface sends and what the run metadata records.
PERTURBATIONS = {
    "rename_identifiers": rename_identifiers,
    "obfuscate_strings": obfuscate_strings,
    "encode_ascii": encode_ascii,
    "refactor_logic": refactor_logic,
    "insert_noops": insert_noops,
    "insert_dummy_computations": insert_dummy_computations,
}

PERTURBATION_LABELS = {
    "rename_identifiers": "Identifier renaming",
    "obfuscate_strings": "String obfuscation",
    "encode_ascii": "ASCII encoding",
    "refactor_logic": "Logic refactoring",
    "insert_noops": "No-op insertion",
    "insert_dummy_computations": "Dummy computations",
}

DEFAULT_PERTURBATIONS = list(PERTURBATIONS)


# ---------------------------------------------------------------------------
# Applying them to one program
# ---------------------------------------------------------------------------

def perturb_source(source: str, kinds: list[str],
                   rng: random.Random) -> tuple[str, list[str], list[str]]:
    """Rewrite one program, preserving what it does.

    Args:
        source: the original program text.
        kinds: which transformations to attempt, as PERTURBATIONS keys.
        rng: seeded, so a perturbed run is reproducible from its metadata.

    Returns:
        `(source, applied, problems)`. On any failure the ORIGINAL source is
        returned with the reason in `problems` -- degrading to the unchanged
        program is always safe, whereas shipping a broken one silently
        corrupts the dataset the perturbation was meant to test.
    """
    # Import here rather than at module scope: generate.py pulls in the
    # provider stack, and the web layer imports this module just to render the
    # checkbox labels.
    from generate import check_runtime_soundness

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return source, [], [f"original_does_not_parse:{exc.lineno}"]

    baseline = set(check_runtime_soundness(source))

    applied: list[str] = []
    for kind in kinds:
        function = PERTURBATIONS.get(kind)
        if function is None:
            continue
        try:
            if function(tree, rng):
                applied.append(kind)
        except Exception as exc:  # noqa: BLE001 - one bad rewrite, not a dead run
            return source, [], [f"{kind}_raised:{type(exc).__name__}"]

    if not applied:
        return source, [], ["no_transformation_applied"]

    ast.fix_missing_locations(tree)
    try:
        rewritten = ast.unparse(tree)
    except Exception as exc:  # noqa: BLE001
        return source, [], [f"unparse_failed:{type(exc).__name__}"]

    # Gate 1: it must still parse.
    try:
        ast.parse(rewritten)
    except SyntaxError as exc:
        return source, [], [f"result_does_not_parse:{exc.lineno}"]

    # Gate 2: it must be no less runnable than the original. Comparing against
    # the baseline rather than requiring zero flags matters -- some generated
    # samples already carry a flag, and the perturbation must not be blamed
    # for a problem it did not introduce.
    introduced = sorted(set(check_runtime_soundness(rewritten)) - baseline)
    if introduced:
        return source, [], [f"soundness_regressed:{','.join(introduced)}"]

    return rewritten, applied, []


# ---------------------------------------------------------------------------
# Applying them to a whole run
# ---------------------------------------------------------------------------

def perturb_run(source_run_id: str, kinds: list[str] | None = None,
                seed: int | None = None, label: str = "",
                on_progress=None) -> dict:
    """Create a new run holding perturbed copies of another run's samples.

    The source run is never modified. The new run carries `derived_from`, so
    the pair can always be found again, and it behaves like any other run --
    which is what lets the existing comparison views show clean against
    perturbed with no special case.

    Args:
        source_run_id: the run whose samples are rewritten.
        kinds: which transformations to attempt. Defaults to all six.
        seed: RNG seed. Recorded in metadata so the run is reproducible.
        label: optional free-text note.
        on_progress: callback receiving log lines.

    Returns:
        A summary dict with the new `run_id` and the per-sample outcome.

    Raises:
        SystemExit: if the source run has no samples to perturb.
    """
    kinds = [k for k in (kinds or DEFAULT_PERTURBATIONS) if k in PERTURBATIONS]
    if not kinds:
        kinds = list(DEFAULT_PERTURBATIONS)

    source_run = runstore.load_run(source_run_id)
    # Prefer the judged manifest: perturbing samples the judges already
    # rejected would waste effort on programs that are leaving anyway.
    manifest = (source_run.read_verified_manifest()
                or source_run.read_generation_manifest())
    if not manifest:
        raise SystemExit(
            f"Run {source_run_id} has no samples to perturb."
        )

    seed = config.SEED if seed is None else seed
    rng = random.Random(seed)

    # One generation on from whatever the source run was, so perturbing an
    # original gives v2.0 and perturbing that gives v3.0. The chain is then
    # readable straight off the run id, which is what the suffix is for.
    source_version = int(source_run.metadata().get("version", 1) or 1)
    version = source_version + 1

    new_run = runstore.create_run(
        source="perturbed",
        label=label or f"perturbed copy of {source_run_id}",
        config_snapshot={
            "derived_from": source_run_id,
            "perturbations": kinds,
            "seed": seed,
        },
        version=version,
        # Marks the run id itself, so a perturbed dataset can never be
        # mistaken for a clean one in a file listing, a log name or a table.
        tag="noisy",
    )
    log = get_logger(new_run.run_id, on_progress)
    log.info(f"Run {new_run.run_id}  (samples version {version})")
    log.info(f"Perturbing {len(manifest)} samples from {source_run_id}"
             f" (version {source_version})")
    log.info(f"Transformations: {', '.join(kinds)}")

    new_run.samples_dir.mkdir(parents=True, exist_ok=True)
    new_manifest: list[dict] = []
    unchanged = 0

    for entry in manifest:
        source = source_run.read_sample_code(entry)
        rewritten, applied, problems = perturb_source(source, kinds, rng)

        (new_run.samples_dir / entry["file"]).write_text(
            rewritten, encoding="utf-8"
        )
        if not applied:
            unchanged += 1
            log.warning(
                f"  {entry['id']}: unchanged ({'; '.join(problems) or 'no reason'})"
            )

        # The label and CWE are carried over deliberately: the transformations
        # preserve semantics, so the ground truth is unchanged by construction.
        # Judging the perturbed run is what tests that assumption rather than
        # trusting it.
        new_manifest.append({
            **entry,
            "perturbations_applied": applied,
            "perturbation_problems": problems,
            "derived_from_run": source_run_id,
        })

    write_json(new_run.generation_manifest_file, new_manifest)

    changed = len(new_manifest) - unchanged
    log.info(f"Perturbed {changed}/{len(new_manifest)} samples"
             + (f"; {unchanged} left unchanged" if unchanged else ""))

    counts = {
        kind: sum(1 for e in new_manifest if kind in e["perturbations_applied"])
        for kind in kinds
    }
    for kind, count in counts.items():
        log.info(f"    {PERTURBATION_LABELS.get(kind, kind):<24} {count}")

    new_run.update_metadata(
        status="perturbed",
        source="perturbed",
        derived_from=source_run_id,
        source_version=source_version,
        perturbations=kinds,
        perturbation_counts=counts,
        perturbation_seed=seed,
        generated_samples=len(new_manifest),
        samples_unchanged=unchanged,
    )

    return {
        "run_id": new_run.run_id,
        "version": version,
        "derived_from": source_run_id,
        "samples": len(new_manifest),
        "changed": changed,
        "unchanged": unchanged,
        "counts": counts,
    }


def main() -> None:
    """Perturb a run from the command line.

    Usage:
        python src/perturb.py <run_id> [transformation ...]
    """
    import sys

    if len(sys.argv) < 2:
        runs = runstore.list_runs()
        print(__doc__.splitlines()[0])
        print("\nUsage: python src/perturb.py <run_id> [transformation ...]")
        print("\nTransformations:")
        for key, name in PERTURBATION_LABELS.items():
            print(f"  {key:<28} {name}")
        print("\nRuns:")
        for run in runs[:10]:
            print(f"  {run.run_id}")
        raise SystemExit(1)

    summary = perturb_run(
        sys.argv[1],
        kinds=sys.argv[2:] or None,
        on_progress=lambda line: print(line),
    )
    print(f"\nNew run: {summary['run_id']}")
    print(f"  {summary['changed']}/{summary['samples']} samples perturbed")


if __name__ == "__main__":
    main()
