"""Every third-party import that runs when the packaged `sestrav` CLI's four
subcommands run must resolve to a declared core dependency.

Scope is deliberately the CLI's reachable set, not the whole repo: most of
`src/` is research/analysis tooling run from a source checkout (the `dev`
extra), which this project has never promised works from a bare
`pip install sestrav`. The reachable set below is derived, not listed: it is
seeded with `src/cli.py` itself (the console script imports it, so its module
scope runs) and with the imports inside its four `cmd_*` functions, then
follows runtime first-party imports transitively, adding every package
`__init__.py` above each module reached, since importing `src.x` runs
`src/__init__.py` first. Third-party imports inside the `cmd_*` bodies are
checked as well as module-scope ones. This guard was added after a
module-scope `import matplotlib` in `functions/stage4_immunogenicity_scoring.py`
(declared only in the `demo` extra) turned out to be the eighth instance of
the class the 2026-08-14 AST audit (see the dated comment above
`[project].dependencies` in pyproject.toml) was meant to close. That audit
checked for presence anywhere in `pyproject.toml`, not presence in
`[project].dependencies` specifically, so a package declared in the wrong
extra passed it silently. This test checks the narrower, correct condition.

Traversal policy (one walker, `_runtime_imports`, used for every scope):
an explicit-stack walk that PRUNES SUBTREES. A pruned region's children are
never pushed, so nothing beneath it is yielded. It prunes exactly three
things: the body and handlers of a `try` with a handler naming ImportError or
ModuleNotFoundError (its `else` and `finally` are still walked, since the
handler does not guard them); the body of `if TYPE_CHECKING:` (its `else` is
still walked); and the body of a nested `def`/`async def`/`lambda`, which runs
only when called. Every other `if` (including `if __name__ == "__main__":`),
`try` (including `except Exception:`), loop, `with` and class body is walked:
the conservative direction, since a false finding is loud and a missed one is
silent.
This is NOT `for node in ast.walk(tree): ... continue`, which skips a node but
still descends into its subtree; the walker test below builds an input on
which the two disagree.
"""

from __future__ import annotations

import ast
from collections import deque
from collections.abc import Iterator
import pathlib
import re
import shutil
import sys
import tomllib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CLI_RELATIVE = "src/cli.py"

# Import name -> PyPI distribution name, for the handful where they differ.
DIST_NAME_OVERRIDES = {
    "ahocorasick": "pyahocorasick",
    "bio": "biopython",
    "sklearn": "scikit-learn",
    "torch_geometric": "torch-geometric",
    "yaml": "pyyaml",
}

# First-party top-level packages; imports of these are never external.
INTRA_REPO = {"src", "functions", "sestrav", "tests", "tools", "scripts", "app", "api"}

# Exception names whose handler marks a `try` as an import guard.
IMPORT_ERROR_NAMES = {"ImportError", "ModuleNotFoundError"}


def _normalize(name: str) -> str:
    """PEP 503 distribution-name normalization, for comparing across separators/case."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _core_dependencies(root: pathlib.Path) -> set[str]:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    names = set()
    for requirement in data["project"]["dependencies"]:
        name = re.split(r"[<>=!~\s;]", requirement, maxsplit=1)[0]
        names.add(_normalize(name))
    return names


def _imported_modules(node: ast.AST) -> list[str]:
    """Absolute module names imported by one AST import node."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and not node.level and node.module:
        return [node.module]
    return []


def _exception_names(expr: ast.expr | None) -> set[str]:
    """Names an `except` clause matches, as written (`E`, `mod.E`, or a tuple of them)."""
    if expr is None:
        return set()
    if isinstance(expr, ast.Tuple):
        return set().union(*(_exception_names(element) for element in expr.elts))
    if isinstance(expr, ast.Name):
        return {expr.id}
    if isinstance(expr, ast.Attribute):
        return {expr.attr}
    return set()


def _is_import_guard(node: ast.Try | ast.TryStar) -> bool:
    return any(_exception_names(handler.type) & IMPORT_ERROR_NAMES for handler in node.handlers)


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _runtime_imports(statements: list[ast.stmt]) -> Iterator[ast.Import | ast.ImportFrom]:
    """Yield the import statements that execute when `statements` execute.

    Subtree-pruning walk; see the module docstring for the exact policy.
    """
    pending: list[ast.AST] = list(reversed(statements))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, (ast.Try, ast.TryStar)) and _is_import_guard(node):
            children: list[ast.AST] = [*node.orelse, *node.finalbody]
        elif isinstance(node, ast.If) and _is_type_checking(node.test):
            children = list(node.orelse)
        else:
            children = list(ast.iter_child_nodes(node))
        pending.extend(reversed(children))


def _runtime_modules(statements: list[ast.stmt]) -> list[str]:
    """Absolute module names imported by the runtime imports in `statements`."""
    return [module for node in _runtime_imports(statements) for module in _imported_modules(node)]


def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _first_party_path(root: pathlib.Path, module: str) -> str | None:
    """Resolve an absolute first-party import to its repository-relative file."""
    relative = module.replace(".", "/")
    candidates = (pathlib.Path(f"{relative}.py"), pathlib.Path(relative) / "__init__.py")
    for candidate in candidates:
        if (root / candidate).is_file():
            return candidate.as_posix()
    return None


def _package_inits(root: pathlib.Path, relative: str) -> list[str]:
    """Every existing `__init__.py` Python runs before it can run `relative`."""
    inits = []
    parts = pathlib.PurePosixPath(relative).parent.parts
    for depth in range(1, len(parts) + 1):
        init = "/".join(parts[:depth]) + "/__init__.py"
        if init != relative and (root / init).is_file():
            inits.append(init)
    return inits


def _cli_command_imports(root: pathlib.Path) -> list[tuple[str, str]]:
    """(`cmd_*` name, module) for every runtime import inside the CLI's commands."""
    tree = _parse(root / CLI_RELATIVE)
    found = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("cmd_"):
            continue
        found.extend((node.name, module) for module in _runtime_modules(node.body))
    return found


def _is_first_party(module: str) -> bool:
    return module.split(".", maxsplit=1)[0] in INTRA_REPO


def _cli_closure(root: pathlib.Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(reachable modules, unresolved first-party imports) for the CLI under `root`.

    Unresolved imports are returned rather than raised: this runs at module scope
    for the real tree, where raising aborts COLLECTION, and a collection error
    means zero tests ran rather than one test going red.
    """
    unresolved: list[str] = []

    def resolve(modules: list[str]) -> list[str]:
        paths = []
        for module in modules:
            if not _is_first_party(module):
                continue
            path = _first_party_path(root, module)
            if path is None:
                unresolved.append(module)
            else:
                paths.append(path)
        return paths

    seeds = [CLI_RELATIVE, *resolve([module for _, module in _cli_command_imports(root)])]
    pending = deque(sorted(set(seeds)))
    reachable: set[str] = set()
    while pending:
        relative = pending.popleft()
        if relative in reachable:
            continue
        reachable.add(relative)
        imported = _runtime_modules(_parse(root / relative).body)
        following = set(resolve(imported)) | set(_package_inits(root, relative))
        pending.extend(sorted(following - reachable))
    return tuple(sorted(reachable)), tuple(sorted(set(unresolved)))


REACHABLE_MODULES, UNRESOLVED_FIRST_PARTY = _cli_closure(REPO_ROOT)


def _third_party_names(modules: list[str]) -> set[str]:
    """Top-level third-party names among absolute module names."""
    stdlib = set(sys.stdlib_module_names)
    names = {module.split(".")[0] for module in modules}
    return {name for name in names if name not in stdlib and name not in INTRA_REPO}


def _undeclared(declared: set[str], where: str, names: set[str]) -> list[str]:
    missing = []
    for name in sorted(names):
        dist = DIST_NAME_OVERRIDES.get(name.lower(), name)
        if _normalize(dist) not in declared:
            missing.append(f"{where}: {name} (-> {dist})")
    return missing


def _undeclared_imports(root: pathlib.Path) -> list[str]:
    """Every runtime third-party import on the CLI path that is not a core dependency."""
    declared = _core_dependencies(root)
    reachable, _ = _cli_closure(root)
    missing = []
    for relative in reachable:
        modules = _runtime_modules(_parse(root / relative).body)
        missing += _undeclared(declared, relative, _third_party_names(modules))
    by_command: dict[str, list[str]] = {}
    for command, module in _cli_command_imports(root):
        by_command.setdefault(command, []).append(module)
    for command, modules in by_command.items():
        where = f"{CLI_RELATIVE}:{command}"
        missing += _undeclared(declared, where, _third_party_names(modules))
    return missing


def test_cli_reachable_modules_declare_every_import_as_a_core_dependency():
    missing = _undeclared_imports(REPO_ROOT)
    assert not missing, (
        "runtime import(s) reachable from `sestrav predict/validate/benchmark/info` "
        "are not declared in [project].dependencies:\n" + "\n".join(missing)
    )


def test_every_first_party_import_in_the_closure_resolves():
    # Replaces an assertion that could not fail. Every entry in REACHABLE_MODULES was
    # returned by _first_party_path only after its own .is_file() check succeeded, so
    # re-asserting .is_file() over that list was tautological; the rename case it claimed
    # to guard aborted collection instead. UNRESOLVED_FIRST_PARTY carries the real signal.
    assert not UNRESOLVED_FIRST_PARTY, (
        "first-party import(s) reachable from the CLI name no file on disk: "
        + ", ".join(UNRESOLVED_FIRST_PARTY)
    )


def test_closure_includes_the_cli_module_and_its_package_inits():
    assert CLI_RELATIVE in REACHABLE_MODULES
    for init in ("src/__init__.py", "functions/__init__.py"):
        assert init in REACHABLE_MODULES, init


# --- The walker's traversal policy, on an input where the two policies disagree ---

POLICY_SOURCE = """\
import top_plain
if TYPE_CHECKING:
    import pruned_type_checking
    if True:
        import pruned_under_type_checking
else:
    import walked_type_checking_else
try:
    import pruned_guarded
    if True:
        import pruned_under_guard
except (ValueError, ImportError):
    import pruned_fallback
else:
    import walked_try_else
finally:
    import walked_try_finally
try:
    import walked_broad_try
except Exception:
    pass
if __name__ == "__main__":
    import walked_main_guard
if flag:
    import walked_plain_if
    for item in items:
        with ctx():
            import walked_nested_block
class Holder:
    import walked_class_body
    def method(self):
        import pruned_method
def helper():
    import pruned_nested_def
"""


def _node_skipping_walk(tree: ast.Module) -> set[str]:
    """The policy this module does NOT use: `continue` on a node inside `ast.walk`.

    `ast.walk` queues a node's children before yielding it, so `continue` skips
    the node but still visits everything beneath it.
    """
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.Try, ast.FunctionDef)):
            continue
        names.update(_imported_modules(node))
    return names


def test_runtime_walker_prunes_subtrees_where_a_node_skipping_walk_does_not():
    tree = ast.parse(POLICY_SOURCE)
    walked = {m for node in _runtime_imports(tree.body) for m in _imported_modules(node)}
    assert walked == {
        "top_plain",
        "walked_type_checking_else",
        "walked_try_else",
        "walked_try_finally",
        "walked_broad_try",
        "walked_main_guard",
        "walked_plain_if",
        "walked_nested_block",
        "walked_class_body",
    }
    skipped_nodes_only = _node_skipping_walk(tree)
    assert walked <= skipped_nodes_only
    # Anti-vacuity: the input must actually separate the two policies.
    assert skipped_nodes_only - walked == {
        "pruned_type_checking",
        "pruned_under_type_checking",
        "pruned_guarded",
        "pruned_under_guard",
        "pruned_fallback",
        "pruned_method",
        "pruned_nested_def",
    }


# --- Mutation tests: each seeds an undeclared import into a COPY of the tree ---

# A real distribution that is declared only in the `demo` extra, not the core set.
PROBE_PACKAGE = "streamlit"


def _copy_cli_tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """Copy pyproject.toml and every .py under src/ and functions/.

    The copy set does not depend on the analysis under test, so a regression that
    shrinks the derived closure cannot also shrink the tree it is measured on.
    """
    root = tmp_path / "repo"
    sources = [REPO_ROOT / "pyproject.toml"]
    for package in ("src", "functions"):
        sources += sorted((REPO_ROOT / package).rglob("*.py"))
    for source in sources:
        target = root / source.relative_to(REPO_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return root


def _mutate(root: pathlib.Path, relative: str, old: str | None, new: str) -> None:
    """Replace the one occurrence of `old` (or prepend `new` when `old` is None)."""
    path = root / relative
    text = path.read_text(encoding="utf-8")
    if old is None:
        text = new + text
    else:
        assert text.count(old) == 1, f"mutation anchor not unique in {relative}: {old!r}"
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def _probe_findings(root: pathlib.Path) -> list[str]:
    return [line for line in _undeclared_imports(root) if f": {PROBE_PACKAGE} " in line]


TRAIN_IMPORT = "    from src.train_classifier import train_models\n"
VALIDATE_DOC = '    """Cross-validate a model configuration on a labeled dataset."""\n'

MUTATIONS = {
    # Control: a plain module-scope import in a module the CLI reaches directly.
    "M6_closure_module_scope": (
        [("src/train_classifier.py", "import os\n", f"import os\nimport {PROBE_PACKAGE}\n")],
        "src/train_classifier.py",
    ),
    # The only route to that module now runs inside an `if` in a command body.
    "M5_reached_through_runtime_if": (
        [
            ("src/train_classifier.py", "import os\n", f"import os\nimport {PROBE_PACKAGE}\n"),
            (CLI_RELATIVE, TRAIN_IMPORT, "    if True:\n    " + TRAIN_IMPORT),
        ],
        "src/train_classifier.py",
    ),
    "M4_third_party_in_command_body": (
        [(CLI_RELATIVE, VALIDATE_DOC, VALIDATE_DOC + f"    import {PROBE_PACKAGE}\n")],
        f"{CLI_RELATIVE}:cmd_validate",
    ),
    "M2_cli_module_scope": (
        [(CLI_RELATIVE, "import argparse\n", f"import argparse\nimport {PROBE_PACKAGE}\n")],
        CLI_RELATIVE,
    ),
    "M3_package_init": (
        [("src/__init__.py", None, f"import {PROBE_PACKAGE}\n")],
        "src/__init__.py",
    ),
}


def test_mutation_probe_package_is_not_a_core_dependency():
    assert _normalize(PROBE_PACKAGE) not in _core_dependencies(REPO_ROOT)


def test_unmutated_copy_reports_nothing(tmp_path):
    root = _copy_cli_tree(tmp_path)
    assert _undeclared_imports(root) == []


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_seeded_undeclared_import_is_reported(tmp_path, mutation):
    edits, expected_where = MUTATIONS[mutation]
    root = _copy_cli_tree(tmp_path)
    for relative, old, new in edits:
        _mutate(root, relative, old, new)
    assert _probe_findings(root) == [f"{expected_where}: {PROBE_PACKAGE} (-> {PROBE_PACKAGE})"]
