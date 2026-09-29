"""Every module-scope third-party import reachable from the packaged `sestrav`
CLI's four subcommands must resolve to a declared core dependency.

Scope is deliberately the CLI's reachable set, not the whole repo: most of
`src/` is research/analysis tooling run from a source checkout (the `dev`
extra), which this project has never promised works from a bare
`pip install sestrav`. The reachable set below is derived from
`src/cli.py`'s four `cmd_*` functions and their unguarded module-scope
first-party imports. This guard was added after a
module-scope `import matplotlib` in `functions/stage4_immunogenicity_scoring.py`
(declared only in the `demo` extra) turned out to be the eighth instance of
the class the 2026-08-14 AST audit (see the dated comment above
`[project].dependencies` in pyproject.toml) was meant to close. That audit
checked for presence anywhere in `pyproject.toml`, not presence in
`[project].dependencies` specifically, so a package declared in the wrong
extra passed it silently. This test checks the narrower, correct condition.
"""

from __future__ import annotations

import ast
from collections import deque
import pathlib
import re
import sys
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

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


def _normalize(name: str) -> str:
    """PEP 503 distribution-name normalization, for comparing across separators/case."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _core_dependencies() -> set[str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
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


# First-party imports that named no file on disk, filled during closure derivation.
UNRESOLVED_FIRST_PARTY: list[str] = []


def _first_party_path(module: str) -> str | None:
    """Resolve an absolute first-party import to its repository-relative file."""
    if module.split(".", maxsplit=1)[0] not in INTRA_REPO:
        return None

    relative = module.replace(".", "/")
    candidates = (pathlib.Path(f"{relative}.py"), pathlib.Path(relative) / "__init__.py")
    for candidate in candidates:
        if (REPO_ROOT / candidate).is_file():
            return candidate.as_posix()
    # Record rather than raise. This runs during REACHABLE_MODULES derivation at module
    # scope, so raising here aborts COLLECTION, and a collection error means zero tests
    # ran rather than one test going red.
    UNRESOLVED_FIRST_PARTY.append(module)
    return None


def _unguarded_function_nodes(statements: list[ast.stmt]):
    """Yield command-body nodes recursively, pruning guarded import blocks."""
    pending: list[ast.AST] = list(reversed(statements))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Try, ast.If)):
            continue
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        pending.extend(reversed(list(ast.iter_child_nodes(node))))


def _cli_direct_first_party_imports() -> set[str]:
    """First-party modules imported by the CLI's command functions."""
    cli_path = REPO_ROOT / "src/cli.py"
    tree = ast.parse(cli_path.read_text(encoding="utf-8"), filename=str(cli_path))
    paths: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("cmd_"):
            continue
        for child in _unguarded_function_nodes(node.body):
            for module in _imported_modules(child):
                path = _first_party_path(module)
                if path:
                    paths.add(path)
    return paths


def _module_scope_first_party_imports(path: pathlib.Path) -> set[str]:
    """Unguarded module-scope first-party imports from one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    paths: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Try, ast.If)):
            continue
        for module in _imported_modules(node):
            imported_path = _first_party_path(module)
            if imported_path:
                paths.add(imported_path)
    return paths


def _cli_reachable_modules() -> tuple[str, ...]:
    """Derive the transitive unguarded first-party closure of CLI commands."""
    pending = deque(sorted(_cli_direct_first_party_imports()))
    reachable: set[str] = set()
    while pending:
        relative = pending.popleft()
        if relative in reachable:
            continue
        reachable.add(relative)
        pending.extend(
            sorted(_module_scope_first_party_imports(REPO_ROOT / relative) - reachable)
        )
    return tuple(sorted(reachable))


REACHABLE_MODULES = _cli_reachable_modules()


def _module_scope_imports(path: pathlib.Path) -> set[str]:
    """Top-level import names, skipping try/except-guarded and conditional (`if`) blocks.

    A guarded import degrades on failure rather than crashing, so it is not a hard
    requirement; an `if` block (most commonly `if TYPE_CHECKING:`) is not evaluated
    at the point a plain `import module` runs.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    stdlib = set(sys.stdlib_module_names)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Try, ast.If)):
            continue
        if isinstance(node, ast.Import):
            candidates = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import - always intra-repo
                continue
            candidates = [node.module.split(".")[0]] if node.module else []
        else:
            continue
        for name in candidates:
            if name in stdlib or name in INTRA_REPO:
                continue
            names.add(name)
    return names


def test_cli_reachable_modules_declare_every_import_as_a_core_dependency():
    declared = _core_dependencies()
    missing = []
    for relative in REACHABLE_MODULES:
        path = REPO_ROOT / relative
        for name in sorted(_module_scope_imports(path)):
            dist = DIST_NAME_OVERRIDES.get(name.lower(), name)
            if _normalize(dist) not in declared:
                missing.append(f"{relative}: {name} (-> {dist})")
    assert not missing, (
        "module-scope import(s) reachable from `sestrav predict/validate/benchmark/info` "
        "are not declared in [project].dependencies:\n" + "\n".join(missing)
    )


def test_every_first_party_import_in_the_closure_resolves():
    # Replaces an assertion that could not fail. Every entry in REACHABLE_MODULES was
    # returned by _first_party_path only after its own .is_file() check succeeded, so
    # re-asserting .is_file() over that list was tautological; the rename case it claimed
    # to guard aborted collection instead. UNRESOLVED_FIRST_PARTY carries the real signal.
    assert not UNRESOLVED_FIRST_PARTY, (
        "first-party import(s) reachable from the CLI name no file on disk: "
        + ", ".join(sorted(UNRESOLVED_FIRST_PARTY))
    )
