"""Every module-scope third-party import reachable from the packaged `sestrav`
CLI's four subcommands must resolve to a declared core dependency.

Scope is deliberately the CLI's reachable set, not the whole repo: most of
`src/` is research/analysis tooling run from a source checkout (the `dev`
extra), which this project has never promised works from a bare
`pip install sestrav`. The reachable set below was traced by hand from
`src/cli.py`'s four `cmd_*` functions on 2026-08-16, the same day a
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
import pathlib
import re
import sys
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Every module reachable at CLI-command scope from `sestrav predict/validate/benchmark/info`.
# Traced by hand from src/cli.py's four cmd_* functions on 2026-08-16; update this list if a
# cmd_* function starts importing a new module.
REACHABLE_MODULES = (
    "functions/stage1_peptide_generation.py",
    "functions/stage2_mhc_binding_prediction.py",
    "functions/stage3_tcr_feature_extraction.py",
    "functions/stage4_immunogenicity_scoring.py",
    "src/train_classifier.py",
    "src/artifact_guard.py",
    "src/ml_utils.py",
    "src/evaluate_metrics.py",
    "src/iedb_data_loader.py",
)

# Import name -> PyPI distribution name, for the handful where they differ.
DIST_NAME_OVERRIDES = {"bio": "biopython", "yaml": "pyyaml", "sklearn": "scikit-learn"}

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


def test_reachable_module_list_still_exists():
    # Guards the fixture list itself: a renamed/moved file would silently drop out of
    # REACHABLE_MODULES and this test's coverage would shrink without anyone noticing.
    missing = [relative for relative in REACHABLE_MODULES if not (REPO_ROOT / relative).is_file()]
    assert not missing, f"REACHABLE_MODULES names file(s) that no longer exist: {missing}"
