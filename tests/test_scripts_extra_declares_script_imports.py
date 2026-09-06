"""The `scripts` extra must satisfy unguarded module-scope imports under `scripts/`.

`scripts/` is not packaged (`pyproject.toml` `tool.setuptools.packages.find`
include is `sestrav*`, `src*`, `functions*`), so these imports are not a PyPI
consumer's problem. They are a source-checkout problem: `pip install -e .`
alone leaves `scripts/benchmark_runner.py`, `scripts/filter_validation_cohorts.py`
and `scripts/ingest_iedb_negatives.py` broken at import.

Two distributions are imported unguarded at module scope and were absent from
`pyproject.toml`:

- `pyahocorasick` (import name `ahocorasick`) in `benchmark_runner.py` and
  `filter_validation_cohorts.py`. Pinned in `requirements.in`; not declared.
- `mhcgnomes` in `ingest_iedb_negatives.py`. Satisfied only transitively via
  `mhcflurry`. Transitive is not a contract - the same argument
  `[project].dependencies` already makes for scipy.

They go in a `[scripts]` extra rather than the base list so a PyPI install is
not taxed with packages it cannot reach. Extras are additive, so the contract
this test guards is `pip install -e ".[scripts]"`.

Scope: only TRACKED files under `scripts/`. A pathspec typo or an unexpected
cwd that makes `git ls-files` return nothing must FAIL, not pass.
"""

from __future__ import annotations

import ast
import pathlib
import re
import subprocess
import sys
import tomllib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

SOURCE_ROOTS = ("scripts", "src", "functions", "tools", "app", "api", "sestrav")

INTRA_REPO = {
    "src",
    "functions",
    "sestrav",
    "tests",
    "tools",
    "scripts",
    "app",
    "api",
    "conftest",
}

DIST_NAME_OVERRIDES = {
    "bio": "biopython",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "ahocorasick": "pyahocorasick",
}

# 83 tracked scripts/*.py at the commit this test was written. Floor is well
# under that so ordinary churn does not trip it, while a scan that collapses
# to nothing still does.
MINIMUM_SCANNED_FILES = 40


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_names(requirements: list[str]) -> set[str]:
    names = set()
    for requirement in requirements:
        name = re.split(r"[<>=!~\s;\[]", requirement, maxsplit=1)[0]
        if name:
            names.add(_normalize(name))
    return names


def _scripts_environment_distributions() -> set[str]:
    """What `pip install -e ".[scripts]"` provides: the base list plus `scripts`.

    Extras are ADDITIVE to `[project].dependencies`. Reading the extra in
    isolation would miss pandas/numpy/etc. that the scripts also import.
    """
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = _requirement_names(data["project"]["dependencies"])
    extra = _requirement_names(
        data["project"].get("optional-dependencies", {}).get("scripts", [])
    )
    return base | extra


def _tracked_script_files() -> list[pathlib.Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "scripts/*.py", "scripts/**/*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"git ls-files unavailable: {result.stderr.strip()}")
    return [REPO_ROOT / entry for entry in result.stdout.split("\0") if entry]


def _is_first_party(name: str, importer: pathlib.Path) -> bool:
    if name in INTRA_REPO:
        return True
    candidates = [REPO_ROOT / root for root in SOURCE_ROOTS]
    directory = importer.parent
    while True:
        candidates.append(directory)
        if directory == REPO_ROOT or REPO_ROOT not in directory.parents:
            break
        directory = directory.parent
    for base in candidates:
        if (base / f"{name}.py").is_file() or (base / name / "__init__.py").is_file():
            return True
    return False


def _module_scope_imports(path: pathlib.Path) -> set[str]:
    """Top-level import names, skipping try/except-guarded and `if` blocks."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    stdlib = set(sys.stdlib_module_names)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Try, ast.If)):
            continue
        if isinstance(node, ast.Import):
            candidates = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            candidates = [node.module.split(".")[0]] if node.module else []
        else:
            continue
        for name in candidates:
            if name in stdlib:
                continue
            names.add(name)
    return names


def test_scripts_extra_declares_every_module_scope_import_in_scripts():
    declared = _scripts_environment_distributions()
    missing = []
    for path in _tracked_script_files():
        for name in sorted(_module_scope_imports(path)):
            if _is_first_party(name, path):
                continue
            dist = DIST_NAME_OVERRIDES.get(name.lower(), name)
            if _normalize(dist) not in declared:
                relative = path.relative_to(REPO_ROOT).as_posix()
                missing.append(f"{relative}: {name} (-> {dist})")
    assert not missing, (
        'module-scope import(s) in scripts/ that `pip install -e ".[scripts]"` '
        "does not provide:\n" + "\n".join(missing)
    )


def test_the_scan_actually_reaches_scripts():
    scanned = _tracked_script_files()
    assert len(scanned) >= MINIMUM_SCANNED_FILES, (
        f"expected at least {MINIMUM_SCANNED_FILES} tracked scripts, found "
        f"{len(scanned)}; the scan is not reaching scripts/ and the companion "
        "test above is therefore vacuous"
    )


def test_the_scan_would_notice_an_undeclared_import():
    declared = _scripts_environment_distributions()
    assert _normalize("definitely-not-a-real-distribution") not in declared
    assert not _is_first_party(
        "definitely_not_a_real_module", REPO_ROOT / "scripts" / "x.py"
    )
