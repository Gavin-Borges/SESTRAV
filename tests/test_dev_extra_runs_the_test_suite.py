"""The `dev` extra must satisfy every module-scope third-party import in `tests/`.

`README.md` documents `pip install -e ".[dev]"` as the way to install the lint and
test tooling, so `pip install -e ".[dev]" && pytest` is a documented path and has
to work. It did not. `pydantic` and `fastapi` were declared only in the `api` and
`demo` extras while three tracked test modules import them at module scope
(`tests/test_config_schema.py`, `tests/test_api_main.py`,
`tests/test_api_log_injection.py`), so a `dev`-only environment aborted during
COLLECTION.

That is why this is worth a gate rather than being left to whoever notices: a
collection error means ZERO tests run, so the failure mode is not "three tests
fail", it is "the suite does not start" - and a run that aborted at collection
cannot be read as evidence about anything else in the suite.

This is the same class as the 2026-08-16 matplotlib incident recorded in
`tests/test_predict_path_dependencies_declared.py`: a package declared in the
WRONG extra passes any check that only asks whether `pyproject.toml` mentions it
somewhere. That test guards the packaged CLI's contract with a consumer. This one
guards the contract `README.md` offers a developer, which is a different promise
over a different dependency set, so neither test subsumes the other.

Scope note: only TRACKED test files are scanned. `tests/wave_test_package/` is
gitignored scratch (`.gitignore:490`) and is absent from a fresh clone and from
CI, so including it would make this test's verdict depend on local scratch
content - the opposite of what a gate is for.
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

# Directories whose presence makes a bare-name import first-party. Tests in this
# repo routinely inject a source directory onto sys.path and then import a module
# by its bare stem (for example `import build_dataset_v4` for
# `scripts/build_dataset_v4.py`), which a naive scan would report as an
# undeclared third-party package.
SOURCE_ROOTS = ("scripts", "src", "functions", "tools", "app", "api", "sestrav")

# First-party top-level packages; imports of these are never external.
INTRA_REPO = {"src", "functions", "sestrav", "tests", "tools", "scripts", "app", "api", "conftest"}

# Import name -> PyPI distribution name, for the handful where they differ.
DIST_NAME_OVERRIDES = {
    "bio": "biopython",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "ahocorasick": "pyahocorasick",
}

# A floor for the anti-vacuity guard below. The tracked suite held 128 Python
# files when this test was written; the floor is set well under that so ordinary
# churn does not trip it, while a scan that collapses to nothing still does.
MINIMUM_SCANNED_FILES = 60


def _normalize(name: str) -> str:
    """PEP 503 distribution-name normalization, for comparing across separators/case."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_names(requirements: list[str]) -> set[str]:
    names = set()
    for requirement in requirements:
        name = re.split(r"[<>=!~\s;\[]", requirement, maxsplit=1)[0]
        if name:
            names.add(_normalize(name))
    return names


def _dev_environment_distributions() -> set[str]:
    """What `pip install -e ".[dev]"` actually provides: the base list plus `dev`.

    Extras are ADDITIVE to `[project].dependencies`, so the base list counts here.
    Reading the extra in isolation is the specific mistake that produced a false
    claim in this repository once already.
    """
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = _requirement_names(data["project"]["dependencies"])
    dev = _requirement_names(data["project"].get("optional-dependencies", {}).get("dev", []))
    return base | dev


def _tracked_test_files() -> list[pathlib.Path]:
    """Tracked `.py` files under `tests/`, which is what a clone and CI actually have."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "tests/*.py", "tests/**/*.py"],
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
    # Also every directory from the importing file up to the repo root, since a
    # test may put its own directory on sys.path and import a sibling by stem.
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
    """Top-level import names, skipping try/except-guarded and conditional (`if`) blocks.

    A guarded import degrades on failure rather than aborting collection, so it is
    not a hard requirement; an `if` block (most commonly `if TYPE_CHECKING:`) is
    not evaluated when a plain `import module` runs.
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
            if name in stdlib:
                continue
            names.add(name)
    return names


def test_dev_extra_declares_every_module_scope_import_in_the_test_suite():
    declared = _dev_environment_distributions()
    missing = []
    for path in _tracked_test_files():
        for name in sorted(_module_scope_imports(path)):
            if _is_first_party(name, path):
                continue
            dist = DIST_NAME_OVERRIDES.get(name.lower(), name)
            if _normalize(dist) not in declared:
                relative = path.relative_to(REPO_ROOT).as_posix()
                missing.append(f"{relative}: {name} (-> {dist})")
    assert not missing, (
        'module-scope import(s) in tests/ that `pip install -e ".[dev]"` does not '
        "provide, so collection aborts before any test runs:\n" + "\n".join(missing)
    )


def test_the_scan_actually_reaches_the_test_suite():
    # Anti-vacuity guard. Every mechanism above can fail OPEN: `git ls-files` could
    # return nothing from an unexpected cwd, and a pathspec typo would silently scan
    # zero files while the test above still passed. A gate that cannot fail is worse
    # than no gate, because it reports safety it never checked.
    scanned = _tracked_test_files()
    assert len(scanned) >= MINIMUM_SCANNED_FILES, (
        f"expected at least {MINIMUM_SCANNED_FILES} tracked test files, found "
        f"{len(scanned)}; the scan is not reaching the suite and the companion "
        "test above is therefore vacuous"
    )


def test_the_scan_would_notice_an_undeclared_import():
    # Proves the detector bites, without mutating pyproject.toml: a name that is
    # neither stdlib, first-party, nor declared must be reported as missing.
    declared = _dev_environment_distributions()
    assert _normalize("definitely-not-a-real-distribution") not in declared
    assert not _is_first_party("definitely_not_a_real_module", REPO_ROOT / "tests" / "x.py")
