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
gitignored scratch (the `tests/wave_test_package/` entry in `.gitignore`) and is
absent from a fresh clone and from
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


def _importorskip_modules(tree: ast.Module) -> set[str]:
    """Top-level module names passed to `pytest.importorskip(...)`.

    Only module-scope calls count. An importorskip inside a function or a class
    body does not gate the module's own top-level imports, so counting it would
    wrongly excuse a genuinely unguarded import.
    """
    out: set[str] = set()
    for node in tree.body:
        call = None
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
        if call is None:
            continue
        func = call.func
        is_importorskip = (
            isinstance(func, ast.Attribute) and func.attr == "importorskip"
        ) or (isinstance(func, ast.Name) and func.id == "importorskip")
        if not is_importorskip or not call.args:
            continue
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            out.add(first.value.split(".")[0])
    return out


def _module_scope_imports(path: pathlib.Path) -> set[str]:
    """Top-level import names that would abort collection if unsatisfied.

    Three guard forms are excluded, because none of them aborts collection:

      - `try`/`except ImportError` degrades on failure rather than raising;
      - an `if` block (most commonly `if TYPE_CHECKING:`) is not evaluated when
        a plain `import module` runs;
      - `pytest.importorskip("mod")` at module scope SKIPS the module when `mod`
        is absent, so every later top-level import of `mod` is reached only when
        it is importable.

    The importorskip form is the established idiom in this repo for optional
    heavy dependencies. Measured at 877850d by an AST walk of module-scope
    statements, not by grep: 14 test modules carry a module-scope
    `pytest.importorskip` for `torch` or `torch_geometric`, and exactly one
    (`test_train_gnn_partial_batch.py`) relies on it to gate a LATER
    module-scope import, which is the mechanism this handling exists for.
    A plain grep reports 16, counting function-scope calls too, so the three
    figures are worth keeping distinct. Treating these as hard requirements
    would demand the `dev` extra pull the whole GNN stack, which is the
    opposite of what `dev` is for.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    stdlib = set(sys.stdlib_module_names)
    skipped = _importorskip_modules(tree)
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
            if name in stdlib or name in skipped:
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


def _first_party_import_targets(path: pathlib.Path) -> set[str]:
    """FULL dotted module names imported at module scope that are first-party.

    `_module_scope_imports` deliberately keeps only the TOP-level name, because
    that is what has to resolve to a distribution. Here the full path is what
    matters: `from src.optimizer import ...` has to lead to `src/optimizer.py`,
    and the top-level name alone (`src`) leads nowhere.

    The same three guard forms are excluded as in `_module_scope_imports`, for
    the same reason: none of them aborts collection.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    skipped = _importorskip_modules(tree)
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Try, ast.If)):
            continue
        if isinstance(node, ast.Import):
            candidates = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import - always intra-repo
                continue
            candidates = [node.module] if node.module else []
        else:
            continue
        for full in candidates:
            top = full.split(".")[0]
            if top in skipped:
                continue
            if _is_first_party(top, path):
                out.add(full)
    return out


def _resolve_repo_module(dotted: str) -> pathlib.Path | None:
    """A dotted first-party name -> the tracked file it imports, or None."""
    parts = dotted.split(".")
    for candidate in (
        REPO_ROOT.joinpath(*parts).with_suffix(".py"),
        REPO_ROOT.joinpath(*parts, "__init__.py"),
    ):
        if candidate.is_file():
            return candidate
    return None


def test_dev_extra_covers_imports_reached_through_a_first_party_module():
    """One level deeper than the scan above, which is where three slipped through.

    The companion test scans what `tests/` imports DIRECTLY. A test that imports
    a first-party module is importing everything that module imports at module
    scope, and `src`/`scripts` are first-party, so the direct scan classifies
    `from src.optimizer import ...` as intra-repo and stops. `src/optimizer.py`
    then does `import pulp` at module scope.

    Measured 2026-09-13, before the companion fix: pulp (declared in no extra at
    all), pyahocorasick and mhcgnomes (declared only in `scripts`, which a `dev`
    install does not pull) were all reachable this way, and the direct scan was
    green throughout. A collection error means ZERO tests run, so this is the
    same severity as the failure the direct scan was written for.

    Scope is deliberately ONE level. A full transitive closure would drag in the
    whole import graph and start reporting optional research dependencies that
    `dev` has never promised, which is the opposite of what this gate is for.
    """
    declared = _dev_environment_distributions()
    missing = []
    for path in _tracked_test_files():
        for dotted in sorted(_first_party_import_targets(path)):
            target = _resolve_repo_module(dotted)
            if target is None:
                continue
            for name in sorted(_module_scope_imports(target)):
                if _is_first_party(name, target):
                    continue
                dist = DIST_NAME_OVERRIDES.get(name.lower(), name)
                if _normalize(dist) not in declared:
                    relative = path.relative_to(REPO_ROOT).as_posix()
                    missing.append(f"{relative} -> {dotted}: {name} (-> {dist})")
    assert not missing, (
        "third-party import(s) reached at module scope through a first-party module "
        'that `pip install -e ".[dev]"` does not provide, so collection aborts '
        "before any test runs:\n" + "\n".join(missing)
    )


def test_the_deeper_scan_actually_resolves_first_party_targets():
    # Anti-vacuity guard for the test above, in the same spirit as the one below.
    # If `_resolve_repo_module` stopped resolving anything - a path-shape change,
    # a wrong cwd - the deeper scan would silently check NOTHING while staying
    # green, which is the failure mode that let these three through in the first
    # place.
    resolved = 0
    for path in _tracked_test_files():
        for dotted in _first_party_import_targets(path):
            if _resolve_repo_module(dotted) is not None:
                resolved += 1
    assert resolved >= MINIMUM_SCANNED_FILES, (
        f"expected the deeper scan to resolve at least {MINIMUM_SCANNED_FILES} "
        f"first-party import targets, resolved {resolved}; it is reaching nothing "
        "and the test above is therefore vacuous"
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
