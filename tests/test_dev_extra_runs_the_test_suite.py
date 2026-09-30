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
import functools
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
    return _is_first_party_from(name, importer.parent)


@functools.lru_cache(maxsize=None)
def _is_first_party_from(name: str, directory: pathlib.Path) -> bool:
    # Keyed on the importing DIRECTORY, not the file: the closure scan asks
    # this for every module it reaches, and most of them share a directory.
    if name in INTRA_REPO:
        return True
    candidates = [REPO_ROOT / root for root in SOURCE_ROOTS]
    # Also every directory from the importing file up to the repo root, since a
    # test may put its own directory on sys.path and import a sibling by stem.
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


def _stubbed_modules(tree: ast.Module) -> set[str]:
    """Top-level names a module assigns into `sys.modules` itself, at module scope.

    A stub satisfies a later import without the real distribution:
    `tests/test_shap_analysis_results_guard.py` installs an empty `shap` module
    object before importing `src.shap_analysis`, behind an `if`, which is why
    module-scope `if`/`try` bodies are searched too. Function bodies are not:
    a stub installed inside a test does not run before collection.
    """
    out: set[str] = set()
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.If, ast.Try)):
            pending.extend([*node.body, *node.orelse, *getattr(node, "finalbody", [])])
            continue
        for target in node.targets if isinstance(node, ast.Assign) else []:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "modules"
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                out.add(target.slice.value.split(".")[0])
    return out


@functools.lru_cache(maxsize=None)
def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@functools.lru_cache(maxsize=None)
def _module_scope_imports(path: pathlib.Path) -> frozenset[str]:
    """Top-level import names that would abort collection if unsatisfied.

    Three guard forms are excluded, because none of them aborts collection:

      - `try`/`except ImportError` degrades on failure rather than raising;
      - an `if` block (most commonly `if TYPE_CHECKING:`) is not evaluated when
        a plain `import module` runs;
      - `pytest.importorskip("mod")` at module scope SKIPS the module when `mod`
        is absent, so every later top-level import of `mod` is reached only when
        it is importable.

    Scope caveat, measured 2026-09-21 against src/verify/structural_gnn.py: the
    "try/except ImportError degrades on failure rather than raising" claim above
    is true of the GUARDED IMPORT ITSELF, and only within that. It says nothing
    about a name the try block binds being referenced somewhere else that Python
    evaluates eagerly - a class-body function ANNOTATION is exactly that case,
    and without `from __future__ import annotations` such a reference raises
    NameError at import time, which `except ImportError` cannot catch. This
    walker could not have seen that defect even in principle: `tree.body` here
    is MODULE scope only, and a class body's own statements (including its
    methods' annotations) are never visited by this loop regardless of whether
    the surrounding import is Try-guarded. This function verifies that every
    module-scope third-party IMPORT STATEMENT resolves to a distribution the
    `dev` extra declares; it does not and cannot verify that everything a
    try-guarded name later touches is equally safe to import without it.

    The importorskip form is the established idiom in this repo for optional
    heavy dependencies. Measured at 877850d by an AST walk of module-scope
    statements, not by grep: 14 test modules carry a module-scope
    `pytest.importorskip` for `torch` or `torch_geometric`, and exactly one
    (`test_train_gnn_partial_batch.py`) relies on it to gate a LATER
    module-scope import, which is the mechanism this handling exists for.
    A plain grep reports 16, counting function-scope calls too, so the three
    figures are worth keeping distinct. Treating these as hard requirements
    would demand the `dev` extra pull the whole GNN stack, which is the
    opposite of what `dev` is for. A module-scope `sys.modules` stub is excluded
    for the same reason; see `_stubbed_modules`.
    """
    tree = _tree(path)
    stdlib = set(sys.stdlib_module_names)
    skipped = _importorskip_modules(tree) | _stubbed_modules(tree)
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
    return frozenset(names)


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


@functools.lru_cache(maxsize=None)
def _first_party_import_targets(path: pathlib.Path) -> frozenset[str]:
    """FULL dotted module names imported at module scope that are first-party.

    `_module_scope_imports` deliberately keeps only the TOP-level name, because
    that is what has to resolve to a distribution. Here the full path is what
    matters: `from src.optimizer import ...` has to lead to `src/optimizer.py`,
    and the top-level name alone (`src`) leads nowhere.

    The same guard forms are excluded as in `_module_scope_imports`, for the
    same reason: none of them aborts collection.

    `from pkg import name` yields `pkg.name` as well as `pkg`, because `name`
    may be a SUBMODULE, whose own module scope then runs: that is how
    `from src import shap_analysis` reaches `import shap`. A candidate that
    names a class or function simply resolves to no file. Relative imports are
    resolved against this file's own package rather than skipped, since
    `src/gnn/__init__.py` reaches `src/gnn/models.py` only that way.
    """
    tree = _tree(path)
    skipped = _importorskip_modules(tree) | _stubbed_modules(tree)
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Try, ast.If)):
            continue
        if isinstance(node, ast.Import):
            candidates = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                package = path.parents[node.level - 1].relative_to(REPO_ROOT).parts
                module = ".".join([*package, *([module] if module else [])])
            candidates = [module, *(f"{module}.{alias.name}" for alias in node.names)]
        else:
            continue
        for full in candidates:
            top = full.split(".")[0]
            if top in skipped:
                continue
            if _is_relative(node) or _is_first_party(top, path):
                out.add(full)
    return frozenset(out)


def _is_relative(node: ast.stmt) -> bool:
    return isinstance(node, ast.ImportFrom) and bool(node.level)


@functools.lru_cache(maxsize=None)
def _resolve_import_files(dotted: str, directory: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Every tracked file that importing `dotted` executes: parent packages first.

    Empty when `dotted` names no module here (for instance a class). Bare stems
    resolve against the same bases `_is_first_party` accepts, since a test may
    put `scripts/` on sys.path and import a script by its name.
    """
    parts = dotted.split(".")
    bases = [REPO_ROOT, *(REPO_ROOT / root for root in SOURCE_ROOTS)]
    while directory != REPO_ROOT and REPO_ROOT in directory.parents:
        bases.append(directory)
        directory = directory.parent
    for base in bases:
        files: list[pathlib.Path] = []
        for depth in range(1, len(parts) + 1):
            here = base.joinpath(*parts[:depth])
            if (here / "__init__.py").is_file():
                files.append(here / "__init__.py")
            elif depth == len(parts) and here.with_suffix(".py").is_file():
                files.append(here.with_suffix(".py"))
            elif not here.is_dir():
                files = []
                break
        if files:
            return tuple(files)
    return ()


@functools.lru_cache(maxsize=None)
def _module_scope_closure(test_path: pathlib.Path) -> dict[pathlib.Path, str]:
    """Every first-party file that importing `test_path` executes -> the import chain.

    Breadth-first over module-scope first-party imports, so each chain reported is
    a shortest one. Guarded imports are not followed, for the reason they are not
    counted: none of them aborts collection.
    """
    chains: dict[pathlib.Path, str] = {}
    frontier = [(test_path, "")]
    while frontier:
        path, chain = frontier.pop(0)
        for dotted in sorted(_first_party_import_targets(path)):
            for target in _resolve_import_files(dotted, path.parent):
                if target != test_path and target not in chains:
                    chains[target] = f"{chain} -> {dotted}" if chain else dotted
                    frontier.append((target, chains[target]))
    return chains


def test_dev_extra_covers_imports_reached_through_first_party_modules():
    """Every module-scope import a tracked test reaches through first-party code.

    The companion test scans what `tests/` imports DIRECTLY. A test that imports
    a first-party module is importing everything that module imports at module
    scope, and everything those modules import in turn, so the direct scan
    classifies `from src.optimizer import ...` as intra-repo and stops.
    `src/optimizer.py` then does `import pulp` at module scope.

    Measured 2026-09-13: pulp, pyahocorasick and mhcgnomes were reachable one
    first-party hop out while the direct scan was green. This test then
    followed exactly ONE hop, on the argument that a full closure would drag in
    optional research dependencies that `dev` has never promised.

    Measured 2026-09-26, that limit was the next blind spot. A clean
    `pip install -e ".[dev]"` aborted collecting
    `tests/test_run_analysis_results_guard.py`, which reaches `import shap`
    TWO hops out (`scripts.run_analysis` -> `src.shap_analysis`), while this
    test was green. The feared noise did not appear: on that tree the full
    closure reported two findings, that one and a `shap` import which the
    importing test satisfies with a `sys.modules` stub, now honoured by
    `_stubbed_modules`. A collection error means ZERO tests run in the file,
    so no depth is a safe place to stop.

    What this cannot see, by construction: imports inside FUNCTION bodies.
    `src.gnn.models` imports torch_geometric inside `GraphEncoderV2.__init__`,
    so a `dev` install collects those tests and they fail one by one at call
    time; each such test has to `pytest.importorskip` the package itself, and
    this test does not check that it does.
    """
    declared = _dev_environment_distributions()
    missing = []
    for path in _tracked_test_files():
        tree = _tree(path)
        excused = _importorskip_modules(tree) | _stubbed_modules(tree)
        for target, chain in sorted(_module_scope_closure(path).items()):
            for name in sorted(_module_scope_imports(target) - excused):
                if _is_first_party(name, target):
                    continue
                dist = DIST_NAME_OVERRIDES.get(name.lower(), name)
                if _normalize(dist) not in declared:
                    relative = path.relative_to(REPO_ROOT).as_posix()
                    missing.append(f"{relative} -> {chain}: {name} (-> {dist})")
    assert not missing, (
        "third-party import(s) reached at module scope through first-party modules "
        'that `pip install -e ".[dev]"` does not provide, so collection aborts '
        "before any test runs:\n" + "\n".join(sorted(set(missing)))
    )


def test_the_closure_follows_more_than_one_first_party_hop(tmp_path, monkeypatch):
    # The regression this closes, in miniature: test -> scripts module -> src
    # module -> an undeclared distribution, two first-party hops out. A module-
    # scope stub must excuse its own name and nothing else.
    for directory in ("scripts", "src", "tests"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "leaf.py").write_text("import undeclared_leaf\n", encoding="utf-8")
    (tmp_path / "scripts" / "runner.py").write_text("from src import leaf\n", encoding="utf-8")
    test = tmp_path / "tests" / "test_probe.py"
    test.write_text(
        "import sys\n"
        "if 'stubbed_dist' not in sys.modules:\n"
        "    sys.modules['stubbed_dist'] = object()\n"
        "from scripts.runner import go\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    chains = {
        path.relative_to(tmp_path).as_posix(): chain
        for path, chain in _module_scope_closure(test).items()
    }
    assert chains["src/leaf.py"] == "scripts.runner -> src.leaf"
    assert _module_scope_imports(tmp_path / "src" / "leaf.py") == {"undeclared_leaf"}
    assert _stubbed_modules(_tree(test)) == {"stubbed_dist"}


def test_the_deeper_scan_actually_resolves_first_party_targets():
    # Anti-vacuity guard for the closure scan, in the same spirit as the one
    # below. If `_resolve_import_files` stopped resolving anything - a path-shape
    # change, a wrong cwd - the closure would silently check NOTHING while
    # staying green, which is the failure mode that let these through before.
    reached = set().union(*(_module_scope_closure(path) for path in _tracked_test_files()))
    assert len(reached) >= MINIMUM_SCANNED_FILES, (
        f"expected the closure scan to reach at least {MINIMUM_SCANNED_FILES} "
        f"distinct first-party files, reached {len(reached)}; it is reaching nothing "
        "and the closure test above is therefore vacuous"
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
