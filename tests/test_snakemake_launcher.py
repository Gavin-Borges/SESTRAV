"""The snakemake launcher is found beside the running interpreter, not on PATH alone.

Why this test exists
--------------------
`tests/test_pipeline_smoke_executes.py` is the one test that EXECUTES the pipeline. It
resolved snakemake with `shutil.which` and skipped when that returned None. PATH is not
where the launcher has to be: when the interpreter running the suite is a conda env's
`python.exe` named directly, without activating the env, the env's `Scripts/` is not on
PATH, so `which` returns None while snakemake is installed and importable. The smoke
test then skipped, and a skip reads as green.

Measured 2026-10-19 by session `129fbcd1`, with PATH as the only variable (same
interpreter, same tree, same argv): without the env's `Scripts/` on PATH the file gave
25 passed and 1 skipped, and with it, 26 passed. At suite scale that is the difference
between 5550 passed / 24 skipped and 5551 / 23, with the collected total equal.

`tests/test_snakemake_shell_quoting.py` already resolved the launcher in two steps,
PATH first and then the launcher directory beside the running interpreter, and its
docstring names this exact hazard. Both files now use one implementation,
`tests/_snakemake_launcher.py`, so the two cannot drift again.

What is asserted
----------------
- with every directory that holds a snakemake launcher removed from PATH, the resolver
  still finds the launcher, beside the running interpreter;
- under that same PATH, the pipeline smoke test RUNS and passes, rather than skipping.

Both skip only when snakemake is not importable at all, because then there is no
launcher anywhere to find.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _path_without_snakemake() -> str:
    """The current PATH minus every directory that holds a snakemake launcher."""
    entries = os.environ.get("PATH", "").split(os.pathsep)
    return os.pathsep.join(e for e in entries if e and shutil.which("snakemake", path=e) is None)


def test_the_launcher_is_found_with_snakemake_off_path(monkeypatch):
    pytest.importorskip("snakemake", reason="no snakemake installed, so no launcher exists")
    monkeypatch.setenv("PATH", _path_without_snakemake())
    assert shutil.which("snakemake") is None, "premise: snakemake is still on PATH"

    from tests._snakemake_launcher import snakemake_launcher

    found = Path(snakemake_launcher())
    interpreter_dir = Path(sys.executable).parent
    assert found.is_file(), found
    assert found.parent in {
        interpreter_dir,
        interpreter_dir / "Scripts",
        interpreter_dir / "bin",
    }, f"found {found}, which is not beside the running interpreter {sys.executable}"


def test_the_pipeline_smoke_test_runs_rather_than_skips_with_snakemake_off_path():
    pytest.importorskip("snakemake", reason="no snakemake installed, so no launcher exists")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_pipeline_smoke_executes.py",
            "-o",
            "addopts=",
            "-q",
            "-rs",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        env=dict(os.environ, PATH=_path_without_snakemake()),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert re.search(r"\b1 passed\b", output), output
    assert not re.search(r"skipped", output, re.IGNORECASE), (
        f"the smoke test skipped although snakemake is installed:\n{output}"
    )


def _calls_which_snakemake(node: ast.AST) -> bool:
    """`shutil.which("snakemake")` consulting the PROCESS PATH. A call given `path=` looks in
    the directories it names, not in PATH, so it is not a launcher lookup at all;
    `_path_without_snakemake` above is one, and it was the first thing this check caught."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "which"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "shutil"
        and bool(node.args)
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "snakemake"
        and not any(keyword.arg == "path" for keyword in node.keywords)
        and len(node.args) < 3  # a third positional argument is `path`
    )


def _reads_sys_executable(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "executable"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _outside_functions(node: ast.AST):
    """Every node below `node` that is not inside a function. A function's decorators sit
    in its `decorator_list`, so they belong to that function's scope, not to this one."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _FUNCTIONS):
            continue
        yield child
        yield from _outside_functions(child)


def _path_only_lookups(source: str) -> list[str]:
    """The scopes in `source` that ask PATH for snakemake without also reading
    `sys.executable`: each function, and the module's own code outside every function
    (reported as `<module>`, and covering class bodies). The module scope matters because
    `NAME = shutil.which(...)` feeding a `skipif` is how this suite gates on a program;
    `tests/` held 15 such module-level calls on 2026-10-19, for bash, grep and git."""
    tree = ast.parse(source)
    offenders = []
    scopes = [
        (func.name, list(ast.walk(func))) for func in ast.walk(tree) if isinstance(func, _FUNCTIONS)
    ]
    scopes.append(("<module>", list(_outside_functions(tree))))
    for name, nodes in scopes:
        if any(_calls_which_snakemake(n) for n in nodes) and not any(
            _reads_sys_executable(n) for n in nodes
        ):
            offenders.append(name)
    return offenders


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            "def f():\n    return shutil.which('snakemake')\n", ["f"], id="function-path-only"
        ),
        pytest.param(
            "def f():\n    return shutil.which('snakemake') or str(Path(sys.executable).parent)\n",
            [],
            id="function-with-fallback",
        ),
        pytest.param(
            "SNAKEMAKE = shutil.which('snakemake')\n", ["<module>"], id="module-path-only"
        ),
        pytest.param(
            "SNAKEMAKE = shutil.which('snakemake') or str(Path(sys.executable).parent)\n",
            [],
            id="module-with-fallback",
        ),
        pytest.param(
            "class T:\n    S = shutil.which('snakemake')\n", ["<module>"], id="class-body"
        ),
        pytest.param(
            "@pytest.mark.skipif(shutil.which('snakemake') is None, reason='x')\ndef test_x():\n    pass\n",
            ["test_x"],
            id="decorator-belongs-to-its-function",
        ),
        pytest.param(
            "S = shutil.which('snakemake', path='x')\n", [], id="path-keyword-is-not-PATH"
        ),
        pytest.param(
            "S = shutil.which('snakemake', None, 'x')\n", [], id="path-positional-is-not-PATH"
        ),
        pytest.param("BASH = shutil.which('bash')\n", [], id="another-program-is-not-counted"),
    ],
)
def test_the_lookup_check_reads_both_scopes(source, expected):
    assert _path_only_lookups(source) == expected


def test_no_test_resolves_snakemake_by_path_alone():
    """The defect CLASS, not one instance: a scope that asks PATH for snakemake must also
    look beside `sys.executable`. Measured per function and for module scope, so a helper
    carrying both steps passes wherever it lives and a lookup with only the first fails,
    naming it. The shared helper is the intended home, but a correct private copy is not
    refused. This also pins that the check flags nothing that exists today: none of the
    module-level `shutil.which` calls already in `tests/` is for snakemake."""
    offenders = [
        f"{path.relative_to(REPO_ROOT).as_posix()}::{name}"
        for path in sorted((REPO_ROOT / "tests").rglob("*.py"))
        for name in _path_only_lookups(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "these resolve snakemake from PATH alone and so skip whenever the env's launcher "
        f"directory is off PATH; use tests._snakemake_launcher.snakemake_launcher: {offenders}"
    )
