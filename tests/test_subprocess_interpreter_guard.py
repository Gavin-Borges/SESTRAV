"""Reject bare Python interpreter names in subprocess argv literals.

A test that spawns ``"python"`` does not reliably run the interpreter running
pytest: on POSIX it is whatever PATH resolves, and on Windows the parent
executable's directory is searched first. Use ``sys.executable``. Each site is
reported as the file:line of the argv[0] string, the line to edit.

Scope is deliberately narrow: ``subprocess.<func>(...)`` called through the
module name, with a list or tuple literal as its first argument or ``args=``.
It does NOT see a subprocess alias or from-import, an argv bound to a
variable first, a ``shell=True`` command string, or ``os.system``/``os.popen``.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BARE_PYTHON = re.compile(r"python(?:3(?:\.\d+)?)?\Z")


def _tracked_test_modules() -> list[Path]:
    """Return tracked Python modules below tests/ from Git's exact listing."""
    result = subprocess.run(
        ["git", "ls-files", "--", "tests"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"tracked-test census requires Git metadata: {result.stderr.strip()}")
    return [
        REPO_ROOT / relative for relative in result.stdout.splitlines() if relative.endswith(".py")
    ]


def _argv_literal(call: ast.Call) -> ast.List | ast.Tuple | None:
    """Return a subprocess call's literal argv, including ``args=`` form."""
    if call.args:
        candidate = call.args[0]
    else:
        candidate = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "args"),
            None,
        )
    return candidate if isinstance(candidate, (ast.List, ast.Tuple)) else None


def _bare_interpreter_sites(path: Path) -> list[str]:
    """Find literal subprocess argv whose first element is a bare Python name."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative = path.relative_to(REPO_ROOT).as_posix()
    sites: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        ):
            continue
        argv = _argv_literal(node)
        if argv is None or not argv.elts:
            continue
        executable = argv.elts[0]
        if not (
            isinstance(executable, ast.Constant)
            and isinstance(executable.value, str)
            and BARE_PYTHON.fullmatch(executable.value)
        ):
            continue
        sites.append(f"{relative}:{executable.lineno}: {executable.value}")
    return sites


def test_tracked_tests_do_not_spawn_bare_python_interpreters():
    sites = [site for path in _tracked_test_modules() for site in _bare_interpreter_sites(path)]
    assert not sites, (
        "subprocess argv literals must start with sys.executable, not a bare "
        "Python interpreter name:\n" + "\n".join(sites)
    )
