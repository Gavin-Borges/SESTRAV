"""A script whose own text documents `python scripts/<name>.py` must support that usage.

Running a file by path puts scripts/ first on sys.path, not the repository root, so its
first-party imports resolve only if the script puts the root there itself, before those
imports run. Otherwise the documented command fails with ModuleNotFoundError wherever the
package is not installed, and imports whatever `src` the environment has installed wherever
one is.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIRST_PARTY = {"src", "functions", "sestrav"}
# The four whose documented usage failed before they put their root on sys.path.
REPAIRED = (
    "evaluate_per_virus.py",
    "fit_calibrator.py",
    "fit_conformal_calibrator.py",
    "run_pandora_structures.py",
)


def _is_first_party(node: ast.stmt) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] in FIRST_PARTY for alias in node.names)
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return node.module.split(".")[0] in FIRST_PARTY
    return False


def _inserts_into_sys_path(statement: ast.stmt) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"insert", "append"}
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "path"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "sys"
        for node in ast.walk(statement)
    )


def _documented_by_path_scripts() -> list[str]:
    selected = []
    for path in sorted((REPO_ROOT / "scripts").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if not re.search(rf"python3?\s+scripts/{re.escape(path.name)}\b", text):
            continue
        if any(_is_first_party(node) for node in ast.walk(ast.parse(text))):
            selected.append(path.name)
    return selected


def test_the_scan_finds_the_documented_scripts():
    found = _documented_by_path_scripts()
    # The four this guard was written for, plus a floor, so a pattern that stops
    # matching fails here instead of passing every case below vacuously.
    assert set(REPAIRED) <= set(found)
    assert len(found) >= 20, found


@pytest.mark.parametrize("name", _documented_by_path_scripts())
def test_the_root_goes_on_sys_path_before_first_party_imports(name):
    body = ast.parse((REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")).body
    top_level = [
        s for s in body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    inserts = [s.lineno for s in top_level if _inserts_into_sys_path(s)]
    assert inserts, f"scripts/{name} documents by-path usage but never puts a root on sys.path"
    first_party = [s.lineno for s in body if _is_first_party(s)]
    if first_party:
        assert min(inserts) < min(first_party), (
            f"scripts/{name} imports first-party code at line {min(first_party)}, "
            f"before it puts a root on sys.path at line {min(inserts)}"
        )


@pytest.mark.parametrize("name", REPAIRED)
def test_documented_help_imports_src_from_its_own_tree(tmp_path, name):
    """Run `--help` from a copy whose planted `src` package exits 97 when imported.

    Exit 97 proves `src` resolved to the copy's own tree. Without the root insert the
    import fails where nothing is installed and resolves to an installed `src` elsewhere,
    so the sentinel never fires in either case.
    """
    (tmp_path / "scripts").mkdir()
    shutil.copy2(REPO_ROOT / "scripts" / name, tmp_path / "scripts")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("raise SystemExit(97)\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, f"scripts/{name}", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 97, result.stdout + result.stderr
