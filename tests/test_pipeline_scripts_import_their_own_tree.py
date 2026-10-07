"""A script that a Snakemake rule runs by path must import first-party code from its own tree.

`python scripts/x.py` puts scripts/ first on sys.path, not the repository root, so a
first-party import there resolves only if the script puts the root on sys.path itself.
Otherwise the import fails wherever the package is not installed (CI's test job installs
none), and resolves to whatever `src` the environment has installed wherever one is.
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
SNAKEFILES = ("pipeline.smk", "standardize_outputs.smk", "Snakefile")
FIRST_PARTY = {"src", "functions", "sestrav"}
# A rule's shell runs a file by path as `python <path>`, `"{sys.executable}" <path>` or
# `{sys.executable:q} <path>`. The format-spec alternative is not cosmetic: when every
# interpolation gained Snakemake's `:q` quoting, a pattern without it matched NOTHING,
# `_scripts_run_by_path()` returned an empty list, and the parametrized test below
# silently collected zero cases. The floor assertion in the first test is what caught it.
RUN_BY_PATH = re.compile(
    r'(?:\bpython3?|\{sys\.executable(?::[^}]*)?\}\\?"?)\s+((?:scripts|src|functions)/[\w/]+\.py)'
)


def _scripts_run_by_path() -> list[str]:
    found: set[str] = set()
    for name in SNAKEFILES:
        found.update(RUN_BY_PATH.findall((REPO_ROOT / name).read_text(encoding="utf-8")))
    return sorted(found)


def _first_party_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names & FIRST_PARTY


def _puts_a_path_on_sys_path_at_module_level(tree: ast.Module) -> bool:
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"insert", "append"}
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "path"
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "sys"
            ):
                return True
    return False


def test_the_pattern_matches_every_spelling_of_the_interpreter():
    """Premise anchor for the pattern itself, not just for its yield.

    Each spelling has been in this repository's workflow files, so a pattern that
    drops one reports zero scripts and makes every assertion below vacuous.
    """
    for spelling in (
        '"{sys.executable}" scripts/x.py',
        "{sys.executable:q} scripts/x.py",
        "python scripts/x.py",
        "python3 scripts/x.py",
    ):
        assert RUN_BY_PATH.findall(spelling) == ["scripts/x.py"], spelling


def test_the_scan_finds_the_scripts_rules_run_by_path():
    scripts = _scripts_run_by_path()
    # The two that failed with ModuleNotFoundError in a real run, plus a floor, so a
    # pattern that stops matching fails here instead of passing every case below vacuously.
    assert {"scripts/run_predig_wrapper.py", "scripts/standardize_outputs.py"} <= set(scripts)
    assert len(scripts) >= 7, scripts


@pytest.mark.parametrize("script", _scripts_run_by_path())
def test_a_script_run_by_path_puts_its_root_on_sys_path(script):
    tree = ast.parse((REPO_ROOT / script).read_text(encoding="utf-8"))
    imported = _first_party_imports(tree)
    if not imported:
        return
    assert _puts_a_path_on_sys_path_at_module_level(tree), (
        f"{script} imports {sorted(imported)} but never puts the repository root on sys.path"
    )


def test_standardize_outputs_imports_src_from_its_own_tree(tmp_path):
    """Run the real script from a copy whose `src` exits with a sentinel code when imported.

    Exit 97 proves the import resolved to the copy's own tree. Without the root insert it
    fails with ModuleNotFoundError where nothing is installed, and resolves to an installed
    `src` elsewhere, so the sentinel never fires in either case.
    """
    (tmp_path / "scripts").mkdir()
    shutil.copy2(REPO_ROOT / "scripts" / "standardize_outputs.py", tmp_path / "scripts")
    planted = tmp_path / "src"
    planted.mkdir()
    (planted / "__init__.py").write_text("", encoding="utf-8")
    (planted / "iedb_data_loader.py").write_text("raise SystemExit(97)\n", encoding="utf-8")
    binding = tmp_path / "binding.csv"
    binding.write_text("peptide,HLA-A*02:01\nGILGFVFTL,0.9\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/standardize_outputs.py",
            "--binding",
            str(binding),
            "--prime",
            "unused",
            "--predig",
            "unused",
            "--sestrav",
            "unused",
            "--output",
            str(tmp_path / "out.csv"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 97, result.stdout + result.stderr
