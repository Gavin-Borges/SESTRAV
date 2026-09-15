"""Tests for the library-coverage scope synchronization gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "check_library_coverage.py"


def _load_module():
    """Import the checker by path because ``tools/`` is not a package."""
    spec = importlib.util.spec_from_file_location("check_library_coverage", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


coverage_gate = _load_module()


def _write_source(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        'if __name__ == "__main__":\n    raise SystemExit(0)\n',
        encoding="utf-8",
    )


def _write_config(root: Path, *entries: str) -> None:
    omit_lines = "".join(f"    {entry}\n" for entry in entries)
    (root / ".coveragerc.library").write_text(
        f"[run]\nomit =\n{omit_lines}[report]\nshow_missing = true\n",
        encoding="utf-8",
    )


@pytest.fixture()
def isolated_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(coverage_gate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(coverage_gate, "CONFIG", tmp_path / ".coveragerc.library")
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "--check"])
    return coverage_gate


def test_in_sync_config_exits_zero(tmp_path, isolated_gate, capsys):
    _write_source(tmp_path, "src/cli.py")
    _write_config(tmp_path, "src/cli.py")

    assert isolated_gate.main() == 0
    assert "omit list is in sync" in capsys.readouterr().out


def test_stale_entry_exits_one_and_names_path(tmp_path, isolated_gate, capsys):
    _write_config(tmp_path, "src/removed.py")

    assert isolated_gate.main() == 1
    output = capsys.readouterr().out
    assert "Stale omit entries" in output
    assert "src/removed.py" in output


def test_missing_entry_exits_one_and_names_path(tmp_path, isolated_gate, capsys):
    _write_source(tmp_path, "functions/unlisted.py")
    _write_config(tmp_path)

    assert isolated_gate.main() == 1
    output = capsys.readouterr().out
    assert "Scripts missing" in output
    assert "functions/unlisted.py" in output


def test_scripts_directory_is_outside_source_roots(tmp_path, isolated_gate, capsys):
    _write_source(tmp_path, "scripts/standalone.py")
    _write_config(tmp_path)

    assert isolated_gate.main() == 0
    assert "scripts/standalone.py" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The PARSER, not just the comparison.
#
# The tests above only ever write concrete `src/...` paths, so the parts of the
# gate that decide WHICH config lines are entries at all went untested. The real
# .coveragerc.library carries glob boilerplate (`*/tests/*`, `*/__init__.py`) and
# lists modules in subdirectories, so a parser regression would flag the
# boilerplate as stale and miss every nested module - both silent, and both
# against the live config rather than a fixture.
# ---------------------------------------------------------------------------


def test_glob_boilerplate_is_not_treated_as_a_stale_entry(isolated_gate, tmp_path, capsys):
    """`*/tests/*` and `*/__init__.py` name no file and must not read as stale."""
    _write_source(tmp_path, "src/cli.py")
    _write_config(tmp_path, "*/tests/*", "*/__init__.py", "src/cli.py")

    assert isolated_gate.main() == 0, capsys.readouterr().out


def test_a_module_in_a_subdirectory_is_discovered(isolated_gate, tmp_path, capsys):
    """The gate walks recursively; a nested script must be found, not reported missing."""
    _write_source(tmp_path, "src/ci/validate_release.py")
    _write_config(tmp_path, "src/ci/validate_release.py")

    assert isolated_gate.main() == 0, capsys.readouterr().out


def test_a_missing_nested_module_is_still_caught(isolated_gate, tmp_path, capsys):
    """The recursive walk must not be so permissive that it stops failing."""
    _write_source(tmp_path, "src/ci/validate_release.py")
    _write_config(tmp_path)

    assert isolated_gate.main() == 1
    assert "src/ci/validate_release.py" in capsys.readouterr().out


def test_an_init_file_is_not_required_in_the_omit_list(isolated_gate, tmp_path, capsys):
    """`__init__.py` is skipped by the walk, so its absence is not a gap."""
    _write_source(tmp_path, "src/__init__.py")
    _write_source(tmp_path, "src/cli.py")
    _write_config(tmp_path, "src/cli.py")

    assert isolated_gate.main() == 0, capsys.readouterr().out


def test_the_live_repository_passes_its_own_gate():
    """Live coverage, not a fixture.

    Without this, a local pytest run still cannot see real .coveragerc.library
    drift - which is the exact gap .claude/rules/deletion-safety.md records, where
    a stale entry passed the whole suite while the gate itself exited 1.
    """
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/check_library_coverage.py", "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
