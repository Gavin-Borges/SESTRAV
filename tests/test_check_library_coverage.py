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
