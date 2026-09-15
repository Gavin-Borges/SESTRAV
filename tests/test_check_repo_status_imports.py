"""The import guard in scripts/check_repo_status.py must survive a blocked DLL.

A native extension whose shared library the OS refuses to load raises OSError,
not ImportError. The guard used to catch ImportError alone, so the exception
escaped and the health checker died before printing anything - in exactly the
degraded environment it exists to diagnose. Observed on a Windows host where an
Application Control policy blocked torch's c10.dll.
"""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_repo_status.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_crs_under_test", _PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("raised", [OSError("blocked by policy"), ImportError("absent")])
def test_check_imports_reports_rather_than_raising(monkeypatch, capsys, raised) -> None:
    """Both failure modes must be reported, and neither may escape."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise raised
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    module = _load_module()

    result = module.check_imports()

    assert result is False
    out = capsys.readouterr().out
    assert "torch is NOT available" in out
    # The exception type is surfaced so a blocked DLL is distinguishable from
    # a package that is simply not installed.
    assert type(raised).__name__ in out
