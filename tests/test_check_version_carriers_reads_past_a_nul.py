"""A stray NUL byte must not hide a stale version carrier from check_version_carriers.py.

``_read_text`` used to return None for any file holding a NUL byte, and the
scan skipped it, so a text carrier with a stale version and one stray NUL was
never compared: the gate passed. A NUL alone does not make a file binary; failing
to decode as UTF-8 does. The reader now drops NUL bytes and still requires UTF-8,
so the text carrier is read while genuine binaries stay skipped.

Measured 2026-10-06 on this repository: 15 tracked files hold a NUL, only one of
them (an SVG) decodes as UTF-8 once its NULs are removed, and the live carrier
set is the same 7 carriers either way. Failing closed on any NUL instead would
have turned the live tree red on that SVG.
"""

from __future__ import annotations

import io
from pathlib import Path

from tools import check_version_carriers as gate


def _badge(version: str) -> str:
    return "![v](https://img.shields.io/" + "badge/version-" + version + "-blue)\n"


def _tree(tmp_path: Path, stale: bytes) -> list[str]:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "3.4.5"\n', encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "agree.md").write_text(_badge("3.4.5"), encoding="utf-8")
    (tmp_path / "docs" / "stale.md").write_bytes(stale)
    return ["pyproject.toml", "docs/agree.md", "docs/stale.md"]


def _run(tmp_path: Path, tracked: list[str]) -> tuple[int, str]:
    output = io.StringIO()
    code = gate.run(tmp_path, tracked_files=tracked, stream=output, required_kinds=())
    return code, output.getvalue()


def test_a_stale_carrier_is_caught_without_a_nul(tmp_path: Path) -> None:
    """Anchor: the plain stale carrier fails the gate."""
    code, output = _run(tmp_path, _tree(tmp_path, _badge("9.9.9").encode("ascii")))
    assert code == 1
    assert "docs/stale.md" + ":1 carries 9.9.9" in output


def test_a_stray_nul_does_not_hide_a_stale_carrier(tmp_path: Path) -> None:
    stale = _badge("9.9.9").encode("ascii") + b"\x00\n"
    code, output = _run(tmp_path, _tree(tmp_path, stale))
    assert code == 1, output
    assert "docs/stale.md" + ":1 carries 9.9.9" in output
    assert "1 of 3 version carriers disagree" in output


def test_a_genuine_binary_is_still_skipped(tmp_path: Path) -> None:
    """Not UTF-8 once its NULs are gone, so it is never read, whatever text it happens to hold."""
    binary = (
        b"\x89PNG\r\n\x1a\n\x00\xff\xfe\xfd" + _badge("9.9.9").encode("ascii") + b"\x00\xc3\x28"
    )
    code, output = _run(tmp_path, _tree(tmp_path, binary))
    assert code == 0, output
    assert "OK: 2 version carriers agree on 3.4.5" in output
