"""scripts/check_secrets.py reads UTF-16 and UTF-32 files as text.

The scanner opened every file as UTF-8. A UTF-16 file read that way is each ASCII
character followed by a NUL byte, so no keyword pattern and no vendor-format
pattern can match it, and a credential in such a file passed every gate that
delegates here: the CI secret-pattern job, pre-push Checks 6a and 6b, and the tree
scan. Windows PowerShell 5.1's `>` redirection and `Out-File` write UTF-16LE with a
byte-order mark by default, which is how a file in this encoding reaches a commit.

The wide decodings are ADDITIVE: the UTF-8 pass is unchanged, so no line it flags
can stop being flagged.

Payloads are assembled at runtime so this module carries no credential-shaped line
of its own; the tree scan reads it.
"""

from __future__ import annotations

import codecs
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_secrets.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_secrets", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assignment() -> str:
    # Mixed alphabet, length > 8, entropy > 3.0. Not a real credential.
    return "api_" + "key" + ' = "' + "a8f3k9d2m1q7x4z0b5" + '"'


def _vendor_token() -> str:
    # A GitHub classic-token shape: the prefix and exactly 36 alphanumerics.
    # Not a real credential.
    body = "A1b2C3d4E5f6G7h8J9k0L1m2N3p4Q5r6S7t8"
    assert len(body) == 36
    return "gh" + "p_" + body


# The credential sits on line 3, so a reported line number is checked as well as
# the verdict. CRLF, because that is what PowerShell writes.
_LINES = ["# exported settings", "", _assignment(), "end"]


def _wide(lines: list[str], codec: str, bom: bytes) -> bytes:
    return bom + ("\r\n".join(lines) + "\r\n").encode(codec)


_WIDE_CASES = [
    pytest.param("utf-16-le", codecs.BOM_UTF16_LE, id="utf-16-le-bom"),
    pytest.param("utf-16-be", codecs.BOM_UTF16_BE, id="utf-16-be-bom"),
    pytest.param("utf-16-le", b"", id="utf-16-le-no-bom"),
    pytest.param("utf-16-be", b"", id="utf-16-be-no-bom"),
    pytest.param("utf-32-le", codecs.BOM_UTF32_LE, id="utf-32-le-bom"),
    pytest.param("utf-32-be", codecs.BOM_UTF32_BE, id="utf-32-be-bom"),
    pytest.param("utf-32-le", b"", id="utf-32-le-no-bom"),
    pytest.param("utf-32-be", b"", id="utf-32-be-no-bom"),
]


@pytest.mark.parametrize(("codec", "bom"), _WIDE_CASES)
def test_assignment_in_a_wide_encoding_is_flagged(tmp_path: Path, codec: str, bom: bytes) -> None:
    mod = _load()
    target = tmp_path / "settings.txt"
    target.write_bytes(_wide(_LINES, codec, bom))
    assert mod.scan_file(str(target)) == [3]


@pytest.mark.parametrize(("codec", "bom"), _WIDE_CASES)
def test_vendor_token_in_a_wide_encoding_is_flagged(tmp_path: Path, codec: str, bom: bytes) -> None:
    mod = _load()
    target = tmp_path / "notes.md"
    target.write_bytes(_wide(["see below", _vendor_token()], codec, bom))
    assert mod.scan_file(str(target)) == [2]


@pytest.mark.parametrize(("codec", "bom"), _WIDE_CASES)
def test_clean_wide_file_is_not_flagged(tmp_path: Path, codec: str, bom: bytes) -> None:
    # Decoding the same bytes four more ways must not manufacture a finding.
    mod = _load()
    target = tmp_path / "clean.txt"
    target.write_bytes(_wide(["# exported settings", "region = us-east-1", "end"], codec, bom))
    assert mod.scan_file(str(target)) == []
    assert str(target) not in mod.UNREADABLE_PATHS


def test_truncated_utf16_file_is_still_scanned(tmp_path: Path) -> None:
    # An odd byte count cannot be valid UTF-16. The decoder must replace the
    # dangling byte rather than raise, or the lines above it go unexamined.
    mod = _load()
    target = tmp_path / "cut.txt"
    target.write_bytes(_wide(_LINES, "utf-16-le", codecs.BOM_UTF16_LE) + b"\x41")
    assert mod.scan_file(str(target)) == [3]
    assert str(target) not in mod.UNREADABLE_PATHS


def test_utf8_file_with_a_nul_byte_keeps_its_utf8_finding(tmp_path: Path) -> None:
    # A NUL is what triggers the wide decodings. The UTF-8 finding must survive
    # them, reported once, at the same line.
    mod = _load()
    target = tmp_path / "mixed.py"
    target.write_bytes(b"x = 1\x00\n" + (_assignment() + "\n").encode("utf-8"))
    assert mod.scan_file(str(target)) == [2]


def _ascii_assignment() -> str:
    # A second credential line, of EVEN length so the UTF-16 reading stays aligned
    # on the bytes that follow it. Not a real credential.
    line = "tok" + "en" + ' = "' + "Q7x4z0b5a8f3k9d2m1" + '"'
    assert len(line) % 2 == 0
    return line


def test_line_flagged_by_two_readings_is_reported_once(tmp_path: Path) -> None:
    # Line 1 holds a UTF-16 credential AND an ASCII one, so the UTF-8 reading and
    # the UTF-16LE reading both flag line 1.
    mod = _load()
    target = tmp_path / "both.txt"
    target.write_bytes(
        _assignment().encode("utf-16-le")
        + _ascii_assignment().encode("ascii")
        + "\n".encode("utf-16-le")
    )
    assert mod.scan_file(str(target)) == [1]


def test_findings_from_two_readings_come_back_in_line_order(tmp_path: Path) -> None:
    # The UTF-16LE reading flags line 1; the UTF-8 reading, which splits at the
    # lone 0x0A byte, flags line 2 instead. Reported as one ordered list.
    mod = _load()
    target = tmp_path / "order.txt"
    target.write_bytes(
        (_assignment() + "\n").encode("utf-16-le") + _ascii_assignment().encode("ascii") + b"\n"
    )
    assert mod.scan_file(str(target)) == [1, 2]


def test_utf8_file_without_a_nul_is_unchanged(tmp_path: Path) -> None:
    mod = _load()
    target = tmp_path / "plain.py"
    target.write_text("x = 1\n" + _assignment() + "\n", encoding="utf-8")
    assert mod.scan_file(str(target)) == [2]


def test_tree_scan_fails_on_a_utf16_credential(tmp_path: Path, capsys) -> None:
    mod = _load()
    (tmp_path / "out.txt").write_bytes(_wide(_LINES, "utf-16-le", codecs.BOM_UTF16_LE))
    assert mod.scan_tree(str(tmp_path), min_files=1) == 1
    out = capsys.readouterr().out
    assert "[FLAGGED]" in out
    # Assembled from two pieces: written whole, a file name followed by a colon and a
    # line number reads to scripts/check_doc_line_citations.py as a citation.
    assert "out.txt" + ":3" in out
    # The value is never printed, in any encoding.
    assert "a8f3k9d2m1q7x4z0b5" not in out
