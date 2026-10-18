"""Three failure paths in scripts/ingest_immunecode.py that reported the wrong thing.

1. A temporary directory whose cleanup RAISES replaced the real failure. `_load_csv`
   downloads into a `tempfile.TemporaryDirectory()`, and `_download_mira` removes its
   `.part` file only best-effort: on Windows a second handle on that file makes the
   removal raise, the error is deliberately swallowed, and the file is left behind. The
   directory's own cleanup then raises `PermissionError` on the way out of the `with`
   block, and THAT is what escapes `main()` instead of the actionable
   "Download failed ... pass it via --input PATH" message. The existing fetch test says
   so in its own docstring: it makes only `os.remove` fail and records that a handle
   still held at cleanup time is not covered.

2. A damaged ZIP's error message depended on its FILENAME, not on any check. `_load_csv`
   dispatches on `zipfile.is_zipfile`, which reads the end-of-central-directory record, so
   a truncated archive answers False and falls through to `pandas.read_csv`. Measured: the
   same truncated bytes named `mira.zip` raise `BadZipFile("File is not a zip file")`,
   because pandas infers zip compression from the extension, while named `mira.dat` they
   raise `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x96`, which tells a reader
   nothing. The clarity was luck.

3. `main()` caught only `FileNotFoundError` and `RuntimeError`, so a `BadZipFile` - the
   damaged-archive refusal from 2, or the one zipfile raises while extracting a member
   that fails its CRC-32 - escaped as a traceback. Both now exit 1 with the reason
   logged, as the script's other refusals do.

No test opens a socket.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import zipfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import ingest_immunecode as ingest  # noqa: E402

MIRA_CSV = (
    "Peptide,HLA Restrictions,Amino Acids\n"
    "YLQPRTFLL,HLA-A*02:01,\n"
    "LTDEMIAQY,HLA-A*02:01,CASSIRSSYEQYF\n"
)


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("peptide-detail-cov2.csv", MIRA_CSV)
    return buffer.getvalue()


PAYLOAD = _zip_bytes()
TRUNCATED = PAYLOAD[: len(PAYLOAD) // 2]
assert TRUNCATED.startswith(b"PK\x03\x04")
assert not zipfile.is_zipfile(io.BytesIO(TRUNCATED))


# --------------------------------------------------------------------------
# 1. A cleanup failure must not replace the download failure
# --------------------------------------------------------------------------


def test_a_file_left_locked_in_the_temporary_directory_does_not_mask_the_failure(
    tmp_path, monkeypatch
):
    """The download fails and leaves an unremovable .part file, exactly as a Windows
    sharing violation does. What escapes main() must be the download's own message."""
    handles: list = []

    def _download_leaving_a_locked_part(dest_dir: str) -> str:
        os.makedirs(dest_dir, exist_ok=True)
        stuck = os.path.join(dest_dir, "ImmuneCODE-MIRA-Release002.1.zip.part")
        handle = open(stuck, "wb")
        handle.write(b"partial")
        handle.flush()
        handles.append(handle)  # still open when the directory is cleaned up
        raise RuntimeError(
            "Download failed: simulated\n"
            "Please download the ImmuneCODE MIRA archive manually from:\n"
            "  https://example.invalid/mira.zip\n"
            "Then pass it via --input PATH."
        )

    monkeypatch.setattr(ingest, "_download_mira", _download_leaving_a_locked_part)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            ingest._load_csv(None)
        assert "Download failed" in str(excinfo.value)
        assert "--input PATH" in str(excinfo.value)
        assert not isinstance(excinfo.value, PermissionError)
    finally:
        for handle in handles:
            handle.close()


def test_main_returns_1_rather_than_raising_when_cleanup_cannot_remove_a_file(
    tmp_path, monkeypatch
):
    handles: list = []

    def _download_leaving_a_locked_part(dest_dir: str) -> str:
        os.makedirs(dest_dir, exist_ok=True)
        stuck = os.path.join(dest_dir, "ImmuneCODE-MIRA-Release002.1.zip.part")
        handle = open(stuck, "wb")
        handle.write(b"partial")
        handle.flush()
        handles.append(handle)
        raise RuntimeError("Download failed: simulated")

    monkeypatch.setattr(ingest, "_download_mira", _download_leaving_a_locked_part)
    out = tmp_path / "out" / "negatives.csv"
    try:
        assert ingest.main(["--output", str(out)]) == 1
        assert not out.exists()
    finally:
        for handle in handles:
            handle.close()


def test_an_extraction_that_leaves_a_locked_file_does_not_mask_a_missing_csv(tmp_path, monkeypatch):
    """_load_from_zip extracts into a temporary directory too. A member left open there
    must not turn 'no peptide-detail CSV in this archive' into a PermissionError."""
    archive = tmp_path / "mira.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr("something-else.csv", MIRA_CSV)
    archive.write_bytes(buffer.getvalue())

    handles: list = []
    real_find = ingest._find_csv_in_dir

    def _find_and_lock(search_dir: str):
        for root, _dirs, files in os.walk(search_dir):
            for name in files:
                handles.append(open(os.path.join(root, name), "rb"))
        return real_find(search_dir)

    monkeypatch.setattr(ingest, "_find_csv_in_dir", _find_and_lock)
    try:
        with pytest.raises(FileNotFoundError, match="peptide-detail"):
            ingest._load_from_zip(str(archive))
    finally:
        for handle in handles:
            handle.close()


# --------------------------------------------------------------------------
# 2. A damaged archive is named as one, whatever the file is called
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["mira.zip", "mira.ZIP", "mira.dat", "archive"])
def test_a_damaged_archive_raises_a_badzipfile_whatever_its_name_is(tmp_path, name):
    damaged = tmp_path / name
    damaged.write_bytes(TRUNCATED)
    with pytest.raises(zipfile.BadZipFile) as excinfo:
        ingest._load_csv(str(damaged))
    assert name in str(excinfo.value)


def test_a_plain_csv_is_still_read_as_a_csv(tmp_path):
    """The magic-byte check must not capture a legitimate CSV input."""
    csv_path = tmp_path / "peptide-detail-cov2.csv"
    csv_path.write_text(MIRA_CSV)
    df, source, verified = ingest._load_csv(str(csv_path))
    assert len(df) == 2
    assert source == str(csv_path)
    assert verified is False


def test_a_csv_whose_first_bytes_merely_start_with_pk_is_not_mistaken_for_an_archive(tmp_path):
    """Only the four-byte local-file-header signature counts, not the letters PK."""
    csv_path = tmp_path / "peptides.csv"
    csv_path.write_text("PKPeptide,HLA Restrictions,Amino Acids\nYLQPRTFLL,HLA-A*02:01,\n")
    df, _source, _verified = ingest._load_csv(str(csv_path))
    assert len(df) == 1


def test_an_intact_archive_is_still_loaded(tmp_path):
    archive = tmp_path / "mira.zip"
    archive.write_bytes(PAYLOAD)
    df, source, verified = ingest._load_csv(str(archive))
    assert len(df) == 2
    assert source == str(archive)
    assert verified is False  # not the pinned release's bytes


# --------------------------------------------------------------------------
# 3. main() reports a damaged archive as a refusal, not a traceback
# --------------------------------------------------------------------------


def _bad_crc_archive() -> bytes:
    """A STORED archive whose one member fails its CRC-32: it opens, and extracting fails."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("peptide-detail-cov2.csv", MIRA_CSV)
    payload = buffer.getvalue()
    start = payload.index(MIRA_CSV.encode("ascii"))
    return payload[:start] + bytes([payload[start] ^ 0x01]) + payload[start + 1 :]


@pytest.mark.parametrize(
    ("name", "body", "reason"),
    [
        ("mira.dat", TRUNCATED, "starts as a ZIP archive but cannot be opened as one"),
        ("mira.zip", TRUNCATED, "starts as a ZIP archive but cannot be opened as one"),
        ("mira.zip", _bad_crc_archive(), "Bad CRC-32"),
    ],
    ids=["truncated-dat", "truncated-zip", "bad-crc-member"],
)
def test_main_returns_1_and_logs_the_reason_for_a_damaged_archive(
    tmp_path, caplog, name, body, reason
):
    damaged = tmp_path / name
    damaged.write_bytes(body)
    assert zipfile.is_zipfile(str(damaged)) == (body is not TRUNCATED)
    out = tmp_path / "out" / "negatives.csv"
    with caplog.at_level(logging.ERROR):
        assert ingest.main(["--input", str(damaged), "--output", str(out)]) == 1
    assert reason in caplog.text
    assert not out.exists()
    assert not (tmp_path / "out" / "negatives_provenance.json").exists()
