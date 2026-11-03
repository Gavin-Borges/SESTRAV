"""scripts/ingest_immunecode.py verifies against the pin the same bytes it extracts.

main() used to load the archive by path and then hash it by path again, in two
separate reads, and the provenance sidecar recorded that second read as
"archive_verified_against_pin". A file replaced between the two reads was
extracted and ingested while the sidecar named the pinned release. The archive
is now read once, and both the verdict and the extraction come from those bytes.

Each test below replaces the file on disk at the moment zipfile opens it, which
is the window the old order left open.
"""

from __future__ import annotations

import hashlib
import io
import json
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
    "KIADYNYKL,HLA-B*07:02,\n"
)
_FIXED_TIME = (2020, 9, 21, 3, 1, 52)
_REAL_ZIPFILE = zipfile.ZipFile


def _zip_bytes(csv_text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            zipfile.ZipInfo("peptide-detail-cov2.csv", date_time=_FIXED_TIME),
            csv_text.encode("ascii"),
        )
    return buffer.getvalue()


PINNED = _zip_bytes(MIRA_CSV)
# Same size, one allele different, so only the digest tells them apart.
OTHER = _zip_bytes(MIRA_CSV.replace("HLA-B*07:02", "HLA-B*07:03"))
assert len(OTHER) == len(PINNED)
assert hashlib.sha256(OTHER).digest() != hashlib.sha256(PINNED).digest()


def _pin_to(monkeypatch, body: bytes) -> None:
    monkeypatch.setattr(ingest, "MIRA_PINNED_SHA256", hashlib.sha256(body).hexdigest())
    monkeypatch.setattr(ingest, "MIRA_PINNED_SIZE_BYTES", len(body))


def _swap_on_open(monkeypatch, path, replacement: bytes) -> list:
    """Overwrite path with replacement the first time zipfile opens anything."""
    opened = []

    def _zipfile(file, *args, **kwargs):
        if not opened:
            with open(path, "wb") as handle:
                handle.write(replacement)
        opened.append(file)
        return _REAL_ZIPFILE(file, *args, **kwargs)

    monkeypatch.setattr(ingest.zipfile, "ZipFile", _zipfile)
    return opened


def _alleles(df) -> set[str]:
    return set(df["HLA Restrictions"])


def test_a_pinned_archive_replaced_mid_load_is_ingested_from_the_bytes_that_were_verified(
    tmp_path, monkeypatch
):
    archive = tmp_path / "mira.zip"
    archive.write_bytes(PINNED)
    _pin_to(monkeypatch, PINNED)
    opened = _swap_on_open(monkeypatch, archive, OTHER)

    df, source, verified = ingest._load_csv(str(archive))

    assert opened, "zipfile was never opened"
    assert archive.read_bytes() == OTHER  # the swap really happened
    assert verified is True
    assert "HLA-B*07:02" in _alleles(df) and "HLA-B*07:03" not in _alleles(df)
    assert source == str(archive)


def test_an_archive_swapped_to_the_pinned_bytes_mid_load_is_not_reported_verified(
    tmp_path, monkeypatch
):
    archive = tmp_path / "mira.zip"
    archive.write_bytes(OTHER)
    _pin_to(monkeypatch, PINNED)
    _swap_on_open(monkeypatch, archive, PINNED)

    df, _source, verified = ingest._load_csv(str(archive))

    assert archive.read_bytes() == PINNED
    assert verified is False
    assert "HLA-B*07:03" in _alleles(df) and "HLA-B*07:02" not in _alleles(df)


def test_the_sidecar_verdict_describes_the_bytes_that_were_ingested(tmp_path, monkeypatch):
    archive = tmp_path / "mira.zip"
    archive.write_bytes(OTHER)
    _pin_to(monkeypatch, PINNED)
    _swap_on_open(monkeypatch, archive, PINNED)
    output = tmp_path / "out.csv"

    assert ingest.main(["--input", str(archive), "--output", str(output)]) == 0

    with open(tmp_path / "out_provenance.json", encoding="utf-8") as handle:
        prov = json.load(handle)
    assert prov["archive_verified_against_pin"] is False
    assert prov["release"] is None
    assert prov["pinned_archive_sha256"] is None


def test_an_archive_replaced_right_after_it_is_read_changes_nothing(tmp_path, monkeypatch):
    # The swap lands straight after the one read, so any second read of the path,
    # for the digest or the extraction, would see OTHER.
    archive = tmp_path / "mira.zip"
    archive.write_bytes(PINNED)
    _pin_to(monkeypatch, PINNED)
    real_open = open

    def _open_then_swap(file, mode="r", *args, **kwargs):
        if os.fspath(file) == str(archive) and "rb" in mode:
            with real_open(file, mode, *args, **kwargs) as handle:
                data = handle.read()
            with real_open(file, "wb") as handle:
                handle.write(OTHER)
            return io.BytesIO(data)
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(ingest, "open", _open_then_swap, raising=False)

    df, _source, verified = ingest._load_csv(str(archive))

    assert archive.read_bytes() == OTHER
    assert verified is True
    assert "HLA-B*07:02" in _alleles(df) and "HLA-B*07:03" not in _alleles(df)


def test_rows_come_from_the_verified_bytes_not_the_extracted_copy(tmp_path, monkeypatch):
    # The extracted CSV is rewritten after extraction; the rows must still be the
    # verified archive's.
    archive = tmp_path / "mira.zip"
    archive.write_bytes(PINNED)
    _pin_to(monkeypatch, PINNED)
    real_find = ingest._find_csv_in_dir

    def _find_and_rewrite(search_dir: str):
        path = real_find(search_dir)
        with open(path, encoding="ascii") as handle:
            text = handle.read()
        with open(path, "w", encoding="ascii") as handle:
            handle.write(text.replace("HLA-B*07:02", "HLA-B*07:03"))
        return path

    monkeypatch.setattr(ingest, "_find_csv_in_dir", _find_and_rewrite)

    df, _source, verified = ingest._load_csv(str(archive))

    assert verified is True
    assert "HLA-B*07:02" in _alleles(df) and "HLA-B*07:03" not in _alleles(df)


def test_the_verdict_covers_every_byte_not_a_prefix(tmp_path, monkeypatch):
    # Identical for their first 2 MiB; the CSV member that differs is stored after it.
    padding = bytes(range(256)) * (8 * 1024)

    def _large(csv_text: str) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(zipfile.ZipInfo("padding.bin", date_time=_FIXED_TIME), padding)
            archive.writestr(
                zipfile.ZipInfo("peptide-detail-cov2.csv", date_time=_FIXED_TIME),
                csv_text.encode("ascii"),
            )
        return buffer.getvalue()

    pinned, other = _large(MIRA_CSV), _large(MIRA_CSV.replace("HLA-B*07:02", "HLA-B*07:03"))
    assert pinned[: len(padding)] == other[: len(padding)] and len(pinned) == len(other)
    _pin_to(monkeypatch, pinned)
    verdicts = {}
    for name, body in (("pinned", pinned), ("other", other)):
        archive = tmp_path / f"{name}.zip"
        archive.write_bytes(body)
        _df, _source, verdicts[name] = ingest._load_csv(str(archive))

    assert verdicts == {"pinned": True, "other": False}


def test_an_auto_download_changed_after_its_check_is_refused(tmp_path, monkeypatch):
    def _download(dest_dir: str) -> str:
        path = os.path.join(dest_dir, "mira.zip")
        with open(path, "wb") as handle:
            handle.write(OTHER)  # what is on disk once extraction begins
        return path

    _pin_to(monkeypatch, PINNED)
    monkeypatch.setattr(ingest, "_download_mira", _download)

    with pytest.raises(RuntimeError, match="no longer matches the pinned sha256"):
        ingest._load_csv(None)


def test_an_intact_auto_download_is_verified(tmp_path, monkeypatch):
    def _download(dest_dir: str) -> str:
        path = os.path.join(dest_dir, "mira.zip")
        with open(path, "wb") as handle:
            handle.write(PINNED)
        return path

    _pin_to(monkeypatch, PINNED)
    monkeypatch.setattr(ingest, "_download_mira", _download)

    df, source, verified = ingest._load_csv(None)

    assert verified is True
    assert source == ingest.MIRA_DOWNLOAD_URL
    assert len(df) == 3
