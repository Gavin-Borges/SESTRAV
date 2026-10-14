"""Extraction is bounded, and an ingest that yields nothing is a failure, not an artifact.

Two measured gaps in `scripts/ingest_vdjdb.py`:

1. `_safe_extract_zip` guarded member PATHS and not member SIZES. Measured: an
   8,273-byte archive holding one deflated member extracted 8,388,608 bytes, with
   no bound of any kind. `fetch_vdjdb` now extracts only an archive whose bytes
   matched the pinned sha256, so the pin is what keeps an arbitrary archive out;
   this bound caps what extraction can write if the pin is ever moved to an
   archive nobody measured, and it must admit the archive pinned today.

2. A `vdjdb.txt` holding only its header was ACCEPTED. Measured: it produced
   "Extracted 0 unique positive epitopes", wrote an empty CSV, and wrote a
   provenance sidecar recording `row_count` 0, all at exit 0. That is exactly the
   shape a truncated extraction leaves, and a downstream build reading that sidecar
   cannot tell it from a VDJdb release that genuinely contributed nothing. An empty
   file and a header cut mid-line already failed loudly; only the header-only case
   slipped through.

A member cannot write more than the size its own header declares - `ZipExtFile`
stops there and then verifies the CRC - so summing the declared sizes before
extracting anything is a sound bound and writes nothing on refusal.
"""

from __future__ import annotations

import csv
import io
import os
import sys
import zipfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import ingest_vdjdb  # noqa: E402

SCHEMA_PATH = os.path.join(_REPO_ROOT, "data", "immunogenicity_dataset_v5_schema.json")
HEADER = ["Epitope", "MHC A", "Epitope species", "Epitope gene"]


def _tsv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


_GOOD_ROW = {
    "Epitope": "GILGFVFTL",
    "MHC A": "HLA-A*02:01",
    "Epitope species": "InfluenzaA",
    "Epitope gene": "M",
}


# --------------------------------------------------------------------------
# 1. Extraction is bounded by total uncompressed size
# --------------------------------------------------------------------------


def _archive_of(uncompressed: int, members: int = 1) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(members):
            archive.writestr(f"part{index}.txt", b"\0" * uncompressed)
    return buffer.getvalue()


# What the pinned archive declares it extracts to: the sum of the uncompressed
# sizes in its central directory, 11 members, read 2026-10-12 by a ranged request
# for the directory alone. Its COMPRESSED size, VDJDB_PINNED_SIZE_BYTES, is about a
# tenth of this, so a bound checked against that would pass while refusing the
# real archive.
PINNED_ARCHIVE_DECLARED_UNCOMPRESSED_BYTES = 425_297_801


def test_the_bound_is_a_positive_int_and_admits_the_pinned_archive():
    bound = ingest_vdjdb.VDJDB_MAX_EXTRACTED_BYTES
    assert type(bound) is int
    assert bound >= PINNED_ARCHIVE_DECLARED_UNCOMPRESSED_BYTES


def test_an_archive_over_the_bound_is_refused_and_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_vdjdb, "VDJDB_MAX_EXTRACTED_BYTES", 4096)
    archive = tmp_path / "bomb.zip"
    archive.write_bytes(_archive_of(1024 * 1024))
    dest = tmp_path / "out"
    dest.mkdir()

    with zipfile.ZipFile(archive) as zip_ref:
        with pytest.raises(RuntimeError, match="4096"):
            ingest_vdjdb._safe_extract_zip(zip_ref, str(dest))

    assert sorted(os.listdir(dest)) == []


def test_the_bound_is_on_the_TOTAL_not_on_one_member(tmp_path, monkeypatch):
    """Many small members can exceed a per-member bound between them."""
    monkeypatch.setattr(ingest_vdjdb, "VDJDB_MAX_EXTRACTED_BYTES", 4096)
    archive = tmp_path / "many.zip"
    archive.write_bytes(_archive_of(1024, members=8))
    dest = tmp_path / "out"
    dest.mkdir()

    with zipfile.ZipFile(archive) as zip_ref:
        with pytest.raises(RuntimeError, match="4096"):
            ingest_vdjdb._safe_extract_zip(zip_ref, str(dest))

    assert sorted(os.listdir(dest)) == []


def test_an_archive_within_the_bound_still_extracts(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_vdjdb, "VDJDB_MAX_EXTRACTED_BYTES", 1024 * 1024)
    archive = tmp_path / "ok.zip"
    archive.write_bytes(_archive_of(2048))
    dest = tmp_path / "out"
    dest.mkdir()

    with zipfile.ZipFile(archive) as zip_ref:
        ingest_vdjdb._safe_extract_zip(zip_ref, str(dest))

    assert sorted(os.listdir(dest)) == ["part0.txt"]
    assert os.path.getsize(dest / "part0.txt") == 2048


def test_the_path_traversal_guard_still_fires(tmp_path):
    """The size bound must not displace the guard that was already there."""
    archive = tmp_path / "evil.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr("../escaped.txt", "x")
    archive.write_bytes(buffer.getvalue())
    dest = tmp_path / "out"
    dest.mkdir()

    with zipfile.ZipFile(archive) as zip_ref:
        with pytest.raises(RuntimeError, match="path traversal"):
            ingest_vdjdb._safe_extract_zip(zip_ref, str(dest))

    assert sorted(os.listdir(dest)) == []


# --------------------------------------------------------------------------
# 1b. A member name zipfile would rewrite is refused
# --------------------------------------------------------------------------
#
# fetch_vdjdb returns os.path.join(output_dir, *name.split("/")) for the table
# member. ZipFile.extract drops "", "." and ".." segments on every platform, and on
# Windows replaces :<>|"?* with "_" and strips trailing dots and spaces from each
# segment. A name it rewrites is written somewhere other than that join, and the
# join can then name a stale file: sub/../vdjdb.txt is written to sub/vdjdb.txt
# while the join spells the top-level vdjdb.txt. Each name below is refused on
# every platform, so an archive is accepted or refused the same way on each.

REWRITTEN_NAMES = [
    "sub/../vdjdb.txt",  # ".." dropped; the join resolves elsewhere
    "./vdjdb.txt",  # "." dropped
    "sub/./vdjdb.txt",
    "sub//vdjdb.txt",  # empty segment dropped
    "sub/vdjdb.txt.",  # trailing dot stripped on Windows
    "sub /vdjdb.txt",  # trailing space stripped on Windows
    "sub/vdjdb?.txt",  # "?" replaced on Windows
    'sub/a"b/vdjdb.txt',
    "sub/a*b.txt",
    "sub/a<b>.txt",
    "sub/a|b.txt",
    "sub/a:b.txt",
    "sub/../",  # a directory entry is held to the same rule
]


@pytest.mark.parametrize("name", REWRITTEN_NAMES)
def test_a_member_name_zipfile_would_rewrite_is_refused_and_nothing_is_written(name, tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr("first.txt", "written first in the archive")
        handle.writestr(name, "" if name.endswith("/") else "x")
    dest = tmp_path / "out"
    dest.mkdir()
    # On Windows, os.path.realpath resolves a segment holding ":" outside dest
    # (measured with CPython 3.11: dest/sub/a:b.txt -> "a:b.txt"), so the
    # traversal guard refuses that name first. Either refusal writes nothing.
    message = "zipfile would rewrite|path traversal" if ":" in name else "zipfile would rewrite"

    with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as zip_ref:
        with pytest.raises(RuntimeError, match=message):
            ingest_vdjdb._safe_extract_zip(zip_ref, str(dest))

    assert sorted(os.listdir(dest)) == []


@pytest.mark.parametrize("name", REWRITTEN_NAMES)
def test_the_name_check_flags_every_rewritten_name_on_any_platform(name):
    assert ingest_vdjdb._zipfile_would_rewrite(name)


ACCEPTED_NAMES = [
    "vdjdb.txt",
    "vdjdb-2026-06-03/",
    "vdjdb-2026-06-03/vdjdb.txt",
    "vdjdb-2026-06-03/vdjdb.slim.meta.txt",
    "a/b/c/vdjdb_full.txt",
    ".hidden/vdjdb.txt",  # a leading dot is kept by zipfile
    "sub/x..y.txt",
]


@pytest.mark.parametrize("name", ACCEPTED_NAMES)
def test_the_name_check_passes_names_zipfile_writes_as_spelled(name):
    assert not ingest_vdjdb._zipfile_would_rewrite(name)


def test_every_accepted_member_lands_exactly_where_its_name_joins(tmp_path):
    """The property fetch_vdjdb relies on, checked against zipfile itself."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        for index, name in enumerate(ACCEPTED_NAMES):
            handle.writestr(name, "" if name.endswith("/") else f"member {index}")
    dest = tmp_path / "out"
    dest.mkdir()

    with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as zip_ref:
        ingest_vdjdb._safe_extract_zip(zip_ref, str(dest))

    for index, name in enumerate(ACCEPTED_NAMES):
        joined = os.path.join(str(dest), *name.rstrip("/").split("/"))
        if name.endswith("/"):
            assert os.path.isdir(joined), name
        else:
            with open(joined, encoding="ascii") as handle:
                assert handle.read() == f"member {index}", name


# --------------------------------------------------------------------------
# 2. An ingest that yields no rows is a failure
# --------------------------------------------------------------------------


def test_a_header_only_table_is_refused_and_writes_no_artifact(tmp_path):
    """The silent case: it used to exit 0 with an empty CSV and a sidecar."""
    tsv = tmp_path / "vdjdb.txt"
    _tsv(tsv, [])
    out = tmp_path / "out.csv"

    with pytest.raises(ValueError, match="no rows"):
        ingest_vdjdb.ingest_vdjdb(str(tsv), str(out), SCHEMA_PATH)

    assert not out.exists()
    assert not (tmp_path / "out_provenance.json").exists()


def test_a_table_whose_every_row_is_filtered_out_is_refused(tmp_path):
    """Nothing distinguishes this from a truncated table downstream, so it fails too."""
    tsv = tmp_path / "vdjdb.txt"
    _tsv(
        tsv,
        [
            {
                "Epitope": "NLVPMVATV",
                "MHC A": "H-2Kb",  # not HLA
                "Epitope species": "CMV",
                "Epitope gene": "pp65",
            }
        ],
    )
    out = tmp_path / "out.csv"

    with pytest.raises(ValueError, match="no rows"):
        ingest_vdjdb.ingest_vdjdb(str(tsv), str(out), SCHEMA_PATH)

    assert not out.exists()


def test_an_empty_file_still_fails_as_it_already_did(tmp_path):
    import pandas as pd

    tsv = tmp_path / "vdjdb.txt"
    tsv.write_text("")
    with pytest.raises(pd.errors.EmptyDataError):
        ingest_vdjdb.ingest_vdjdb(str(tsv), str(tmp_path / "out.csv"), SCHEMA_PATH)


def test_one_surviving_row_is_still_written(tmp_path):
    """The refusal must be on an EMPTY result, not on a small one."""
    tsv = tmp_path / "vdjdb.txt"
    _tsv(tsv, [_GOOD_ROW, {**_GOOD_ROW, "Epitope": "NLVPMVATV", "MHC A": "H-2Kb"}])
    out = tmp_path / "out.csv"

    ingest_vdjdb.ingest_vdjdb(str(tsv), str(out), SCHEMA_PATH)

    assert out.exists()
    assert (tmp_path / "out_provenance.json").exists()
