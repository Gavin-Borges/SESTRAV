"""ImmuneCODE MIRA release 002.1 is refused, with the reason, and no pair is invented.

The script pins release 002.1, and that release cannot supply what the script
writes. Read 2026-10-12 from the archive itself (sha256 matching the pin): it has
no peptide-detail-cov2.csv; its peptide-detail-ci.csv and peptide-detail-cii.csv
share the seven columns in DETAIL_COLUMNS below, none of them a peptide column
the script reads or any HLA column; and HLA appears only in subject-metadata.csv,
two columns per locus, one row per experiment's subject - a genotype. On main the
script raised a ValueError traceback from `_require_column` by both routes.

These tests pin:

* an archive shaped like 002.1, by --input, by the auto-download (through the real
  `_download_mira`, with the pin pointed at the synthetic archive), or its detail
  CSV by --input: main() returns 1, logs a refusal that names the missing columns
  and explains the genotype layout, writes nothing, and never reaches
  `_extract_negatives`;
* a table missing only its peptide column is refused without the 002.1 account,
  and one missing only its HLA column with it;
* the download-failure text points at the pinned URL and the listing page, says
  002.1 is refused by either route, and the URL the script held before appears
  nowhere in it;
* opt-in: with SESTRAV_IMMUNECODE_ARCHIVE naming a local copy of the real archive,
  that copy matches the pin and is refused the same way.

No test opens a socket.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import logging
import os
import sys
import urllib.request
import urllib.response
import zipfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import ingest_immunecode as ingest  # noqa: E402

# The URL the script held before; its host answered 404 for it on 2026-10-12.
DEAD_URL = "https://clients.adaptivebiotech.com/pub/covid-2020/immuneCODE-MIRA-release002.1.zip"

DETAIL_COLUMNS = [
    "TCR BioIdentity",
    "TCR Nucleotide Sequence",
    "Experiment",
    "ORF Coverage",
    "Amino Acids",
    "Start Index in Genome",
    "End Index in Genome",
]
# Synthetic rows in the release's shape; no value here is copied from it.
DETAIL_CSV = (
    ",".join(DETAIL_COLUMNS) + "\n"
    'CASSAAAAAAF+TCRBV01-01+TCRBJ01-01,NNNNNN,eX1,ORF1ab,"YLQPRTFLL,KIADYNYKL",1,2\n'
    "CASSCCCCCCF+TCRBV02-01+TCRBJ02-01,NNNNNN,eX2,ORF1ab,LTDEMIAQY,3,4\n"
)
SUBJECT_CSV = (
    "Experiment,Subject,HLA-A,HLA-A,HLA-B,HLA-B,HLA-C,HLA-C\n"
    "eX1,1,A*02:01:01,A*24:02:01,B*07:02:01,B*08:01:01,C*07:01:01,C*07:02:01\n"
    "eX2,2,A*01:01:01,A*02:01:01,B*15:01:01,B*44:02:01,C*03:04:01,C*05:01:01\n"
)
PREFIX = "ImmuneCODE-MIRA-Release002.1/"


def _archive_like_002_1() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(PREFIX + "peptide-detail-ci.csv", DETAIL_CSV)
        archive.writestr(PREFIX + "peptide-detail-cii.csv", DETAIL_CSV)
        archive.writestr(PREFIX + "subject-metadata.csv", SUBJECT_CSV)
        archive.writestr(PREFIX + "readme.txt", "* subject-metadata.csv\n")
    return buffer.getvalue()


LIKE_002_1 = _archive_like_002_1()

# Phrases the refusal must carry: what is missing, and why the genotypes cannot stand in.
MISSING_BOTH = "no peptide column (tried Peptide"
MISSING_HLA = "no HLA restriction column (tried HLA Restrictions"
GENOTYPE = "HLA appears only as each subject's genotype in subject-metadata.csv"
NOT_PRESENTED = "not which one presented a peptide"


@pytest.fixture
def never_extracts(monkeypatch):
    def _never(df):
        raise AssertionError("_extract_negatives was reached for a table that cannot supply pairs")

    monkeypatch.setattr(ingest, "_extract_negatives", _never)


def _assert_refused(caplog, out, *phrases) -> None:
    assert not out.exists()
    assert not out.with_name(out.stem + "_provenance.json").exists()
    for phrase in phrases:
        assert phrase in caplog.text, phrase


def test_an_archive_shaped_like_002_1_is_refused_by_input(tmp_path, caplog, never_extracts):
    archive = tmp_path / "ImmuneCODE-MIRA-Release002.1.zip"
    archive.write_bytes(LIKE_002_1)
    out = tmp_path / "out" / "negatives.csv"
    with caplog.at_level(logging.ERROR):
        assert ingest.main(["--input", str(archive), "--output", str(out)]) == 1
    _assert_refused(caplog, out, MISSING_BOTH, MISSING_HLA, GENOTYPE, NOT_PRESENTED)


def test_its_detail_csv_is_refused_by_input(tmp_path, caplog, never_extracts):
    csv_path = tmp_path / "peptide-detail-ci.csv"
    csv_path.write_text(DETAIL_CSV)
    out = tmp_path / "out" / "negatives.csv"
    with caplog.at_level(logging.ERROR):
        assert ingest.main(["--input", str(csv_path), "--output", str(out)]) == 1
    _assert_refused(caplog, out, "peptide-detail-ci.csv cannot supply", GENOTYPE)


_REAL_BUILD_OPENER = urllib.request.build_opener


def _serve(body: bytes):
    class _Handler(urllib.request.HTTPSHandler):
        def https_open(self, req):
            response = urllib.response.addinfourl(
                io.BytesIO(body), http.client.HTTPMessage(), req.full_url, 200
            )
            response.msg = "OK"  # type: ignore[attr-defined]
            return response

    return lambda *handlers: _REAL_BUILD_OPENER(_Handler())


def test_the_auto_download_is_verified_and_then_refused(
    tmp_path, monkeypatch, caplog, never_extracts
):
    """The real _download_mira runs: the archive passes every check, then is refused."""
    monkeypatch.setattr(ingest, "MIRA_PINNED_SHA256", hashlib.sha256(LIKE_002_1).hexdigest())
    monkeypatch.setattr(ingest, "MIRA_PINNED_SIZE_BYTES", len(LIKE_002_1))
    monkeypatch.setattr(urllib.request, "build_opener", _serve(LIKE_002_1))
    out = tmp_path / "out" / "negatives.csv"
    with caplog.at_level(logging.ERROR):
        assert ingest.main(["--output", str(out)]) == 1
    _assert_refused(caplog, out, "ImmuneCODE-MIRA-Release002.1.zip cannot supply", GENOTYPE)
    assert "Download failed" not in caplog.text


@pytest.mark.parametrize(
    ("header", "row", "expect", "absent"),
    [
        ("Peptide,Amino Acids", "YLQPRTFLL,", [MISSING_HLA, GENOTYPE], ["no peptide column"]),
        (
            "HLA Restrictions,Amino Acids",
            "HLA-A*02:01,",
            ["no peptide column (tried Peptide"],
            [GENOTYPE, "no HLA restriction column"],
        ),
    ],
    ids=["no-hla", "no-peptide"],
)
def test_only_a_missing_hla_column_carries_the_002_1_account(
    tmp_path, caplog, never_extracts, header, row, expect, absent
):
    csv_path = tmp_path / "table.csv"
    csv_path.write_text(header + "\n" + row + "\n")
    out = tmp_path / "out" / "negatives.csv"
    with caplog.at_level(logging.ERROR):
        assert ingest.main(["--input", str(csv_path), "--output", str(out)]) == 1
    _assert_refused(caplog, out, *expect)
    for phrase in absent:
        assert phrase not in caplog.text, phrase


def test_the_download_failure_text_points_at_the_live_url_and_says_002_1_is_refused(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(urllib.request, "build_opener", _serve(b"<html>not a zip</html>"))
    with pytest.raises(RuntimeError) as raised:
        ingest._download_mira(str(tmp_path))
    message = str(raised.value)
    assert ingest.MIRA_DOWNLOAD_URL in message
    assert ingest.MIRA_LISTING_URL in message
    assert "release\n002.1 is refused by either route" in message
    assert DEAD_URL not in message


def test_the_url_the_script_held_before_appears_nowhere_in_it():
    with open(ingest.__file__, encoding="utf-8") as handle:
        assert DEAD_URL not in handle.read()


# Opt-in, because the archive is not in the repository: point this at a local copy.
_REAL_ARCHIVE = os.environ.get("SESTRAV_IMMUNECODE_ARCHIVE")


@pytest.mark.skipif(not _REAL_ARCHIVE, reason="set SESTRAV_IMMUNECODE_ARCHIVE to a local copy")
def test_the_real_archive_matches_the_pin_and_is_refused(tmp_path, caplog, never_extracts):
    assert _REAL_ARCHIVE is not None
    assert ingest._read_archive(_REAL_ARCHIVE)[1]
    out = tmp_path / "out" / "negatives.csv"
    with caplog.at_level(logging.INFO):
        assert ingest.main(["--input", _REAL_ARCHIVE, "--output", str(out)]) == 1
    _assert_refused(caplog, out, MISSING_BOTH, MISSING_HLA, GENOTYPE)
    assert "--input matches the pinned release 002.1 archive" in caplog.text
