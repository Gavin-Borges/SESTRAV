"""The ImmuneCODE MIRA archive is pinned by sha256, and verified before it is used.

`scripts/ingest_immunecode.py` downloaded the MIRA archive and checked it only
STRUCTURALLY: that it opens as a ZIP and that every member passes its CRC-32. That
catches a transfer cut short or damaged in flight and says nothing about a
SUBSTITUTED archive. The transfer was also unbounded, so an endless body could
fill the disk.

These tests pin:

* a module-level sha256 and byte count for release 002.1, both well formed;
* verification of the downloaded bytes against both, after the structural check
  and BEFORE the archive takes its final name, with a refusal that names the
  expected and the actual digest;
* a digest over the WHOLE file: two archives identical in their first 1 MiB and
  different after it must not both match;
* a transfer bounded by the pinned size, so an oversized body is cut off;
* nothing left behind by a refusal: no archive under its final name, no `.part`.

No test opens a socket: `_download_mira` is driven with an in-memory opener.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import os
import re
import sys
import urllib.request
import urllib.response
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


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    """A STORED archive with fixed timestamps, so equal inputs give equal bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in members.items():
            archive.writestr(zipfile.ZipInfo(name, date_time=_FIXED_TIME), data)
    return buffer.getvalue()


PAYLOAD = _zip_bytes({"peptide-detail-cov2.csv": MIRA_CSV.encode("ascii")})
# A substituted archive of EXACTLY the same size, so the size check cannot be
# what rejects it and the digest check is what the test measures. One allele
# differs; the member name and every length are identical.
OTHER = _zip_bytes(
    {"peptide-detail-cov2.csv": MIRA_CSV.replace("HLA-B*07:02", "HLA-B*07:03").encode("ascii")}
)
assert len(OTHER) == len(PAYLOAD)
assert hashlib.sha256(OTHER).hexdigest() != hashlib.sha256(PAYLOAD).hexdigest()

# Two archives larger than the 1 MiB read size, identical in their first 2 MiB
# and different only in a small member stored after it, whose header and CRC sit
# past that point too. A digest of any prefix up to 2 MiB cannot tell them apart.
_PADDING = bytes(range(256)) * (8 * 1024)  # 2 MiB, not all one byte
assert len(_PADDING) == 2 * 1024 * 1024
LARGE = _zip_bytes(
    {"peptide-detail-cov2.csv": MIRA_CSV.encode("ascii"), "pad.bin": _PADDING, "z.txt": b"A"}
)
LARGE_OTHER = _zip_bytes(
    {"peptide-detail-cov2.csv": MIRA_CSV.encode("ascii"), "pad.bin": _PADDING, "z.txt": b"B"}
)
assert len(LARGE) == len(LARGE_OTHER) > 2 * 1024 * 1024
assert LARGE[: 2 * 1024 * 1024] == LARGE_OTHER[: 2 * 1024 * 1024]
assert LARGE != LARGE_OTHER


_REAL_BUILD_OPENER = urllib.request.build_opener


def _serve(body: bytes):
    """Replace build_opener with one whose https handler answers with `body`.

    The real build_opener is captured at import, so the replacement cannot
    recurse into itself.
    """

    class _Handler(urllib.request.HTTPSHandler):
        def https_open(self, req):
            response = urllib.response.addinfourl(
                io.BytesIO(body), http.client.HTTPMessage(), req.full_url, 200
            )
            response.msg = "OK"  # type: ignore[attr-defined]
            return response

    return lambda *handlers: _REAL_BUILD_OPENER(_Handler())


def _pin_to(monkeypatch, body: bytes) -> None:
    """Point the pin at `body`, so a test measures the check and not the literal."""
    monkeypatch.setattr(ingest, "MIRA_PINNED_SHA256", hashlib.sha256(body).hexdigest())
    monkeypatch.setattr(ingest, "MIRA_PINNED_SIZE_BYTES", len(body))


@pytest.fixture
def pinned_to_payload(monkeypatch):
    _pin_to(monkeypatch, PAYLOAD)


# --------------------------------------------------------------------------
# The pin itself
# --------------------------------------------------------------------------


def test_the_pinned_values_are_well_formed():
    assert re.fullmatch(r"[0-9a-f]{64}", ingest.MIRA_PINNED_SHA256)
    assert type(ingest.MIRA_PINNED_SIZE_BYTES) is int
    assert ingest.MIRA_PINNED_SIZE_BYTES > 0


def test_the_pin_is_not_a_placeholder():
    """An unresolved pin must never be spelled as a value that could match bytes."""
    assert set(ingest.MIRA_PINNED_SHA256) != {"0"}
    assert set(ingest.MIRA_PINNED_SHA256) != {"f"}


# --------------------------------------------------------------------------
# Verification of the download
# --------------------------------------------------------------------------


def test_an_archive_matching_the_pin_is_accepted(tmp_path, monkeypatch, pinned_to_payload):
    monkeypatch.setattr(urllib.request, "build_opener", _serve(PAYLOAD))
    path = ingest._download_mira(str(tmp_path))
    with open(path, "rb") as handle:
        assert handle.read() == PAYLOAD
    assert sorted(os.listdir(tmp_path)) == [os.path.basename(path)]


def test_a_substituted_archive_is_refused_even_though_it_is_a_valid_zip(
    tmp_path, monkeypatch, pinned_to_payload
):
    """The structural check passes on OTHER; only the digest separates them."""
    assert ingest._zip_is_intact.__doc__  # the structural check this one outlives
    monkeypatch.setattr(urllib.request, "build_opener", _serve(OTHER))
    with pytest.raises(RuntimeError) as excinfo:
        ingest._download_mira(str(tmp_path))
    message = str(excinfo.value)
    assert hashlib.sha256(PAYLOAD).hexdigest() in message
    assert hashlib.sha256(OTHER).hexdigest() in message
    assert sorted(os.listdir(tmp_path)) == []


def test_the_digest_covers_the_whole_archive_not_its_first_mebibyte(tmp_path, monkeypatch):
    """Both archives are over 2 MiB and share their first 2 MiB; only the end differs."""
    _pin_to(monkeypatch, LARGE)
    monkeypatch.setattr(urllib.request, "build_opener", _serve(LARGE))
    assert os.path.getsize(ingest._download_mira(str(tmp_path / "a"))) == len(LARGE)
    monkeypatch.setattr(urllib.request, "build_opener", _serve(LARGE_OTHER))
    with pytest.raises(RuntimeError, match="does not match the pinned release"):
        ingest._download_mira(str(tmp_path / "b"))
    assert sorted(os.listdir(tmp_path / "b")) == []


def test_an_archive_of_the_wrong_size_is_refused(tmp_path, monkeypatch, pinned_to_payload):
    monkeypatch.setattr(ingest, "MIRA_PINNED_SIZE_BYTES", len(PAYLOAD) + 1)
    monkeypatch.setattr(urllib.request, "build_opener", _serve(PAYLOAD))
    with pytest.raises(RuntimeError, match=str(len(PAYLOAD) + 1)):
        ingest._download_mira(str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == []


def test_a_transfer_longer_than_the_pinned_size_is_cut_off(
    tmp_path, monkeypatch, pinned_to_payload
):
    """Unbounded before: shutil.copyfileobj read whatever the server sent."""
    monkeypatch.setattr(urllib.request, "build_opener", _serve(PAYLOAD * 40))
    with pytest.raises(RuntimeError, match="exceeds"):
        ingest._download_mira(str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == []
