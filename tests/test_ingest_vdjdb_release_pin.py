"""scripts/ingest_vdjdb.py ingests ONE pinned VDJdb release, and only bytes it verified.

The fetcher used to resolve `/releases/latest` and then verify the downloaded
bytes against the sha256 digest THAT SAME API RESPONSE reported for the asset.
That is a transport-integrity check and not a pin: when upstream publishes a new
release, `latest` moves, and the URL, the digest and the size all move with it,
so the pipeline silently ingests a different corpus and the sidecar records only
that a fetch happened.

These tests pin what makes it a pin instead:

* the release endpoint names a fixed tag, so `latest` is never consulted;
* the API's own `tag_name` (absent counts as a mismatch), asset name, digest and
  size must each equal the pinned value, and a mismatch is refused BEFORE any
  byte is downloaded;
* the digest and size handed to the downloader are the PINNED values, not the
  API's, so the pin governs the bytes even if the API misreports them;
* a refusal is FATAL: in each of the four cases the O6 review measured failing
  open - an intact archive of another release already on disk, a release
  republished under another tag, one republished with another digest, and no
  network - `ingest_vdjdb` raises and writes neither the CSV nor its sidecar,
  where it used to ingest a stale table and record the pinned release.

No test opens a socket: the API call is routed through an in-memory
urllib.request.urlopen, socket connects are refused, and `_download_verified` is
replaced by a fake that fails the test if a pin mismatch ever reaches it.
"""

from __future__ import annotations

import http.client
import io
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.request
import urllib.response
import zipfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import ingest_vdjdb  # noqa: E402

SCHEMA_PATH = os.path.join(_REPO_ROOT, "data", "immunogenicity_dataset_v5_schema.json")

_HEADER = "Epitope\tMHC A\tEpitope species\tEpitope gene\n"
TSV_TEXT = (
    _HEADER + "GILGFVFTL\tHLA-A*02:01\tInfluenzaA\tM\n" + "NLVPMVATV\tHLA-A*02:01\tCMV\tpp65\n"
)
STALE_TSV = _HEADER + "YLQPRTFLL\tHLA-A*02:01\tSARSCoV2\tSpike\n"

_ABSENT = object()


def _asset(
    name: str | None = None,
    digest: str | None = None,
    size: object | None = None,
    url: str | None = None,
) -> dict:
    """One release asset, defaulting to exactly what the pin expects."""
    return {
        "name": ingest_vdjdb.VDJDB_PINNED_ASSET_NAME if name is None else name,
        "digest": ("sha256:" + ingest_vdjdb.VDJDB_PINNED_SHA256) if digest is None else digest,
        "size": ingest_vdjdb.VDJDB_PINNED_SIZE_BYTES if size is None else size,
        "browser_download_url": (
            ingest_vdjdb.VDJDB_DOWNLOAD_PATH_PREFIX.join(
                ["https://github.com", ingest_vdjdb.VDJDB_PINNED_RELEASE_TAG + "/x.zip"]
            )
            if url is None
            else url
        ),
    }


def _release(tag: object = None, assets: list[dict] | None = None) -> dict:
    """A release payload; tag=_ABSENT leaves `tag_name` out of it altogether."""
    release: dict = {"assets": [_asset()] if assets is None else assets}
    if tag is not _ABSENT:
        release["tag_name"] = ingest_vdjdb.VDJDB_PINNED_RELEASE_TAG if tag is None else tag
    return release


@pytest.fixture
def api(monkeypatch):
    """Serve one in-memory release payload; record every URL opened; refuse sockets."""
    state: dict = {"release": _release(), "calls": [], "offline": False}

    def _urlopen(url, *args, **kwargs):
        target = getattr(url, "full_url", url)
        state["calls"].append(target)
        if state["offline"]:
            raise urllib.error.URLError("simulated: no network")
        body = io.BytesIO(json.dumps(state["release"]).encode("utf-8"))
        response = urllib.response.addinfourl(body, http.client.HTTPMessage(), target, 200)
        response.msg = "OK"  # type: ignore[attr-defined]
        return response

    def _no_connect(self, *args, **kwargs):
        raise AssertionError("a socket connect was attempted during a no-network test")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(socket.socket, "connect", _no_connect)
    return state


def _never_download(*args, **kwargs):
    raise AssertionError("a download was started despite a pin mismatch")


# --------------------------------------------------------------------------
# The pin itself
# --------------------------------------------------------------------------


def test_the_release_endpoint_names_a_fixed_tag_and_never_latest():
    assert ingest_vdjdb.VDJDB_RELEASES_API.endswith(
        "/releases/tags/" + ingest_vdjdb.VDJDB_PINNED_RELEASE_TAG
    )
    assert "/releases/latest" not in ingest_vdjdb.VDJDB_RELEASES_API


def test_the_pinned_values_are_well_formed():
    assert re.fullmatch(r"[0-9a-f]{64}", ingest_vdjdb.VDJDB_PINNED_SHA256)
    assert type(ingest_vdjdb.VDJDB_PINNED_SIZE_BYTES) is int
    assert ingest_vdjdb.VDJDB_PINNED_SIZE_BYTES > 0
    assert ingest_vdjdb.VDJDB_PINNED_ASSET_NAME.endswith(".zip")
    assert ingest_vdjdb.VDJDB_PINNED_RELEASE_TAG


def test_a_matching_release_resolves_to_the_pinned_digest_and_size(api):
    _url, digest, size = ingest_vdjdb._resolve_vdjdb_asset()
    assert digest == ingest_vdjdb.VDJDB_PINNED_SHA256
    assert size == ingest_vdjdb.VDJDB_PINNED_SIZE_BYTES
    assert api["calls"] == [ingest_vdjdb.VDJDB_RELEASES_API]


# --------------------------------------------------------------------------
# A pin mismatch is refused, and refused before any byte is downloaded
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag", ["2027-01-01-ZENODO", _ABSENT, "", 20260603], ids=["other", "absent", "empty", "int"]
)
def test_a_release_not_reported_under_the_pinned_tag_is_refused(tag, api):
    """An endpoint that does not say which release it is describing is not the pin."""
    api["release"] = _release(tag=tag)
    with pytest.raises(RuntimeError, match="not the pinned"):
        ingest_vdjdb._resolve_vdjdb_asset()


def test_an_asset_whose_api_digest_is_not_the_pin_is_refused(api):
    other = "f" * 64
    api["release"] = _release(assets=[_asset(digest="sha256:" + other)])
    with pytest.raises(RuntimeError, match="does not match the pinned"):
        ingest_vdjdb._resolve_vdjdb_asset()


def test_an_asset_whose_api_size_is_not_the_pin_is_refused(api):
    api["release"] = _release(assets=[_asset(size=ingest_vdjdb.VDJDB_PINNED_SIZE_BYTES + 1)])
    with pytest.raises(RuntimeError, match="does not match the pinned"):
        ingest_vdjdb._resolve_vdjdb_asset()


def test_a_zip_asset_under_another_name_is_not_accepted(api):
    """The old code took the first .zip asset, whatever it was called."""
    api["release"] = _release(assets=[_asset(name="vdjdb-2027-01-01.zip")])
    with pytest.raises(RuntimeError, match=ingest_vdjdb.VDJDB_PINNED_ASSET_NAME):
        ingest_vdjdb._resolve_vdjdb_asset()


@pytest.mark.parametrize(
    "release",
    [
        _release(assets=[_asset(digest="sha256:" + "f" * 64)]),
        _release(assets=[_asset(size=1)]),
        _release(assets=[_asset(name="vdjdb-2027-01-01.zip")]),
        _release(tag=_ABSENT),
    ],
    ids=["wrong-digest", "wrong-size", "wrong-asset-name", "no-tag"],
)
def test_no_byte_is_downloaded_when_the_pin_does_not_match(api, release, tmp_path, monkeypatch):
    api["release"] = release
    monkeypatch.setattr(ingest_vdjdb, "_download_verified", _never_download)
    out = str(tmp_path / "data")
    assert ingest_vdjdb.fetch_vdjdb(out) is None


def test_the_pinned_digest_and_size_are_what_reach_the_downloader(api, tmp_path, monkeypatch):
    """Returning the API's values would let a misreporting API choose the bytes."""
    seen: dict = {}

    def _record(url, zip_path, expected_sha256, max_bytes, extract_to):
        seen["sha256"] = expected_sha256
        seen["max_bytes"] = max_bytes
        raise ingest_vdjdb._ExtractionFailed("stop here; nothing is written")

    monkeypatch.setattr(ingest_vdjdb, "_download_verified", _record)
    ingest_vdjdb.fetch_vdjdb(str(tmp_path / "data"))
    assert seen["sha256"] == ingest_vdjdb.VDJDB_PINNED_SHA256
    assert seen["max_bytes"] == ingest_vdjdb.VDJDB_PINNED_SIZE_BYTES


# --------------------------------------------------------------------------
# A refusal is fatal: the four fail-open cases now write nothing
# --------------------------------------------------------------------------


def _archive(tsv: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("vdjdb.txt", tsv)
    return buffer.getvalue()


# An intact archive of another release, and the first 100 bytes of one, the
# shape an interrupted download used to leave behind.
OTHER_RELEASE = _archive(STALE_TSV)
TRUNCATED = _archive(TSV_TEXT)[:100]

SCENARIOS = {
    "other-release-cache": {"archive": OTHER_RELEASE, "offline": True},
    "tag-republished": {"archive": TRUNCATED, "release": _release(tag="2027-01-01-ZENODO")},
    "digest-republished": {
        "archive": TRUNCATED,
        "release": _release(assets=[_asset(digest="sha256:" + "f" * 64)]),
    },
    "offline": {"archive": TRUNCATED, "offline": True},
}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_a_refused_or_unverified_archive_is_fatal_and_nothing_is_written(
    scenario, api, tmp_path, monkeypatch
):
    """Each of these used to ingest the stale table at exit 0 and write a sidecar
    recording the pinned release."""
    case = SCENARIOS[scenario]
    data = tmp_path / "data"
    data.mkdir()
    (data / "vdjdb.zip").write_bytes(case["archive"])
    (data / "vdjdb.txt").write_text(STALE_TSV)
    api["offline"] = case.get("offline", False)
    if "release" in case:
        api["release"] = case["release"]
    monkeypatch.setattr(ingest_vdjdb, "_download_verified", _never_download)
    monkeypatch.chdir(tmp_path)  # fetch_vdjdb() defaults to ./data
    out = tmp_path / "out.csv"

    with pytest.raises(FileNotFoundError, match="Could not find or fetch VDJdb input"):
        ingest_vdjdb.ingest_vdjdb(None, str(out), SCHEMA_PATH)

    assert not out.exists()
    assert not (tmp_path / "out_provenance.json").exists()
    assert (data / "vdjdb.txt").read_text() == STALE_TSV  # untouched, and unused
    assert sorted(os.listdir(data)) == ["vdjdb.txt", "vdjdb.zip"]
