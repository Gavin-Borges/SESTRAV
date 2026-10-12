"""Tests for the VDJdb archive download in scripts/ingest_vdjdb.py.

fetch_vdjdb() resolves the pinned release through the GitHub API and downloads
its .zip asset. These tests pin five properties of that download:

* the asset URL is refused before it is opened unless it is https on
  github.com under this repository's release downloads, and every redirect
  target (301, 302, 303, 307 and 308) is checked the same way BEFORE urllib
  follows it, with only GitHub's release-asset hosts added for redirects;
* the transfer has a bounded socket timeout and reads no more bytes than the
  pinned size;
* the bytes go to a ``.part`` file, are checked against the pinned sha256, are
  extracted, and only then take the name ``vdjdb.zip``, so no failed download
  leaves its archive under that name;
* a damaged ``vdjdb.zip`` already on disk is replaced by a verified download.
  If that download fails before extraction starts, the damaged archive is
  kept and any table already beside it is still used; once extraction starts,
  the damaged archive is removed before anything is written, so neither a
  later failure nor a process killed mid-extraction leaves it to fall back on;
* an intact ``vdjdb.zip`` is reused without network access, and extracted
  again when nothing extracted from it is left.

No test opens a socket. The GitHub API call goes through an in-memory
urllib.request.urlopen, and the asset transfer through in-memory https, http
and ftp handlers that serve real redirect responses, so urllib's own redirect
machinery runs. Socket connects are refused outright.
"""

from __future__ import annotations

import errno
import hashlib
import http.client
import io
import json
import os
import shutil
import socket
import sys
import urllib.error
import urllib.request
import urllib.response
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import ingest_vdjdb  # noqa: E402

ASSET_URL = (
    "https://github.com/antigenomics/vdjdb-db/releases/download/"
    "2026-06-03-ZENODO/vdjdb-2026-06-03.zip"
)
CDN_URL = "https://objects.githubusercontent.com/release-asset/vdjdb-2026-06-03.zip"
REPO_PREFIX = "https://github.com/antigenomics/vdjdb-db/releases/download/"
TSV_TEXT = "antigen.epitope\tmhc.a\nGILGFVFTL\tHLA-A*02:01\n"


def _zip_bytes(compression: int = zipfile.ZIP_DEFLATED, name: str = "vdjdb.txt") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        archive.writestr(name, TSV_TEXT)
    return buffer.getvalue()


PAYLOAD = _zip_bytes()
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


def _release(
    url: str = ASSET_URL,
    digest: str | None = "sha256:" + PAYLOAD_SHA256,
    size: object = len(PAYLOAD),
) -> dict:
    asset: dict = {"name": "vdjdb-2026-06-03.zip", "browser_download_url": url}
    if digest is not None:
        asset["digest"] = digest
    if size is not None:
        asset["size"] = size
    return {"tag_name": "2026-06-03-ZENODO", "assets": [asset]}


class _Body(io.BytesIO):
    """A response body with an optional per-read hook and a mid-transfer stall."""

    def __init__(
        self,
        payload: bytes,
        on_read=None,
        fail_after: int = -1,
        fail_with: BaseException | None = None,
        chunk: int = 0,
        delivered: list[int] | None = None,
    ):
        super().__init__(payload)
        self._on_read = on_read
        self._fail_after = fail_after
        self._fail_with = fail_with or TimeoutError("simulated stalled transfer")
        self._chunk = chunk
        self._delivered = delivered
        self._reads = 0

    def read(self, size: int | None = -1) -> bytes:
        if self._on_read is not None:
            self._on_read()
        self._reads += 1
        if self._fail_after >= 0 and self._reads > self._fail_after:
            raise self._fail_with
        if self._fail_after >= 0:
            # Hand back only half of what is left, so the transfer is truncated.
            remaining = len(self.getvalue()) - self.tell()
            return super().read(max(1, remaining // 2))
        if self._chunk:
            size = self._chunk if size is None or size < 0 else min(size, self._chunk)
        data = super().read(size)
        if self._delivered is not None:
            self._delivered.append(len(data))
        return data


def _response(url: str, code: int = 200, body: bytes = b"", location=None, **body_kw):
    headers = http.client.HTTPMessage()
    if location is not None:
        headers["Location"] = location
    response = urllib.response.addinfourl(_Body(body, **body_kw), headers, url, code)
    # urllib's HTTPErrorProcessor reads .msg, which addinfourl does not define.
    response.msg = "OK" if code == 200 else "Found"  # type: ignore[attr-defined]
    return response


def _serve(state: dict, req):
    state["calls"].append((req.full_url, req.timeout))
    factory = state["routes"].get(req.full_url)
    if factory is None:
        raise AssertionError(f"unexpected URL opened: {req.full_url}")
    return factory()


class _FakeHTTPS(urllib.request.HTTPSHandler):
    def __init__(self, state):
        super().__init__()
        self._state = state

    def https_open(self, req):
        return _serve(self._state, req)


class _FakeHTTP(urllib.request.HTTPHandler):
    def __init__(self, state):
        super().__init__()
        self._state = state

    def http_open(self, req):
        return _serve(self._state, req)


class _FakeFTP(urllib.request.FTPHandler):
    def __init__(self, state):
        super().__init__()
        self._state = state

    def ftp_open(self, req):
        return _serve(self._state, req)


@pytest.fixture
def network(monkeypatch):
    """Route the API call and every asset request to in-memory responses.

    ``state["routes"]`` maps a URL to a response factory; ``state["calls"]``
    records (url, timeout) for every request; ``state["offline"]`` makes the
    API call fail as it would with no network.
    """
    state: dict = {"release": _release(), "routes": {}, "calls": [], "offline": False}
    # The synthetic PAYLOAD stands in for the pinned archive, so the pin points at
    # it for the duration of each test. The pin's real literals, and the refusals
    # that compare a release against them, live in test_ingest_vdjdb_release_pin.py;
    # these tests are about the transfer, not about which release is pinned.
    monkeypatch.setattr(ingest_vdjdb, "VDJDB_PINNED_SHA256", PAYLOAD_SHA256)
    monkeypatch.setattr(ingest_vdjdb, "VDJDB_PINNED_SIZE_BYTES", len(PAYLOAD))
    real_build_opener = urllib.request.build_opener

    def _build_opener(*handlers):
        return real_build_opener(*handlers, _FakeHTTPS(state), _FakeHTTP(state), _FakeFTP(state))

    def _urlopen(url, *args, **kwargs):
        target = getattr(url, "full_url", url)
        if target == ingest_vdjdb.VDJDB_RELEASES_API:
            state["calls"].append((target, kwargs.get("timeout")))
            if state["offline"]:
                raise urllib.error.URLError("simulated: no network")
            return _response(target, body=json.dumps(state["release"]).encode("utf-8"))
        # Anything else is what plain urlopen does: urllib's default handlers,
        # which follow every redirect unchecked.
        return _build_opener().open(url, *args, **kwargs)

    def _no_connect(self, *args, **kwargs):
        raise AssertionError("a socket connect was attempted during a no-network test")

    monkeypatch.setattr(urllib.request, "build_opener", _build_opener)
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(socket.socket, "connect", _no_connect)
    return state


def _route(state: dict, url: str, code: int = 200, body: bytes = PAYLOAD, location=None, **kw):
    state["routes"][url] = lambda: _response(url, code, body, location, **kw)


def _redirect(state: dict, url: str, location: str, code: int = 302) -> None:
    _route(state, url, code=code, body=b"", location=location)


# Every status urllib's HTTPRedirectHandler follows for a GET.
REDIRECT_CODES = [301, 302, 303, 307, 308]


def _serve_via_cdn(state: dict, **kw) -> None:
    _redirect(state, ASSET_URL, CDN_URL)
    _route(state, CDN_URL, **kw)


def _asset_urls(state: dict) -> list[str]:
    return [url for url, _ in state["calls"] if url != ingest_vdjdb.VDJDB_RELEASES_API]


def _pin(sha256: str | None = None, size: int | None = None) -> None:
    """Move the pin a test is measured against.

    The ``network`` fixture monkeypatches both pin constants, so it has already
    recorded their real values and restores them at teardown; reassigning them
    here is undone with the rest. A test that serves its own archive, or that
    drives a code path reached only once the API agrees with the pin, points the
    pin at what it serves, so the resolve-time comparison is not what it measures.
    """
    if sha256 is not None:
        ingest_vdjdb.VDJDB_PINNED_SHA256 = sha256
    if size is not None:
        ingest_vdjdb.VDJDB_PINNED_SIZE_BYTES = size


def _repin_to(state: dict, body: bytes) -> None:
    """Point the release AND the pin at ``body``."""
    digest = hashlib.sha256(body).hexdigest()
    state["release"] = _release(digest="sha256:" + digest, size=len(body))
    _pin(sha256=digest, size=len(body))


def _out(tmp_path) -> str:
    return str(tmp_path / "data")


def _tsv(tmp_path) -> str:
    return os.path.join(_out(tmp_path), "vdjdb.txt")


def _zip(tmp_path) -> str:
    return os.path.join(_out(tmp_path), "vdjdb.zip")


def test_verified_download_lands_extracts_and_leaves_no_part_file(tmp_path, network):
    _serve_via_cdn(network)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) == _tsv(tmp_path)
    with open(_zip(tmp_path), "rb") as handle:
        assert handle.read() == PAYLOAD
    assert sorted(os.listdir(_out(tmp_path))) == ["vdjdb.txt", "vdjdb.zip"]


REFUSED_ASSET_URLS = [
    # Scheme is not https.
    "http://github.com/antigenomics/vdjdb-db/releases/download/t/vdjdb.zip",
    # https, but the netloc is not exactly github.com.
    "https://example.org/antigenomics/vdjdb-db/releases/download/t/vdjdb.zip",
    "https://github.com.attacker.example/antigenomics/vdjdb-db/releases/download/t/v.zip",
    "https://github.com@attacker.example/antigenomics/vdjdb-db/releases/download/t/v.zip",
    "https://github.com:8443/antigenomics/vdjdb-db/releases/download/t/vdjdb.zip",
    "https://notgithub.com/antigenomics/vdjdb-db/releases/download/t/vdjdb.zip",
    # https on github.com, but not under this repository's release downloads.
    "https://github.com/someone-else/vdjdb-db/releases/download/t/vdjdb.zip",
    "https://github.com/x/antigenomics/vdjdb-db/releases/download/t/vdjdb.zip",
    # The prefix matches, but ".." segments, encoded or not, climb out of it.
    REPO_PREFIX + "../../../someone-else/repo/releases/download/t/v.zip",
    REPO_PREFIX + "%2e%2e/%2E%2E/%2e%2e/someone-else/repo/releases/download/t/v.zip",
    REPO_PREFIX + "t/..%2f..%2f..%2fsomeone-else/v.zip",
    REPO_PREFIX + "t/..%5c..%5c..%5csomeone-else/v.zip",
    # urlsplit silently drops tab, CR and LF, so the check would otherwise
    # pass a URL that differs from the one handed to the opener.
    "https://git" + chr(9) + "hub.com/antigenomics/vdjdb-db/releases/download/t/v.zip",
    REPO_PREFIX + "t/v.zip" + chr(13) + chr(10) + "X-Injected: 1",
    # A space or a non-ASCII character has no place in a URL GitHub returns.
    REPO_PREFIX + "t/v .zip",
    REPO_PREFIX + "t/v" + chr(0xE9) + ".zip",
]


@pytest.mark.parametrize("url", REFUSED_ASSET_URLS)
def test_off_allowlist_asset_url_is_never_opened(url, tmp_path, network, capsys):
    network["release"] = _release(url=url)
    _route(network, url)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None
    assert _asset_urls(network) == []
    assert not os.path.exists(_zip(tmp_path))
    assert "Refusing VDJdb download URL" in capsys.readouterr().out


REFUSED_REDIRECT_TARGETS = [
    # A downgrade to http, or a hop to ftp, which urllib would follow.
    "http://objects.githubusercontent.com/release-asset/vdjdb.zip",
    "ftp://objects.githubusercontent.com/release-asset/vdjdb.zip",
    # Not a GitHub release-asset host.
    "https://attacker.example/vdjdb.zip",
    "https://objects.githubusercontent.com.attacker.example/vdjdb.zip",
    "https://attackergithubusercontent.com/vdjdb.zip",
    # githubusercontent.com hosts that serve user-writable content.
    "https://raw.githubusercontent.com/someone/repo/main/vdjdb.zip",
    "https://gist.githubusercontent.com/someone/abc/raw/vdjdb.zip",
    # A port on an allowed host.
    "https://objects.githubusercontent.com:8443/release-asset/vdjdb.zip",
]


@pytest.mark.parametrize("code", REDIRECT_CODES)
@pytest.mark.parametrize("target", REFUSED_REDIRECT_TARGETS)
def test_refused_redirect_target_is_never_requested(target, code, tmp_path, network, capsys):
    _redirect(network, ASSET_URL, target, code=code)
    _route(network, target)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None
    assert _asset_urls(network) == [ASSET_URL]
    assert sorted(os.listdir(_out(tmp_path))) == []
    assert "Refusing VDJdb download URL" in capsys.readouterr().out


@pytest.mark.parametrize("code", REDIRECT_CODES)
def test_an_unchecked_intermediate_hop_is_refused_even_if_the_last_is_allowed(
    code, tmp_path, network
):
    downgrade = "http://objects.githubusercontent.com/release-asset/vdjdb.zip"
    _redirect(network, ASSET_URL, downgrade, code=code)
    _redirect(network, downgrade, CDN_URL)
    _route(network, CDN_URL)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None
    assert _asset_urls(network) == [ASSET_URL]
    assert not os.path.exists(_zip(tmp_path))


@pytest.mark.parametrize(
    "host", ["objects.githubusercontent.com", "release-assets.githubusercontent.com"]
)
def test_redirect_to_a_github_release_asset_host_is_followed(host, tmp_path, network):
    target = f"https://{host}/release-asset/vdjdb.zip?sig=abc"
    _redirect(network, ASSET_URL, target)
    _route(network, target)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) == _tsv(tmp_path)
    assert _asset_urls(network) == [ASSET_URL, target]


def test_a_hop_within_the_repository_downloads_is_followed(tmp_path, network):
    renamed = REPO_PREFIX + "2026-06-03-ZENODO/vdjdb-renamed.zip"
    _redirect(network, ASSET_URL, renamed)
    _redirect(network, renamed, CDN_URL)
    _route(network, CDN_URL)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) == _tsv(tmp_path)


def test_refusal_message_omits_the_redirect_query(tmp_path, network, capsys):
    # A signed CDN URL carries its signature in the query; QUERYMARKER stands in.
    target = "https://attacker.example/vdjdb.zip?sig=QUERYMARKER"
    _redirect(network, ASSET_URL, target)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None
    out = capsys.readouterr().out
    assert "Refusing VDJdb download URL 'https://attacker.example/vdjdb.zip'" in out
    assert "QUERYMARKER" not in out


def test_every_asset_request_has_a_bounded_socket_timeout(tmp_path, network):
    _serve_via_cdn(network)
    ingest_vdjdb.fetch_vdjdb(_out(tmp_path))
    timeouts = [t for url, t in network["calls"] if url != ingest_vdjdb.VDJDB_RELEASES_API]
    assert len(timeouts) == 2
    for timeout in timeouts:
        assert isinstance(timeout, (int, float)) and 0 < timeout <= 300, timeout


@pytest.mark.parametrize(
    "wrong",
    ["0" * 64, PAYLOAD_SHA256[:8] + "0" * 56],  # the second shares the first 8 hex
    ids=["all-zero", "same-first-8"],
)
def test_digest_mismatch_leaves_no_archive_and_no_part_file(wrong, tmp_path, network, capsys):
    assert wrong != PAYLOAD_SHA256
    network["release"] = _release(digest="sha256:" + wrong)
    _pin(sha256=wrong)  # the API agrees with the pin; the BYTES are what differ
    _serve_via_cdn(network)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None
    assert sorted(os.listdir(_out(tmp_path))) == []
    assert "checksum mismatch" in capsys.readouterr().out


@pytest.mark.parametrize(
    "digest",
    [
        None,
        "",
        "md5:" + "0" * 32,
        "sha256:" + PAYLOAD_SHA256[:8],  # a proper prefix of the right digest
        "sha256:" + PAYLOAD_SHA256[:-1],
        "sha256:" + "z" * 64,
        "md5:" + PAYLOAD_SHA256,  # the right value under the wrong algorithm
    ],
    ids=["absent", "empty", "md5", "prefix8", "prefix63", "not-hex", "wrong-label"],
)
def test_asset_without_a_valid_sha256_digest_is_not_downloaded(digest, tmp_path, network, capsys):
    network["release"] = _release(digest=digest)
    _serve_via_cdn(network)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None
    assert _asset_urls(network) == []
    assert "sha256 digest" in capsys.readouterr().out


def test_uppercase_hex_digest_from_the_api_is_accepted(tmp_path, network):
    network["release"] = _release(digest="sha256:" + PAYLOAD_SHA256.upper())
    _serve_via_cdn(network)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) == _tsv(tmp_path)


def test_transfer_longer_than_the_reported_size_is_cut_off(tmp_path, network, capsys):
    network["release"] = _release(size=len(PAYLOAD) - 1)
    _pin(size=len(PAYLOAD) - 1)  # so the cap, not the pin comparison, is measured
    _serve_via_cdn(network)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None
    assert sorted(os.listdir(_out(tmp_path))) == []
    assert "exceeds" in capsys.readouterr().out


def test_reading_stops_at_the_reported_size(tmp_path, network, capsys):
    """The cap is checked while streaming, not after the whole body is read."""
    delivered: list[int] = []
    _serve_via_cdn(network, body=PAYLOAD * 50, chunk=64, delivered=delivered)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None
    assert sum(delivered) <= len(PAYLOAD) + 64, sum(delivered)
    assert sorted(os.listdir(_out(tmp_path))) == []
    assert "exceeds" in capsys.readouterr().out


@pytest.mark.parametrize(
    "size", [None, 0, "41017409", True], ids=["absent", "zero", "string", "bool"]
)
def test_asset_without_a_usable_size_is_not_downloaded(size, tmp_path, network, capsys):
    network["release"] = _release(size=size)
    _serve_via_cdn(network)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None
    assert _asset_urls(network) == []
    assert "size" in capsys.readouterr().out


def test_interrupted_transfer_leaves_no_archive_under_the_final_name(tmp_path, network):
    _serve_via_cdn(network, fail_after=1)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None
    assert sorted(os.listdir(_out(tmp_path))) == []


def test_keyboard_interrupt_mid_transfer_still_removes_the_part_file(tmp_path, network):
    _serve_via_cdn(network, fail_after=1, fail_with=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        ingest_vdjdb.fetch_vdjdb(_out(tmp_path))
    assert sorted(os.listdir(_out(tmp_path))) == []


def test_bytes_stream_to_a_part_file_never_to_the_final_name(tmp_path, network):
    seen: list[tuple[bool, bool]] = []

    def _observe():
        seen.append((os.path.exists(_zip(tmp_path)), os.path.exists(_zip(tmp_path) + ".part")))

    _serve_via_cdn(network, on_read=_observe)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) == _tsv(tmp_path)
    assert seen, "the fake response was never read"
    assert not any(final for final, _ in seen), seen
    assert any(part for _, part in seen), seen


def test_failed_extraction_leaves_no_archive_under_the_final_name(tmp_path, network):
    """vdjdb.zip must name an archive that was extracted, or a crash part way
    through extraction would leave it to be trusted on every later run."""
    unsafe = _zip_bytes(name="../outside.txt")
    network["release"] = _release(
        digest="sha256:" + hashlib.sha256(unsafe).hexdigest(), size=len(unsafe)
    )
    _serve_via_cdn(network, body=unsafe)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None
    assert sorted(os.listdir(_out(tmp_path))) == []
    assert not (tmp_path / "outside.txt").exists()


def _fail_mid_extraction(monkeypatch, exc: BaseException) -> None:
    """Make zipfile write 10 bytes of a member, then raise exc.

    zipfile copies each member with shutil.copyfileobj, and this script's own
    download loop does not use it, so only extraction is affected.
    """

    def _fail(source, target, length=0):
        target.write(source.read(10))
        raise exc

    monkeypatch.setattr(shutil, "copyfileobj", _fail)


def _disk_full() -> OSError:
    return OSError(errno.ENOSPC, "No space left on device (simulated)")


@pytest.fixture
def disk_fills_mid_extraction(monkeypatch):
    _fail_mid_extraction(monkeypatch, _disk_full())


def _corrupt_member_crc(payload: bytes) -> bytes:
    """Flip one byte of a STORED member's data, keeping the archive structure."""
    start = payload.index(TSV_TEXT.encode("ascii"))
    return payload[:start] + bytes([payload[start] ^ 0x01]) + payload[start + 1 :]


DAMAGED = [PAYLOAD[: len(PAYLOAD) // 2], _corrupt_member_crc(_zip_bytes(zipfile.ZIP_STORED))]
DAMAGED_IDS = ["truncated", "bad-crc"]


def _place(tmp_path, zip_bytes: bytes | None = None, tsv: str | None = None) -> None:
    os.makedirs(_out(tmp_path))
    if zip_bytes is not None:
        with open(_zip(tmp_path), "wb") as handle:
            handle.write(zip_bytes)
    if tsv is not None:
        with open(_tsv(tmp_path), "w", encoding="ascii") as handle:
            handle.write(tsv)


@pytest.mark.parametrize("damaged", DAMAGED, ids=DAMAGED_IDS)
def test_damaged_existing_archive_is_replaced_by_a_verified_download(damaged, tmp_path, network):
    _place(tmp_path, damaged)
    _serve_via_cdn(network)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) == _tsv(tmp_path)
    assert _asset_urls(network) == [ASSET_URL, CDN_URL]
    with open(_zip(tmp_path), "rb") as handle:
        assert handle.read() == PAYLOAD


@pytest.mark.parametrize("damaged", DAMAGED, ids=DAMAGED_IDS)
def test_damaged_archive_is_kept_when_the_fetch_fails_offline(damaged, tmp_path, network):
    """Offline, the earlier extraction is still used, as before this check existed."""
    _place(tmp_path, damaged, tsv=TSV_TEXT)
    network["offline"] = True
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) == _tsv(tmp_path)
    with open(_zip(tmp_path), "rb") as handle:
        assert handle.read() == damaged
    assert sorted(os.listdir(_out(tmp_path))) == ["vdjdb.txt", "vdjdb.zip"]


def test_damaged_archive_is_kept_when_the_download_fails_verification(tmp_path, network):
    damaged = DAMAGED[0]
    _place(tmp_path, damaged, tsv=TSV_TEXT)
    network["release"] = _release(digest="sha256:" + "0" * 64)
    _serve_via_cdn(network)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) == _tsv(tmp_path)
    with open(_zip(tmp_path), "rb") as handle:
        assert handle.read() == damaged


OLD_TSV = "antigen.epitope\tmhc.a\n" + "NLVPMVATV\tHLA-A*02:01\n" * 50


def test_table_half_written_by_a_failed_extraction_is_not_returned(
    tmp_path, network, disk_fills_mid_extraction, capsys
):
    """A damaged archive beside a good table, and the verified replacement's
    extraction overwrites that table and then fails: the table on disk is no
    longer the earlier one, so it must not be returned as if it were."""
    _place(tmp_path, DAMAGED[0], tsv=OLD_TSV)
    _serve_via_cdn(network)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None
    out = capsys.readouterr().out
    assert os.path.getsize(_tsv(tmp_path)) == 10  # what this run left behind
    # The damaged archive is removed too, so a later run cannot fall back to it.
    assert sorted(os.listdir(_out(tmp_path))) == ["vdjdb.txt"]
    assert "may have been partly overwritten; not using them" in out


def _serve_corrupt_member_with_its_own_digest(state: dict) -> None:
    """An upstream archive whose member fails its CRC still matches its digest."""
    corrupt = DAMAGED[1]
    _repin_to(state, corrupt)
    _serve_via_cdn(state, body=corrupt)


@pytest.mark.parametrize("failure", ["disk-full", "bad-crc-member"])
def test_a_table_half_written_in_one_run_is_not_used_by_the_next(
    failure, tmp_path, network, monkeypatch, capsys
):
    _place(tmp_path, DAMAGED[0], tsv=OLD_TSV)
    with monkeypatch.context() as patch:
        if failure == "disk-full":
            _serve_via_cdn(network)
            _fail_mid_extraction(patch, _disk_full())
        else:  # zipfile writes the member, then raises BadZipFile (not an OSError)
            _serve_corrupt_member_with_its_own_digest(network)
        assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None
    out = capsys.readouterr().out
    assert os.path.getsize(_tsv(tmp_path)) != len(OLD_TSV)
    network["offline"] = True
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None  # the next run, offline
    assert sorted(os.listdir(_out(tmp_path))) == ["vdjdb.txt"]
    assert "may have been partly overwritten" in out


def test_an_interrupt_during_extraction_propagates_and_removes_the_damaged_archive(
    tmp_path, network, monkeypatch
):
    _place(tmp_path, DAMAGED[0], tsv=OLD_TSV)
    _serve_via_cdn(network)
    with monkeypatch.context() as patch:
        _fail_mid_extraction(patch, KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            ingest_vdjdb.fetch_vdjdb(_out(tmp_path))
    assert sorted(os.listdir(_out(tmp_path))) == ["vdjdb.txt"]
    network["offline"] = True
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None


def test_a_part_file_that_cannot_be_removed_does_not_bring_back_the_damaged_archive(
    tmp_path, network, monkeypatch
):
    """On Windows a second handle on vdjdb.zip.part makes removing it raise.
    The cleanup stops there and the PermissionError replaces the extraction
    failure, so fetch_vdjdb takes its generic branch, which falls back to the
    table beside vdjdb.zip whenever that file exists. The damaged archive must
    already be gone, or the table this run half wrote is returned."""
    _place(tmp_path, DAMAGED[0], tsv=OLD_TSV)
    _serve_via_cdn(network)
    real_remove = os.remove

    def _remove(path, *args, **kwargs):
        if str(path).endswith(".part"):
            raise PermissionError(errno.EACCES, "in use by another process (simulated)", path)
        return real_remove(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        _fail_mid_extraction(patch, _disk_full())
        patch.setattr(os, "remove", _remove)
        assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None
    assert os.path.getsize(_tsv(tmp_path)) == 10  # what this run left behind
    assert not os.path.exists(_zip(tmp_path))


def test_a_member_extracted_onto_the_archive_name_is_removed_after_a_failure(
    tmp_path, network, monkeypatch
):
    """The archive name is cleared as extraction starts, but a member named
    vdjdb.zip is extracted onto it. After a later failure that file must go
    too, or the next offline run falls back to the table this run half wrote."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("vdjdb.zip", "not an archive")
        archive.writestr("vdjdb.txt", TSV_TEXT)
    payload = buffer.getvalue()
    _place(tmp_path, DAMAGED[0], tsv=OLD_TSV)
    _repin_to(network, payload)
    _serve_via_cdn(network, body=payload)
    real_copy = shutil.copyfileobj
    copies: list[int] = []

    def _fail_on_the_second_member(source, target, length=0):
        copies.append(1)
        if len(copies) == 1:
            return real_copy(source, target)
        target.write(source.read(10))
        raise _disk_full()

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copyfileobj", _fail_on_the_second_member)
        assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None
    assert sorted(os.listdir(_out(tmp_path))) == ["vdjdb.txt"]
    network["offline"] = True
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None


@pytest.mark.parametrize("failure", ["refused-url", "refused-redirect", "oversize"])
def test_damaged_archive_is_kept_when_the_fetch_fails_before_extraction(failure, tmp_path, network):
    _place(tmp_path, DAMAGED[0], tsv=TSV_TEXT)
    if failure == "refused-url":
        network["release"] = _release(url="https://example.org/vdjdb.zip")
        _route(network, "https://example.org/vdjdb.zip")
    elif failure == "refused-redirect":
        _redirect(network, ASSET_URL, "https://attacker.example/vdjdb.zip")
    else:
        network["release"] = _release(size=len(PAYLOAD) - 1)
        _pin(size=len(PAYLOAD) - 1)  # so this leg still fails on the cap
        _serve_via_cdn(network)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) == _tsv(tmp_path)
    with open(_zip(tmp_path), "rb") as handle:
        assert handle.read() == DAMAGED[0]


def test_first_run_partial_extraction_is_not_returned(tmp_path, network, disk_fills_mid_extraction):
    _serve_via_cdn(network)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None
    assert sorted(os.listdir(_out(tmp_path))) == ["vdjdb.txt"]  # 10 bytes, unused


def test_archive_takes_its_final_name_only_after_extraction(tmp_path, network, monkeypatch):
    """A process killed mid-extraction runs no cleanup at all; it must not
    leave a verified vdjdb.zip that the next run trusts beside a partial table."""
    seen: list[bool] = []
    real_copy = shutil.copyfileobj

    def _observe(source, target, length=0):
        seen.append(os.path.exists(_zip(tmp_path)))
        return real_copy(source, target)

    monkeypatch.setattr(shutil, "copyfileobj", _observe)
    _serve_via_cdn(network)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) == _tsv(tmp_path)
    assert seen == [False]
    assert os.path.exists(_zip(tmp_path))


@pytest.mark.parametrize("damaged", DAMAGED, ids=DAMAGED_IDS)
def test_damaged_archive_is_gone_before_any_member_is_written(
    damaged, tmp_path, network, monkeypatch
):
    """A process killed mid-extraction runs no cleanup. If the damaged archive
    were still there while members are written, the next offline run would
    fall back to a table this one half wrote."""
    _place(tmp_path, damaged, tsv=OLD_TSV)
    seen: list[bool] = []
    real_copy = shutil.copyfileobj

    def _observe(source, target, length=0):
        seen.append(os.path.exists(_zip(tmp_path)))
        return real_copy(source, target)

    monkeypatch.setattr(shutil, "copyfileobj", _observe)
    _serve_via_cdn(network)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) == _tsv(tmp_path)
    assert seen == [False]
    with open(_zip(tmp_path), "rb") as handle:
        assert handle.read() == PAYLOAD


def test_stray_table_without_an_archive_is_not_used_when_offline(tmp_path, network):
    """With no vdjdb.zip there is nothing to fall back on, as before."""
    _place(tmp_path, tsv=TSV_TEXT)
    network["offline"] = True
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None


@pytest.mark.parametrize("damaged", DAMAGED, ids=DAMAGED_IDS)
def test_damaged_archive_with_no_table_offline_returns_none(damaged, tmp_path, network):
    """A damaged archive is never extracted, so nothing raises."""
    _place(tmp_path, damaged)
    network["offline"] = True
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) is None


def test_intact_existing_archive_is_reused_without_network(tmp_path, network):
    _place(tmp_path, PAYLOAD, tsv=TSV_TEXT)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) == _tsv(tmp_path)
    assert network["calls"] == []


def test_intact_archive_with_nothing_extracted_is_extracted_again(tmp_path, network):
    _place(tmp_path, PAYLOAD)
    assert ingest_vdjdb.fetch_vdjdb(_out(tmp_path)) == _tsv(tmp_path)
    assert network["calls"] == []
    with open(_tsv(tmp_path), encoding="ascii") as handle:
        assert handle.read() == TSV_TEXT
