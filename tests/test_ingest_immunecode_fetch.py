"""Tests for the ImmuneCODE MIRA archive download in scripts/ingest_immunecode.py.

With no --input, the script downloads one fixed URL, MIRA_DOWNLOAD_URL, into a
temporary directory. These tests pin five properties of that download:

* the start URL, and every redirect target (301, 302, 303, 307 and 308) before
  urllib requests it, must be https on one of the two allowed hosts exactly:
  adaptivepublic.blob.core.windows.net, where the publisher's listing page links
  the archive, or clients.adaptivebiotech.com;
* every request has a bounded socket timeout;
* the bytes go to a ``.part`` file, which takes the archive's name only once it
  opens as a ZIP whose every member passes its CRC-32 check and matches the
  pinned size and sha256, and any failure, KeyboardInterrupt included, removes
  the ``.part`` file. The pin itself is tested in test_immunecode_archive_pin.py;
  here it points at the synthetic archive each test serves;
* a refused download, a network error mid-transfer or a damaged archive makes
  main() return 1 with the reason logged, instead of raising out of it, and an
  error removing the ``.part`` file does not replace that outcome;
* main() downloads into a directory under tempfile's root that no longer exists
  when it returns, and leaves nothing in that root, for the successful run and
  the failed runs tested here.

No test opens a socket. The transfer goes through in-memory https, http and ftp
handlers that serve real redirect responses, so urllib's own redirect machinery
runs, and any socket connect is recorded, refused and fails the test.
"""

from __future__ import annotations

import errno
import hashlib
import http.client
import io
import logging
import os
import socket
import sys
import tempfile
import urllib.request
import urllib.response
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import ingest_immunecode  # noqa: E402

HOST_URL = "https://adaptivepublic.blob.core.windows.net/"
OTHER_HOST_URL = "https://clients.adaptivebiotech.com/"
ZIP_NAME = "ImmuneCODE-MIRA-Release002.1.zip"
URL = HOST_URL + "publishedproject-supplements/covid-2020/" + ZIP_NAME
ALLOWED = "not https on adaptivepublic.blob.core.windows.net or clients.adaptivebiotech.com"
MIRA_CSV = (
    "Peptide,HLA Restrictions,Amino Acids\n"
    "YLQPRTFLL,HLA-A*02:01,\n"
    "LTDEMIAQY,HLA-A*02:01,CASSIRSSYEQYF\n"
    "KIADYNYKL,HLA-B*07:02,\n"
)


def _zip_bytes(compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        archive.writestr("peptide-detail-cov2.csv", MIRA_CSV)
    return buffer.getvalue()


def _damaged_member(payload: bytes) -> bytes:
    """Flip one byte of a STORED member's data, keeping the archive structure."""
    start = payload.index(MIRA_CSV.encode("ascii"))
    return payload[:start] + bytes([payload[start] ^ 0x01]) + payload[start + 1 :]


PAYLOAD = _zip_bytes()
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()
TRUNCATED = PAYLOAD[: len(PAYLOAD) // 2]
BAD_CRC = _damaged_member(_zip_bytes(zipfile.ZIP_STORED))


def test_the_url_these_tests_serve_is_the_scripts_url():
    assert ingest_immunecode.MIRA_DOWNLOAD_URL == URL


class _Body(io.BytesIO):
    """A response body with an optional per-read hook and a mid-transfer failure."""

    def __init__(self, payload: bytes, on_read=None, fail_after: int = -1, fail_with=None):
        super().__init__(payload)
        self._on_read = on_read
        self._fail_after = fail_after
        self._fail_with = fail_with or TimeoutError("simulated stalled transfer")
        self._reads = 0

    def read(self, size: int | None = -1) -> bytes:
        if self._on_read is not None:
            self._on_read()
        self._reads += 1
        if self._fail_after >= 0 and self._reads > self._fail_after:
            raise self._fail_with
        if self._fail_after >= 0:
            # Hand back only part of the body, so the transfer is truncated.
            return super().read(max(1, len(self.getvalue()) // 4))
        return super().read(size)


def _response(url: str, code: int, body: bytes, location=None, **body_kw):
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
    """Route every request to in-memory responses and refuse real sockets.

    ``state["routes"]`` maps a URL to a response factory and ``state["calls"]``
    records (url, timeout) for every request opened.
    """
    state: dict = {"routes": {}, "calls": [], "connects": []}
    # The synthetic PAYLOAD stands in for the pinned release-002.1 archive, so
    # the pin points at it for the duration of each test. The pin's real literals
    # live in test_immunecode_archive_pin.py; these tests are about the transfer,
    # the URL allowlist and the .part staging, not about which bytes are pinned.
    monkeypatch.setattr(ingest_immunecode, "MIRA_PINNED_SHA256", PAYLOAD_SHA256)
    monkeypatch.setattr(ingest_immunecode, "MIRA_PINNED_SIZE_BYTES", len(PAYLOAD))
    real_build_opener = urllib.request.build_opener

    def _build_opener(*handlers):
        return real_build_opener(*handlers, _FakeHTTPS(state), _FakeHTTP(state), _FakeFTP(state))

    def _urlopen(url, *args, **kwargs):
        # What plain urlopen and urlretrieve do: urllib's default handlers,
        # which follow every http, https and ftp redirect unchecked.
        return _build_opener().open(url, *args, **kwargs)

    def _no_connect(self, *args, **kwargs):
        state["connects"].append(args)
        raise OSError("a socket connect was attempted during a no-network test")

    monkeypatch.setattr(urllib.request, "build_opener", _build_opener)
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(socket.socket, "connect", _no_connect)
    yield state
    assert state["connects"] == [], state["connects"]


def _route(state: dict, url: str, body: bytes = PAYLOAD, **body_kw) -> None:
    state["routes"][url] = lambda: _response(url, 200, body, **body_kw)


def _pin_size_to(body: bytes) -> None:
    """Admit `body` past the pinned-size cap, so a later check is what rejects it.

    The ``network`` fixture monkeypatched MIRA_PINNED_SIZE_BYTES, so it restores
    the real value at teardown. BAD_CRC is STORED, so it is longer than PAYLOAD
    and would otherwise be cut off by the cap rather than failing the structural
    check it was written to exercise. The cap is tested in
    test_immunecode_archive_pin.py.
    """
    ingest_immunecode.MIRA_PINNED_SIZE_BYTES = len(body)


def _redirect(state: dict, url: str, location: str, code: int = 302) -> None:
    state["routes"][url] = lambda: _response(url, code, b"", location)


# Every status urllib's HTTPRedirectHandler follows for a GET.
REDIRECT_CODES = [301, 302, 303, 307, 308]


def _opened(state: dict) -> list[str]:
    return [url for url, _ in state["calls"]]


def _listing(path) -> list[str]:
    return sorted(os.listdir(path)) if os.path.isdir(path) else []


def test_verified_download_returns_the_archive_and_leaves_no_part_file(tmp_path, network):
    _route(network, URL)
    path = ingest_immunecode._download_mira(str(tmp_path))
    assert path == os.path.join(str(tmp_path), ZIP_NAME)
    with open(path, "rb") as handle:
        assert handle.read() == PAYLOAD
    assert _listing(tmp_path) == [ZIP_NAME]


REFUSED_START_URLS = [
    # Scheme is not https.
    "http://clients.adaptivebiotech.com/pub/covid-2020/immuneCODE-MIRA-release002.1.zip",
    "ftp://clients.adaptivebiotech.com/pub/covid-2020/immuneCODE-MIRA-release002.1.zip",
    # https, but the netloc is not exactly clients.adaptivebiotech.com.
    "https://example.org/pub/covid-2020/immuneCODE-MIRA-release002.1.zip",
    "https://clients.adaptivebiotech.com.attacker.example/pub/covid-2020/m.zip",
    "https://clients.adaptivebiotech.com@attacker.example/pub/covid-2020/m.zip",
    "https://clients.adaptivebiotech.com:8443/pub/covid-2020/m.zip",
    "https://adaptivebiotech.com/pub/covid-2020/immuneCODE-MIRA-release002.1.zip",
    "https://www.adaptivebiotech.com/pub/covid-2020/immuneCODE-MIRA-release002.1.zip",
    # urlsplit silently drops tab, CR and LF, so the check would otherwise
    # pass a URL that differs from the one handed to the opener.
    "https://clients.adaptive" + chr(9) + "biotech.com/pub/covid-2020/m.zip",
    URL + chr(13) + chr(10) + "X-Injected: 1",
    # A space or a non-ASCII character has no place in this URL.
    HOST_URL + "pub/covid-2020/immuneCODE MIRA.zip",
    HOST_URL + "pub/covid-2020/immuneCODE-MIRA-r" + chr(0xE9) + "lease.zip",
    # The blob host the archive moved to, spelled any other way.
    "http://adaptivepublic.blob.core.windows.net/publishedproject-supplements/m.zip",
    "https://adaptivepublic.blob.core.windows.net.attacker.example/m.zip",
    "https://adaptivepublic.blob.core.windows.net:8443/m.zip",
    "https://user@adaptivepublic.blob.core.windows.net/m.zip",
    "https://otheraccount.blob.core.windows.net/publishedproject-supplements/m.zip",
    "https://blob.core.windows.net/publishedproject-supplements/m.zip",
]


@pytest.mark.parametrize("url", REFUSED_START_URLS)
def test_off_allowlist_start_url_is_never_opened(url, tmp_path, network, monkeypatch):
    monkeypatch.setattr(ingest_immunecode, "MIRA_DOWNLOAD_URL", url)
    _route(network, url)
    with pytest.raises(RuntimeError, match="Refusing ImmuneCODE download URL"):
        ingest_immunecode._download_mira(str(tmp_path))
    assert _opened(network) == []
    assert _listing(tmp_path) == []


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        (REFUSED_START_URLS[0], ALLOWED),
        (REFUSED_START_URLS[3], ALLOWED),
        (REFUSED_START_URLS[8], "it holds a control, space or non-ASCII character"),
        (REFUSED_START_URLS[10], "it holds a control, space or non-ASCII character"),
        (REFUSED_START_URLS[16], ALLOWED),
    ],
    ids=["http", "look-alike", "tab", "space", "other-storage-account"],
)
def test_the_refusal_names_its_reason(url, reason, tmp_path, network, monkeypatch):
    monkeypatch.setattr(ingest_immunecode, "MIRA_DOWNLOAD_URL", url)
    with pytest.raises(RuntimeError) as raised:
        ingest_immunecode._download_mira(str(tmp_path))
    assert reason in str(raised.value)


REFUSED_REDIRECT_TARGETS = [
    # A downgrade to http, or a hop to ftp, both of which urllib would follow.
    "http://clients.adaptivebiotech.com/pub/covid-2020/immuneCODE-MIRA-release002.1.zip",
    "ftp://clients.adaptivebiotech.com/pub/covid-2020/immuneCODE-MIRA-release002.1.zip",
    # Any other host, a look-alike, a port, or userinfo.
    "https://attacker.example/immuneCODE-MIRA-release002.1.zip",
    "https://clients.adaptivebiotech.com.attacker.example/m.zip",
    "https://clients.adaptivebiotech.com:8443/pub/covid-2020/m.zip",
    "https://user@clients.adaptivebiotech.com/pub/covid-2020/m.zip",
    "http://adaptivepublic.blob.core.windows.net/publishedproject-supplements/m.zip",
    "https://otheraccount.blob.core.windows.net/publishedproject-supplements/m.zip",
    "https://adaptivepublic.blob.core.windows.net.attacker.example/m.zip",
]


def test_the_allowlist_is_exactly_the_two_publisher_hosts():
    assert ingest_immunecode.MIRA_DOWNLOAD_HOSTS == {
        "adaptivepublic.blob.core.windows.net",
        "clients.adaptivebiotech.com",
    }
    assert ingest_immunecode.MIRA_DOWNLOAD_URL.startswith(HOST_URL)


@pytest.mark.parametrize("target", REFUSED_REDIRECT_TARGETS)
def test_refused_redirect_target_is_never_requested(target, tmp_path, network):
    _redirect(network, URL, target)
    _route(network, target)
    with pytest.raises(RuntimeError, match="Refusing ImmuneCODE download URL"):
        ingest_immunecode._download_mira(str(tmp_path))
    assert _opened(network) == [URL]
    assert _listing(tmp_path) == []


def test_an_unchecked_intermediate_hop_is_refused_even_if_the_last_is_allowed(tmp_path, network):
    downgrade = "http://clients.adaptivebiotech.com/pub/covid-2020/m.zip"
    final = HOST_URL + "pub/covid-2020/m.zip"
    _redirect(network, URL, downgrade)
    _redirect(network, downgrade, final)
    _route(network, final)
    with pytest.raises(RuntimeError, match="Refusing ImmuneCODE download URL"):
        ingest_immunecode._download_mira(str(tmp_path))
    assert _opened(network) == [URL]
    assert _listing(tmp_path) == []


@pytest.mark.parametrize("code", REDIRECT_CODES)
def test_a_downgrade_is_refused_whatever_the_redirect_status(code, tmp_path, network):
    downgrade = "http://clients.adaptivebiotech.com/pub/covid-2020/m.zip"
    _redirect(network, URL, downgrade, code=code)
    _route(network, downgrade)
    with pytest.raises(RuntimeError, match="Refusing ImmuneCODE download URL"):
        ingest_immunecode._download_mira(str(tmp_path))
    assert _opened(network) == [URL]
    assert _listing(tmp_path) == []


@pytest.mark.parametrize("code", REDIRECT_CODES)
@pytest.mark.parametrize("host", [HOST_URL, OTHER_HOST_URL], ids=["same-host", "other-allowed"])
def test_a_redirect_to_an_allowed_host_is_followed(code, host, tmp_path, network):
    target = host + "pub/covid-2020/renamed.zip?sig=abc"
    _redirect(network, URL, target, code=code)
    _route(network, target)
    path = ingest_immunecode._download_mira(str(tmp_path))
    assert _opened(network) == [URL, target]
    with open(path, "rb") as handle:
        assert handle.read() == PAYLOAD


def test_refusal_message_omits_the_redirect_query(tmp_path, network):
    # A signed URL carries its signature in the query; QUERYMARKER stands in.
    _redirect(network, URL, "https://attacker.example/m.zip?sig=QUERYMARKER")
    with pytest.raises(RuntimeError) as raised:
        ingest_immunecode._download_mira(str(tmp_path))
    assert "Refusing ImmuneCODE download URL 'https://attacker.example/m.zip'" in str(raised.value)
    assert "QUERYMARKER" not in str(raised.value)


def test_every_request_has_a_bounded_socket_timeout(tmp_path, network):
    target = HOST_URL + "pub/covid-2020/renamed.zip"
    _redirect(network, URL, target)
    _route(network, target)
    ingest_immunecode._download_mira(str(tmp_path))
    timeouts = [timeout for _, timeout in network["calls"]]
    assert len(timeouts) == 2
    for timeout in timeouts:
        assert isinstance(timeout, (int, float)) and 0 < timeout <= 300, timeout


@pytest.mark.parametrize(
    "body",
    [b"<html><body>Sign in</body></html>", TRUNCATED, BAD_CRC],
    ids=["not-a-zip", "truncated", "bad-crc"],
)
def test_a_damaged_download_is_refused_and_leaves_nothing(body, tmp_path, network):
    _route(network, URL, body=body)
    _pin_size_to(body)
    with pytest.raises(RuntimeError, match="ZIP archive"):
        ingest_immunecode._download_mira(str(tmp_path))
    assert _listing(tmp_path) == []


def test_interrupted_transfer_leaves_nothing_behind(tmp_path, network):
    _route(network, URL, fail_after=1)
    with pytest.raises(RuntimeError, match="Download failed"):
        ingest_immunecode._download_mira(str(tmp_path))
    assert _listing(tmp_path) == []


def test_keyboard_interrupt_mid_transfer_still_removes_the_part_file(tmp_path, network):
    _route(network, URL, fail_after=1, fail_with=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        ingest_immunecode._download_mira(str(tmp_path))
    assert _listing(tmp_path) == []


def test_bytes_stream_to_a_part_file_never_to_the_final_name(tmp_path, network):
    final = os.path.join(str(tmp_path), ZIP_NAME)
    seen: list[tuple[bool, bool]] = []

    def _observe():
        seen.append((os.path.exists(final), os.path.exists(final + ".part")))

    _route(network, URL, on_read=_observe)
    ingest_immunecode._download_mira(str(tmp_path))
    assert seen, "the fake response was never read"
    assert not any(at_final for at_final, _ in seen), seen
    assert any(at_part for _, at_part in seen), seen


@pytest.fixture
def temp_root(tmp_path, monkeypatch):
    """Point tempfile at an empty directory, so what a run leaves there is visible."""
    root = tmp_path / "tmp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


@pytest.fixture
def download_dirs(monkeypatch):
    """Record each directory main() hands to _download_mira, then run the real one."""
    seen: list[str] = []
    real = ingest_immunecode._download_mira

    def _spy(dest_dir):
        seen.append(dest_dir)
        return real(dest_dir)

    monkeypatch.setattr(ingest_immunecode, "_download_mira", _spy)
    return seen


def _assert_downloaded_into_a_removed_temporary_dir(dirs: list[str], root) -> None:
    assert len(dirs) == 1, dirs
    dest, top = os.path.realpath(dirs[0]), os.path.realpath(root)
    assert dest != top and os.path.commonpath([dest, top]) == top, (dest, top)
    assert not os.path.exists(dirs[0]), dirs[0]
    assert _listing(root) == []


def test_auto_download_writes_negatives_and_leaves_no_temporary_file(
    tmp_path, network, temp_root, download_dirs
):
    _route(network, URL)
    out = tmp_path / "out" / "negatives.csv"
    assert ingest_immunecode.main(["--output", str(out)]) == 0
    with open(out, encoding="utf-8") as handle:
        assert handle.readline().startswith("peptide,label,hla_allele")
    _assert_downloaded_into_a_removed_temporary_dir(download_dirs, temp_root)


@pytest.mark.parametrize(
    ("serve", "reason"),
    [
        (
            lambda state: (_route(state, URL, body=BAD_CRC), _pin_size_to(BAD_CRC)),
            "not a complete, undamaged ZIP archive",
        ),
        (lambda state: _route(state, URL, fail_after=1), "simulated stalled transfer"),
        (
            lambda state: _route(
                state, URL, fail_after=1, fail_with=ConnectionResetError("simulated reset")
            ),
            "simulated reset",
        ),
        (
            lambda state: _redirect(state, URL, "http://clients.adaptivebiotech.com/m.zip"),
            "Refusing ImmuneCODE download URL",
        ),
    ],
    ids=["bad-crc", "timeout", "connection-reset", "refused-redirect"],
)
def test_a_failed_auto_download_returns_1_and_leaves_no_file(
    serve, reason, tmp_path, network, temp_root, download_dirs, caplog
):
    serve(network)
    out = tmp_path / "out" / "negatives.csv"
    with caplog.at_level(logging.ERROR):
        assert ingest_immunecode.main(["--output", str(out)]) == 1
    assert reason in caplog.text
    assert not out.exists()
    _assert_downloaded_into_a_removed_temporary_dir(download_dirs, temp_root)


def test_a_part_file_that_cannot_be_removed_still_returns_1(
    tmp_path, network, temp_root, monkeypatch, caplog
):
    """On Windows a second handle on the .part file makes removing it raise.
    That error must not replace the download failure and escape main().
    Only os.remove is made to fail here: a handle still held when the
    temporary directory is cleaned up makes that cleanup raise too, which this
    test does not cover."""
    _route(network, URL, body=BAD_CRC)
    _pin_size_to(BAD_CRC)
    real_remove = os.remove

    def _remove(path, *args, **kwargs):
        if str(path).endswith(".part"):
            raise PermissionError(errno.EACCES, "in use by another process (simulated)", path)
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(os, "remove", _remove)
    out = tmp_path / "out" / "negatives.csv"
    with caplog.at_level(logging.ERROR):
        assert ingest_immunecode.main(["--output", str(out)]) == 1
    assert "not a complete, undamaged ZIP archive" in caplog.text
    assert not out.exists()
