"""Offline tests for the verified Stage-4 model fetcher."""

from __future__ import annotations

import functools
import hashlib
import io
import json
import os
import sys
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPHandler, HTTPSHandler, ProxyHandler, Request
from urllib.response import addinfourl

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import fetch_verified_stage4_model as fetcher  # noqa: E402
from src.artifact_integrity import ArtifactIntegrityError  # noqa: E402

URL = "https://example.invalid/rf_31feature_integrated.joblib"


class FakeResponse:
    def __init__(self, data: bytes, declared_length: int | None = None) -> None:
        self._data = data
        self._offset = 0
        self.headers = {}
        if declared_length is not None:
            self.headers["Content-Length"] = str(declared_length)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def _write_manifest(
    root: Path,
    entries: dict[str, bytes],
    *,
    size_delta: int = 0,
    digest_override: str | None = None,
) -> Path:
    artifacts = {}
    for key, data in entries.items():
        artifacts[key] = {
            "sha256": digest_override or hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data) + size_delta,
        }
    manifest = root / "model_artifact_checksums.json"
    manifest.write_text(json.dumps({"generated_utc": None, "artifacts": artifacts}))
    return manifest


def _install_response(
    monkeypatch: pytest.MonkeyPatch, data: bytes, length: int | None = None
) -> None:
    monkeypatch.setattr(fetcher, "urlopen", lambda *args, **kwargs: FakeResponse(data, length))


def test_success_is_atomic(tmp_path, monkeypatch):
    data = b"verified-model"
    target = tmp_path / "rf.joblib"
    manifest = _write_manifest(tmp_path, {target.name: data})

    class AtomicResponse(FakeResponse):
        def read(self, size: int) -> bytes:
            assert not target.exists()
            return super().read(size)

    monkeypatch.setattr(fetcher, "urlopen", lambda *args, **kwargs: AtomicResponse(data, len(data)))
    assert fetcher.fetch_stage4_model(URL, target, manifest_path=manifest) == target
    assert target.read_bytes() == data
    assert not target.with_suffix(".joblib.tmp").exists()


def test_digest_mismatch_removes_destination_and_partial(tmp_path, monkeypatch):
    data = b"wrong-model"
    target = tmp_path / "rf.joblib"
    manifest = _write_manifest(tmp_path, {target.name: data}, digest_override="0" * 64)
    _install_response(monkeypatch, data, len(data))
    with pytest.raises(ArtifactIntegrityError, match="Checksum verification failed"):
        fetcher.fetch_stage4_model(URL, target, manifest_path=manifest)
    assert not target.exists()
    assert not target.with_suffix(".joblib.tmp").exists()


def test_size_mismatch_removes_partial(tmp_path, monkeypatch):
    data = b"short"
    target = tmp_path / "rf.joblib"
    manifest = _write_manifest(tmp_path, {target.name: data}, size_delta=1)
    _install_response(monkeypatch, data, len(data))
    with pytest.raises(ArtifactIntegrityError, match="Size verification failed"):
        fetcher.fetch_stage4_model(URL, target, manifest_path=manifest)
    assert not target.exists()
    assert not target.with_suffix(".joblib.tmp").exists()


def test_truncated_stream_removes_partial(tmp_path, monkeypatch):
    data = b"short"
    target = tmp_path / "rf.joblib"
    manifest = _write_manifest(tmp_path, {target.name: data})
    _install_response(monkeypatch, data, len(data) + 9)
    with pytest.raises(ArtifactIntegrityError, match="Truncated download"):
        fetcher.fetch_stage4_model(URL, target, manifest_path=manifest)
    assert not target.exists()
    assert not target.with_suffix(".joblib.tmp").exists()


def test_http_error_leaves_no_files(tmp_path, monkeypatch):
    target = tmp_path / "rf.joblib"
    manifest = _write_manifest(tmp_path, {target.name: b"model"})

    def fail(*args, **kwargs):
        raise HTTPError(URL, 503, "unavailable", {}, None)

    monkeypatch.setattr(fetcher, "urlopen", fail)
    with pytest.raises(HTTPError):
        fetcher.fetch_stage4_model(URL, target, manifest_path=manifest)
    assert not target.exists()
    assert not target.with_suffix(".joblib.tmp").exists()


class CountingStream(FakeResponse):
    """Streams ``total`` zero bytes on demand and counts how many were read."""

    def __init__(self, total: int) -> None:
        super().__init__(b"")
        self.total = total
        self.sent = 0

    def read(self, size: int) -> bytes:
        n = min(size, self.total - self.sent)
        self.sent += n
        return b"\0" * n


def test_oversize_stream_is_cut_off_at_the_manifest_size(tmp_path, monkeypatch):
    data = b"verified-model"
    target = tmp_path / "rf.joblib"
    manifest = _write_manifest(tmp_path, {target.name: data})
    response = CountingStream(64 * fetcher.CHUNK_SIZE)
    monkeypatch.setattr(fetcher, "urlopen", lambda *args, **kwargs: response)
    with pytest.raises(ArtifactIntegrityError, match="exceeds the manifest size"):
        fetcher.fetch_stage4_model(URL, target, manifest_path=manifest)
    assert response.sent <= len(data) + fetcher.CHUNK_SIZE
    assert not target.exists()
    assert not target.with_suffix(".joblib.tmp").exists()


def test_connection_reset_mid_stream_removes_partial(tmp_path, monkeypatch):
    # test_http_error_leaves_no_files cannot see the cleanup: urlopen raises there
    # before the .tmp is opened. Here one chunk is on disk when the stream dies.
    data = b"m" * (fetcher.CHUNK_SIZE + 1)
    target = tmp_path / "rf.joblib"
    manifest = _write_manifest(tmp_path, {target.name: data})

    class ResetAfterFirstChunk(FakeResponse):
        def read(self, size: int) -> bytes:
            if self._offset:
                raise ConnectionResetError("reset by peer")
            return super().read(size)

    monkeypatch.setattr(fetcher, "urlopen", lambda *args, **kwargs: ResetAfterFirstChunk(data))
    with pytest.raises(ConnectionResetError):
        fetcher.fetch_stage4_model(URL, target, manifest_path=manifest)
    assert not target.exists()
    assert not target.with_suffix(".joblib.tmp").exists()


@pytest.mark.parametrize("url", ["http://example.invalid/model", "file:///tmp/model", "model"])
def test_non_https_url_is_refused_before_network_or_filesystem(tmp_path, monkeypatch, url):
    target = tmp_path / "missing" / "rf.joblib"
    monkeypatch.setattr(fetcher, "urlopen", lambda *args, **kwargs: pytest.fail("network called"))
    with pytest.raises(ValueError, match="must use https"):
        fetcher.fetch_stage4_model(url, target)
    assert not target.parent.exists()


def test_existing_valid_file_performs_zero_network_calls(tmp_path, monkeypatch):
    data = b"already-present"
    target = tmp_path / "rf.joblib"
    target.write_bytes(data)
    manifest = _write_manifest(tmp_path, {target.name: data})
    monkeypatch.setattr(fetcher, "urlopen", lambda *args, **kwargs: pytest.fail("network called"))
    assert fetcher.fetch_stage4_model(URL, target, manifest_path=manifest) == target


def test_absent_manifest_fails_before_network(tmp_path, monkeypatch):
    target = tmp_path / "rf.joblib"
    monkeypatch.setattr(fetcher, "urlopen", lambda *args, **kwargs: pytest.fail("network called"))
    with pytest.raises(ArtifactIntegrityError, match="manifest not found"):
        fetcher.fetch_stage4_model(URL, target)


def test_absent_key_fails_before_network(tmp_path, monkeypatch):
    target = tmp_path / "rf.joblib"
    manifest = _write_manifest(tmp_path, {"another.joblib": b"model"})
    monkeypatch.setattr(fetcher, "urlopen", lambda *args, **kwargs: pytest.fail("network called"))
    with pytest.raises(ArtifactIntegrityError, match="No checksum entry"):
        fetcher.fetch_stage4_model(URL, target, manifest_path=manifest)


def test_artifact_outside_manifest_directory_fails_before_network(tmp_path, monkeypatch):
    manifest_root = tmp_path / "models"
    manifest_root.mkdir()
    manifest = _write_manifest(manifest_root, {"rf.joblib": b"model"})
    target = tmp_path / "outside" / "rf.joblib"
    monkeypatch.setattr(fetcher, "urlopen", lambda *args, **kwargs: pytest.fail("network called"))
    with pytest.raises(ArtifactIntegrityError, match="lies outside"):
        fetcher.fetch_stage4_model(URL, target, manifest_path=manifest)


def test_nested_model_uses_manifest_relative_key_not_basename(tmp_path, monkeypatch):
    root_data = b"root-model"
    nested_data = b"nested-model"
    target = tmp_path / "v5" / "rf.joblib"
    manifest = _write_manifest(
        tmp_path,
        {"rf.joblib": root_data, "v5/rf.joblib": nested_data},
    )
    _install_response(monkeypatch, nested_data, len(nested_data))
    fetcher.fetch_stage4_model(URL, target, manifest_path=manifest)
    assert target.read_bytes() == nested_data


def test_nested_model_cannot_borrow_root_basename_entry(tmp_path, monkeypatch):
    target = tmp_path / "v5" / "rf.joblib"
    manifest = _write_manifest(tmp_path, {"rf.joblib": b"root-model"})
    monkeypatch.setattr(fetcher, "urlopen", lambda *args, **kwargs: pytest.fail("network called"))
    with pytest.raises(ArtifactIntegrityError, match="expected key 'v5/rf.joblib'"):
        fetcher.fetch_stage4_model(URL, target, manifest_path=manifest)


MIRROR_URL = "https://mirror.example.invalid/rf_31feature_integrated.joblib"
PLAIN_URL = "http://mirror.example.invalid/rf_31feature_integrated.joblib"


class CannedTransport(HTTPSHandler, HTTPHandler):
    """Answers http and https requests from a table and never opens a socket.

    It replaces both default transports, so a redirect wrongly followed to http is
    answered here, with valid bytes, instead of reaching the network.
    """

    def __init__(self, routes: dict[str, tuple[int, str | None, bytes]]) -> None:
        super().__init__()
        self.routes = routes
        self.requested: list[str] = []

    def _answer(self, req: Request) -> addinfourl:
        self.requested.append(req.full_url)
        code, location, body = self.routes[req.full_url]
        headers = Message()
        if location is not None:
            headers["Location"] = location
        response = addinfourl(io.BytesIO(body), headers, req.full_url, code)
        response.msg = "canned"
        return response

    http_open = https_open = _answer


def _route_through(
    monkeypatch: pytest.MonkeyPatch, routes: dict[str, tuple[int, str | None, bytes]]
) -> CannedTransport:
    # The real opener and redirect handler stay in place; only the transport is
    # swapped, and ProxyHandler({}) keeps the host's proxy settings out of it.
    transport = CannedTransport(routes)
    opener = functools.partial(fetcher.build_https_opener, ProxyHandler({}), transport)
    monkeypatch.setattr(fetcher, "build_https_opener", opener)
    return transport


def test_https_to_http_redirect_is_refused(tmp_path, monkeypatch):
    """FAILS IF: a redirect to http is followed. The http URL serves the right
    bytes, so only the redirect check stops this download."""
    data = b"verified-model"
    target = tmp_path / "rf.joblib"
    manifest = _write_manifest(tmp_path, {target.name: data})
    transport = _route_through(
        monkeypatch, {URL: (302, PLAIN_URL, b""), PLAIN_URL: (200, None, data)}
    )
    with pytest.raises(HTTPError, match="non-https"):
        fetcher.fetch_stage4_model(URL, target, manifest_path=manifest)
    assert transport.requested == [URL]
    assert not target.exists()
    assert not target.with_suffix(".joblib.tmp").exists()


def test_https_to_https_redirect_is_followed(tmp_path, monkeypatch):
    data = b"verified-model"
    target = tmp_path / "rf.joblib"
    manifest = _write_manifest(tmp_path, {target.name: data})
    transport = _route_through(
        monkeypatch, {URL: (302, MIRROR_URL, b""), MIRROR_URL: (200, None, data)}
    )
    assert fetcher.fetch_stage4_model(URL, target, manifest_path=manifest) == target
    assert target.read_bytes() == data
    assert transport.requested == [URL, MIRROR_URL]


@pytest.mark.parametrize("newurl", [PLAIN_URL, "ftp://mirror.example.invalid/model"])
def test_redirect_handler_refuses_every_non_https_target(newurl):
    # urllib's own handler would follow both of these.
    handler = fetcher.HttpsOnlyRedirectHandler()
    with pytest.raises(HTTPError, match="non-https"):
        handler.redirect_request(Request(URL), io.BytesIO(), 302, "Found", Message(), newurl)
