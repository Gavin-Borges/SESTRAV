"""Tests for scripts/fetch_verified_mhcflurry.py.

The URL guard is the reason the script's single urlopen call may carry a
scoped Bandit B310 suppression, so these tests prove the guard refuses every
non-https or off-host URL BEFORE any network or filesystem access. No test
here opens a socket: urlopen is replaced with a tripwire, and socket connects
are refused outright, so a guard regression fails loudly instead of quietly
reaching the network.

The download tests pin a second property: the archive is hashed while it is
still the ``.part`` file, and only a matching archive is renamed to its final
name. Step two (``mhcflurry-downloads fetch --already-downloaded-dir``) reads
the final name from that directory, so an archive that failed its digest must
never appear there, and neither may a leftover ``.part``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import socket
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "fetch_verified_mhcflurry.py"
CONFIG = REPO_ROOT / "config.yaml"


def _load_module():
    spec = importlib.util.spec_from_file_location("fetch_verified_mhcflurry", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fetcher = _load_module()


@pytest.fixture
def urlopen_calls(monkeypatch):
    """Replace urlopen with a tripwire and refuse every socket connect."""
    calls: list[object] = []

    def _tripwire(*args, **kwargs):
        calls.append(args)
        raise AssertionError("urlopen was reached; the URL guard did not refuse first")

    def _no_connect(self, *args, **kwargs):
        raise AssertionError("a socket connect was attempted during a no-network test")

    monkeypatch.setattr(fetcher, "urlopen", _tripwire)
    monkeypatch.setattr(socket.socket, "connect", _no_connect)
    return calls


REFUSED_URLS = [
    # Scheme is not https.
    "http://github.com/openvax/mhcflurry/releases/download/x/models.tar.bz2",
    "file:///etc/passwd",
    "ftp://github.com/models.tar.bz2",
    "data:text/plain,hello",
    "github.com/openvax/mhcflurry/models.tar.bz2",
    "//github.com/openvax/mhcflurry/models.tar.bz2",
    # https, but the host is not on the allowlist.
    "https://example.org/models.tar.bz2",
    "https://github.com.attacker.example/models.tar.bz2",
    "https://github.com@attacker.example/models.tar.bz2",
    "https://objects.githubusercontent.com/models.tar.bz2",
    # https on the allowed host, but no filename to write.
    "https://github.com/",
]


@pytest.mark.parametrize("url", REFUSED_URLS)
def test_download_refuses_before_any_network_or_disk_access(url, tmp_path, urlopen_calls):
    output_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="MHCflurry archive URL"):
        fetcher.download_archive(url, output_dir, "0" * 64)
    assert urlopen_calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("url", "fragment"),
    [
        ("http://github.com/models.tar.bz2", "must use https, got scheme 'http'"),
        ("file:///etc/passwd", "must use https, got scheme 'file'"),
        ("https://example.org/models.tar.bz2", "host 'example.org' is not in the allowlist"),
    ],
)
def test_refusal_names_the_offending_part(url, fragment, urlopen_calls):
    with pytest.raises(ValueError) as excinfo:
        fetcher.validate_archive_url(url)
    assert fragment in str(excinfo.value)
    assert urlopen_calls == []


def test_the_pinned_config_url_passes_the_guard():
    """The live config.yaml URL must satisfy the guard, or the image build fails."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    url = config["mhcflurry_model_archive_url"]
    filename = fetcher.validate_archive_url(url)
    assert filename.endswith(".tar.bz2")
    assert filename == Path(url).name


PAYLOAD = b"synthetic archive bytes, not a real model"
GOOD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()
ALLOWED_URL = "https://github.com/openvax/mhcflurry/releases/download/x/models.tar.bz2"


@pytest.fixture
def fake_download(monkeypatch):
    """Serve PAYLOAD from an in-memory urlopen and refuse every socket connect.

    Returns the list of (url, kwargs) pairs urlopen was called with.
    """
    requested: list[tuple[str, dict]] = []

    def _fake_urlopen(url, *args, **kwargs):
        requested.append((url, kwargs))
        return io.BytesIO(PAYLOAD)

    def _no_connect(self, *args, **kwargs):
        raise AssertionError("a socket connect was attempted during a no-network test")

    monkeypatch.setattr(fetcher, "urlopen", _fake_urlopen)
    monkeypatch.setattr(socket.socket, "connect", _no_connect)
    return requested


def test_allowed_url_downloads_and_verifies_without_network(tmp_path, fake_download):
    """An https URL on the allowed host reaches urlopen, and a matching archive lands."""
    archive = fetcher.download_archive(ALLOWED_URL, tmp_path / "out", GOOD_SHA256)

    assert [url for url, _ in fake_download] == [ALLOWED_URL]
    assert archive == tmp_path / "out" / "models.tar.bz2"
    assert archive.read_bytes() == PAYLOAD
    assert not archive.with_suffix(archive.suffix + ".part").exists()


def test_a_mismatched_download_never_reaches_its_final_name(tmp_path, fake_download):
    """The digest is checked on the .part file, so a bad archive is never renamed.

    Before this was fixed the .part was renamed first and hashed second, so a
    mismatch raised with the unverified archive already under the name step two
    reads.
    """
    output_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetcher.download_archive(ALLOWED_URL, output_dir, "0" * 64)

    assert not (output_dir / "models.tar.bz2").exists()
    assert not (output_dir / "models.tar.bz2.part").exists()
    assert list(output_dir.iterdir()) == []


def test_urlopen_is_given_a_finite_timeout(tmp_path, fake_download):
    """Without a timeout a stalled server hangs the image build indefinitely."""
    fetcher.download_archive(ALLOWED_URL, tmp_path / "out", GOOD_SHA256)

    (_, kwargs), = fake_download
    timeout = kwargs.get("timeout")
    assert isinstance(timeout, (int, float)) and 0 < timeout <= 600, kwargs


def test_cli_url_mode_leaves_nothing_behind_on_a_mismatch(tmp_path, fake_download, monkeypatch):
    """The command singularity.def and the CLI hint run: a mismatch exits non-zero
    and leaves the output directory empty."""
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_verified_mhcflurry.py", "--url", ALLOWED_URL, "--sha256", "0" * 64,
         "--output-dir", str(output_dir)],
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetcher.main()
    assert list(output_dir.iterdir()) == []


def test_cli_url_mode_accepts_an_uppercase_digest(tmp_path, fake_download, monkeypatch):
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_verified_mhcflurry.py", "--url", ALLOWED_URL, "--sha256", GOOD_SHA256.upper(),
         "--output-dir", str(output_dir)],
    )
    assert fetcher.main() == 0
    assert (output_dir / "models.tar.bz2").read_bytes() == PAYLOAD


def test_verify_archive_reports_a_mismatch(tmp_path):
    archive = tmp_path / "models.tar.bz2"
    archive.write_bytes(PAYLOAD)
    fetcher.verify_archive(archive, GOOD_SHA256)
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetcher.verify_archive(archive, "0" * 64)
