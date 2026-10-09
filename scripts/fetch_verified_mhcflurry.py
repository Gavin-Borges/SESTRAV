"""Download or verify the pinned MHCflurry presentation-model archive."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

# The only host the pinned archive is fetched from. Any other host is refused
# before a socket is opened, so pointing config.yaml at a mirror is a reviewed
# edit to this constant rather than a silent one. GitHub answers with a redirect
# to its asset CDN, which urllib follows; the sha256 pin, not this allowlist, is
# what vouches for the bytes that arrive.
ALLOWED_ARCHIVE_HOSTS = frozenset({"github.com"})

# Seconds urlopen may wait on the connect and on each socket read. It bounds a
# stalled server, not the total transfer time of the archive.
DOWNLOAD_TIMEOUT_SECONDS = 60


def validate_archive_url(url: str) -> str:
    """Refuse any archive URL that is not HTTPS on an allowed host.

    Returns the archive filename taken from the URL path. This runs before any
    network or filesystem access, so a file://, http://, ftp:// or custom-scheme
    URL, or an off-host one, fails here with a clear error and never reaches
    urlopen.
    """
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        raise ValueError(
            f"MHCflurry archive URL must use https, got scheme {parsed.scheme!r}"
        )
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_ARCHIVE_HOSTS:
        raise ValueError(
            f"MHCflurry archive URL host {host!r} is not in the allowlist "
            f"{sorted(ALLOWED_ARCHIVE_HOSTS)}"
        )
    filename = Path(parsed.path).name
    if not filename:
        raise ValueError("MHCflurry archive URL has no filename")
    return filename


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"MHCflurry archive checksum mismatch: expected {expected_sha256}, "
            f"got {actual}"
        )
    print(f"Verified MHCflurry archive sha256: {actual}")


def download_archive(url: str, output_dir: Path, expected_sha256: str) -> Path:
    """Download to a .part file, verify it, and only then give it its final name.

    Step two (`mhcflurry-downloads fetch --already-downloaded-dir`) reads the
    final name from output_dir, so an archive that fails its digest must never
    appear under it. Any failure, including a mismatch, removes the .part file.
    """
    filename = validate_archive_url(url)

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        # The B310 suppression on the next line rests on validate_archive_url()
        # above, which has already refused every scheme but https and every host
        # outside ALLOWED_ARCHIVE_HOSTS. Bandit cannot follow that call; the
        # guard is what makes the suppression honest, not a claim that urlopen
        # is safe in general.
        with urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:  # nosec B310
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output)
        verify_archive(partial, expected_sha256)
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url")
    source.add_argument("--archive", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.url and args.output_dir is None:
        parser.error("--output-dir is required with --url")
    return args


def main() -> int:
    args = parse_args()
    expected_sha256 = args.sha256.lower()
    if args.url:
        download_archive(args.url, args.output_dir, expected_sha256)
    else:
        verify_archive(args.archive, expected_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
