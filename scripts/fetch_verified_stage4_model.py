"""Fetch the Stage-4 RF model and verify it against its checksum manifest."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from http.client import HTTPMessage
from pathlib import Path
from typing import IO, Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import (
    BaseHandler,
    HTTPRedirectHandler,
    OpenerDirector,
    Request,
    build_opener,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_integrity import (  # noqa: E402
    ArtifactIntegrityError,
    default_manifest_path_for,
    load_checksum_manifest,
    sha256_file,
    verify_artifact_checksum,
)

MODEL_URL_ENV = "SESTRAV_STAGE4_MODEL_URL"
DEFAULT_DESTINATION = PROJECT_ROOT / "models" / "rf_31feature_integrated.joblib"
CHUNK_SIZE = 1024 * 1024


def _validate_url(url: str) -> None:
    """Reject a non-HTTPS starting URL before filesystem or network access.

    Redirect targets are checked separately, by HttpsOnlyRedirectHandler.
    """
    scheme = urlsplit(url).scheme.lower()
    if scheme != "https":
        raise ValueError(f"Stage-4 model URL must use https, got scheme {scheme!r}")


class HttpsOnlyRedirectHandler(HTTPRedirectHandler):
    """Follow a redirect only when its target URL is https.

    urllib's default HTTPRedirectHandler also follows redirects to http and ftp,
    so checking the starting URL alone would not keep the transfer on https.
    """

    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        scheme = urlsplit(newurl).scheme.lower()
        if scheme != "https":
            raise HTTPError(
                newurl,
                code,
                f"Refusing redirect to a non-https URL (scheme {scheme!r})",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_https_opener(*handlers: BaseHandler) -> OpenerDirector:
    """Return an opener whose only redirect handler is HttpsOnlyRedirectHandler."""
    return build_opener(HttpsOnlyRedirectHandler, *handlers)


def urlopen(url: str, *, timeout: float) -> Any:
    """Open ``url``, following a redirect only when its target is https."""
    return build_https_opener().open(url, timeout=timeout)


def _manifest_entry(destination: Path, manifest: Path) -> tuple[str, int]:
    """Resolve exactly one manifest-relative artifact entry, without basename fallback."""
    if not manifest.is_file():
        raise ArtifactIntegrityError(f"Checksum manifest not found: {manifest}")
    try:
        key = destination.resolve().relative_to(manifest.parent.resolve()).as_posix()
    except ValueError as exc:
        raise ArtifactIntegrityError(
            f"Artifact '{destination}' lies outside the directory of manifest '{manifest}'."
        ) from exc

    entry = load_checksum_manifest(manifest).get("artifacts", {}).get(key)
    if not isinstance(entry, Mapping):
        raise ArtifactIntegrityError(
            f"No checksum entry found for '{destination}' in manifest '{manifest}' "
            f"(expected key '{key}')."
        )
    expected_sha256 = entry.get("sha256")
    expected_size = entry.get("size_bytes")
    if not isinstance(expected_sha256, str) or not isinstance(expected_size, int):
        raise ArtifactIntegrityError(f"Invalid checksum entry for '{key}' in '{manifest}'.")
    return expected_sha256, expected_size


def _verify_size(path: Path, expected_size: int) -> None:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ArtifactIntegrityError(
            f"Size verification failed for '{path}'. Expected {expected_size}, got {actual_size}."
        )


def fetch_stage4_model(
    url: str,
    destination: str | Path = DEFAULT_DESTINATION,
    *,
    manifest_path: str | Path | None = None,
    timeout: float = 60.0,
) -> Path:
    """Fetch, verify, and atomically install the Stage-4 model.

    An existing valid destination is returned without opening a network connection.
    The manifest-relative key is resolved before download so same-basename files in
    nested model directories cannot borrow the root model's checksum.
    """
    _validate_url(url)
    target = Path(destination)
    manifest = (
        Path(manifest_path) if manifest_path is not None else default_manifest_path_for(target)
    )
    expected_sha256, expected_size = _manifest_entry(target, manifest)

    if target.is_file():
        _verify_size(target, expected_size)
        verify_artifact_checksum(target, manifest_path=manifest, required=True)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".tmp")
    partial.unlink(missing_ok=True)
    try:
        # HTTPS is enforced on the starting URL by _validate_url and on every
        # redirect by HttpsOnlyRedirectHandler. The manifest size and digest, not
        # the transport, vouch for the bytes received.
        with urlopen(url, timeout=timeout) as response, partial.open("wb") as output:
            declared_length = response.headers.get("Content-Length")
            received = 0
            while chunk := response.read(CHUNK_SIZE):
                received += len(chunk)
                # Stop at the manifest size rather than after the stream ends, so a
                # wrong or hostile URL cannot fill the disk before the size check.
                if received > expected_size:
                    raise ArtifactIntegrityError(
                        f"Download exceeds the manifest size of {expected_size} bytes; "
                        "refusing to write past it."
                    )
                output.write(chunk)
        if declared_length is not None and received != int(declared_length):
            raise ArtifactIntegrityError(
                f"Truncated download: expected {declared_length} bytes from HTTP, got {received}."
            )
        _verify_size(partial, expected_size)
        actual_sha256 = sha256_file(partial)
        if actual_sha256 != expected_sha256:
            raise ArtifactIntegrityError(
                f"Checksum verification failed for '{target}'. "
                f"Expected {expected_sha256}, got {actual_sha256}."
            )
        partial.replace(target)
        verify_artifact_checksum(target, manifest_path=manifest, required=True)
    except BaseException:
        partial.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get(MODEL_URL_ENV))
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if not args.url:
        parser.error(f"--url or {MODEL_URL_ENV} is required")
    path = fetch_stage4_model(args.url, args.output, manifest_path=args.manifest)
    print(f"Verified Stage-4 model: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
