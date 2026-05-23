"""
Helpers for artifact checksum verification and manifest maintenance.

SESTRAV model files are generated locally and intentionally excluded from git.
That means the practical hardening path is to generate sidecar checksums when
artifacts are created, and verify them again before sensitive loads.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


MODEL_CHECKSUM_MANIFEST = "model_artifact_checksums.json"


class ArtifactIntegrityError(RuntimeError):
    """Raised when a checksum manifest is missing or an artifact mismatches it."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_manifest_path_for(path: str | Path, manifest_name: str = MODEL_CHECKSUM_MANIFEST) -> Path:
    """Return the default checksum manifest path for an artifact."""
    artifact = Path(path)
    return artifact.parent / manifest_name


def _manifest_key(path: Path, manifest_path: Path) -> str:
    return str(path.resolve().relative_to(manifest_path.parent.resolve())).replace("\\", "/")


def load_checksum_manifest(manifest_path: str | Path) -> dict:
    """Load a checksum manifest, returning an empty structure if it is absent."""
    manifest = Path(manifest_path)
    if not manifest.is_file():
        return {"generated_utc": None, "artifacts": {}}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ArtifactIntegrityError(f"Checksum manifest has invalid format: {manifest}")
    return payload


def update_checksum_manifest(
    manifest_path: str | Path,
    artifact_paths: Iterable[str | Path],
) -> Path:
    """Upsert checksum rows for the provided artifacts into a manifest."""
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = load_checksum_manifest(manifest)
    artifacts = dict(payload.get("artifacts", {}))

    for artifact_path in artifact_paths:
        artifact = Path(artifact_path)
        if not artifact.is_file():
            continue
        key = _manifest_key(artifact, manifest)
        artifacts[key] = {
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        }

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_artifact_checksum(
    path: str | Path,
    manifest_path: str | Path | None = None,
    required: bool = False,
) -> bool:
    """Verify an artifact against a checksum manifest.

    Returns True when verification was performed and passed.
    Returns False when no applicable manifest entry exists and verification was optional.
    Raises ArtifactIntegrityError on mismatch or when verification is required but unavailable.
    """
    artifact = Path(path)
    if not artifact.is_file():
        raise ArtifactIntegrityError(f"Artifact not found for checksum verification: {artifact}")

    manifest = Path(manifest_path) if manifest_path else default_manifest_path_for(artifact)
    payload = load_checksum_manifest(manifest)
    artifacts = payload.get("artifacts", {})
    if not artifacts:
        if required:
            raise ArtifactIntegrityError(
                f"Checksum manifest required for '{artifact}', but '{manifest}' was not found or is empty."
            )
        return False

    keys = []
    try:
        keys.append(_manifest_key(artifact, manifest))
    except ValueError:
        pass
    keys.append(artifact.name)
    entry = next((artifacts[key] for key in keys if key in artifacts), None)
    if entry is None:
        if required:
            raise ArtifactIntegrityError(
                f"No checksum entry found for '{artifact}' in manifest '{manifest}'."
            )
        return False

    expected = entry.get("sha256")
    actual = sha256_file(artifact)
    if expected != actual:
        raise ArtifactIntegrityError(
            f"Checksum verification failed for '{artifact}'. Expected {expected}, got {actual}."
        )
    return True


def load_verified_joblib(
    path: str | Path,
    manifest_path: str | Path | None = None,
    required_checksum: bool = False,
):
    """Verify a joblib artifact when possible, then load it."""
    from joblib import load as joblib_load

    verify_artifact_checksum(path, manifest_path=manifest_path, required=required_checksum)
    return joblib_load(path)
