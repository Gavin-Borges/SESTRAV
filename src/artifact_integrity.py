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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
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


def default_manifest_path_for(
    path: str | Path, manifest_name: str = MODEL_CHECKSUM_MANIFEST
) -> Path:
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

    # An artifact is only ever matched against its canonical manifest-relative key.
    # Basenames are not unique across the repo (models/, models/v5/ and
    # models/allele_aware/ all carry files with identical names but different
    # contents), so any bare-filename fallback would let one artifact be checked
    # against another artifact's digest. There is no safe fallback: fail closed.
    key: str | None
    try:
        key = _manifest_key(artifact, manifest)
    except ValueError:
        key = None

    entry = artifacts.get(key) if key is not None else None
    if entry is None:
        if required:
            if key is None:
                raise ArtifactIntegrityError(
                    f"Artifact '{artifact}' lies outside the directory of manifest "
                    f"'{manifest}', so no entry in it can describe this artifact."
                )
            raise ArtifactIntegrityError(
                f"No checksum entry found for '{artifact}' in manifest '{manifest}' "
                f"(expected key '{key}')."
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


def _relative_to_project_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def write_provenance_sidecar(
    output_path: str | Path,
    *,
    script: str,
    extra: dict[str, object] | None = None,
) -> Path:
    """Write a `<output>.provenance.json` sidecar recording the output's own sha256.

    Mirrors the pattern proven in `scripts/assess_calibration.py` (the one
    sidecar that already PASSes `_local/integrity/integrity_check.py`'s
    provenance check): `artifact` + `sha256` are the two fields that check
    resolves and verifies, and the file is written with `newline=""` so the
    LF that `json.dumps` produces is not rewritten to CRLF on Windows - the
    hash recorded must match the hash git stores under the `results/*.provenance.json`
    `eol=lf` pin in `.gitattributes`, or the check fails on a byte-identical file.

    `extra` is for anything the artifact's reproducibility depends on but that
    isn't captured by the artifact's own bytes - for a script that scores an
    untracked, gitignored model file, that means the model's own path and
    sha256, recorded here specifically because the model can be silently
    overwritten in place after the benchmark ran (see the TSNAdb 0.99
    incident, D-series, 2026-08-12: the model that produced it was overwritten
    with no checksum captured, making the figure permanently unreproducible).
    Pass `{"model_path": ..., "model_sha256": ...}` for that case.
    """
    output_path = Path(output_path)
    payload: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": script,
        "artifact": _relative_to_project_root(output_path),
        "sha256": sha256_file(output_path),
    }
    if extra:
        payload.update(extra)
    sidecar_path = output_path.with_suffix(output_path.suffix + ".provenance.json")
    with sidecar_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(payload, indent=2) + "\n")
    return sidecar_path


def model_provenance_fields(model_path: str | Path) -> dict[str, object]:
    """Return `{model_path, model_sha256}` for embedding in a benchmark's own
    provenance sidecar via `write_provenance_sidecar`'s `extra` argument.

    `model_sha256` is `None` when the model file is not present locally (it is
    gitignored in this repo) rather than raising, matching how
    `check_provenance` treats a missing referenced artifact as a benign SKIP,
    not a FAIL.
    """
    model_path = Path(model_path)
    return {
        "model_path": _relative_to_project_root(model_path),
        "model_sha256": sha256_file(model_path) if model_path.is_file() else None,
    }
