from __future__ import annotations

import json

import pytest

import hashlib
from pathlib import Path

from src.artifact_integrity import (
    ArtifactIntegrityError,
    MODEL_CHECKSUM_MANIFEST,
    default_manifest_path_for,
    load_checksum_manifest,
    load_verified_joblib,
    sha256_file,
    update_checksum_manifest,
    verify_artifact_checksum,
)


def _write(path: Path, data: bytes = b"hello-sestrav") -> Path:
    path.write_bytes(data)
    return path


def test_update_and_verify_checksum_manifest(tmp_path):
    artifact = tmp_path / "rf_30feature_integrated.joblib"
    artifact.write_bytes(b"trusted-model-bytes")

    manifest = tmp_path / MODEL_CHECKSUM_MANIFEST
    update_checksum_manifest(manifest, [artifact])

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert "artifacts" in payload
    assert "rf_30feature_integrated.joblib" in payload["artifacts"]
    assert verify_artifact_checksum(artifact, manifest_path=manifest, required=True) is True


def test_verify_artifact_checksum_rejects_tampering(tmp_path):
    artifact = tmp_path / "ann_30feature_integrated.pt"
    artifact.write_bytes(b"checkpoint-v1")
    manifest = tmp_path / MODEL_CHECKSUM_MANIFEST
    update_checksum_manifest(manifest, [artifact])

    artifact.write_bytes(b"checkpoint-v2")
    with pytest.raises(ArtifactIntegrityError):
        verify_artifact_checksum(artifact, manifest_path=manifest, required=True)


def test_verify_artifact_checksum_optional_when_manifest_missing(tmp_path):
    artifact = tmp_path / "xgb_30feature_integrated.joblib"
    artifact.write_bytes(b"xgb-bytes")

    assert verify_artifact_checksum(artifact, required=False) is False
    with pytest.raises(ArtifactIntegrityError):
        verify_artifact_checksum(artifact, required=True)


def test_sha256_file_matches_hashlib(tmp_path):
    artifact = _write(tmp_path / "a.bin", b"abc123")
    assert sha256_file(artifact) == hashlib.sha256(b"abc123").hexdigest()


def test_sha256_file_handles_multichunk(tmp_path):
    # Larger than the 1 MiB read chunk to exercise the streaming loop.
    blob = b"x" * (1024 * 1024 + 17)
    artifact = _write(tmp_path / "big.bin", blob)
    assert sha256_file(artifact) == hashlib.sha256(blob).hexdigest()


def test_default_manifest_path_for(tmp_path):
    artifact = tmp_path / "models" / "m.joblib"
    assert default_manifest_path_for(artifact) == tmp_path / "models" / MODEL_CHECKSUM_MANIFEST


def test_default_manifest_path_custom_name(tmp_path):
    artifact = tmp_path / "m.joblib"
    assert default_manifest_path_for(artifact, "custom.json") == tmp_path / "custom.json"


def test_load_manifest_absent_returns_empty(tmp_path):
    assert load_checksum_manifest(tmp_path / "missing.json") == {
        "generated_utc": None,
        "artifacts": {},
    }


def test_load_manifest_valid(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"generated_utc": "t", "artifacts": {"a": {}}}), encoding="utf-8")
    assert load_checksum_manifest(path)["artifacts"] == {"a": {}}


def test_load_manifest_invalid_artifacts_type_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"artifacts": ["not", "a", "dict"]}), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        load_checksum_manifest(path)


def test_update_manifest_records_size_and_hash(tmp_path):
    artifact = _write(tmp_path / "a.bin", b"data")
    manifest_path = tmp_path / MODEL_CHECKSUM_MANIFEST
    assert update_checksum_manifest(manifest_path, [artifact]) == manifest_path
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["generated_utc"] is not None
    entry = payload["artifacts"]["a.bin"]
    assert entry["sha256"] == sha256_file(artifact)
    assert entry["size_bytes"] == len(b"data")


def test_update_manifest_skips_missing_files(tmp_path):
    manifest_path = tmp_path / MODEL_CHECKSUM_MANIFEST
    update_checksum_manifest(manifest_path, [tmp_path / "nope.bin"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["artifacts"] == {}


def test_update_manifest_upserts_existing(tmp_path):
    manifest_path = tmp_path / MODEL_CHECKSUM_MANIFEST
    update_checksum_manifest(manifest_path, [_write(tmp_path / "a.bin", b"one")])
    update_checksum_manifest(manifest_path, [_write(tmp_path / "b.bin", b"two")])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(payload["artifacts"]) == {"a.bin", "b.bin"}


def test_verify_missing_artifact_raises(tmp_path):
    with pytest.raises(ArtifactIntegrityError, match="not found"):
        verify_artifact_checksum(tmp_path / "ghost.bin")


def test_verify_entry_missing_optional_returns_false(tmp_path):
    artifact = _write(tmp_path / "a.bin")
    manifest_path = tmp_path / MODEL_CHECKSUM_MANIFEST
    update_checksum_manifest(manifest_path, [_write(tmp_path / "b.bin")])
    assert verify_artifact_checksum(artifact, manifest_path) is False


def test_verify_entry_missing_required_raises(tmp_path):
    artifact = _write(tmp_path / "a.bin")
    manifest_path = tmp_path / MODEL_CHECKSUM_MANIFEST
    update_checksum_manifest(manifest_path, [_write(tmp_path / "b.bin")])
    with pytest.raises(ArtifactIntegrityError, match="No checksum entry"):
        verify_artifact_checksum(artifact, manifest_path, required=True)


def test_verify_default_manifest_path(tmp_path):
    artifact = _write(tmp_path / "a.bin")
    update_checksum_manifest(default_manifest_path_for(artifact), [artifact])
    assert verify_artifact_checksum(artifact) is True


def test_load_verified_joblib_roundtrip(tmp_path):
    joblib = pytest.importorskip("joblib")
    obj = {"weights": [1, 2, 3]}
    artifact = tmp_path / "model.joblib"
    joblib.dump(obj, artifact)
    manifest_path = tmp_path / MODEL_CHECKSUM_MANIFEST
    update_checksum_manifest(manifest_path, [artifact])
    assert load_verified_joblib(artifact, manifest_path) == obj


def test_load_verified_joblib_mismatch_raises(tmp_path):
    joblib = pytest.importorskip("joblib")
    artifact = tmp_path / "model.joblib"
    joblib.dump({"a": 1}, artifact)
    manifest_path = tmp_path / MODEL_CHECKSUM_MANIFEST
    update_checksum_manifest(manifest_path, [artifact])
    joblib.dump({"a": 2}, artifact)  # tamper after manifest written
    with pytest.raises(ArtifactIntegrityError):
        load_verified_joblib(artifact, manifest_path, required_checksum=True)
