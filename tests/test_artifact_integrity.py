from __future__ import annotations

import json

import pytest

from src.artifact_integrity import (
    ArtifactIntegrityError,
    MODEL_CHECKSUM_MANIFEST,
    update_checksum_manifest,
    verify_artifact_checksum,
)


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
