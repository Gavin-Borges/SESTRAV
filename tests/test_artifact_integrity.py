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


def _write_manifest(manifest_path: Path, entries: dict) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"generated_utc": "2026-01-01T00:00:00Z", "artifacts": entries}),
        encoding="utf-8",
    )
    return manifest_path


def _entry(path: Path) -> dict:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _dual_manifest_layout(tmp_path: Path, name: str = "rf_31feature_integrated.joblib"):
    """Mirror the real repo: models/<name> and models/v5/<name>, each with its own
    sibling manifest, same basename, different bytes."""
    root = tmp_path / "models"
    v5 = root / "v5"
    v5.mkdir(parents=True)

    root_artifact = _write(root / name, b"root-model-bytes")
    v5_artifact = _write(v5 / name, b"v5-model-bytes-different")

    root_manifest = _write_manifest(
        root / MODEL_CHECKSUM_MANIFEST, {name: _entry(root_artifact)}
    )
    v5_manifest = _write_manifest(v5 / MODEL_CHECKSUM_MANIFEST, {name: _entry(v5_artifact)})
    return root_artifact, root_manifest, v5_artifact, v5_manifest


def test_colliding_basenames_verify_against_their_own_manifests(tmp_path):
    root_artifact, root_manifest, v5_artifact, v5_manifest = _dual_manifest_layout(tmp_path)

    assert verify_artifact_checksum(root_artifact, root_manifest, required=True) is True
    assert verify_artifact_checksum(v5_artifact, v5_manifest, required=True) is True


def test_v5_artifact_not_matched_by_root_manifest_basename(tmp_path):
    _, root_manifest, v5_artifact, _ = _dual_manifest_layout(tmp_path)

    assert verify_artifact_checksum(v5_artifact, root_manifest) is False
    with pytest.raises(ArtifactIntegrityError, match="No checksum entry"):
        verify_artifact_checksum(v5_artifact, root_manifest, required=True)


def test_root_artifact_not_matched_by_v5_manifest_basename(tmp_path):
    root_artifact, _, _, v5_manifest = _dual_manifest_layout(tmp_path)

    assert verify_artifact_checksum(root_artifact, v5_manifest) is False
    with pytest.raises(ArtifactIntegrityError, match="lies outside the directory"):
        verify_artifact_checksum(root_artifact, v5_manifest, required=True)


def test_bare_name_entry_never_matches_out_of_tree_artifact(tmp_path):
    """Even when the bare-name entry's digest happens to match the artifact bytes,
    an entry that does not describe this path must not verify it."""
    subdir_a = tmp_path / "a"
    subdir_a.mkdir()
    artifact = _write(subdir_a / "model.bin", b"content")

    manifest_path = _write_manifest(
        tmp_path / "b" / MODEL_CHECKSUM_MANIFEST, {artifact.name: _entry(artifact)}
    )

    assert verify_artifact_checksum(artifact, manifest_path) is False
    with pytest.raises(ArtifactIntegrityError, match="lies outside the directory"):
        verify_artifact_checksum(artifact, manifest_path, required=True)


def test_nested_relative_key_still_verifies(tmp_path):
    """update_checksum_manifest writes 'v5/<name>' for a subdirectory artifact
    (as promote_gnn does for models/gnn/); that key must still resolve."""
    root = tmp_path / "models"
    v5 = root / "v5"
    v5.mkdir(parents=True)
    artifact = _write(v5 / "rf_31feature_integrated.joblib", b"v5-bytes")

    manifest_path = root / MODEL_CHECKSUM_MANIFEST
    update_checksum_manifest(manifest_path, [artifact])

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "v5/rf_31feature_integrated.joblib" in payload["artifacts"]
    assert verify_artifact_checksum(artifact, manifest_path, required=True) is True


def test_nested_key_does_not_leak_to_sibling_basename(tmp_path):
    """A manifest holding only the nested key must not verify a same-named file
    sitting directly beside the manifest."""
    root = tmp_path / "models"
    v5 = root / "v5"
    v5.mkdir(parents=True)
    nested = _write(v5 / "model.joblib", b"nested-bytes")
    sibling = _write(root / "model.joblib", b"sibling-bytes")

    manifest_path = root / MODEL_CHECKSUM_MANIFEST
    update_checksum_manifest(manifest_path, [nested])

    assert verify_artifact_checksum(sibling, manifest_path) is False
    with pytest.raises(ArtifactIntegrityError, match="No checksum entry"):
        verify_artifact_checksum(sibling, manifest_path, required=True)
