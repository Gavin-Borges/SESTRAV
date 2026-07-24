import hashlib

import joblib
import pytest
from src.artifact_integrity import default_manifest_path_for, update_checksum_manifest
from src.core.config import SestravConfig
from src.core.model_registry import ModelRegistry


def test_model_registry_resolution(tmp_path):
    # Mock a config
    config = SestravConfig.model_construct(output_dir=tmp_path)
    registry = ModelRegistry(config)

    # Should resolve to the default models dir
    resolved = registry.resolve_model("some_model.joblib")
    assert resolved.name == "some_model.joblib"
    assert "models" in resolved.parts


def test_model_registry_missing_model(tmp_path):
    config = SestravConfig.model_construct(output_dir=tmp_path)
    registry = ModelRegistry(config)

    with pytest.raises(FileNotFoundError):
        registry.load("nonexistent_model.joblib")


def _registry(tmp_path):
    return ModelRegistry(SestravConfig.model_construct(output_dir=tmp_path))


def test_artifact_checksum_matches_hashlib(tmp_path):
    artifact = tmp_path / "artifact.bin"
    payload = b"sestrav-model-bytes" * 1000  # exceed the 4096-byte read chunk
    artifact.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert _registry(tmp_path).artifact_checksum(artifact) == expected


class _FakeEstimator:
    def __init__(self, n_features_in_):
        self.n_features_in_ = n_features_in_


def test_validate_signature_accepts_matching_feature_count(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump(_FakeEstimator(30), model_path)
    update_checksum_manifest(default_manifest_path_for(model_path), [model_path])
    assert _registry(tmp_path).validate_signature(model_path, expected_features=30) is True


def test_validate_signature_rejects_mismatched_feature_count(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump(_FakeEstimator(21), model_path)
    update_checksum_manifest(default_manifest_path_for(model_path), [model_path])
    assert _registry(tmp_path).validate_signature(model_path, expected_features=30) is False


def test_validate_signature_ignores_non_joblib_artifacts(tmp_path):
    # Non-.joblib artifacts are not introspected and are treated as valid.
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"not really a torch file")
    assert _registry(tmp_path).validate_signature(model_path, expected_features=30) is True


def test_load_rejects_unsupported_extension(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"data")
    # Bypass the models/ resolution so we exercise the extension dispatch directly.
    monkeypatch.setattr(registry, "resolve_model", lambda name: artifact)
    with pytest.raises(ValueError, match="Unsupported model extension"):
        registry.load("model.bin")


def test_load_joblib_roundtrip(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    artifact = tmp_path / "model.joblib"
    joblib.dump({"params": [1, 2, 3]}, artifact)
    update_checksum_manifest(default_manifest_path_for(artifact), [artifact])
    monkeypatch.setattr(registry, "resolve_model", lambda name: artifact)
    assert registry.load("model.joblib") == {"params": [1, 2, 3]}


def test_validate_signature_handles_unreadable_joblib(tmp_path):
    # A .joblib that can't be deserialized hits the except branch -> fail-closed (invalid).
    model_path = tmp_path / "corrupt.joblib"
    model_path.write_bytes(b"not a real joblib payload")
    assert _registry(tmp_path).validate_signature(model_path, expected_features=30) is False


def test_validate_signature_rejects_tampered_checksum(tmp_path):
    # Model bytes no longer match the recorded sha256 -> mismatch -> fail-closed (invalid).
    model_path = tmp_path / "model.joblib"
    joblib.dump(_FakeEstimator(30), model_path)
    update_checksum_manifest(default_manifest_path_for(model_path), [model_path])
    model_path.write_bytes(b"tampered bytes that do not match the recorded checksum")
    assert _registry(tmp_path).validate_signature(model_path, expected_features=30) is False


def test_validate_signature_model_without_feature_attr(tmp_path):
    # Estimator lacking n_features_in_ -> getattr returns None -> treated valid.
    model_path = tmp_path / "model.joblib"
    joblib.dump(object(), model_path)
    update_checksum_manifest(default_manifest_path_for(model_path), [model_path])
    assert _registry(tmp_path).validate_signature(model_path, expected_features=30) is True


def test_load_torch_roundtrip(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    registry = _registry(tmp_path)
    artifact = tmp_path / "model.pt"
    torch.save({"w": torch.zeros(3)}, artifact)
    monkeypatch.setattr(registry, "resolve_model", lambda name: artifact)
    loaded = registry.load("model.pt")
    assert "w" in loaded
    assert torch.equal(loaded["w"], torch.zeros(3))


def test_load_torch_corrupt_raises_runtime_error(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    registry = _registry(tmp_path)
    artifact = tmp_path / "model.pth"
    artifact.write_bytes(b"not a real torch checkpoint")
    monkeypatch.setattr(registry, "resolve_model", lambda name: artifact)
    with pytest.raises(RuntimeError, match="Failed to load torch model"):
        registry.load("model.pth")
