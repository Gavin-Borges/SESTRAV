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


def test_validate_signature_rejects_a_torch_checkpoint_with_no_manifest(tmp_path):
    """A .pt artifact is not introspected, but it IS checksum-verified.

    This test previously asserted True here, on the reasoning that a non-joblib
    artifact is simply "not introspected". That conflated two separate halves of
    the contract. The FEATURE comparison is genuinely skipped for a torch
    checkpoint, which exposes no n_features_in_. The CHECKSUM requirement is not
    suffix-specific, and skipping it made a method named validate_signature
    return True for bytes it had never verified.
    """
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"not really a torch file")
    assert _registry(tmp_path).validate_signature(model_path, expected_features=30) is False


def test_validate_signature_rejects_an_artifact_that_does_not_exist(tmp_path):
    """The sharpest form of the fail-open: nothing on disk at all.

    `validate_signature(Path("does/not/exist.pt"), 31)` returned True, so a
    caller asking whether an artifact was trustworthy got a yes for a file that
    was not there.
    """
    missing = tmp_path / "absent.pt"
    assert not missing.exists()
    assert _registry(tmp_path).validate_signature(missing, expected_features=30) is False


def test_validate_signature_accepts_a_torch_checkpoint_with_a_matching_checksum(tmp_path):
    """Fail-closed must not mean fail-always: a verified .pt is still valid."""
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"torch-checkpoint-bytes")
    update_checksum_manifest(default_manifest_path_for(model_path), [model_path])
    assert _registry(tmp_path).validate_signature(model_path, expected_features=30) is True


def test_validate_signature_rejects_a_tampered_torch_checkpoint(tmp_path):
    """The manifest entry must be checked against the bytes, not merely present."""
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"torch-checkpoint-bytes")
    update_checksum_manifest(default_manifest_path_for(model_path), [model_path])
    model_path.write_bytes(b"tampered-checkpoint-bytes")
    assert _registry(tmp_path).validate_signature(model_path, expected_features=30) is False


def test_validate_signature_rejects_an_extension_load_would_refuse(tmp_path):
    """The two halves of the registry must agree about the same path.

    load() raises ValueError("Unsupported model extension") for .bin, so
    reporting a valid signature for it put the pair in direct contradiction.
    """
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"data")
    update_checksum_manifest(default_manifest_path_for(model_path), [model_path])
    assert _registry(tmp_path).validate_signature(model_path, expected_features=30) is False


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


def test_load_torch_normalizes_numpy_internal_rename(tmp_path, monkeypatch):
    """A numpy rename breaks the np._core getattr chain; it must still normalize.

    The artifact is a VALID checkpoint so the only thing that can fail is the
    getattr chain: without the delattr this test loads cleanly and raises nothing.
    """
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    registry = _registry(tmp_path)
    artifact = tmp_path / "model.pt"
    torch.save({"w": torch.tensor([1.0])}, artifact)
    monkeypatch.setattr(registry, "resolve_model", lambda name: artifact)
    monkeypatch.delattr(numpy, "_core")
    with pytest.raises(RuntimeError, match="Failed to load torch model") as excinfo:
        registry.load("model.pt")
    assert isinstance(excinfo.value.__cause__, AttributeError)


def test_validate_signature_lets_an_unexpected_error_propagate(tmp_path, monkeypatch):
    """A bug must surface, not be reported as an invalid signature.

    validate_signature's contract is that a VERIFICATION failure returns False.
    Under the previous bare `except Exception` any error at all became False, so
    a coding error in the load path was indistinguishable from a tampered or
    corrupt artifact: both produced a quiet "invalid model" with no traceback.

    The narrowed tuple keeps every genuine artifact-failure mode returning False
    (the tests above pin those) while letting anything outside it propagate.
    RecursionError stands in for "a class of failure nobody anticipated"; it is
    not in the tuple and is not a subclass of anything in it.
    """
    model_path = tmp_path / "model.joblib"
    joblib.dump(_FakeEstimator(30), model_path)

    def _boom(*_args, **_kwargs):
        raise RecursionError("unanticipated failure inside the load path")

    monkeypatch.setattr("src.core.model_registry.load_verified_joblib", _boom)

    with pytest.raises(RecursionError):
        _registry(tmp_path).validate_signature(model_path, expected_features=30)
