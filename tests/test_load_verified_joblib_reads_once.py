"""`load_verified_joblib` must deserialize exactly the bytes it verified. It used
to hash the artifact by path and then reopen the path to unpickle it, so a file
replaced between the two opens was loaded without being checked."""

import builtins
import inspect
import io
import json
import os
from pathlib import Path

import joblib
import pytest

import src.artifact_integrity as ai

GOOD = {"who": "good", "pad": "x" * 64}
EVIL = {"who": "EVIL", "pad": "y" * 64}


def _fixture(tmp_path: Path, obj=GOOD, **dump_kw):
    artifact = tmp_path / "rf_31feature_integrated.joblib"
    joblib.dump(obj, artifact, **dump_kw)
    manifest = tmp_path / ai.MODEL_CHECKSUM_MANIFEST
    ai.update_checksum_manifest(manifest, [artifact])
    return artifact, manifest


def test_bytes_swapped_after_verification_are_never_deserialized(tmp_path, monkeypatch):
    """The version check runs between the checksum and the load, so a swap inside
    it stands in for a concurrent writer. Refusing or loading GOOD is safe."""
    artifact, manifest = _fixture(tmp_path)
    evil = io.BytesIO()
    joblib.dump(EVIL, evil)

    def _swap(path, required=False):
        Path(path).write_bytes(evil.getvalue())
        return True

    monkeypatch.setattr(ai, "verify_artifact_library_versions", _swap)
    try:
        loaded = ai.load_verified_joblib(artifact, manifest, required_checksum=True)
    except ai.ArtifactIntegrityError as refusal:
        # Name the only refusal this setup can legitimately raise. The artifact
        # exists, its manifest entry is intact and the version check is patched
        # out, so a loader that hashes after the swap fails the checksum and
        # nothing else does. A bare `except ...: return` would also swallow an
        # integrity error raised for an unrelated reason, and then the swap
        # would never be adjudicated at all.
        assert "Checksum verification failed" in str(refusal), (
            f"refused, but not over the swapped bytes: {refusal}"
        )
    else:
        assert loaded == GOOD, "deserialized bytes that were never verified"


def test_the_artifact_is_opened_exactly_once(tmp_path, monkeypatch):
    """The same property wherever the window sits: one open leaves no second read."""
    artifact, manifest = _fixture(tmp_path)
    target = artifact.resolve()
    opens: list[object] = []
    real_open = io.open

    def _counting_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file).resolve() == target:
            opens.append(file)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", _counting_open)
    monkeypatch.setattr(builtins, "open", _counting_open)
    assert ai.load_verified_joblib(artifact, manifest, required_checksum=True) == GOOD
    assert len(opens) == 1, f"artifact opened {len(opens)} times"


def test_the_default_requires_a_checksum(tmp_path):
    artifact = tmp_path / "model.joblib"
    joblib.dump(GOOD, artifact)  # and no manifest at all
    default = inspect.signature(ai.load_verified_joblib).parameters["required_checksum"].default
    assert default is True
    with pytest.raises(ai.ArtifactIntegrityError, match="Checksum manifest required"):
        ai.load_verified_joblib(artifact)


def test_an_explicit_opt_out_still_loads_and_logs_the_skip(tmp_path, caplog):
    artifact = tmp_path / "model.joblib"
    joblib.dump(GOOD, artifact)
    with caplog.at_level("WARNING", logger="src.artifact_integrity"):
        assert ai.load_verified_joblib(artifact, required_checksum=False) == GOOD
    assert "Checksum verification SKIPPED" in caplog.text


@pytest.mark.parametrize("required", [True, False])
def test_a_tampered_artifact_is_refused(tmp_path, required):
    artifact, manifest = _fixture(tmp_path)
    joblib.dump(EVIL, artifact)
    with pytest.raises(ai.ArtifactIntegrityError, match="Checksum verification failed"):
        ai.load_verified_joblib(artifact, manifest, required_checksum=required)


def test_a_missing_artifact_is_an_integrity_error(tmp_path):
    artifact, manifest = _fixture(tmp_path)
    artifact.unlink()  # its manifest entry survives
    with pytest.raises(ai.ArtifactIntegrityError, match="Artifact not found"):
        ai.load_verified_joblib(artifact, manifest, required_checksum=True)


@pytest.mark.parametrize("entry", [{"size_bytes": 1}, {"sha256": None}, "not-a-dict"])
@pytest.mark.parametrize("required", [True, False])
def test_an_entry_without_a_digest_fails_closed(tmp_path, required, entry):
    """An entry naming the artifact with no usable digest is refused either way."""
    artifact, manifest = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifacts"][artifact.name] = entry
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ai.ArtifactIntegrityError):
        ai.load_verified_joblib(artifact, manifest, required_checksum=required)


@pytest.mark.parametrize("compress", [0, 3, ("gzip", 3), ("bz2", 3), ("xz", 3)])
def test_every_compression_loads_from_the_verified_buffer(tmp_path, compress):
    artifact, manifest = _fixture(tmp_path, compress=compress)
    assert ai.load_verified_joblib(artifact, manifest, required_checksum=True) == GOOD
