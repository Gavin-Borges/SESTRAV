from __future__ import annotations

import json

import pytest

import hashlib
from importlib import metadata as importlib_metadata
from pathlib import Path

from src.artifact_integrity import (
    ARTIFACT_FORMAT_LIBRARIES,
    ARTIFACT_LIBRARY_PACKAGES,
    ArtifactIntegrityError,
    LIBRARY_VERSIONS_FIELD,
    MODEL_CHECKSUM_MANIFEST,
    artifact_library_dependencies,
    default_manifest_path_for,
    comparable_library_versions,
    library_version_drift,
    library_versions,
    load_checksum_manifest,
    load_verified_joblib,
    provenance_sidecar_path_for,
    sha256_file,
    update_checksum_manifest,
    verify_artifact_checksum,
    verify_artifact_library_versions,
    write_provenance_sidecar,
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


def test_optional_verification_warns_when_the_manifest_has_no_entries(tmp_path, caplog):
    """A skipped verification must leave a trace.

    No PRODUCTION caller inspects this return value. All three discard it:
    load_verified_joblib in src/artifact_integrity.py, and _load_torch_checkpoint
    in both src/baseline_comparison.py and functions/stage4_immunogenicity_scoring.py.
    So a bare False was a sensitive load proceeding completely unverified with
    nothing in the log to say so, and baseline_comparison loads a model that way.

    Scoped to production deliberately: tests DO read it, including the
    assertion two lines below and tests/test_stage4_conformal.py, which uses
    the call directly as a condition. An earlier draft of this docstring said
    "no caller in the repository", which this file refutes on its own face.

    Cited by symbol, never by line. Line citations here must carry a baseline
    entry in docs/line_citations.json, whose exempt ledger sits at its ceiling,
    so an unpinned path:NNN reddens a required check. A symbol does not move.
    """
    artifact = _write(tmp_path / "model.joblib")
    manifest = tmp_path / MODEL_CHECKSUM_MANIFEST
    manifest.write_text(json.dumps({"artifacts": {}}), encoding="utf-8")

    with caplog.at_level("WARNING", logger="src.artifact_integrity"):
        assert verify_artifact_checksum(artifact, manifest, required=False) is False

    assert "SKIPPED" in caplog.text
    assert "model.joblib" in caplog.text


def test_optional_verification_warns_when_the_entry_is_absent(tmp_path, caplog):
    artifact = _write(tmp_path / "model.joblib")
    manifest = tmp_path / MODEL_CHECKSUM_MANIFEST
    manifest.write_text(
        json.dumps({"artifacts": {"unrelated.joblib": {"sha256": "00"}}}), encoding="utf-8"
    )

    with caplog.at_level("WARNING", logger="src.artifact_integrity"):
        assert verify_artifact_checksum(artifact, manifest, required=False) is False

    assert "SKIPPED" in caplog.text
    assert "no entry in manifest" in caplog.text


# ---------------------------------------------------------------------------
# library_versions: what an artifact was written under
# ---------------------------------------------------------------------------


def test_provenance_sidecar_records_library_versions(tmp_path):
    """The sidecar must carry the versions the artifact was written under.

    An existence assertion alone would survive the field being written as an
    empty dict, so this reads a real version back out of it and compares it to
    what the running interpreter reports.
    """
    artifact = _write(tmp_path / "rf_31feature_integrated.joblib", b"model-bytes")
    sidecar = write_provenance_sidecar(artifact, script="src/train_classifier.py")

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    recorded = payload[LIBRARY_VERSIONS_FIELD]
    assert recorded["scikit-learn"] == importlib_metadata.version("scikit-learn")
    assert recorded["numpy"] == importlib_metadata.version("numpy")
    assert recorded["joblib"] == importlib_metadata.version("joblib")


def test_library_versions_covers_every_package_the_artifacts_pickle(tmp_path):
    """The recorded set must not silently shrink.

    These five are what a pickle opcode walk over models/ resolves: sklearn,
    numpy and joblib in every .joblib, xgboost in the booster dumps, torch in
    every .pth and .pt. Dropping one would make an artifact that no longer
    loads look fully provenanced.
    """
    artifact = _write(tmp_path / "m.joblib")
    sidecar = write_provenance_sidecar(artifact, script="src/train_classifier.py")

    recorded = json.loads(sidecar.read_text(encoding="utf-8"))[LIBRARY_VERSIONS_FIELD]
    assert set(recorded) == set(ARTIFACT_LIBRARY_PACKAGES)
    assert {"scikit-learn", "joblib", "numpy", "xgboost", "torch"} <= set(recorded)


def test_library_versions_records_absent_package_as_null():
    """An uninstalled package is a null, not an exception and not an omission."""
    resolved = library_versions(["numpy", "sestrav-package-that-is-not-installed"])
    assert resolved["numpy"] == importlib_metadata.version("numpy")
    assert resolved["sestrav-package-that-is-not-installed"] is None


def test_library_versions_is_json_serializable(tmp_path):
    """The field is written straight into JSON, so every value must survive it."""
    assert json.loads(json.dumps(library_versions())) == library_versions()


def test_provenance_sidecar_extra_still_wins_over_library_versions(tmp_path):
    """`extra` is applied after the standard fields, and stays that way."""
    artifact = _write(tmp_path / "m.joblib")
    sidecar = write_provenance_sidecar(
        artifact, script="s.py", extra={LIBRARY_VERSIONS_FIELD: {"numpy": "0.0.0"}}
    )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload[LIBRARY_VERSIONS_FIELD] == {"numpy": "0.0.0"}


def test_provenance_sidecar_path_appends_rather_than_replaces(tmp_path):
    """Two artifacts differing only by extension must not share one sidecar."""
    joblib_artifact = tmp_path / "model.joblib"
    torch_artifact = tmp_path / "model.pth"
    assert provenance_sidecar_path_for(joblib_artifact) != provenance_sidecar_path_for(
        torch_artifact
    )
    assert provenance_sidecar_path_for(joblib_artifact).name == "model.joblib.provenance.json"


def test_provenance_sidecar_keeps_its_lf_newlines(tmp_path):
    """The added field must not reintroduce CRLF on Windows: the recorded sha256
    has to match what git stores under the results/*.provenance.json eol=lf pin."""
    artifact = _write(tmp_path / "m.joblib")
    sidecar = write_provenance_sidecar(artifact, script="s.py")
    assert b"\r\n" not in sidecar.read_bytes()


# ---------------------------------------------------------------------------
# The load-time version gate
# ---------------------------------------------------------------------------

_ABSENT_PACKAGE = "sestrav-package-that-is-not-installed"


def _sidecar_with_versions(artifact: Path, versions: dict) -> Path:
    """Write a provenance sidecar by hand so the recorded versions can be made
    to differ from this environment without touching the artifact bytes."""
    sidecar = provenance_sidecar_path_for(artifact)
    sidecar.write_text(
        json.dumps(
            {
                "generated_utc": "2026-01-01T00:00:00+00:00",
                "script": "src/train_classifier.py",
                "artifact": artifact.name,
                "sha256": sha256_file(artifact),
                LIBRARY_VERSIONS_FIELD: versions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return sidecar


def _verified_joblib_fixture(tmp_path: Path, versions: dict | None):
    """A joblib artifact whose CHECKSUM is valid, so only the version check can
    fail. Returns (artifact, manifest_path)."""
    joblib = pytest.importorskip("joblib")
    artifact = tmp_path / "rf_31feature_integrated.joblib"
    joblib.dump({"weights": [1, 2, 3]}, artifact)
    manifest_path = tmp_path / MODEL_CHECKSUM_MANIFEST
    update_checksum_manifest(manifest_path, [artifact])
    if versions is not None:
        _sidecar_with_versions(artifact, versions)
    return artifact, manifest_path


def test_library_version_drift_reports_a_changed_version():
    drift = library_version_drift(
        {"scikit-learn": "1.3.0"}, running={"scikit-learn": "1.8.0"}
    )
    assert drift == ["scikit-learn: artifact written under 1.3.0, running 1.8.0"]


def test_library_version_drift_reports_a_package_absent_here():
    drift = library_version_drift({"xgboost": "3.2.0"}, running={"xgboost": None})
    assert drift == ["xgboost: artifact written under 3.2.0, not installed here"]


def test_library_version_drift_is_empty_when_versions_match():
    assert library_version_drift({"numpy": "2.4.6"}, running={"numpy": "2.4.6"}) == []


def test_recorded_null_version_is_not_drift():
    """A package that was absent when the artifact was written cannot be a
    dependency of it, so installing it later is not drift."""
    assert library_version_drift({"torch": None}, running={"torch": "2.13.0"}) == []


def test_library_version_drift_ignores_a_malformed_record():
    assert library_version_drift("not-a-mapping") == []
    assert library_version_drift(None) == []


def test_verify_library_versions_passes_against_a_sidecar_it_just_wrote(tmp_path, caplog):
    artifact = _write(tmp_path / "m.joblib")
    write_provenance_sidecar(artifact, script="s.py")

    with caplog.at_level("WARNING", logger="src.artifact_integrity"):
        assert verify_artifact_library_versions(artifact, required=True) is True
    assert caplog.text == ""


def test_verify_library_versions_warns_when_nothing_was_recorded(tmp_path, caplog):
    """A sidecar that records no versions is UNMEASURED, not clean, and must
    leave a trace rather than reading as a pass."""
    artifact = _write(tmp_path / "m.joblib")

    with caplog.at_level("WARNING", logger="src.artifact_integrity"):
        assert verify_artifact_library_versions(artifact, required=True) is False
    assert "SKIPPED" in caplog.text
    assert LIBRARY_VERSIONS_FIELD in caplog.text


@pytest.mark.parametrize(
    "recorded",
    [
        pytest.param({}, id="empty-mapping"),
        pytest.param({"numpy": None, "joblib": None}, id="every-value-null"),
        pytest.param({"numpy": 2, "joblib": ["a"]}, id="every-value-non-string"),
    ],
)
def test_verify_library_versions_treats_every_uncomparable_record_alike(
    tmp_path, caplog, recorded
):
    """Three sidecars with identical information content, namely nothing that
    can be compared, must produce identical verdicts.

    This is the regression anchor for a fail-open. The guard used to reject only
    an EMPTY mapping while the comparison ignored every non-string value, so a
    record of all nulls passed the guard, compared nothing, found no drift, and
    returned True, documented as "every recorded version matched", with no log
    line at all. It is not an exotic input: `library_versions` writes exactly
    that shape for a package that was not installed, and the sidecar carries no
    checksum of its own, so it is both the normal shape of an uninformative
    record and the shape a tampered one would have.

    Asserting the three are EQUIVALENT is what makes this test bite. Asserting
    only that the empty mapping warns would have passed against the defect.
    """
    artifact = _write(tmp_path / "m.joblib")
    _sidecar_with_versions(artifact, recorded)

    with caplog.at_level("WARNING", logger="src.artifact_integrity"):
        assert verify_artifact_library_versions(artifact, required=True) is False
    assert "SKIPPED" in caplog.text
    assert LIBRARY_VERSIONS_FIELD in caplog.text


def test_comparable_library_versions_keeps_only_string_values():
    """The single definition the guard and the comparison must share, so they
    cannot drift back apart into two notions of "has something to compare"."""
    assert comparable_library_versions(
        {"numpy": "2.4.6", "torch": None, "joblib": 3}
    ) == {"numpy": "2.4.6"}
    assert comparable_library_versions({}) == {}
    assert comparable_library_versions("not-a-mapping") == {}
    assert comparable_library_versions(None) == {}


def test_verify_library_versions_warns_on_an_unreadable_sidecar(tmp_path, caplog):
    artifact = _write(tmp_path / "m.joblib")
    provenance_sidecar_path_for(artifact).write_text("{not json", encoding="utf-8")

    with caplog.at_level("WARNING", logger="src.artifact_integrity"):
        assert verify_artifact_library_versions(artifact, required=True) is False
    assert "SKIPPED" in caplog.text


def test_verify_library_versions_warns_on_drift_when_not_required(tmp_path, caplog):
    artifact = _write(tmp_path / "m.joblib")
    _sidecar_with_versions(artifact, {"scikit-learn": "0.0.1-not-this-one"})

    with caplog.at_level("WARNING", logger="src.artifact_integrity"):
        assert verify_artifact_library_versions(artifact, required=False) is False
    assert "DRIFT" in caplog.text
    assert "scikit-learn" in caplog.text


def test_verify_library_versions_raises_on_drift_when_required(tmp_path):
    artifact = _write(tmp_path / "m.joblib")
    _sidecar_with_versions(artifact, {"scikit-learn": "0.0.1-not-this-one"})

    with pytest.raises(ArtifactIntegrityError, match="Library version drift"):
        verify_artifact_library_versions(artifact, required=True)


def test_load_verified_joblib_raises_on_library_drift_when_checksum_required(tmp_path):
    """The gate that fires on the sensitive load path. The checksum here is
    VALID: an intact pickle written by a different scikit-learn is exactly the
    case a digest cannot see."""
    artifact, manifest_path = _verified_joblib_fixture(
        tmp_path, {"scikit-learn": "0.0.1-not-this-one"}
    )
    assert verify_artifact_checksum(artifact, manifest_path, required=True) is True

    with pytest.raises(ArtifactIntegrityError, match="Library version drift"):
        load_verified_joblib(artifact, manifest_path, required_checksum=True)


def test_load_verified_joblib_warns_but_loads_on_drift_when_not_required(tmp_path, caplog):
    artifact, manifest_path = _verified_joblib_fixture(
        tmp_path, {"scikit-learn": "0.0.1-not-this-one"}
    )

    with caplog.at_level("WARNING", logger="src.artifact_integrity"):
        assert load_verified_joblib(artifact, manifest_path) == {"weights": [1, 2, 3]}
    assert "DRIFT" in caplog.text


def test_load_verified_joblib_still_loads_when_no_versions_were_recorded(tmp_path, caplog):
    """Every artifact on disk predates this field. Raising on an absent record
    would make required_checksum=True unusable, so absence warns and loads."""
    artifact, manifest_path = _verified_joblib_fixture(tmp_path, None)

    with caplog.at_level("WARNING", logger="src.artifact_integrity"):
        assert load_verified_joblib(artifact, manifest_path, required_checksum=True) == {
            "weights": [1, 2, 3]
        }
    assert "SKIPPED" in caplog.text


def test_load_verified_joblib_passes_with_the_running_versions_recorded(tmp_path, caplog):
    artifact, manifest_path = _verified_joblib_fixture(tmp_path, library_versions())

    with caplog.at_level("WARNING", logger="src.artifact_integrity"):
        assert load_verified_joblib(artifact, manifest_path, required_checksum=True) == {
            "weights": [1, 2, 3]
        }
    assert caplog.text == ""


def test_recorded_version_of_an_uninstalled_package_is_drift(tmp_path, monkeypatch):
    """A recorded version this environment cannot supply is drift end to end.

    The absent package has to be made RELEVANT to the format first. Only the
    distributions `artifact_library_dependencies` names are adjudicated, and by
    construction every name in the real mapping is installed here, so recording
    `_ABSENT_PACKAGE` alone against an unpatched `.joblib` would be filtered out
    and reported as UNMEASURED - which is the correct verdict for an irrelevant
    package and the wrong fixture for this property.
    """
    monkeypatch.setitem(
        ARTIFACT_FORMAT_LIBRARIES,
        ".joblib",
        ARTIFACT_FORMAT_LIBRARIES[".joblib"] + (_ABSENT_PACKAGE,),
    )
    artifact = _write(tmp_path / "m.joblib")
    _sidecar_with_versions(artifact, {_ABSENT_PACKAGE: "1.0.0"})

    with pytest.raises(ArtifactIntegrityError, match="not installed here"):
        verify_artifact_library_versions(artifact, required=True)


# ---------------------------------------------------------------------------
# Scoping the comparison to the format that actually depends on the library
# ---------------------------------------------------------------------------


def test_artifact_library_dependencies_scopes_each_shipped_format():
    """`.joblib` must not name torch and `.pth` must not name the pickle stack.

    Asserting the exact sets, rather than that each is non-empty, is what makes
    this bite: the defect being anchored here was a single UNION applied to
    every format.
    """
    assert artifact_library_dependencies("models/rf_mode31.joblib") == (
        "scikit-learn",
        "joblib",
        "numpy",
        "xgboost",
    )
    assert artifact_library_dependencies("models/gnn_best.pth") == ("torch",)
    assert artifact_library_dependencies("models/gnn_best.pt") == ("torch",)
    assert artifact_library_dependencies("results/h2_tier_a_summary.csv") == ()
    assert "torch" not in artifact_library_dependencies("m.joblib")


def test_every_scoped_package_is_one_the_writer_records():
    """The mapping may only narrow `ARTIFACT_LIBRARY_PACKAGES`, never extend it.

    A format scoped to a distribution the writer never records would compare
    nothing and report UNMEASURED forever, which is a fail-open wearing the
    costume of a stricter rule.
    """
    for suffix, packages in ARTIFACT_FORMAT_LIBRARIES.items():
        assert set(packages) <= set(ARTIFACT_LIBRARY_PACKAGES), suffix
        assert packages, suffix


def test_torch_drift_does_not_block_an_sklearn_artifact(tmp_path, caplog):
    """THE REGRESSION ANCHOR. A torch upgrade must not break loading an RF.

    An `rf_*.joblib` is a graph of sklearn, numpy and joblib objects and holds
    no torch reference, so torch's version cannot change how it deserializes.
    The writer records the whole environment, which is right; adjudicating the
    whole environment made `required=True` raise here, which was not.
    """
    artifact = _write(tmp_path / "rf_31feature_integrated.joblib")
    recorded = dict(library_versions())
    recorded["torch"] = "0.0.1-a-torch-this-environment-does-not-have"
    _sidecar_with_versions(artifact, recorded)

    with caplog.at_level("WARNING", logger="src.artifact_integrity"):
        assert verify_artifact_library_versions(artifact, required=True) is True
    assert caplog.text == ""


def test_torch_drift_DOES_block_a_torch_checkpoint(tmp_path):
    """The non-vacuity partner of the test above, on the same bumped package.

    Without this, scoping could be implemented as "never compare torch" and the
    anchor above would still pass.
    """
    artifact = _write(tmp_path / "gnn_best.pth")
    recorded = dict(library_versions())
    recorded["torch"] = "0.0.1-a-torch-this-environment-does-not-have"
    _sidecar_with_versions(artifact, recorded)

    with pytest.raises(ArtifactIntegrityError, match="Library version drift"):
        verify_artifact_library_versions(artifact, required=True)


def test_sklearn_drift_still_blocks_a_joblib_artifact(tmp_path):
    """The second non-vacuity partner: scoping must not disarm the gate it
    narrows. Same artifact as the anchor, a package that IS its dependency."""
    artifact = _write(tmp_path / "rf_31feature_integrated.joblib")
    recorded = dict(library_versions())
    recorded["scikit-learn"] = "0.0.1-not-this-one"
    _sidecar_with_versions(artifact, recorded)

    with pytest.raises(ArtifactIntegrityError, match="Library version drift"):
        verify_artifact_library_versions(artifact, required=True)


def test_an_artifact_that_loads_through_no_library_verifies_silently(tmp_path, caplog):
    """A results CSV depends on no library, so drift in any of them is not
    evidence about it. Reporting every CSV as UNMEASURED would train the reader
    to ignore the warning that matters."""
    artifact = _write(tmp_path / "h2_tier_a_summary.csv", b"peptide,label\nSIINFEKL,1\n")
    recorded = dict(library_versions())
    recorded["torch"] = "0.0.1-a-torch-this-environment-does-not-have"
    recorded["scikit-learn"] = "0.0.1-not-this-one"
    _sidecar_with_versions(artifact, recorded)

    with caplog.at_level("WARNING", logger="src.artifact_integrity"):
        assert verify_artifact_library_versions(artifact, required=True) is True
    assert caplog.text == ""


def test_a_joblib_recording_only_irrelevant_packages_is_unmeasured(tmp_path, caplog):
    """Scoping must not turn an uninformative record into a pass. A `.joblib`
    whose sidecar names torch ALONE has nothing comparable, which is the
    UNMEASURED verdict, not the clean one."""
    artifact = _write(tmp_path / "m.joblib")
    _sidecar_with_versions(artifact, {"torch": library_versions(["torch"])["torch"]})

    with caplog.at_level("WARNING", logger="src.artifact_integrity"):
        assert verify_artifact_library_versions(artifact, required=True) is False
    assert "SKIPPED" in caplog.text
