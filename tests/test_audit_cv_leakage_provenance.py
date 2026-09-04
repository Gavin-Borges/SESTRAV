"""The leakage audit's provenance sidecar must digest its FEATURE inputs.

Before this, the sidecar recorded dataset_sha256 alone, so a run was
reproducible in its rows but not in its features. On 2026-09-04 a stale binding
matrix was found to reach 0.79% of the v5 corpus's tested_negative rows, turning
the zero-fill into a label proxy worth roughly -0.154 pooled AUC-PR, and no
artifact of the affected runs recorded which matrix had been opened. The
self-similarity and antigen-processing caches carry the same exposure.

The three input paths are module constants pointing at real repo files, two of
which are gitignored and therefore absent from a fresh clone, so these tests
monkeypatch them onto tmp_path fixtures rather than depending on the workstation.
"""

import hashlib
import json

import pytest

from scripts import audit_cv_leakage as acl


@pytest.fixture
def fake_inputs(tmp_path, monkeypatch):
    """Point the three feature-input constants at real files under PROJECT_ROOT."""
    root = tmp_path / "repo"
    (root / "models").mkdir(parents=True)
    (root / "data").mkdir()
    paths = {
        "binding_matrix": root / "models" / "peptide_binding_matrix_v5.csv",
        "antigen_processing_cache": root / "data" / "antigen_processing_cache.csv",
        "self_similarity_cache": root / "data" / "self_similarity_cache.csv",
    }
    for name, p in paths.items():
        p.write_text(f"contents of {name}\n", encoding="utf-8")
    monkeypatch.setattr(acl, "PROJECT_ROOT", root)
    monkeypatch.setattr(acl, "BINDING_MATRIX_PATH", paths["binding_matrix"])
    monkeypatch.setattr(acl, "ANTIGEN_PROCESSING_CACHE_PATH", paths["antigen_processing_cache"])
    monkeypatch.setattr(acl, "SELF_SIMILARITY_CACHE_PATH", paths["self_similarity_cache"])
    return root, paths


def test_feature_input_provenance_names_all_three_inputs(fake_inputs):
    _, paths = fake_inputs
    prov = acl._feature_input_provenance()
    assert set(prov) == set(paths)


def test_feature_input_provenance_digests_match_the_files(fake_inputs):
    _, paths = fake_inputs
    prov = acl._feature_input_provenance()
    for name, p in paths.items():
        assert prov[name]["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()


def test_feature_input_provenance_paths_are_repo_relative_posix(fake_inputs):
    prov = acl._feature_input_provenance()
    assert prov["binding_matrix"]["path"] == "models/peptide_binding_matrix_v5.csv"
    assert prov["antigen_processing_cache"]["path"] == "data/antigen_processing_cache.csv"
    assert prov["self_similarity_cache"]["path"] == "data/self_similarity_cache.csv"


def test_a_changed_cache_changes_the_recorded_digest(fake_inputs):
    _, paths = fake_inputs
    before = acl._feature_input_provenance()["self_similarity_cache"]["sha256"]
    paths["self_similarity_cache"].write_text("a different proteome\n", encoding="utf-8")
    after = acl._feature_input_provenance()["self_similarity_cache"]["sha256"]
    assert before != after


def test_written_sidecar_carries_feature_inputs(fake_inputs):
    root, paths = fake_inputs
    dataset = root / "data" / "immunogenicity_dataset_v5.csv"
    dataset.write_text("peptide,label\nAAAAAAAAA,1\n", encoding="utf-8")
    out = root / "results" / "cv_leakage_audit.csv"
    out.parent.mkdir()
    out.write_text("metric,value\noverall_peptide_overlap_pct,0.0\n", encoding="utf-8")

    acl._write_provenance(out, dataset, n_rows=1)

    sidecar = json.loads(
        acl._provenance_path(out).read_text(encoding="utf-8")
    )
    assert "dataset_sha256" in sidecar
    assert set(sidecar["feature_inputs"]) == set(paths)
    for name, p in paths.items():
        assert sidecar["feature_inputs"][name]["sha256"] == (
            hashlib.sha256(p.read_bytes()).hexdigest()
        )
