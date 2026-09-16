"""Provenance sidecars for `src/train_classifier.py`'s artifacts.

The trainer produces the production model artifacts and, until this file
existed, was the only writer in the repository that emitted no
`.provenance.json` sidecar while eight other modules did. Its checksum manifest
records what each artifact HASHES TO and nothing about what produced it, so a
manifest entry cannot distinguish a mode-31 model fitted on the v5 corpus from
one fitted on v4 under the same filename.

Every test here is hermetic: tiny synthetic corpora and a mock binding matrix
under `tmp_path`, never the 128 MB production model or the live corpus.

Separate file rather than an addition to `tests/test_train_classifier.py`
because that file is concurrently edited on an unmerged branch.
"""

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.artifact_integrity import sha256_file  # noqa: E402
from src.features import BINDING_ALLELE_COLUMNS  # noqa: E402
from src.train_classifier import (  # noqa: E402
    train_models,
    training_provenance_fields,
)

_AAS = "ACDEFGHIKLMNPQRSTVWY"

# Mode 31 is the production track (docs/model_cards/rf_31feature_integrated.md),
# so it is the mode these tests drive.
_MODE = 31
_RF_ARTIFACT = "rf_31feature_integrated.joblib"

# Every artifact a mode-31 run writes wholesale, i.e. the set that must carry a
# sidecar. Listed literally rather than derived from planned_artifact_paths() so
# a silent shrink of that enumeration cannot silently shrink this assertion too.
_EXPECTED_ARTIFACTS = [
    "rf_31feature_integrated.joblib",
    "xgb_31feature_integrated.joblib",
    "training_results.csv",
    "training_results_mode31.csv",
    "training_subgroup_metrics.csv",
    "feature_importances.csv",
    "rf_oof_predictions.csv",
    "rf_oof_predictions_mode31.csv",
    "optimal_thresholds.json",
]


def _make_peptides(n):
    """Deterministic distinct valid 9-mers (standard AA only)."""
    base = list("SLLMWITQV")
    peps = []
    for i in range(n):
        p = base.copy()
        p[0] = _AAS[i % 20]
        p[1] = _AAS[(i // 20) % 20]
        peps.append("".join(p))
    return peps


def _mock_binding_csv(tmp_path, peptides, name="binding.csv", score=0.6):
    data = {"peptide": peptides}
    for col in BINDING_ALLELE_COLUMNS:
        data[col] = [score] * len(peptides)
    path = tmp_path / name
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def _training_csv(tmp_path, n=30, name="train.csv"):
    peps = _make_peptides(n)
    labels = ([0, 1] * (n // 2 + 1))[:n]
    df = pd.DataFrame({"peptide": peps, "label": labels, "virus": ["EBV"] * n})
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path, peps


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """One mode-31 training run, shared by the assertions below.

    Module-scoped because a train_models call is the expensive part of this file
    and every sidecar assertion reads the same run's output directory.
    """
    tmp_path = tmp_path_factory.mktemp("trainprov")
    data_path, peps = _training_csv(tmp_path)
    binding = _mock_binding_csv(tmp_path, peps)
    model_dir = tmp_path / "models31"
    train_models(
        str(data_path),
        model_dir=str(model_dir),
        n_cv_folds=3,
        feature_mode=_MODE,
        binding_matrix_path=str(binding),
    )
    return {"model_dir": model_dir, "data_path": data_path, "binding": binding}


# ---------------------------------------------------------------------------
# training_provenance_fields: the payload, independent of a training run
# ---------------------------------------------------------------------------


def test_provenance_fields_record_corpus_matrix_and_mode(tmp_path):
    data_path, peps = _training_csv(tmp_path)
    binding = _mock_binding_csv(tmp_path, peps)

    fields = training_provenance_fields(data_path, _MODE, binding_matrix_path=binding)

    assert fields["feature_mode"] == _MODE
    assert fields["training_data_sha256"] == sha256_file(data_path)
    assert fields["binding_matrix_sha256"] == sha256_file(binding)
    # _relative_to_project_root falls back to the basename for a path outside
    # the repository, which is what tmp_path always is.
    assert fields["training_data_path"] == data_path.name
    assert fields["binding_matrix_path"] == binding.name


def test_provenance_fields_digest_tracks_corpus_content(tmp_path):
    """A different corpus must produce a different digest.

    An existence assertion on the key cannot see a helper that records a
    constant, or that hashes the wrong file.
    """
    first, peps = _training_csv(tmp_path, n=30, name="a.csv")
    second, _ = _training_csv(tmp_path, n=40, name="b.csv")
    binding = _mock_binding_csv(tmp_path, peps)

    a = training_provenance_fields(first, _MODE, binding_matrix_path=binding)
    b = training_provenance_fields(second, _MODE, binding_matrix_path=binding)

    assert a["training_data_sha256"] != b["training_data_sha256"]
    assert a["binding_matrix_sha256"] == b["binding_matrix_sha256"]


def test_provenance_fields_digest_tracks_binding_matrix_content(tmp_path):
    """The binding matrix is not a lesser input than the corpus.

    `prepare_features_30` zero-fills every peptide the matrix omits without
    raising, so two models fitted on one corpus and two matrices are different
    models. A sidecar that pinned only the corpus would call them identical.
    """
    data_path, peps = _training_csv(tmp_path)
    low = _mock_binding_csv(tmp_path, peps, name="low.csv", score=0.1)
    high = _mock_binding_csv(tmp_path, peps, name="high.csv", score=0.9)

    a = training_provenance_fields(data_path, _MODE, binding_matrix_path=low)
    b = training_provenance_fields(data_path, _MODE, binding_matrix_path=high)

    assert a["binding_matrix_sha256"] != b["binding_matrix_sha256"]
    assert a["training_data_sha256"] == b["training_data_sha256"]


def test_provenance_fields_omit_inputs_the_mode_does_not_read(tmp_path):
    """A mode-31 payload must not carry cache fields mode 31 cannot have.

    Recording them as null would assert the caches were consulted and found
    empty, which is a different claim from "this mode does not read them".
    """
    data_path, peps = _training_csv(tmp_path)
    binding = _mock_binding_csv(tmp_path, peps)

    fields = training_provenance_fields(data_path, _MODE, binding_matrix_path=binding)

    assert "antigen_processing_cache_path" not in fields
    assert "self_similarity_cache_path" not in fields
    assert "binding_matrix_path" in fields


def test_provenance_fields_record_caches_for_mode_35(tmp_path):
    data_path, peps = _training_csv(tmp_path)
    binding = _mock_binding_csv(tmp_path, peps)
    ap_cache = tmp_path / "ap.csv"
    ap_cache.write_text("peptide,netchop_score\n", encoding="utf-8")
    sim_cache = tmp_path / "sim.csv"
    sim_cache.write_text("peptide,self_similarity_max_identity\n", encoding="utf-8")

    fields = training_provenance_fields(
        data_path,
        35,
        binding_matrix_path=binding,
        antigen_processing_cache_path=ap_cache,
        self_similarity_cache_path=sim_cache,
    )

    assert fields["antigen_processing_cache_sha256"] == sha256_file(ap_cache)
    assert fields["self_similarity_cache_sha256"] == sha256_file(sim_cache)


def test_provenance_fields_absent_input_is_none_not_an_exception(tmp_path):
    """Mirrors model_provenance_fields: a missing file is a benign None."""
    missing = tmp_path / "gone.csv"
    fields = training_provenance_fields(missing, _MODE)
    assert fields["training_data_sha256"] is None
    assert fields["training_data_path"] == "gone.csv"


# ---------------------------------------------------------------------------
# train_models: the sidecars actually reach disk
# ---------------------------------------------------------------------------


def test_train_models_writes_a_sidecar_for_every_artifact(trained_run):
    """THE MUTATION TEST.

    Delete the `write_provenance_sidecar` loop from `train_models` and this
    fails on the first missing `.provenance.json`.
    """
    model_dir = trained_run["model_dir"]
    missing = [
        name for name in _EXPECTED_ARTIFACTS if not (model_dir / f"{name}.provenance.json").is_file()
    ]
    assert missing == [], f"artifacts written with no provenance sidecar: {missing}"


def test_model_sidecar_pins_the_inputs_that_determined_it(trained_run):
    """The sidecar's whole point: the INPUTS, not only the artifact.

    A sidecar recording only the artifact's own sha256 detects an overwrite but
    cannot say what corpus or matrix produced the bytes.
    """
    model_dir = trained_run["model_dir"]
    sidecar = model_dir / f"{_RF_ARTIFACT}.provenance.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))

    assert payload["script"] == "src/train_classifier.py"
    assert payload["artifact"] == _RF_ARTIFACT
    assert payload["sha256"] == sha256_file(model_dir / _RF_ARTIFACT)
    assert payload["feature_mode"] == _MODE
    assert payload["training_data_sha256"] == sha256_file(trained_run["data_path"])
    assert payload["binding_matrix_sha256"] == sha256_file(trained_run["binding"])


def test_every_sidecar_records_the_same_input_set(trained_run):
    """One run, one input set, so the input half must be identical everywhere.

    Only the artifact-identifying half may differ. A per-artifact divergence
    here would mean an artifact was hashed against inputs it was not built from.
    """
    model_dir = trained_run["model_dir"]
    input_keys = ("feature_mode", "training_data_sha256", "binding_matrix_sha256")
    payloads = {}
    for name in _EXPECTED_ARTIFACTS:
        payloads[name] = json.loads(
            (model_dir / f"{name}.provenance.json").read_text(encoding="utf-8")
        )

    reference = {k: payloads[_RF_ARTIFACT][k] for k in input_keys}
    for name, payload in payloads.items():
        assert {k: payload[k] for k in input_keys} == reference, name
        # ... while each still identifies its own artifact.
        assert payload["artifact"] == name
        assert payload["sha256"] == sha256_file(model_dir / name)


def test_sidecars_are_lf_terminated(trained_run):
    """`.gitattributes` pins `*.provenance.json` to LF wherever one is tracked.

    A CRLF sidecar records a Windows-only hash that disagrees with the git blob,
    which is a provenance FAIL on a byte-identical file. `write_provenance_sidecar`
    opens with `newline=""` for this reason; the assertion guards the invariant
    at the trainer's own call site rather than trusting it.
    """
    raw = (trained_run["model_dir"] / f"{_RF_ARTIFACT}.provenance.json").read_bytes()
    assert b"\r\n" not in raw
