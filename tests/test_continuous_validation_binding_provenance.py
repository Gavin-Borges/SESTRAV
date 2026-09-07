"""The monthly IEDB benchmark must record the binding matrix, not just the model.

`src/continuous_validation.py` scores a FRESHLY FETCHED IEDB export against a
FROZEN binding matrix, on a cron, and opens a GitHub Issue on a regression. It
already records `model_path` and `model_sha256`, added after a benchmark recorded
no model hash, the model was overwritten, and the figure became permanently
unreproducible.

The binding matrix was the input left out, and it is not a lesser one. A mode-31
score is a function of the matrix's COVERAGE as much as of the model, because
`prepare_features_30` substitutes `np.zeros(10)` for every peptide the matrix
omits without raising. Coverage here necessarily decays as IEDB grows: any
peptide added since the matrix was built cannot be in it. On the v5 corpus that
substitution is worth roughly 0.15 AUC-PR, which is larger than the 3% regression
threshold this job alerts on - and it pushes in the direction that keeps the alarm
quiet, because a stronger label proxy looks like stability.
"""

import json

import pandas as pd
import pytest

import src.continuous_validation as cv
from src.artifact_integrity import binding_matrix_provenance_fields


def _matrix(tmp_path, peptides, name="bm.csv"):
    path = tmp_path / name
    pd.DataFrame({"peptide": list(peptides), "bind_A0101": [0.5] * len(peptides)}).to_csv(
        path, index=False
    )
    return path


def _frame(peptides):
    return pd.DataFrame({"peptide": list(peptides), "label": [1, 0] * (len(peptides) // 2)})


PEPS = ["AAAAAAAAA", "CCCCCCCCC", "DDDDDDDDD", "EEEEEEEEE"]


# --- provenance fields -----------------------------------------------------


def test_provenance_fields_hash_an_existing_matrix(tmp_path):
    path = _matrix(tmp_path, PEPS)
    fields = binding_matrix_provenance_fields(path)
    assert set(fields) == {"binding_matrix_path", "binding_matrix_sha256"}
    assert isinstance(fields["binding_matrix_sha256"], str)
    assert len(fields["binding_matrix_sha256"]) == 64


def test_provenance_sha_is_none_for_an_absent_matrix(tmp_path):
    """Absent must not raise - it mirrors model_provenance_fields exactly."""
    fields = binding_matrix_provenance_fields(tmp_path / "nope.csv")
    assert fields["binding_matrix_sha256"] is None


def test_an_in_place_rebuild_changes_the_hash(tmp_path):
    """The failure this guards: same path, different content, silently swapped."""
    path = _matrix(tmp_path, PEPS)
    before = binding_matrix_provenance_fields(path)["binding_matrix_sha256"]
    _matrix(tmp_path, PEPS + ["FFFFFFFFF", "GGGGGGGGG"], name="bm.csv")
    after = binding_matrix_provenance_fields(path)["binding_matrix_sha256"]
    assert before != after


# --- coverage --------------------------------------------------------------


def test_full_coverage_reports_one(tmp_path):
    got = cv.binding_matrix_coverage(_frame(PEPS), str(_matrix(tmp_path, PEPS)))
    assert got == {"binding_matrix_covered": 4, "binding_matrix_coverage": 1.0}


def test_partial_coverage_is_measured(tmp_path):
    got = cv.binding_matrix_coverage(_frame(PEPS), str(_matrix(tmp_path, PEPS[:1])))
    assert got["binding_matrix_covered"] == 1
    assert got["binding_matrix_coverage"] == pytest.approx(0.25)


def test_zero_coverage_is_reported_not_crashed(tmp_path):
    """The decayed-matrix endpoint: nothing matches, every row zero-filled."""
    got = cv.binding_matrix_coverage(_frame(PEPS), str(_matrix(tmp_path, ["ZZZZZZZZZ"])))
    assert got["binding_matrix_covered"] == 0
    assert got["binding_matrix_coverage"] == 0.0


def test_absent_matrix_degrades_to_none_rather_than_raising(tmp_path):
    got = cv.binding_matrix_coverage(_frame(PEPS), str(tmp_path / "absent.csv"))
    assert got == {"binding_matrix_covered": None, "binding_matrix_coverage": None}


# --- wiring ----------------------------------------------------------------


def test_run_records_the_matrix_in_the_payload(tmp_path):
    """A score with no recorded matrix is not reproducible. Pin the wiring."""
    matrix = _matrix(tmp_path, PEPS)
    baseline = tmp_path / "baselines.json"
    baseline.write_text(json.dumps({cv.BASELINE_KEY: {"auc_pr": 0.80}}))
    results = tmp_path / "out"

    code = cv.run(
        _write_inputs(tmp_path),
        baseline_path=str(baseline),
        results_dir=str(results),
        binding_matrix_path=str(matrix),
        score_fn=lambda df, m, b: {"auc_pr": 0.79, "n_peptides": len(df), "n_positive": 2},
        marker_path=str(tmp_path / "MARKER"),
        today="2026-09-04",
    )
    assert code == 0
    payload = json.loads((results / "benchmark_latest.json").read_text())
    assert payload["binding_matrix_path"].endswith("bm.csv")
    assert len(payload["binding_matrix_sha256"]) == 64


def test_score_iedb_export_merges_the_coverage_fields():
    """Guards that coverage is threaded into the metrics, not merely computable."""
    import inspect

    source = inspect.getsource(cv.score_iedb_export)
    assert "**binding_matrix_coverage(df, binding_matrix_path)" in source


def _write_inputs(tmp_path):
    """Mirror of the helper in tests/test_continuous_validation.py."""
    path = tmp_path / "in.csv"
    pd.DataFrame(
        {"peptide": PEPS, "label": [1, 0, 1, 0]}
    ).to_csv(path, index=False)
    return [str(path)]
