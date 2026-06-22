"""Targeted unit tests covering discrete, dependency-free code paths.

Each test targets a specific uncovered range in a module that was otherwise
at 0% or near-0% coverage, pushing whole-repo coverage over the 35% gate.
Modules targeted:
  - src/iedb_data_loader.py  private filename helpers  (lines 43-60)
  - src/train_classifier.py  feature-matrix builders   (lines 60-71, 99,
                                                         150-156, 161-184)
  - src/train_classifier.py  _cross_validate paths     (lines 280, 308-309, 321)
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.iedb_data_loader import _label_from_filename, _virus_from_filename
from src.train_classifier import (
    BINDING_ALLELE_COLUMNS,
    _cross_validate,
    _get_protein_name_from_header,
    prepare_features,
    prepare_features_30_graph,
    prepare_features_50,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_PEPTIDES = ["CLGGLLTMV", "RAKFKQLL"]


def _mock_binding_csv(tmp_path, peptides=None):
    """Write a minimal mock binding matrix with all 10 allele columns."""
    if peptides is None:
        peptides = [_PEPTIDES[0]]  # only first peptide present
    data = {"peptide": peptides}
    for col in BINDING_ALLELE_COLUMNS:
        data[col] = [0.7] * len(peptides)
    csv_path = tmp_path / "mock_binding.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# iedb_data_loader - _label_from_filename (lines 43-48)
# ---------------------------------------------------------------------------


def test_label_from_filename_positive():
    assert _label_from_filename("EBV_T-cell_positive_epitopes.xlsx") == 1


def test_label_from_filename_negative():
    assert _label_from_filename("HPV16_T-cell_negative_epitopes.xlsx") == 0


def test_label_from_filename_unknown():
    assert _label_from_filename("some_other_file.xlsx") is None


# ---------------------------------------------------------------------------
# iedb_data_loader - _virus_from_filename (lines 53-60)
# ---------------------------------------------------------------------------


def test_virus_from_filename_ebv():
    assert _virus_from_filename("EBV_positive.xlsx") == "EBV"


def test_virus_from_filename_hpv16():
    assert _virus_from_filename("HPV16_data.xlsx") == "HPV16"


def test_virus_from_filename_hvp16_typo():
    assert _virus_from_filename("HVP16_something.xlsx") == "HPV16"


def test_virus_from_filename_hpv11():
    assert _virus_from_filename("HPV11_epitopes.csv") == "HPV11"


def test_virus_from_filename_unknown():
    assert _virus_from_filename("unknown_dataset.csv") is None


# ---------------------------------------------------------------------------
# train_classifier - prepare_features (lines 60-71)
# ---------------------------------------------------------------------------


def test_prepare_features_no_binding_shape():
    df = pd.DataFrame({"peptide": _PEPTIDES})
    result = prepare_features(df, include_binding=False)
    assert result.shape[0] == 2
    assert "binding_score" not in result.columns


def test_prepare_features_no_binding_no_nan():
    df = pd.DataFrame({"peptide": _PEPTIDES})
    assert not prepare_features(df).isnull().any().any()


def test_prepare_features_with_binding_shape():
    df = pd.DataFrame({"peptide": _PEPTIDES, "binding_score": [0.8, 0.3]})
    result = prepare_features(df, include_binding=True, binding_col="binding_score")
    assert result.shape[0] == 2
    assert "binding_score" in result.columns


def test_prepare_features_nan_binding_replaced():
    df = pd.DataFrame({"peptide": _PEPTIDES, "binding_score": [0.8, float("nan")]})
    result = prepare_features(df, include_binding=True, binding_col="binding_score")
    assert not result.isnull().any().any()


# ---------------------------------------------------------------------------
# train_classifier - prepare_features_30 missing-peptide branch (line 99)
# ---------------------------------------------------------------------------


def test_prepare_features_30_missing_peptide_zeros(tmp_path):
    from src.train_classifier import prepare_features_30

    csv_path = _mock_binding_csv(tmp_path, peptides=[_PEPTIDES[0]])
    df = pd.DataFrame({"peptide": _PEPTIDES})
    result = prepare_features_30(df, str(csv_path))
    assert result.shape[0] == 2
    assert result.iloc[1][BINDING_ALLELE_COLUMNS[0]] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# train_classifier - prepare_features_30_graph (lines 150-156)
# ---------------------------------------------------------------------------


def test_prepare_features_30_graph_shape(tmp_path):
    csv_path = _mock_binding_csv(tmp_path)
    df = pd.DataFrame({"peptide": [_PEPTIDES[0]]})
    result = prepare_features_30_graph(df, str(csv_path))
    assert result.shape[0] == 1
    assert result.shape[1] == 62


def test_prepare_features_30_graph_no_nan(tmp_path):
    csv_path = _mock_binding_csv(tmp_path)
    df = pd.DataFrame({"peptide": [_PEPTIDES[0]]})
    result = prepare_features_30_graph(df, str(csv_path))
    assert not result.isnull().any().any()


# ---------------------------------------------------------------------------
# train_classifier - prepare_features_50 (lines 161-184)
# ---------------------------------------------------------------------------


def test_prepare_features_50_shape(tmp_path):
    csv_path = _mock_binding_csv(tmp_path, peptides=[_PEPTIDES[0]])
    df = pd.DataFrame({"peptide": _PEPTIDES})
    result = prepare_features_50(df, str(csv_path))
    assert result.shape == (2, 50)


def test_prepare_features_50_missing_peptide_zeros(tmp_path):
    csv_path = _mock_binding_csv(tmp_path, peptides=[_PEPTIDES[0]])
    df = pd.DataFrame({"peptide": _PEPTIDES})
    result = prepare_features_50(df, str(csv_path))
    assert result.iloc[1][BINDING_ALLELE_COLUMNS[0]] == pytest.approx(0.0)


def test_prepare_features_50_raises_insufficient_alleles(tmp_path):
    csv_path = tmp_path / "bad_binding.csv"
    pd.DataFrame({"peptide": [_PEPTIDES[0]], "bind_A0101": [0.5]}).to_csv(
        csv_path, index=False
    )
    df = pd.DataFrame({"peptide": [_PEPTIDES[0]]})
    with pytest.raises(ValueError, match="only 1/10"):
        prepare_features_50(df, str(csv_path))


# ---------------------------------------------------------------------------
# train_classifier - _get_protein_name_from_header empty case (line 280)
# ---------------------------------------------------------------------------


def test_get_protein_name_from_header_empty_string():
    assert _get_protein_name_from_header("") == ""


# ---------------------------------------------------------------------------
# train_classifier - _cross_validate StratifiedKFold path (lines 308-309)
# and sample_weights branch (line 321)
# ---------------------------------------------------------------------------


def _cv_fixture(n=40):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    metadata = pd.DataFrame(
        {
            "peptide": [f"P{i}" for i in range(n)],
            "virus": ["EBV"] * n,
            "protein": ["PROT_A"] * (n // 2) + ["PROT_B"] * (n // 2),
        }
    )
    return X, y, metadata


def test_cross_validate_skf_path_returns_metrics():
    X, y, meta = _cv_fixture()
    avg, std, _, oof_df = _cross_validate(
        X, y, meta,
        RandomForestClassifier,
        {"n_estimators": 5, "random_state": 0},
        n_splits=3, use_lopo=False,
    )
    assert "auc_pr" in avg
    assert len(oof_df) == len(X)


def test_cross_validate_sample_weights_path():
    X, y, meta = _cv_fixture()
    weights = np.ones(len(y))
    avg, _, _, _ = _cross_validate(
        X, y, meta,
        RandomForestClassifier,
        {"n_estimators": 5, "random_state": 0},
        n_splits=3, use_lopo=False, sample_weights=weights,
    )
    assert "auc_pr" in avg
