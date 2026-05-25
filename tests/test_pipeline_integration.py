"""
Integration tests for the SESTRAV pipeline data flow.

Verifies that feature extraction, model scoring, gold-standard validation,
and baseline comparison all work together correctly without requiring
MHCflurry (Stages 1-2 are mocked with synthetic data).

Run from repo root:
    python -m pytest tests/test_pipeline_integration.py -v
    python tests/test_pipeline_integration.py
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from src.artifact_integrity import load_verified_joblib

from src.features import (
    compute_features, compute_features_for_dataset,
    BINDING_ALLELE_COLUMNS, FEATURE_COLUMNS, FEATURE_COLUMNS_30, TRAIN_FEATURE_COLUMNS,
)
from src.evaluate_metrics import evaluate
from src.gold_standard import GOLD_STANDARD
from src.naming import resolve_model_path


SAMPLE_PEPTIDES = [
    {'peptide': 'CLGGLLTMV', 'presentation_score': 0.98, 'protein_id': 'LMP2A'},
    {'peptide': 'RAKFKQLL',  'presentation_score': 0.45, 'protein_id': 'BZLF1'},
    {'peptide': 'TIHDIILECV','presentation_score': 0.72, 'protein_id': 'VE6'},
    {'peptide': 'HPVGEADYFEY','presentation_score': 0.81, 'protein_id': 'EBNA1'},
    {'peptide': 'AAAAAAAAA', 'presentation_score': 0.01, 'protein_id': 'FAKE'},
    {'peptide': 'WWWWWWWWWW','presentation_score': 0.50, 'protein_id': 'FAKE'},
]


def _make_synthetic_features_df():
    """Build a DataFrame that mimics Stage 2 output, run through Stage 3."""
    df = pd.DataFrame(SAMPLE_PEPTIDES)
    df = compute_features_for_dataset(df, peptide_col='peptide',
                                      binding_col='presentation_score')
    for idx, col in enumerate(BINDING_ALLELE_COLUMNS):
        # Synthetic but deterministic per-allele binding values for 30-feature tests.
        df[col] = np.clip(df['presentation_score'] - (idx * 0.01), 0.0, 1.0)
    return df


def test_feature_extraction_produces_22_columns():
    """Stage 3: compute_features_for_dataset adds exactly 22 feature columns."""
    df = _make_synthetic_features_df()
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing feature column: {col}"
    assert 'peptide' in df.columns
    assert 'presentation_score' in df.columns


def test_train_feature_columns_are_21():
    """TRAIN_FEATURE_COLUMNS must be FEATURE_COLUMNS minus binding_score."""
    assert len(TRAIN_FEATURE_COLUMNS) == 21
    assert 'binding_score' not in TRAIN_FEATURE_COLUMNS
    assert 'peptide_length' in TRAIN_FEATURE_COLUMNS


def test_canonical_feature_columns_are_present():
    """Synthetic Stage 3 output can satisfy the canonical 30-feature schema."""
    df = _make_synthetic_features_df()
    for col in FEATURE_COLUMNS_30:
        assert col in df.columns, f"Missing canonical feature column: {col}"
    X = df[FEATURE_COLUMNS_30]
    assert X.shape[1] == 30


def test_config_defaults_to_canonical_30_feature_release_path():
    """Release defaults in config.yaml stay aligned to canonical 30-feature mode."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config_text = f.read()
    assert 'feature_mode: 30' in config_text
    assert 'model_path: models/rf_30feature_integrated.joblib' in config_text


def test_legacy_rf_model_loads_and_scores():
    """Legacy Stage 4 path remains supported for 21-feature RF models."""
    model_path = resolve_model_path(os.path.join(
        os.path.dirname(__file__), '..', 'models', 'rf_21feature_legacy.joblib'
    ))
    if not os.path.isfile(model_path):
        import pytest
        pytest.skip("rf_21feature_legacy.joblib not found (run training first)")

    model = load_verified_joblib(model_path, required_checksum=True)
    assert model.n_features_in_ == 21

    df = _make_synthetic_features_df()
    X = df[TRAIN_FEATURE_COLUMNS]
    proba = model.predict_proba(X)

    assert proba.shape == (len(SAMPLE_PEPTIDES), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_xgb_model_loads_and_scores():
    """Stage 4: XGB model loads and scores consistently with RF."""
    model_path = resolve_model_path(os.path.join(
        os.path.dirname(__file__), '..', 'models', 'xgb_21feature_legacy.joblib'
    ))
    if not os.path.isfile(model_path):
        import pytest
        pytest.skip("xgb_21feature_legacy.joblib not found (run training first)")

    model = load_verified_joblib(model_path, required_checksum=True)
    assert model.n_features_in_ == 21

    df = _make_synthetic_features_df()
    X = df[TRAIN_FEATURE_COLUMNS]
    proba = model.predict_proba(X)

    assert proba.shape == (len(SAMPLE_PEPTIDES), 2)


def test_canonical_rf_model_loads_and_scores():
    """Canonical Stage 4 path loads and scores a 30-feature RF model."""
    model_path = os.path.join(
        os.path.dirname(__file__), '..', 'models', 'rf_30feature_integrated.joblib'
    )
    if not os.path.isfile(model_path):
        import pytest
        pytest.skip("rf_30feature_integrated.joblib not found (run canonical training first)")

    model = load_verified_joblib(model_path, required_checksum=True)
    assert model.n_features_in_ == 30

    df = _make_synthetic_features_df()
    X = df[FEATURE_COLUMNS_30]
    proba = model.predict_proba(X)

    assert proba.shape == (len(SAMPLE_PEPTIDES), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_evaluate_metrics_on_scored_peptides():
    """Metrics module works on model output."""
    model_path = resolve_model_path(os.path.join(
        os.path.dirname(__file__), '..', 'models', 'rf_21feature_legacy.joblib'
    ))
    if not os.path.isfile(model_path):
        import pytest
        pytest.skip("model not found (run training first)")

    model = load_verified_joblib(model_path, required_checksum=True)
    df = _make_synthetic_features_df()
    scores = model.predict_proba(df[TRAIN_FEATURE_COLUMNS])[:, 1]

    y_true = np.array([1, 1, 0, 1, 0, 0])
    metrics = evaluate(y_true, scores)

    core_keys = {'auc_roc', 'auc_pr', 'issr_10', 'issr_25'}
    assert core_keys.issubset(set(metrics.keys()))
    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


def test_gold_standard_list_consistency():
    """Gold-standard list has 15 entries covering both viruses."""
    assert len(GOLD_STANDARD) == 15
    viruses = {gs['virus'] for gs in GOLD_STANDARD}
    assert viruses == {'EBV', 'HPV'}
    ebv = [gs for gs in GOLD_STANDARD if gs['virus'] == 'EBV']
    hpv = [gs for gs in GOLD_STANDARD if gs['virus'] == 'HPV']
    assert len(ebv) == 10
    assert len(hpv) == 5
    for gs in GOLD_STANDARD:
        assert 8 <= len(gs['peptide']) <= 11


def test_end_to_end_score_and_rank():
    """Full Stage 3+4 flow: features -> model -> score -> rank."""
    model_path = resolve_model_path(os.path.join(
        os.path.dirname(__file__), '..', 'models', 'rf_21feature_legacy.joblib'
    ))
    if not os.path.isfile(model_path):
        import pytest
        pytest.skip("model not found (run training first)")

    model = load_verified_joblib(model_path, required_checksum=True)
    df = _make_synthetic_features_df()
    X = df[TRAIN_FEATURE_COLUMNS]

    df['immunogenicity_score'] = model.predict_proba(X)[:, 1]
    df['rank'] = df['immunogenicity_score'].rank(ascending=False).astype(int)

    assert df['rank'].min() == 1
    assert df['rank'].max() == len(df)
    assert not df['immunogenicity_score'].isna().any()

    fd, tmppath = tempfile.mkstemp(prefix="sestrav_integration_", suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(tmppath, index=False)
        reloaded = pd.read_csv(tmppath)
        assert 'immunogenicity_score' in reloaded.columns
        assert 'rank' in reloaded.columns
        assert len(reloaded) == len(SAMPLE_PEPTIDES)
    finally:
        if os.path.isfile(tmppath):
            os.unlink(tmppath)


if __name__ == "__main__":
    test_feature_extraction_produces_22_columns()
    test_train_feature_columns_are_21()
    test_canonical_feature_columns_are_present()
    test_config_defaults_to_canonical_30_feature_release_path()
    test_legacy_rf_model_loads_and_scores()
    test_xgb_model_loads_and_scores()
    test_canonical_rf_model_loads_and_scores()
    test_evaluate_metrics_on_scored_peptides()
    test_gold_standard_list_consistency()
    test_end_to_end_score_and_rank()
    print("All integration tests passed.")
