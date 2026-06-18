"""Tests for FEATURE_COLUMNS_33 definition and prepare_features_33 in train_classifier."""

import numpy as np
import pandas as pd

from src.features import (
    FEATURE_COLUMNS_31,
    FEATURE_COLUMNS_33,
)


def test_feature_columns_33_length():
    assert len(FEATURE_COLUMNS_33) == 33


def test_feature_columns_33_is_superset_of_31():
    assert set(FEATURE_COLUMNS_31).issubset(set(FEATURE_COLUMNS_33))


def test_feature_columns_33_adds_processing_columns():
    extra = [c for c in FEATURE_COLUMNS_33 if c not in FEATURE_COLUMNS_31]
    assert sorted(extra) == ['netchop_score', 'tap_score']


def test_feature_columns_33_no_duplicates():
    assert len(FEATURE_COLUMNS_33) == len(set(FEATURE_COLUMNS_33))


def _make_minimal_train_df():
    peptides = ['AAAAAAAA', 'ACDEFGHIK', 'LMVKQSRTY', 'NPQRSTVWY']
    return pd.DataFrame({
        'peptide': peptides,
        'label': [1, 0, 1, 0],
        'virus': ['EBV', 'HPV', 'EBV', 'HPV'],
    })


def _write_binding_matrix(peptides, path):
    allele_cols = [
        'bind_A0101', 'bind_A0201', 'bind_A0301', 'bind_A1101', 'bind_A2402',
        'bind_B0702', 'bind_B0801', 'bind_B2705', 'bind_B3501', 'bind_B4402',
    ]
    rows = [{'peptide': p, **{c: 0.5 for c in allele_cols}} for p in peptides]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_ap_cache(peptides, path, netchop=0.7, tap=0.6):
    rows = [{'peptide': p, 'netchop_score': netchop, 'tap_score': tap} for p in peptides]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_prepare_features_33_shape(tmp_path):
    """prepare_features_33 returns a DataFrame with 33 columns per peptide."""
    from src.train_classifier import prepare_features_33

    df = _make_minimal_train_df()
    bm_path = tmp_path / "bm.csv"
    _write_binding_matrix(df['peptide'].tolist(), bm_path)
    cache_path = tmp_path / "ap_cache.csv"
    _write_ap_cache(df['peptide'].tolist(), cache_path)

    X = prepare_features_33(df, str(bm_path), str(cache_path))
    assert X.shape == (len(df), 33)
    assert list(X.columns) == FEATURE_COLUMNS_33


def test_prepare_features_33_score_values(tmp_path):
    """netchop_score and tap_score columns reflect cache values exactly."""
    from src.train_classifier import prepare_features_33

    df = _make_minimal_train_df()
    bm_path = tmp_path / "bm.csv"
    _write_binding_matrix(df['peptide'].tolist(), bm_path)
    cache_path = tmp_path / "ap_cache.csv"
    _write_ap_cache(df['peptide'].tolist(), cache_path, netchop=0.42, tap=0.88)

    X = prepare_features_33(df, str(bm_path), str(cache_path))
    np.testing.assert_allclose(X['netchop_score'].values, 0.42, atol=1e-6)
    np.testing.assert_allclose(X['tap_score'].values, 0.88, atol=1e-6)


def test_prepare_features_33_no_nans_when_cache_complete(tmp_path):
    """prepare_features_33 produces no NaNs when all peptides are in cache."""
    from src.train_classifier import prepare_features_33

    df = _make_minimal_train_df()
    bm_path = tmp_path / "bm.csv"
    _write_binding_matrix(df['peptide'].tolist(), bm_path)
    cache_path = tmp_path / "ap_cache.csv"
    _write_ap_cache(df['peptide'].tolist(), cache_path)

    X = prepare_features_33(df, str(bm_path), str(cache_path))
    assert not X.isnull().any().any()


def test_train_classifier_argparse_accepts_mode_33():
    """train_classifier.py argparse does not reject --feature-mode 33."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, 'src/train_classifier.py', '--help'],
        capture_output=True, text=True,
        cwd='.'
    )
    assert '33' in result.stdout
