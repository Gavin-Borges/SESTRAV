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
    assert sorted(extra) == ["netchop_score", "tap_score"]


def test_feature_columns_33_no_duplicates():
    assert len(FEATURE_COLUMNS_33) == len(set(FEATURE_COLUMNS_33))


def _make_minimal_train_df():
    peptides = ["AAAAAAAA", "ACDEFGHIK", "LMVKQSRTY", "NPQRSTVWY"]
    return pd.DataFrame(
        {
            "peptide": peptides,
            "label": [1, 0, 1, 0],
            "virus": ["EBV", "HPV", "EBV", "HPV"],
        }
    )


def _write_binding_matrix(peptides, path):
    allele_cols = [
        "bind_A0101",
        "bind_A0201",
        "bind_A0301",
        "bind_A1101",
        "bind_A2402",
        "bind_B0702",
        "bind_B0801",
        "bind_B2705",
        "bind_B3501",
        "bind_B4402",
    ]
    rows = [{"peptide": p, **{c: 0.5 for c in allele_cols}} for p in peptides]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_ap_cache(peptides, path, netchop=0.7, tap=0.6):
    rows = [{"peptide": p, "netchop_score": netchop, "tap_score": tap} for p in peptides]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_prepare_features_33_shape(tmp_path):
    """prepare_features_33 returns a DataFrame with 33 columns per peptide."""
    from src.train_classifier import prepare_features_33

    df = _make_minimal_train_df()
    bm_path = tmp_path / "bm.csv"
    _write_binding_matrix(df["peptide"].tolist(), bm_path)
    cache_path = tmp_path / "ap_cache.csv"
    _write_ap_cache(df["peptide"].tolist(), cache_path)

    X = prepare_features_33(df, str(bm_path), str(cache_path))
    assert X.shape == (len(df), 33)
    assert list(X.columns) == FEATURE_COLUMNS_33


def test_prepare_features_33_score_values(tmp_path):
    """netchop_score and tap_score columns reflect cache values exactly."""
    from src.train_classifier import prepare_features_33

    df = _make_minimal_train_df()
    bm_path = tmp_path / "bm.csv"
    _write_binding_matrix(df["peptide"].tolist(), bm_path)
    cache_path = tmp_path / "ap_cache.csv"
    _write_ap_cache(df["peptide"].tolist(), cache_path, netchop=0.42, tap=0.88)

    X = prepare_features_33(df, str(bm_path), str(cache_path))
    np.testing.assert_allclose(X["netchop_score"].values, 0.42, atol=1e-6)
    np.testing.assert_allclose(X["tap_score"].values, 0.88, atol=1e-6)


def test_prepare_features_33_no_nans_when_cache_complete(tmp_path):
    """prepare_features_33 produces no NaNs when all peptides are in cache."""
    from src.train_classifier import prepare_features_33

    df = _make_minimal_train_df()
    bm_path = tmp_path / "bm.csv"
    _write_binding_matrix(df["peptide"].tolist(), bm_path)
    cache_path = tmp_path / "ap_cache.csv"
    _write_ap_cache(df["peptide"].tolist(), cache_path)

    X = prepare_features_33(df, str(bm_path), str(cache_path))
    assert not X.isnull().any().any()


def _run_train_classifier(*args):
    """Drive the module's argparse in a subprocess, from the repo root."""
    import pathlib
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-m", "src.train_classifier", *args],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(__file__).resolve().parents[1],
    )


def test_train_classifier_argparse_accepts_mode_33(tmp_path):
    """`--feature-mode 33` is in argparse's `choices`, not merely named in the help text.

    Asserting `"33" in --help` output is NOT sufficient: the rendered help *prose* is
    independent of `choices`, and "33" survives on the --antigen-processing-cache and
    --no-fold-impute help strings alone, so that assertion stays green even if "33" were
    dropped from `choices` and the flag became unusable - the exact "registered in one
    table, missing from another" failure this test exists to catch. Drive a real parse
    instead and require only that argparse accepted the value.
    """
    result = _run_train_classifier(
        "--feature-mode",
        "33",
        "--data",
        str(tmp_path / "does_not_exist.csv"),
        "--model-dir",
        str(tmp_path / "out"),
    )
    combined = result.stdout + result.stderr
    assert "invalid choice" not in combined, combined


def test_train_classifier_argparse_rejects_unregistered_mode_34(tmp_path):
    """Negative control for the test above.

    Without this, `assert "invalid choice" not in output` could pass for a reason having
    nothing to do with `choices` - a parser that never ran, or a renamed argparse message.
    Mode 34 is deliberately not registered, so it must be rejected.
    """
    result = _run_train_classifier(
        "--feature-mode",
        "34",
        "--data",
        str(tmp_path / "does_not_exist.csv"),
        "--model-dir",
        str(tmp_path / "out"),
    )
    assert "invalid choice" in (result.stdout + result.stderr)


def test_batch_experiment_runner_accepts_mode_33():
    from scripts.batch_experiment_runner import VALID_FEATURE_MODES

    assert 33 in VALID_FEATURE_MODES
