"""Tests for FEATURE_COLUMNS_31 definition and prepare_features_31 in train_classifier."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS_30,
    FEATURE_COLUMNS_31,
    PHYSICO_COLUMNS,
    BINDING_ALLELE_COLUMNS,
)


def test_feature_columns_31_length():
    assert len(FEATURE_COLUMNS_31) == 31


def test_feature_columns_31_is_superset_of_30():
    assert set(FEATURE_COLUMNS_30).issubset(set(FEATURE_COLUMNS_31))


def test_feature_columns_31_adds_peptide_length():
    extra = [c for c in FEATURE_COLUMNS_31 if c not in FEATURE_COLUMNS_30]
    assert extra == ["peptide_length"]


def test_feature_columns_31_no_duplicates():
    assert len(FEATURE_COLUMNS_31) == len(set(FEATURE_COLUMNS_31))


def test_feature_columns_31_physico_and_binding_present():
    for col in PHYSICO_COLUMNS:
        assert col in FEATURE_COLUMNS_31, f"Missing physico column: {col}"
    for col in BINDING_ALLELE_COLUMNS:
        assert col in FEATURE_COLUMNS_31, f"Missing binding column: {col}"


def _make_minimal_train_df():
    """Minimal dataset with required columns for prepare_features_31."""
    peptides = ["AAAAAAAA", "ACDEFGHIK", "LMVKQSRTY", "NPQRSTVWY"]
    return pd.DataFrame(
        {
            "peptide": peptides,
            "label": [1, 0, 1, 0],
            "virus": ["EBV", "HPV", "EBV", "HPV"],
        }
    )


def _make_mock_binding_matrix(peptides):
    """Minimal binding matrix with required 10 allele columns."""
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
    return pd.DataFrame(rows)


def test_prepare_features_31_shape(tmp_path):
    """prepare_features_31 returns a DataFrame with 31 columns per peptide."""
    from src.train_classifier import prepare_features_31

    df = _make_minimal_train_df()
    bm = _make_mock_binding_matrix(df["peptide"].tolist())
    bm_path = tmp_path / "binding_matrix.csv"
    bm.to_csv(bm_path, index=False)

    X = prepare_features_31(df, str(bm_path))
    assert X.shape == (len(df), 31)
    assert list(X.columns) == FEATURE_COLUMNS_31


def test_prepare_features_31_peptide_length_values(tmp_path):
    """peptide_length column equals actual peptide lengths."""
    from src.train_classifier import prepare_features_31

    df = _make_minimal_train_df()
    bm = _make_mock_binding_matrix(df["peptide"].tolist())
    bm_path = tmp_path / "binding_matrix.csv"
    bm.to_csv(bm_path, index=False)

    X = prepare_features_31(df, str(bm_path))
    expected_lengths = df["peptide"].str.len().values
    np.testing.assert_array_equal(X["peptide_length"].values, expected_lengths)


def test_prepare_features_31_no_nans(tmp_path):
    """prepare_features_31 produces no NaN values for well-formed input."""
    from src.train_classifier import prepare_features_31

    df = _make_minimal_train_df()
    bm = _make_mock_binding_matrix(df["peptide"].tolist())
    bm_path = tmp_path / "binding_matrix.csv"
    bm.to_csv(bm_path, index=False)

    X = prepare_features_31(df, str(bm_path))
    assert not X.isnull().any().any(), "Unexpected NaNs in 31-feature matrix"


def test_prepare_features_31_missing_binding_matrix_raises():
    """prepare_features_31 raises ValueError when binding matrix is missing a required allele."""
    from src.train_classifier import prepare_features_31

    df = _make_minimal_train_df()
    # Binding matrix with only 5 allele columns (insufficient)
    incomplete_bm = pd.DataFrame([{"peptide": p, "bind_A0101": 0.5} for p in df["peptide"]])
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        incomplete_bm.to_csv(f, index=False)
        tmp_path = f.name
    try:
        with pytest.raises(ValueError, match="only.*10 expected allele columns"):
            prepare_features_31(df, tmp_path)
    finally:
        os.unlink(tmp_path)


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


def test_train_classifier_argparse_accepts_mode_31(tmp_path):
    """`--feature-mode 31` is in argparse's `choices`, not merely named in the help text.

    Asserting `"31" in --help` output is NOT sufficient: the rendered help *prose* is
    independent of `choices`, and "31" survives on the --feature-mode description line
    ("31 canonical") and on the unrelated --cv-group-by help, so that assertion stays
    green even if "31" were dropped from `choices` and the flag became unusable - the
    exact "registered in one table, missing from another" failure this test exists to
    catch. Drive a real parse instead and require only that argparse accepted the value.
    """
    result = _run_train_classifier(
        "--feature-mode",
        "31",
        "--data",
        str(tmp_path / "does_not_exist.csv"),
        "--model-dir",
        str(tmp_path / "out"),
    )
    combined = result.stdout + result.stderr
    assert "invalid choice" not in combined, combined


def test_train_classifier_argparse_rejects_unregistered_mode_32(tmp_path):
    """Negative control for the test above.

    Without this, `assert "invalid choice" not in output` could pass for a reason having
    nothing to do with `choices` - a parser that never ran, or a renamed argparse message.
    Mode 32 is deliberately not registered, so it must be rejected.
    """
    result = _run_train_classifier(
        "--feature-mode",
        "32",
        "--data",
        str(tmp_path / "does_not_exist.csv"),
        "--model-dir",
        str(tmp_path / "out"),
    )
    assert "invalid choice" in (result.stdout + result.stderr)


def test_batch_experiment_runner_accepts_mode_31():
    from scripts.batch_experiment_runner import VALID_FEATURE_MODES

    assert 31 in VALID_FEATURE_MODES
