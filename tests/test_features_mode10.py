"""Tests for FEATURE_COLUMNS_10 and prepare_features_10 in train_classifier.

Mode 10 is the binding-only ablation: the exact converse of mode 21 (physico-only,
no binding). It isolates the 10 per-allele MHCflurry presentation scores with no
physicochemistry and no peptide_length, so that the contribution of binding alone
can be measured on the same peptide-grouped CV path as the other arms.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS_10, BINDING_ALLELE_COLUMNS


def test_feature_columns_10_length():
    assert len(FEATURE_COLUMNS_10) == 10


def test_feature_columns_10_equals_binding_allele_columns():
    assert FEATURE_COLUMNS_10 == BINDING_ALLELE_COLUMNS


def test_feature_columns_10_no_duplicates():
    assert len(FEATURE_COLUMNS_10) == len(set(FEATURE_COLUMNS_10))


def test_feature_columns_10_has_no_physico_or_length():
    assert "peptide_length" not in FEATURE_COLUMNS_10
    for col in FEATURE_COLUMNS_10:
        assert col.startswith("bind_"), f"Unexpected non-binding column: {col}"


def test_mode_10_and_mode_21_partition_mode_31_exactly():
    """The "exact converse" claim in src/features.py is checked, not just asserted in prose.

    Mode 10 (binding-only) and mode 21 (physico-only, TRAIN_FEATURE_COLUMNS) must be
    disjoint and together account for every mode-31 column. If this ever fails, the two
    ablation arms overlap or leave a gap, and neither one measures what its name claims.
    """
    from src.features import FEATURE_COLUMNS_31, TRAIN_FEATURE_COLUMNS

    mode_10, mode_21 = set(FEATURE_COLUMNS_10), set(TRAIN_FEATURE_COLUMNS)
    assert mode_10 & mode_21 == set(), f"arms overlap on: {sorted(mode_10 & mode_21)}"
    assert mode_10 | mode_21 == set(FEATURE_COLUMNS_31), (
        f"arms do not sum to mode 31; missing: {sorted(set(FEATURE_COLUMNS_31) - (mode_10 | mode_21))}, "
        f"extra: {sorted((mode_10 | mode_21) - set(FEATURE_COLUMNS_31))}"
    )
    assert len(FEATURE_COLUMNS_10) + len(TRAIN_FEATURE_COLUMNS) == len(FEATURE_COLUMNS_31)


def _make_minimal_train_df():
    peptides = ["AAAAAAAA", "ACDEFGHIK", "LMVKQSRTY", "NPQRSTVWY"]
    return pd.DataFrame(
        {
            "peptide": peptides,
            "label": [1, 0, 1, 0],
            "virus": ["EBV", "HPV", "EBV", "HPV"],
        }
    )


def _make_mock_binding_matrix(peptides):
    rows = [{"peptide": p, **{c: 0.5 for c in BINDING_ALLELE_COLUMNS}} for p in peptides]
    return pd.DataFrame(rows)


def test_prepare_features_10_shape(tmp_path):
    """prepare_features_10 returns a DataFrame with 10 columns per peptide."""
    from src.train_classifier import prepare_features_10

    df = _make_minimal_train_df()
    bm = _make_mock_binding_matrix(df["peptide"].tolist())
    bm_path = tmp_path / "binding_matrix.csv"
    bm.to_csv(bm_path, index=False)

    X = prepare_features_10(df, str(bm_path))
    assert X.shape == (len(df), 10)
    assert list(X.columns) == FEATURE_COLUMNS_10


def test_prepare_features_10_values_match_binding_matrix(tmp_path):
    """Binding values pass through unchanged from the binding matrix."""
    from src.train_classifier import prepare_features_10

    df = _make_minimal_train_df()
    bm = _make_mock_binding_matrix(df["peptide"].tolist())
    bm_path = tmp_path / "binding_matrix.csv"
    bm.to_csv(bm_path, index=False)

    X = prepare_features_10(df, str(bm_path))
    assert (X.to_numpy() == 0.5).all()


def test_prepare_features_10_preserves_per_allele_alignment(tmp_path):
    """Each allele's score lands under its OWN column, not merely under some binding column.

    The sibling test above fills every allele with the same 0.5, so it cannot distinguish
    a correct name-based join from an arbitrary column permutation. Here each allele gets a
    distinct value and the matrix is written to disk with its allele columns REVERSED, so a
    positional join would transpose them and every assertion below would fail.
    """
    from src.train_classifier import prepare_features_10

    df = _make_minimal_train_df()
    expected = {col: 0.01 * (i + 1) for i, col in enumerate(BINDING_ALLELE_COLUMNS)}
    bm = pd.DataFrame([{"peptide": p, **expected} for p in df["peptide"]])
    bm = bm[["peptide"] + list(reversed(BINDING_ALLELE_COLUMNS))]
    bm_path = tmp_path / "binding_matrix.csv"
    bm.to_csv(bm_path, index=False)

    X = prepare_features_10(df, str(bm_path))
    for col in FEATURE_COLUMNS_10:
        assert (X[col] == expected[col]).all(), f"{col} carries the wrong allele's score"


def test_prepare_features_10_missing_peptide_zero_filled(tmp_path):
    """A peptide absent from the binding matrix gets zero-filled, not dropped or NaN."""
    from src.train_classifier import prepare_features_10

    df = _make_minimal_train_df()
    bm = _make_mock_binding_matrix(df["peptide"].tolist()[:-1])  # drop the last peptide
    bm_path = tmp_path / "binding_matrix.csv"
    bm.to_csv(bm_path, index=False)

    X = prepare_features_10(df, str(bm_path))
    assert X.shape == (len(df), 10)
    assert not X.isnull().any().any()
    np.testing.assert_array_equal(X.iloc[-1].to_numpy(), np.zeros(10))


def test_prepare_features_10_missing_binding_matrix_raises():
    """prepare_features_10 raises ValueError when the binding matrix is missing an allele."""
    from src.train_classifier import prepare_features_10
    import tempfile
    import os

    df = _make_minimal_train_df()
    incomplete_bm = pd.DataFrame([{"peptide": p, "bind_A0101": 0.5} for p in df["peptide"]])
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        incomplete_bm.to_csv(f, index=False)
        tmp_path = f.name
    try:
        with pytest.raises(ValueError, match="only.*10 expected allele columns"):
            prepare_features_10(df, tmp_path)
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


def test_train_classifier_argparse_accepts_mode_10(tmp_path):
    """`--feature-mode 10` is in argparse's `choices`, not merely named in the help text.

    Asserting `"10" in --help` output is NOT sufficient: the help *prose* itself reads
    "Feature mode: 10 (binding-only, no physicochemistry ...)", so that assertion stays
    green even if "10" were dropped from `choices` and the flag became unusable - the
    exact "registered in one table, missing from another" failure this test exists to
    catch. Drive a real parse instead and require only that argparse accepted the value.
    """
    result = _run_train_classifier(
        "--feature-mode",
        "10",
        "--data",
        str(tmp_path / "does_not_exist.csv"),
        "--model-dir",
        str(tmp_path / "out"),
    )
    combined = result.stdout + result.stderr
    assert "invalid choice" not in combined, combined


def test_train_classifier_argparse_rejects_unregistered_mode(tmp_path):
    """Negative control for the test above.

    Without this, `assert "invalid choice" not in output` could pass for a reason having
    nothing to do with `choices` - a parser that never ran, or a renamed argparse message.
    Mode 11 is deliberately not registered, so it must be rejected.
    """
    result = _run_train_classifier(
        "--feature-mode",
        "11",
        "--data",
        str(tmp_path / "does_not_exist.csv"),
        "--model-dir",
        str(tmp_path / "out"),
    )
    assert "invalid choice" in (result.stdout + result.stderr)


def test_batch_experiment_runner_accepts_mode_10():
    from scripts.batch_experiment_runner import VALID_FEATURE_MODES

    assert 10 in VALID_FEATURE_MODES
