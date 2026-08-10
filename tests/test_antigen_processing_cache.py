"""Tests for load_antigen_processing_cache in src/features.py."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    load_antigen_processing_cache,
    antigen_processing_cache_medians,
    ANTIGEN_PROCESSING_COLUMNS,
)


def _write_cache(path, rows):
    pd.DataFrame(rows, columns=["peptide", "netchop_score", "tap_score"]).to_csv(path, index=False)


def test_load_cache_adds_score_columns(tmp_path):
    """Cache loader attaches netchop_score and tap_score to the DataFrame."""
    cache_path = tmp_path / "cache.csv"
    _write_cache(
        cache_path,
        [
            ("AAAAAAAAAA", 0.8, 0.6),
            ("CCCCCCCCC", 0.3, 0.4),
        ],
    )
    df = pd.DataFrame({"peptide": ["AAAAAAAAAA", "CCCCCCCCC"]})
    result = load_antigen_processing_cache(str(cache_path), df)
    assert "netchop_score" in result.columns
    assert "tap_score" in result.columns
    np.testing.assert_allclose(result["netchop_score"].values, [0.8, 0.3], atol=1e-6)
    np.testing.assert_allclose(result["tap_score"].values, [0.6, 0.4], atol=1e-6)


def test_load_cache_median_imputation(tmp_path):
    """Peptides absent from cache receive median imputation."""
    cache_path = tmp_path / "cache.csv"
    _write_cache(
        cache_path,
        [
            ("AAAAAAAAAA", 0.6, 0.4),
            ("CCCCCCCCC", 0.8, 0.8),
        ],
    )
    df = pd.DataFrame({"peptide": ["AAAAAAAAAA", "UNKNOWNPEP"]})
    result = load_antigen_processing_cache(str(cache_path), df)

    expected_netchop_median = 0.7  # median of [0.6, 0.8]
    expected_tap_median = 0.6  # median of [0.4, 0.8]
    assert result.loc[0, "netchop_score"] == pytest.approx(0.6)
    assert result.loc[1, "netchop_score"] == pytest.approx(expected_netchop_median)
    assert result.loc[1, "tap_score"] == pytest.approx(expected_tap_median)


def test_load_cache_no_nans_after_imputation(tmp_path):
    """Result DataFrame has no NaN values after median imputation."""
    cache_path = tmp_path / "cache.csv"
    _write_cache(
        cache_path,
        [
            ("AAAAAAAAAA", 0.5, 0.5),
        ],
    )
    df = pd.DataFrame({"peptide": ["AAAAAAAAAA", "MISSING_PEP"]})
    result = load_antigen_processing_cache(str(cache_path), df)
    assert not result[["netchop_score", "tap_score"]].isnull().any().any()


def test_load_cache_preserves_original_columns(tmp_path):
    """Original DataFrame columns are preserved alongside score columns."""
    cache_path = tmp_path / "cache.csv"
    _write_cache(cache_path, [("AAAAAAAAAA", 0.5, 0.5)])
    df = pd.DataFrame({"peptide": ["AAAAAAAAAA"], "label": [1], "virus": ["EBV"]})
    result = load_antigen_processing_cache(str(cache_path), df)
    assert "label" in result.columns
    assert "virus" in result.columns
    assert result.loc[0, "label"] == 1


def test_load_cache_deduplicates_cache(tmp_path):
    """Duplicate peptides in cache are deduplicated (first-seen wins)."""
    cache_path = tmp_path / "cache.csv"
    _write_cache(
        cache_path,
        [
            ("AAAAAAAAAA", 0.9, 0.7),
            ("AAAAAAAAAA", 0.1, 0.1),  # duplicate - should be ignored
        ],
    )
    df = pd.DataFrame({"peptide": ["AAAAAAAAAA"]})
    result = load_antigen_processing_cache(str(cache_path), df)
    assert result.loc[0, "netchop_score"] == pytest.approx(0.9)


def test_load_cache_empty_cache_produces_nan(tmp_path):
    """An empty cache produces NaN scores (median imputation of empty series = NaN)."""
    cache_path = tmp_path / "cache.csv"
    pd.DataFrame(columns=["peptide", "netchop_score", "tap_score"]).to_csv(cache_path, index=False)
    df = pd.DataFrame({"peptide": ["AAAAAAAAAA"]})
    result = load_antigen_processing_cache(str(cache_path), df)
    # Median of empty series is NaN; imputed value is therefore also NaN
    assert result["netchop_score"].isna().all()
    assert result["tap_score"].isna().all()


# ---------------------------------------------------------------------------
# impute=False (Phase 0 step 6: in-fold imputation support)
# ---------------------------------------------------------------------------


def test_load_cache_impute_false_leaves_missing_as_nan(tmp_path):
    cache_path = tmp_path / "cache.csv"
    _write_cache(cache_path, [("AAAAAAAAAA", 0.6, 0.4)])
    df = pd.DataFrame({"peptide": ["AAAAAAAAAA", "UNKNOWNPEP"]})
    result = load_antigen_processing_cache(str(cache_path), df, impute=False)
    assert result.loc[0, "netchop_score"] == pytest.approx(0.6)
    assert pd.isna(result.loc[1, "netchop_score"])
    assert pd.isna(result.loc[1, "tap_score"])


def test_load_cache_impute_true_is_still_the_default(tmp_path):
    cache_path = tmp_path / "cache.csv"
    _write_cache(cache_path, [("AAAAAAAAAA", 0.6, 0.4), ("CCCCCCCCC", 0.8, 0.8)])
    df = pd.DataFrame({"peptide": ["UNKNOWNPEP"]})
    default_call = load_antigen_processing_cache(str(cache_path), df)
    explicit_call = load_antigen_processing_cache(str(cache_path), df, impute=True)
    pd.testing.assert_frame_equal(default_call, explicit_call)


def test_antigen_processing_cache_medians(tmp_path):
    cache_path = tmp_path / "cache.csv"
    _write_cache(cache_path, [("AAAAAAAAAA", 0.6, 0.4), ("CCCCCCCCC", 0.8, 0.8)])
    medians = antigen_processing_cache_medians(str(cache_path))
    assert set(medians) == set(ANTIGEN_PROCESSING_COLUMNS)
    assert medians["netchop_score"] == pytest.approx(0.7)
    assert medians["tap_score"] == pytest.approx(0.6)
