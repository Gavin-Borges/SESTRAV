from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.compute_binding_zerofill_attribution import (
    KEY_COLUMNS,
    align_oof,
    auc_attribution,
    population_mask,
)


def test_population_flags_use_corrected_repository_definitions() -> None:
    frame = pd.DataFrame(
        {
            "label": [1, 1, 0, 0, 0],
            "negative_origin": [None, "iedb_positive", "iedb_api", "tested_negative", "decoy"],
            "virus": ["CMV", "CMV", "CMV", "CMV", "CMV"],
        }
    )
    assert population_mask(frame, "def_a").tolist() == [True, True, True, False, False]
    assert population_mask(frame, "def_b").tolist() == [True, True, True, True, False]
    assert population_mask(frame, "pooled").tolist() == [True] * 5


def test_synthetic_pair_decomposition_has_expected_cross_group_delta() -> None:
    labels = pd.Series([1, 0, 1, 0])
    scores = pd.Series([0.6, 0.4, 0.2, 0.1])
    covered = pd.Series([True, True, False, False])
    result = auc_attribution(labels, scores, covered)
    assert result["pooled_auc_roc"] == pytest.approx(0.75)
    assert result["within_coverage_auc_roc"] == pytest.approx(1.0)
    assert result["cross_coverage_delta_auc_roc"] == pytest.approx(-0.25)
    assert result["coverage_is_degenerate"] is False


def test_single_coverage_group_reports_a_delta_that_was_not_computed() -> None:
    labels = pd.Series([1, 0, 1, 0])
    scores = pd.Series([0.6, 0.4, 0.2, 0.1])
    all_covered = auc_attribution(labels, scores, pd.Series([True] * 4))
    none_covered = auc_attribution(labels, scores, pd.Series([False] * 4))

    for result in (all_covered, none_covered):
        # The delta is set to zero because no cross-coverage pair exists, so the
        # flag is what separates it from a zero a comparison produced.
        assert result["cross_coverage_delta_auc_roc"] == 0.0
        assert result["coverage_is_degenerate"] is True
        assert result["within_coverage_auc_roc"] == result["pooled_auc_roc"]


def _oof(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "peptide": ["AAA", "BBB"],
            "virus": ["V", "V"],
            "strain": [np.nan, "S"],
            "protein": ["P", "P"],
            "negative_origin": [np.nan, "iedb_api"],
            "hla_allele": ["HLA-A*01:01", "HLA-A*01:01"],
            "label": [1, 0],
            "fold": [1, 1],
            "score": scores,
        }
    )


def test_alignment_is_keyed_and_rejects_missing_rows() -> None:
    left = _oof([0.9, 0.1])
    right = _oof([0.8, 0.2]).iloc[::-1].reset_index(drop=True)
    aligned = align_oof(left, right)
    assert aligned["mode10_score"].tolist() == [0.8, 0.2]
    assert all(column in aligned for column in KEY_COLUMNS)

    with pytest.raises(ValueError, match="OOF key mismatch"):
        align_oof(left, right.iloc[:1])
