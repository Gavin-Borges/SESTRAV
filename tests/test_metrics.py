"""
Unit tests for the SESTRAV shared evaluation metrics module.

Run from repo root:
    python -m pytest tests/test_metrics.py -v
    python -m tests.test_metrics           (standalone)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from src.evaluate_metrics import (
    evaluate,
    issr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    summarize_fold_metrics,
)


def test_perfect_predictions():
    """Perfect separation should yield AUC-ROC = 1.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    result = evaluate(y_true, y_scores)
    assert result["auc_roc"] == 1.0, f"Expected AUC-ROC=1.0, got {result['auc_roc']}"
    assert result["auc_pr"] == 1.0, f"Expected AUC-PR=1.0, got {result['auc_pr']}"


def test_random_predictions():
    """Random predictions should give AUC-ROC near 0.5."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_scores = np.random.rand(1000)
    result = evaluate(y_true, y_scores)
    assert 0.40 < result["auc_roc"] < 0.60, f"Random AUC-ROC out of range: {result['auc_roc']}"


def test_issr_perfect():
    """ISSR@10 should be 1.0 when every peptide in the top 10% is a true positive."""
    y_true = np.array([0] * 90 + [1] * 10)
    y_scores = np.array(list(range(100)))  # positives have highest scores
    assert issr_at_k(y_true, y_scores, 10) == 1.0


def test_metric_keys():
    """evaluate() must return at least the 4 core metric keys plus extended metrics."""
    y_true = np.array([0, 1, 0, 1])
    y_scores = np.array([0.2, 0.8, 0.3, 0.9])
    result = evaluate(y_true, y_scores)
    core_keys = {"auc_roc", "auc_pr", "issr_10", "issr_25"}
    extended_keys = {"precision_10", "recall_10", "ndcg_10", "precision_25", "recall_25", "ndcg_25"}
    band_keys = {
        f"{metric}_{k}_band_{side}"
        for metric in ("issr", "precision", "recall")
        for k in (10, 25)
        for side in ("lo", "hi")
    }
    assert core_keys.issubset(set(result.keys())), (
        f"Missing core keys: {core_keys - set(result.keys())}"
    )
    assert extended_keys.issubset(set(result.keys())), (
        f"Missing extended keys: {extended_keys - set(result.keys())}"
    )
    assert band_keys.issubset(set(result.keys())), (
        f"Missing band keys: {band_keys - set(result.keys())}"
    )


# -- Tie band (achievable [lo, hi] range at the top-K cutoff) ----------------
#
# issr_at_k/recall_at_k select the top K% via a bare argsort with no tie-break,
# so a tie at the cutoff makes the point estimate order-dependent (see the
# characterization tests above this line's sibling work on tie behaviour).
# These tests check the exact band evaluate() now returns alongside each point
# estimate, computed independently of argsort's own (platform-dependent) tie
# resolution.


def test_tie_band_brackets_the_point_estimate_and_matches_hand_computed_bounds():
    """1 row clearly above the cutoff; 3 tied rows compete for 1 remaining slot.

    n=8, k=25 -> top_k_count=2. The row at 0.9 is certain-in. The tied group at
    0.6 (1 positive, 2 negative) competes for the other slot: best case takes
    the tied positive (2/2), worst case takes a tied negative (1/2).
    """
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    y_scores = np.array([0.9, 0.6, 0.6, 0.6, 0.3, 0.2, 0.1, 0.05])
    result = evaluate(y_true, y_scores)

    assert result["issr_25_band_lo"] == pytest.approx(0.5)
    assert result["issr_25_band_hi"] == pytest.approx(1.0)
    assert result["issr_25_band_lo"] <= result["issr_25"] <= result["issr_25_band_hi"]
    # precision_at_k delegates to issr_at_k, so it shares issr's band exactly.
    assert result["precision_25_band_lo"] == result["issr_25_band_lo"]
    assert result["precision_25_band_hi"] == result["issr_25_band_hi"]


def test_recall_tie_band_uses_positive_count_as_denominator():
    """Same fixture as above; recall's denominator is total positives (2)."""
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    y_scores = np.array([0.9, 0.6, 0.6, 0.6, 0.3, 0.2, 0.1, 0.05])
    result = evaluate(y_true, y_scores)

    assert result["recall_25_band_lo"] == pytest.approx(0.5)  # 1 of 2 positives
    assert result["recall_25_band_hi"] == pytest.approx(1.0)  # 2 of 2 positives
    assert result["recall_25_band_lo"] <= result["recall_25"] <= result["recall_25_band_hi"]


def test_band_has_zero_width_when_no_tie_at_the_cutoff():
    """Distinct scores at the cutoff: the band collapses to the point estimate."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    result = evaluate(y_true, y_scores)

    assert result["issr_10_band_lo"] == result["issr_10_band_hi"] == result["issr_10"]
    assert result["recall_10_band_lo"] == result["recall_10_band_hi"] == result["recall_10"]


def test_issr_at_k_empty():
    assert issr_at_k([], [], 10) == 0.0


def test_precision_at_k_equals_issr():
    y_true = np.array([0, 1, 0, 1])
    y_scores = np.array([0.1, 0.9, 0.2, 0.8])
    assert precision_at_k(y_true, y_scores, 50) == issr_at_k(y_true, y_scores, 50)


def test_recall_at_k():
    # Two positives at the top -> top-50% recall captures both.
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert recall_at_k(y_true, y_scores, 50) == 1.0


def test_ndcg_at_k_perfect_and_degenerate():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert 0.0 < ndcg_at_k(y_true, y_scores, 100) <= 1.0
    # Fewer than two items -> NaN.
    assert np.isnan(ndcg_at_k([1], [0.5], 10))


def test_evaluate_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        evaluate([], [])


def test_evaluate_single_class_returns_nan_auc():
    result = evaluate(np.array([1, 1, 1]), np.array([0.2, 0.5, 0.9]))
    assert np.isnan(result["auc_roc"])
    assert np.isnan(result["auc_pr"])
    # Ranking metrics are still computable.
    assert not np.isnan(result["issr_10"])


def test_summarize_fold_metrics():
    folds = [
        {"auc_roc": 0.8, "auc_pr": 0.7},
        {"auc_roc": 0.9, "auc_pr": 0.6},
    ]
    avg, std = summarize_fold_metrics(folds)
    assert avg["auc_roc"] == pytest.approx(0.85)
    assert std["auc_pr"] == pytest.approx(0.05)


def test_summarize_fold_metrics_empty_raises():
    with pytest.raises(ValueError, match="No fold metrics"):
        summarize_fold_metrics([])


if __name__ == "__main__":
    test_perfect_predictions()
    test_random_predictions()
    test_issr_perfect()
    test_metric_keys()
    print("All evaluation metric tests passed.")
