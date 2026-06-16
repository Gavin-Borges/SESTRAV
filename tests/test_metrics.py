"""
Unit tests for the SESTRAV shared evaluation metrics module.

Run from repo root:
    python -m pytest tests/test_metrics.py -v
    python -m tests.test_metrics           (standalone)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    assert result['auc_roc'] == 1.0, f"Expected AUC-ROC=1.0, got {result['auc_roc']}"
    assert result['auc_pr'] == 1.0, f"Expected AUC-PR=1.0, got {result['auc_pr']}"


def test_random_predictions():
    """Random predictions should give AUC-ROC near 0.5."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_scores = np.random.rand(1000)
    result = evaluate(y_true, y_scores)
    assert 0.40 < result['auc_roc'] < 0.60, f"Random AUC-ROC out of range: {result['auc_roc']}"


def test_issr_perfect():
    """ISSR@10 should be 1.0 when all true positives are in top 10%."""
    y_true = np.array([0]*90 + [1]*10)
    y_scores = np.array(list(range(100)))  # positives have highest scores
    assert issr_at_k(y_true, y_scores, 10) == 1.0


def test_metric_keys():
    """evaluate() must return at least the 4 core metric keys plus extended metrics."""
    y_true = np.array([0, 1, 0, 1])
    y_scores = np.array([0.2, 0.8, 0.3, 0.9])
    result = evaluate(y_true, y_scores)
    core_keys = {'auc_roc', 'auc_pr', 'issr_10', 'issr_25'}
    extended_keys = {'precision_10', 'recall_10', 'ndcg_10',
                     'precision_25', 'recall_25', 'ndcg_25'}
    assert core_keys.issubset(set(result.keys())), f"Missing core keys: {core_keys - set(result.keys())}"
    assert extended_keys.issubset(set(result.keys())), f"Missing extended keys: {extended_keys - set(result.keys())}"


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
