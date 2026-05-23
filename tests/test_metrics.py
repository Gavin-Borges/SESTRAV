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
from src.evaluate_metrics import evaluate, issr_at_k


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


if __name__ == "__main__":
    test_perfect_predictions()
    test_random_predictions()
    test_issr_perfect()
    test_metric_keys()
    print("All evaluation metric tests passed.")
