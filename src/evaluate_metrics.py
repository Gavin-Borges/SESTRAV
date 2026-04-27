"""
SESTRAV Shared Evaluation Metrics Module

Provides the 4 standard metrics used to evaluate ALL SESTRAV models:
RF, XGBoost, ANN, binding-only baseline, PredIG, and PRIME.

Metric definitions and units:
  auc_roc  : Area Under the ROC Curve, unitless [0, 1].
             0.5 = random, 1.0 = perfect discrimination.
  auc_pr   : Area Under the Precision-Recall Curve, unitless [0, 1].
             PRIMARY metric — robust to class imbalance.
             Baseline equals the positive class prevalence.
  issr_10  : Immune-Stimulating Success Rate at top 10%, unitless [0, 1].
             Fraction of true positives among the highest-scoring 10% of
             predictions.  Matches PredIG methodology (Farriol-Duran 2025).
  issr_25  : Same as issr_10, but for the top 25%.

Label convention: y_true binary (0 = non-immunogenic, 1 = immunogenic).
Score convention: y_scores continuous — higher = more immunogenic.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def issr_at_k(y_true, y_scores, k):
    """
    Immune-Stimulating Success Rate at top K%.
    Fraction of true positives among the top K% of predictions,
    matching the metric used by PredIG (Farriol-Duran et al., 2025).
    """
    n = len(y_scores)
    top_k_count = max(1, int(n * k / 100))
    top_k_indices = np.argsort(y_scores)[-top_k_count:]
    return float(np.mean(np.array(y_true)[top_k_indices]))


def evaluate(y_true, y_scores):
    """
    Compute all SESTRAV evaluation metrics.

    Args:
        y_true:   array-like of binary labels (0 = non-immunogenic, 1 = immunogenic)
        y_scores: array-like of predicted scores/probabilities (higher = more immunogenic)

    Returns:
        dict with keys: auc_roc, auc_pr, issr_10, issr_25
        Returns NaN for AUC metrics if only one class is present in y_true.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        auc_roc = float('nan')
        auc_pr = float('nan')
    else:
        auc_roc = roc_auc_score(y_true, y_scores)
        auc_pr = average_precision_score(y_true, y_scores)

    return {
        'auc_roc':  auc_roc,
        'auc_pr':   auc_pr,
        'issr_10':  issr_at_k(y_true, y_scores, 10),
        'issr_25':  issr_at_k(y_true, y_scores, 25),
    }
