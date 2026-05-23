"""
SESTRAV Shared Evaluation Metrics Module

Provides the standard metrics used to evaluate ALL SESTRAV models:
RF, XGBoost, ANN, GNN, binding-only baseline, PredIG, and PRIME.

Core metrics (always returned by evaluate()):
  auc_roc  : Area Under the ROC Curve, unitless [0, 1].
             0.5 = random, 1.0 = perfect discrimination.
  auc_pr   : Area Under the Precision-Recall Curve, unitless [0, 1].
             PRIMARY metric — robust to class imbalance.
             Baseline equals the positive class prevalence.
  issr_10  : Immune-Stimulating Success Rate at top 10%, unitless [0, 1].
             Fraction of true positives among the highest-scoring 10% of
             predictions.  Matches PredIG methodology (Farriol-Duran 2025).
  issr_25  : Same as issr_10, but for the top 25%.

Extended metrics (also returned by evaluate()):
  precision_10, recall_10, ndcg_10 : Precision, Recall, and NDCG at top 10%.
  precision_25, recall_25, ndcg_25 : Same for top 25%.

Cross-validation helper:
  summarize_fold_metrics() : Mean +/- std across k-fold CV results.

Label convention: y_true binary (0 = non-immunogenic, 1 = immunogenic).
Score convention: y_scores continuous — higher = more immunogenic.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score


def issr_at_k(y_true, y_scores, k):
    """
    Immune-Stimulating Success Rate at top K%.
    Fraction of true positives among the top K% of predictions,
    matching the metric used by PredIG (Farriol-Duran et al., 2025).
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    if len(y_true) == 0:
        return 0.0

    n = len(y_scores)
    top_k_count = max(1, int(n * k / 100))
    top_k_indices = np.argsort(y_scores)[-top_k_count:]
    return float(np.mean(y_true[top_k_indices]))


def precision_at_k(y_true, y_scores, k):
    """Precision among the top K% highest-scoring predictions.

    For binary labels, this is identical to issr_at_k.
    """
    return issr_at_k(y_true, y_scores, k)


def recall_at_k(y_true, y_scores, k):
    """Recall captured within the top K% highest-scoring predictions."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_pos = max(1, int(y_true.sum()))
    n = len(y_scores)
    top_k_count = max(1, int(n * k / 100))
    top_k_indices = np.argsort(y_scores)[-top_k_count:]
    return float(y_true[top_k_indices].sum() / n_pos)


def ndcg_at_k(y_true, y_scores, k):
    """Normalized Discounted Cumulative Gain at top K%."""
    y_true = np.asarray(y_true).reshape(1, -1)
    y_scores = np.asarray(y_scores).reshape(1, -1)
    n_items = y_true.shape[1]
    top_k_count = max(1, int(n_items * k / 100))
    return float(ndcg_score(y_true, y_scores, k=top_k_count))


def evaluate(y_true, y_scores):
    """
    Compute all SESTRAV evaluation metrics.

    Args:
        y_true:   array-like of binary labels (0 = non-immunogenic, 1 = immunogenic)
        y_scores: array-like of predicted scores/probabilities (higher = more immunogenic)

    Returns:
        dict with keys: auc_roc, auc_pr, issr_10, issr_25,
        precision_10, recall_10, ndcg_10, precision_25, recall_25, ndcg_25.
        Returns NaN for AUC metrics if only one class is present in y_true.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate empty arrays")

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        auc_roc = float('nan')
        auc_pr = float('nan')
    else:
        auc_roc = roc_auc_score(y_true, y_scores)
        auc_pr = average_precision_score(y_true, y_scores)

    return {
        'auc_roc':      auc_roc,
        'auc_pr':       auc_pr,
        'issr_10':      issr_at_k(y_true, y_scores, 10),
        'issr_25':      issr_at_k(y_true, y_scores, 25),
        'precision_10': precision_at_k(y_true, y_scores, 10),
        'recall_10':    recall_at_k(y_true, y_scores, 10),
        'ndcg_10':      ndcg_at_k(y_true, y_scores, 10),
        'precision_25': precision_at_k(y_true, y_scores, 25),
        'recall_25':    recall_at_k(y_true, y_scores, 25),
        'ndcg_25':      ndcg_at_k(y_true, y_scores, 25),
    }


def summarize_fold_metrics(fold_metrics_list):
    """Compute mean and std across k-fold CV results.

    Args:
        fold_metrics_list: list of dicts from evaluate(), one per fold.

    Returns:
        (avg_dict, std_dict) where each maps metric name to float.
    """
    if not fold_metrics_list:
        raise ValueError("No fold metrics to summarize")

    keys = fold_metrics_list[0].keys()
    avg = {k: float(np.nanmean([m[k] for m in fold_metrics_list])) for k in keys}
    std = {k: float(np.nanstd([m[k] for m in fold_metrics_list])) for k in keys}
    return avg, std
