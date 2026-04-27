"""
Helpers for subgroup-aware evaluation and threshold analysis.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from src.evaluate_metrics import evaluate


def _safe_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """Return metric dict with NaN fallback for invalid slices."""
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        return {
            "auc_roc": np.nan,
            "auc_pr": np.nan,
            "issr_10": np.nan,
            "issr_25": np.nan,
        }
    return evaluate(y_true, y_scores)


def evaluate_subgroups(
    fold_df: pd.DataFrame,
    score_col: str,
    label_col: str = "label",
    group_columns: Optional[Iterable[str]] = None,
    min_group_size: int = 15,
) -> List[Dict]:
    """
    Compute overall and subgroup metrics on one validation fold.

    Returns a list of metric rows with fields:
      subgroup_key, subgroup_value, n_samples, n_positive, n_negative, metric columns
    """
    rows: List[Dict] = []
    y_true = fold_df[label_col].to_numpy()
    y_scores = fold_df[score_col].to_numpy()

    overall = _safe_metrics(y_true, y_scores)
    rows.append(
        {
            "subgroup_key": "overall",
            "subgroup_value": "all",
            "n_samples": int(len(fold_df)),
            "n_positive": int((y_true == 1).sum()),
            "n_negative": int((y_true == 0).sum()),
            **overall,
        }
    )

    if not group_columns:
        return rows

    for group_col in group_columns:
        if group_col not in fold_df.columns:
            continue
        for group_value, gdf in fold_df.groupby(group_col, dropna=False):
            y_g = gdf[label_col].to_numpy()
            s_g = gdf[score_col].to_numpy()
            row = {
                "subgroup_key": group_col,
                "subgroup_value": "missing" if pd.isna(group_value) else str(group_value),
                "n_samples": int(len(gdf)),
                "n_positive": int((y_g == 1).sum()),
                "n_negative": int((y_g == 0).sum()),
            }
            if len(gdf) < min_group_size:
                row.update(
                    {
                        "auc_roc": np.nan,
                        "auc_pr": np.nan,
                        "issr_10": np.nan,
                        "issr_25": np.nan,
                    }
                )
            else:
                row.update(_safe_metrics(y_g, s_g))
            rows.append(row)
    return rows


def pick_operating_threshold(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    group_col: str = "virus",
    min_group_size: int = 15,
) -> Dict[str, float]:
    """
    Choose threshold balancing recall and precision with subgroup robustness.

    Objective:
      1) maximize minimum subgroup F1 (guard against skew)
      2) then maximize overall F1
      3) then maximize overall recall
    """
    if df.empty:
        raise ValueError("Cannot tune threshold on empty dataframe")
    work = df[[score_col, label_col] + ([group_col] if group_col in df.columns else [])].copy()
    work = work.dropna(subset=[score_col, label_col])
    thresholds = np.unique(np.quantile(work[score_col].to_numpy(), np.linspace(0.05, 0.95, 91)))
    best = None

    for thr in thresholds:
        y_true = work[label_col].to_numpy().astype(int)
        y_pred = (work[score_col].to_numpy() >= thr).astype(int)
        overall_precision = precision_score(y_true, y_pred, zero_division=0)
        overall_recall = recall_score(y_true, y_pred, zero_division=0)
        overall_f1 = (
            2 * overall_precision * overall_recall / (overall_precision + overall_recall)
            if (overall_precision + overall_recall) > 0
            else 0.0
        )

        subgroup_f1s = []
        if group_col in work.columns:
            for _, gdf in work.groupby(group_col, dropna=False):
                if len(gdf) < min_group_size:
                    continue
                y_g = gdf[label_col].to_numpy().astype(int)
                p_g = (gdf[score_col].to_numpy() >= thr).astype(int)
                prec_g = precision_score(y_g, p_g, zero_division=0)
                rec_g = recall_score(y_g, p_g, zero_division=0)
                f1_g = (
                    2 * prec_g * rec_g / (prec_g + rec_g)
                    if (prec_g + rec_g) > 0
                    else 0.0
                )
                subgroup_f1s.append(float(f1_g))
        min_subgroup_f1 = float(min(subgroup_f1s)) if subgroup_f1s else overall_f1

        candidate = {
            "threshold": float(thr),
            "overall_precision": float(overall_precision),
            "overall_recall": float(overall_recall),
            "overall_f1": float(overall_f1),
            "min_subgroup_f1": float(min_subgroup_f1),
        }
        if best is None:
            best = candidate
            continue
        if (
            candidate["min_subgroup_f1"] > best["min_subgroup_f1"]
            or (
                np.isclose(candidate["min_subgroup_f1"], best["min_subgroup_f1"])
                and candidate["overall_f1"] > best["overall_f1"]
            )
            or (
                np.isclose(candidate["min_subgroup_f1"], best["min_subgroup_f1"])
                and np.isclose(candidate["overall_f1"], best["overall_f1"])
                and candidate["overall_recall"] > best["overall_recall"]
            )
        ):
            best = candidate

    return best if best is not None else {}
