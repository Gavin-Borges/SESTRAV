"""Unit tests for src/subgroup_eval.py.

Pure pandas/sklearn helpers: overall + subgroup metric rows and the
subgroup-robust operating-threshold search. Tests use small deterministic
frames covering the degenerate (single-class / small-group / empty) branches.
"""

import numpy as np
import pandas as pd
import pytest

from src import subgroup_eval as se


def _frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.uniform(size=n)
    label = (score > 0.5).astype(int)
    label[0], label[1] = 1, 0  # ensure both classes
    virus = np.where(np.arange(n) % 2 == 0, "EBV", "HPV")
    return pd.DataFrame({"score": score, "label": label, "virus": virus})


def test_safe_metrics_single_class_returns_nan():
    out = se._safe_metrics(np.array([1, 1, 1]), np.array([0.2, 0.8, 0.5]))
    assert all(np.isnan(v) for v in out.values())


def test_safe_metrics_two_classes_delegates_to_evaluate():
    out = se._safe_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert "auc_pr" in out and not np.isnan(out["auc_pr"])


def test_evaluate_subgroups_overall_only():
    df = _frame()
    rows = se.evaluate_subgroups(df, score_col="score")
    assert len(rows) == 1
    assert rows[0]["subgroup_key"] == "overall"
    assert rows[0]["n_samples"] == len(df)
    assert rows[0]["n_positive"] + rows[0]["n_negative"] == len(df)


def test_evaluate_subgroups_with_groups():
    df = _frame(n=60)
    rows = se.evaluate_subgroups(
        df, score_col="score", group_columns=["virus"], min_group_size=5
    )
    keys = {r["subgroup_key"] for r in rows}
    assert keys == {"overall", "virus"}
    virus_rows = [r for r in rows if r["subgroup_key"] == "virus"]
    assert {r["subgroup_value"] for r in virus_rows} == {"EBV", "HPV"}


def test_evaluate_subgroups_small_group_gets_nan():
    df = _frame(n=40)
    # Force a tiny subgroup that falls below min_group_size.
    df.loc[df.index[:3], "virus"] = "RARE"
    rows = se.evaluate_subgroups(
        df, score_col="score", group_columns=["virus"], min_group_size=15
    )
    rare = next(r for r in rows if r.get("subgroup_value") == "RARE")
    assert np.isnan(rare["auc_pr"])


def test_evaluate_subgroups_missing_column_skipped():
    df = _frame()
    rows = se.evaluate_subgroups(
        df, score_col="score", group_columns=["does_not_exist"]
    )
    assert len(rows) == 1  # only the overall row


def test_evaluate_subgroups_nan_group_value_labelled_missing():
    df = _frame(n=40)
    df.loc[df.index[:20], "virus"] = np.nan
    rows = se.evaluate_subgroups(
        df, score_col="score", group_columns=["virus"], min_group_size=5
    )
    assert any(r.get("subgroup_value") == "missing" for r in rows)


def test_pick_operating_threshold_returns_dict():
    df = _frame(n=80)
    best = se.pick_operating_threshold(df, score_col="score", group_col="virus")
    assert set(best) >= {
        "threshold",
        "overall_precision",
        "overall_recall",
        "overall_f1",
        "min_subgroup_f1",
    }
    assert 0.0 <= best["overall_f1"] <= 1.0


def test_pick_operating_threshold_no_group_col():
    df = _frame(n=40)[["score", "label"]]
    best = se.pick_operating_threshold(df, score_col="score", group_col="virus")
    # With no group column, min_subgroup_f1 falls back to overall_f1.
    assert best["min_subgroup_f1"] == best["overall_f1"]


def test_pick_operating_threshold_empty_raises():
    with pytest.raises(ValueError, match="empty dataframe"):
        se.pick_operating_threshold(pd.DataFrame({"score": [], "label": []}))
