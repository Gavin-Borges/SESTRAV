"""Gate 1's threshold and the comparator it is anchored on.

The GNN retry pre-registration's section 2.1 asks for two things in writing:
name the authoritative Gate 1 comparator, and record that GATE1_AUC_PR_MIN does
not change as a consequence. Register row D3 names it - the POOLED mode-31
AUC-PR from results/pooled_cv_metrics_mode31.csv, not the fold-mean from
models/v5/training_results_mode31.csv - and src/verify/promote_gnn.py's comment
now says so. A comment records a commitment; it cannot hold anyone to it.

These tests hold it, by re-deriving D3's numeric claims from the two tracked
sources and requiring the threshold to stay where the ruling left it.
"""

from __future__ import annotations

import pandas as pd

from src.verify.promote_gnn import GATE1_AUC_PR_MIN

POOLED_SOURCE = "results/pooled_cv_metrics_mode31.csv"
FOLD_MEAN_SOURCE = "models/v5/training_results_mode31.csv"
# The measured GNN figure this gate was applied to, from the module docstring's
# scorecard. Used only to size the comparator difference against the margin.
MEASURED_GNN_AUC_PR = 0.6458


def _pooled() -> float:
    frame = pd.read_csv(POOLED_SOURCE)
    return float(frame.loc[frame["metric"] == "mode31_pooled_auc_pr", "value"].iloc[0])


def _fold_mean() -> float:
    frame = pd.read_csv(FOLD_MEAN_SOURCE)
    return float(frame.loc[frame["metric"] == "auc_pr", "rf_cv_mean"].iloc[0])


def test_gate1_threshold_is_the_value_the_ruling_left_it_at() -> None:
    """Moving this bound is an owner decision, not a consequence of D3."""
    assert GATE1_AUC_PR_MIN == 0.65


def test_the_two_comparators_are_distinct_quantities() -> None:
    """D3's claim, re-derived: the pooled and fold-mean figures are not the same.

    They round to 0.6055 and 0.6058 at four decimals. A test that only checked
    the rounded strings would pass if both sources held the same number, so the
    raw values are required to differ as well.
    """
    pooled, fold_mean = _pooled(), _fold_mean()
    assert pooled != fold_mean
    assert round(pooled, 4) == 0.6055
    assert round(fold_mean, 4) == 0.6058


def test_the_fold_mean_source_does_not_carry_the_pooled_figure() -> None:
    """D3 says that file "contains no 0.6055", which is why it is excluded."""
    text = pd.read_csv(FOLD_MEAN_SOURCE).to_csv(index=False)
    assert "0.6055" not in text


def test_naming_the_right_comparator_cannot_move_the_threshold() -> None:
    """Why GATE1_AUC_PR_MIN stays 0.65: the correction is far below the margin.

    The gap between the two comparators is about 3e-04, while the measured GNN
    figure misses the threshold by about 4.2e-03. A difference an order of
    magnitude smaller than the shortfall cannot justify moving a bound quoted to
    two decimals, and cannot flip the verdict either.
    """
    comparator_gap = abs(_fold_mean() - _pooled())
    fail_margin = GATE1_AUC_PR_MIN - MEASURED_GNN_AUC_PR

    assert 0 < comparator_gap < 1e-3
    assert fail_margin > 0, "the measured GNN figure is below the threshold"
    assert comparator_gap * 10 < fail_margin
