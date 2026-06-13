"""
Unit tests for SESTRAV consensus rank aggregation and paired bootstrapping.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.consensus_ensemble import min_max_scale, compute_borda_scores, run_consensus
from src.statistical_bootstrap import paired_bootstrap_comparison

def test_min_max_scale():
    # Regular case
    s = pd.Series([1.0, 2.0, 3.0])
    scaled = min_max_scale(s)
    pd.testing.assert_series_equal(scaled, pd.Series([0.0, 0.5, 1.0]))

    # Constant case
    s_const = pd.Series([2.0, 2.0, 2.0])
    scaled_const = min_max_scale(s_const)
    pd.testing.assert_series_equal(scaled_const, pd.Series([0.5, 0.5, 0.5]))

    # NaN / empty case
    s_nan = pd.Series([np.nan, np.nan])
    scaled_nan = min_max_scale(s_nan)
    pd.testing.assert_series_equal(scaled_nan, pd.Series([0.5, 0.5]))

def test_borda_scores():
    # Setup dataframe where A is [10, 20, 30] and B is [0.3, 0.2, 0.1]
    # Ranking for A: index 2 (rank 1), index 1 (rank 2), index 0 (rank 3)
    # Ranking for B: index 0 (rank 1), index 1 (rank 2), index 2 (rank 3)
    df = pd.DataFrame({
        "model_A": [10.0, 20.0, 30.0],
        "model_B": [0.3, 0.2, 0.1]
    })
    
    # uniform weights
    scores = compute_borda_scores(df, ["model_A", "model_B"])
    # Borda sum:
    # index 0: A_rank = 3 (score 3-3=0), B_rank = 1 (score 3-1=2) -> total 2
    # index 1: A_rank = 2 (score 3-2=1), B_rank = 2 (score 3-2=1) -> total 2
    # index 2: A_rank = 1 (score 3-1=2), B_rank = 3 (score 3-3=0) -> total 2
    # Constant sum = 2 -> scales to 0.5
    pd.testing.assert_series_equal(scores, pd.Series([0.5, 0.5, 0.5]))

    # Custom weights: model_A weight = 2.0, model_B weight = 1.0
    # Borda weighted sum:
    # index 0: 0*2 + 2*1 = 2
    # index 1: 1*2 + 1*1 = 3
    # index 2: 2*2 + 0*1 = 4
    # Scaled range [2, 4] -> [0.0, 0.5, 1.0]
    scores_weighted = compute_borda_scores(df, ["model_A", "model_B"], weights={"model_A": 2.0, "model_B": 1.0})
    pd.testing.assert_series_equal(scores_weighted, pd.Series([0.0, 0.5, 1.0]))

def test_paired_bootstrap():
    # Construct a dataset where ref_col predicts label perfectly, comp_col predicts randomly
    np.random.seed(42)
    n = 100
    labels = np.random.choice([0, 1], size=n)
    
    # ref_col has higher values for label=1
    ref_scores = labels * 0.8 + np.random.normal(0, 0.1, size=n)
    # comp_col is random noise
    comp_scores = np.random.normal(0, 0.5, size=n)
    
    df = pd.DataFrame({
        "label": labels,
        "ref": ref_scores,
        "comp": comp_scores
    })
    
    res = paired_bootstrap_comparison(
        df,
        ref_col="ref",
        comp_col="comp",
        label_col="label",
        n_resamples=500, # small count for quick test
        seed=42
    )
    
    assert "error" not in res
    assert res["n_samples"] == 100
    
    # Verify we got metrics
    assert "auc_pr" in res
    assert "auc_roc" in res
    assert "issr_10" in res
    
    # Delta should be positive since ref is superior to comp
    assert res["auc_roc"]["delta_base"] > 0
    assert 0.0 <= res["auc_roc"]["p_value"] <= 1.0
    assert res["auc_roc"]["significant_95"] in ["yes", "no"]
