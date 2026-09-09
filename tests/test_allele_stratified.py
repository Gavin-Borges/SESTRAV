"""tests/test_allele_stratified.py
================================
Unit tests for allele-stratified concordance metrics and bootstrap CI logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_allele_stratified import (
    compute_stratum_concordance,
    compute_stratified_metrics,
    stratified_bootstrap_ci,
)


def test_compute_stratum_concordance_perfect():
    pos = np.array([0.9, 0.8])
    neg = np.array([0.2, 0.1, 0.3])
    auc, pairs = compute_stratum_concordance(pos, neg)
    assert pairs == 6
    assert auc == 1.0


def test_compute_stratum_concordance_inverted():
    pos = np.array([0.1, 0.2])
    neg = np.array([0.8, 0.9])
    auc, pairs = compute_stratum_concordance(pos, neg)
    assert pairs == 4
    assert auc == 0.0


def test_compute_stratum_concordance_ties():
    pos = np.array([0.5, 0.8])
    neg = np.array([0.5, 0.2])
    # pairs:
    # (0.5, 0.5) -> tie (0.5)
    # (0.5, 0.2) -> pos > neg (1.0)
    # (0.8, 0.5) -> pos > neg (1.0)
    # (0.8, 0.2) -> pos > neg (1.0)
    # Total: 3.5 / 4 = 0.875
    auc, pairs = compute_stratum_concordance(pos, neg)
    assert pairs == 4
    assert auc == 0.875


def test_compute_stratum_concordance_empty():
    auc, pairs = compute_stratum_concordance(np.array([]), np.array([0.5]))
    assert pairs == 0
    assert np.isnan(auc)


def test_compute_stratified_metrics_multi_strata():
    # Stratum A: 2 pos, 2 neg (4 pairs), perfect (auc = 1.0)
    # Stratum B: 1 pos, 2 neg (2 pairs), inverted (auc = 0.0)
    # Stratum C: 2 pos, 0 neg (0 pairs, single class)
    # Pair-weighted (MH) concordance: (4 * 1.0 + 2 * 0.0) / 6 = 4/6 = 0.6667
    # Row-weighted concordance: (4 samples * 1.0 + 3 samples * 0.0) / 7 = 4/7 = 0.5714
    data = [
        {"allele": "A", "label": 1, "score": 0.9},
        {"allele": "A", "label": 1, "score": 0.8},
        {"allele": "A", "label": 0, "score": 0.2},
        {"allele": "A", "label": 0, "score": 0.1},
        {"allele": "B", "label": 1, "score": 0.1},
        {"allele": "B", "label": 0, "score": 0.8},
        {"allele": "B", "label": 0, "score": 0.9},
        {"allele": "C", "label": 1, "score": 0.5},
        {"allele": "C", "label": 1, "score": 0.6},
    ]
    df = pd.DataFrame(data)
    res = compute_stratified_metrics(df, score_col="score")

    assert res["total_pairs"] == 6
    assert res["two_class_strata_count"] == 2
    assert res["total_strata_count"] == 3
    assert pytest.approx(res["mh_concordance"], abs=1e-4) == 4 / 6
    assert pytest.approx(res["row_weighted_concordance"], abs=1e-4) == 4 / 7


def test_stratified_bootstrap_ci():
    # Verify deterministic reproducibility and valid interval
    data = [
        {"allele": "HLA-A*02:01", "label": 1, "score": 0.8},
        {"allele": "HLA-A*02:01", "label": 1, "score": 0.7},
        {"allele": "HLA-A*02:01", "label": 0, "score": 0.3},
        {"allele": "HLA-A*02:01", "label": 0, "score": 0.2},
    ]
    df = pd.DataFrame(data)
    low, high = stratified_bootstrap_ci(df, score_col="score", n_resamples=500, seed=12345)
    assert 0.0 <= low <= high <= 1.0
    assert low == 1.0
    assert high == 1.0
