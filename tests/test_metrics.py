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


def test_precision_at_k_is_a_delegation_not_an_independent_check():
    """
    precision_at_k is `return issr_at_k(y_true, y_scores, k)`.

    So test_precision_at_k_equals_issr above compares a function to itself and
    cannot fail for any input. It is kept because the delegation is the intended
    contract, but it is NOT independent corroboration of issr_at_k, and it must
    not be read as a second opinion on any issr value.
    """
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    for k in (10, 25, 50, 100):
        assert precision_at_k(y_true, y_scores, k) == issr_at_k(y_true, y_scores, k)


# -- Tie behaviour at the top-K cutoff ---------------------------------------
#
# issr_at_k and recall_at_k each select the top K% with a bare
# `np.argsort(y_scores)[-top_k_count:]`, which applies no tie-break. When more
# rows share the cutoff score than there are slots left, WHICH of them is kept
# depends on the row order, so the metric is order-dependent.
#
# This is not hypothetical for SESTRAV: rf_oof_score is discretized to multiples
# of 1/200 (that arm used n_estimators=200), so ties at the cutoff are common
# rather than rare.
#
# These tests CHARACTERIZE the current behaviour. They deliberately pin no
# published value and assert nothing about which sort algorithm wins - only the
# achievable band and the fact of order-dependence. Both hold on any platform
# and under any numpy sort kind.
#
# One measured result is worth stating, because it is not obvious and it rules
# out the cheapest-looking fix: passing kind="stable" does NOT make these metrics
# order-invariant. A stable sort preserves the INPUT row order among equal keys,
# so it only makes the result reproducible for a fixed row order; permuting the
# rows still moves the value, and the band below is unchanged. Verified against
# both stable-ascending and stable-descending variants. Genuine invariance needs
# either a tie-break on a real secondary key (e.g. the peptide) or an estimator
# that does not depend on tie order at all.

# 4 rows strictly above the cutoff, 6 tied AT it, 10 below. For n=20 and k=25
# the cutoff admits 5 rows: the 4 clear ones plus exactly ONE slot that the 6
# tied rows compete for. 3 of those 6 are positive, so that slot decides.
_TIE_SCORES = np.array([0.9] * 4 + [0.5] * 6 + [0.1] * 10)
_TIE_LABELS = np.array([1] * 4 + [1, 1, 1, 0, 0, 0] + [0] * 10)
_TIE_K = 25

# Distinct scores, same shape, as a control.
_CLEAN_SCORES = np.linspace(0.01, 0.99, 20)
_CLEAN_LABELS = np.zeros(20, dtype=int)
_CLEAN_LABELS[[0, 1, 16, 17, 19]] = 1


def _values_across_permutations(fn, y_true, y_scores, k, n_perm=300):
    """Collect the distinct values fn takes over row permutations of one multiset."""
    seen = set()
    for seed in range(n_perm):
        order = np.random.default_rng(seed).permutation(len(y_scores))
        seen.add(fn(y_true[order], y_scores[order], k))
    return seen


def test_issr_at_k_is_order_dependent_when_ties_span_the_cutoff():
    """The same multiset of (label, score) pairs yields more than one ISSR."""
    seen = _values_across_permutations(issr_at_k, _TIE_LABELS, _TIE_SCORES, _TIE_K)

    assert len(seen) > 1, f"expected order-dependence, got the single value {seen}"
    # 4 clear positives, plus a contested slot that is positive or not.
    assert seen == {4 / 5, 5 / 5}


def test_recall_at_k_is_independently_order_dependent():
    """
    recall_at_k carries its OWN unguarded argsort, so it must be shown to be
    order-dependent directly.

    Asserting only that recall equals issr times a constant would be an algebraic
    identity between two functions sharing a top-K set: it holds just as well on
    an input with no ties at all, and so proves nothing about tie handling.
    """
    seen = _values_across_permutations(recall_at_k, _TIE_LABELS, _TIE_SCORES, _TIE_K)

    assert len(seen) > 1, f"expected order-dependence, got the single value {seen}"
    n_pos = int(_TIE_LABELS.sum())
    assert seen == {4 / n_pos, 5 / n_pos}


def test_every_numpy_sort_kind_stays_within_the_tie_band():
    """
    No sort kind escapes the band, but they do NOT agree with each other.

    Which kind lands where is an implementation detail of the installed numpy and
    is deliberately not asserted - that would make this test a platform pin
    rather than a characterization.
    """
    band = set()
    for kind in ("quicksort", "stable", "heapsort", "mergesort"):
        idx = np.argsort(_TIE_SCORES, kind=kind)[-5:]
        band.add(float(np.mean(_TIE_LABELS[idx])))

    assert band.issubset({4 / 5, 5 / 5})
    assert len(band) > 1, "sort kinds agreed; the fixture no longer exercises a tie"


def test_distinct_scores_are_order_invariant():
    """
    Control. Without ties at the cutoff both metrics take exactly one value.

    This is what makes the two order-dependence tests above meaningful: it shows
    they detect ties rather than merely detecting permutation.
    """
    issr_seen = _values_across_permutations(
        issr_at_k, _CLEAN_LABELS, _CLEAN_SCORES, _TIE_K
    )
    recall_seen = _values_across_permutations(
        recall_at_k, _CLEAN_LABELS, _CLEAN_SCORES, _TIE_K
    )

    assert len(issr_seen) == 1, f"distinct scores should be invariant, got {issr_seen}"
    assert len(recall_seen) == 1, (
        f"distinct scores should be invariant, got {recall_seen}"
    )


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
