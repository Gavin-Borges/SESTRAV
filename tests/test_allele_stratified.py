"""tests/test_allele_stratified.py
================================
Unit tests for allele-stratified concordance metrics and bootstrap CI logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_allele_stratified import (
    _stratum_dominance,
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


# ---------------------------------------------------------------------------
# Paired delta CI.
#
# The script reports two MARGINAL intervals, one per score column, drawn from
# two deliberately different RNG seeds. Overlapping marginals are not a test of
# their difference, and the difference is what the within-allele question asks.
# These fixtures are synthetic on purpose: the real cohort lives under results/,
# which is read-denied here, and a test that needs it would not run in CI either.
# ---------------------------------------------------------------------------

ALLELES = ("HLA-A*02:01", "HLA-A*01:01", "HLA-B*07:02", "HLA-B*44:02")


def _synthetic_cohort(kind: str, seed: int = 7, n_per_class: int = 15) -> pd.DataFrame:
    """Four strata, n_per_class positives and negatives each.

    kind='monotone': model is a strictly increasing affine transform of raw, so
                     every pairwise comparison has the same sign in both arms and
                     the paired delta is EXACTLY zero in every resample.
    kind='noise':    model is drawn independently of the label; raw is informative.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for allele in ALLELES:
        for label in (1, 0):
            for _ in range(n_per_class):
                raw = float(rng.normal(0.6 if label == 1 else 0.0, 1.0))
                model = 2.0 * raw + 1.0 if kind == "monotone" else float(rng.normal(0.0, 1.0))
                rows.append(
                    {
                        "allele": allele,
                        "label": label,
                        "immunogenicity_score": model,
                        "presentation_score": raw,
                    }
                )
    return pd.DataFrame(rows)


def test_paired_delta_is_zero_for_monotone_transform():
    """An order-preserving transform cannot change concordance, so delta is 0."""
    res = stratified_bootstrap_ci(
        _synthetic_cohort("monotone"),
        score_col="immunogenicity_score",
        compare_col="presentation_score",
        n_resamples=500,
        seed=20260909,
    )
    assert res["delta_point"] == pytest.approx(0.0, abs=1e-12)
    assert res["delta_ci"][0] == pytest.approx(0.0, abs=1e-12)
    assert res["delta_ci"][1] == pytest.approx(0.0, abs=1e-12)
    assert res["delta_excludes_zero"] is False
    assert res["score_ci"] == pytest.approx(res["compare_ci"], abs=1e-12)


def test_paired_delta_strictly_below_zero_for_noise_model():
    """A model with no signal must lose to an informative raw score, detectably.

    The final assertion is the point of this test: the fixture is required to keep
    the two MARGINAL intervals overlapping, which is the exact condition under
    which the old two-interval reporting was inconclusive and the paired contrast
    earns its keep.
    """
    res = stratified_bootstrap_ci(
        _synthetic_cohort("noise"),
        score_col="immunogenicity_score",
        compare_col="presentation_score",
        n_resamples=1000,
        seed=20260909,
    )
    assert res["delta_point"] < 0.0
    assert res["delta_ci"][1] < 0.0, f"upper bound {res['delta_ci'][1]} is not below 0"
    assert res["delta_excludes_zero"] is True

    lo_m, hi_m = res["score_ci"]
    lo_r, hi_r = res["compare_ci"]
    assert lo_m < hi_r and lo_r < hi_m, (
        "fixture must keep the MARGINAL intervals overlapping - that is the "
        "condition under which the paired test is the only one that can answer"
    )


def test_paired_marginal_matches_unpaired_call():
    """Drawing indices instead of values must not perturb the RNG stream.

    If this fails, every already-published marginal CI has silently moved.
    """
    df = _synthetic_cohort("noise")
    marginal = stratified_bootstrap_ci(
        df, score_col="immunogenicity_score", n_resamples=400, seed=20260909
    )
    paired = stratified_bootstrap_ci(
        df,
        score_col="immunogenicity_score",
        compare_col="presentation_score",
        n_resamples=400,
        seed=20260909,
    )
    assert paired["score_ci"] == pytest.approx(marginal, abs=1e-12)


def test_backwards_compatible_two_tuple():
    """Without compare_col the historical 2-tuple shape is preserved."""
    out = stratified_bootstrap_ci(
        _synthetic_cohort("noise"), score_col="immunogenicity_score", n_resamples=200, seed=1
    )
    assert isinstance(out, tuple) and len(out) == 2
    assert 0.0 <= out[0] <= out[1] <= 1.0


def test_paired_mode_uses_one_shared_row_mask():
    """Per-column dropna would desynchronise the two arms and break the pairing.

    NaNs are injected into ONE column only. Under a shared mask the surviving rows
    are identical in both arms, so a monotone fixture still gives exactly zero.
    """
    df = _synthetic_cohort("monotone")
    df.loc[df.index[:5], "presentation_score"] = np.nan
    res = stratified_bootstrap_ci(
        df,
        score_col="immunogenicity_score",
        compare_col="presentation_score",
        n_resamples=300,
        seed=20260909,
    )
    assert res["delta_point"] == pytest.approx(0.0, abs=1e-12)
    assert res["paired_pairs"] > 0


def test_paired_mode_with_no_two_class_stratum_returns_nan_dict():
    df = _synthetic_cohort("noise")
    df["label"] = 1
    res = stratified_bootstrap_ci(
        df,
        score_col="immunogenicity_score",
        compare_col="presentation_score",
        n_resamples=50,
        seed=1,
    )
    assert np.isnan(res["delta_point"])
    assert res["paired_pairs"] == 0
    assert res["delta_excludes_zero"] is False


def test_stratum_dominance_flags_a_single_dominant_allele():
    strata = [
        {"allele": "HLA-A*02:01", "pairs": 1666},
        {"allele": "HLA-A*01:01", "pairs": 150},
        {"allele": "HLA-B*07:02", "pairs": 100},
        {"allele": "HLA-B*44:02", "pairs": 0},
    ]
    dom = _stratum_dominance(strata, total_pairs=1916)
    assert dom["top_allele"] == "HLA-A*02:01"
    assert dom["top_allele_pairs"] == 1666
    assert dom["top_allele_pair_share"] == pytest.approx(1666 / 1916)
    assert dom["n_strata_contributing"] == 3, "a zero-pair stratum must not count"
    assert dom["dominance_warning"] is True


def test_stratum_dominance_silent_when_evenly_spread():
    strata = [{"allele": f"A{i}", "pairs": 100} for i in range(5)]
    dom = _stratum_dominance(strata, total_pairs=500)
    assert dom["dominance_warning"] is False
    assert dom["top_allele_pair_share"] == pytest.approx(0.2)
