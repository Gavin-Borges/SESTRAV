"""Tests for src/conformal.py - cross Venn-Abers probability intervals.

Synthetic in-memory data only; no file I/O and no trained SESTRAV artifact.

Every numeric threshold below was measured first and then loosened, never
tightened to fit. The measured values are recorded next to each assertion so a
reviewer can tell a real margin from a threshold reverse-engineered from one
lucky run.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from src.conformal import (
    _ivap_pairs,
    fit_venn_abers,
    interval_width,
    oof_intervals,
    predict_interval,
)

# Peptide generator. Rows of one peptide share a score and a label, which is the
# regime where peptide leakage can actually reach a score-only calibrator: a
# calibrator that maps score to probability can only be corrupted by a group
# whose rows concentrate at one score.
_N_GROUPS = 150
_PER_GROUP = 32


def _peptide_data(
    seed: int, n_groups: int = _N_GROUPS, per_group: int = _PER_GROUP
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (scores, labels, groups, truth) with truth = E[label | score]."""
    rng = np.random.default_rng(seed)
    score_g = rng.uniform(0.05, 0.95, n_groups)
    label_g = (rng.uniform(size=n_groups) < score_g).astype(np.float64)
    scores = np.repeat(score_g, per_group)
    return (
        scores,
        np.repeat(label_g, per_group),
        np.repeat(np.arange(n_groups), per_group),
        scores.copy(),
    )


def _exchangeable_data(seed: int, n: int = 1500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row per group, label ~ Bernoulli(score): the well-specified case."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.0, 1.0, n)
    labels = (rng.uniform(size=n) < scores).astype(np.float64)
    return scores, labels, np.arange(n)


def _rows_whose_group_appears_in_another_fold(groups: np.ndarray, fold_index: np.ndarray) -> int:
    """Count rows some other fold could have leaked into. Zero means disjoint."""
    leaked = 0
    for g in np.unique(groups):
        member = groups == g
        if np.unique(fold_index[member]).size > 1:
            leaked += int(member.sum())
    return leaked


# --------------------------------------------------------------------------
# Property 1: group disjointness
# --------------------------------------------------------------------------


def test_groups_argument_is_required() -> None:
    scores, labels, groups = _exchangeable_data(0, n=200)
    with pytest.raises(TypeError):
        fit_venn_abers(scores, labels)  # type: ignore[call-arg]


def test_calibration_folds_are_group_disjoint_and_a_violation_is_detectable() -> None:
    scores, labels, groups, _ = _peptide_data(0, n_groups=60, per_group=8)

    grouped = fit_venn_abers(scores, labels, groups, n_folds=5, random_state=7)
    assert _rows_whose_group_appears_in_another_fold(groups, grouped.fold_index) == 0

    # Same data, same detector, grouping deliberately broken by declaring every
    # row its own group. The detector fires, so the zero above is a measurement
    # and not a check that can never fail.
    leaky = fit_venn_abers(scores, labels, np.arange(scores.size), n_folds=5, random_state=7)
    assert _rows_whose_group_appears_in_another_fold(groups, leaky.fold_index) > 0

    # The consequence that matters: an out-of-fold prediction merges the folds a
    # row does not sit in, so under grouping none of those folds holds any row
    # of that row's peptide.
    for k in range(grouped.n_folds):
        held = set(np.unique(groups[grouped.fold_index == k]).tolist())
        others = set(np.unique(groups[grouped.fold_index != k]).tolist())
        assert held.isdisjoint(others)


# --------------------------------------------------------------------------
# Property 2: interval invariants
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_interval_invariants_on_random_inputs(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_groups, per_group = 40, 6
    # Rounded scores so tied calibration values are exercised too.
    score_g = np.round(rng.normal(0.0, 1.0, n_groups), 2)
    scores = np.repeat(score_g, per_group) + rng.normal(0.0, 0.05, n_groups * per_group)
    labels = (rng.uniform(size=scores.size) < 0.5).astype(np.float64)
    groups = np.repeat(np.arange(n_groups), per_group)
    model = fit_venn_abers(scores, labels, groups, n_folds=4, random_state=seed)

    # Queries deliberately span outside the calibration range in both directions.
    query = np.concatenate([rng.uniform(-6.0, 6.0, 120), score_g[:10], [-1e9, 1e9]])
    lower, upper = predict_interval(model, query)

    assert lower.shape == upper.shape == query.shape
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert np.all(lower >= 0.0) and np.all(upper <= 1.0)
    assert np.all(lower <= upper)
    assert np.array_equal(interval_width(model, query), upper - lower)

    oof_lower, oof_upper = oof_intervals(model)
    assert oof_lower.shape == oof_upper.shape == scores.shape
    assert np.all(np.isfinite(oof_lower)) and np.all(np.isfinite(oof_upper))
    assert np.all(oof_lower >= 0.0) and np.all(oof_upper <= 1.0)
    assert np.all(oof_lower <= oof_upper)


def test_cell_cache_matches_a_naive_per_query_ivap_fit() -> None:
    """The one non-obvious optimisation in the module must be exact.

    _ivap_pairs evaluates one representative score per cell between adjacent
    distinct calibration scores instead of refitting per query point. If that
    equivalence were wrong every interval would be silently wrong, so it is
    checked against the literal definition rather than assumed.
    """

    def naive(cal: np.ndarray, lab: np.ndarray, query: np.ndarray) -> tuple:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        p0 = np.empty(query.size)
        p1 = np.empty(query.size)
        for i, s in enumerate(query):
            point = np.array([s])
            x = np.concatenate([cal, point])
            p0[i] = iso.fit(x, np.concatenate([lab, [0.0]])).predict(point)[0]
            p1[i] = iso.fit(x, np.concatenate([lab, [1.0]])).predict(point)[0]
        return p0, p1

    rng = np.random.default_rng(1)
    for trial in range(6):
        size = int(rng.integers(8, 90))
        # Rounding controls how many calibration scores are tied.
        cal = np.round(rng.uniform(0.0, 1.0, size), int(rng.integers(1, 4)))
        lab = (rng.uniform(size=size) < cal).astype(np.float64)
        query = np.concatenate(
            [rng.uniform(-0.3, 1.3, 40), rng.choice(cal, 10), [-9.0, 9.0]]
        )
        cached = _ivap_pairs(cal, lab, query)
        reference = naive(cal, lab, query)
        assert np.array_equal(cached[0], reference[0]), f"p0 mismatch on trial {trial}"
        assert np.array_equal(cached[1], reference[1]), f"p1 mismatch on trial {trial}"


def test_rejects_malformed_input() -> None:
    scores, labels, groups = _exchangeable_data(3, n=200)
    with pytest.raises(ValueError, match="length mismatch"):
        fit_venn_abers(scores[:-1], labels, groups)
    with pytest.raises(ValueError, match="binary"):
        fit_venn_abers(scores, labels + 0.5, groups)
    with pytest.raises(ValueError, match="finite"):
        bad = scores.copy()
        bad[0] = np.nan
        fit_venn_abers(bad, labels, groups)
    with pytest.raises(ValueError, match="both classes"):
        fit_venn_abers(scores, np.zeros_like(labels), groups)
    with pytest.raises(ValueError, match="exceeds"):
        fit_venn_abers(scores, labels, np.zeros_like(groups), n_folds=5)


# --------------------------------------------------------------------------
# Property 3: validity signal, and its degradation without grouping
# --------------------------------------------------------------------------


def test_intervals_track_the_true_probability_on_exchangeable_data() -> None:
    """Positive case: exchangeable rows, label ~ Bernoulli(score).

    The bracket rate asserted here is an empirical property at this n, NOT a
    coverage guarantee: Venn-Abers width shrinks faster than estimation error,
    so this fraction falls as n grows. See the guarantee note in src/conformal.py.
    """
    scores, labels, groups = _exchangeable_data(11)
    model = fit_venn_abers(scores, labels, groups, n_folds=5, random_state=3)
    lower, upper = oof_intervals(model)
    mid = 0.5 * (lower + upper)

    # Calibration in the large. Measured 0.0010.
    assert abs(float(mid.mean() - labels.mean())) < 0.02
    # Point estimate tracks the generator. Measured mean|mid - score| = 0.0198.
    assert float(np.mean(np.abs(mid - scores))) < 0.05
    # Binned calibration over deciles of the prediction. Measured max 0.0258.
    edges = np.quantile(mid, np.linspace(0.0, 1.0, 11))
    bin_of = np.clip(np.searchsorted(edges, mid, side="right") - 1, 0, 9)
    for b in range(10):
        sel = bin_of == b
        assert abs(float(labels[sel].mean() - mid[sel].mean())) < 0.10
    # Bracket rate of the true probability. Measured 0.740 (0.670 at seed 12).
    bracket = float(np.mean((lower <= scores) & (scores <= upper)))
    assert bracket > 0.55


def test_broken_grouping_degrades_the_intervals() -> None:
    """Negative case: the same grouped data fitted while ignoring the groups.

    Three degradations are asserted, all measured across 8 seeds and holding
    8/8 pairwise; the assertions run 3 seeds and compare seed-averaged values,
    which is where the margins below come from.
    """
    seeds = (0, 1, 2)
    stats: dict[str, list[tuple[float, float, float]]] = {"grouped": [], "ignored": []}
    for seed in seeds:
        scores, labels, groups, truth = _peptide_data(seed)
        # Held-out peptides, disjoint from training, for the honest reference.
        new_scores, new_labels, _, _ = _peptide_data(seed + 500)
        for name, g in (("grouped", groups), ("ignored", np.arange(scores.size))):
            model = fit_venn_abers(scores, labels, g, n_folds=5, random_state=1)
            lower, upper = oof_intervals(model)
            mid = 0.5 * (lower + upper)
            new_lower, new_upper = predict_interval(model, new_scores)
            apparent = float(np.mean((mid - labels) ** 2))
            honest = float(np.mean((0.5 * (new_lower + new_upper) - new_labels) ** 2))
            stats[name].append(
                (
                    float(np.mean((lower <= truth) & (truth <= upper))),
                    float(np.mean(upper - lower)),
                    honest - apparent,
                )
            )

    grouped = np.array(stats["grouped"]).mean(axis=0)
    ignored = np.array(stats["ignored"]).mean(axis=0)

    # 1. Bracket rate of the true probability drops. Measured 0.093 vs 0.058.
    assert grouped[0] > ignored[0] + 0.01

    # 2. Intervals collapse to spurious confidence, because a row's near
    #    duplicates sit in the folds scoring it. Measured 0.074 vs 0.014.
    assert grouped[1] > 3.0 * ignored[1]

    # 3. The self-assessment turns optimistic: out-of-fold Brier score beats the
    #    held-out-peptide Brier score only when grouping is broken. Measured
    #    -0.010 (conservative) vs +0.018 (optimistic).
    assert grouped[2] < 0.0
    assert ignored[2] > 0.0
    assert ignored[2] > grouped[2] + 0.01


# --------------------------------------------------------------------------
# Property 4: determinism
# --------------------------------------------------------------------------


def test_same_random_state_gives_identical_output() -> None:
    scores, labels, groups, _ = _peptide_data(4, n_groups=60, per_group=8)
    query = np.linspace(0.0, 1.0, 97)

    first = fit_venn_abers(scores, labels, groups, n_folds=5, random_state=42)
    second = fit_venn_abers(scores, labels, groups, n_folds=5, random_state=42)
    assert np.array_equal(first.fold_index, second.fold_index)

    lo_a, hi_a = predict_interval(first, query)
    lo_b, hi_b = predict_interval(second, query)
    assert np.array_equal(lo_a, lo_b) and np.array_equal(hi_a, hi_b)

    oof_a = oof_intervals(first)
    oof_b = oof_intervals(second)
    assert np.array_equal(oof_a[0], oof_b[0]) and np.array_equal(oof_a[1], oof_b[1])


def test_random_state_none_is_also_reproducible() -> None:
    scores, labels, groups, _ = _peptide_data(5, n_groups=60, per_group=8)
    first = fit_venn_abers(scores, labels, groups, n_folds=5)
    second = fit_venn_abers(scores, labels, groups, n_folds=5)
    assert np.array_equal(first.fold_index, second.fold_index)
