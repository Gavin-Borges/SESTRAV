"""Cross Venn-Abers probability intervals for binary immunogenicity scores.

Maps a real-valued model score to a pair [lower, upper] that brackets a
calibrated probability. Pure library: no file I/O, no model loading, no CLI.

Method: Vovk, Petej and Fedorova (2015), "Large-scale probabilistic predictors
with and without guarantees of validity", NeurIPS 28. An inductive Venn-Abers
predictor (IVAP) is built per fold from ``sklearn.isotonic.IsotonicRegression``;
the per-fold pairs are merged across folds with the paper's geometric-mean rule.

WHAT IS GUARANTEED, AND WHAT IS NOT. Read this before quoting the output.

1. A single IVAP pair is valid in the Venn sense (Vovk et al. 2015, Sec. 3):
   under exchangeability of the calibration examples, one of the two endpoints
   is a calibrated probability for the realised label. That statement is about
   the PAIR, not about either endpoint alone.
2. The IVAP guarantee assumes the underlying scorer never saw the calibration
   examples. Scores produced by a model trained on these same rows void it.
3. CVAP has NO proven validity guarantee. Vovk et al. introduce the cross
   variant explicitly as the "without guarantees of validity" half of their
   title, justified empirically. Anything this module outputs inherits that.
4. [lower, upper] is NOT a confidence interval for the true conditional
   probability and has no nominal coverage level. Its width reflects the
   influence of one extra observation, so it shrinks roughly like the inverse
   of the local isotonic block size, while the estimation error shrinks only
   like the inverse square root of it. The fraction of test points whose true
   probability falls inside the interval therefore DECREASES as n grows. Do not
   report the interval as "N percent coverage" of anything.
5. Rows here are grouped by peptide and are NOT exchangeable at row level.
   Grouped folds stop a peptide appearing on both sides of a fold, which is
   what keeps the interval targeting the peptide-independent probability. It
   does not restore row-level exchangeability, so 1 remains unavailable even
   for a single fold.
6. What the endpoints are: the merge rule is applied separately to the K lower
   endpoints and the K upper endpoints. The resulting interval provably
   contains Vovk et al.'s merged CVAP point probability, because
   x -> x / (x + c) is increasing in x.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold

FloatArray = NDArray[np.float64]

# Guards log(0) in the geometric-mean merge. A single fold reporting an exact
# 0 or 1 must still be able to drive the merged endpoint to 0 or 1.
_EPS = 1e-12

__all__ = [
    "CrossVennAbers",
    "fit_venn_abers",
    "interval_width",
    "oof_intervals",
    "predict_interval",
]


@dataclass(frozen=True, eq=False)
class CrossVennAbers:
    """Fitted CVAP: the calibration rows and the group-disjoint fold they sit in."""

    scores: FloatArray
    labels: FloatArray
    groups: NDArray[Any]
    fold_index: NDArray[np.int64]
    n_folds: int
    random_state: int | None


def fit_venn_abers(
    scores: NDArray[Any],
    labels: NDArray[Any],
    groups: NDArray[Any],
    n_folds: int = 5,
    random_state: int | None = None,
) -> CrossVennAbers:
    """Assign calibration rows to n_folds folds that are disjoint by group.

    ``groups`` is required, not optional: passing one group per row reproduces
    the leaky ungrouped behaviour, and that has to be an explicit caller choice.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    g = np.asarray(groups).ravel()
    if s.size == 0:
        raise ValueError("scores must be non-empty")
    if not (s.size == y.size == g.size):
        raise ValueError(f"length mismatch: scores={s.size} labels={y.size} groups={g.size}")
    if not bool(np.all(np.isfinite(s))):
        raise ValueError("scores must all be finite")
    if not bool(np.all((y == 0.0) | (y == 1.0))):
        raise ValueError("labels must be binary 0/1")
    if np.unique(y).size < 2:
        raise ValueError("labels must contain both classes")
    n_groups = int(np.unique(g).size)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_folds > n_groups:
        raise ValueError(f"n_folds={n_folds} exceeds the {n_groups} distinct groups")

    splitter = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=random_state is not None,
        random_state=random_state,
    )
    fold_index = np.empty(s.size, dtype=np.int64)
    for k, (_, held) in enumerate(splitter.split(s.reshape(-1, 1), y, groups=g)):
        fold_index[held] = k
    return CrossVennAbers(
        scores=s,
        labels=y,
        groups=g,
        fold_index=fold_index,
        n_folds=n_folds,
        random_state=random_state,
    )


def _ivap_pairs(
    cal_scores: FloatArray, cal_labels: FloatArray, query: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """IVAP endpoints at each query score, calibrated on one fold."""
    p0 = np.empty(query.size, dtype=np.float64)
    p1 = np.empty(query.size, dtype=np.float64)
    if query.size == 0:
        return p0, p1

    # Isotonic regression on a chain depends only on the ORDER of the points, so
    # p0/p1 are constant between adjacent distinct calibration scores. Evaluating
    # one representative per cell caps the work at 2M+1 fits for M distinct
    # calibration scores, however many query points there are.
    grid = np.unique(cal_scores)
    idx = np.searchsorted(grid, query, side="left")
    on_grid = grid[np.minimum(idx, grid.size - 1)] == query
    cell = 2 * idx + on_grid.astype(np.int64)

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    for c in np.unique(cell):
        sel = cell == c
        point = np.array([query[sel][0]], dtype=np.float64)
        x = np.concatenate([cal_scores, point])
        p0[sel] = iso.fit(x, np.concatenate([cal_labels, [0.0]])).predict(point)[0]
        p1[sel] = iso.fit(x, np.concatenate([cal_labels, [1.0]])).predict(point)[0]
    return p0, p1


def _fold_pairs(model: CrossVennAbers, query: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Per-fold IVAP endpoints, shaped (n_folds, n_query)."""
    p0 = np.empty((model.n_folds, query.size), dtype=np.float64)
    p1 = np.empty_like(p0)
    for k in range(model.n_folds):
        sel = model.fold_index == k
        p0[k], p1[k] = _ivap_pairs(model.scores[sel], model.labels[sel], query)
    return p0, p1


def _merge(p: FloatArray, weights: FloatArray) -> FloatArray:
    """Geometric-mean merge of per-fold probabilities (Vovk et al. 2015)."""
    q = np.clip(p, _EPS, 1.0 - _EPS)
    used = weights.sum(axis=0)
    log_p = (weights * np.log(q)).sum(axis=0) / used
    log_1mp = (weights * np.log1p(-q)).sum(axis=0) / used
    with np.errstate(over="ignore"):
        merged: FloatArray = 1.0 / (1.0 + np.exp(log_1mp - log_p))
    return merged


def _endpoints(
    p0: FloatArray, p1: FloatArray, weights: FloatArray
) -> tuple[FloatArray, FloatArray]:
    lower = _merge(p0, weights)
    upper = _merge(p1, weights)
    # The merge is monotone in each fold's endpoint and p0 <= p1 holds per fold,
    # so lower <= upper is already implied; this only absorbs float64 rounding.
    return (
        np.clip(np.minimum(lower, upper), 0.0, 1.0),
        np.clip(np.maximum(lower, upper), 0.0, 1.0),
    )


def _as_query(scores: NDArray[Any]) -> FloatArray:
    q = np.asarray(scores, dtype=np.float64).ravel()
    if q.size and not bool(np.all(np.isfinite(q))):
        raise ValueError("scores must all be finite")
    return q


def predict_interval(
    model: CrossVennAbers, scores: NDArray[Any]
) -> tuple[FloatArray, FloatArray]:
    """Interval for new scores, merged over all folds.

    Valid only for rows whose group is absent from the calibration data; the
    caller owns that, since a bare score carries no group.
    """
    query = _as_query(scores)
    p0, p1 = _fold_pairs(model, query)
    return _endpoints(p0, p1, np.ones_like(p0))


def oof_intervals(model: CrossVennAbers) -> tuple[FloatArray, FloatArray]:
    """Out-of-fold intervals for the calibration rows themselves.

    Each row is merged over the folds that do not hold it. Because folds are
    group-disjoint, none of those folds holds any row of that row's group.
    """
    p0, p1 = _fold_pairs(model, model.scores)
    weights = (
        np.arange(model.n_folds, dtype=np.int64)[:, None] != model.fold_index[None, :]
    ).astype(np.float64)
    return _endpoints(p0, p1, weights)


def interval_width(model: CrossVennAbers, scores: NDArray[Any]) -> FloatArray:
    """upper - lower for new scores."""
    lower, upper = predict_interval(model, scores)
    width: FloatArray = upper - lower
    return width
