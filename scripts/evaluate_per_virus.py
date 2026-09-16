"""
evaluate_per_virus.py - Per-virus AUC evaluation with bootstrap CIs.

Implements the Week 3 evaluation protocol (Part 16, Amendment 6):
  AUC-ROC with 95% CI (1000-rep bootstrap) -- primary metric
  AUC-PR with 95% CI (1000-rep bootstrap) -- co-primary metric
  Precision at 10% recall -- practical vaccine shortlisting cutpoint
  ECE (Expected Calibration Error) -- per-virus calibration check
  AUC-ROC by peptide length (9-mer vs 8+10+11-mer) -- zero-imputation confound
  AUC-ROC on real-tested-negative rows only -- scientifically honest metric

Model comparison uses paired bootstrap (src.statistical_bootstrap) rather than
DeLong's test: paired bootstrap gives empirical CIs for correlated AUC estimates
without assuming a Gaussian AUC distribution, which is preferable for small n.

Amendment 6 exit criterion (checked automatically). It is adjudicated on the
DECOY-FREE column auc_roc_real_neg_only, not on the full-negative-set auc_roc,
which is contaminated by binding-matrix coverage:
  EBV: real-negative-only AUC-ROC point estimate >= 0.57
  HPV: real-negative-only AUC-ROC point estimate >= 0.58
Both columns are reported side by side, and every adjudicated virus also carries a
derived per-virus validity floor. See the "Exit criterion" constants block and
chance_ceiling() for the contamination evidence, the derivation, and what is
deliberately NOT re-derived here.

Usage:
  python scripts/evaluate_per_virus.py \\
      --predictions data/oof_mode31_v5.csv \\
      [--compare      data/oof_mode33_v5.csv] \\
      [--output-json  results/per_virus_eval_v5.json] \\
      [--output-csv   results/per_virus_eval_v5.csv] \\
      [--n-bootstrap  1000] \\
      [--min-virus-size 20]

OOF prediction CSV expected columns:
  label            binary int (0/1)
  score            model probability / score (higher = more immunogenic)
  virus            virus name matching v5 dataset vocabulary
  peptide          amino acid sequence (used for length subgroup split)
  negative_origin  enum: tested_negative | self_proteome_decoy | ... (optional)
  hla_allele       HLA allele string (optional; used by subgroup_eval only)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from src.statistical_bootstrap import paired_bootstrap_comparison

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_N_BOOTSTRAP: int = 1000
CI_LEVEL: float = 0.95
# Origins that represent genuine experimentally-confirmed negatives. Both the
# bulk-export path ("tested_negative") and the IEDB REST/API-bridge path
# ("iedb_api") are real assay-confirmed negatives; this matches the real-negative
# definition in build_dataset_v5.py (_real_neg_origins). Synthetic decoys
# (allele_matched_nonbinder, self_proteome_decoy) are NOT real negatives.
REAL_NEG_ORIGINS: frozenset[str] = frozenset({"tested_negative", "iedb_api"})
MIN_SAMPLES_DEFAULT: int = 20

# ---------------------------------------------------------------------------
# Exit criterion: which column, which level, and what a level has to clear
# ---------------------------------------------------------------------------
# The two AUC-ROC columns this script emits are not interchangeable, and the exit
# criterion is adjudicated on the second one.
#
# HEADLINE_AUC_COL scores positives against EVERY negative, decoys included, and is
# contaminated by binding-matrix coverage. Measured on the tracked mode-31 OOF frame
# (models/v5/rf_oof_predictions_mode31.csv, 35,555 rows) against
# models/peptide_binding_matrix_v5.csv: row coverage is 100.00% for tested_negative
# (22,466 rows) and 100.00% for iedb_api (1,956 rows) but 7.01% for the synthetic
# allele_matched_nonbinder decoys (3,112 rows). src/train_classifier.py zero-fills an
# uncovered peptide with no guard (:159, :189, :293, :400), so "all ten binding
# features are zero" is very nearly the indicator "this row is a synthetic decoy". On
# DENV that indicator ALONE scores AUC-ROC 0.9837, ABOVE the 0.9769 reached by the
# max binding feature it stands in for. Wherever decoys are present, part of what
# auc_roc measures is decoy provenance rather than immunogenicity.
#
# HONEST_AUC_COL restricts the negative set to REAL_NEG_ORIGINS, both of which are
# 100% covered, so the coverage indicator is constant there and carries no signal.
HEADLINE_AUC_COL: str = "auc_roc"
HONEST_AUC_COL: str = "auc_roc_real_neg_only"

# Amendment 6 target panel: the same nine viruses as CANON in
# scripts/compute_loo_binding_confound.py. On the shipped artifact
# results/per_virus_eval_v5_mode31.csv the filter n_pos >= 100 reproduces this panel
# exactly, but the panel is the definition and the filter is only a coincidence of
# that artifact.
TARGET_PANEL: frozenset[str] = frozenset(
    {"CMV", "DENV", "EBV", "HBV", "HCV", "HIV-1", "HPV", "IAV", "SARS-CoV-2"}
)

# One-sided 95% normal deviate. Used only by chance_ceiling.
FLOOR_Z: float = 1.6448536269514722

# Amendment 6 levels. CARRIED OVER UNCHANGED FROM THE CONTAMINATED SCALE; THEY ARE
# NOT RE-DERIVED HERE, DELIBERATELY. They entered the repo in 317b5d4 (2026-06-25)
# citing Amendment 6 of the owner's planning record, which is not tracked: a grep
# finds carriers and no derivation anywhere in the tree. CHANGELOG.md records
# "HPV >= 0.58 (achieved 0.598)", so the level originally tracked a then-measured
# value. Re-deriving a level from today's results would repeat exactly that mistake
# on a new scale, so this module does not do it. The LEVEL is an owner policy
# quantity and an input to an open venue decision; the COLUMN it is compared against
# is a technical defect, and that is what is fixed here. What IS derived is the
# validity floor below (chance_ceiling), the minimum level that can mean anything.
#
# Applying an unchanged level to the honest column is strictly STRICTER, never
# looser. On the shipped artifact EBV moves from PASS (auc_roc 0.711) to FAIL
# (honest 0.556) and HPV stays FAIL (0.482 on both, since it has no decoys). No
# virus is made to pass by this change.
EXIT_CRITERION: dict[str, dict[str, float]] = {
    "EBV": {HONEST_AUC_COL: 0.57},
    "HPV": {HONEST_AUC_COL: 0.58},
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def _safe_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return float(auc(rec, prec))


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Non-parametric bootstrap CI for a scalar metric.

    Returns (point_estimate, lower_ci, upper_ci) at CI_LEVEL confidence.
    Bootstrap samples that are single-class reuse the point estimate to
    avoid NaN contamination in the CI distribution.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_score)
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        samples[i] = point if len(np.unique(yt)) < 2 else metric_fn(yt, ys)
    alpha = (1.0 - CI_LEVEL) / 2.0
    lower = float(np.quantile(samples, alpha))
    upper = float(np.quantile(samples, 1.0 - alpha))
    return float(point), lower, upper


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float | None:
    """ECE via equal-width probability bins.

    Returns None when scores are outside [0, 1] (uncalibrated raw scores).
    Lower ECE is better; a perfectly calibrated model scores 0.
    """
    if y_prob.min() < -1e-6 or y_prob.max() > 1.0 + 1e-6:
        return None
    y_prob = np.clip(y_prob, 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include the right edge only in the last bin so a score of exactly 1.0
        # lands somewhere. A half-open final bin drops those rows from every bin
        # while n still counts them, which silently understates ECE. Matches
        # expected_calibration_error in scripts/fit_calibrator.py, which
        # already does this.
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def precision_at_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    recall_threshold: float = 0.10,
) -> float | None:
    """Highest precision achievable when recall >= recall_threshold.

    Returns None when both classes are not present or threshold is unreachable.
    """
    if len(np.unique(y_true)) < 2:
        return None
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    mask = rec >= recall_threshold
    if not mask.any():
        return None
    return float(prec[mask].max())


# ---------------------------------------------------------------------------
# Per-virus evaluation
# ---------------------------------------------------------------------------


def evaluate_virus(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "label",
    origin_col: str = "negative_origin",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
) -> dict:
    """Compute all six Amendment 6 mandatory metrics for one virus slice."""
    y_true = df[label_col].to_numpy(dtype=float)
    y_score = df[score_col].to_numpy(dtype=float)

    n_pos = int((y_true == 1).sum())
    n_neg_total = int((y_true == 0).sum())

    if origin_col in df.columns:
        real_neg_mask = (y_true == 0) & (df[origin_col].fillna("").isin(REAL_NEG_ORIGINS).to_numpy())
        n_neg_real = int(real_neg_mask.sum())
        n_neg_decoy = n_neg_total - n_neg_real
    else:
        real_neg_mask = np.zeros(len(df), dtype=bool)
        n_neg_real = 0
        n_neg_decoy = n_neg_total

    result: dict = {
        "n_pos": n_pos,
        "n_neg_real": n_neg_real,
        "n_neg_decoy": n_neg_decoy,
        "n_total": len(df),
    }

    can_eval = len(np.unique(y_true)) >= 2
    can_bootstrap = can_eval and len(df) >= MIN_SAMPLES_DEFAULT

    # Metric 1+2: AUC-ROC and AUC-PR with bootstrap CIs
    for metric_name, fn in (("auc_roc", _safe_auc_roc), ("auc_pr", _safe_auc_pr)):
        if can_bootstrap:
            point, lo, hi = bootstrap_metric(y_true, y_score, fn, n_bootstrap, seed)
        elif can_eval:
            point, lo, hi = fn(y_true, y_score), float("nan"), float("nan")
        else:
            point = lo = hi = float("nan")
        result[metric_name] = point
        result[f"{metric_name}_lower"] = lo
        result[f"{metric_name}_upper"] = hi

    # Metric 3: Precision at 10% recall
    result["precision_at_10pct_recall"] = (
        precision_at_recall(y_true, y_score, 0.10) if can_eval else None
    )

    # Metric 4: ECE
    result["ece"] = expected_calibration_error(y_true, y_score) if can_eval else None

    # Metric 5: AUC-ROC by peptide length (9-mer vs non-9-mer)
    if "peptide" in df.columns:
        lengths = df["peptide"].fillna("").str.len()
        for group_name, mask_series in (
            ("9mer", lengths == 9),
            ("non9mer", lengths != 9),
        ):
            mask = mask_series.to_numpy()
            yt_g, ys_g = y_true[mask], y_score[mask]
            key = f"auc_roc_{group_name}"
            if len(yt_g) >= MIN_SAMPLES_DEFAULT and len(np.unique(yt_g)) >= 2:
                result[key] = _safe_auc_roc(yt_g, ys_g)
            else:
                result[key] = None
    else:
        result["auc_roc_9mer"] = None
        result["auc_roc_non9mer"] = None

    # Metric 6: AUC-ROC on real-tested-negative subset (positives + real negatives only)
    pos_mask = y_true == 1
    real_subset = pos_mask | real_neg_mask
    yt_r, ys_r = y_true[real_subset], y_score[real_subset]
    if len(yt_r) >= MIN_SAMPLES_DEFAULT and len(np.unique(yt_r)) >= 2:
        result["auc_roc_real_neg_only"] = _safe_auc_roc(yt_r, ys_r)
    else:
        result["auc_roc_real_neg_only"] = None

    return result


def evaluate_all_viruses(
    df: pd.DataFrame,
    score_col: str,
    virus_col: str = "virus",
    label_col: str = "label",
    origin_col: str = "negative_origin",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    min_virus_size: int = MIN_SAMPLES_DEFAULT,
    seed: int = 42,
    logger: logging.Logger | None = None,
) -> dict[str, dict]:
    """Return per-virus metric dicts for all viruses meeting min_virus_size."""
    log = logger or logging.getLogger("evaluate_per_virus")
    results: dict[str, dict] = {}
    for virus, group in df.groupby(virus_col):
        if len(group) < min_virus_size:
            log.warning("Skipping %s: %d rows < min %d", virus, len(group), min_virus_size)
            continue
        log.info("Evaluating %s  (%d rows)", virus, len(group))
        results[str(virus)] = evaluate_virus(
            group.reset_index(drop=True),
            score_col=score_col,
            label_col=label_col,
            origin_col=origin_col,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    return results


# ---------------------------------------------------------------------------
# Exit criterion
# ---------------------------------------------------------------------------


def chance_ceiling(n_pos: int, n_neg: int, z: float = FLOOR_Z) -> float | None:
    """Derived per-virus validity floor: the highest AUC-ROC chance alone reaches.

    DERIVATION, reproducible from tracked inputs alone. Under the null "the scores
    carry no information about the label", AUC-ROC is the normalised Mann-Whitney U
    statistic. Its exact null mean is 0.5 and its exact null variance with no ties is

        Var_0 = (n_pos + n_neg + 1) / (12 * n_pos * n_neg)

    so the one-sided 95% upper bound on what a chance-level model scores at that
    sample size is

        floor = 0.5 + 1.6449 * sqrt(Var_0)

    A threshold BELOW this number cannot separate a working model from a coin flip
    at that virus's n, so no defensible level sits below it.

    SECOND INSTRUMENT, and the exact limits of its agreement. With no ties, Var_0 is
    the EXACT null variance rather than an approximation; only the normal
    95th-percentile step approximates anything. Ties make the exact null variance
    strictly smaller,

        Var_tied = ((N + 1) - sum(t**3 - t) / (N * (N - 1))) / (12 * n_pos * n_neg)

    with N = n_pos + n_neg and t running over tie-group sizes, so the tie-free form
    never UNDERSTATES the null spread. A label-permutation null agrees with the
    returned value to within Monte Carlo error: at 4,000 permutations on the shipped
    honest-scale slices the closed form was the larger in all four cases, by 0.0003
    to 0.0013 (EBV 0.5627 vs 0.5624, HPV 0.5539 vs 0.5528, SARS-CoV-2 0.5179 vs
    0.5176, HBV 0.5410 vs 0.5397).

    Do NOT read that as a guarantee on the PERCENTILE. An earlier draft of this
    docstring claimed the closed form is always the larger, and a test written to
    that claim failed: at 2,000 permutations on a heavily tied synthetic slice the
    permutation value came out 0.0018 higher, which is about one Monte Carlo
    standard error for a 95th percentile at that many reps. Raising the same case to
    40,000 permutations restores the ordering (0.564438 against 0.564167). The
    variance inequality is exact; the percentile ordering is not.
    tests/test_evaluate_per_virus.py checks the variance inequality deterministically
    and the percentile agreement only to a stated tolerance.

    WHY THIS IS NOT CIRCULAR. The expression reads n_pos and n_neg and nothing else.
    It never touches a score, a label ordering or an observed AUC, so it cannot be
    moved by making the model better or worse and it cannot be tuned to let a named
    virus pass. On the shipped artifact it lets NEITHER gated virus pass: HPV honest
    0.4820 against floor 0.5539, EBV honest 0.5557 against floor 0.5627.

    WHAT IT IS NOT, stated rather than implied. This is a validity FLOOR, not an exit
    target. Clearing it means only "distinguishable from chance", which for
    SARS-CoV-2 (floor 0.5179 on 2,473 positives and 980 real negatives) is a far
    weaker claim than the exit criterion makes; reporting a virus as having PASSED
    because it cleared its floor would overstate the result, and this module does not
    do that. It also carries no multiplicity correction across the adjudicated
    viruses, because the criterion is decided per virus and not as a family, and it
    says nothing about whether the honest column is free of any other confound.

    Returns None when either arm is empty, where the quantity is undefined.
    """
    if n_pos < 1 or n_neg < 1:
        return None
    var_null = (n_pos + n_neg + 1.0) / (12.0 * n_pos * n_neg)
    return float(0.5 + z * np.sqrt(var_null))


def _virus_floor(metrics: dict) -> float | None:
    """chance_ceiling on the HONEST-scale arms: positives and REAL negatives."""
    try:
        n_pos = int(metrics.get("n_pos") or 0)
        n_neg = int(metrics.get("n_neg_real") or 0)
    except (TypeError, ValueError):
        return None
    return chance_ceiling(n_pos, n_neg)


def _honest_value(metrics: dict) -> float | None:
    """HONEST_AUC_COL as a real float, or None when it is absent or NaN."""
    val = metrics.get(HONEST_AUC_COL)
    if val is None:
        return None
    val = float(val)
    return None if val != val else val  # nan != nan


def check_exit_criterion(
    results: dict[str, dict],
) -> tuple[bool, str]:
    """Check the Amendment 6 exit criterion on the HONEST (decoy-free) scale.

    Reads HONEST_AUC_COL, never HEADLINE_AUC_COL. The contaminated value is printed
    beside it, labelled, so the contrast stays visible; it is not gated on. See the
    constants block for the coverage measurements that make auc_roc unusable as a
    gate wherever decoys are present.

    Both viruses use the point estimate rather than the CI lower bound: with only 72
    real negatives for EBV, CI-lower is dominated by bootstrap sampling noise and is
    unachievable at any realistic AUC. Point estimate is the correct instrument for
    sparse-negative viruses.

    Fails closed three ways. A missing virus fails; an undefined honest value fails
    rather than falling back to the contaminated column; and a level sitting below
    its own derived validity floor fails, because a pass against such a level would
    not be distinguishable from chance.

    Returns (passed, detail_message).
    """
    msgs: list[str] = []
    passed = True

    for virus in ("EBV", "HPV"):
        level = EXIT_CRITERION[virus][HONEST_AUC_COL]
        metrics = results.get(virus, {})
        if not metrics:
            msgs.append(f"{virus}: no results (too few rows or missing)")
            passed = False
            continue

        floor = _virus_floor(metrics)
        if floor is not None and level < floor:
            msgs.append(
                f"{virus}: threshold {level} is BELOW its derived validity floor "
                f"{floor:.4f}, so a pass would not be distinguishable from chance"
            )
            passed = False
            continue

        honest = _honest_value(metrics)
        if honest is None:
            msgs.append(
                f"{virus}: {HONEST_AUC_COL} undefined (too few real negatives); the "
                f"criterion cannot be evaluated on the honest scale"
            )
            passed = False
            continue

        ok = honest >= level
        if not ok:
            passed = False
        msgs.append(
            f"{virus} honest AUC-ROC={honest:.3f} ({'PASS' if ok else 'FAIL'},"
            f" threshold>={level}, chance floor {_fmt(floor, 4)})"
            f" [contaminated auc_roc={_fmt(metrics.get(HEADLINE_AUC_COL))}, not gated]"
        )

    return passed, "; ".join(msgs)


def adjudicate_viruses(results: dict[str, dict]) -> list[dict]:
    """Adjudicate every two-class virus on BOTH scales plus the derived floor.

    "Two-class" means HEADLINE_AUC_COL is defined, i.e. the slice carries both
    labels. Each row names its own populations (in_target_panel, decoy_free) so any
    count drawn from this list can state the population it came from. On the shipped
    artifact "three decoy-free viruses" is correct for the nine-virus target panel
    and "six" is correct for all twelve two-class viruses; both describe the same
    data, and an unlabelled count is ambiguous between them.

    exit_status is populated only for the viruses EXIT_CRITERION names. For every
    other virus the only derived verdict available is beats_chance_floor, which is a
    validity check and NOT an exit criterion (see chance_ceiling).
    """
    rows: list[dict] = []
    for virus, metrics in sorted(results.items()):
        raw = metrics.get(HEADLINE_AUC_COL)
        if raw is None or float(raw) != float(raw):
            continue
        headline = float(raw)
        honest = _honest_value(metrics)
        floor = _virus_floor(metrics)
        level = EXIT_CRITERION.get(virus, {}).get(HONEST_AUC_COL)
        n_decoy = int(metrics.get("n_neg_decoy") or 0)

        exit_status: str | None = None
        if level is not None:
            if floor is not None and level < floor:
                exit_status = "INVALID"
            elif honest is None:
                exit_status = "UNDEFINED"
            else:
                exit_status = "PASS" if honest >= level else "FAIL"

        rows.append(
            {
                "virus": virus,
                "n_pos": int(metrics.get("n_pos") or 0),
                "n_neg_real": int(metrics.get("n_neg_real") or 0),
                "n_neg_decoy": n_decoy,
                "in_target_panel": virus in TARGET_PANEL,
                "decoy_free": n_decoy == 0,
                "auc_roc_contaminated": headline,
                "auc_roc_honest": honest,
                "decoy_inflation": None if honest is None else headline - honest,
                "chance_floor": floor,
                "beats_chance_floor": (
                    None if (honest is None or floor is None) else bool(honest >= floor)
                ),
                "exit_threshold": level,
                "exit_status": exit_status,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def _fmt(val: object, decimals: int = 3) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def format_table(results: dict[str, dict]) -> str:
    """ASCII table of per-virus metrics."""
    header = (
        f"{'Virus':<12} {'n_pos':>6} {'n_neg_r':>8} {'n_neg_d':>8}"
        f" {'AUC-ROC':>8} {'[95%CI]':>16}"
        f" {'AUC-PR':>8} {'[95%CI]':>16}"
        f" {'P@R10%':>7} {'ECE':>6}"
        f" {'ROC_9m':>7} {'ROC_r':>6}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for virus, m in sorted(results.items()):
        ci_roc = f"[{_fmt(m.get('auc_roc_lower'))},{_fmt(m.get('auc_roc_upper'))}]"
        ci_pr = f"[{_fmt(m.get('auc_pr_lower'))},{_fmt(m.get('auc_pr_upper'))}]"
        rows.append(
            f"{virus:<12}"
            f" {m.get('n_pos', 0):>6}"
            f" {m.get('n_neg_real', 0):>8}"
            f" {m.get('n_neg_decoy', 0):>8}"
            f" {_fmt(m.get('auc_roc')):>8}"
            f" {ci_roc:>16}"
            f" {_fmt(m.get('auc_pr')):>8}"
            f" {ci_pr:>16}"
            f" {_fmt(m.get('precision_at_10pct_recall')):>7}"
            f" {_fmt(m.get('ece')):>6}"
            f" {_fmt(m.get('auc_roc_9mer')):>7}"
            f" {_fmt(m.get('auc_roc_real_neg_only')):>6}"
        )
    return "\n".join(rows)


def format_adjudication_table(results: dict[str, dict]) -> str:
    """Both AUC-ROC scales side by side, with the derived floor and the verdict.

    Every count in the footer names its population, because the two natural
    populations here give different right answers to the same question.
    """
    adj = adjudicate_viruses(results)
    header = (
        f"{'Virus':<16} {'panel':>5} {'n_pos':>6} {'n_negR':>7} {'n_negD':>7}"
        f" {'ROC_all':>8} {'ROC_real':>9} {'inflation':>10}"
        f" {'floor':>7} {'>floor':>7} {'thresh':>7} {'exit':>8}"
    )
    sep = "-" * len(header)
    rows = [
        "Per-virus adjudication. ROC_all is the CONTAMINATED column (all negatives,",
        "decoys included); ROC_real is the HONEST column (real assay-confirmed",
        "negatives only) and is the one the exit criterion is decided on. floor is the",
        "derived validity floor (chance_ceiling); clearing it means only that the virus",
        "is distinguishable from chance, which is NOT the exit criterion.",
        "",
        header,
        sep,
    ]
    for r in adj:
        beats = r["beats_chance_floor"]
        rows.append(
            f"{r['virus']:<16}"
            f" {('T' if r['in_target_panel'] else '-'):>5}"
            f" {r['n_pos']:>6} {r['n_neg_real']:>7} {r['n_neg_decoy']:>7}"
            f" {_fmt(r['auc_roc_contaminated'], 4):>8}"
            f" {_fmt(r['auc_roc_honest'], 4):>9}"
            f" {_fmt(r['decoy_inflation'], 4):>10}"
            f" {_fmt(r['chance_floor'], 4):>7}"
            f" {('N/A' if beats is None else ('yes' if beats else 'no')):>7}"
            f" {('-' if r['exit_threshold'] is None else _fmt(r['exit_threshold'], 2)):>7}"
            f" {(r['exit_status'] or '-'):>8}"
        )

    panel = [r for r in adj if r["in_target_panel"]]
    rows.append(sep)
    rows.append(
        f"Populations: {len(adj)} two-class viruses, of which {len(panel)} are in the"
        f" Amendment 6 target panel."
    )
    for label, pop in (("all two-class", adj), ("target panel", panel)):
        if not pop:
            continue
        honest = [r["auc_roc_honest"] for r in pop if r["auc_roc_honest"] is not None]
        cont = [r["auc_roc_contaminated"] for r in pop]
        beats = sum(1 for r in pop if r["beats_chance_floor"])
        free = sum(1 for r in pop if r["decoy_free"])
        rows.append(
            f"  {label} (n={len(pop)}): mean ROC_all={np.mean(cont):.4f},"
            f" mean ROC_real={np.mean(honest) if honest else float('nan'):.4f},"
            f" {beats}/{len(pop)} above their own chance floor,"
            f" {free}/{len(pop)} decoy-free."
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def compare_predictions(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    score_col_a: str,
    score_col_b: str,
    label_col: str = "label",
    virus_col: str = "virus",
    logger: logging.Logger | None = None,
) -> dict[str, dict]:
    """Per-virus paired bootstrap comparison of two models.

    Both DataFrames must have the same row order (same OOF partition).
    Uses src.statistical_bootstrap.paired_bootstrap_comparison.
    """
    log = logger or logging.getLogger("evaluate_per_virus")

    if len(df_a) != len(df_b):
        log.error(
            "Row count mismatch: primary=%d compare=%d; cannot do paired comparison",
            len(df_a),
            len(df_b),
        )
        return {}

    merged = df_a[[label_col, virus_col, score_col_a]].copy()
    merged["_score_b"] = df_b[score_col_b].to_numpy()

    comparisons: dict[str, dict] = {}
    for virus, group in merged.groupby(virus_col):
        if len(group) < MIN_SAMPLES_DEFAULT:
            continue
        log.info("Comparing models on %s (%d rows)", virus, len(group))
        result = paired_bootstrap_comparison(
            group.reset_index(drop=True),
            ref_col=score_col_a,
            comp_col="_score_b",
            label_col=label_col,
        )
        comparisons[str(virus)] = result
    return comparisons


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate_per_virus.py",
        description=(
            "Per-virus AUC evaluation with bootstrap CIs. "
            "Implements the Amendment 6 Week 3 protocol."
        ),
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        metavar="PATH",
        help="OOF predictions CSV (label, score, virus, peptide, negative_origin).",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        metavar="PATH",
        help="Second model OOF CSV for paired comparison (same row order).",
    )
    parser.add_argument(
        "--score-col",
        default="score",
        metavar="COL",
        help="Score column in --predictions (default: score).",
    )
    parser.add_argument(
        "--compare-score-col",
        default="score",
        metavar="COL",
        help="Score column in --compare file (default: score).",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        metavar="COL",
    )
    parser.add_argument(
        "--virus-col",
        default="virus",
        metavar="COL",
    )
    parser.add_argument(
        "--origin-col",
        default="negative_origin",
        metavar="COL",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write full results to JSON (optional).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write per-virus metrics table to CSV (optional).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help=f"Bootstrap resamples (default: {DEFAULT_N_BOOTSTRAP}).",
    )
    parser.add_argument(
        "--min-virus-size",
        type=int,
        default=MIN_SAMPLES_DEFAULT,
        help=f"Skip viruses with fewer rows (default: {MIN_SAMPLES_DEFAULT}).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    random.seed(42)
    np.random.seed(42)

    args = parse_args(argv)
    setup_logging(args.verbose)
    logger = logging.getLogger("evaluate_per_virus")

    if not args.predictions.exists():
        logger.error("Predictions file not found: %s", args.predictions)
        return 1

    df = pd.read_csv(args.predictions)

    if args.score_col not in df.columns:
        logger.error("Score column %r not found. Available: %s", args.score_col, list(df.columns))
        return 1
    if args.label_col not in df.columns:
        logger.error("Label column %r not found.", args.label_col)
        return 1
    if args.virus_col not in df.columns:
        logger.error("Virus column %r not found.", args.virus_col)
        return 1

    results = evaluate_all_viruses(
        df,
        score_col=args.score_col,
        virus_col=args.virus_col,
        label_col=args.label_col,
        origin_col=args.origin_col,
        n_bootstrap=args.n_bootstrap,
        min_virus_size=args.min_virus_size,
        logger=logger,
    )

    if not results:
        logger.error("No virus had sufficient rows to evaluate.")
        return 1

    print("\n" + format_table(results))
    print("\n" + format_adjudication_table(results))

    passed, criterion_msg = check_exit_criterion(results)
    print(f"\nAmendment 6 exit criterion: {'PASS' if passed else 'FAIL'}")
    print(f"  {criterion_msg}")

    comparisons: dict[str, dict] = {}
    if args.compare is not None:
        if not args.compare.exists():
            logger.error("Compare file not found: %s", args.compare)
            return 1
        df_b = pd.read_csv(args.compare)
        comparisons = compare_predictions(
            df,
            df_b,
            score_col_a=args.score_col,
            score_col_b=args.compare_score_col,
            label_col=args.label_col,
            virus_col=args.virus_col,
            logger=logger,
        )
        if comparisons:
            print("\n--- Paired bootstrap comparison (primary vs compare) ---")
            for virus, comp in sorted(comparisons.items()):
                delta_roc = comp.get("delta_auc_roc", {})
                print(
                    f"  {virus:<12}  delta AUC-ROC={delta_roc.get('mean', float('nan')):.4f}"
                    f"  p={delta_roc.get('p_value', float('nan')):.3f}"
                )

    output = {
        "predictions_file": str(args.predictions),
        "n_bootstrap": args.n_bootstrap,
        "min_virus_size": args.min_virus_size,
        "per_virus": results,
        "adjudication": adjudicate_viruses(results),
        "exit_criterion_passed": passed,
        "exit_criterion_detail": criterion_msg,
        "comparisons": comparisons,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        logger.info("Wrote JSON: %s", args.output_json)

    if args.output_csv:
        rows = [{"virus": v, **m} for v, m in results.items()]
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        logger.info("Wrote CSV: %s", args.output_csv)

    # rc=0 criterion met; rc=2 ran OK but criterion not met; rc=1 fatal error
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
