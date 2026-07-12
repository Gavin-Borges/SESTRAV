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

Amendment 6 exit criterion (checked automatically):
  EBV: AUC-ROC point estimate >= 0.57
  HPV: AUC-ROC point estimate >= 0.58

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

# Amendment 6 exit criterion thresholds.
EXIT_CRITERION: dict[str, dict[str, float]] = {
    "EBV": {"auc_roc": 0.57},
    "HPV": {"auc_roc": 0.58},
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
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
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


def check_exit_criterion(
    results: dict[str, dict],
) -> tuple[bool, str]:
    """Check Amendment 6 exit criterion using thresholds from EXIT_CRITERION.

    Both EBV and HPV use point estimate (not CI-lower): with only 14 IEDB-sourced
    tested negatives for EBV, CI-lower is dominated by bootstrap sampling noise and
    is unachievable at any realistic AUC. Point estimate is the correct instrument
    for sparse-negative viruses.

    Returns (passed, detail_message).
    """
    msgs: list[str] = []
    passed = True

    ebv_threshold = EXIT_CRITERION["EBV"]["auc_roc"]
    hpv_threshold = EXIT_CRITERION["HPV"]["auc_roc"]

    ebv = results.get("EBV", {})
    if ebv:
        val = ebv.get("auc_roc", float("nan"))
        ok = val == val and val >= ebv_threshold  # nan == nan is False
        msgs.append(
            f"EBV AUC-ROC={val:.3f} ({'PASS' if ok else 'FAIL'}, threshold>={ebv_threshold})"
        )
        if not ok:
            passed = False
    else:
        msgs.append("EBV: no results (too few rows or missing)")
        passed = False

    hpv = results.get("HPV", {})
    if hpv:
        val = hpv.get("auc_roc", float("nan"))
        ok = val == val and val >= hpv_threshold
        msgs.append(
            f"HPV AUC-ROC={val:.3f} ({'PASS' if ok else 'FAIL'}, threshold>={hpv_threshold})"
        )
        if not ok:
            passed = False
    else:
        msgs.append("HPV: no results (too few rows or missing)")
        passed = False

    return passed, "; ".join(msgs)


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
