"""
SESTRAV Statistical Bootstrapping Module.
Performs paired bootstrapping (N=10,000) to evaluate metric differences between predictors.

Why Paired Bootstrapping replaces DeLong's test for nested models:
DeLong's test assumes normal distributions and sample independence. For nested machine
learning models or models evaluated on overlapping/correlated peptide groupings, this
assumption is violated, rendering DeLong's test invalid and prone to false positives.
Paired bootstrapping resamples peptide indices (not independent labels) for both models
simultaneously, preserving the joint covariance between the predictions without distribution
assumptions, leading to mathematically sound confidence intervals and empirical p-values.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from src.evaluate_metrics import issr_at_k
from joblib import Parallel, delayed


def _bootstrap_resample_indices(
    seed: int, n_clean: int, n_resamples: int
) -> list[np.ndarray]:
    """Generate reproducible, distinct bootstrap draws in the parent process."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n_clean, size=n_clean) for _ in range(n_resamples)]


def _bootstrap_iter(y, ref_scores, comp_scores, idx):
    y_b = y[idx]
    if len(np.unique(y_b)) < 2:
        return None
    ref_b = ref_scores[idx]
    comp_b = comp_scores[idx]

    ap = average_precision_score(y_b, ref_b) - average_precision_score(y_b, comp_b)
    roc = roc_auc_score(y_b, ref_b) - roc_auc_score(y_b, comp_b)
    issr = issr_at_k(y_b, ref_b, 10) - issr_at_k(y_b, comp_b, 10)
    return (ap, roc, issr)


def paired_bootstrap_comparison(
    df: pd.DataFrame,
    ref_col: str,
    comp_col: str,
    label_col: str = "label",
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Perform paired bootstrapping over peptide indices to compare two models.
    Returns delta metrics, 95% confidence intervals, and empirical p-values.
    """
    # Extract clean vectors (drop rows with missing scores or labels)
    clean_df = df.dropna(subset=[ref_col, comp_col, label_col])
    n_clean = len(clean_df)
    if n_clean < 10:
        return {"error": "Too few clean samples for bootstrapping"}

    y = clean_df[label_col].values
    ref_scores = clean_df[ref_col].values
    comp_scores = clean_df[comp_col].values

    # Base metric calculations
    base_ref_ap = average_precision_score(y, ref_scores)
    base_comp_ap = average_precision_score(y, comp_scores)
    base_delta_ap = base_ref_ap - base_comp_ap

    base_ref_roc = roc_auc_score(y, ref_scores)
    base_comp_roc = roc_auc_score(y, comp_scores)
    base_delta_roc = base_ref_roc - base_comp_roc

    base_ref_issr = issr_at_k(y, ref_scores, 10)
    base_comp_issr = issr_at_k(y, comp_scores, 10)
    base_delta_issr = base_ref_issr - base_comp_issr

    indices = _bootstrap_resample_indices(seed, n_clean, n_resamples)
    results_parallel = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(_bootstrap_iter)(y, ref_scores, comp_scores, idx) for idx in indices
    )

    _ap: list[float] = []
    _roc: list[float] = []
    _issr: list[float] = []
    for res in results_parallel:
        if res is not None:
            _ap.append(res[0])
            _roc.append(res[1])
            _issr.append(res[2])

    deltas_ap = np.array(_ap)
    deltas_roc = np.array(_roc)
    deltas_issr = np.array(_issr)

    # Helper to compute CIs and empirical p-values
    def analyze_delta(deltas, base_delta):
        ci_low = float(np.percentile(deltas, 2.5))
        ci_high = float(np.percentile(deltas, 97.5))

        # Empirical two-sided p-value: fraction of deltas of opposite sign of base_delta
        # (or standard test of delta crossing zero)
        p_val = float(min(np.mean(deltas <= 0), np.mean(deltas >= 0)) * 2)
        sig = "yes" if ci_low > 0 or ci_high < 0 else "no"
        return {
            "delta_mean": float(np.mean(deltas)),
            "delta_base": float(base_delta),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_val,
            "significant_95": sig,
        }

    return {
        "n_samples": n_clean,
        "auc_pr": analyze_delta(deltas_ap, base_delta_ap),
        "auc_roc": analyze_delta(deltas_roc, base_delta_roc),
        "issr_10": analyze_delta(deltas_issr, base_delta_issr),
    }


def main():
    parser = argparse.ArgumentParser(description="Run paired bootstrap statistics")
    parser.add_argument("--merged-csv", required=True, help="Merged scores CSV file")
    parser.add_argument(
        "--ref-col", required=True, help="Reference model column name (e.g. rf_oof_score)"
    )
    parser.add_argument(
        "--comp-cols",
        required=True,
        help="Comma-separated comparison column names (e.g. prime_score,predig_score)",
    )
    parser.add_argument("--label-col", default="label", help="Ground-truth binary label column")
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=10000,
        help="Number of bootstrap resamples (default 10,000)",
    )
    parser.add_argument("--output-json", required=True, help="JSON output file for stats summary")
    args = parser.parse_args()

    df = pd.read_csv(args.merged_csv)
    cols = [c.strip() for c in args.comp_cols.split(",")]

    results = {}
    for col in cols:
        if col not in df.columns:
            print(f"[bootstrap] Column '{col}' not found inMerged CSV. Skipping.")
            continue
        print(
            f"[bootstrap] Comparing {args.ref_col} vs {col} (paired bootstrap, N={args.n_resamples})..."
        )
        res = paired_bootstrap_comparison(
            df,
            ref_col=args.ref_col,
            comp_col=col,
            label_col=args.label_col,
            n_resamples=args.n_resamples,
        )
        results[col] = res

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[bootstrap] Saved statistical results to {args.output_json}")


if __name__ == "__main__":
    main()
