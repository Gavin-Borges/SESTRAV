"""
Compute ANN 5-fold CV summary metrics from existing OOF predictions.

Reads models/ann_oof_predictions.csv (produced by a prior ann_benchmark run),
computes per-fold metrics, summarizes mean ± std, and appends ANN columns to
models/training_results.csv so all 3 models (RF, XGBoost, ANN) can be compared.

Usage:
    python scripts/compute_ann_baseline_summary.py
    python scripts/compute_ann_baseline_summary.py --oof-file models/ann_oof_predictions.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Allow running from SESTRAV-Dev root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate_metrics import evaluate


def compute_fold_metrics(oof_df):
    """Compute per-fold metrics from an OOF predictions DataFrame.

    Args:
        oof_df: DataFrame with columns: peptide, label, score, fold

    Returns:
        List of metric dicts, one per fold.
    """
    folds = sorted(oof_df["fold"].unique())
    fold_metrics = []
    for fold in folds:
        fold_data = oof_df[oof_df["fold"] == fold]
        y_true = fold_data["label"].values
        y_score = fold_data["score"].values
        m = evaluate(y_true, y_score)
        m["fold"] = fold
        fold_metrics.append(m)
        print(f"  Fold {fold}: AUC-ROC={m['auc_roc']:.4f}  AUC-PR={m['auc_pr']:.4f}  "
              f"ISSR@10={m['issr_10']:.4f}  ISSR@25={m['issr_25']:.4f}")
    return fold_metrics


def summarize(fold_metrics):
    """Compute mean ± std across folds, excluding the 'fold' key."""
    keys = [k for k in fold_metrics[0] if k != "fold"]
    avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    std = {k: float(np.std([m[k] for m in fold_metrics])) for k in keys}
    return avg, std


def main():
    parser = argparse.ArgumentParser(description="Compute ANN baseline CV summary")
    parser.add_argument(
        "--oof-file",
        default="models/ann_oof_predictions.csv",
        help="Path to ANN OOF predictions CSV (default: models/ann_oof_predictions.csv)",
    )
    parser.add_argument(
        "--results-file",
        default="models/training_results.csv",
        help="Path to training_results.csv to update (default: models/training_results.csv)",
    )
    parser.add_argument(
        "--output-summary",
        default="models/ann_cv_summary.csv",
        help="Path to write standalone ANN CV summary CSV",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.oof_file):
        print(f"ERROR: OOF file not found: {args.oof_file}")
        sys.exit(1)

    print(f"\nLoading ANN OOF predictions from: {args.oof_file}")
    oof_df = pd.read_csv(args.oof_file)
    print(f"  Loaded {len(oof_df)} rows across folds: {sorted(oof_df['fold'].unique())}")
    print(f"  Peptides: {oof_df['peptide'].nunique()} unique")
    print(f"  Class balance: {oof_df['label'].mean():.2%} positive")

    print(f"\n{'=' * 60}")
    print("ANN 5-fold Cross-Validation Metrics:")
    print(f"{'=' * 60}")
    fold_metrics = compute_fold_metrics(oof_df)
    avg, std = summarize(fold_metrics)

    print(f"\n  Mean:  " + "  ".join(f"{k}={v:.4f}" for k, v in avg.items()
                                      if k in ("auc_roc", "auc_pr", "issr_10", "issr_25")))
    print(f"  Stdev: " + "  ".join(f"{k}={v:.4f}" for k, v in std.items()
                                      if k in ("auc_roc", "auc_pr", "issr_10", "issr_25")))

    # Write standalone ANN summary
    summary_rows = []
    for key in avg:
        summary_rows.append({
            "metric": key,
            "ann_cv_mean": avg[key],
            "ann_cv_std": std[key],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_summary, index=False)
    print(f"\nANN CV summary written to: {args.output_summary}")

    # Merge into training_results.csv if it exists
    if os.path.isfile(args.results_file):
        results_df = pd.read_csv(args.results_file)
        ann_lookup = summary_df.set_index("metric")
        results_df["ann_cv_mean"] = results_df["metric"].map(ann_lookup["ann_cv_mean"])
        results_df["ann_cv_std"] = results_df["metric"].map(ann_lookup["ann_cv_std"])
        results_df.to_csv(args.results_file, index=False)
        print(f"Updated training_results.csv with ANN columns: {args.results_file}")
    else:
        print(f"WARNING: {args.results_file} not found — skipping merge.")
        print(f"         ANN-only summary is at {args.output_summary}")

    print(f"\n{'=' * 60}")
    print("ANN Baseline Summary (for external validation phase context):")
    print(f"{'=' * 60}")
    print(f"  AUC-ROC : {avg['auc_roc']:.4f} +/- {std['auc_roc']:.4f}")
    print(f"  AUC-PR  : {avg['auc_pr']:.4f} +/- {std['auc_pr']:.4f}")
    print(f"  ISSR@10 : {avg['issr_10']:.4f} +/- {std['issr_10']:.4f}")
    print(f"  ISSR@25 : {avg['issr_25']:.4f} +/- {std['issr_25']:.4f}")
    print(f"\nCompare vs legacy RF baseline (from training_results.csv):")
    print(f"  RF AUC-ROC: 0.7268  RF AUC-PR: 0.8317  RF ISSR@10: 0.8429")
    return avg, std


if __name__ == "__main__":
    main()
