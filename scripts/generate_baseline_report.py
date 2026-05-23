"""
SESTRAV 5-Model Baseline Comparison Report
==========================================

Reads training_results.csv (RF + XGBoost + ANN) and any available GNN result
CSVs to produce a consolidated comparison table for the external validation phase.

This is the frozen legacy (allele-blind, 30-feature) baseline that will be used
to measure the improvement from allele-aware training.

Usage:
    python scripts/generate_baseline_report.py
    python scripts/generate_baseline_report.py --output results/baseline_report.csv
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


METRIC_DISPLAY = {
    "auc_roc": "AUC-ROC",
    "auc_pr":  "AUC-PR",
    "issr_10": "ISSR@10",
    "issr_25": "ISSR@25",
}


def load_training_results(path):
    """Load RF, XGBoost, and ANN CV metrics from training_results.csv."""
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path).set_index("metric")
    models = {}
    if "rf_cv_mean" in df.columns:
        models["RandomForest (RF-30)"] = {
            "mean": df["rf_cv_mean"].to_dict(),
            "std":  df["rf_cv_std"].to_dict(),
        }
    if "xgb_cv_mean" in df.columns:
        models["XGBoost (XGB-30)"] = {
            "mean": df["xgb_cv_mean"].to_dict(),
            "std":  df["xgb_cv_std"].to_dict(),
        }
    if "ann_cv_mean" in df.columns:
        models["ANN 256-128-64 (30-feat)"] = {
            "mean": df["ann_cv_mean"].to_dict(),
            "std":  df["ann_cv_std"].to_dict(),
        }
    return models


def load_gnn_results(models_dir):
    """Load GNN benchmark CSVs if available."""
    models = {}

    seq_path = os.path.join(models_dir, "gnn_sequence_benchmark.csv")
    if os.path.isfile(seq_path):
        df = pd.read_csv(seq_path)
        for _, row in df.iterrows():
            name = row["model"]
            models[name] = {
                "mean": {
                    "auc_roc": row["auc_roc_mean"],
                    "auc_pr":  row["auc_pr_mean"],
                    "issr_10": row["issr_10_mean"],
                    "issr_25": row["issr_25_mean"],
                },
                "std": {
                    "auc_roc": row["auc_roc_std"],
                    "auc_pr":  row["auc_pr_std"],
                    "issr_10": 0.0,
                    "issr_25": 0.0,
                },
            }

    bi_path = os.path.join(models_dir, "gnn_bipartite_benchmark.csv")
    if os.path.isfile(bi_path):
        df = pd.read_csv(bi_path)
        for _, row in df.iterrows():
            name = row["model"]
            models[name] = {
                "mean": {
                    "auc_roc": row["auc_roc_mean"],
                    "auc_pr":  row["auc_pr_mean"],
                    "issr_10": row["issr_10_mean"],
                    "issr_25": row["issr_25_mean"],
                },
                "std": {
                    "auc_roc": row["auc_roc_std"],
                    "auc_pr":  row["auc_pr_std"],
                    "issr_10": 0.0,
                    "issr_25": 0.0,
                },
            }
    return models


def build_comparison_table(all_models, metrics=None):
    """Build a wide-format comparison DataFrame."""
    if metrics is None:
        metrics = ["auc_roc", "auc_pr", "issr_10", "issr_25"]

    rows = []
    for model_name, data in all_models.items():
        row = {"Model": model_name}
        for m in metrics:
            mean_val = data["mean"].get(m, float("nan"))
            std_val  = data["std"].get(m, float("nan"))
            row[METRIC_DISPLAY.get(m, m)] = (
                f"{mean_val:.4f} +/- {std_val:.4f}"
                if not (np.isnan(mean_val) or np.isnan(std_val))
                else "N/A"
            )
            row[f"{METRIC_DISPLAY.get(m, m)}_mean"] = mean_val
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate 5-model baseline comparison report")
    parser.add_argument("--training-results", default="models/training_results.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output", default="results/baseline_comparison_report.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    print(f"\nSESTRAV 5-Model Baseline Comparison Report")
    print(f"{'=' * 70}")

    all_models = {}

    # Load RF / XGBoost / ANN from training_results.csv
    tab_models = load_training_results(args.training_results)
    if tab_models:
        all_models.update(tab_models)
        print(f"Loaded tabular models from: {args.training_results}")
    else:
        print(f"WARNING: {args.training_results} not found or incomplete.")

    # Load GNN results
    gnn_models = load_gnn_results(args.models_dir)
    if gnn_models:
        all_models.update(gnn_models)
        print(f"Loaded {len(gnn_models)} GNN model(s) from: {args.models_dir}/gnn_*.csv")
    else:
        print(f"Note: No GNN benchmark results found in {args.models_dir}/")
        print(f"      Run 'python -m src.gnn_benchmark --data immunogenicity_dataset.csv' to generate them.")

    if not all_models:
        print("ERROR: No model results found. Run training and benchmarks first.")
        sys.exit(1)

    # Build comparison table
    table = build_comparison_table(all_models)
    table_sorted = table.sort_values("AUC-PR_mean", ascending=False).drop(
        columns=[c for c in table.columns if c.endswith("_mean")],
    )

    print(f"\n{'=' * 70}")
    print("5-fold CV Performance Comparison (Legacy Allele-Blind Baseline):")
    print(f"{'=' * 70}")
    print(table_sorted.to_string(index=False))

    # Save
    table_sorted.to_csv(args.output, index=False)
    print(f"\nComparison table saved to: {args.output}")

    # Best model summary
    best_row = table.sort_values("AUC-PR_mean", ascending=False).iloc[0]
    print(f"\n{'=' * 70}")
    print(f"LEGACY BASELINE FROZEN:")
    print(f"  Best model: {best_row['Model']}")
    print(f"  Best AUC-PR: {best_row['AUC-PR_mean']:.4f}")
    print(f"  RF (primary) AUC-PR: {all_models.get('RandomForest (RF-30)', {}).get('mean', {}).get('auc_pr', float('nan')):.4f}")
    print(f"\nThis baseline is now frozen for comparison with allele-aware models.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
