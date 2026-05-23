"""
Reproducible Tier A external validation review summary.

Reads processed artifacts from a frozen run directory and writes
processed/tier_a_analysis_summary.json for the analysis memo.

Usage:
    python -m src.external_validation_tier_a_review \\
        --run-dir results/external_tool_outputs/extval_20260520_1607_gb_tierA
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd


def _read_csv(path: str) -> pd.DataFrame:
    if os.path.isfile(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def build_summary(run_dir: str) -> dict:
    processed = os.path.join(run_dir, "processed")
    manifests = os.path.join(run_dir, "manifests")

    metrics_int = _read_csv(os.path.join(processed, "metrics_intersection.csv"))
    metrics_excl = _read_csv(os.path.join(processed, "metrics_overlap_excluded.csv"))
    metrics_only = _read_csv(os.path.join(processed, "metrics_overlap_only.csv"))
    fdr = _read_csv(os.path.join(processed, "fdr_primary_tests.csv"))
    virus_w = _read_csv(os.path.join(processed, "virus_weighted_metrics.csv"))
    length_df = _read_csv(os.path.join(processed, "length_stratified_auc_pr.csv"))
    holdout = _read_csv(os.path.join(processed, "holdout_spotlight.csv"))

    mcda_path = os.path.join(processed, "mcda_verdicts.json")
    mcda = []
    if os.path.isfile(mcda_path):
        mcda = json.loads(open(mcda_path, encoding="utf-8").read())

    coverage = {}
    cov_path = os.path.join(manifests, "coverage_summary.json")
    if os.path.isfile(cov_path):
        coverage = json.loads(open(cov_path, encoding="utf-8").read())

    overlap = {}
    ov_path = os.path.join(manifests, "training_overlap.json")
    if os.path.isfile(ov_path):
        overlap = json.loads(open(ov_path, encoding="utf-8").read())

    cross_virus_path = os.path.join(
        os.path.dirname(os.path.dirname(run_dir)), "external_validation_cross_virus.csv"
    )
    if not os.path.isfile(cross_virus_path):
        cross_virus_path = os.path.join("results", "external_validation_cross_virus.csv")
    cross_virus = _read_csv(cross_virus_path)

    def _tool_row(df: pd.DataFrame, tool: str) -> dict:
        if df.empty or "tool" not in df.columns:
            return {}
        row = df[df["tool"] == tool]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": os.path.basename(run_dir.rstrip("/\\")),
        "coverage": coverage,
        "training_overlap": overlap,
        "mcda_verdicts": mcda,
        "intersection_metrics": {
            "SESTRAV RF (30-feat)": _tool_row(metrics_int, "SESTRAV RF (30-feat)"),
            "PredIG-Path": _tool_row(metrics_int, "PredIG-Path"),
            "PRIME 2.1": _tool_row(metrics_int, "PRIME 2.1"),
            "Binding-only (max)": _tool_row(metrics_int, "Binding-only (max)"),
        },
        "overlap_excluded_n": int(metrics_excl["n_peptides"].iloc[0])
        if not metrics_excl.empty and "n_peptides" in metrics_excl.columns
        else None,
        "overlap_only_n": int(metrics_only["n_peptides"].iloc[0])
        if not metrics_only.empty and "n_peptides" in metrics_only.columns
        else None,
        "overlap_excluded_auc_pr": {
            "SESTRAV RF (30-feat)": _tool_row(metrics_excl, "SESTRAV RF (30-feat)").get("auc_pr"),
            "PRIME 2.1": _tool_row(metrics_excl, "PRIME 2.1").get("auc_pr"),
            "PredIG-Path": _tool_row(metrics_excl, "PredIG-Path").get("auc_pr"),
        },
        "overlap_only_auc_pr": {
            "SESTRAV RF (30-feat)": _tool_row(metrics_only, "SESTRAV RF (30-feat)").get("auc_pr"),
            "PRIME 2.1": _tool_row(metrics_only, "PRIME 2.1").get("auc_pr"),
            "PredIG-Path": _tool_row(metrics_only, "PredIG-Path").get("auc_pr"),
        },
        "virus_weighted_auc_pr": virus_w.set_index("tool")["weighted_auc_pr"].to_dict()
        if not virus_w.empty and "tool" in virus_w.columns
        else {},
        "fdr_primary_tests": fdr.to_dict(orient="records") if not fdr.empty else [],
        "cross_virus": cross_virus.to_dict(orient="records") if not cross_virus.empty else [],
        "holdout_n": len(holdout) if not holdout.empty else 0,
        "length_stratified_9mer_rf_auc_pr": None,
        "length_stratified_9mer_prime_auc_pr": None,
    }

    if not length_df.empty:
        nine = length_df[length_df["length_group"] == "9-mer"]
        rf = nine[nine["tool"] == "SESTRAV RF (30-feat)"]
        pr = nine[nine["tool"] == "PRIME 2.1 (max)"]
        if not rf.empty:
            summary["length_stratified_9mer_rf_auc_pr"] = float(rf["auc_pr"].iloc[0])
        if not pr.empty:
            summary["length_stratified_9mer_prime_auc_pr"] = float(pr["auc_pr"].iloc[0])

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier A external validation review summary")
    parser.add_argument(
        "--run-dir",
        default="results/external_tool_outputs/extval_20260520_1607_gb_tierA",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    out_path = args.output or os.path.join(run_dir, "processed", "tier_a_analysis_summary.json")
    summary = build_summary(run_dir)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[tier-a-review] Wrote {out_path}")


if __name__ == "__main__":
    main()
