"""
End-to-end SESTRAV bias/skew finalization runner.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd

from src.ann_benchmark import train_ann
from src.baseline_comparison import compare_methods
from src.data_bias_audit import _collect_raw_records, refresh_dataset, write_audit_reports
from src.final_validation_report import run_final_validation
from src.gold_standard_sensitivity import run_gold_standard_sensitivity
from src.train_classifier import train_models


def _gate_status(
    bias_summary: Dict,
    subgroup_csv: str,
    threshold_json: str,
    sensitivity_delta_csv: str,
) -> Dict[str, bool]:
    """Compute release-gate booleans from produced artifacts."""
    gates: Dict[str, bool] = {}

    subgroup_df = pd.read_csv(subgroup_csv) if os.path.isfile(subgroup_csv) else pd.DataFrame()
    severe_underperf = False
    if not subgroup_df.empty:
        ss = subgroup_df[subgroup_df["subgroup_key"] == "virus"].copy()
        if not ss.empty:
            severe_underperf = bool((ss["auc_pr"] < 0.55).fillna(False).any())
    gates["no_severe_subgroup_underperformance"] = not severe_underperf

    threshold_ok = False
    if os.path.isfile(threshold_json):
        with open(threshold_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        threshold_ok = bool(payload.get("overall_precision", 0.0) >= 0.55 and payload.get("overall_recall", 0.0) >= 0.55)
    gates["acceptable_threshold_tradeoff"] = threshold_ok

    sensitivity_ok = True
    if os.path.isfile(sensitivity_delta_csv):
        deltas = pd.read_csv(sensitivity_delta_csv)
        if not deltas.empty:
            sensitivity_ok = bool((deltas["delta_recovery_top25"] > -0.20).all())
    gates["gold_standard_not_brittle"] = sensitivity_ok

    provenance_ok = bool(bias_summary.get("n_total", 0) > 0 and bias_summary.get("raw_n_records", 0) > 0)
    gates["dataset_provenance_complete"] = provenance_ok
    gates["all_passed"] = all(gates.values())
    return gates


def run_bias_skew_finalization(
    source_data_dir: str,
    data_csv: str = "immunogenicity_dataset.csv",
    model_dir: str = "models",
    results_dir: str = "results",
    feature_mode: int = 30,
    binding_matrix_path: str = "models/peptide_binding_matrix.csv",
) -> str:
    """Execute the full finalization pipeline and return release summary path."""
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    provenance_csv = os.path.join(results_dir, "immunogenicity_provenance.csv")
    audit_csv = os.path.join(results_dir, "data_bias_audit_summary.csv")
    audit_md = os.path.join(results_dir, "data_bias_audit.md")
    raw_records = _collect_raw_records(source_data_dir, include_hpv11=False)
    refresh_dataset(
        source_data_dir=source_data_dir,
        output_csv=data_csv,
        provenance_csv=provenance_csv,
        include_hpv11=False,
    )
    _, bias_summary = write_audit_reports(
        dataset_csv=data_csv,
        raw_records=raw_records,
        output_csv=audit_csv,
        output_md=audit_md,
    )

    train_models(
        data_path=data_csv,
        model_dir=model_dir,
        n_cv_folds=5,
        random_state=42,
        feature_mode=feature_mode,
        binding_matrix_path=binding_matrix_path if feature_mode == 30 else None,
    )
    train_ann(
        data_path=data_csv,
        model_dir=model_dir,
        n_cv_folds=5,
        random_state=42,
    )

    compare_df = compare_methods(results_dir=results_dir, model_dir=model_dir)
    compare_df.to_csv(os.path.join(results_dir, "baseline_comparison.csv"), index=False)

    model_path = os.path.join(
        model_dir,
        "rf_30feature_integrated.joblib" if feature_mode == 30 else "rf_21feature_legacy.joblib",
    )
    run_final_validation(
        results_dir=results_dir,
        model_dir=model_dir,
        data_path=data_csv,
        binding_matrix_path=binding_matrix_path,
        model_path=model_path,
    )

    gs_sens_csv = os.path.join(results_dir, "gold_standard_sensitivity.csv")
    gs_sens_md = os.path.join(results_dir, "gold_standard_sensitivity.md")
    run_gold_standard_sensitivity(results_dir, gs_sens_csv, gs_sens_md)

    gates = _gate_status(
        bias_summary=bias_summary,
        subgroup_csv=os.path.join(model_dir, "training_subgroup_metrics.csv"),
        threshold_json=os.path.join(model_dir, "optimal_thresholds.json"),
        sensitivity_delta_csv=os.path.join(results_dir, "gold_standard_sensitivity_deltas.csv"),
    )

    threshold_payload = {}
    threshold_json_path = os.path.join(model_dir, "optimal_thresholds.json")
    if os.path.isfile(threshold_json_path):
        with open(threshold_json_path, "r", encoding="utf-8") as f:
            threshold_payload = json.load(f)

    summary_path = os.path.join(results_dir, "release_readiness_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(
            "# SESTRAV Bias/Skew Finalization Summary\n\n"
            "## Dataset updates\n"
            f"- Refreshed dataset: `{data_csv}`\n"
            f"- Provenance table: `{provenance_csv}`\n"
            f"- Bias audit: `{audit_csv}` and `{audit_md}`\n\n"
            "## Model/evaluation outputs\n"
            f"- Training CV summary: `{os.path.join(model_dir, 'training_results.csv')}`\n"
            f"- Subgroup CV summary: `{os.path.join(model_dir, 'training_subgroup_metrics.csv')}`\n"
            f"- ANN subgroup CV summary: `{os.path.join(model_dir, 'ann_subgroup_metrics.csv')}`\n"
            f"- Baseline comparison: `{os.path.join(results_dir, 'baseline_comparison.csv')}`\n"
            f"- H2 Tier A fold metrics: `{os.path.join(results_dir, 'h2_tier_a_fold_metrics.csv')}`\n"
            f"- H2 Tier A summary: `{os.path.join(results_dir, 'h2_tier_a_summary.csv')}`\n"
            f"- Gold-standard sensitivity: `{gs_sens_csv}`\n\n"
            "## Threshold decision\n"
            f"- Selected threshold: `{threshold_payload.get('threshold', np.nan):.4f}`\n"
            f"- Overall precision: `{threshold_payload.get('overall_precision', np.nan):.4f}`\n"
            f"- Overall recall: `{threshold_payload.get('overall_recall', np.nan):.4f}`\n"
            f"- Min subgroup F1 (virus): `{threshold_payload.get('min_subgroup_f1', np.nan):.4f}`\n\n"
            "## Release gate\n"
            f"- No severe subgroup underperformance: `{gates['no_severe_subgroup_underperformance']}`\n"
            f"- Acceptable threshold tradeoff: `{gates['acceptable_threshold_tradeoff']}`\n"
            f"- Gold-standard sensitivity non-brittle: `{gates['gold_standard_not_brittle']}`\n"
            f"- Dataset provenance complete: `{gates['dataset_provenance_complete']}`\n"
            f"- Final release-ready status: **`{gates['all_passed']}`**\n"
        )
    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SESTRAV bias/skew finalization bundle")
    parser.add_argument(
        "--source-data-dir",
        default=os.path.join("..", "..", "09_Data"),
        help="Path to raw IEDB source files used for refresh",
    )
    parser.add_argument("--data-csv", default="immunogenicity_dataset.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--feature-mode", type=int, default=30, choices=[21, 30])
    parser.add_argument("--binding-matrix", default="models/peptide_binding_matrix.csv")
    args = parser.parse_args()

    out = run_bias_skew_finalization(
        source_data_dir=args.source_data_dir,
        data_csv=args.data_csv,
        model_dir=args.model_dir,
        results_dir=args.results_dir,
        feature_mode=args.feature_mode,
        binding_matrix_path=args.binding_matrix,
    )
    print(f"Release summary written to: {out}")
