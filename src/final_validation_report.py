"""
SESTRAV Final Validation Bundle

Runs the core end-of-semester validation artifacts in one place:
  1) Gold-standard validation report
  2) Baseline comparison (RF/XGB/ANN vs binding-only)
  3) H2 Tier A labeled ISSR enrichment evaluation

This intentionally excludes long-term or optional analyses.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import pandas as pd

from src.baseline_comparison import compare_methods
from src.gold_standard import full_validation_report
from src.h2_tier_a_evaluation import run_h2_tier_a
from src.naming import canonical_output_filename, resolve_model_path


def run_final_validation(
    results_dir: str = "results",
    model_dir: str = "models",
    data_path: str = "immunogenicity_dataset.csv",
    binding_matrix_path: str = "models/peptide_binding_matrix.csv",
    model_path: str = "models/rf_30feature_integrated.joblib",
    dataset_mode: str = "modeB_updated",
    dataset_version: str = "IEDB-20260424-EBV_HPV16_UPDATED-v2",
) -> Tuple[str, str, str]:
    """Generate all final validation artifacts and return key output paths."""
    model_path = resolve_model_path(model_path)
    os.makedirs(results_dir, exist_ok=True)

    # 1) Gold-standard stage report
    gs_df = full_validation_report(results_dir)
    gs_path = os.path.join(results_dir, "gold_standard_validation.csv")
    gs_df.to_csv(gs_path, index=False)

    # 2) Baseline comparison
    baseline_df = compare_methods(results_dir, model_dir)
    baseline_path = os.path.join(results_dir, "baseline_comparison.csv")
    baseline_df.to_csv(baseline_path, index=False)

    # 3) H2 Tier A (labeled benchmark)
    _, h2_summary_df = run_h2_tier_a(
        data_path=data_path,
        model_path=model_path,
        binding_matrix_path=binding_matrix_path,
        output_dir=results_dir,
    )
    h2_summary_csv = os.path.join(results_dir, "h2_tier_a_summary.csv")
    h2_summary_md = os.path.join(results_dir, "h2_tier_a_summary.md")

    decision_row = h2_summary_df[h2_summary_df["method"] == "h2_decision"].iloc[0]
    h2_supported = bool(decision_row["h2_supported_ratio_gte_2_and_stable"])
    r10 = float(decision_row["issr_10_ratio_integrated_over_binding"])
    r25 = float(decision_row["issr_25_ratio_integrated_over_binding"])

    final_md_path = os.path.join(results_dir, "final_validation_report.md")
    status = "SUPPORTED" if h2_supported else "NOT SUPPORTED"
    has_stage_outputs = bool(gs_df["stage1_found"].any()) or bool(gs_df.get("stage2_found", pd.Series(dtype=bool)).any())
    has_baseline_outputs = not baseline_df.empty

    with open(final_md_path, "w", encoding="utf-8") as f:
        f.write(
            f"# SESTRAV Final Validation Report\n\n"
            f"## Core Outputs\n"
            f"- Gold-standard stage validation: `{gs_path}`\n"
            f"- Baseline comparison: `{baseline_path}`\n"
            f"- H2 Tier A summary: `{h2_summary_csv}`\n"
            f"- H2 Tier A markdown: `{h2_summary_md}`\n\n"
            f"## H2 Tier A Headline\n"
            f"- R10 (ISSR@10 integrated / binding-only): `{r10:.4f}`\n"
            f"- R25 (ISSR@25 integrated / binding-only): `{r25:.4f}`\n"
            f"- Decision (R10 >= 2 and stable denominator): **{status}**\n"
        )
        if not has_stage_outputs or not has_baseline_outputs:
            f.write(
                "\n## Data Availability Note\n"
                "- Some stage output files were not found in `results/` during validation.\n"
                "- Gold-standard stage recovery and baseline comparison may be partial or empty.\n"
                "- Run the full pipeline first (`snakemake --snakefile pipeline.smk --cores 4`) and rerun this report.\n"
            )

    # Write mode/version-tagged aliases to reduce cross-run interpretation mix-ups.
    tagged = {
        "gold_standard_validation": gs_df,
        "baseline_comparison": baseline_df,
        "h2_tier_a_summary": h2_summary_df,
    }
    for base_name, df in tagged.items():
        tagged_name = canonical_output_filename(base_name, dataset_mode, dataset_version)
        df.to_csv(os.path.join(results_dir, tagged_name), index=False)

    return gs_path, baseline_path, final_md_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SESTRAV final validation bundle")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--data", default="immunogenicity_dataset.csv")
    parser.add_argument("--binding-matrix", default="models/peptide_binding_matrix.csv")
    parser.add_argument("--model-path", default="models/rf_30feature_integrated.joblib")
    parser.add_argument("--dataset-mode", default="modeB_updated")
    parser.add_argument("--dataset-version", default="IEDB-20260424-EBV_HPV16_UPDATED-v2")
    args = parser.parse_args()

    run_final_validation(
        results_dir=args.results_dir,
        model_dir=args.model_dir,
        data_path=args.data,
        binding_matrix_path=args.binding_matrix,
        model_path=args.model_path,
        dataset_mode=args.dataset_mode,
        dataset_version=args.dataset_version,
    )
