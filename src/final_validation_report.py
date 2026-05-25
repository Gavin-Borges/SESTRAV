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
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
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
    binding_matrix_path: str = "models/peptide_binding_matrix_v3.csv",
    model_path: str = "models/rf_30feature_integrated.joblib",
    dataset_mode: str = "expansion_alpha",
    dataset_version: str = "2.0.0-alpha",
    freeze_mode: bool = False,
) -> Tuple[str, str, str]:
    """Generate all final validation artifacts and return key output paths."""
    model_path = resolve_model_path(model_path)
    os.makedirs(results_dir, exist_ok=True)
    run_started_at = datetime.now(timezone.utc).isoformat()

    def _sha256_or_missing(path: str) -> str:
        if not os.path.isfile(path):
            return "missing"
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    if freeze_mode and not os.path.isfile(model_path):
        raise RuntimeError(
            f"Freeze mode requires a trained model at '{model_path}', but no file exists."
        )

    tmp_dir = tempfile.mkdtemp(prefix=".final_validation_tmp_", dir=results_dir)
    try:
        # 1) Gold-standard stage report
        gs_df = full_validation_report(
            results_dir,
            strict_stems=freeze_mode,
            require_stage4_score=freeze_mode,
        )
        gs_filename = "gold_standard_validation.csv"
        gs_tmp_path = os.path.join(tmp_dir, gs_filename)
        gs_df.to_csv(gs_tmp_path, index=False)

        # 2) Baseline comparison
        baseline_df = compare_methods(
            results_dir,
            model_dir,
            strict_ann_loading=freeze_mode,
            binding_matrix_path=binding_matrix_path,
        )
        baseline_filename = "baseline_comparison.csv"
        baseline_tmp_path = os.path.join(tmp_dir, baseline_filename)
        baseline_df.to_csv(baseline_tmp_path, index=False)

        # 3) H2 Tier A (labeled benchmark)
        _, h2_summary_df = run_h2_tier_a(
            data_path=data_path,
            model_path=model_path,
            binding_matrix_path=binding_matrix_path,
            output_dir=tmp_dir,
        )
        h2_summary_csv_name = "h2_tier_a_summary.csv"
        h2_summary_md_name = "h2_tier_a_summary.md"

        decision_mask = h2_summary_df["method"] == "h2_decision"
        if not decision_mask.any():
            raise RuntimeError("H2 summary is missing the 'h2_decision' row.")
        decision_row = h2_summary_df.loc[decision_mask].iloc[0]
        h2_supported = bool(decision_row["h2_supported_ratio_gte_2_and_stable"])
        r10 = float(decision_row["issr_10_ratio_integrated_over_binding"])
        r25 = float(decision_row["issr_25_ratio_integrated_over_binding"])

        has_stage_outputs = bool(gs_df["stage1_found"].any()) or bool(
            gs_df.get("stage2_found", pd.Series(dtype=bool)).any()
        )
        has_baseline_outputs = not baseline_df.empty
        if freeze_mode and (not has_stage_outputs or not has_baseline_outputs):
            raise RuntimeError(
                "Freeze mode requires non-empty stage outputs and baseline comparison."
            )

        # 4) Final Markdown report
        final_md_name = "final_validation_report.md"
        final_md_tmp_path = os.path.join(tmp_dir, final_md_name)
        status = "SUPPORTED" if h2_supported else "NOT SUPPORTED"
        input_hashes = {
            "data_path": {"path": data_path, "sha256": _sha256_or_missing(data_path)},
            "binding_matrix_path": {
                "path": binding_matrix_path,
                "sha256": _sha256_or_missing(binding_matrix_path),
            },
            "model_path": {"path": model_path, "sha256": _sha256_or_missing(model_path)},
        }

        gs_path = os.path.join(results_dir, gs_filename)
        baseline_path = os.path.join(results_dir, baseline_filename)
        h2_summary_csv = os.path.join(results_dir, h2_summary_csv_name)
        h2_summary_md = os.path.join(results_dir, h2_summary_md_name)
        final_md_path = os.path.join(results_dir, final_md_name)
        with open(final_md_tmp_path, "w", encoding="utf-8") as f:
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
                f"- Decision (R10 >= 2 and stable denominator): **{status}**\n\n"
                f"## Run Metadata\n"
                f"- Generated at (UTC): `{run_started_at}`\n"
                f"- Freeze mode: `{freeze_mode}`\n"
                f"- Input hashes:\n"
                f"  - Data: `{input_hashes['data_path']['sha256']}`\n"
                f"  - Binding matrix: `{input_hashes['binding_matrix_path']['sha256']}`\n"
                f"  - Model: `{input_hashes['model_path']['sha256']}`\n"
            )
            if not has_stage_outputs or not has_baseline_outputs:
                f.write(
                    "\n## Data Availability Note\n"
                    "- Some stage output files were not found in `results/` during validation.\n"
                    "- Gold-standard stage recovery and baseline comparison may be partial or empty.\n"
                    "- Run the full pipeline first (`snakemake --snakefile pipeline.smk --cores 4`) and rerun this report.\n"
                )

        # 5) Write mode/version-tagged aliases in temp area.
        tagged = {
            "gold_standard_validation": gs_df,
            "baseline_comparison": baseline_df,
            "h2_tier_a_summary": h2_summary_df,
        }
        tagged_names = []
        for base_name, df in tagged.items():
            tagged_name = canonical_output_filename(base_name, dataset_mode, dataset_version)
            tagged_names.append(tagged_name)
            df.to_csv(os.path.join(tmp_dir, tagged_name), index=False)

        # 6) Freeze status artifact
        freeze_status_name = "freeze_status.json"
        with open(os.path.join(tmp_dir, freeze_status_name), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "valid": True,
                    "freeze_mode": freeze_mode,
                    "generated_at_utc": run_started_at,
                    "dataset_mode": dataset_mode,
                    "dataset_version": dataset_version,
                    "h2": {"r10": r10, "r25": r25, "status": status},
                    "inputs": input_hashes,
                },
                f,
                indent=2,
            )

        # 7) Publish atomically after full success.
        publish_names = [
            gs_filename,
            baseline_filename,
            "h2_tier_a_fold_metrics.csv",
            h2_summary_csv_name,
            h2_summary_md_name,
            final_md_name,
            freeze_status_name,
            *tagged_names,
        ]
        for name in publish_names:
            src = os.path.join(tmp_dir, name)
            dst = os.path.join(results_dir, name)
            if not os.path.isfile(src):
                raise RuntimeError(f"Expected validation artifact was not produced: {src}")
            os.replace(src, dst)
    except Exception as exc:
        with open(os.path.join(results_dir, "freeze_status.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "valid": False,
                    "freeze_mode": freeze_mode,
                    "generated_at_utc": run_started_at,
                    "error": str(exc),
                },
                f,
                indent=2,
            )
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return gs_path, baseline_path, final_md_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SESTRAV final validation bundle")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--data", default="immunogenicity_dataset.csv")
    parser.add_argument("--binding-matrix", default="models/peptide_binding_matrix_v3.csv")
    parser.add_argument("--model-path", default="models/rf_30feature_integrated.joblib")
    parser.add_argument("--dataset-mode", default="expansion_alpha")
    parser.add_argument("--dataset-version", default="2.0.0-alpha")
    parser.add_argument("--freeze-mode", action="store_true")
    args = parser.parse_args()

    run_final_validation(
        results_dir=args.results_dir,
        model_dir=args.model_dir,
        data_path=args.data,
        binding_matrix_path=args.binding_matrix,
        model_path=args.model_path,
        dataset_mode=args.dataset_mode,
        dataset_version=args.dataset_version,
        freeze_mode=args.freeze_mode,
    )
