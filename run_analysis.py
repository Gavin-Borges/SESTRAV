"""
SESTRAV Post-Pipeline Analysis Runner

Runs all analysis tasks on existing pipeline output (no MHCflurry needed):
  1. Gold-standard validation (both viruses)
  2. Baseline comparison (RF vs XGB vs binding-only)
  3. SHAP explainability (summary, bar, waterfall plots)

Usage:
    python run_analysis.py [--results-dir results] [--model-dir models]
"""

import os
import argparse

import yaml

from src.gold_standard import full_validation_report
from src.baseline_comparison import compare_methods, print_comparison
from src.shap_analysis import run_shap_analysis


def _load_config(config_path="config.yaml"):
    """Load config.yaml for feature mode and binding matrix defaults."""
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


def main(results_dir='results', model_dir='models'):
    os.makedirs(results_dir, exist_ok=True)
    cfg = _load_config()

    print("\n" + "#" * 70)
    print("# SESTRAV POST-PIPELINE ANALYSIS")
    print("#" * 70)

    # --- 1. Gold-standard validation ---
    print("\n" + "=" * 70)
    print("STEP 1: Gold-Standard Validation (15 epitopes, both viruses)")
    print("=" * 70)

    report = full_validation_report(results_dir)
    gs_path = os.path.join(results_dir, 'gold_standard_validation.csv')
    report.to_csv(gs_path, index=False)
    print(f"\nSaved to {gs_path}")

    # --- 2. Baseline comparison ---
    print("\n" + "=" * 70)
    print("STEP 2: Baseline Comparison (RF vs XGB vs Binding-only)")
    print("=" * 70)

    comparison = compare_methods(results_dir, model_dir)
    comp_path = os.path.join(results_dir, 'baseline_comparison.csv')
    comparison.to_csv(comp_path, index=False)
    print_comparison(comparison)
    print(f"\nSaved to {comp_path}")

    # --- 3. SHAP analysis ---
    print("\n" + "=" * 70)
    print("STEP 3: SHAP Explainability Analysis")
    print("=" * 70)

    feature_mode = cfg.get('feature_mode', 30)
    run_shap_analysis(results_dir, model_dir, results_dir,
                      feature_mode=feature_mode)

    # --- Summary ---
    print("\n" + "#" * 70)
    print("# ANALYSIS COMPLETE")
    print("#" * 70)
    print(f"\nAll outputs in {results_dir}/:")
    for f in sorted(os.listdir(results_dir)):
        if not os.path.isdir(os.path.join(results_dir, f)):
            size = os.path.getsize(os.path.join(results_dir, f))
            print(f"  {f:<50s} {size:>10,d} bytes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SESTRAV post-pipeline analysis')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--model-dir', default='models')
    args = parser.parse_args()
    main(args.results_dir, args.model_dir)
