"""
Regenerate all 11 PNGs for results/shareout_20260426/.

Requires: sestrav conda env (or equivalent with sklearn, shap,
xgboost, matplotlib, joblib).

Steps:
  1. Retrain RF+XGB (30-feature) to produce .joblib models (needed for SHAP)
  2. Generate stage-4 panel plots from existing ranked CSVs
  3. Generate CV training plots (ROC, PR, score distributions)
  4. Generate SHAP plots (beeswarm, bar, waterfall)
  5. Generate calibration reliability diagram
  6. Copy all PNGs into results/shareout_20260426/
"""

import os
import sys
import shutil

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

SHAREOUT_DIR = os.path.join("results", "shareout_20260426")
RESULTS_DIR = "results"
MODEL_DIR = "models"
DATA_PATH = "immunogenicity_dataset.csv"
BINDING_MATRIX = os.path.join(MODEL_DIR, "peptide_binding_matrix.csv")

os.makedirs(SHAREOUT_DIR, exist_ok=True)


def step1_retrain_models():
    """Retrain RF and XGB with 30-feature mode to get .joblib files."""
    from src.train_classifier import train_models

    rf_path = os.path.join(MODEL_DIR, "rf_30feature_integrated.joblib")
    xgb_path = os.path.join(MODEL_DIR, "xgb_30feature_integrated.joblib")
    if os.path.isfile(rf_path) and os.path.isfile(xgb_path):
        print("[Step 1] Models already exist, skipping retrain")
        return

    print("[Step 1] Retraining RF + XGB (30-feature integrated)...")
    train_models(
        DATA_PATH,
        MODEL_DIR,
        n_cv_folds=5,
        random_state=42,
        feature_mode=30,
        binding_matrix_path=BINDING_MATRIX,
    )
    print("[Step 1] Done\n")


def step2_stage4_plots():
    """Generate top-20 and score-distribution PNGs from existing ranked CSVs."""
    import pandas as pd
    from functions.stage4_immunogenicity_scoring import plot_immunogenicity_scores

    antigens = ["HPV16_18_panel8", "EBV_B95_8_panel8"]
    for ag in antigens:
        ranked_path = os.path.join(RESULTS_DIR, f"{ag}_ranked.csv")
        if not os.path.isfile(ranked_path):
            print(f"[Step 2] WARNING: {ranked_path} not found, skipping {ag}")
            continue
        ranked_df = pd.read_csv(ranked_path)
        print(f"[Step 2] Generating plots for {ag} ({len(ranked_df)} peptides)...")
        plot_immunogenicity_scores(ranked_df, ag)

    print("[Step 2] Done\n")


def step3_training_plots():
    """Generate CV ROC, PR, and score distribution PNGs."""
    from src.training_plots import generate_training_plots

    print("[Step 3] Generating CV training plots (30-feature)...")
    generate_training_plots(
        DATA_PATH,
        output_dir=RESULTS_DIR,
        random_state=42,
        feature_mode=30,
        binding_matrix_path=BINDING_MATRIX,
    )
    print("[Step 3] Done\n")


def step4_shap_plots():
    """Generate SHAP beeswarm, bar, and waterfall PNGs."""
    from src.shap_analysis import run_shap_analysis

    rf_path = os.path.join(MODEL_DIR, "rf_30feature_integrated.joblib")
    if not os.path.isfile(rf_path):
        print("[Step 4] WARNING: RF model not found, skipping SHAP")
        return

    print("[Step 4] Generating SHAP plots (30-feature)...")
    run_shap_analysis(
        results_dir=RESULTS_DIR,
        model_dir=MODEL_DIR,
        output_dir=RESULTS_DIR,
        feature_mode=30,
    )
    print("[Step 4] Done\n")


def step5_calibration_plot():
    """Generate calibration reliability diagram."""
    from src.calibration_analysis import run_calibration_analysis

    v2_oof = os.path.join(MODEL_DIR, "rf_oof_predictions.csv")
    v1_oof = os.path.join(MODEL_DIR, "v1_backup", "rf_oof_predictions.csv")

    if not os.path.isfile(v2_oof):
        print("[Step 5] WARNING: v2 OOF predictions not found, skipping calibration")
        return

    print("[Step 5] Generating calibration reliability diagram...")
    run_calibration_analysis(
        v2_oof_path=v2_oof,
        v1_oof_path=v1_oof if os.path.isfile(v1_oof) else None,
        output_dir=RESULTS_DIR,
    )
    print("[Step 5] Done\n")


def step6_copy_to_shareout():
    """Copy all expected PNGs from results/ into results/shareout_20260426/."""
    expected = [
        "HPV16_18_panel8_top20_immunogenicity.png",
        "EBV_B95_8_panel8_top20_immunogenicity.png",
        "HPV16_18_panel8_score_distribution.png",
        "EBV_B95_8_panel8_score_distribution.png",
        "cv_roc_curves.png",
        "cv_precision_recall_curves.png",
        "cv_score_distributions.png",
        "shap_summary_rf.png",
        "shap_bar_rf.png",
        "shap_waterfall_top_gs.png",
        "calibration_reliability_diagram.png",
    ]

    copied = 0
    missing = []
    for fname in expected:
        src = os.path.join(RESULTS_DIR, fname)
        dst = os.path.join(SHAREOUT_DIR, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing.append(fname)

    print(f"[Step 6] Copied {copied}/{len(expected)} PNGs to {SHAREOUT_DIR}")
    if missing:
        print(f"[Step 6] Missing: {missing}")
    print("[Step 6] Done\n")


if __name__ == "__main__":
    print("=" * 70)
    print("SESTRAV Shareout PNG Regeneration")
    print("=" * 70)
    step1_retrain_models()
    step2_stage4_plots()
    step3_training_plots()
    step4_shap_plots()
    step5_calibration_plot()
    step6_copy_to_shareout()

    pngs = [f for f in os.listdir(SHAREOUT_DIR) if f.endswith(".png")]
    print(f"\nFinal: {len(pngs)} PNGs in {SHAREOUT_DIR}/")
    for p in sorted(pngs):
        print(f"  {p}")
