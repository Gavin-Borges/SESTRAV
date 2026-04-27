"""
SESTRAV SHAP Explainability Analysis

Generates SHAP (SHapley Additive exPlanations) plots for the trained RF
and XGBoost immunogenicity classifiers, providing feature-level interpretability.

Outputs:
  1. SHAP summary beeswarm plot (global feature importance + direction)
  2. SHAP bar plot (mean |SHAP| per feature)
  3. SHAP values CSV for downstream analysis
  4. Gold-standard waterfall plot (local explanation for top-ranked GS epitope)

SHAP TreeExplainer is exact for tree models (RF, XGB) — no approximation.

Usage (after pipeline.py has run):
    python -m src.shap_analysis --results-dir results --model-dir models
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from joblib import load as joblib_load

from src.features import TRAIN_FEATURE_COLUMNS, FEATURE_COLUMNS_30
from src.gold_standard import GOLD_STANDARD, VIRUS_FILE_MAP
from src.naming import proteome_id_candidates, resolve_model_path


FEATURE_DISPLAY_NAMES = {
    'peptide_length': 'Peptide length',
    'p4_hydrophobicity': 'p4 hydrophobicity',
    'p5_hydrophobicity': 'p5 hydrophobicity',
    'p6_hydrophobicity': 'p6 hydrophobicity',
    'p7_hydrophobicity': 'p7 hydrophobicity',
    'p8_hydrophobicity': 'p8 hydrophobicity',
    'p4_aromaticity': 'p4 aromaticity',
    'p5_aromaticity': 'p5 aromaticity',
    'p6_aromaticity': 'p6 aromaticity',
    'p7_aromaticity': 'p7 aromaticity',
    'p8_aromaticity': 'p8 aromaticity',
    'p4_vdw_volume': 'p4 VdW volume',
    'p5_vdw_volume': 'p5 VdW volume',
    'p6_vdw_volume': 'p6 VdW volume',
    'p7_vdw_volume': 'p7 VdW volume',
    'p8_vdw_volume': 'p8 VdW volume',
    'p4_charge': 'p4 charge',
    'p5_charge': 'p5 charge',
    'p6_charge': 'p6 charge',
    'p7_charge': 'p7 charge',
    'p8_charge': 'p8 charge',
    'bind_A0101': 'Bind HLA-A*01:01',
    'bind_A0201': 'Bind HLA-A*02:01',
    'bind_A0301': 'Bind HLA-A*03:01',
    'bind_A1101': 'Bind HLA-A*11:01',
    'bind_A2402': 'Bind HLA-A*24:02',
    'bind_B0702': 'Bind HLA-B*07:02',
    'bind_B0801': 'Bind HLA-B*08:01',
    'bind_B2705': 'Bind HLA-B*27:05',
    'bind_B3501': 'Bind HLA-B*35:01',
    'bind_B4402': 'Bind HLA-B*44:02',
}


def _load_features(results_dir, max_samples=2000):
    """Load and optionally subsample feature data from pipeline output.

    Combines EBV and HPV features, subsamples if total exceeds max_samples
    (SHAP on 26k rows is slow; 2k is visually equivalent for summary plots).
    """
    frames = []
    for virus, prefix in VIRUS_FILE_MAP.items():
        path = None
        for cand in proteome_id_candidates(prefix):
            p = os.path.join(results_dir, f"{cand}_features.csv")
            if os.path.isfile(p):
                path = p
                break
        if path and os.path.isfile(path):
            df = pd.read_csv(path)
            df['virus'] = virus
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No feature CSVs found in {results_dir}")

    combined = pd.concat(frames, ignore_index=True)
    if len(combined) > max_samples:
        combined = combined.sample(n=max_samples, random_state=42)
    return combined


def run_shap_analysis(results_dir, model_dir='models', output_dir='results',
                      feature_mode=21):
    """Compute SHAP values and generate all plots.

    Args:
        feature_mode: 21 for sequence-only legacy models, 30 for integrated
                      models that include per-allele binding features.
    """
    os.makedirs(output_dir, exist_ok=True)

    if feature_mode == 30:
        rf_path = os.path.join(model_dir, 'rf_30feature_integrated.joblib')
        xgb_path = os.path.join(model_dir, 'xgb_30feature_integrated.joblib')
        feature_cols = FEATURE_COLUMNS_30
    else:
        rf_path = os.path.join(model_dir, 'rf_21feature_legacy.joblib')
        xgb_path = os.path.join(model_dir, 'xgb_21feature_legacy.joblib')
        feature_cols = TRAIN_FEATURE_COLUMNS

    rf_model = joblib_load(resolve_model_path(rf_path))
    xgb_model = joblib_load(resolve_model_path(xgb_path))

    combined = _load_features(results_dir)
    cols = [c for c in feature_cols if c in combined.columns]
    X = combined[cols].copy()

    display_names = [FEATURE_DISPLAY_NAMES.get(c, c) for c in cols]
    X_display = X.copy()
    X_display.columns = display_names

    for model, name, tag in [
        (rf_model, 'RandomForest', 'rf'),
        (xgb_model, 'XGBoost', 'xgb'),
    ]:
        print(f"\n[SHAP] Computing TreeExplainer for {name}...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        except (ValueError, TypeError) as exc:
            print(f"  WARNING: TreeExplainer failed for {name}: {exc}")
            print(f"  Skipping {name} SHAP (known shap/xgboost compatibility issue)")
            continue

        # shap_values can be: list of 2 arrays (old API), 3D ndarray (n, f, 2),
        # or 2D ndarray (n, f) for binary XGB.  We always want class-1 values.
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_vals = shap_values[:, :, 1]
        else:
            shap_vals = shap_values

        shap_df = pd.DataFrame(shap_vals, columns=cols)
        shap_df.to_csv(os.path.join(output_dir, f'shap_values_{tag}.csv'), index=False)
        print(f"  SHAP values saved ({shap_vals.shape[0]} samples x {shap_vals.shape[1]} features)")

        # Beeswarm summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_vals,
            X_display,
            show=False,
            plot_size=(10, 8),
        )
        plt.title(f'{name} — SHAP Feature Importance', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_summary_{tag}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Summary beeswarm plot saved")

        # Bar plot (mean |SHAP|)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_vals,
            X_display,
            plot_type='bar',
            show=False,
            plot_size=(10, 8),
        )
        plt.title(f'{name} — Mean |SHAP| Feature Importance', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_bar_{tag}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Mean |SHAP| bar plot saved")

    _shap_gold_standard_waterfall(rf_model, results_dir, output_dir,
                                   feature_cols=feature_cols)

    print(f"\n[SHAP] All outputs saved to {output_dir}/")


def _shap_gold_standard_waterfall(model, results_dir, output_dir,
                                   feature_cols=None):
    """Generate a waterfall plot for the highest-ranked gold-standard epitope."""
    if feature_cols is None:
        feature_cols = TRAIN_FEATURE_COLUMNS
    gs_peptides = {gs['peptide'] for gs in GOLD_STANDARD}

    best_peptide = None
    best_rank = float('inf')
    best_row = None

    for virus, prefix in VIRUS_FILE_MAP.items():
        ranked_path = None
        for cand in proteome_id_candidates(prefix):
            p = os.path.join(results_dir, f"{cand}_ranked.csv")
            if os.path.isfile(p):
                ranked_path = p
                break
        if ranked_path is None:
            continue
        df = pd.read_csv(ranked_path)
        gs_rows = df[df['peptide'].isin(gs_peptides)]
        if gs_rows.empty:
            continue
        top = gs_rows.loc[gs_rows['rank'].idxmin()]
        if top['rank'] < best_rank:
            best_rank = top['rank']
            best_peptide = top['peptide']
            best_row = top

    if best_peptide is None:
        print("[SHAP] No gold-standard epitopes found in ranked output; skipping waterfall")
        return

    cols = [c for c in feature_cols if c in best_row.index]
    x_single = pd.DataFrame([best_row[cols].values], columns=cols)
    display_names = [FEATURE_DISPLAY_NAMES.get(c, c) for c in cols]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(x_single)

    vals = shap_values.values
    base = shap_values.base_values
    data = shap_values.data

    # Extract class-1 slice from RF (3D) or use directly for XGB (2D)
    if vals.ndim == 3:
        vals = vals[0, :, 1]
    else:
        vals = vals[0]

    if isinstance(base, np.ndarray) and base.ndim == 2:
        base = base[0, 1]
    elif isinstance(base, np.ndarray) and base.ndim == 1:
        base = base[0]
    else:
        base = float(base)

    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = data[0]

    sv = shap.Explanation(
        values=vals,
        base_values=base,
        data=data,
        feature_names=display_names,
    )

    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(sv, show=False)
    plt.title(f'SHAP Waterfall — {best_peptide} (rank {int(best_rank)})', fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_waterfall_top_gs.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Waterfall plot saved for {best_peptide} (rank {int(best_rank)})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SESTRAV SHAP explainability analysis')
    parser.add_argument('--results-dir', default='results',
                        help='Directory containing pipeline feature CSVs')
    parser.add_argument('--model-dir', default='models',
                        help='Directory containing .joblib model files')
    parser.add_argument('--output-dir', default='results',
                        help='Directory for SHAP output files')
    parser.add_argument('--feature-mode', type=int, default=21, choices=[21, 30],
                        help='Feature mode: 21 (sequence-only) or 30 (integrated)')
    args = parser.parse_args()

    run_shap_analysis(args.results_dir, args.model_dir, args.output_dir,
                      feature_mode=args.feature_mode)
