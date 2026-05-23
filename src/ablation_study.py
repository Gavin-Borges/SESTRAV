"""
SESTRAV Ablation Study — Feature Group Contribution Analysis

Evaluates the contribution of different feature groups to immunogenicity
prediction by training the best ANN architecture on each subset.

Feature group definitions (from CMB 523 Project 2):
  physico_20  : 20 TCR-contact physicochemical features only
  binding_10  : 10 per-allele MHC binding features only
  sestrav_21  : physico + peptide_length (legacy training track)
  combined_30 : physico + binding (canonical 30-feature track)
  full_31     : physico + binding + peptide_length

Project 2 results (5-fold CV, best ANN: 256-128-64 ReLU d0.2):
  physico_20:  AUC-ROC=0.5766, AUC-PR=0.7719
  binding_10:  AUC-ROC=0.7273, AUC-PR=0.8509
  sestrav_21:  AUC-ROC=0.6215, AUC-PR=0.7844
  combined_30: AUC-ROC=0.6699, AUC-PR=0.8252
  full_31:     AUC-ROC=0.7433, AUC-PR=0.8639  (best overall)

Usage:
    python -m src.ablation_study --data immunogenicity_dataset.csv \\
        --binding-matrix models/peptide_binding_matrix.csv
"""

import argparse
import os
import numpy as np
import pandas as pd

from src.features import PHYSICO_COLUMNS, BINDING_ALLELE_COLUMNS
from src.evaluate_metrics import summarize_fold_metrics
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES
from src.model import (
    set_seeds, get_device, compute_pos_weight, run_cv,
    SEED, N_FOLDS,
)
from src.train_classifier import prepare_features_30


# Feature group definitions
FEATURE_GROUPS = {
    "physico_20": PHYSICO_COLUMNS,
    "binding_10": BINDING_ALLELE_COLUMNS,
    "sestrav_21": PHYSICO_COLUMNS + ["peptide_length"],
    "combined_30": PHYSICO_COLUMNS + BINDING_ALLELE_COLUMNS,
    "full_31": PHYSICO_COLUMNS + BINDING_ALLELE_COLUMNS + ["peptide_length"],
}


def run_ablation(data_path, binding_matrix_path, config=None,
                 n_folds=N_FOLDS, output_dir='models'):
    """Run ablation study across all feature groups.

    Args:
        data_path:           Path to immunogenicity_dataset.csv.
        binding_matrix_path: Path to peptide_binding_matrix.csv.
        config:              Architecture config dict.
        n_folds:             CV folds.
        output_dir:          Directory for result CSVs.

    Returns:
        pd.DataFrame of ablation results sorted by AUC-PR.
    """
    if config is None:
        config = {"hidden": [256, 128, 64], "dropout": 0.2, "activation": "relu"}

    set_seeds(SEED)
    device = get_device()
    print(f"Device: {device}")

    # Load data
    df = pd.read_csv(data_path)
    gs_mask = df['peptide'].isin(GOLD_STANDARD_EPITOPES)
    pool = df[~gs_mask].copy()
    y = pool['label'].values
    virus = pool['virus'].values if 'virus' in pool.columns else np.zeros(len(pool))
    strat_key = np.array([f"{l}_{v}" for l, v in zip(y, virus)])
    pos_weight = compute_pos_weight(y)

    print(f"Training pool: {len(pool)} records")
    print(f"Class balance: {np.mean(y):.2%} positive")

    X_30 = prepare_features_30(pool, binding_matrix_path)

    # Add peptide_length column
    lengths = pool['peptide'].str.len().values
    X_full = X_30.copy()
    X_full['peptide_length'] = lengths

    # Run ablation
    config_name = "-".join(str(h) for h in config["hidden"])
    print(f"\nArchitecture: {config_name} {config['activation']} d{config['dropout']}")

    results = []
    for group_name, group_cols in FEATURE_GROUPS.items():
        available = [c for c in group_cols if c in X_full.columns]
        if len(available) < len(group_cols):
            missing = set(group_cols) - set(X_full.columns)
            print(f"\n  WARNING: {group_name} missing {len(missing)} columns: {missing}")
            continue

        X_group = X_full[available].values
        n_features = X_group.shape[1]
        print(f"\nAblation: {group_name} ({n_features} features)...")

        # Adjust architecture input dim
        group_config = config.copy()

        fold_metrics = run_cv(X_group, y, strat_key, group_config, pos_weight,
                              n_folds=n_folds, device=device)
        avg, std = summarize_fold_metrics(fold_metrics)

        print(f"  AUC-ROC={avg['auc_roc']:.4f} +/- {std['auc_roc']:.4f}  "
              f"AUC-PR={avg['auc_pr']:.4f} +/- {std['auc_pr']:.4f}")

        results.append({
            "feature_set": group_name,
            "n_features": n_features,
            "auc_roc_mean": avg["auc_roc"],
            "auc_roc_std": std["auc_roc"],
            "auc_pr_mean": avg["auc_pr"],
            "auc_pr_std": std["auc_pr"],
            "issr_10_mean": avg["issr_10"],
            "issr_25_mean": avg["issr_25"],
        })

    df_results = pd.DataFrame(results)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'ablation_study_results.csv')
    df_results.to_csv(out_path, index=False)
    print(f"\n{'=' * 60}")
    print("Ablation Study Results:")
    print(f"{'=' * 60}")
    print(df_results.to_string(index=False))
    print(f"\nResults saved to {out_path}")

    return df_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SESTRAV Ablation Study — Feature group contribution analysis'
    )
    parser.add_argument('--data', required=True,
                        help='Path to immunogenicity_dataset.csv')
    parser.add_argument('--binding-matrix', required=True,
                        help='Path to peptide_binding_matrix.csv')
    parser.add_argument('--architecture', default='256-128-64',
                        help='Hidden layer sizes (default: 256-128-64)')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--output-dir', default='models')
    args = parser.parse_args()

    hidden = [int(x) for x in args.architecture.split('-')]
    config = {"hidden": hidden, "dropout": args.dropout, "activation": args.activation}

    run_ablation(
        args.data,
        args.binding_matrix,
        config=config,
        n_folds=args.cv_folds,
        output_dir=args.output_dir,
    )
