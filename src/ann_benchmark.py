"""
SESTRAV ANN Benchmark — CMB 523 Neural Network Comparison

Trains feedforward neural networks (MLPs) on the SESTRAV immunogenicity dataset,
enabling direct comparison with RF and XGBoost classifiers.

Supports two modes:
  1. Single architecture (default): trains the specified architecture with k-fold CV
  2. Architecture search (--search): evaluates all 14 Project 2 configurations

Feature modes:
  --feature-mode 21  Legacy: 21 sequence-only features (binding_score excluded)
  --feature-mode 30  Canonical: 20 physico + 10 multi-allele binding features

Project 2 best architecture (30-feature, 5-fold CV):
  256-128-64 ReLU dropout 0.2 → AUC-PR=0.8252 +/- 0.0248

Usage:
    # Single architecture (default: 256-128-64)
    python -m src.ann_benchmark --data immunogenicity_dataset.csv --feature-mode 30

    # Architecture search
    python -m src.ann_benchmark --data immunogenicity_dataset.csv --feature-mode 30 --search

    # Legacy 21-feature mode (backward compatible)
    python -m src.ann_benchmark --data immunogenicity_dataset.csv --feature-mode 21
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch

from src.artifact_integrity import MODEL_CHECKSUM_MANIFEST, update_checksum_manifest
from src.features import (
    compute_features, TRAIN_FEATURE_COLUMNS, FEATURE_COLUMNS_30,
)
from src.evaluate_metrics import evaluate, summarize_fold_metrics
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES
from src.model import (
    FlexibleMLP, set_seeds, get_device, compute_pos_weight,
    train_one_fold, run_cv, train_final_model,
    ARCH_CONFIGS, SEED, N_FOLDS,
)


def _prepare_features_21(df):
    """Compute 21-feature matrix from peptide sequences (legacy mode)."""
    records = []
    for _, row in df.iterrows():
        feats = compute_features(row['peptide'], binding_score=0.0)
        records.append(feats)
    return pd.DataFrame(records)[TRAIN_FEATURE_COLUMNS]


def _prepare_features_30(df, binding_matrix_path):
    """Compute 30-feature matrix (20 physico + 10 binding)."""
    from src.train_classifier import prepare_features_30
    return prepare_features_30(df, binding_matrix_path)


def _parse_architecture(arch_string):
    """Parse architecture string like '256-128-64' into a list of ints."""
    return [int(x) for x in arch_string.split('-')]


def run_architecture_search(X, y, strat_key, pos_weight, configs=None,
                            n_folds=N_FOLDS, device=None):
    """Evaluate multiple architectures via k-fold CV.

    Args:
        X:          np.ndarray feature matrix.
        y:          np.ndarray binary labels.
        strat_key:  stratification keys.
        pos_weight: class imbalance weight.
        configs:    list of architecture config dicts (default: ARCH_CONFIGS).
        n_folds:    CV folds.
        device:     torch.device.

    Returns:
        pd.DataFrame of results sorted by AUC-PR, and the best config dict.
    """
    if configs is None:
        configs = ARCH_CONFIGS
    if device is None:
        device = get_device()

    results = []
    for config in configs:
        name = config.get("name", str(config["hidden"]))
        print(f"\n  {name}...", end="", flush=True)

        fold_metrics = run_cv(X, y, strat_key, config, pos_weight,
                              n_folds=n_folds, device=device)
        avg, std = summarize_fold_metrics(fold_metrics)

        print(f" AUC-PR={avg['auc_pr']:.4f} +/- {std['auc_pr']:.4f}")

        results.append({
            "config": name,
            "auc_roc_mean": avg["auc_roc"],
            "auc_roc_std": std["auc_roc"],
            "auc_pr_mean": avg["auc_pr"],
            "auc_pr_std": std["auc_pr"],
            "issr_10_mean": avg["issr_10"],
            "issr_25_mean": avg["issr_25"],
            **config,  # Include hidden, dropout, activation for reference
        })

    df = pd.DataFrame(results).sort_values("auc_pr_mean", ascending=False)
    best_row = df.iloc[0]
    best_config = {
        "hidden": best_row["hidden"],
        "dropout": best_row["dropout"],
        "activation": best_row["activation"],
    }
    print(f"\n  Best: {best_row['config']} (AUC-PR={best_row['auc_pr_mean']:.4f})")

    return df, best_config


def train_ann(data_path, model_dir='models', n_cv_folds=5, random_state=42,
              feature_mode=21, binding_matrix_path=None,
              architecture=None, search=False):
    """Full ANN training pipeline with CV and final model serialization.

    Args:
        data_path:           Path to immunogenicity_dataset.csv.
        model_dir:           Output directory for model artifacts.
        n_cv_folds:          Number of CV folds.
        random_state:        Random seed.
        feature_mode:        21 (legacy) or 30 (canonical).
        binding_matrix_path: Required for feature_mode=30.
        architecture:        Hidden layer sizes (list of ints or dash-separated string).
        search:              If True, run architecture search.
    """
    set_seeds(random_state)
    os.makedirs(model_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # --- Load data ---
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {data_path}")

    gs_mask = df['peptide'].isin(GOLD_STANDARD_EPITOPES)
    train_pool = df[~gs_mask].copy()
    print(f"Held out {gs_mask.sum()} gold-standard records")
    print(f"Training pool: {len(train_pool)} records")

    # --- Prepare features ---
    if feature_mode == 30:
        if binding_matrix_path is None:
            raise ValueError("--binding-matrix is required for feature-mode 30")
        X_df = _prepare_features_30(train_pool, binding_matrix_path)
        feature_cols = FEATURE_COLUMNS_30
        mode_label = "30-feature multi-allele"
    else:
        X_df = _prepare_features_21(train_pool)
        feature_cols = TRAIN_FEATURE_COLUMNS
        mode_label = "21-feature sequence-only"

    X = X_df.values
    y = train_pool['label'].values

    print(f"Features: {X.shape[1]} ({mode_label})")
    print(f"Class balance: {np.mean(y):.2%} positive, {1 - np.mean(y):.2%} negative")

    pos_weight = compute_pos_weight(y)
    virus = train_pool['virus'].values if 'virus' in train_pool.columns else np.zeros(len(train_pool))
    strat_key = np.array([f"{l}_{v}" for l, v in zip(y, virus)])

    # --- Architecture search or single-architecture CV ---
    if search:
        print(f"\n{'=' * 60}")
        print(f"Architecture Search: {len(ARCH_CONFIGS)} configs x {n_cv_folds}-fold CV")
        print(f"{'=' * 60}")

        search_df, best_config = run_architecture_search(
            X, y, strat_key, pos_weight, n_folds=n_cv_folds, device=device,
        )

        search_path = os.path.join(model_dir, 'ann_architecture_search.csv')
        search_df.to_csv(search_path, index=False)
        print(f"  Architecture search results saved to {search_path}")

        config = best_config
    else:
        # Use specified architecture or default
        if architecture is None:
            if feature_mode == 30:
                # Project 2 best: 256-128-64
                hidden = [256, 128, 64]
            else:
                # Legacy default: 64-32
                hidden = [64, 32]
        elif isinstance(architecture, str):
            hidden = _parse_architecture(architecture)
        else:
            hidden = architecture

        config = {"hidden": hidden, "dropout": 0.2, "activation": "relu"}

    config_name = "-".join(str(h) for h in config["hidden"])
    print(f"\n{'=' * 60}")
    print(f"ANN ({config_name} {config['activation']} d{config['dropout']}) "
          f"{n_cv_folds}-fold cross-validation:")
    print(f"{'=' * 60}")

    fold_metrics = run_cv(X, y, strat_key, config, pos_weight,
                          n_folds=n_cv_folds, device=device)

    # Print fold results
    for i, m in enumerate(fold_metrics, 1):
        print(f"    Fold {i}: AUC-ROC={m['auc_roc']:.4f}  "
              f"AUC-PR={m['auc_pr']:.4f}  "
              f"ISSR@10={m['issr_10']:.4f}  "
              f"ISSR@25={m['issr_25']:.4f}")

    avg, std = summarize_fold_metrics(fold_metrics)
    print(f"  Mean:  " + "  ".join(f"{k}={v:.4f}" for k, v in avg.items()
                                    if k in ('auc_roc', 'auc_pr', 'issr_10', 'issr_25')))
    print(f"  Stdev: " + "  ".join(f"{k}={v:.4f}" for k, v in std.items()
                                    if k in ('auc_roc', 'auc_pr', 'issr_10', 'issr_25')))

    # --- Final model training ---
    print(f"\n{'=' * 60}")
    print("Retraining final ANN on full training pool...")
    print(f"{'=' * 60}")

    half = len(X) // 5
    final_model, final_scaler = train_final_model(
        X, y, X[:half], y[:half], config, pos_weight, device=device,
    )

    # --- Save model ---
    if feature_mode == 30:
        model_stem = 'ann_30feature_integrated'
    else:
        model_stem = 'ann_21feature_legacy'

    model_path = os.path.join(model_dir, f'{model_stem}.pt')
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'scaler_mean': torch.tensor(final_scaler.mean_),
        'scaler_scale': torch.tensor(final_scaler.scale_),
        'n_features': X.shape[1],
        'architecture': {
            'hidden': config['hidden'],
            'activation': config['activation'],
            'dropout': config['dropout'],
        },
    }, model_path)
    print(f"  ANN saved to {model_path}")

    # --- Save CV results ---
    results_path = os.path.join(model_dir, 'training_results.csv')
    if os.path.isfile(results_path):
        results_df = pd.read_csv(results_path)
        results_df['ann_cv_mean'] = [avg.get(m, None) for m in results_df['metric']]
        results_df['ann_cv_std'] = [std.get(m, None) for m in results_df['metric']]
    else:
        rows = []
        for key in avg:
            rows.append({
                'metric': key,
                'ann_cv_mean': avg[key],
                'ann_cv_std': std[key],
            })
        results_df = pd.DataFrame(rows)

    results_df.to_csv(results_path, index=False)
    print(f"  CV results appended to {results_path}")

    checksum_manifest = os.path.join(model_dir, MODEL_CHECKSUM_MANIFEST)
    update_checksum_manifest(
        checksum_manifest,
        [
            model_path,
            results_path,
            *( [search_path] if search else [] ),
        ],
    )
    print(f"  Artifact checksums updated in {checksum_manifest}")

    return final_model, final_scaler, avg, std


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SESTRAV ANN benchmark')
    parser.add_argument('--data', required=True,
                        help='Path to immunogenicity_dataset.csv')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--feature-mode', type=int, default=30, choices=[21, 30],
                        help='Feature mode: 21 (legacy) or 30 (canonical, default)')
    parser.add_argument('--binding-matrix', default='models/peptide_binding_matrix.csv',
                        help='Path to peptide_binding_matrix.csv (required for mode 30)')
    parser.add_argument('--architecture', default=None,
                        help='Hidden layer sizes, e.g., "256-128-64" (default: auto)')
    parser.add_argument('--search', action='store_true',
                        help='Run architecture search over 14 configurations')
    args = parser.parse_args()

    train_ann(
        args.data,
        model_dir=args.model_dir,
        n_cv_folds=args.cv_folds,
        feature_mode=args.feature_mode,
        binding_matrix_path=args.binding_matrix,
        architecture=args.architecture,
        search=args.search,
    )
