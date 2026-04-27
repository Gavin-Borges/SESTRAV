"""
SESTRAV Training Visualization Module

Generates cross-validation evaluation plots for the training report:
  1. ROC curves (RF, XGB) with AUC annotation
  2. Precision-Recall curves with AP annotation
  3. Score distribution (positive vs negative classes)

Usage:
    python -m src.training_plots --data immunogenicity_dataset.csv
    python -m src.training_plots --data immunogenicity_dataset.csv \\
        --feature-mode 30 --binding-matrix models/peptide_binding_matrix.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve
from xgboost import XGBClassifier

from src.features import compute_features, TRAIN_FEATURE_COLUMNS, FEATURE_COLUMNS_30
from src.train_classifier import prepare_features, prepare_features_30
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES


def generate_training_plots(data_path, output_dir='results', random_state=42,
                            feature_mode=21, binding_matrix_path=None):
    """Generate all training evaluation plots using 5-fold CV.

    Args:
        data_path: path to immunogenicity_dataset.csv
        output_dir: directory for output PNGs
        random_state: CV random seed
        feature_mode: 21 (legacy sequence-only) or 30 (canonical integrated)
        binding_matrix_path: required when feature_mode=30
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    gs_mask = df['peptide'].isin(GOLD_STANDARD_EPITOPES)
    train_pool = df[~gs_mask].copy()

    if feature_mode == 30:
        if binding_matrix_path is None:
            raise ValueError("--binding-matrix is required for --feature-mode 30")
        X = prepare_features_30(train_pool, binding_matrix_path)
        mode_label = "30-feature integrated"
    else:
        X = prepare_features(train_pool, include_binding=False)
        mode_label = "21-feature sequence-only"

    y = train_pool['label'].values
    print(f"[Plots] Feature mode: {mode_label} ({X.shape[1]} columns)")

    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    spw = n_neg / max(n_pos, 1)

    models = {
        'RF': (RandomForestClassifier, dict(
            n_estimators=200, class_weight='balanced',
            random_state=random_state, n_jobs=1)),
        'XGB': (XGBClassifier, dict(
            n_estimators=200, scale_pos_weight=spw,
            random_state=random_state, eval_metric='logloss', nthread=1)),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    colors = {'RF': '#4C72B0', 'XGB': '#DD8452'}

    # Collect all fold predictions for aggregate curves
    all_preds = {name: {'y_true': [], 'y_score': []} for name in models}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        for name, (cls, kwargs) in models.items():
            model = cls(**kwargs)
            model.fit(X_tr, y_tr)
            scores = model.predict_proba(X_val)[:, 1]
            all_preds[name]['y_true'].extend(y_val)
            all_preds[name]['y_score'].extend(scores)

    # --- Plot 1: ROC Curves ---
    fig, ax = plt.subplots(figsize=(8, 7))
    for name in models:
        yt = np.array(all_preds[name]['y_true'])
        ys = np.array(all_preds[name]['y_score'])
        fpr, tpr, _ = roc_curve(yt, ys)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(yt, ys)
        ax.plot(fpr, tpr, color=colors[name], lw=2,
                label=f'{name} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title(f'ROC Curves — 5-Fold CV ({mode_label})', fontsize=14)
    ax.legend(loc='lower right', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_roc_curves.png'), dpi=150)
    plt.close()
    print("[Plots] ROC curves saved")

    # --- Plot 2: Precision-Recall Curves ---
    fig, ax = plt.subplots(figsize=(8, 7))
    prevalence = np.mean(y)
    for name in models:
        yt = np.array(all_preds[name]['y_true'])
        ys = np.array(all_preds[name]['y_score'])
        precision, recall, _ = precision_recall_curve(yt, ys)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(yt, ys)
        ax.plot(recall, precision, color=colors[name], lw=2,
                label=f'{name} (AP = {ap:.3f})')

    ax.axhline(y=prevalence, color='gray', linestyle='--', lw=1, alpha=0.7,
               label=f'Baseline prevalence ({prevalence:.2f})')
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title(f'Precision-Recall Curves — 5-Fold CV ({mode_label})', fontsize=14)
    ax.legend(loc='lower left', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_precision_recall_curves.png'), dpi=150)
    plt.close()
    print("[Plots] Precision-Recall curves saved")

    # --- Plot 3: Score Distribution (positive vs negative) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, name in enumerate(models):
        yt = np.array(all_preds[name]['y_true'])
        ys = np.array(all_preds[name]['y_score'])

        ax = axes[idx]
        ax.hist(ys[yt == 1], bins=40, alpha=0.7, color='#55A868',
                label=f'Positive (n={np.sum(yt==1)})', density=True)
        ax.hist(ys[yt == 0], bins=40, alpha=0.7, color='#C44E52',
                label=f'Negative (n={np.sum(yt==0)})', density=True)
        ax.set_xlabel('Predicted Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{name} Score Distribution', fontsize=13)
        ax.legend(fontsize=11)

    plt.suptitle(f'Predicted Score by True Class — 5-Fold CV ({mode_label})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_score_distributions.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("[Plots] Score distributions saved")

    print(f"\n[Plots] All training plots saved to {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SESTRAV training evaluation plots')
    parser.add_argument('--data', required=True, help='Path to immunogenicity_dataset.csv')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--feature-mode', type=int, default=21, choices=[21, 30],
                        help='Feature mode: 21 (sequence-only) or 30 (integrated)')
    parser.add_argument('--binding-matrix', default=None,
                        help='Path to peptide_binding_matrix.csv (required for --feature-mode 30)')
    args = parser.parse_args()
    generate_training_plots(args.data, args.output_dir,
                            feature_mode=args.feature_mode,
                            binding_matrix_path=args.binding_matrix)
