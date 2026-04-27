"""
SESTRAV Virus-Specific Ablation Study

Trains separate EBV-only and HPV16-only RandomForest classifiers and compares
their within-virus performance against the pooled (virus-agnostic) model.

This addresses the biological question: does pooling immunogenicity data
across viruses with different TCR-contact patterns help or hurt prediction?

Usage:
    python -m src.virus_specific_ablation --data immunogenicity_dataset.csv
    python -m src.virus_specific_ablation --data immunogenicity_dataset.csv \
        --feature-mode 30 --binding-matrix models/peptide_binding_matrix.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from src.evaluate_metrics import evaluate
from src.features import FEATURE_COLUMNS_30, PHYSICO_COLUMNS, BINDING_ALLELE_COLUMNS, TRAIN_FEATURE_COLUMNS
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES
from src.train_classifier import prepare_features, prepare_features_30


def _safe_evaluate(y_true, y_scores):
    """Evaluate with NaN fallback for degenerate slices."""
    if len(np.unique(y_true)) < 2:
        return {k: np.nan for k in ('auc_roc', 'auc_pr', 'issr_10', 'issr_25')}
    return evaluate(y_true, y_scores)


def run_ablation(data_path, output_dir='results', n_cv_folds=5, random_state=42,
                 feature_mode=21, binding_matrix_path=None):
    """Run virus-specific vs pooled ablation study."""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    df = df[~df['peptide'].isin(GOLD_STANDARD_EPITOPES)].copy()
    print(f"Training pool (gold-standard excluded): {len(df)} records")

    if 'virus' not in df.columns:
        raise ValueError("Dataset must have a 'virus' column for ablation study")

    viruses = sorted(df['virus'].dropna().unique())
    print(f"Viruses found: {viruses}")

    rf_kwargs = dict(
        n_estimators=200,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=1,
    )

    all_results = []

    for virus in viruses:
        virus_df = df[df['virus'] == virus].copy()
        n_pos = int((virus_df['label'] == 1).sum())
        n_neg = int((virus_df['label'] == 0).sum())
        print(f"\n{'=' * 60}")
        print(f"{virus}: {len(virus_df)} peptides ({n_pos} pos, {n_neg} neg)")
        print(f"{'=' * 60}")

        if n_neg < n_cv_folds or n_pos < n_cv_folds:
            print(f"  SKIP: too few samples for {n_cv_folds}-fold CV")
            continue

        if feature_mode == 30:
            X_virus = prepare_features_30(virus_df, binding_matrix_path)
        else:
            X_virus = prepare_features(virus_df, include_binding=False)
        y_virus = virus_df['label'].values

        # --- Virus-specific model ---
        skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
        virus_fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_virus, y_virus), 1):
            X_tr, X_val = X_virus.iloc[train_idx], X_virus.iloc[val_idx]
            y_tr, y_val = y_virus[train_idx], y_virus[val_idx]

            model = RandomForestClassifier(**rf_kwargs)
            model.fit(X_tr, y_tr)
            scores = model.predict_proba(X_val)[:, 1]
            m = _safe_evaluate(y_val, scores)
            virus_fold_metrics.append(m)
            print(f"  [{virus}-specific] Fold {fold_idx}: AUC-ROC={m['auc_roc']:.4f}  "
                  f"AUC-PR={m['auc_pr']:.4f}")

        virus_avg = {k: np.nanmean([fm[k] for fm in virus_fold_metrics])
                     for k in virus_fold_metrics[0]}
        virus_std = {k: np.nanstd([fm[k] for fm in virus_fold_metrics])
                     for k in virus_fold_metrics[0]}

        # --- Pooled model evaluated on this virus only ---
        if feature_mode == 30:
            X_all = prepare_features_30(df, binding_matrix_path)
        else:
            X_all = prepare_features(df, include_binding=False)
        y_all = df['label'].values
        virus_mask_all = (df['virus'] == virus).values

        pooled_fold_metrics = []
        skf_pooled = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
        for fold_idx, (train_idx, val_idx) in enumerate(skf_pooled.split(X_all, y_all), 1):
            X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
            y_tr, y_val = y_all[train_idx], y_all[val_idx]

            model = RandomForestClassifier(**rf_kwargs)
            model.fit(X_tr, y_tr)

            virus_val_mask = virus_mask_all[val_idx]
            if virus_val_mask.sum() == 0:
                continue
            scores_virus = model.predict_proba(X_val[virus_val_mask])[:, 1]
            y_virus_val = y_val[virus_val_mask]
            m = _safe_evaluate(y_virus_val, scores_virus)
            pooled_fold_metrics.append(m)
            print(f"  [Pooled->{virus}] Fold {fold_idx}: AUC-ROC={m['auc_roc']:.4f}  "
                  f"AUC-PR={m['auc_pr']:.4f}")

        pooled_avg = {k: np.nanmean([fm[k] for fm in pooled_fold_metrics])
                      for k in pooled_fold_metrics[0]}
        pooled_std = {k: np.nanstd([fm[k] for fm in pooled_fold_metrics])
                      for k in pooled_fold_metrics[0]}

        for metric in virus_avg:
            all_results.append({
                'virus': virus,
                'metric': metric,
                'virus_specific_mean': virus_avg[metric],
                'virus_specific_std': virus_std[metric],
                'pooled_mean': pooled_avg[metric],
                'pooled_std': pooled_std[metric],
                'delta': virus_avg[metric] - pooled_avg[metric],
                'n_virus_samples': len(virus_df),
                'n_pooled_samples': len(df),
            })

    results_df = pd.DataFrame(all_results)
    output_path = os.path.join(output_dir, 'virus_specific_ablation.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Ablation results saved to {output_path}")
    print(f"{'=' * 60}")

    md_path = os.path.join(output_dir, 'virus_specific_ablation.md')
    _write_ablation_report(results_df, md_path, feature_mode)
    print(f"Ablation report saved to {md_path}")

    return results_df


def _write_ablation_report(results_df, output_path, feature_mode):
    """Generate a markdown summary of the ablation study."""
    lines = [
        "# Virus-Specific vs Pooled Model Ablation Study",
        "",
        f"**Feature mode:** {feature_mode}",
        "",
        "Positive delta = virus-specific model outperforms pooled model on that virus.",
        "",
    ]

    for virus in sorted(results_df['virus'].unique()):
        vdf = results_df[results_df['virus'] == virus]
        n = int(vdf['n_virus_samples'].iloc[0])
        lines.append(f"## {virus} (n={n})")
        lines.append("")
        lines.append("| Metric | Virus-Specific | Pooled | Delta |")
        lines.append("|--------|---------------|--------|-------|")
        for _, row in vdf.iterrows():
            lines.append(
                f"| {row['metric']} "
                f"| {row['virus_specific_mean']:.4f} +/- {row['virus_specific_std']:.4f} "
                f"| {row['pooled_mean']:.4f} +/- {row['pooled_std']:.4f} "
                f"| {row['delta']:+.4f} |"
            )
        lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SESTRAV virus-specific ablation study')
    parser.add_argument('--data', required=True, help='Path to immunogenicity_dataset.csv')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--feature-mode', type=int, default=21, choices=[21, 30])
    parser.add_argument('--binding-matrix', default=None,
                        help='Path to peptide_binding_matrix.csv (required for --feature-mode 30)')
    args = parser.parse_args()
    run_ablation(args.data, args.output_dir, n_cv_folds=args.cv_folds,
                 feature_mode=args.feature_mode, binding_matrix_path=args.binding_matrix)
