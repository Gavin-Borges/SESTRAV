"""
SESTRAV Offline Model Training Script

Trains RandomForest and XGBoost classifiers on cleaned IEDB immunogenicity data.
Produces serialized .joblib models for use in the production scoring pipeline.

Feature modes:
  --feature-mode 21  (default, legacy)
      21 sequence-only features (p4-p8 physicochemical + peptide_length).
      binding_score excluded because IEDB exports lack allele info.

  --feature-mode 30
      20 physicochemical features + 10 per-allele MHCflurry binding scores.
      Requires --binding-matrix pointing to peptide_binding_matrix.csv
      (produced by CMB 523 Project 2).

Evaluation uses stratified 5-fold cross-validation on the full training pool
for reliable metric estimates, then retrains on the full pool for the final
serialized model.

Usage:
    python -m src.train_classifier --data immunogenicity_dataset.csv
    python -m src.train_classifier --data immunogenicity_dataset.csv \\
        --feature-mode 30 --binding-matrix models/peptide_binding_matrix.csv
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from joblib import dump

from src.features import (
    compute_features, FEATURE_COLUMNS, TRAIN_FEATURE_COLUMNS,
    FEATURE_COLUMNS_30, BINDING_ALLELE_COLUMNS, PHYSICO_COLUMNS,
)
from src.evaluate_metrics import evaluate
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES
from src.subgroup_eval import evaluate_subgroups, pick_operating_threshold


def prepare_features(df, include_binding=False, binding_col='binding_score'):
    """Compute feature matrix for all peptides in the DataFrame.

    When include_binding is False (training mode), returns 21 features
    (excluding binding_score).  When True (inference mode), returns all 22.
    """
    feature_records = []
    for _, row in df.iterrows():
        peptide = row['peptide']
        binding = 0.0
        if include_binding:
            binding = row.get(binding_col, 0.0)
            if pd.isna(binding):
                binding = 0.0
        feats = compute_features(peptide, binding_score=binding)
        feature_records.append(feats)
    cols = FEATURE_COLUMNS if include_binding else TRAIN_FEATURE_COLUMNS
    return pd.DataFrame(feature_records)[cols]


def prepare_features_30(df, binding_matrix_path):
    """Build the 30-feature matrix by joining physico features with per-allele binding.

    Returns a DataFrame with columns matching FEATURE_COLUMNS_30.
    """
    binding_df = pd.read_csv(binding_matrix_path)
    bind_cols_present = [c for c in BINDING_ALLELE_COLUMNS if c in binding_df.columns]
    if len(bind_cols_present) < 10:
        raise ValueError(
            f"Binding matrix has only {len(bind_cols_present)}/10 expected allele columns"
        )
    binding_lookup = binding_df.set_index('peptide')[bind_cols_present]

    physico_records = []
    for _, row in df.iterrows():
        feats = compute_features(row['peptide'], binding_score=0.0)
        physico_records.append(feats)
    physico_df = pd.DataFrame(physico_records)[PHYSICO_COLUMNS]

    peptides = df['peptide'].values
    bind_rows = []
    for pep in peptides:
        if pep in binding_lookup.index:
            bind_rows.append(binding_lookup.loc[pep].values)
        else:
            bind_rows.append(np.zeros(10))
    bind_df = pd.DataFrame(bind_rows, columns=BINDING_ALLELE_COLUMNS)

    return pd.concat([physico_df.reset_index(drop=True),
                      bind_df.reset_index(drop=True)], axis=1)[FEATURE_COLUMNS_30]


def _cross_validate(
    X,
    y,
    metadata,
    model_cls,
    model_kwargs,
    n_splits=5,
    random_state=42,
    subgroup_columns=None,
    min_group_size=15,
):
    """Run stratified k-fold CV and return aggregate, subgroup rows, and OOF scores."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []
    subgroup_rows = []
    oof_rows = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = model_cls(**model_kwargs)
        model.fit(X_tr, y_tr)
        scores = model.predict_proba(X_val)[:, 1]
        m = evaluate(y_val, scores)
        fold_metrics.append(m)
        fold_meta = metadata.iloc[val_idx].copy().reset_index(drop=True)
        fold_meta["label"] = y_val
        fold_meta["score"] = scores
        fold_meta["fold"] = fold_idx
        oof_rows.append(fold_meta)

        for row in evaluate_subgroups(
            fold_meta,
            score_col="score",
            label_col="label",
            group_columns=subgroup_columns,
            min_group_size=min_group_size,
        ):
            subgroup_rows.append(
                {
                    "fold": fold_idx,
                    **row,
                }
            )
        print(f"    Fold {fold_idx}: AUC-ROC={m['auc_roc']:.4f}  "
              f"AUC-PR={m['auc_pr']:.4f}  "
              f"ISSR@10={m['issr_10']:.4f}  "
              f"ISSR@25={m['issr_25']:.4f}")

    avg = {}
    std = {}
    for key in fold_metrics[0]:
        vals = [fm[key] for fm in fold_metrics]
        avg[key] = float(np.mean(vals))
        std[key] = float(np.std(vals))
    subgroup_df = pd.DataFrame(subgroup_rows)
    oof_df = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
    return avg, std, subgroup_df, oof_df


def train_models(data_path, model_dir='models', n_cv_folds=5, random_state=42,
                  feature_mode=21, binding_matrix_path=None):
    """
    Full training pipeline:
    1. Load cleaned IEDB data
    2. Remove gold-standard epitopes from training
    3. Compute feature matrix (21 or 30 features depending on mode)
    4. Stratified 5-fold cross-validation for reliable metric estimates
    5. Retrain on full pool with class-imbalance handling
    6. Serialize final models
    """
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {data_path}")

    gs_mask = df['peptide'].isin(GOLD_STANDARD_EPITOPES)
    gold_standard_df = df[gs_mask].copy()
    train_pool = df[~gs_mask].copy()
    print(f"Held out {len(gold_standard_df)} gold-standard epitope records")
    print(f"Training pool: {len(train_pool)} records")

    if feature_mode == 30:
        if binding_matrix_path is None:
            raise ValueError("--binding-matrix is required for feature-mode 30")
        X = prepare_features_30(train_pool, binding_matrix_path)
        feature_cols_used = FEATURE_COLUMNS_30
        mode_label = "30-feature multi-allele"
    else:
        X = prepare_features(train_pool, include_binding=False)
        feature_cols_used = TRAIN_FEATURE_COLUMNS
        mode_label = "21-feature sequence-only"

    y = train_pool['label'].values
    metadata_cols = [c for c in ["peptide", "virus", "strain", "protein"] if c in train_pool.columns]
    metadata = train_pool[metadata_cols].copy().reset_index(drop=True)
    print(f"Features: {X.shape[1]} ({mode_label})")
    print(f"Class balance: {np.mean(y):.2%} positive, {1 - np.mean(y):.2%} negative")

    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    spw = n_neg / max(n_pos, 1)

    rf_kwargs = dict(
        n_estimators=200,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=1,
    )
    xgb_kwargs = dict(
        n_estimators=200,
        scale_pos_weight=spw,
        random_state=random_state,
        eval_metric='logloss',
        nthread=1,
    )

    print(f"\n{'=' * 60}")
    print(f"RandomForest {n_cv_folds}-fold cross-validation:")
    print(f"{'=' * 60}")
    subgroup_columns = [c for c in ["virus", "strain", "protein"] if c in train_pool.columns]
    rf_avg, rf_std, rf_subgroups, rf_oof = _cross_validate(
        X, y, metadata, RandomForestClassifier, rf_kwargs,
        n_splits=n_cv_folds, random_state=random_state,
        subgroup_columns=subgroup_columns,
    )
    print(f"  Mean:  " + "  ".join(f"{k}={v:.4f}" for k, v in rf_avg.items()))
    print(f"  Stdev: " + "  ".join(f"{k}={v:.4f}" for k, v in rf_std.items()))

    print(f"\n{'=' * 60}")
    print(f"XGBoost {n_cv_folds}-fold cross-validation:")
    print(f"{'=' * 60}")
    xgb_avg, xgb_std, xgb_subgroups, xgb_oof = _cross_validate(
        X, y, metadata, XGBClassifier, xgb_kwargs,
        n_splits=n_cv_folds, random_state=random_state,
        subgroup_columns=subgroup_columns,
    )
    print(f"  Mean:  " + "  ".join(f"{k}={v:.4f}" for k, v in xgb_avg.items()))
    print(f"  Stdev: " + "  ".join(f"{k}={v:.4f}" for k, v in xgb_std.items()))

    print(f"\n{'=' * 60}")
    print("Retraining final models on full training pool...")
    print(f"{'=' * 60}")

    if feature_mode == 21:
        rf_stem = 'rf_21feature_legacy'
        xgb_stem = 'xgb_21feature_legacy'
    else:
        rf_stem = f'rf_{feature_mode}feature_integrated'
        xgb_stem = f'xgb_{feature_mode}feature_integrated'

    rf_final = RandomForestClassifier(**rf_kwargs)
    rf_final.fit(X, y)
    rf_path = os.path.join(model_dir, f'{rf_stem}.joblib')
    dump(rf_final, rf_path)
    print(f"  RandomForest saved to {rf_path}")

    xgb_final = XGBClassifier(**xgb_kwargs)
    xgb_final.fit(X, y)
    xgb_path = os.path.join(model_dir, f'{xgb_stem}.joblib')
    dump(xgb_final, xgb_path)
    print(f"  XGBoost saved to {xgb_path}")

    results_rows = []
    for metric_key in rf_avg:
        results_rows.append({
            'metric': metric_key,
            'rf_cv_mean': rf_avg[metric_key],
            'rf_cv_std': rf_std[metric_key],
            'xgb_cv_mean': xgb_avg[metric_key],
            'xgb_cv_std': xgb_std[metric_key],
        })
    results_df = pd.DataFrame(results_rows)
    results_path = os.path.join(model_dir, 'training_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nCV comparison saved to {results_path}")

    subgroup_metrics = pd.concat(
        [
            rf_subgroups.assign(method="RandomForest"),
            xgb_subgroups.assign(method="XGBoost"),
        ],
        ignore_index=True,
    )
    subgroup_path = os.path.join(model_dir, "training_subgroup_metrics.csv")
    subgroup_metrics.to_csv(subgroup_path, index=False)
    print(f"Subgroup CV metrics saved to {subgroup_path}")

    feature_imp = pd.DataFrame({
        'feature': feature_cols_used,
        'rf_importance': rf_final.feature_importances_,
        'xgb_importance': xgb_final.feature_importances_,
    }).sort_values('rf_importance', ascending=False)
    imp_path = os.path.join(model_dir, 'feature_importances.csv')
    feature_imp.to_csv(imp_path, index=False)
    print(f"Feature importances saved to {imp_path}")

    if not rf_oof.empty:
        rf_oof_out = rf_oof.copy()
        rf_oof_out["method"] = "RandomForest"
        rf_oof_out["feature_mode"] = feature_mode
        rf_oof_path = os.path.join(model_dir, "rf_oof_predictions.csv")
        rf_oof_out.to_csv(rf_oof_path, index=False)
        threshold_payload = pick_operating_threshold(
            rf_oof_out,
            score_col="score",
            label_col="label",
            group_col="virus",
            min_group_size=15,
        )
        threshold_payload["method"] = "RandomForest"
        threshold_payload["feature_mode"] = feature_mode
        threshold_payload["selection_objective"] = (
            "maximize minimum subgroup F1, then overall F1, then recall"
        )
        threshold_path = os.path.join(model_dir, "optimal_thresholds.json")
        with open(threshold_path, "w", encoding="utf-8") as f:
            json.dump(threshold_payload, f, indent=2)
        print(f"OOF predictions saved to {rf_oof_path}")
        print(f"Optimal threshold summary saved to {threshold_path}")

    return rf_final, xgb_final, rf_avg, xgb_avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SESTRAV immunogenicity classifiers')
    parser.add_argument('--data', required=True, help='Path to immunogenicity_dataset.csv')
    parser.add_argument('--model-dir', default='models', help='Output directory for serialized models')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds (default: 5)')
    parser.add_argument('--feature-mode', type=int, default=21, choices=[21, 30],
                        help='Feature mode: 21 (sequence-only) or 30 (physico + multi-allele binding)')
    parser.add_argument('--binding-matrix', default=None,
                        help='Path to peptide_binding_matrix.csv (required for --feature-mode 30)')
    args = parser.parse_args()
    train_models(args.data, args.model_dir, n_cv_folds=args.cv_folds,
                 feature_mode=args.feature_mode, binding_matrix_path=args.binding_matrix)
