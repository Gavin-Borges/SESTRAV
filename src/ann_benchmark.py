"""
SESTRAV ANN Benchmark — CMB 523 Neural Network Comparison

Trains a simple feedforward neural network (MLP) on the same 21 sequence-only
features and IEDB immunogenicity data used by the RF and XGBoost classifiers,
enabling a direct apples-to-apples comparison.

Architecture:
    Input (21) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) →
    Dropout(0.3) → Dense(1, Sigmoid)

Training:
    - Binary cross-entropy loss with class weighting
    - Adam optimizer (lr=1e-3)
    - Early stopping on validation AUC-PR (patience=15)
    - Stratified 5-fold CV for metric estimates (same folds as tree models)
    - Final model retrained on full pool

Usage:
    python -m src.ann_benchmark --data immunogenicity_dataset.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from joblib import dump

from src.features import compute_features, TRAIN_FEATURE_COLUMNS
from src.evaluate_metrics import evaluate
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES
from src.subgroup_eval import evaluate_subgroups


class ImmunogenicityMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _prepare_features(df):
    """Compute 21-feature matrix from peptide sequences."""
    records = []
    for _, row in df.iterrows():
        feats = compute_features(row['peptide'], binding_score=0.0)
        records.append(feats)
    return pd.DataFrame(records)[TRAIN_FEATURE_COLUMNS]


def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _predict(model, X_tensor, device):
    model.eval()
    return model(X_tensor.to(device)).cpu().numpy()


def _train_mlp(X_train, y_train, X_val, y_val, class_weight, device,
               max_epochs=200, patience=15, lr=1e-3, batch_size=64):
    """Train MLP with early stopping on validation AUC-PR."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    g = torch.Generator()
    g.manual_seed(0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=g)

    model = ImmunogenicityMLP(X_train.shape[1]).to(device)
    criterion = nn.BCELoss(weight=None, reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc_pr = -1
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            batch_weights = torch.where(
                y_batch >= 0.5,
                torch.tensor(class_weight[1], device=device),
                torch.tensor(class_weight[0], device=device),
            )
            loss = (criterion(preds, y_batch) * batch_weights).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)

        val_scores = _predict(model, X_val_t, device)
        val_metrics = evaluate(y_val, val_scores)
        auc_pr = val_metrics['auc_pr']

        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, scaler


def _seed_everything(seed):
    """Pin all RNG sources for reproducible ANN training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_ann(data_path, model_dir='models', n_cv_folds=5, random_state=42):
    """Full ANN training pipeline with CV and final model serialization."""
    _seed_everything(random_state)
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {data_path}")

    gs_mask = df['peptide'].isin(GOLD_STANDARD_EPITOPES)
    train_pool = df[~gs_mask].copy()
    print(f"Held out {gs_mask.sum()} gold-standard records")
    print(f"Training pool: {len(train_pool)} records")

    X = _prepare_features(train_pool)
    y = train_pool['label'].values
    metadata_cols = [c for c in ["peptide", "virus", "strain"] if c in train_pool.columns]
    metadata = train_pool[metadata_cols].copy().reset_index(drop=True)
    subgroup_columns = [c for c in ["virus", "strain"] if c in train_pool.columns]
    print(f"Features: {X.shape[1]} (sequence-only)")
    print(f"Class balance: {np.mean(y):.2%} positive, {1 - np.mean(y):.2%} negative")

    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    total = n_neg + n_pos
    class_weight = {0: total / (2 * n_neg), 1: total / (2 * n_pos)}

    skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
    fold_metrics = []
    subgroup_rows = []
    oof_rows = []

    print(f"\n{'=' * 60}")
    print(f"ANN (MLP) {n_cv_folds}-fold cross-validation:")
    print(f"{'=' * 60}")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y[train_idx], y[val_idx]

        model, scaler = _train_mlp(
            X_tr, y_tr, X_val, y_val, class_weight, device
        )

        X_val_s = scaler.transform(X_val)
        X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
        scores = _predict(model, X_val_t, device)
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
            min_group_size=15,
        ):
            subgroup_rows.append({"fold": fold_idx, **row})

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

    print(f"  Mean:  " + "  ".join(f"{k}={v:.4f}" for k, v in avg.items()))
    print(f"  Stdev: " + "  ".join(f"{k}={v:.4f}" for k, v in std.items()))

    print(f"\n{'=' * 60}")
    print("Retraining final ANN on full training pool...")
    print(f"{'=' * 60}")

    X_full = X.values
    y_full = y
    half = len(X_full) // 5
    final_model, final_scaler = _train_mlp(
        X_full, y_full,
        X_full[:half], y_full[:half],
        class_weight, device,
        max_epochs=200, patience=20,
    )

    model_path = os.path.join(model_dir, 'ann_21feature_legacy.pt')
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'scaler_mean': torch.tensor(final_scaler.mean_),
        'scaler_scale': torch.tensor(final_scaler.scale_),
        'n_features': X.shape[1],
    }, model_path)
    print(f"  ANN saved to {model_path}")

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

    subgroup_df = pd.DataFrame(subgroup_rows)
    subgroup_path = os.path.join(model_dir, "ann_subgroup_metrics.csv")
    subgroup_df.to_csv(subgroup_path, index=False)
    print(f"  ANN subgroup CV metrics saved to {subgroup_path}")

    if oof_rows:
        oof_df = pd.concat(oof_rows, ignore_index=True)
        oof_path = os.path.join(model_dir, "ann_oof_predictions.csv")
        oof_df.to_csv(oof_path, index=False)
        print(f"  ANN OOF predictions saved to {oof_path}")

    return final_model, final_scaler, avg, std


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SESTRAV ANN benchmark')
    parser.add_argument('--data', required=True, help='Path to immunogenicity_dataset.csv')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--cv-folds', type=int, default=5)
    args = parser.parse_args()
    train_ann(args.data, args.model_dir, n_cv_folds=args.cv_folds)
