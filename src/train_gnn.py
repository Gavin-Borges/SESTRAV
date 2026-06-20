"""
SESTRAV Optional Deep Learning Module: Graph Neural Network (GNN)

Transforms peptide sequences into chain molecular graphs, then classifies
immunogenicity using Graph Convolutional Networks (GCN) in base PyTorch.
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.train_classifier import prepare_features, prepare_features_50
from src.evaluate_metrics import evaluate
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES

from src.gnn.graph_builder import GraphBuilder
from src.gnn.models import GraphPredictor

class GraphPeptideDataset(Dataset):
    def __init__(self, df, feature_matrix, labels=None, max_len=11, cache_dir=None, use_spatial=False):
        self.sequences = df['peptide'].values
        self.physico_features = torch.tensor(feature_matrix.values, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        self.max_len = max_len
        self.cache_dir = cache_dir
        self.use_spatial = use_spatial

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        node_feats = GraphBuilder.sequence_to_node_features(seq, max_len=self.max_len)
        if self.use_spatial and self.cache_dir:
            adj = GraphBuilder.build_spatial_adj(seq, self.cache_dir, max_len=self.max_len)
        else:
            adj = GraphBuilder.build_chain_adj(max_len=self.max_len)
        physico = self.physico_features[idx]
        if self.labels is not None:
            return node_feats, physico, adj, self.labels[idx]
        return node_feats, physico, adj

import time

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    # Profiling variables
    prof_construction = 0.0
    prof_encoding = 0.0
    prof_forward = 0.0
    prof_backward = 0.0

    for node_x, feat_x, adj, y in dataloader:
        t0 = time.perf_counter()
        node_x, feat_x, adj, y = node_x.to(device), feat_x.to(device), adj.to(device), y.to(device)
        t1 = time.perf_counter()
        prof_construction += (t1 - t0)
        
        optimizer.zero_grad()
        
        # We manually call encoder and fusion to profile encoding vs forward pass
        t_enc0 = time.perf_counter()
        gnn_out = model.encoder(node_x, adj)
        t_enc1 = time.perf_counter()
        prof_encoding += (t_enc1 - t_enc0)
        
        t_fwd0 = time.perf_counter()
        physico_out = model.physico_block(feat_x)
        fused = torch.cat((gnn_out, physico_out), dim=1)
        logits = model.fusion_block(fused).squeeze(1)
        loss = criterion(logits, y)
        t_fwd1 = time.perf_counter()
        prof_forward += (t_fwd1 - t_fwd0)
        
        t_bwd0 = time.perf_counter()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_bwd1 = time.perf_counter()
        prof_backward += (t_bwd1 - t_bwd0)
        
        total_loss += loss.item()
        
    print(f"    [Profile] Data: {prof_construction:.4f}s | Encode: {prof_encoding:.4f}s | Fwd: {prof_forward:.4f}s | Bwd: {prof_backward:.4f}s")
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for node_x, feat_x, adj, y in dataloader:
            node_x, feat_x, adj = node_x.to(device), feat_x.to(device), adj.to(device)
            logits = model(node_x, feat_x, adj)
            probs = torch.sigmoid(logits)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs (Python, NumPy, Torch) for reproducible GNN runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_gnn(data_path, model_dir='models/gnn', epochs=15, batch_size=64, lr=1e-3, feature_mode=21, binding_matrix_path=None, seed=42):
    from src.core.config import SestravConfig
    from src.core.feature_store import FeatureStore

    config = SestravConfig.load("config.yaml")
    store = FeatureStore(config.output_dir)

    set_seed(seed)
    torch.autograd.set_detect_anomaly(True)
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    df = pd.read_csv(data_path)
    gs_mask = df['peptide'].isin(GOLD_STANDARD_EPITOPES)
    train_pool = df[~gs_mask].copy().reset_index(drop=True)
    print(f"Training pool: {len(train_pool)} records")

    # 2. Extract physicochemical features (with Cache resolution)
    # Include dataset fingerprint so switching datasets invalidates the cache.
    import hashlib as _hl
    _data_tag = _hl.md5(open(data_path, "rb").read(65536)).hexdigest()[:8]  # nosec B324
    cache_name = f"physico_features_mode{feature_mode}_{_data_tag}.csv"
    X_feats = store.load_cached_features(cache_name)
    if X_feats is None:
        print(f"Extracting SESTRAV physicochemical features (mode {feature_mode})...")
        if feature_mode == 50:
            if binding_matrix_path is None:
                raise ValueError("binding_matrix_path required for feature mode 50")
            X_feats = prepare_features_50(train_pool, binding_matrix_path)
        else:
            X_feats = prepare_features(train_pool, include_binding=False)
        store.save_cached_features(X_feats, cache_name)
    else:
        # If cache contains non-feature columns (e.g. metadata), align with target features
        non_feat_cols = [c for c in ['peptide', 'label', 'protein', 'allele'] if c in X_feats.columns]
        if non_feat_cols:
            X_feats = X_feats.drop(columns=non_feat_cols)
    y = train_pool['label'].values
    
    # 3. Stratified K-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_metrics = []
    oof_rows = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_feats, y), 1):
        print(f"\n--- Fold {fold} ---")
        df_train, df_val = train_pool.iloc[train_idx], train_pool.iloc[val_idx]
        X_train, X_val = X_feats.iloc[train_idx], X_feats.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize continuous features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Convert to Datasets with config parameters
        train_dataset = GraphPeptideDataset(
            df_train, pd.DataFrame(X_train_scaled), y_train,
            max_len=config.max_peptide_length,
            cache_dir=config.structural_cache_dir,
            use_spatial=config.use_spatial_adj
        )
        val_dataset = GraphPeptideDataset(
            df_val, pd.DataFrame(X_val_scaled), y_val,
            max_len=config.max_peptide_length,
            cache_dir=config.structural_cache_dir,
            use_spatial=config.use_spatial_adj
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        # Initialize model
        model = GraphPredictor(num_continuous_features=X_feats.shape[1]).to(device)
        
        # Positive weight for class imbalance
        pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(1, y_train.sum())]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Train
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            
        # Evaluate
        val_labels, val_preds = evaluate_model(model, val_loader, device)
        m = evaluate(val_labels, val_preds)
        fold_metrics.append(m)
        print(f"Fold {fold} - AUC-ROC: {m['auc_roc']:.4f} | AUC-PR: {m['auc_pr']:.4f} | ISSR@10: {m['issr_10']:.4f}")
        
        for i, idx_val in enumerate(val_idx):
            oof_rows.append({
                'peptide': train_pool['peptide'].iloc[idx_val],
                'label': val_labels[i],
                'gnn_oof_score': val_preds[i]
            })
 
    # Save OOF predictions
    oof_df = pd.DataFrame(oof_rows)
    oof_path = os.path.join(os.path.dirname(model_dir), 'gnn_oof_predictions.csv')
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved GNN OOF predictions to {oof_path}")
 
    # Summary Metrics
    avg = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    std = {k: np.std([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    
    print(f"\n{'=' * 40}")
    print("Graph Neural Network (GNN) 5-Fold CV Results:")
    print(f"{'=' * 40}")
    print(f"Mean AUC-ROC: {avg['auc_roc']:.4f} (±{std['auc_roc']:.4f})")
    print(f"Mean AUC-PR:  {avg['auc_pr']:.4f} (±{std['auc_pr']:.4f})")
    print(f"Mean ISSR@10: {avg['issr_10']:.4f} (±{std['issr_10']:.4f})")
    
    # Retrain on full dataset
    print("\nRetraining final GNN model on all data...")
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_feats)
    full_dataset = GraphPeptideDataset(
        train_pool, pd.DataFrame(X_full_scaled), y,
        max_len=config.max_peptide_length,
        cache_dir=config.structural_cache_dir,
        use_spatial=config.use_spatial_adj
    )
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    model_final = GraphPredictor(num_continuous_features=X_feats.shape[1]).to(device)
    pos_weight_full = torch.tensor([(len(y) - y.sum()) / max(1, y.sum())]).to(device)
    criterion_final = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)
    optimizer_final = optim.Adam(model_final.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(epochs):
        train_epoch(model_final, full_loader, criterion_final, optimizer_final, device)
        
    # Save model and scaler
    torch.save(model_final.state_dict(), os.path.join(model_dir, 'structural_gnn_v2.pth'))  # nosec B614
    import joblib
    joblib.dump(scaler_full, os.path.join(model_dir, 'gnn_scaler.joblib'))
    print(f"Final GNN model and scaler saved to {model_dir}/")
 
# ---------------------------------------------------------------------------
# GNN v2.1 — GINEConv + ESM-2 node embeddings
# ---------------------------------------------------------------------------

class GraphPeptideDatasetV2(torch.utils.data.Dataset):
    """PyG-compatible dataset returning Data objects with ESM-2 node features.

    Each item:
        x:        (max_len, 320)  — ESM-2 per-residue embeddings
        edge_index: (2, num_edges) — chain graph, local node indices
        edge_attr: (num_edges, 3)  — one-hot edge type features
        physico:  (1, num_features) — batches to (B, num_features) via PyG collation
        y:        (1,)             — batches to (B,) via PyG collation
    """
    def __init__(self, df, feature_matrix, labels, esm2_cache, max_len=11):
        self.sequences = df['peptide'].values
        self.physico_features = torch.tensor(feature_matrix.values, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        self.esm2_cache = esm2_cache
        self.max_len = max_len
        self.edge_index, self.edge_attr = GraphBuilder.build_pyg_chain_graph(max_len)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        from torch_geometric.data import Data
        seq = self.sequences[idx]
        node_feats = self.esm2_cache[seq]  # (max_len, 320)
        label = self.labels[idx].view(1) if self.labels is not None else torch.zeros(1)
        return Data(
            x=node_feats,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            physico=self.physico_features[idx].unsqueeze(0),
            y=label,
        )


def train_epoch_v2(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_model_v2(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            probs = torch.sigmoid(model(batch))
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.view(-1).cpu().numpy())
    return np.array(all_labels), np.array(all_preds)


def train_gnn_v2(data_path, model_dir='models/gnn', epochs=20, batch_size=64,
                 lr=3e-4, feature_mode=21, esm2_cache_path='data/esm2_embeddings.pt', seed=42):
    """Train GNN v2.1: GINEConv message passing with ESM-2 per-residue node embeddings."""
    from src.core.config import SestravConfig
    from src.core.feature_store import FeatureStore
    from src.gnn.models import GraphPredictorV2
    from torch_geometric.loader import DataLoader as PyGDataLoader

    config = SestravConfig.load("config.yaml")
    store = FeatureStore(config.output_dir)

    set_seed(seed)
    torch.autograd.set_detect_anomaly(True)
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load ESM-2 embeddings cache
    if not os.path.exists(esm2_cache_path):
        raise FileNotFoundError(
            f"ESM-2 embeddings not found at {esm2_cache_path}. "
            "Run: python scripts/precompute_esm2_embeddings.py first."
        )
    print(f"Loading ESM-2 embeddings from {esm2_cache_path} ...")
    esm2_cache = torch.load(esm2_cache_path, weights_only=True)  # nosec B614
    print(f"Loaded {len(esm2_cache)} peptide embeddings")

    # Load data
    df = pd.read_csv(data_path)
    gs_mask = df['peptide'].isin(GOLD_STANDARD_EPITOPES)
    train_pool = df[~gs_mask].copy().reset_index(drop=True)
    print(f"Training pool: {len(train_pool)} records")

    # Verify ESM-2 coverage
    missing = set(train_pool['peptide'].values) - set(esm2_cache.keys())
    if missing:
        raise ValueError(
            f"{len(missing)} peptides missing from ESM-2 cache. "
            "Re-run: python scripts/precompute_esm2_embeddings.py"
        )

    # Extract physicochemical features (same cache as v1, fused at output)
    import hashlib as _hl
    _data_tag = _hl.md5(open(data_path, "rb").read(65536)).hexdigest()[:8]  # nosec B324
    cache_name = f"physico_features_mode{feature_mode}_{_data_tag}.csv"
    X_feats = store.load_cached_features(cache_name)
    if X_feats is None:
        print(f"Extracting physicochemical features (mode {feature_mode}) ...")
        X_feats = prepare_features(train_pool, include_binding=False)
        store.save_cached_features(X_feats, cache_name)
    else:
        non_feat_cols = [c for c in ['peptide', 'label', 'protein', 'allele'] if c in X_feats.columns]
        if non_feat_cols:
            X_feats = X_feats.drop(columns=non_feat_cols)
    y = train_pool['label'].values

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_metrics = []
    oof_rows = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_feats, y), 1):
        print(f"\n--- Fold {fold} ---")
        df_train = train_pool.iloc[train_idx]
        df_val = train_pool.iloc[val_idx]
        X_train, X_val = X_feats.iloc[train_idx], X_feats.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        train_dataset = GraphPeptideDatasetV2(
            df_train, pd.DataFrame(X_train_scaled), y_train, esm2_cache,
            max_len=config.max_peptide_length,
        )
        val_dataset = GraphPeptideDatasetV2(
            df_val, pd.DataFrame(X_val_scaled), y_val, esm2_cache,
            max_len=config.max_peptide_length,
        )

        train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = GraphPredictorV2(num_continuous_features=X_feats.shape[1]).to(device)

        pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(1, y_train.sum())]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            loss = train_epoch_v2(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs} — loss: {loss:.4f}")

        val_labels, val_preds = evaluate_model_v2(model, val_loader, device)
        m = evaluate(val_labels, val_preds)
        fold_metrics.append(m)
        print(f"Fold {fold} - AUC-ROC: {m['auc_roc']:.4f} | AUC-PR: {m['auc_pr']:.4f} | ISSR@10: {m['issr_10']:.4f}")

        for i, idx_val in enumerate(val_idx):
            oof_rows.append({
                'peptide': train_pool['peptide'].iloc[idx_val],
                'label': val_labels[i],
                'gnn_oof_score': val_preds[i],
            })

    # Save OOF predictions
    oof_df = pd.DataFrame(oof_rows)
    oof_path = os.path.join(os.path.dirname(model_dir), 'gnn_oof_predictions.csv')
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved GNN OOF predictions to {oof_path}")

    # Summary
    avg = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    std = {k: np.std([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    print(f"\n{'=' * 40}")
    print("Graph Neural Network v2.1 (GINEConv + ESM-2) 5-Fold CV Results:")
    print(f"{'=' * 40}")
    print(f"Mean AUC-ROC: {avg['auc_roc']:.4f} (±{std['auc_roc']:.4f})")
    print(f"Mean AUC-PR:  {avg['auc_pr']:.4f} (±{std['auc_pr']:.4f})")
    print(f"Mean ISSR@10: {avg['issr_10']:.4f} (±{std['issr_10']:.4f})")

    # Retrain on full data
    print("\nRetraining final GNN v2.1 model on all data ...")
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_feats)
    full_dataset = GraphPeptideDatasetV2(
        train_pool, pd.DataFrame(X_full_scaled), y, esm2_cache,
        max_len=config.max_peptide_length,
    )
    full_loader = PyGDataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    model_final = GraphPredictorV2(num_continuous_features=X_feats.shape[1]).to(device)
    pos_weight_full = torch.tensor([(len(y) - y.sum()) / max(1, y.sum())]).to(device)
    criterion_final = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)
    optimizer_final = optim.Adam(model_final.parameters(), lr=lr, weight_decay=1e-4)
    scheduler_final = optim.lr_scheduler.CosineAnnealingLR(optimizer_final, T_max=epochs)

    for epoch in range(epochs):
        train_epoch_v2(model_final, full_loader, criterion_final, optimizer_final, device)
        scheduler_final.step()

    torch.save(model_final.state_dict(), os.path.join(model_dir, 'structural_gnn_v2.pth'))  # nosec B614
    import joblib
    joblib.dump(scaler_full, os.path.join(model_dir, 'gnn_scaler.joblib'))
    print(f"Final GNN v2.1 model saved to {model_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GNN model on immunogenicity data')
    parser.add_argument('--data', required=True, help='Path to immunogenicity_dataset.csv')
    parser.add_argument('--model-dir', default='models/gnn', help='Output directory')
    parser.add_argument('--epochs', type=int, default=15, help='Training epochs per fold')
    parser.add_argument('--feature-mode', type=int, default=21, help='Feature mode (21 or 50)')
    parser.add_argument('--binding-matrix', default=None, help='Path to peptide_binding_matrix.csv')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible runs')
    parser.add_argument('--architecture', choices=['v1', 'v2'], default='v2',
                        help='GNN architecture: v1 (dense-adj GCN) or v2 (GINEConv + ESM-2)')
    parser.add_argument('--esm2-cache', default='data/esm2_embeddings.pt',
                        help='Path to pre-computed ESM-2 embeddings (required for v2)')
    args = parser.parse_args()

    if args.architecture == 'v2':
        train_gnn_v2(
            args.data, args.model_dir, epochs=args.epochs,
            feature_mode=args.feature_mode, esm2_cache_path=args.esm2_cache, seed=args.seed,
        )
    else:
        train_gnn(
            args.data, args.model_dir, epochs=args.epochs,
            feature_mode=args.feature_mode, binding_matrix_path=args.binding_matrix, seed=args.seed,
        )

