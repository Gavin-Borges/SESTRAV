"""
SESTRAV Optional Deep Learning Module: Graph Neural Network (GNN)

Transforms peptide sequences into chain molecular graphs, then classifies
immunogenicity using Graph Convolutional Networks (GCN) in base PyTorch.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.features import TRAIN_FEATURE_COLUMNS, FEATURE_COLUMNS_50
from src.train_classifier import prepare_features, prepare_features_50
from src.evaluate_metrics import evaluate
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

def build_chain_adj(max_len=11):
    """Build a normalized adjacency matrix for a chain graph with self-loops."""
    # A is adjacency matrix, I is identity
    A = torch.zeros((max_len, max_len))
    for i in range(max_len):
        A[i, i] = 1.0 # self loop
        if i > 0:
            A[i, i-1] = 1.0
        if i < max_len - 1:
            A[i, i+1] = 1.0
            
    # D is degree vector
    d = torch.sum(A, dim=1)
    d_inv_sqrt = torch.pow(d, -0.5)
    # Set inf to 0 just in case (though it shouldn't happen here)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    
    # D^(-1/2) A D^(-1/2)
    norm_A = torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)
    return norm_A

def sequence_to_node_features(seq, max_len=11):
    """Convert peptide to node feature matrix (max_len, num_features)."""
    # Features per node: one-hot encoded AA (size 20)
    features = torch.zeros((max_len, 20))
    for i, aa in enumerate(seq[:max_len]):
        if aa in AA_TO_IDX:
            features[i, AA_TO_IDX[aa]] = 1.0
    return features

class GraphPeptideDataset(Dataset):
    def __init__(self, df, feature_matrix, labels=None):
        self.sequences = df['peptide'].values
        self.physico_features = torch.tensor(feature_matrix.values, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        node_feats = sequence_to_node_features(seq)
        physico = self.physico_features[idx]
        if self.labels is not None:
            return node_feats, physico, self.labels[idx]
        return node_feats, physico

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: (batch, max_len, in_features)
        # adj: (max_len, max_len)
        support = torch.matmul(x, self.weight) # (batch, max_len, out_features)
        output = torch.matmul(adj, support)    # (batch, max_len, out_features)
        return output + self.bias

class PeptideGNN(nn.Module):
    def __init__(self, num_continuous_features, dropout_rate=0.3):
        super(PeptideGNN, self).__init__()
        
        # Precomputed normalized adjacency matrix
        self.register_buffer('adj', build_chain_adj())
        
        # GCN layers for Graph feature extraction
        self.gcn1 = GCNLayer(20, 32)
        self.gcn2 = GCNLayer(32, 64)
        
        # Dense layers for continuous SESTRAV physicochemical features
        self.physico_block = nn.Sequential(
            nn.Linear(num_continuous_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Fusion block
        self.fusion_block = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, node_x, feat_x):
        # node_x shape: (batch, max_len, 20)
        h = torch.relu(self.gcn1(node_x, self.adj))
        h = torch.relu(self.gcn2(h, self.adj)) # (batch, max_len, 64)
        
        # Global mean pooling over nodes
        gnn_out = torch.mean(h, dim=1) # (batch, 64)
        
        # feat_x shape: (batch, num_continuous_features)
        physico_out = self.physico_block(feat_x)
        
        # Concatenate and classify
        fused = torch.cat((gnn_out, physico_out), dim=1)
        out = self.fusion_block(fused)
        return out.squeeze(1)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for node_x, feat_x, y in dataloader:
        node_x, feat_x, y = node_x.to(device), feat_x.to(device), y.to(device)
        
        optimizer.zero_grad()
        if torch.isnan(node_x).any():
            print("NaN in node_x!")
            import sys; sys.exit(1)
        if torch.isnan(feat_x).any():
            print("NaN in feat_x!")
            import sys; sys.exit(1)
        logits = model(node_x, feat_x)
        if torch.isnan(logits).any():
            print("NaN in logits!")
            import sys; sys.exit(1)
        loss = criterion(logits, y)
        if torch.isnan(loss).any():
            print("NaN in loss!")
            import sys; sys.exit(1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for node_x, feat_x, y in dataloader:
            node_x, feat_x = node_x.to(device), feat_x.to(device)
            logits = model(node_x, feat_x)
            probs = torch.sigmoid(logits)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

def train_gnn(data_path, model_dir='models/gnn', epochs=15, batch_size=64, lr=1e-3, feature_mode=21, binding_matrix_path=None):
    torch.autograd.set_detect_anomaly(True)
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    df = pd.read_csv(data_path)
    gs_mask = df['peptide'].isin(GOLD_STANDARD_EPITOPES)
    train_pool = df[~gs_mask].copy().reset_index(drop=True)
    print(f"Training pool: {len(train_pool)} records")

    # 2. Extract physicochemical features
    print(f"Extracting SESTRAV physicochemical features (mode {feature_mode})...")
    if feature_mode == 50:
        if binding_matrix_path is None:
            raise ValueError("binding_matrix_path required for feature mode 50")
        X_feats = prepare_features_50(train_pool, binding_matrix_path)
    else:
        X_feats = prepare_features(train_pool, include_binding=False)
    y = train_pool['label'].values
    
    # 3. Stratified K-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_feats, y), 1):
        print(f"\n--- Fold {fold} ---")
        df_train, df_val = train_pool.iloc[train_idx], train_pool.iloc[val_idx]
        X_train, X_val = X_feats.iloc[train_idx], X_feats.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize continuous features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Convert to Datasets
        train_dataset = GraphPeptideDataset(df_train, pd.DataFrame(X_train_scaled), y_train)
        val_dataset = GraphPeptideDataset(df_val, pd.DataFrame(X_val_scaled), y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        # Initialize model
        model = PeptideGNN(num_continuous_features=X_feats.shape[1]).to(device)
        
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

    # Summary Metrics
    avg = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    std = {k: np.std([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    
    print(f"\n{'=' * 40}")
    print(f"Graph Neural Network (GNN) 5-Fold CV Results:")
    print(f"{'=' * 40}")
    print(f"Mean AUC-ROC: {avg['auc_roc']:.4f} (±{std['auc_roc']:.4f})")
    print(f"Mean AUC-PR:  {avg['auc_pr']:.4f} (±{std['auc_pr']:.4f})")
    print(f"Mean ISSR@10: {avg['issr_10']:.4f} (±{std['issr_10']:.4f})")
    
    # Retrain on full dataset
    print("\nRetraining final GNN model on all data...")
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_feats)
    full_dataset = GraphPeptideDataset(train_pool, pd.DataFrame(X_full_scaled), y)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    model_final = PeptideGNN(num_continuous_features=X_feats.shape[1]).to(device)
    pos_weight_full = torch.tensor([(len(y) - y.sum()) / max(1, y.sum())]).to(device)
    criterion_final = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)
    optimizer_final = optim.Adam(model_final.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(epochs):
        train_epoch(model_final, full_loader, criterion_final, optimizer_final, device)
        
    # Save model and scaler
    torch.save(model_final.state_dict(), os.path.join(model_dir, 'gnn_model.pth'))
    import joblib
    joblib.dump(scaler_full, os.path.join(model_dir, 'gnn_scaler.joblib'))
    print(f"Final GNN model and scaler saved to {model_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GNN model on immunogenicity data')
    parser.add_argument('--data', required=True, help='Path to immunogenicity_dataset.csv')
    parser.add_argument('--model-dir', default='models/gnn', help='Output directory')
    parser.add_argument('--epochs', type=int, default=15, help='Training epochs per fold')
    parser.add_argument('--feature-mode', type=int, default=21, help='Feature mode (21 or 50)')
    parser.add_argument('--binding-matrix', default=None, help='Path to peptide_binding_matrix.csv')
    args = parser.parse_args()
    
    train_gnn(args.data, args.model_dir, epochs=args.epochs, feature_mode=args.feature_mode, binding_matrix_path=args.binding_matrix)
