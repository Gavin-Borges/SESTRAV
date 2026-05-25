"""
SESTRAV Optional Deep Learning Module: 1D-CNN (ANN)

Trains a 1D Convolutional Neural Network (CNN) on one-hot encoded peptide sequences
fused with the SESTRAV canonical physicochemical descriptor set.
Provides a deep learning track alternative to XGBoost/RandomForest.
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

# Amino acid vocabulary for One-Hot Encoding
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

def sequence_to_tensor(seq, max_len=11):
    """Convert a peptide sequence to a one-hot encoded matrix of shape (20, max_len)."""
    tensor = torch.zeros((len(AA_VOCAB), max_len))
    for i, aa in enumerate(seq[:max_len]):
        if aa in AA_TO_IDX:
            tensor[AA_TO_IDX[aa], i] = 1.0
    return tensor

class PeptideDataset(Dataset):
    def __init__(self, df, feature_matrix, labels=None):
        self.sequences = df['peptide'].values
        self.features = torch.tensor(feature_matrix.values, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_tensor = sequence_to_tensor(seq)
        feat_tensor = self.features[idx]
        if self.labels is not None:
            return seq_tensor, feat_tensor, self.labels[idx]
        return seq_tensor, feat_tensor

class PeptideCNN(nn.Module):
    def __init__(self, num_continuous_features, dropout_rate=0.3):
        super(PeptideCNN, self).__init__()
        
        # 1D CNN for Sequence feature extraction
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) # Global Max Pooling
        )
        
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

    def forward(self, seq_x, feat_x):
        # seq_x shape: (batch, 20, max_len)
        cnn_out = self.conv_block(seq_x) # shape: (batch, 64, 1)
        cnn_out = cnn_out.squeeze(-1)    # shape: (batch, 64)
        
        # feat_x shape: (batch, num_continuous_features)
        physico_out = self.physico_block(feat_x)
        
        # Concatenate and classify
        fused = torch.cat((cnn_out, physico_out), dim=1)
        out = self.fusion_block(fused)
        return out.squeeze(1)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for seq_x, feat_x, y in dataloader:
        seq_x, feat_x, y = seq_x.to(device), feat_x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(seq_x, feat_x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for seq_x, feat_x, y in dataloader:
            seq_x, feat_x = seq_x.to(device), feat_x.to(device)
            logits = model(seq_x, feat_x)
            probs = torch.sigmoid(logits)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

def train_ann(data_path, model_dir='models/ann', epochs=15, batch_size=64, lr=1e-3, feature_mode=21, binding_matrix_path=None):
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
        train_dataset = PeptideDataset(df_train, pd.DataFrame(X_train_scaled), y_train)
        val_dataset = PeptideDataset(df_val, pd.DataFrame(X_val_scaled), y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        # Initialize model
        model = PeptideCNN(num_continuous_features=X_feats.shape[1]).to(device)
        
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
    print(f"1D-CNN (ANN) 5-Fold CV Results:")
    print(f"{'=' * 40}")
    print(f"Mean AUC-ROC: {avg['auc_roc']:.4f} (±{std['auc_roc']:.4f})")
    print(f"Mean AUC-PR:  {avg['auc_pr']:.4f} (±{std['auc_pr']:.4f})")
    print(f"Mean ISSR@10: {avg['issr_10']:.4f} (±{std['issr_10']:.4f})")
    
    # Retrain on full dataset
    print("\nRetraining final ANN model on all data...")
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_feats)
    full_dataset = PeptideDataset(train_pool, pd.DataFrame(X_full_scaled), y)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    model_final = PeptideCNN(num_continuous_features=X_feats.shape[1]).to(device)
    pos_weight_full = torch.tensor([(len(y) - y.sum()) / max(1, y.sum())]).to(device)
    criterion_final = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)
    optimizer_final = optim.Adam(model_final.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(epochs):
        train_epoch(model_final, full_loader, criterion_final, optimizer_final, device)
        
    # Save model and scaler
    torch.save(model_final.state_dict(), os.path.join(model_dir, 'ann_model.pth'))
    import joblib
    joblib.dump(scaler_full, os.path.join(model_dir, 'ann_scaler.joblib'))
    print(f"Final model and scaler saved to {model_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 1D-CNN ANN model on immunogenicity data')
    parser.add_argument('--data', required=True, help='Path to immunogenicity_dataset.csv')
    parser.add_argument('--model-dir', default='models/ann', help='Output directory')
    parser.add_argument('--epochs', type=int, default=15, help='Training epochs per fold')
    parser.add_argument('--feature-mode', type=int, default=21, help='Feature mode (21 or 50)')
    parser.add_argument('--binding-matrix', default=None, help='Path to peptide_binding_matrix.csv')
    args = parser.parse_args()
    
    train_ann(args.data, args.model_dir, epochs=args.epochs, feature_mode=args.feature_mode, binding_matrix_path=args.binding_matrix)
