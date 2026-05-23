"""
SESTRAV GNN Benchmark — Graph Neural Network Comparison

Implements GCN, GAT, and Bipartite Peptide-Allele GNN architectures for
peptide immunogenicity prediction, backported from CMB 523 Project 2.

Each peptide is represented as a molecular graph:
  - Nodes: individual amino acid residues (8-11 per peptide)
  - Node features: 4 physicochemical properties per residue
    (hydrophobicity, aromaticity, VdW volume, charge)
  - Edges: bidirectional sequential backbone adjacency

The GNN learns position-dependent importance and inter-residue relationships
through message passing, capturing structural context that fixed-position
tabular features cannot represent.

Project 2 results (5-fold CV, v2 dataset):
  GCN (2-layer):            AUC-PR=0.7781, AUC-ROC=0.6138
  GAT (2-layer, 4-head):    AUC-PR=0.7956, AUC-ROC=0.6366  (best GNN)
  Bipartite Peptide-Allele: AUC-PR=0.7886, AUC-ROC=0.6124

Note: GNNs underperform the tabular RF (AUC-PR=0.8102) and ANN
(AUC-PR=0.8252) on this dataset.  Included as an exploratory benchmark
to characterize the representation space, not as a production classifier.

Requires: pip install -r requirements-gnn.txt

Usage:
    python -m src.gnn_benchmark --data immunogenicity_dataset.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from src.features import KD_HYDRO, VDW_VOL, AROMATIC, CHARGE
from src.evaluate_metrics import evaluate, summarize_fold_metrics
from src.model import set_seeds, get_device, SEED, N_FOLDS, LEARNING_RATE, WEIGHT_DECAY
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def peptide_to_graph(peptide):
    """Convert a peptide string to a PyTorch Geometric Data object.

    Node features (4 per residue):
        - Kyte-Doolittle hydrophobicity
        - Aromaticity (binary: F/W/Y/H)
        - Van der Waals volume
        - Formal charge at pH 7

    Edges: bidirectional sequential backbone adjacency.
    """
    n = len(peptide)
    x = torch.zeros(n, 4, dtype=torch.float32)
    for i, aa in enumerate(peptide):
        x[i, 0] = KD_HYDRO.get(aa, 0.0)
        x[i, 1] = float(AROMATIC.get(aa, 0))
        x[i, 2] = VDW_VOL.get(aa, 0.0)
        x[i, 3] = float(CHARGE.get(aa, 0))

    edge_list = []
    for i in range(n - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


def build_graph_dataset(peptides, labels):
    """Convert peptide sequences and labels to a list of PyG Data objects."""
    graphs = []
    for pep, lab in zip(peptides, labels):
        g = peptide_to_graph(pep)
        g.y = torch.tensor([lab], dtype=torch.float32)
        graphs.append(g)
    return graphs


# ---------------------------------------------------------------------------
# GNN Architectures
# ---------------------------------------------------------------------------

def _kaiming_init_gnn(module):
    """Apply Kaiming (He) initialization to all Linear layers in a GNN."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class PeptideGCN(nn.Module):
    """Two-layer Graph Convolutional Network for peptide immunogenicity.

    Message passing over the backbone graph captures inter-residue
    physicochemical context that tabular models cannot represent.
    """

    def __init__(self, in_channels=4, hidden=64, out_hidden=32, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_hidden, 1)
        _kaiming_init_gnn(self)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze(-1)


class PeptideGAT(nn.Module):
    """Two-layer Graph Attention Network variant.

    Uses attention-weighted message passing so the model can learn which
    neighboring residues matter most for each position's contribution
    to immunogenicity.
    """

    def __init__(self, in_channels=4, hidden=64, out_hidden=32,
                 dropout=0.3, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden // heads, heads=heads,
                             dropout=dropout)
        self.conv2 = GATConv(hidden, out_hidden, heads=1, concat=False,
                             dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_hidden, 1)
        _kaiming_init_gnn(self)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze(-1)


class BipartitePeptideAlleleGNN(nn.Module):
    """Lightweight bipartite message-passing model (peptide <-> alleles).

    Each sample is represented as:
      - 1 peptide node with 20 physicochemical features
      - 10 allele nodes with learned embeddings
      - 10 edges carrying MHC binding scores
    """

    def __init__(self, peptide_dim=20, n_alleles=10, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.allele_embed = nn.Embedding(n_alleles, hidden_dim)
        self.peptide_proj = nn.Linear(peptide_dim, hidden_dim)
        self.edge_proj = nn.Linear(1, hidden_dim)
        self.msg_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        _kaiming_init_gnn(self)

    def forward(self, peptide_x, binding_x):
        # peptide_x: (B, 20), binding_x: (B, 10)
        bsz, n_alleles = binding_x.shape
        pep_h = self.peptide_proj(peptide_x)  # (B, H)
        allele_ids = torch.arange(n_alleles, device=peptide_x.device)
        allele_h = self.allele_embed(allele_ids).unsqueeze(0).expand(bsz, -1, -1)  # (B, A, H)
        pep_exp = pep_h.unsqueeze(1).expand(-1, n_alleles, -1)  # (B, A, H)
        edge_h = self.edge_proj(binding_x.unsqueeze(-1))  # (B, A, H)
        msg = torch.cat([pep_exp, allele_h, edge_h], dim=-1)
        msg = F.relu(self.msg_proj(msg))  # (B, A, H)
        agg = msg.mean(dim=1)  # (B, H)
        return self.out(agg).squeeze(-1)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ---------------------------------------------------------------------------
# Training Loops
# ---------------------------------------------------------------------------

def train_gnn_one_fold(model, train_graphs, val_graphs, pos_weight,
                       max_epochs=150, patience=10, lr=LEARNING_RATE,
                       batch_size=64, device=None):
    """Train a GNN model on one fold with early stopping on AUC-PR."""
    if device is None:
        device = get_device()
    model = model.to(device)

    pw = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=WEIGHT_DECAY)

    best_auc_pr = -1.0
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        np.random.shuffle(train_graphs)
        for i in range(0, len(train_graphs), batch_size):
            batch_graphs = train_graphs[i:i + batch_size]
            batch = Batch.from_data_list(batch_graphs).to(device)
            labels = batch.y.to(device)

            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_batch = Batch.from_data_list(val_graphs).to(device)
            val_logits = model(val_batch).cpu().numpy()
            val_probs = _sigmoid(val_logits)
            val_labels = np.array([g.y.item() for g in val_graphs])

        if len(np.unique(val_labels)) < 2:
            auc_pr = 0.0
        else:
            auc_pr = average_precision_score(val_labels, val_probs)

        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_batch = Batch.from_data_list(val_graphs).to(device)
        val_logits = model(val_batch).cpu().numpy()
        val_probs = _sigmoid(val_logits)
        val_labels = np.array([g.y.item() for g in val_graphs])

    return evaluate(val_labels, val_probs)


def _train_bipartite_one_fold(model, Xp_train, Xb_train, y_train,
                              Xp_val, Xb_val, y_val, pos_weight,
                              max_epochs=120, patience=10, batch_size=64,
                              device=None):
    """Train a bipartite GNN on one fold."""
    if device is None:
        device = get_device()
    model = model.to(device)

    xpt = torch.tensor(Xp_train, dtype=torch.float32).to(device)
    xbt = torch.tensor(Xb_train, dtype=torch.float32).to(device)
    yt = torch.tensor(y_train, dtype=torch.float32).to(device)
    xpv = torch.tensor(Xp_val, dtype=torch.float32).to(device)
    xbv = torch.tensor(Xb_val, dtype=torch.float32).to(device)

    pw = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state, best_ap, wait = None, -1.0, 0
    n_train = len(y_train)
    for _ in range(max_epochs):
        model.train()
        idx = np.random.permutation(n_train)
        for start in range(0, n_train, batch_size):
            b = idx[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(xpt[b], xbt[b])
            loss = criterion(logits, yt[b])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(xpv, xbv)).cpu().numpy()
        if len(np.unique(y_val)) < 2:
            ap = 0.0
        else:
            ap = average_precision_score(y_val, probs)
        if ap > best_ap:
            best_ap = ap
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(xpv, xbv)).cpu().numpy()
    return evaluate(y_val, val_probs)


# ---------------------------------------------------------------------------
# CV Runners
# ---------------------------------------------------------------------------

def run_gnn_cv(peptides, labels, strat_key, model_type="gcn",
               n_folds=N_FOLDS, pos_weight=None, device=None):
    """Run full k-fold stratified CV for a GNN architecture.

    Args:
        peptides:   array of peptide strings.
        labels:     array of binary labels.
        strat_key:  array of stratification keys (label_virus).
        model_type: "gcn" or "gat".
        n_folds:    number of CV folds.
        pos_weight: float, inverse-frequency class weight.
        device:     torch.device.

    Returns:
        (fold_metrics_list, avg_dict, std_dict)
    """
    if not HAS_PYG:
        raise ImportError(
            "torch_geometric is required for GNN benchmark. "
            "Install with: pip install -r requirements-gnn.txt"
        )

    if device is None:
        device = get_device()

    peptides = np.asarray(peptides)
    labels = np.asarray(labels)

    if pos_weight is None:
        n_pos = int(labels.sum())
        pos_weight = (len(labels) - n_pos) / max(n_pos, 1)

    all_graphs = build_graph_dataset(peptides, labels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(peptides, strat_key), 1):
        set_seeds(SEED)

        train_graphs = [all_graphs[i] for i in train_idx]
        val_graphs = [all_graphs[i] for i in val_idx]

        train_feats = torch.cat([g.x for g in train_graphs], dim=0).numpy()
        node_scaler = StandardScaler().fit(train_feats)
        for g in train_graphs:
            g.x = torch.tensor(node_scaler.transform(g.x.numpy()), dtype=torch.float32)
        for g in val_graphs:
            g.x = torch.tensor(node_scaler.transform(g.x.numpy()), dtype=torch.float32)

        if model_type == "gat":
            model = PeptideGAT(in_channels=4, hidden=64, out_hidden=32,
                               dropout=0.3, heads=4)
        else:
            model = PeptideGCN(in_channels=4, hidden=64, out_hidden=32,
                               dropout=0.3)

        metrics = train_gnn_one_fold(
            model, train_graphs, val_graphs, pos_weight, device=device,
        )
        fold_metrics.append(metrics)
        print(f"    Fold {fold_idx}: AUC-ROC={metrics['auc_roc']:.4f}  "
              f"AUC-PR={metrics['auc_pr']:.4f}  "
              f"ISSR@10={metrics['issr_10']:.4f}  "
              f"ISSR@25={metrics['issr_25']:.4f}")

    avg, std = summarize_fold_metrics(fold_metrics)
    return fold_metrics, avg, std


def run_gnn_benchmark(peptides, labels, strat_key, pos_weight=None,
                      device=None):
    """Run both GCN and GAT benchmarks and return comparison DataFrame.

    Returns:
        pd.DataFrame with columns: model, auc_roc_mean, auc_roc_std,
        auc_pr_mean, auc_pr_std, issr_10_mean, issr_25_mean.
    """
    results = []

    for model_type, model_name in [("gcn", "GCN (2-layer)"),
                                    ("gat", "GAT (2-layer, 4-head)")]:
        print(f"\n{'=' * 60}")
        print(f"GNN Benchmark: {model_name}")
        print(f"{'=' * 60}")

        fold_metrics, avg, std = run_gnn_cv(
            peptides, labels, strat_key,
            model_type=model_type,
            pos_weight=pos_weight,
            device=device,
        )

        results.append({
            "model": model_name,
            "auc_roc_mean": avg["auc_roc"],
            "auc_roc_std": std["auc_roc"],
            "auc_pr_mean": avg["auc_pr"],
            "auc_pr_std": std["auc_pr"],
            "issr_10_mean": avg["issr_10"],
            "issr_25_mean": avg["issr_25"],
        })

        print(f"  Mean AUC-ROC: {avg['auc_roc']:.4f} +/- {std['auc_roc']:.4f}")
        print(f"  Mean AUC-PR:  {avg['auc_pr']:.4f} +/- {std['auc_pr']:.4f}")
        print(f"  ISSR@10:      {avg['issr_10']:.4f}")
        print(f"  ISSR@25:      {avg['issr_25']:.4f}")

    df_results = pd.DataFrame(results)
    best = df_results.sort_values("auc_pr_mean", ascending=False).iloc[0]
    print(f"\n  Best GNN by AUC-PR: {best['model']} "
          f"(AUC-PR={best['auc_pr_mean']:.4f}, AUC-ROC={best['auc_roc_mean']:.4f})")
    return df_results


def run_bipartite_gnn_benchmark(X_physico, X_binding, labels, strat_key,
                                pos_weight=None, n_folds=N_FOLDS, device=None):
    """Run exploratory bipartite peptide-allele GNN benchmark.

    Args:
        X_physico: np.ndarray of shape (n, 20) — physicochemical features.
        X_binding: np.ndarray of shape (n, 10) — per-allele binding scores.
        labels:    np.ndarray of binary labels.
        strat_key: stratification keys.
        pos_weight: class weight.
        n_folds:   CV folds.
        device:    torch.device.

    Returns:
        pd.DataFrame with one row of benchmark results.
    """
    if device is None:
        device = get_device()

    X_physico = np.asarray(X_physico, dtype=np.float32)
    X_binding = np.asarray(X_binding, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    if pos_weight is None:
        n_pos = int(labels.sum())
        pos_weight = (len(labels) - n_pos) / max(1, n_pos)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_physico, strat_key), 1):
        set_seeds(SEED)

        physico_scaler = StandardScaler().fit(X_physico[train_idx])
        binding_scaler = StandardScaler().fit(X_binding[train_idx])
        Xp_tr = physico_scaler.transform(X_physico[train_idx]).astype(np.float32)
        Xb_tr = binding_scaler.transform(X_binding[train_idx]).astype(np.float32)
        Xp_val = physico_scaler.transform(X_physico[val_idx]).astype(np.float32)
        Xb_val = binding_scaler.transform(X_binding[val_idx]).astype(np.float32)

        model = BipartitePeptideAlleleGNN(
            peptide_dim=X_physico.shape[1],
            n_alleles=X_binding.shape[1],
            hidden_dim=64,
            dropout=0.3,
        )
        metrics = _train_bipartite_one_fold(
            model,
            Xp_tr, Xb_tr, labels[train_idx],
            Xp_val, Xb_val, labels[val_idx],
            pos_weight=pos_weight, device=device,
        )
        fold_metrics.append(metrics)
        print(
            f"    Bipartite Fold {fold_idx}: "
            f"AUC-ROC={metrics['auc_roc']:.4f} AUC-PR={metrics['auc_pr']:.4f} "
            f"ISSR@10={metrics['issr_10']:.4f} ISSR@25={metrics['issr_25']:.4f}"
        )

    avg, std = summarize_fold_metrics(fold_metrics)
    return pd.DataFrame([{
        "model": "Bipartite Peptide-Allele GNN (lightweight)",
        "auc_roc_mean": avg["auc_roc"],
        "auc_roc_std": std["auc_roc"],
        "auc_pr_mean": avg["auc_pr"],
        "auc_pr_std": std["auc_pr"],
        "issr_10_mean": avg["issr_10"],
        "issr_25_mean": avg["issr_25"],
    }])


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='SESTRAV GNN Benchmark — Graph Neural Network comparison'
    )
    parser.add_argument('--data', required=True,
                        help='Path to immunogenicity_dataset.csv')
    parser.add_argument('--binding-matrix', default='models/peptide_binding_matrix.csv',
                        help='Path to peptide_binding_matrix.csv (for bipartite GNN)')
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--output-dir', default='models',
                        help='Directory for result CSVs')
    parser.add_argument('--skip-bipartite', action='store_true',
                        help='Skip the bipartite peptide-allele GNN')
    args = parser.parse_args()

    if not HAS_PYG:
        print("ERROR: torch_geometric not installed.")
        print("Install with: pip install -r requirements-gnn.txt")
        return

    set_seeds(SEED)
    device = get_device()
    print(f"Device: {device}")

    df = pd.read_csv(args.data)
    gs_mask = df['peptide'].isin(GOLD_STANDARD_EPITOPES)
    pool = df[~gs_mask].copy()
    print(f"Loaded {len(df)} records, held out {gs_mask.sum()} gold-standard")
    print(f"Training pool: {len(pool)} records")

    peptides = pool['peptide'].values
    labels = pool['label'].values
    virus = pool['virus'].values if 'virus' in pool.columns else np.zeros(len(pool))
    strat_key = np.array([f"{l}_{v}" for l, v in zip(labels, virus)])

    n_pos = int(labels.sum())
    pos_weight = (len(labels) - n_pos) / max(1, n_pos)

    # --- Sequence-graph GNN benchmark (GCN + GAT) ---
    print("\n" + "=" * 70)
    print("SEQUENCE-GRAPH GNN BENCHMARKS (GCN + GAT)")
    print("=" * 70)
    seq_results = run_gnn_benchmark(
        peptides, labels, strat_key,
        pos_weight=pos_weight, device=device,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    seq_path = os.path.join(args.output_dir, 'gnn_sequence_benchmark.csv')
    seq_results.to_csv(seq_path, index=False)
    print(f"\nSequence GNN results saved to {seq_path}")

    # --- Bipartite peptide-allele GNN benchmark ---
    if not args.skip_bipartite and os.path.isfile(args.binding_matrix):
        from src.features import PHYSICO_COLUMNS, BINDING_ALLELE_COLUMNS
        from src.train_classifier import prepare_features_30

        print("\n" + "=" * 70)
        print("BIPARTITE PEPTIDE-ALLELE GNN BENCHMARK")
        print("=" * 70)

        X_30 = prepare_features_30(pool, args.binding_matrix)
        X_physico = X_30[PHYSICO_COLUMNS].values
        X_binding = X_30[BINDING_ALLELE_COLUMNS].values

        bi_results = run_bipartite_gnn_benchmark(
            X_physico, X_binding, labels, strat_key,
            pos_weight=pos_weight, n_folds=args.cv_folds, device=device,
        )

        bi_path = os.path.join(args.output_dir, 'gnn_bipartite_benchmark.csv')
        bi_results.to_csv(bi_path, index=False)
        print(f"\nBipartite GNN results saved to {bi_path}")
    elif args.skip_bipartite:
        print("\nSkipped bipartite GNN (--skip-bipartite)")
    else:
        print(f"\nSkipped bipartite GNN (binding matrix not found: {args.binding_matrix})")


if __name__ == '__main__':
    main()
