"""
SESTRAV-CORE Structural Graph Neural Network Framework
Implements 3D spatial graph mappings, custom PyG Dataset, and mixed-precision GNN.
"""

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import logging
import json
from pathlib import Path
from typing import Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("structural-gnn")

# Safe PyG imports
try:
    from torch_geometric.data import Data, Dataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GINEConv, global_mean_pool

    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from src.features import KD_HYDRO, VDW_VOL, AROMATIC, CHARGE

# 34 pocket residue index mapping according to NetMHCpan
MHC_POCKET_COUNT = 34


def generate_canonical_groove_coords(
    peptide: str, allele_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate deterministic 3D coordinates (x, y, z) approximating the MHC-I groove interface.

    Peptide residues lie in a beta-strand configuration along the X-axis.
    MHC pocket residues are divided into:
      - Helix 1 (Y = -4.0 Å, Z = 3.0 Å)
      - Helix 2 (Y = 4.0 Å, Z = 3.0 Å)
      - Floor sheet (Y = 0.0 Å, Z = -3.0 Å)
    """
    p_len = len(peptide)
    # 1. Peptide coordinates (X-axis translation of 3.2 Å per residue)
    pep_coords = np.zeros((p_len, 3))
    for i in range(p_len):
        pep_coords[i, 0] = 3.2 * i
        pep_coords[i, 1] = 0.0 + (0.5 if i % 2 == 0 else -0.5)  # slight side-chain oscillation
        pep_coords[i, 2] = 0.0

    # 2. MHC pocket coordinates (34 residues mapping to helices/floor)
    mhc_coords = np.zeros((MHC_POCKET_COUNT, 3))
    for i in range(MHC_POCKET_COUNT):
        x_val = (i % 12) * 2.8  # span parallel to the peptide
        if i < 12:  # Helix 1
            mhc_coords[i] = [x_val, -4.0, 3.0]
        elif i < 24:  # Helix 2
            mhc_coords[i] = [x_val, 4.0, 3.0]
        else:  # Floor beta-sheet
            mhc_coords[i] = [x_val - 3.0, 0.0, -3.0]

    return pep_coords, mhc_coords


def generate_edge_features(
    pos: torch.Tensor, charges: torch.Tensor, d_max: float = 8.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes bidirectional edge indices for pairwise distances d <= 8.0 Å,
    and returns edge attributes: [distance, Coulomb potential proxy, Lennard-Jones proxy].
    """
    n = pos.size(0)
    # Pairwise coordinate diffs
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N, N, 3)
    dist = torch.norm(diff, p=2, dim=-1)  # (N, N)

    # Exclude self loops and filter by distance threshold
    mask = (dist <= d_max) & (~torch.eye(n, dtype=torch.bool, device=pos.device))
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()

    # Selected distances
    flat_dist = dist[mask].unsqueeze(-1)

    # Coulomb potential proxy: q_i * q_j / d_ij
    q_i = charges[edge_index[0]].unsqueeze(-1)
    q_j = charges[edge_index[1]].unsqueeze(-1)
    coulomb = (q_i * q_j) / (flat_dist + 1e-5)

    # Lennard-Jones potential proxy: (1/d^12) - (1/d^6)
    lj = (1.0 / (flat_dist**12 + 1e-6)) - (1.0 / (flat_dist**6 + 1e-6))

    edge_attr = torch.cat([flat_dist, coulomb, lj], dim=-1)
    return edge_index, edge_attr


class StructuralPeptideMHCDataset(Dataset if HAS_PYG else object):  # type: ignore[misc]
    """
    Custom PyTorch Geometric Dataset dynamically scaling to load structural graphs.
    """

    def __init__(self, df: pd.DataFrame, transform=None, pre_transform=None):
        if HAS_PYG:
            super().__init__("", transform, pre_transform)
        self.peptides = df["peptide"].values
        self.alleles = df["allele"].values
        self.labels = df["label"].values
        self.proteins = df["protein"].values if "protein" in df.columns else ["Unknown"] * len(df)

        # Vectorization: Pre-embed MHC node features for the 10 canonical alleles
        self.unique_alleles = sorted(list(set(self.alleles)))
        self.allele_to_idx = {a: i for i, a in enumerate(self.unique_alleles)}

        # Load exact 34-residue structural strings
        pseudo_seq_path = Path("src/verify/mhc_pseudo_sequences.json")
        if pseudo_seq_path.exists():
            with open(pseudo_seq_path, "r") as f:
                self.pseudo_seqs = json.load(f)
        else:
            self.pseudo_seqs = {}

        self.mhc_node_tensors = torch.zeros(
            (len(self.unique_alleles), MHC_POCKET_COUNT, 5), dtype=torch.float32
        )
        self.mhc_charges_tensors = torch.zeros(
            (len(self.unique_alleles), MHC_POCKET_COUNT), dtype=torch.float32
        )

        for i, allele in enumerate(self.unique_alleles):
            if allele in self.pseudo_seqs:
                seq = self.pseudo_seqs[allele]
                if len(seq) != MHC_POCKET_COUNT:
                    # A silently truncated or padded pseudo-sequence is a wrong
                    # feature vector with no error, not a smaller correct one -
                    # see docs/claims_register.md D30. Padding invents residues
                    # and truncation discards them; either way the four
                    # physicochemical features derived from the affected
                    # positions are fabricated. Fail loud instead of masking a
                    # data bug as a shorter allele.
                    raise ValueError(
                        f"pseudo_seqs[{allele!r}] is {len(seq)} chars, expected "
                        f"{MHC_POCKET_COUNT}. Check src/verify/mhc_pseudo_sequences.json."
                    )
            else:
                logger.warning(
                    f"No pseudo-sequence for allele {allele!r}; using an all-alanine "
                    f"placeholder ({MHC_POCKET_COUNT} residues). Pocket features for "
                    "this allele carry no real structural signal."
                )
                seq = "A" * MHC_POCKET_COUNT
            for j, aa in enumerate(seq):
                self.mhc_node_tensors[i, j, 0] = KD_HYDRO.get(aa, 0.0)
                self.mhc_node_tensors[i, j, 1] = float(AROMATIC.get(aa, 0))
                self.mhc_node_tensors[i, j, 2] = VDW_VOL.get(aa, 110.0)
                self.mhc_node_tensors[i, j, 3] = float(CHARGE.get(aa, 0))
                self.mhc_node_tensors[i, j, 4] = 1.0  # MHC indicator
                self.mhc_charges_tensors[i, j] = float(CHARGE.get(aa, 0))

    def len(self) -> int:
        return len(self.peptides)

    def get(self, idx: int) -> Data:
        peptide = self.peptides[idx]
        allele = self.alleles[idx]
        label = self.labels[idx]

        # 1. Generate coordinates
        pep_coords, mhc_coords = generate_canonical_groove_coords(peptide, allele)
        pos = np.concatenate([pep_coords, mhc_coords], axis=0)
        pos_t = torch.tensor(pos, dtype=torch.float32)

        n_pep = len(peptide)

        # 2. Extract node features
        pep_node_feats = torch.zeros(n_pep, 5, dtype=torch.float32)
        pep_charges = torch.zeros(n_pep, dtype=torch.float32)

        for i, aa in enumerate(peptide):
            pep_node_feats[i, 0] = KD_HYDRO.get(aa, 0.0)
            pep_node_feats[i, 1] = float(AROMATIC.get(aa, 0))
            pep_node_feats[i, 2] = VDW_VOL.get(aa, 0.0)
            pep_node_feats[i, 3] = float(CHARGE.get(aa, 0))
            pep_node_feats[i, 4] = 0.0  # Peptide indicator
            pep_charges[i] = float(CHARGE.get(aa, 0))

        # Vectorized assembly
        allele_idx = self.allele_to_idx[allele]
        mhc_feats = self.mhc_node_tensors[allele_idx]
        mhc_charges = self.mhc_charges_tensors[allele_idx]

        node_feats = torch.cat([pep_node_feats, mhc_feats], dim=0)
        charges = torch.cat([pep_charges, mhc_charges], dim=0)

        # 3. Generate edges and edge attributes
        edge_index, edge_attr = generate_edge_features(pos_t, charges, d_max=8.0)

        y_t = torch.tensor([label], dtype=torch.float32)

        if HAS_PYG:
            return Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr, pos=pos_t, y=y_t)
        else:

            class SimpleData:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

            return SimpleData(
                x=node_feats, edge_index=edge_index, edge_attr=edge_attr, pos=pos_t, y=y_t
            )


class StructuralGNN(nn.Module):
    """
    Multi-viral structural GNN architecture using GINEConv message passing.
    """

    def __init__(
        self, node_dim: int = 5, edge_dim: int = 3, hidden_dim: int = 64, dropout: float = 0.3
    ):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required to initialize StructuralGNN.")
        # Node and edge projection layers
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # Message passing layers (GINEConv supports edge attributes matching node hidden dimensions)
        self.conv1_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINEConv(self.conv1_nn, train_eps=True)

        self.conv2_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINEConv(self.conv2_nn, train_eps=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Initial projection
        h_node = F.relu(self.node_proj(x))
        h_edge = F.relu(self.edge_proj(edge_attr))

        # Conv layer 1
        h_node = self.conv1(h_node, edge_index, edge_attr=h_edge)
        h_node = F.relu(h_node)
        h_node = self.dropout(h_node)

        # Conv layer 2
        h_node = self.conv2(h_node, edge_index, edge_attr=h_edge)
        h_node = F.relu(h_node)
        h_node = self.dropout(h_node)

        # Global readout pooling (mean node state representation)
        pooled = global_mean_pool(h_node, batch)
        return self.fc(pooled).squeeze(-1)


def train_structural_gnn(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_name: str = "cuda",
) -> StructuralGNN:
    """
    Optimized training loop with mixed-precision (AMP) and gradient clipping.
    """
    if not HAS_PYG:
        raise ImportError("PyTorch Geometric is not installed in the current environment.")

    device = torch.device(
        device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu"
    )
    logger.info(f"Training Structural GNN on device: {device}")

    # Load dataset
    train_dataset = StructuralPeptideMHCDataset(df_train)
    val_dataset = StructuralPeptideMHCDataset(df_val)

    # Fast DataLoader configuration per debate protocol
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=min(os.cpu_count() or 4, 4),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=min(os.cpu_count() or 4, 4),
        persistent_workers=True,
    )

    model = StructuralGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # AMP Scaler for mixed precision
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Autocast context for mixed precision
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(batch)
                loss = criterion(logits, batch.y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Gradient clipping to stabilize message passing
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * batch.num_graphs

        logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / len(train_dataset):.4f}")

    model.eval()
    return model
