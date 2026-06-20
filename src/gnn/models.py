import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (batch, max_len, in_features)
        # adj: (max_len, max_len)
        support = torch.matmul(x, self.weight) # (batch, max_len, out_features)
        output = torch.matmul(adj, support)    # (batch, max_len, out_features)
        return output + self.bias

class GraphEncoder(nn.Module):
    """Encodes the peptide graph using GCN layers."""
    def __init__(self, in_features: int = 20, hidden_dim1: int = 32, hidden_dim2: int = 64):
        super().__init__()
        self.gcn1 = GCNLayer(in_features, hidden_dim1)
        self.gcn2 = GCNLayer(hidden_dim1, hidden_dim2)
        
    def forward(self, node_x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.gcn1(node_x, adj))
        h = torch.relu(self.gcn2(h, adj)) # (batch, max_len, 64)
        # Global mean pooling over nodes
        return torch.mean(h, dim=1) # (batch, 64)

class GraphPredictor(nn.Module):
    """
    Combines the GraphEncoder with an MLP head for predicting immunogenicity.
    """
    def __init__(self, num_continuous_features: int, dropout_rate: float = 0.3):
        super(GraphPredictor, self).__init__()

        self.encoder = GraphEncoder()

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

    def forward(self, node_x: torch.Tensor, feat_x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        gnn_out = self.encoder(node_x, adj)
        physico_out = self.physico_block(feat_x)

        # Concatenate and classify
        fused = torch.cat((gnn_out, physico_out), dim=1)
        out = self.fusion_block(fused)
        return out.squeeze(1)


# ---------------------------------------------------------------------------
# GNN v2.1: GINEConv + ESM-2 node embeddings
# ---------------------------------------------------------------------------

class GraphEncoderV2(nn.Module):
    """GINEConv-based encoder consuming pre-computed ESM-2 per-residue embeddings.

    Expects PyG-format inputs: flat (total_nodes, node_dim) node tensor,
    edge_index with batch-offset node indices, and a batch vector.
    """
    def __init__(self, node_dim: int = 320, hidden_dim: int = 256,
                 out_dim: int = 128, edge_dim: int = 3):
        super().__init__()
        from torch_geometric.nn import GINEConv, global_mean_pool
        self._global_mean_pool = global_mean_pool

        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            edge_dim=edge_dim,
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            ),
            edge_dim=edge_dim,
        )
        for conv in (self.conv1, self.conv2):
            for m in conv.nn.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = F.relu(self.conv2(h, edge_index, edge_attr))
        return self._global_mean_pool(h, batch)  # (num_graphs, out_dim)


class GraphPredictorV2(nn.Module):
    """GINEConv + ESM-2 immunogenicity predictor. Accepts a PyG batched Data object.

    data.x: (total_nodes, node_dim) — pre-computed ESM-2 residue embeddings
    data.edge_index: (2, total_edges) — chain graph with batch-offset indices
    data.edge_attr: (total_edges, 3) — one-hot edge type features
    data.physico: (batch_size, num_continuous_features) — SESTRAV physicochemical features
    data.y: (batch_size,) — binary immunogenicity labels
    """
    def __init__(self, num_continuous_features: int, node_dim: int = 320,
                 dropout_rate: float = 0.3):
        super().__init__()
        self.encoder = GraphEncoderV2(node_dim=node_dim)

        self.physico_block = nn.Sequential(
            nn.Linear(num_continuous_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        for m in self.physico_block.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.fusion_block = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )
        for m in self.fusion_block.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, data) -> torch.Tensor:
        gnn_out = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        physico_out = self.physico_block(data.physico)
        fused = torch.cat((gnn_out, physico_out), dim=1)
        return self.fusion_block(fused).squeeze(1)

