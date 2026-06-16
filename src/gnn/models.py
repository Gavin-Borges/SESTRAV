import torch
import torch.nn as nn

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

