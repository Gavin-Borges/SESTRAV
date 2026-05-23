import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class PeptideGNN(torch.nn.Module):
    def __init__(self, node_feature_dim=6, hidden_dim=32, num_classes=1):
        """
        SESTRAV 2.0 Graph Neural Network for Peptide Immunogenicity.
        
        Args:
            node_feature_dim: Number of features per residue node (e.g., VdW volume, charge, hydrophobicity).
            hidden_dim: Dimension of hidden representations.
            num_classes: Output dimension (1 for binary classification probability).
        """
        super(PeptideGNN, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Dense classification head
        self.fc1 = torch.nn.Linear(hidden_dim, 16)
        self.fc2 = torch.nn.Linear(16, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph Convolutions over peptide residue nodes
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling (e.g. mean over all nodes in a peptide graph)
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        
        return torch.sigmoid(x)
