import torch
from typing import Dict

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

class GraphBuilder:
    """Builds chain graph representations from peptide sequences."""
    
    @staticmethod
    def build_chain_adj(max_len: int = 11) -> torch.Tensor:
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
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        
        # D^(-1/2) A D^(-1/2)
        norm_A = torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)
        return norm_A

    @staticmethod
    def sequence_to_node_features(seq: str, max_len: int = 11) -> torch.Tensor:
        """Convert peptide to node feature matrix (max_len, num_features)."""
        # Features per node: one-hot encoded AA (size 20)
        features = torch.zeros((max_len, 20))
        for i, aa in enumerate(seq[:max_len]):
            if aa in AA_TO_IDX:
                features[i, AA_TO_IDX[aa]] = 1.0
        return features

