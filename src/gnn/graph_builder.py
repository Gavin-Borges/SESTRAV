import torch
from typing import Dict

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

class GraphBuilder:
    """Builds chain graph representations from peptide sequences."""
    
    @staticmethod
    def build_chain_adj(max_len: int = 11) -> torch.Tensor:
        """
        Builds a normalized adjacency matrix for a 1D sequence chain graph.
        """
        adj = torch.zeros((max_len, max_len), dtype=torch.float32)
        for i in range(max_len):
            adj[i, i] = 1.0  # Self-loop
            if i > 0:
                adj[i, i-1] = 1.0
            if i < max_len - 1:
                adj[i, i+1] = 1.0

        # Degree normalization (D^-0.5 * A * D^-0.5)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        return norm_adj

    @staticmethod
    def build_spatial_adj(peptide: str, cache_dir: str, max_len: int = 11, distance_threshold: float = 8.0) -> torch.Tensor:
        """
        Builds a normalized adjacency matrix using pre-computed 3D spatial distances
        (e.g. from AlphaFold PDBs). 
        Edges are created if distance <= threshold, and weighted as 1/distance.
        
        Args:
            peptide: The sequence to look up in the structural cache.
            cache_dir: Directory containing '{peptide}_dist.pt' distance matrices.
            max_len: Padding length.
            distance_threshold: Angstrom cutoff for edge creation.
        """
        import os
        cache_path = os.path.join(cache_dir, f"{peptide}_dist.pt")
        
        # Fallback to chain graph if structure is not cached
        if not os.path.exists(cache_path):
            return GraphBuilder.build_chain_adj(max_len)
            
        # Load precomputed pairwise distance matrix (L x L)
        dist_matrix = torch.load(cache_path)
        L = dist_matrix.size(0)
        
        adj = torch.zeros((max_len, max_len), dtype=torch.float32)
        
        # Create weighted edges
        for i in range(min(L, max_len)):
            for j in range(min(L, max_len)):
                if i == j:
                    adj[i, j] = 1.0  # Self-loop
                else:
                    d = dist_matrix[i, j].item()
                    if d <= distance_threshold and d > 0:
                        adj[i, j] = 1.0 / d
                        
        # Degree normalization
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        return norm_adj

    @staticmethod
    def sequence_to_node_features(seq: str, max_len: int = 11) -> torch.Tensor:
        """Convert peptide to node feature matrix (max_len, num_features)."""
        # Features per node: one-hot encoded AA (size 20)
        features = torch.zeros((max_len, 20))
        for i, aa in enumerate(seq[:max_len]):
            if aa in AA_TO_IDX:
                features[i, AA_TO_IDX[aa]] = 1.0
        return features

