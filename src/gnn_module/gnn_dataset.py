import torch
from torch_geometric.data import Data, Dataset

class PeptideGraphDataset(Dataset):
    def __init__(self, peptide_data_list):
        """
        Custom dataset for SESTRAV 2.0 GNN peptide representations.
        
        Args:
            peptide_data_list: List of dictionaries containing 'sequence' and 'label'.
        """
        super().__init__()
        self.data_list = self._process_peptides(peptide_data_list)

    def _process_peptides(self, peptide_list):
        graphs = []
        for item in peptide_list:
            seq = item['sequence']
            label = item['label']
            
            # Nodes: sequence length. Features per node: e.g., 6.
            # In production, these should be mapped via Kyte-Doolittle and other tables.
            x = torch.rand((len(seq), 6), dtype=torch.float)
            
            # Edges: Sequential backbone connections
            edges = []
            for i in range(len(seq) - 1):
                edges.append([i, i+1])
                edges.append([i+1, i])
            
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                
            y = torch.tensor([[label]], dtype=torch.float)
            graphs.append(Data(x=x, edge_index=edge_index, y=y))
        return graphs

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
