import pytest
import pandas as pd
import torch
import numpy as np
from src.verify.structural_gnn import (
    generate_canonical_groove_coords,
    generate_edge_features,
    StructuralPeptideMHCDataset,
    HAS_PYG
)

def test_generate_canonical_groove_coords():
    peptide = "GLFYTRTGL"
    allele = "HLA-A*02:01"
    
    pep_coords, mhc_coords = generate_canonical_groove_coords(peptide, allele)
    assert isinstance(pep_coords, np.ndarray)
    assert isinstance(mhc_coords, np.ndarray)
    assert pep_coords.shape == (9, 3)
    assert mhc_coords.shape == (34, 3)
    
    # Check X-axis translation for peptide
    assert pep_coords[1, 0] == 3.2
    assert pep_coords[0, 1] == 0.5  # side-chain oscillation
    assert pep_coords[1, 1] == -0.5

def test_generate_edge_features():
    # Simple coordinates: 3 nodes
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [10.0, 0.0, 0.0]  # Node 2 is > 8 Å from Node 0
    ], dtype=torch.float32)
    
    charges = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float32)
    edge_index, edge_attr = generate_edge_features(pos, charges, d_max=8.0)
    
    assert edge_index.shape[0] == 2
    # Node 0 <-> Node 1 are within 8 Å (bidirectional edge)
    # Node 2 is > 8 Å from Node 0 and Node 1, but Node 1 <-> Node 2 distance is 7 Å (within 8 Å!)
    # Node 0 and Node 1 dist = 3.0 Å
    # Node 1 and Node 2 dist = 7.0 Å
    # So we expect edges:
    # 0 -> 1, 1 -> 0, 1 -> 2, 2 -> 1
    # Total edges = 4
    assert edge_index.shape[1] == 4
    
    # Check edge attributes: [distance, Coulomb, LJ]
    # For edge 0 -> 1 (distance = 3.0 Å, charges = 1.0 and -1.0)
    # Coulomb potential = -1.0 / 3.0 = -0.333
    assert edge_attr.shape[1] == 3

def test_dataset_generation():
    df = pd.DataFrame({
        "peptide": ["GLFYTRTGL", "AAYSDQWAL"],
        "allele": ["HLA-A*02:01", "HLA-A*24:02"],
        "label": [1, 0]
    })
    
    dataset = StructuralPeptideMHCDataset(df)
    assert len(dataset) == 2
    
    data = dataset.get(0)
    # Peptide length 9 + MHC 34 = 43 nodes
    assert data.x.shape == (43, 5)
    assert data.pos.shape == (43, 3)
    assert data.y.item() == 1.0

@pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
def test_gnn_model_and_training():
    from src.verify.structural_gnn import StructuralGNN, train_structural_gnn
    
    # Test model initialization
    model = StructuralGNN()
    assert isinstance(model, torch.nn.Module)
    
    # Test model forward pass
    df = pd.DataFrame({
        "peptide": ["GLFYTRTGL", "AAYSDQWAL"],
        "allele": ["HLA-A*02:01", "HLA-A*24:02"],
        "label": [1, 0]
    })
    dataset = StructuralPeptideMHCDataset(df)
    
    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))
    
    logits = model(batch)
    assert logits.shape == (2,)
    
    # Test mock training loop
    trained_model = train_structural_gnn(df, df, epochs=1, batch_size=2)
    assert isinstance(trained_model, torch.nn.Module)
