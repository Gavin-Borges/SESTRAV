import pytest
import pandas as pd
import torch
import numpy as np
from src.verify.structural_gnn import (
    generate_canonical_groove_coords,
    generate_edge_features,
    StructuralPeptideMHCDataset,
    HAS_PYG,
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
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],  # Node 2 is > 8 Å from Node 0
        ],
        dtype=torch.float32,
    )

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
    df = pd.DataFrame(
        {
            "peptide": ["GLFYTRTGL", "AAYSDQWAL"],
            "allele": ["HLA-A*02:01", "HLA-A*24:02"],
            "label": [1, 0],
        }
    )

    dataset = StructuralPeptideMHCDataset(df)
    assert len(dataset) == 2

    data = dataset.get(0)
    # Peptide length 9 + MHC 34 = 43 nodes
    assert data.x.shape == (43, 5)
    assert data.pos.shape == (43, 3)
    assert data.y.item() == 1.0


def _minimal_df():
    return pd.DataFrame(
        {
            "peptide": ["GLFYTRTGL", "AAYSDQWAL"],
            "allele": ["HLA-A*02:01", "HLA-A*24:02"],
            "label": [1, 0],
        }
    )


@pytest.mark.skipif(
    HAS_PYG,
    reason=(
        "StructuralPeptideMHCDataset inherits from Dataset when PyG is installed; "
        "PyG's Dataset.__setattr__ requires super().__init__(), so the HAS_PYG=False "
        "branch (lines 99->101) can only be exercised in a PyG-absent environment."
    ),
)
def test_dataset_haspyg_false_init(monkeypatch):
    """Covers branch 99->101: when HAS_PYG is False the super().__init__ call
    inside StructuralPeptideMHCDataset.__init__ is skipped (PyG-absent env only)."""
    import src.verify.structural_gnn as sgnn

    monkeypatch.setattr(sgnn, "HAS_PYG", False)
    ds = sgnn.StructuralPeptideMHCDataset(_minimal_df())
    assert len(ds.peptides) == 2


@pytest.mark.skipif(
    HAS_PYG, reason=("Same Dataset.__setattr__ constraint as test_dataset_haspyg_false_init.")
)
def test_dataset_haspyg_false_get(monkeypatch):
    """Covers lines 175-178: when HAS_PYG is False, get() returns a SimpleData
    object instead of a torch_geometric Data object (PyG-absent env only)."""
    import src.verify.structural_gnn as sgnn

    monkeypatch.setattr(sgnn, "HAS_PYG", False)
    ds = sgnn.StructuralPeptideMHCDataset(_minimal_df())
    item = ds.get(0)
    assert hasattr(item, "x") and hasattr(item, "edge_index")
    assert type(item).__name__ == "SimpleData"


def test_structural_gnn_init_no_pyg_raises(monkeypatch):
    """Covers line 187: StructuralGNN.__init__ raises ImportError if HAS_PYG=False."""
    import src.verify.structural_gnn as sgnn

    monkeypatch.setattr(sgnn, "HAS_PYG", False)
    with pytest.raises(ImportError, match="torch_geometric"):
        sgnn.StructuralGNN()


def test_train_structural_gnn_no_pyg_raises(monkeypatch):
    """Covers line 249: train_structural_gnn raises ImportError if HAS_PYG=False."""
    import src.verify.structural_gnn as sgnn

    monkeypatch.setattr(sgnn, "HAS_PYG", False)
    with pytest.raises(ImportError, match="PyTorch Geometric"):
        sgnn.train_structural_gnn(_minimal_df(), _minimal_df())


def test_dataset_no_pseudo_seqs_file(monkeypatch, tmp_path):
    """Covers line 116: pseudo_seqs defaults to {} when the JSON file is absent."""
    import src.verify.structural_gnn as sgnn

    # Change CWD to tmp_path so the relative path 'src/verify/mhc_pseudo_sequences.json'
    # does not resolve to the actual file.
    monkeypatch.chdir(tmp_path)
    ds = sgnn.StructuralPeptideMHCDataset(_minimal_df())
    assert ds.pseudo_seqs == {}


@pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
def test_gnn_model_and_training():
    from src.verify.structural_gnn import StructuralGNN, train_structural_gnn

    # Test model initialization
    model = StructuralGNN()
    assert isinstance(model, torch.nn.Module)

    # Test model forward pass
    df = pd.DataFrame(
        {
            "peptide": ["GLFYTRTGL", "AAYSDQWAL"],
            "allele": ["HLA-A*02:01", "HLA-A*24:02"],
            "label": [1, 0],
        }
    )
    dataset = StructuralPeptideMHCDataset(df)

    from torch_geometric.loader import DataLoader

    loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))

    logits = model(batch)
    assert logits.shape == (2,)

    # Test mock training loop
    trained_model = train_structural_gnn(df, df, epochs=1, batch_size=2)
    assert isinstance(trained_model, torch.nn.Module)
