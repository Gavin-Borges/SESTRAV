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
        "branch can only be exercised in a PyG-absent environment."
    ),
)
def test_dataset_haspyg_false_init(monkeypatch):
    """Covers the HAS_PYG=False branch: the super().__init__ call
    inside StructuralPeptideMHCDataset.__init__ is skipped (PyG-absent env only)."""
    import src.verify.structural_gnn as sgnn

    monkeypatch.setattr(sgnn, "HAS_PYG", False)
    ds = sgnn.StructuralPeptideMHCDataset(_minimal_df())
    assert len(ds.peptides) == 2


@pytest.mark.skipif(
    HAS_PYG, reason=("Same Dataset.__setattr__ constraint as test_dataset_haspyg_false_init.")
)
def test_dataset_haspyg_false_get(monkeypatch):
    """Covers get()'s HAS_PYG=False branch: it returns a SimpleData
    object instead of a torch_geometric Data object (PyG-absent env only)."""
    import src.verify.structural_gnn as sgnn

    monkeypatch.setattr(sgnn, "HAS_PYG", False)
    ds = sgnn.StructuralPeptideMHCDataset(_minimal_df())
    item = ds.get(0)
    assert hasattr(item, "x") and hasattr(item, "edge_index")
    assert type(item).__name__ == "SimpleData"


def test_structural_gnn_init_no_pyg_raises(monkeypatch):
    """Covers StructuralGNN.__init__'s guard: it raises ImportError if HAS_PYG=False."""
    import src.verify.structural_gnn as sgnn

    monkeypatch.setattr(sgnn, "HAS_PYG", False)
    with pytest.raises(ImportError, match="torch_geometric"):
        sgnn.StructuralGNN()


def test_train_structural_gnn_no_pyg_raises(monkeypatch):
    """Covers train_structural_gnn's guard: it raises ImportError if HAS_PYG=False."""
    import src.verify.structural_gnn as sgnn

    monkeypatch.setattr(sgnn, "HAS_PYG", False)
    with pytest.raises(ImportError, match="PyTorch Geometric"):
        sgnn.train_structural_gnn(_minimal_df(), _minimal_df())


def test_dataset_no_pseudo_seqs_file(monkeypatch, tmp_path):
    """Covers the missing-file branch: pseudo_seqs defaults to {} when the JSON is absent."""
    import src.verify.structural_gnn as sgnn

    # Change CWD to tmp_path so the relative path 'src/verify/mhc_pseudo_sequences.json'
    # does not resolve to the actual file.
    monkeypatch.chdir(tmp_path)
    ds = sgnn.StructuralPeptideMHCDataset(_minimal_df())
    assert ds.pseudo_seqs == {}


def _write_pseudo_seqs(tmp_path, table):
    """Plant a pseudo-sequence JSON at the relative path the dataset reads."""
    import json

    target = tmp_path / "src" / "verify"
    target.mkdir(parents=True)
    (target / "mhc_pseudo_sequences.json").write_text(json.dumps(table), encoding="utf-8")


def test_dataset_raises_on_wrong_length_pseudo_sequence(monkeypatch, tmp_path):
    """A malformed entry must fail loud, not be silently padded (D30).

    Before this guard, a 33-character entry was silently `.ljust(34, "A")`-ed,
    fabricating a 34th residue and reporting nothing. `.ljust()` appends, so the
    preceding 33 positions survive intact - the damage is an invented residue
    yielding four invented features, not a shifted frame.
    """
    import src.verify.structural_gnn as sgnn

    _write_pseudo_seqs(
        tmp_path,
        {
            "HLA-A*02:01": "A" * (sgnn.MHC_POCKET_COUNT - 1),  # one short
            "HLA-A*24:02": "A" * sgnn.MHC_POCKET_COUNT,
        },
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=r"is 33 chars, expected 34"):
        sgnn.StructuralPeptideMHCDataset(_minimal_df())


def test_dataset_warns_and_placeholders_for_an_absent_allele(monkeypatch, tmp_path, caplog):
    """Absent is a coverage gap, not a data bug: warn + explicit placeholder.

    This is the deliberate other half of the guard above. It must stay distinct
    from the raise, or a missing allele would become a hard failure and callers
    would be tempted to reintroduce silent padding to get past it.
    """
    import logging

    import src.verify.structural_gnn as sgnn

    _write_pseudo_seqs(tmp_path, {"HLA-A*02:01": "A" * sgnn.MHC_POCKET_COUNT})
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.WARNING):
        ds = sgnn.StructuralPeptideMHCDataset(_minimal_df())

    assert "HLA-A*24:02" in caplog.text
    assert "no real structural signal" in caplog.text
    # The placeholder is all-alanine, so its charge row is uniformly zero.
    idx = ds.allele_to_idx["HLA-A*24:02"]
    assert float(ds.mhc_charges_tensors[idx].abs().sum()) == 0.0


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
