"""Tests for GraphPeptideDataset and set_seed in src/train_gnn.py.

GraphPeptideDataset builds torch.utils.data.Dataset items from peptide sequences
and a pre-computed physicochemical feature matrix. Tests exercise:
  - construction and __len__
  - __getitem__ tensor shapes (node features, adjacency, physico)
  - label passthrough and label-less mode
  - padding behaviour for sequences shorter than max_len
  - use_spatial fallback when cache file is absent
  - set_seed determinism
"""

import numpy as np
import pandas as pd
import torch
import pytest

from src.train_gnn import GraphPeptideDataset, set_seed


@pytest.fixture(autouse=True)
def _isolated_global_flags():
    """Restore the process-global flags set_seed writes, so this file leaks nothing.

    set_seed sets deterministic-algorithms mode and the two cudnn flags on the
    whole interpreter, deliberately: scoping them to a run is the trainers' job,
    not set_seed's. A test module that calls set_seed directly bypasses those
    trainers, so without this fixture every test that runs after this file would
    silently execute under deterministic algorithms. The same fixture guards
    tests/test_train_gnn_seed.py for the same reason.
    """
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(
    peptides,
    n_features=21,
    labels=None,
    max_len=11,
    cache_dir=None,
    use_spatial=False,
    alleles="HLA-A*02:01",
):
    df = pd.DataFrame({"peptide": peptides})
    if alleles is not None:
        df["hla_allele"] = alleles
    n = len(peptides)
    feat_matrix = pd.DataFrame(
        np.random.default_rng(0).standard_normal((n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    if labels is None:
        labels = np.zeros(n)
    return GraphPeptideDataset(
        df, feat_matrix, labels, max_len=max_len, cache_dir=cache_dir, use_spatial=use_spatial
    )


# ---------------------------------------------------------------------------
# Construction + __len__
# ---------------------------------------------------------------------------


def test_len_single():
    ds = _make_dataset(["CLGGLLTMV"])
    assert len(ds) == 1


def test_len_batch():
    ds = _make_dataset(["CLGGLLTMV", "RAKFKQLL", "TIHDIILECV"])
    assert len(ds) == 3


# ---------------------------------------------------------------------------
# __getitem__ with labels
# ---------------------------------------------------------------------------


def test_getitem_returns_four_tensors():
    ds = _make_dataset(["CLGGLLTMV"], labels=np.array([1.0]))
    item = ds[0]
    assert len(item) == 4


def test_node_features_shape():
    ds = _make_dataset(["CLGGLLTMV"], max_len=11)
    node_feats, _, _, _ = ds[0]
    assert node_feats.shape == (11, 20)


def test_adj_shape():
    ds = _make_dataset(["CLGGLLTMV"], max_len=11)
    _, _, adj, _ = ds[0]
    assert adj.shape == (11, 11)


def test_physico_shape():
    n_features = 21
    ds = _make_dataset(["CLGGLLTMV"], n_features=n_features, max_len=11)
    _, physico, _, _ = ds[0]
    assert physico.shape == (n_features,)


def test_label_dtype_float32():
    ds = _make_dataset(["CLGGLLTMV"], labels=np.array([1.0]))
    _, _, _, label = ds[0]
    assert label.dtype == torch.float32


def test_label_value_preserved():
    ds = _make_dataset(["CLGGLLTMV", "RAKFKQLL"], labels=np.array([1.0, 0.0]))
    assert ds[0][3].item() == pytest.approx(1.0)
    assert ds[1][3].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# __getitem__ without labels
# ---------------------------------------------------------------------------


def test_getitem_no_labels_returns_three_tensors():
    df = pd.DataFrame({"peptide": ["CLGGLLTMV"]})
    feat = pd.DataFrame(np.zeros((1, 21)), columns=[f"f{i}" for i in range(21)])
    ds = GraphPeptideDataset(df, feat, labels=None)
    item = ds[0]
    assert len(item) == 3


# ---------------------------------------------------------------------------
# Sequence padding (peptide shorter than max_len)
# ---------------------------------------------------------------------------


def test_short_sequence_zero_pads():
    ds = _make_dataset(["ACDE"], max_len=11)  # 4 AAs, padded to 11
    node_feats, _, _, _ = ds[0]
    assert node_feats.shape == (11, 20)
    # Padded positions (indices 4-10) should be all-zero
    assert node_feats[4:].sum().item() == pytest.approx(0.0)


def test_9mer_node_features_populated():
    ds = _make_dataset(["CLGGLLTMV"], max_len=11)  # 9 AAs
    node_feats, _, _, _ = ds[0]
    # Positions 0-8 should have exactly one hot-1 per row
    assert node_feats[:9].sum(dim=1).min().item() == pytest.approx(1.0)
    # Padded positions 9-10 should be zero
    assert node_feats[9:].sum().item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# use_spatial fallback: missing cache → chain adj
# ---------------------------------------------------------------------------


def test_use_spatial_missing_cache_falls_back(tmp_path):
    ds = _make_dataset(["CLGGLLTMV"], max_len=11, cache_dir=str(tmp_path), use_spatial=True)
    node_feats, physico, adj_spatial, label = ds[0]

    ds_chain = _make_dataset(["CLGGLLTMV"], max_len=11, use_spatial=False)
    _, _, adj_chain, _ = ds_chain[0]

    torch.testing.assert_close(adj_spatial, adj_chain)


def test_use_spatial_without_allele_column_raises(tmp_path):
    # The structural cache is keyed by (peptide, allele). A corpus with no allele
    # column cannot address it, and the old peptide-only lookup missed every
    # entry silently instead of saying so.
    with pytest.raises(ValueError, match="hla_allele"):
        _make_dataset(
            ["CLGGLLTMV"],
            max_len=11,
            cache_dir=str(tmp_path),
            use_spatial=True,
            alleles=None,
        )


def test_use_spatial_uses_each_row_own_allele(tmp_path):
    # Two rows, two different alleles, a cached matrix for the SECOND one only.
    # If the dataset broadcast row 0's allele to every row, row 1 would miss and
    # fall back, so this is what pins the per-row lookup.
    from src.gnn.graph_builder import GraphBuilder, structural_cache_filename

    dist = torch.full((11, 11), 2.0, dtype=torch.float32)
    torch.save(  # nosec B614 - test fixture, trusted tensor
        dist, tmp_path / structural_cache_filename("CCCCCCCCC", "HLA-B*07:02")
    )
    ds = _make_dataset(
        ["AAAAAAAAA", "CCCCCCCCC"],
        max_len=11,
        cache_dir=str(tmp_path),
        use_spatial=True,
        alleles=["HLA-A*02:01", "HLA-B*07:02"],
    )
    chain = GraphBuilder.build_chain_adj(max_len=11)
    assert torch.allclose(ds[0][2], chain), "row 0 has no cached matrix and must fall back"
    assert not torch.allclose(ds[1][2], chain), "row 1 has a cached matrix and must use it"


def test_use_spatial_reads_the_allele_keyed_cache(tmp_path):
    from src.gnn.graph_builder import GraphBuilder, structural_cache_filename

    dist = torch.full((11, 11), 2.0, dtype=torch.float32)
    torch.save(  # nosec B614 - test fixture, trusted tensor
        dist, tmp_path / structural_cache_filename("CLGGLLTMV", "HLA-A*02:01")
    )
    ds = _make_dataset(["CLGGLLTMV"], max_len=11, cache_dir=str(tmp_path), use_spatial=True)
    _, _, adj_spatial, _ = ds[0]
    assert not torch.allclose(adj_spatial, GraphBuilder.build_chain_adj(max_len=11))


# ---------------------------------------------------------------------------
# Adjacency matrix properties
# ---------------------------------------------------------------------------


def test_adj_symmetric():
    ds = _make_dataset(["CLGGLLTMV"], max_len=11)
    _, _, adj, _ = ds[0]
    torch.testing.assert_close(adj, adj.T)


def test_adj_nonnegative():
    ds = _make_dataset(["CLGGLLTMV"], max_len=11)
    _, _, adj, _ = ds[0]
    assert (adj >= 0).all()


# ---------------------------------------------------------------------------
# set_seed determinism
# ---------------------------------------------------------------------------


def test_set_seed_torch_deterministic():
    set_seed(123)
    t1 = torch.randn(10)
    set_seed(123)
    t2 = torch.randn(10)
    torch.testing.assert_close(t1, t2)


def test_set_seed_numpy_deterministic():
    set_seed(99)
    a1 = np.random.rand(5)
    set_seed(99)
    a2 = np.random.rand(5)
    np.testing.assert_array_equal(a1, a2)
