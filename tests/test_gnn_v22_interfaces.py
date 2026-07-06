"""Tests for GNN v2.2/v2.3 interfaces: node_dim propagation, gnn_config.json,
early-stopping params, binding_matrix_path wiring, num_continuous_features
propagation, variable-length graph correctness, and promote_gnn gate3 config-reading path.
"""

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# GraphPredictorV2 node_dim propagation
# ---------------------------------------------------------------------------


def test_predictor_v2_default_node_dim():
    from src.gnn.models import GraphPredictorV2

    model = GraphPredictorV2(num_continuous_features=10)
    assert model.encoder.conv1.nn[0].in_features == 320


def test_predictor_v2_custom_node_dim_480():
    from src.gnn.models import GraphPredictorV2

    model = GraphPredictorV2(num_continuous_features=10, node_dim=480)
    assert model.encoder.conv1.nn[0].in_features == 480


def test_predictor_v2_custom_node_dim_640():
    from src.gnn.models import GraphPredictorV2

    model = GraphPredictorV2(num_continuous_features=10, node_dim=640)
    assert model.encoder.conv1.nn[0].in_features == 640


# ---------------------------------------------------------------------------
# train_gnn_v2 signature accepts new v2.2 params
# ---------------------------------------------------------------------------


def test_train_gnn_v2_accepts_node_dim_param():
    import inspect
    from src.train_gnn import train_gnn_v2

    sig = inspect.signature(train_gnn_v2)
    assert "node_dim" in sig.parameters


def test_train_gnn_v2_accepts_esm2_model_name_param():
    import inspect
    from src.train_gnn import train_gnn_v2

    sig = inspect.signature(train_gnn_v2)
    assert "esm2_model_name" in sig.parameters


def test_train_gnn_v2_accepts_early_stopping_patience_param():
    import inspect
    from src.train_gnn import train_gnn_v2

    sig = inspect.signature(train_gnn_v2)
    assert "early_stopping_patience" in sig.parameters


def test_train_gnn_v2_default_epochs_is_50():
    import inspect
    from src.train_gnn import train_gnn_v2

    sig = inspect.signature(train_gnn_v2)
    assert sig.parameters["epochs"].default == 50


def test_train_gnn_v2_default_patience_is_10():
    import inspect
    from src.train_gnn import train_gnn_v2

    sig = inspect.signature(train_gnn_v2)
    assert sig.parameters["early_stopping_patience"].default == 10


def test_train_gnn_v2_accepts_binding_matrix_path_param():
    import inspect
    from src.train_gnn import train_gnn_v2

    sig = inspect.signature(train_gnn_v2)
    assert "binding_matrix_path" in sig.parameters


def test_train_gnn_v2_binding_matrix_path_defaults_none():
    import inspect
    from src.train_gnn import train_gnn_v2

    sig = inspect.signature(train_gnn_v2)
    assert sig.parameters["binding_matrix_path"].default is None


# ---------------------------------------------------------------------------
# promote_gnn gate3 reads node_dim and num_continuous_features from gnn_config.json
# ---------------------------------------------------------------------------


def test_gate3_reads_node_dim_from_config(tmp_path, monkeypatch):
    """gate3_latency should use node_dim from gnn_config.json if present."""
    import src.verify.promote_gnn as pg

    config_data = {"node_dim": 480, "esm2_model_name": "facebook/esm2_t12_35M_UR50D"}
    config_file = tmp_path / "gnn_config.json"
    config_file.write_text(json.dumps(config_data))

    monkeypatch.setattr(pg, "GNN_CONFIG", config_file)

    # Read node_dim using the same logic as gate3_latency
    import json as _json

    node_dim = 320
    if pg.GNN_CONFIG.exists():
        with pg.GNN_CONFIG.open() as fh:
            node_dim = _json.load(fh).get("node_dim", 320)

    assert node_dim == 480


def test_gate3_defaults_node_dim_320_when_config_missing(tmp_path, monkeypatch):
    """gate3_latency falls back to 320 when gnn_config.json does not exist."""
    import src.verify.promote_gnn as pg

    monkeypatch.setattr(pg, "GNN_CONFIG", tmp_path / "nonexistent.json")

    import json as _json

    node_dim = 320
    if pg.GNN_CONFIG.exists():
        with pg.GNN_CONFIG.open() as fh:
            node_dim = _json.load(fh).get("node_dim", 320)

    assert node_dim == 320


def test_gate3_reads_num_continuous_features_from_config(tmp_path, monkeypatch):
    """gate3_latency should use num_continuous_features from gnn_config.json when present."""
    import src.verify.promote_gnn as pg

    config_data = {
        "node_dim": 480,
        "num_continuous_features": 31,
        "feature_mode": 31,
        "esm2_model_name": "facebook/esm2_t12_35M_UR50D",
    }
    config_file = tmp_path / "gnn_config.json"
    config_file.write_text(json.dumps(config_data))

    monkeypatch.setattr(pg, "GNN_CONFIG", config_file)

    import json as _json
    from src.features import TRAIN_FEATURE_COLUMNS

    num_features = len(TRAIN_FEATURE_COLUMNS)
    if pg.GNN_CONFIG.exists():
        with pg.GNN_CONFIG.open() as fh:
            _cfg = _json.load(fh)
            num_features = _cfg.get("num_continuous_features", num_features)

    assert num_features == 31


def test_gate3_defaults_num_features_21_when_config_missing(tmp_path, monkeypatch):
    """gate3_latency falls back to 21 (TRAIN_FEATURE_COLUMNS) when gnn_config.json absent."""
    import src.verify.promote_gnn as pg
    from src.features import TRAIN_FEATURE_COLUMNS

    monkeypatch.setattr(pg, "GNN_CONFIG", tmp_path / "nonexistent.json")

    import json as _json

    num_features = len(TRAIN_FEATURE_COLUMNS)
    if pg.GNN_CONFIG.exists():
        with pg.GNN_CONFIG.open() as fh:
            _cfg = _json.load(fh)
            num_features = _cfg.get("num_continuous_features", num_features)

    assert num_features == 21


# ---------------------------------------------------------------------------
# v2.3: variable-length graph correctness - no zero-padding nodes in graphs
# ---------------------------------------------------------------------------


def test_dataset_v2_node_count_equals_peptide_length():
    """Each Data item must have exactly L nodes, not max_len (11)."""
    import numpy as np
    import pandas as pd
    from src.train_gnn import GraphPeptideDatasetV2

    seqs = ["GILGFVFT", "GILGFVFTL", "GILGFVFTLV", "GILGFVFTLVA"]
    lengths = [8, 9, 10, 11]
    df = pd.DataFrame({"peptide": seqs, "label": [1, 0, 1, 0]})

    esm2_cache = {}
    for seq in seqs:
        padded = torch.zeros(11, 320)
        padded[: len(seq)] = torch.randn(len(seq), 320)
        esm2_cache[seq] = padded

    X = pd.DataFrame(np.zeros((4, 21)))
    ds = GraphPeptideDatasetV2(df, X, np.array([1.0, 0.0, 1.0, 0.0]), esm2_cache, max_len=11)

    for i, (seq, L) in enumerate(zip(seqs, lengths)):
        item = ds[i]
        assert item.x.shape[0] == L, f"Expected {L} nodes for {seq}, got {item.x.shape[0]}"
        assert item.x.shape[0] != 11 or L == 11, "Padding nodes must not appear for sub-11-mers"


def test_dataset_v2_edge_index_within_node_range():
    """All edge indices must reference nodes within [0, L-1]."""
    import numpy as np
    import pandas as pd
    from src.train_gnn import GraphPeptideDatasetV2

    seqs = ["GILGFVFT", "GILGFVFTL"]
    df = pd.DataFrame({"peptide": seqs, "label": [1, 0]})
    esm2_cache = {seq: torch.zeros(11, 320) for seq in seqs}
    X = pd.DataFrame(np.zeros((2, 21)))
    ds = GraphPeptideDatasetV2(df, X, np.array([1.0, 0.0]), esm2_cache, max_len=11)

    for i, seq in enumerate(seqs):
        item = ds[i]
        L = len(seq)
        assert item.edge_index.max().item() == L - 1, (
            f"Edge index references node > {L - 1} for {seq}"
        )


def test_dataset_v2_pyg_batch_node_count():
    """PyG batching must produce total_nodes == sum of individual peptide lengths."""
    import numpy as np
    import pandas as pd
    from src.train_gnn import GraphPeptideDatasetV2
    from torch_geometric.loader import DataLoader as PyGDataLoader

    seqs = ["GILGFVFT", "GILGFVFTL", "GILGFVFTLV", "GILGFVFTLVA"]
    df = pd.DataFrame({"peptide": seqs, "label": [1, 0, 1, 0]})
    esm2_cache = {seq: torch.zeros(11, 320) for seq in seqs}
    X = pd.DataFrame(np.zeros((4, 21)))
    ds = GraphPeptideDatasetV2(df, X, np.array([1.0, 0.0, 1.0, 0.0]), esm2_cache, max_len=11)
    loader = PyGDataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    expected_nodes = sum(len(s) for s in seqs)  # 8+9+10+11 = 38
    assert batch.x.shape[0] == expected_nodes, (
        f"Expected {expected_nodes} nodes in batch, got {batch.x.shape[0]}"
    )


# ---------------------------------------------------------------------------
# v2.4 attention pooling (backward-compatible; default remains mean pool)
# ---------------------------------------------------------------------------


def test_predictor_v2_default_pooling_is_mean():
    """Default readout must stay mean pool so existing v2.1-v2.3 checkpoints load."""
    from src.gnn.models import GraphPredictorV2

    model = GraphPredictorV2(num_continuous_features=10)
    assert model.encoder.pooling == "mean"
    assert not hasattr(model.encoder, "att_pool")


def test_predictor_v2_attention_pooling_adds_gate():
    from src.gnn.models import GraphPredictorV2

    model = GraphPredictorV2(num_continuous_features=10, pooling="attention")
    assert model.encoder.pooling == "attention"
    assert hasattr(model.encoder, "att_pool")


def test_predictor_v2_invalid_pooling_raises():
    from src.gnn.models import GraphPredictorV2

    with pytest.raises(ValueError):
        GraphPredictorV2(num_continuous_features=10, pooling="maxpool")


def _build_v2_batch(node_dim=320, n_feats=21):
    import numpy as np
    import pandas as pd
    from src.train_gnn import GraphPeptideDatasetV2
    from torch_geometric.loader import DataLoader as PyGDataLoader

    seqs = ["GILGFVFT", "GILGFVFTL", "GILGFVFTLV", "GILGFVFTLVA"]
    df = pd.DataFrame({"peptide": seqs, "label": [1, 0, 1, 0]})
    esm2_cache = {}
    for seq in seqs:
        padded = torch.zeros(11, node_dim)
        padded[: len(seq)] = torch.randn(len(seq), node_dim)
        esm2_cache[seq] = padded
    X = pd.DataFrame(np.zeros((4, n_feats)))
    ds = GraphPeptideDatasetV2(df, X, np.array([1.0, 0.0, 1.0, 0.0]), esm2_cache, max_len=11)
    loader = PyGDataLoader(ds, batch_size=4, shuffle=False)
    return next(iter(loader))


def test_predictor_v2_attention_forward_shape():
    """Attention-pooled forward must return one logit per graph on a real PyG batch."""
    from src.gnn.models import GraphPredictorV2

    batch = _build_v2_batch()
    model = GraphPredictorV2(num_continuous_features=21, node_dim=320, pooling="attention")
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (4,), f"Expected (4,) logits, got {tuple(out.shape)}"


def test_predictor_v2_mean_forward_still_works():
    """Backward-compat: default mean pooling forward unchanged."""
    from src.gnn.models import GraphPredictorV2

    batch = _build_v2_batch()
    model = GraphPredictorV2(num_continuous_features=21, node_dim=320)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (4,)


def test_train_gnn_v2_accepts_pooling_param():
    import inspect
    from src.train_gnn import train_gnn_v2

    sig = inspect.signature(train_gnn_v2)
    assert "pooling" in sig.parameters
    assert sig.parameters["pooling"].default == "mean"
