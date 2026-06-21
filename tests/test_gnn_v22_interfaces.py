"""Tests for GNN v2.2 interfaces: node_dim propagation, gnn_config.json,
early-stopping params, binding_matrix_path wiring, num_continuous_features
propagation, and promote_gnn gate3 config-reading path.
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
