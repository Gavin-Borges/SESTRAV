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
# promote_gnn gate3 reads node_dim, num_continuous_features and pooling from
# gnn_config.json
#
# These call the real gate3_latency. An earlier version of each test only
# re-implemented the config read inline (a local `json.load(...).get(...)`)
# and asserted that json round-tripped a dict it had just written, so all four
# passed with gate3_latency replaced by a raiser and promote_gnn's config-read
# branch stayed at zero coverage across the whole suite. The heavy I/O is stubbed
# the same way tests/test_promote_gnn_runner.py does it (real files for the
# existence checks, a fake RF, a state dict handed back from torch.load), and the
# assertion is on the architecture gate3 actually instantiates.
# ---------------------------------------------------------------------------


def _fake_rf(n_features):
    """Stand-in for the joblib RF whose only role here is the latency baseline."""
    import numpy as np

    class _FakeRF:
        n_features_in_ = n_features

        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    return _FakeRF()


def _run_gate3(monkeypatch, tmp_path, *, state_dict, n_features, checkpoint_path=None):
    """Call the real gate3_latency and return (GateResult, kwargs it built the GNN with).

    Everything gate3 touches on disk is redirected: a real (stub) checkpoint and
    RF file so the existence checks pass, load_verified_joblib and torch.load
    stubbed, and GraphPredictorV2 wrapped so the constructor kwargs gate3 derives
    from the config are observable. The wrapper returns a genuine model, so a
    node_dim/num_features/pooling that disagrees with *state_dict* still fails
    loudly in load_state_dict.
    """
    import src.artifact_integrity as artifact_integrity
    import src.gnn.models as models
    import src.verify.promote_gnn as pg

    checkpoint = tmp_path / "structural_gnn_v2.pth"
    checkpoint.write_bytes(b"stub")
    rf_model = tmp_path / "rf_31feature_integrated.joblib"
    rf_model.write_bytes(b"stub")

    built: dict[str, object] = {}
    real_cls = models.GraphPredictorV2

    def _record(*args, **kwargs):
        built.update(kwargs)
        return real_cls(*args, **kwargs)

    monkeypatch.setattr(models, "GraphPredictorV2", _record)
    monkeypatch.setattr(pg, "GNN_CHECKPOINT", checkpoint)
    monkeypatch.setattr(pg, "RF_MODEL_PATH", rf_model)
    monkeypatch.setattr(
        artifact_integrity, "load_verified_joblib", lambda *a, **k: _fake_rf(n_features)
    )
    monkeypatch.setattr(torch, "load", lambda *a, **k: state_dict)

    result = pg.gate3_latency(checkpoint_path)
    # "ratio=" only appears once the GNN was built, loaded and actually timed.
    assert "ratio=" in str(result.value), f"gate3_latency did not complete: {result.value}"
    return result, built


def test_gate3_reads_node_dim_from_config(tmp_path, monkeypatch):
    """gate3_latency must build the GNN with node_dim from gnn_config.json."""
    import src.verify.promote_gnn as pg
    from src.features import TRAIN_FEATURE_COLUMNS
    from src.gnn.models import GraphPredictorV2

    n_features = len(TRAIN_FEATURE_COLUMNS)
    config_file = tmp_path / "gnn_config.json"
    config_file.write_text(
        json.dumps({"node_dim": 480, "esm2_model_name": "facebook/esm2_t12_35M_UR50D"})
    )
    monkeypatch.setattr(pg, "GNN_CONFIG", config_file)

    state = GraphPredictorV2(num_continuous_features=n_features, node_dim=480).state_dict()
    _, built = _run_gate3(monkeypatch, tmp_path, state_dict=state, n_features=n_features)

    assert built["node_dim"] == 480


def test_gate3_defaults_node_dim_320_when_config_missing(tmp_path, monkeypatch):
    """gate3_latency falls back to 320 when gnn_config.json does not exist."""
    import src.verify.promote_gnn as pg
    from src.features import TRAIN_FEATURE_COLUMNS
    from src.gnn.models import GraphPredictorV2

    n_features = len(TRAIN_FEATURE_COLUMNS)
    monkeypatch.setattr(pg, "GNN_CONFIG", tmp_path / "nonexistent.json")

    state = GraphPredictorV2(num_continuous_features=n_features).state_dict()
    _, built = _run_gate3(monkeypatch, tmp_path, state_dict=state, n_features=n_features)

    assert built["node_dim"] == 320


def test_gate3_reads_num_continuous_features_from_config(tmp_path, monkeypatch):
    """gate3_latency must build the GNN with num_continuous_features from the config."""
    import src.verify.promote_gnn as pg
    from src.gnn.models import GraphPredictorV2

    config_file = tmp_path / "gnn_config.json"
    config_file.write_text(
        json.dumps(
            {
                "node_dim": 320,
                "num_continuous_features": 31,
                "feature_mode": 31,
                "esm2_model_name": "facebook/esm2_t6_8M_UR50D",
            }
        )
    )
    monkeypatch.setattr(pg, "GNN_CONFIG", config_file)

    state = GraphPredictorV2(num_continuous_features=31).state_dict()
    _, built = _run_gate3(monkeypatch, tmp_path, state_dict=state, n_features=31)

    assert built["num_continuous_features"] == 31


def test_gate3_defaults_num_features_21_when_config_missing(tmp_path, monkeypatch):
    """gate3_latency falls back to 21 (TRAIN_FEATURE_COLUMNS) when gnn_config.json absent."""
    import src.verify.promote_gnn as pg
    from src.features import TRAIN_FEATURE_COLUMNS
    from src.gnn.models import GraphPredictorV2

    n_features = len(TRAIN_FEATURE_COLUMNS)
    assert n_features == 21
    monkeypatch.setattr(pg, "GNN_CONFIG", tmp_path / "nonexistent.json")

    state = GraphPredictorV2(num_continuous_features=n_features).state_dict()
    _, built = _run_gate3(monkeypatch, tmp_path, state_dict=state, n_features=n_features)

    assert built["num_continuous_features"] == 21


def test_gate3_reads_pooling_from_config(tmp_path, monkeypatch):
    """The v2.4 readout is the third key gate3 reads, and nothing else covers it.

    An attention checkpoint carries encoder.att_pool weights that a mean-pooled
    GraphPredictorV2 has no slot for, so ignoring this key makes gate3 time the
    wrong architecture (GNN rule 8/9).
    """
    import src.verify.promote_gnn as pg
    from src.features import TRAIN_FEATURE_COLUMNS
    from src.gnn.models import GraphPredictorV2

    n_features = len(TRAIN_FEATURE_COLUMNS)
    config_file = tmp_path / "gnn_config.json"
    config_file.write_text(json.dumps({"node_dim": 320, "pooling": "attention"}))
    monkeypatch.setattr(pg, "GNN_CONFIG", config_file)

    state = GraphPredictorV2(num_continuous_features=n_features, pooling="attention").state_dict()
    _, built = _run_gate3(monkeypatch, tmp_path, state_dict=state, n_features=n_features)

    assert built["pooling"] == "attention"


def test_gate3_reads_the_checkpoint_siblings_config_not_the_tracked_one(tmp_path, monkeypatch):
    """With a --checkpoint override, the config must come from the checkpoint's directory.

    gate3_latency documents `checkpoint_path.parent / "gnn_config.json"` as the
    config source for an override, so that node_dim stays matched to the file
    being timed. The tracked GNN_CONFIG here names a different node_dim on
    purpose: reading it instead would build a 320-dim model for a 480-dim
    checkpoint.
    """
    import src.verify.promote_gnn as pg
    from src.features import TRAIN_FEATURE_COLUMNS
    from src.gnn.models import GraphPredictorV2

    n_features = len(TRAIN_FEATURE_COLUMNS)

    tracked_config = tmp_path / "tracked" / "gnn_config.json"
    tracked_config.parent.mkdir()
    tracked_config.write_text(json.dumps({"node_dim": 320}))
    monkeypatch.setattr(pg, "GNN_CONFIG", tracked_config)

    scratch_dir = tmp_path / "scratch_run"
    scratch_dir.mkdir()
    scratch_checkpoint = scratch_dir / "gnn.pth"
    scratch_checkpoint.write_bytes(b"stub")
    (scratch_dir / "gnn_config.json").write_text(json.dumps({"node_dim": 480}))

    state = GraphPredictorV2(num_continuous_features=n_features, node_dim=480).state_dict()
    _, built = _run_gate3(
        monkeypatch,
        tmp_path,
        state_dict=state,
        n_features=n_features,
        checkpoint_path=scratch_checkpoint,
    )

    assert built["node_dim"] == 480


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


def test_dataset_v2_threads_edge_mode_to_emitted_graphs():
    """--edge-mode must reach the Data items, not just the builder.

    Builder-level tests pass even if the dataset never forwards the flag, which
    would silently run the full graph while the run is recorded as an ablation.
    """
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
    labels = np.array([1.0, 0.0, 1.0, 0.0])

    full = GraphPeptideDatasetV2(df, X, labels, esm2_cache, max_len=11)
    ablated = GraphPeptideDatasetV2(
        df, X, labels, esm2_cache, max_len=11, edge_mode="self-loop-only"
    )

    for i, L in enumerate(lengths):
        assert full[i].edge_index.shape[1] == L + 2 * (L - 1)
        assert ablated[i].edge_index.shape[1] == L
        assert (ablated[i].edge_index[0] == ablated[i].edge_index[1]).all()


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
