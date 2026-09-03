"""Unit tests for src/gnn/models.py (GCN encoder + graph predictor, both v1 and v2).

Exercises forward passes with small synthetic batches on CPU. GraphPredictor and
GraphPredictorV2 both contain BatchNorm1d, so batch size > 1 is required.
"""

import pytest

torch = pytest.importorskip("torch")

from src.gnn.models import GCNLayer, GraphEncoder, GraphPredictor, GraphEncoderV2, GraphPredictorV2
from src.gnn.graph_builder import GraphBuilder


MAX_LEN = 11


def test_gcn_layer_forward_shape():
    layer = GCNLayer(in_features=20, out_features=8)
    x = torch.zeros(4, MAX_LEN, 20)
    adj = torch.eye(MAX_LEN)
    out = layer(x, adj)
    assert out.shape == (4, MAX_LEN, 8)


def test_graph_encoder_pools_to_fixed_dim():
    enc = GraphEncoder(in_features=20, hidden_dim1=16, hidden_dim2=32)
    node_x = torch.rand(3, MAX_LEN, 20)
    adj = torch.eye(MAX_LEN)
    out = enc(node_x, adj)
    # Global mean pool over nodes -> (batch, hidden_dim2).
    assert out.shape == (3, 32)


def test_graph_predictor_forward_returns_logit_per_sample():
    model = GraphPredictor(num_continuous_features=30)
    batch = 4  # > 1 so BatchNorm1d is happy in train mode
    node_x = torch.rand(batch, MAX_LEN, 20)
    feat_x = torch.rand(batch, 30)
    adj = torch.eye(MAX_LEN)
    out = model(node_x, feat_x, adj)
    assert out.shape == (batch,)


def test_graph_predictor_eval_mode_single_sample():
    model = GraphPredictor(num_continuous_features=12)
    model.eval()  # eval mode lets BatchNorm handle a single sample
    node_x = torch.rand(1, MAX_LEN, 20)
    feat_x = torch.rand(1, 12)
    adj = torch.eye(MAX_LEN)
    with torch.no_grad():
        out = model(node_x, feat_x, adj)
    assert out.shape == (1,)


# ---------------------------------------------------------------------------
# GNN v2.1 - GraphEncoderV2 + GraphPredictorV2 (GINEConv + ESM-2 node dim)
# ---------------------------------------------------------------------------


def _make_pyg_batch(batch_size: int, num_features: int, node_dim: int = 320):
    """Build a synthetic PyG batch compatible with GraphPredictorV2."""
    from torch_geometric.data import Data, Batch

    edge_index, edge_attr = GraphBuilder.build_pyg_chain_graph(MAX_LEN)
    data_list = [
        Data(
            x=torch.rand(MAX_LEN, node_dim),
            edge_index=edge_index,
            edge_attr=edge_attr,
            physico=torch.rand(1, num_features),
            y=torch.tensor([float(i % 2)]),
        )
        for i in range(batch_size)
    ]
    return Batch.from_data_list(data_list)


def test_pyg_chain_graph_edge_count():
    edge_index, edge_attr = GraphBuilder.build_pyg_chain_graph(MAX_LEN)
    expected = MAX_LEN + 2 * (MAX_LEN - 1)  # self-loops + bidirectional edges
    assert edge_index.shape == (2, expected)
    assert edge_attr.shape == (expected, 3)


def test_pyg_chain_graph_edge_attrs_one_hot():
    _, edge_attr = GraphBuilder.build_pyg_chain_graph(MAX_LEN)
    # Each row should have exactly one 1.0 and two 0.0 values
    assert (edge_attr.sum(dim=1) == 1.0).all()


def test_graph_encoder_v2_output_shape():
    enc = GraphEncoderV2(node_dim=32, hidden_dim=16, out_dim=8, edge_dim=3)
    batch = _make_pyg_batch(batch_size=4, num_features=10, node_dim=32)
    out = enc(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    assert out.shape == (4, 8)


def test_graph_predictor_v2_forward_shape():
    model = GraphPredictorV2(num_continuous_features=30, node_dim=32)
    # Rebuild with small node_dim to avoid 320-dim overhead in tests
    model.encoder = GraphEncoderV2(node_dim=32, hidden_dim=16, out_dim=128, edge_dim=3)
    batch = _make_pyg_batch(batch_size=4, num_features=30, node_dim=32)
    out = model(batch)
    assert out.shape == (4,)


# ---------------------------------------------------------------------------
# Gradient flow - the graph encoder must actually receive a gradient
#
# Every other test in this file is a forward-shape test, so detaching the
# encoder output in GraphPredictor.forward / GraphPredictorV2.forward leaves the
# whole suite green while reducing the model to a physico-only MLP. These two
# assert the loss reaches the first conv weight of each encoder.
# ---------------------------------------------------------------------------


def test_graph_predictor_gradient_reaches_encoder():
    """v1: backward must leave a non-zero grad on the first GCN layer weight."""
    torch.manual_seed(0)
    model = GraphPredictor(num_continuous_features=30)
    batch = 4  # > 1 so BatchNorm1d is happy in train mode
    node_x = torch.rand(batch, MAX_LEN, 20)
    feat_x = torch.rand(batch, 30)
    adj = torch.eye(MAX_LEN)
    y = torch.tensor([float(i % 2) for i in range(batch)])

    out = model(node_x, feat_x, adj)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
    loss.backward()

    grad = model.encoder.gcn1.weight.grad
    assert grad is not None, "no gradient reached GraphEncoder.gcn1 - encoder is detached"
    assert grad.norm().item() > 0.0, "gradient reached GraphEncoder.gcn1 but is all zeros"
    # Control: the physico branch trains either way, so its grad separates a
    # severed encoder from a backward pass that never ran at all.
    assert model.physico_block[0].weight.grad is not None


def test_graph_predictor_v2_gradient_reaches_encoder():
    """v2 production path: backward must reach GINEConv 1's first Linear weight."""
    torch.manual_seed(0)
    model = GraphPredictorV2(num_continuous_features=30, node_dim=32)
    model.encoder = GraphEncoderV2(node_dim=32, hidden_dim=16, out_dim=128, edge_dim=3)
    batch = _make_pyg_batch(batch_size=4, num_features=30, node_dim=32)

    out = model(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, batch.y.view(-1))
    loss.backward()

    grad = model.encoder.conv1.nn[0].weight.grad
    assert grad is not None, "no gradient reached GraphEncoderV2.conv1 - encoder is detached"
    assert grad.norm().item() > 0.0, "gradient reached GraphEncoderV2.conv1 but is all zeros"
    assert model.physico_block[0].weight.grad is not None


def test_graph_predictor_v2_eval_single_sample():
    model = GraphPredictorV2(num_continuous_features=12, node_dim=32)
    model.encoder = GraphEncoderV2(node_dim=32, hidden_dim=16, out_dim=128, edge_dim=3)
    model.eval()
    batch = _make_pyg_batch(batch_size=1, num_features=12, node_dim=32)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (1,)
