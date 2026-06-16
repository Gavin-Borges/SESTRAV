"""Unit tests for src/gnn/models.py (GCN encoder + graph predictor).

Exercises forward passes with small synthetic batches on CPU. GraphPredictor
contains a BatchNorm1d, so a batch size > 1 is used.
"""

import pytest

torch = pytest.importorskip("torch")

from src.gnn.models import GCNLayer, GraphEncoder, GraphPredictor


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
