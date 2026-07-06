"""Unit tests for src/model.py (FlexibleMLP + training utilities).

Runs on CPU with tiny synthetic data and very short epoch budgets so the
training loop, early stopping, CV runner, and final-model path are all
exercised quickly without GPUs or real datasets.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src import model as M


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    n, d = 60, 5
    X = rng.normal(size=(n, d)).astype("float32")
    # Linearly separable-ish label so AUC-PR is meaningful but cheap.
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype("float32")
    # Guarantee both classes are present.
    y[0], y[1] = 1.0, 0.0
    return X, y


def test_get_device_returns_torch_device():
    dev = M.get_device()
    assert isinstance(dev, torch.device)
    assert dev.type in {"cpu", "cuda"}


def test_set_seeds_is_deterministic():
    M.set_seeds(123)
    a = torch.rand(3)
    M.set_seeds(123)
    b = torch.rand(3)
    assert torch.allclose(a, b)


def test_flexible_mlp_forward_shape():
    M.set_seeds(0)
    net = M.FlexibleMLP(input_dim=5, hidden_sizes=[8, 4], dropout=0.1)
    out = net(torch.zeros(7, 5))
    assert out.shape == (7,)  # squeezed single logit per row


@pytest.mark.parametrize("activation", ["relu", "gelu", "leaky"])
def test_flexible_mlp_activations(activation):
    net = M.FlexibleMLP(input_dim=3, hidden_sizes=[4], dropout=0.0, activation=activation)
    assert net(torch.zeros(2, 3)).shape == (2,)


def test_flexible_mlp_unknown_activation_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        M.FlexibleMLP(input_dim=3, hidden_sizes=[4], dropout=0.0, activation="swish")


def test_compute_pos_weight():
    # 2 negatives, 1 positive -> 2.0
    assert M.compute_pos_weight([0, 0, 1]) == 2.0


def test_compute_pos_weight_no_positives_raises():
    with pytest.raises(ValueError, match="No positive samples"):
        M.compute_pos_weight([0, 0, 0])


def test_sigmoid_bounds_and_clip():
    assert M._sigmoid(0.0) == pytest.approx(0.5)
    # Extreme values must not overflow thanks to clipping.
    assert M._sigmoid(np.array([-1e6, 1e6])).tolist() == pytest.approx([0.0, 1.0])


def test_train_one_fold_returns_metrics_and_scaler(toy_data):
    X, y = toy_data
    M.set_seeds(0)
    net = M.FlexibleMLP(input_dim=X.shape[1], hidden_sizes=[8], dropout=0.0)
    metrics, scaler = M.train_one_fold(
        net,
        X[:40],
        y[:40],
        X[40:],
        y[40:],
        pos_weight=1.0,
        max_epochs=3,
        patience=2,
        device=torch.device("cpu"),
    )
    assert isinstance(metrics, dict)
    assert hasattr(scaler, "transform")


def test_train_one_fold_single_class_val(toy_data):
    # Validation set with one class triggers the auc_pr=0 branch.
    X, y = toy_data
    y_val = np.ones(10, dtype="float32")
    net = M.FlexibleMLP(input_dim=X.shape[1], hidden_sizes=[4], dropout=0.0)
    metrics, _ = M.train_one_fold(
        net,
        X[:40],
        y[:40],
        X[:10],
        y_val,
        pos_weight=1.0,
        max_epochs=2,
        patience=1,
        device=torch.device("cpu"),
    )
    assert isinstance(metrics, dict)


def test_run_cv_returns_per_fold_metrics(toy_data):
    X, y = toy_data
    config = {"hidden": [8], "dropout": 0.0, "activation": "relu"}
    metrics = M.run_cv(
        X,
        y,
        strat_key=y,
        config=config,
        pos_weight=1.0,
        n_folds=2,
        device=torch.device("cpu"),
    )
    assert len(metrics) == 2
    assert all(isinstance(m, dict) for m in metrics)


def test_train_final_model(toy_data):
    X, y = toy_data
    config = {"hidden": [8], "dropout": 0.0, "activation": "relu"}
    model, scaler = M.train_final_model(
        X[:40],
        y[:40],
        X[40:],
        y[40:],
        config,
        pos_weight=1.0,
        max_epochs=2,
        patience=1,
        device=torch.device("cpu"),
    )
    assert isinstance(model, M.FlexibleMLP)
    assert hasattr(scaler, "transform")
