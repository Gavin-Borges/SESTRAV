"""Extended coverage tests for src/model.py.

Targets the 7 uncovered statements and 8 branch misses remaining after
test_model.py:
  - get_device: CUDA path (line 70)
  - set_seeds: CUDA seed paths (lines 79-81)
  - train_one_fold: device=None auto-detect (177), exhaust-all-epochs branch
    (202->232), best_state-is-None branch (232->235)
  - run_cv: device=None auto-detect (259)
  - train_final_model: device=None auto-detect (291)
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src import model as M


# ---------------------------------------------------------------------------
# Shared toy dataset (tiny - tests must run in < 1 s on CPU)
# ---------------------------------------------------------------------------


@pytest.fixture
def toy():
    rng = np.random.default_rng(1)
    n, d = 40, 4
    X = rng.normal(size=(n, d)).astype("float32")
    y = (X[:, 0] > 0).astype("float32")
    y[0], y[1] = 1.0, 0.0
    return X, y


# ---------------------------------------------------------------------------
# get_device - CUDA path (line 70)
# ---------------------------------------------------------------------------


class TestGetDeviceCuda:
    def test_cuda_available_returns_cuda_device(self):
        with patch("src.model.torch.cuda.is_available", return_value=True):
            dev = M.get_device()
        assert dev == torch.device("cuda")


# ---------------------------------------------------------------------------
# set_seeds - CUDA seed paths (lines 79-81)
# ---------------------------------------------------------------------------


class TestSetSeedsCuda:
    def test_cuda_available_sets_all_cuda_seeds(self):
        mock_cuda = MagicMock()
        mock_cuda.is_available.return_value = True
        mock_cuda.manual_seed_all = MagicMock()
        mock_backends = MagicMock()

        with (
            patch("src.model.torch.cuda", mock_cuda),
            patch("src.model.torch.backends", mock_backends),
        ):
            M.set_seeds(42)

        mock_cuda.manual_seed_all.assert_called_once_with(42)
        assert mock_backends.cudnn.deterministic is True
        assert mock_backends.cudnn.benchmark is False


# ---------------------------------------------------------------------------
# train_one_fold - device auto-detect (line 177)
# ---------------------------------------------------------------------------


class TestTrainOneFoldDeviceNone:
    def test_device_none_falls_back_to_get_device(self, toy):
        X, y = toy
        M.set_seeds(0)
        net = M.FlexibleMLP(input_dim=X.shape[1], hidden_sizes=[4], dropout=0.0)
        metrics, scaler = M.train_one_fold(
            net,
            X[:30],
            y[:30],
            X[30:],
            y[30:],
            pos_weight=1.0,
            max_epochs=1,
            patience=5,
        )
        assert isinstance(metrics, dict)
        assert hasattr(scaler, "transform")


# ---------------------------------------------------------------------------
# train_one_fold - exhaust all epochs without early stopping (line 202->232)
# ---------------------------------------------------------------------------


class TestTrainOneFoldExhaustsEpochs:
    def test_patience_larger_than_epochs_exhausts_loop(self, toy):
        X, y = toy
        M.set_seeds(0)
        net = M.FlexibleMLP(input_dim=X.shape[1], hidden_sizes=[4], dropout=0.0)
        metrics, _ = M.train_one_fold(
            net,
            X[:30],
            y[:30],
            X[30:],
            y[30:],
            pos_weight=1.0,
            max_epochs=2,
            patience=100,
            device=torch.device("cpu"),
        )
        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# train_one_fold - best_state is None when max_epochs=0 (line 232->235)
# ---------------------------------------------------------------------------


class TestTrainOneFoldBestStateNone:
    def test_zero_epochs_skips_load_state_dict(self, toy):
        X, y = toy
        M.set_seeds(0)
        net = M.FlexibleMLP(input_dim=X.shape[1], hidden_sizes=[4], dropout=0.0)
        metrics, _ = M.train_one_fold(
            net,
            X[:30],
            y[:30],
            X[30:],
            y[30:],
            pos_weight=1.0,
            max_epochs=0,
            patience=5,
            device=torch.device("cpu"),
        )
        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# run_cv - device auto-detect (line 259)
# ---------------------------------------------------------------------------


class TestRunCvDeviceNone:
    def test_device_none_falls_back_to_get_device(self, toy):
        X, y = toy
        config = {"hidden": [4], "dropout": 0.0, "activation": "relu"}
        fold_metrics = M.run_cv(
            X,
            y,
            strat_key=y,
            config=config,
            pos_weight=1.0,
            n_folds=2,
        )
        assert len(fold_metrics) == 2
        assert all(isinstance(m, dict) for m in fold_metrics)


# ---------------------------------------------------------------------------
# train_final_model - device auto-detect (line 291)
# ---------------------------------------------------------------------------


class TestTrainFinalModelDeviceNone:
    def test_device_none_falls_back_to_get_device(self, toy):
        X, y = toy
        config = {"hidden": [4], "dropout": 0.0, "activation": "relu"}
        model, scaler = M.train_final_model(
            X[:30],
            y[:30],
            X[30:],
            y[30:],
            config,
            pos_weight=1.0,
            max_epochs=1,
            patience=5,
        )
        assert isinstance(model, M.FlexibleMLP)
        assert hasattr(scaler, "transform")
