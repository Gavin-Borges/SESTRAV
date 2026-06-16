"""End-to-end unit tests for ``score_immunogenicity`` and plotting.

These exercise the real decision logic on Stage 4's scoring critical path — the
prototype RandomForest fallback, the joblib/.pt model branches (with the heavy
loader mocked or a real lightweight checkpoint), calibration, freeze-mode guards,
and the plotting helper. They complement the pure-helper tests in
``test_stage4_scoring_units.py``.
"""

import os

import numpy as np
import pandas as pd
import pytest

import functions.stage4_immunogenicity_scoring as s4
from src.features import (
    FEATURE_COLUMNS,
    TRAIN_FEATURE_COLUMNS,
    FEATURE_COLUMNS_30,
)


def _feature_frame(cols, n=16, seed=0):
    """Build a DataFrame with the given feature columns plus a peptide label."""
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(size=n) for c in cols}
    data["peptide"] = [f"PEP{i:03d}" for i in range(n)]
    return pd.DataFrame(data)


def _run_in_results_dir(monkeypatch, tmp_path):
    """cd into a temp dir that has a results/ folder for CSV/PNG output."""
    (tmp_path / "results").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)


# --- prototype RandomForest fallback -------------------------------------------------

def test_prototype_path_with_binding_score(monkeypatch, tmp_path):
    _run_in_results_dir(monkeypatch, tmp_path)
    df = _feature_frame(FEATURE_COLUMNS)  # FEATURE_COLUMNS includes binding_score
    ranked, model = s4.score_immunogenicity(df, "HPV16", model_path=None)

    assert "immunogenicity_score" in ranked.columns
    assert ranked["immunogenicity_score"].between(0, 1).all()
    # Deterministic contiguous ranks 1..N, sorted descending by score.
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))
    assert ranked["immunogenicity_score"].is_monotonic_decreasing
    assert model is not None
    assert os.path.isfile("results/HPV16_ranked.csv")


def test_prototype_path_with_presentation_score(monkeypatch, tmp_path):
    _run_in_results_dir(monkeypatch, tmp_path)
    cols = [c for c in FEATURE_COLUMNS if c != "binding_score"]
    df = _feature_frame(cols)
    df["presentation_score"] = np.linspace(0.1, 0.9, len(df))
    ranked, _ = s4.score_immunogenicity(df, "EBV", model_path=None)
    assert "immunogenicity_score" in ranked.columns


def test_freeze_mode_without_model_raises(monkeypatch, tmp_path):
    _run_in_results_dir(monkeypatch, tmp_path)
    df = _feature_frame(FEATURE_COLUMNS)
    with pytest.raises(RuntimeError, match="Freeze mode requires a trained model"):
        s4.score_immunogenicity(df, "HPV16", model_path=None, freeze_mode=True)


# --- joblib model branch (heavy loader mocked) ---------------------------------------

class _FakeSklearnModel:
    def __init__(self, n_features):
        self.n_features_in_ = n_features

    def predict_proba(self, X):
        # Monotonic-in-row-sum probabilities, shape (n, 2).
        p1 = 1.0 / (1.0 + np.exp(-np.asarray(X).sum(axis=1)))
        return np.column_stack([1 - p1, p1])


@pytest.mark.parametrize(
    "cols",
    [TRAIN_FEATURE_COLUMNS, FEATURE_COLUMNS_30],
    ids=["train21", "f30"],
)
def test_joblib_branch_scores(monkeypatch, tmp_path, cols):
    _run_in_results_dir(monkeypatch, tmp_path)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"stub")  # only needs to exist; loader is mocked
    monkeypatch.setattr(
        s4, "load_verified_joblib", lambda p, required_checksum=True: _FakeSklearnModel(len(cols))
    )
    df = _feature_frame(cols)
    ranked, model = s4.score_immunogenicity(df, "HPV16", model_path=str(model_path))
    assert isinstance(model, _FakeSklearnModel)
    assert ranked["immunogenicity_score"].between(0, 1).all()
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))


def test_joblib_missing_loader_warns(monkeypatch, tmp_path, capsys):
    """When joblib support is unavailable the prototype fallback still scores."""
    _run_in_results_dir(monkeypatch, tmp_path)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"stub")
    monkeypatch.setattr(s4, "load_verified_joblib", None)
    df = _feature_frame(FEATURE_COLUMNS)
    ranked, _ = s4.score_immunogenicity(df, "HPV16", model_path=str(model_path))
    assert "immunogenicity_score" in ranked.columns
    assert "joblib not available" in capsys.readouterr().out


def test_model_path_alias_resolution(monkeypatch, tmp_path):
    _run_in_results_dir(monkeypatch, tmp_path)
    real_path = tmp_path / "resolved.joblib"
    real_path.write_bytes(b"stub")
    # A calibrator alongside the model exercises the calibrated_score assignment.
    (tmp_path / "platt_calibrator.joblib").write_bytes(b"stub")
    monkeypatch.setattr(s4, "resolve_model_path", lambda p: str(real_path))
    monkeypatch.setattr(
        s4, "load_verified_joblib",
        lambda p, required_checksum=True: _FakeSklearnModel(len(TRAIN_FEATURE_COLUMNS)),
    )
    df = _feature_frame(TRAIN_FEATURE_COLUMNS)
    ranked, _ = s4.score_immunogenicity(df, "HPV16", model_path="alias://model")
    assert "immunogenicity_score" in ranked.columns
    assert "calibrated_score" in ranked.columns


# --- calibration with a calibrator present -------------------------------------------

class _FakeCalibrator:
    def predict_proba(self, logits):
        p1 = 1.0 / (1.0 + np.exp(-np.asarray(logits).ravel()))
        return np.column_stack([1 - p1, p1])


def test_apply_calibration_with_calibrator(monkeypatch, tmp_path):
    (tmp_path / "platt_calibrator.joblib").write_bytes(b"stub")
    monkeypatch.setattr(
        s4, "load_verified_joblib", lambda p, required_checksum=True: _FakeCalibrator()
    )
    scores = np.array([0.2, 0.5, 0.8])
    out, applied = s4._apply_calibration(scores, str(tmp_path))
    assert applied is True
    assert out.shape == scores.shape
    assert np.all((out >= 0) & (out <= 1))


# --- PyTorch .pt branch (real lightweight checkpoint) --------------------------------

def _save_ann_checkpoint(path, n_features):
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class _MLP(nn.Module):
        def __init__(self, input_dim, hidden=(64, 32), dropout=0.3):
            super().__init__()
            layers, prev = [], input_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = _MLP(n_features)
    torch.save(
        {
            "n_features": n_features,
            "scaler_mean": torch.zeros(n_features),
            "scaler_scale": torch.ones(n_features),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    # The loader enforces checksum verification (required=True); generate the
    # sidecar manifest with the project's own helper so the safe-load path runs.
    from src.artifact_integrity import default_manifest_path_for, update_checksum_manifest

    update_checksum_manifest(default_manifest_path_for(path), [path])


def test_pytorch_branch_scores(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    _run_in_results_dir(monkeypatch, tmp_path)
    pt_path = tmp_path / "ann.pt"
    _save_ann_checkpoint(str(pt_path), len(FEATURE_COLUMNS_30))
    df = _feature_frame(FEATURE_COLUMNS_30)
    ranked, model = s4.score_immunogenicity(df, "HPV16", model_path=str(pt_path))
    assert ranked["immunogenicity_score"].between(0, 1).all()
    assert model is not None


def test_pytorch_branch_mc_dropout(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    _run_in_results_dir(monkeypatch, tmp_path)
    pt_path = tmp_path / "ann.pt"
    _save_ann_checkpoint(str(pt_path), len(FEATURE_COLUMNS_30))
    df = _feature_frame(FEATURE_COLUMNS_30)
    ranked, _ = s4.score_immunogenicity(
        df, "HPV16", model_path=str(pt_path), mc_dropout=True
    )
    assert "mc_score" in ranked.columns
    assert "uncertainty_std" in ranked.columns
    assert (ranked["uncertainty_std"] >= 0).all()


def test_pytorch_freeze_mode_missing_features_raises(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    _run_in_results_dir(monkeypatch, tmp_path)
    pt_path = tmp_path / "ann.pt"
    _save_ann_checkpoint(str(pt_path), len(FEATURE_COLUMNS_30))
    # Drop a required feature so the freeze-mode guard fires.
    df = _feature_frame(FEATURE_COLUMNS_30[:-1])
    with pytest.raises(RuntimeError, match="Missing"):
        s4.score_immunogenicity(df, "HPV16", model_path=str(pt_path), freeze_mode=True)


# --- plotting ------------------------------------------------------------------------

def test_plot_immunogenicity_scores(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    ranked = pd.DataFrame(
        {
            "peptide": [f"PEP{i}" for i in range(25)],
            "immunogenicity_score": np.linspace(0.0, 1.0, 25),
        }
    )
    s4.plot_immunogenicity_scores(ranked, "HPV16", top_n=20)
    assert os.path.isfile("results/HPV16_top20_immunogenicity.png")
    assert os.path.isfile("results/HPV16_score_distribution.png")
