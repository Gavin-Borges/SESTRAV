"""
Tests for verifying that SESTRAV ANN checkpoints load correctly and safely.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baseline_comparison import _load_torch_checkpoint, _score_with_ann
from src.features import compute_features_for_dataset, BINDING_ALLELE_COLUMNS

def test_ann_30feature_checkpoint_loads_and_scores():
    """Verify that the 30-feature ANN checkpoint loads safely with weights_only=True."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed; skipping ANN loading test")

    model_path = os.path.join("models", "ann_30feature_integrated.pt")
    if not os.path.isfile(model_path):
        pytest.skip(f"ANN checkpoint not found at: {model_path}")

    # 1. Load the checkpoint safely
    checkpoint = _load_torch_checkpoint(model_path)
    assert checkpoint is not None, "Checkpoint should not be None"
    
    # 2. Verify expected keys and feature count
    assert "n_features" in checkpoint, "n_features key must exist"
    assert checkpoint["n_features"] == 30, f"Expected 30 features, got {checkpoint['n_features']}"
    assert "model_state_dict" in checkpoint, "model_state_dict key must exist"
    assert "scaler_mean" in checkpoint, "scaler_mean key must exist"
    assert "scaler_scale" in checkpoint, "scaler_scale key must exist"

    # 3. Verify float32 or native float types (no float64 tensors or numpy Float64DType wrapper)
    scaler_mean = checkpoint["scaler_mean"]
    scaler_scale = checkpoint["scaler_scale"]

    if isinstance(scaler_mean, torch.Tensor):
        assert scaler_mean.dtype == torch.float32, f"Expected float32 scaler_mean, got {scaler_mean.dtype}"
    else:
        assert isinstance(scaler_mean, (list, float)), f"Expected list or float for scaler_mean, got {type(scaler_mean)}"

    if isinstance(scaler_scale, torch.Tensor):
        assert scaler_scale.dtype == torch.float32, f"Expected float32 scaler_scale, got {scaler_scale.dtype}"
    else:
        assert isinstance(scaler_scale, (list, float)), f"Expected list or float for scaler_scale, got {type(scaler_scale)}"

    # 4. Verify scoring works on dummy data with 30-feature schema
    df = pd.DataFrame([
        {"peptide": "CLGGLLTMV", "presentation_score": 0.9},
        {"peptide": "RAKFKQLL", "presentation_score": 0.4},
    ])
    
    # Compute base 22 features
    df = compute_features_for_dataset(df, peptide_col="peptide", binding_col="presentation_score")
    
    # Add binding allele features to reach 30 features (8 allele columns)
    for idx, col in enumerate(BINDING_ALLELE_COLUMNS):
        df[col] = np.clip(df["presentation_score"] - (idx * 0.01), 0.0, 1.0)

    # Score with the ANN model path
    scores = _score_with_ann(df, model_path)
    assert scores.shape[0] == 2
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0), "Scores must be probability outputs"
