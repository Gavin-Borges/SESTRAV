"""
Lightweight smoke tests for optional ANN/GNN modules.

These tests are intentionally non-blocking for environments that do not
install optional dependencies (torch / torch-geometric).
"""

import os
import sys

import numpy as np
import pandas as pd

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import compute_features_for_dataset, BINDING_ALLELE_COLUMNS


def test_optional_module_imports_are_safe():
    """Optional modules should either import or fail with clear dependency errors."""
    ann_error = None
    gnn_error = None
    try:
        import src.ann_benchmark  # noqa: F401
    except Exception as exc:  # pragma: no cover - dependency-specific behavior
        ann_error = exc
    try:
        import src.gnn_benchmark  # noqa: F401
    except Exception as exc:  # pragma: no cover - dependency-specific behavior
        gnn_error = exc

    # Either import succeeds or the error explicitly reflects optional deps.
    if ann_error is not None:
        assert "torch" in str(ann_error).lower()
    if gnn_error is not None:
        assert "torch" in str(gnn_error).lower()


def test_baseline_ann_scores_probabilities_when_checkpoint_exists():
    """If an ANN checkpoint exists, baseline ANN scores should be probabilities."""
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch not installed; optional ANN path unavailable")

    from src import baseline_comparison as bc

    model_path = None
    for candidate in [
        os.path.join("models", "ann_30feature_integrated.pt"),
        os.path.join("models", "ann_21feature_legacy.pt"),
    ]:
        if os.path.isfile(candidate):
            model_path = candidate
            break
    if model_path is None:
        pytest.skip("no ANN checkpoint found in models/")

    df = pd.DataFrame(
        [
            {"peptide": "CLGGLLTMV", "presentation_score": 0.9},
            {"peptide": "RAKFKQLL", "presentation_score": 0.4},
            {"peptide": "TIHDIILECV", "presentation_score": 0.7},
        ]
    )
    df = compute_features_for_dataset(df, peptide_col="peptide", binding_col="presentation_score")
    for idx, col in enumerate(BINDING_ALLELE_COLUMNS):
        df[col] = np.clip(df["presentation_score"] - (idx * 0.01), 0.0, 1.0)

    scores = bc._score_with_ann(df, model_path)
    assert scores.shape[0] == len(df)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
