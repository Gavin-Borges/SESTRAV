"""Unit tests for Stage 4 immunogenicity scoring helpers.

These cover the pure, in-process decision logic on the scoring critical path
(name sanitisation, threshold application, and the no-calibrator branch) without
requiring a trained model or PyTorch. They complement the model-loading and
freeze-mode tests in ``test_freeze_usability_guards.py``.
"""

import json

import numpy as np
import pandas as pd

from functions.stage4_immunogenicity_scoring import (
    _apply_calibration,
    _apply_thresholds,
    _sanitize_name,
)


def test_sanitize_name_allows_safe_chars():
    assert _sanitize_name("HIV-1_gag") == "HIV-1_gag"
    assert _sanitize_name("Proteome123") == "Proteome123"


def test_sanitize_name_neutralises_path_and_shell_chars():
    # Path-traversal and separators must not survive into a filename.
    assert "/" not in _sanitize_name("../../etc/passwd")
    assert "\\" not in _sanitize_name("a\\b")
    assert _sanitize_name("a b;rm -rf c") == "a_b_rm_-rf_c"
    # Every disallowed character collapses to an underscore (length preserved).
    raw = "x/y:z*?"
    out = _sanitize_name(raw)
    assert len(out) == len(raw)
    assert out == "x_y_z__"


def test_apply_thresholds_no_file_is_noop(tmp_path):
    df = pd.DataFrame({"immunogenicity_score": [0.1, 0.9]})
    _apply_thresholds(df, str(tmp_path))
    assert "immunogenic" not in df.columns


def test_apply_thresholds_uses_threshold_key(tmp_path):
    (tmp_path / "optimal_thresholds.json").write_text(json.dumps({"threshold": 0.5}))
    df = pd.DataFrame({"immunogenicity_score": [0.2, 0.5, 0.8]})
    _apply_thresholds(df, str(tmp_path))
    assert df["immunogenic"].tolist() == [0, 1, 1]  # >= threshold is positive


def test_apply_thresholds_falls_back_to_f1_threshold_key(tmp_path):
    (tmp_path / "optimal_thresholds.json").write_text(json.dumps({"f1_threshold": 0.7}))
    df = pd.DataFrame({"immunogenicity_score": [0.6, 0.7, 0.9]})
    _apply_thresholds(df, str(tmp_path))
    assert df["immunogenic"].tolist() == [0, 1, 1]


def test_apply_thresholds_prefers_calibrated_score_column(tmp_path):
    (tmp_path / "optimal_thresholds.json").write_text(json.dumps({"threshold": 0.5}))
    # Raw score would flip the call; calibrated_score must take precedence.
    df = pd.DataFrame({"immunogenicity_score": [0.9], "calibrated_score": [0.1]})
    _apply_thresholds(df, str(tmp_path))
    assert df["immunogenic"].tolist() == [0]


def test_apply_calibration_without_calibrator_returns_originals(tmp_path):
    scores = np.array([0.1, 0.4, 0.95])
    out, applied = _apply_calibration(scores, str(tmp_path))
    assert applied is False
    assert np.array_equal(out, scores)
