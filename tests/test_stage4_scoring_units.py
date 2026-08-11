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
    _resolve_thresholds_path,
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


def test_resolve_thresholds_path_prefers_explicit_config_path(tmp_path):
    """An explicit config path wins over the model-dir convention.

    Before this resolver existed, config.yaml's ``thresholds_path`` was read by
    nothing: the file was located purely by convention from the model directory,
    so repointing the config key was a silent no-op.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "optimal_thresholds.json").write_text(json.dumps({"threshold": 0.5}))
    explicit = tmp_path / "elsewhere.json"
    explicit.write_text(json.dumps({"threshold": 0.9}))

    assert _resolve_thresholds_path(str(model_dir), str(explicit)) == str(explicit)


def test_resolve_thresholds_path_falls_back_to_model_dir(tmp_path):
    """With no explicit path, behaviour is identical to the pre-fix convention."""
    convention = tmp_path / "optimal_thresholds.json"
    convention.write_text(json.dumps({"threshold": 0.5}))

    assert _resolve_thresholds_path(str(tmp_path), None) == str(convention)
    # A configured path that does not exist must not mask the model-dir copy.
    assert _resolve_thresholds_path(str(tmp_path), str(tmp_path / "missing.json")) == str(
        convention
    )


def test_resolve_thresholds_path_returns_none_when_nothing_exists(tmp_path):
    assert _resolve_thresholds_path(str(tmp_path), None) is None


def test_apply_thresholds_honours_explicit_path(tmp_path):
    """The explicit path actually drives the cut, not just the resolver."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "optimal_thresholds.json").write_text(json.dumps({"threshold": 0.5}))
    explicit = tmp_path / "elsewhere.json"
    explicit.write_text(json.dumps({"threshold": 0.9}))

    df = pd.DataFrame({"immunogenicity_score": [0.6, 0.95]})
    _apply_thresholds(df, str(model_dir), str(explicit))
    # Under the model-dir copy (0.5) both rows would be positive.
    assert df["immunogenic"].tolist() == [0, 1]


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
