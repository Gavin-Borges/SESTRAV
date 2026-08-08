"""Overwrite-guard tests for src/train_gnn.py's --model-dir.

train_gnn.py now delegates to the shared src/artifact_guard.py helper (see
that module's docstring), but is deliberately NOT registered in
tests/test_artifact_guard_contract.py's GUARDED_MODULES: its planned paths
include a file in the PARENT of the directory it guards, not just files
inside it. A run training into models/gnn also writes
models/gnn_oof_predictions*.csv one level up, and that file is a tracked
release artifact too, so planned_gnn_artifact_paths() must name it. Two of
the contract file's six checks assume every planned path lives directly
under the given directory (that is what makes "point the flag at a fresh
directory" a complete fix), which is false for this module by design. This
file gives train_gnn.py the same rigor, adapted for that shape - the same
reason test_data_bias_audit_guard.py exists as its own file rather than
joining the contract registry.
"""

from __future__ import annotations

import os

import pytest

from src import train_gnn as tg


def _model_dir(tmp_path):
    """A model_dir one level under tmp_path, so its parent (tmp_path itself)
    is available for the OOF predictions path without escaping the sandbox."""
    d = tmp_path / "gnn_run"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Planned-path enumeration
# ---------------------------------------------------------------------------


def test_planned_paths_v1_includes_parent_oof_file(tmp_path):
    model_dir = _model_dir(tmp_path)
    paths = tg.planned_gnn_artifact_paths(str(model_dir), "mean", "v1")
    names = {os.path.basename(p) for p in paths}
    assert names == {
        "structural_gnn_v2.pth",
        "gnn_scaler.joblib",
        "gnn_oof_predictions.csv",
    }
    oof_path = next(p for p in paths if os.path.basename(p) == "gnn_oof_predictions.csv")
    assert os.path.dirname(oof_path) == str(tmp_path)
    assert not oof_path.startswith(str(model_dir) + os.sep)


def test_planned_paths_v2_mean_pooling_writes_tagged_and_untagged_copies(tmp_path):
    model_dir = _model_dir(tmp_path)
    paths = tg.planned_gnn_artifact_paths(str(model_dir), "mean", "v2")
    names = {os.path.basename(p) for p in paths}
    assert "structural_gnn_v2_mean.pth" in names
    assert "structural_gnn_v2.pth" in names
    assert "gnn_oof_predictions_mean.csv" in names
    assert "gnn_oof_predictions.csv" in names


def test_planned_paths_v2_other_pooling_skips_the_untagged_copy(tmp_path):
    model_dir = _model_dir(tmp_path)
    paths = tg.planned_gnn_artifact_paths(str(model_dir), "max", "v2")
    names = {os.path.basename(p) for p in paths}
    assert "structural_gnn_v2_max.pth" in names
    assert "gnn_oof_predictions_max.csv" in names
    assert "structural_gnn_v2.pth" not in names
    assert "gnn_oof_predictions.csv" not in names


# ---------------------------------------------------------------------------
# Guard behaviour
# ---------------------------------------------------------------------------


def test_guard_is_silent_on_an_empty_directory(tmp_path):
    model_dir = _model_dir(tmp_path)
    tg._guard_output_dir(str(model_dir), "mean", "v1", False)


def test_guard_refuses_when_the_in_directory_artifact_exists(tmp_path):
    model_dir = _model_dir(tmp_path)
    (model_dir / "structural_gnn_v2.pth").write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        tg._guard_output_dir(str(model_dir), "mean", "v1", False)
    assert "structural_gnn_v2.pth" in str(exc.value)


def test_guard_refuses_when_the_parent_oof_file_exists(tmp_path):
    """The hazard this module's guard exists specifically to catch: a stale
    OOF predictions CSV one level above model_dir, with model_dir itself
    still empty."""
    model_dir = _model_dir(tmp_path)
    (tmp_path / "gnn_oof_predictions.csv").write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        tg._guard_output_dir(str(model_dir), "mean", "v1", False)
    assert "gnn_oof_predictions.csv" in str(exc.value)


def test_allow_overwrite_disarms_the_guard_for_both_locations(tmp_path):
    model_dir = _model_dir(tmp_path)
    (model_dir / "structural_gnn_v2.pth").write_text("published", encoding="utf-8")
    (model_dir / "gnn_scaler.joblib").write_text("published", encoding="utf-8")
    (tmp_path / "gnn_oof_predictions.csv").write_text("published", encoding="utf-8")
    tg._guard_output_dir(str(model_dir), "mean", "v1", True)


def test_error_names_the_flag_hint_and_every_collision(tmp_path):
    model_dir = _model_dir(tmp_path)
    (model_dir / "structural_gnn_v2.pth").write_text("published", encoding="utf-8")
    (model_dir / "gnn_scaler.joblib").write_text("published", encoding="utf-8")
    (tmp_path / "gnn_oof_predictions.csv").write_text("published", encoding="utf-8")

    with pytest.raises(FileExistsError) as exc:
        tg._guard_output_dir(str(model_dir), "mean", "v1", False)
    message = str(exc.value)

    assert "--model-dir" in message
    assert "--allow-overwrite" in message
    assert "train_gnn_v2(..., allow_overwrite=True)" in message
    assert "3 existing artifact(s)" in message
    assert "structural_gnn_v2.pth" in message
    assert "gnn_scaler.joblib" in message
    assert "gnn_oof_predictions.csv" in message
    assert "Note that the OOF predictions are written to the parent of --model-dir" in message


# ---------------------------------------------------------------------------
# Wiring: the guard must actually run inside both public entry points
# ---------------------------------------------------------------------------


def test_guard_runs_before_any_write_in_train_gnn_and_train_gnn_v2():
    """A defined-but-uncalled guard would pass every guard-behaviour test above.
    Checked by source order, not a live call: train_gnn_v2 calls set_seed()
    before its guard, so a monkeypatch-and-invoke test would raise on
    set_seed regardless of whether the guard is wired, proving nothing. Source
    order against the first write (os.makedirs) is unambiguous instead."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src" / "train_gnn.py").read_text(
        encoding="utf-8"
    )

    def _guard_precedes_first_makedirs(fn_start: int) -> None:
        guard_pos = source.index("_guard_output_dir(", fn_start)
        makedirs_pos = source.index("os.makedirs(model_dir", fn_start)
        assert guard_pos < makedirs_pos

    train_gnn_start = source.index("def train_gnn(")
    train_gnn_v2_start = source.index("def train_gnn_v2(")
    _guard_precedes_first_makedirs(train_gnn_start)
    _guard_precedes_first_makedirs(train_gnn_v2_start)


def test_cli_advertises_allow_overwrite():
    """--allow-overwrite must be a real argparse flag, not just an internal
    parameter, so a stray import above cannot silently make this vacuous."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "src.train_gnn", "--help"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0
    assert "--allow-overwrite" in result.stdout
    assert "--model-dir" in result.stdout
