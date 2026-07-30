"""Overwrite-guard tests for src/ann_benchmark.py.

`python -m src.ann_benchmark` used to default `--model-dir` to `models`, the
directory holding the published release artifacts. A benchmark run on a clean
clone therefore rewrote the tracked `models/training_results.csv` in place.
This is the same defect class the train_classifier repair addressed; these
tests pin the fix.

Most cases exercise the guard directly, since a full `train_ann` call needs a
real dataset and a CV run. Two cases go through `train_ann` itself to pin the
wiring: without them, deleting the guard call from `train_ann` would leave
every other test in this file green. Both are cheap because the guard runs
before the dataset is read, so they never reach training.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch", reason="ANN benchmark requires torch")

from src.ann_benchmark import (  # noqa: E402
    _ann_model_stem,
    _guard_output_dir,
    planned_ann_artifact_paths,
    train_ann,
)


def test_model_stem_matches_feature_mode():
    assert _ann_model_stem(30) == "ann_30feature_integrated"
    assert _ann_model_stem(21) == "ann_21feature_legacy"


def test_planned_paths_cover_the_serialized_model():
    paths = [os.path.basename(p) for p in planned_ann_artifact_paths("models", 30)]
    assert "ann_30feature_integrated.pt" in paths


def test_search_output_is_only_planned_when_searching():
    without = [os.path.basename(p) for p in planned_ann_artifact_paths("models", 30, search=False)]
    with_search = [
        os.path.basename(p) for p in planned_ann_artifact_paths("models", 30, search=True)
    ]
    assert "ann_architecture_search.csv" not in without
    assert "ann_architecture_search.csv" in with_search


def test_training_results_is_not_guarded():
    """training_results.csv is merged into, not replaced, so it must stay exempt.

    The train_classifier step of the same run writes it moments earlier and
    the ANN step adds ann_cv_mean/ann_cv_std columns to it. Guarding it would
    make every legitimate RF-then-ANN run into one directory fail.
    """
    for search in (False, True):
        names = [os.path.basename(p) for p in planned_ann_artifact_paths("models", 30, search)]
        assert "training_results.csv" not in names


def test_guard_passes_on_an_empty_directory(tmp_path):
    _guard_output_dir(str(tmp_path), 30, search=False, allow_overwrite=False)


def test_guard_refuses_to_clobber_an_existing_model(tmp_path):
    (tmp_path / "ann_30feature_integrated.pt").write_bytes(b"published")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _guard_output_dir(str(tmp_path), 30, search=False, allow_overwrite=False)

    assert (tmp_path / "ann_30feature_integrated.pt").read_bytes() == b"published"


def test_guard_names_the_blocking_file_and_the_escape_hatch(tmp_path):
    (tmp_path / "ann_21feature_legacy.pt").write_bytes(b"stale")

    with pytest.raises(FileExistsError) as excinfo:
        _guard_output_dir(str(tmp_path), 21, search=False, allow_overwrite=False)

    message = str(excinfo.value)
    assert "ann_21feature_legacy.pt" in message
    assert "--allow-overwrite" in message


def test_allow_overwrite_disarms_the_guard(tmp_path):
    (tmp_path / "ann_30feature_integrated.pt").write_bytes(b"published")
    _guard_output_dir(str(tmp_path), 30, search=False, allow_overwrite=True)


def test_guard_ignores_a_different_feature_modes_artifact(tmp_path):
    """A mode-21 run must not be blocked by a mode-30 model sitting alongside."""
    (tmp_path / "ann_30feature_integrated.pt").write_bytes(b"other mode")
    _guard_output_dir(str(tmp_path), 21, search=False, allow_overwrite=False)


def test_train_ann_requires_an_explicit_model_dir(tmp_path):
    """No default destination: omitting model_dir is a TypeError, not a write to models/.

    The argparse layer is covered by tests/test_entry_point_help_smoke.py; this
    pins the function signature, which a caller could still reach directly.
    """
    with pytest.raises(TypeError):
        train_ann(str(tmp_path / "does_not_exist.csv"))


def test_train_ann_actually_calls_the_guard(tmp_path):
    """The guard must be wired into train_ann, not merely defined beside it.

    Every other test here exercises _guard_output_dir directly and would stay
    green if the call site were deleted. This one goes through the real entry
    point. It never trains: the guard runs before the dataset is read, which is
    why a nonexistent --data path still produces FileExistsError rather than
    FileNotFoundError.
    """
    sentinel = tmp_path / "ann_21feature_legacy.pt"
    sentinel.write_bytes(b"published")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        train_ann(
            str(tmp_path / "does_not_exist.csv"),
            model_dir=str(tmp_path),
            feature_mode=21,
        )

    assert sentinel.read_bytes() == b"published"
