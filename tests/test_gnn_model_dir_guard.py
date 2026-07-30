"""Overwrite-guard tests for src/train_gnn.py and src/gnn_benchmark.py.

These were the last two entry points carrying the silent-overwrite defect that
the train_classifier / ann_benchmark repairs closed elsewhere. `train_gnn`
defaulted `--model-dir` to `models/gnn` and `gnn_benchmark` defaulted
`--output-dir` to `models`, so a run that omitted the flag rewrote tracked
release artifacts in place.

`train_gnn` is the harder of the two: it writes the OOF predictions into the
PARENT of `model_dir`, so a guard scoped to `model_dir` alone would have missed
`models/gnn_oof_predictions.csv` and `models/gnn_oof_predictions_mean.csv`,
both tracked. The parent-directory cases below pin exactly that.

The guards are exercised directly rather than through a real training run,
which would need ESM-2 embeddings and a GPU budget. Both entry points call the
guard before any training work, so the unit under test is the whole behavior.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch", reason="GNN modules require torch")

from src.train_gnn import (  # noqa: E402
    _guard_output_dir as guard_train_gnn,
    planned_gnn_artifact_paths,
)


# ---------------------------------------------------------------------------
# src/train_gnn.py - planned path set
# ---------------------------------------------------------------------------


def test_v2_mean_run_plans_both_tagged_and_canonical_names():
    """A mean-pool run writes each artifact twice; both names must be guarded."""
    names = [os.path.basename(p) for p in planned_gnn_artifact_paths("models/gnn", "mean", "v2")]
    assert "gnn_config_mean.json" in names
    assert "gnn_config.json" in names
    assert "structural_gnn_v2_mean.pth" in names
    assert "structural_gnn_v2.pth" in names


def test_attention_run_does_not_plan_the_canonical_names():
    """Only mean-pool writes the untagged canonical copies - see the OOF write site."""
    names = [
        os.path.basename(p) for p in planned_gnn_artifact_paths("models/gnn", "attention", "v2")
    ]
    assert "gnn_config_attention.json" in names
    assert "gnn_config.json" not in names
    assert "structural_gnn_v2.pth" not in names


def test_oof_predictions_are_planned_in_the_parent_directory():
    """The OOF CSVs land one level above model_dir, and both are tracked artifacts."""
    paths = planned_gnn_artifact_paths("models/gnn", "mean", "v2")
    oof = [p for p in paths if "oof_predictions" in os.path.basename(p)]
    assert oof, "no OOF path planned"
    for p in oof:
        assert os.path.dirname(p) == "models", f"{p} is not in the parent of model_dir"


def test_scratch_model_dir_also_redirects_the_oof_predictions():
    """A scratch --model-dir must not leave the OOF writes pointing at models/."""
    paths = planned_gnn_artifact_paths(os.path.join("models", "scratch", "run1"), "mean", "v2")
    for p in paths:
        assert "scratch" in p, f"{p} escaped the scratch directory"


def test_v1_architecture_plans_its_own_smaller_set():
    """v1 writes no config, no platt scaler and no pooling-tagged names."""
    names = [os.path.basename(p) for p in planned_gnn_artifact_paths("models/gnn", "mean", "v1")]
    assert "structural_gnn_v2.pth" in names
    assert "gnn_scaler.joblib" in names
    assert "gnn_oof_predictions.csv" in names
    assert "gnn_config.json" not in names
    assert "gnn_platt_scaler.joblib" not in names


# ---------------------------------------------------------------------------
# src/train_gnn.py - guard behavior
# ---------------------------------------------------------------------------


def test_train_gnn_guard_passes_on_an_empty_directory(tmp_path):
    model_dir = tmp_path / "gnn"
    model_dir.mkdir()
    guard_train_gnn(str(model_dir), "mean", "v2", allow_overwrite=False)


def test_train_gnn_guard_refuses_to_clobber_a_tracked_config(tmp_path):
    model_dir = tmp_path / "gnn"
    model_dir.mkdir()
    (model_dir / "gnn_config.json").write_text('{"published": true}')

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        guard_train_gnn(str(model_dir), "mean", "v2", allow_overwrite=False)

    assert (model_dir / "gnn_config.json").read_text() == '{"published": true}'


def test_train_gnn_guard_catches_an_oof_file_in_the_parent(tmp_path):
    """The regression that a model_dir-only guard would have missed."""
    model_dir = tmp_path / "gnn"
    model_dir.mkdir()
    sentinel = tmp_path / "gnn_oof_predictions.csv"
    sentinel.write_text("peptide,label,gnn_oof_score\nAAA,1,0.99\n")

    with pytest.raises(FileExistsError) as excinfo:
        guard_train_gnn(str(model_dir), "mean", "v2", allow_overwrite=False)

    assert "gnn_oof_predictions.csv" in str(excinfo.value)
    assert "0.99" in sentinel.read_text()


def test_train_gnn_guard_names_the_blocking_file_and_the_escape_hatch(tmp_path):
    model_dir = tmp_path / "gnn"
    model_dir.mkdir()
    (model_dir / "structural_gnn_v2.pth").write_bytes(b"published")

    with pytest.raises(FileExistsError) as excinfo:
        guard_train_gnn(str(model_dir), "mean", "v2", allow_overwrite=False)

    message = str(excinfo.value)
    assert "structural_gnn_v2.pth" in message
    assert "--allow-overwrite" in message


def test_train_gnn_allow_overwrite_disarms_the_guard(tmp_path):
    model_dir = tmp_path / "gnn"
    model_dir.mkdir()
    (model_dir / "gnn_config.json").write_text("{}")
    guard_train_gnn(str(model_dir), "mean", "v2", allow_overwrite=True)


def test_attention_run_is_not_blocked_by_the_canonical_artifacts(tmp_path):
    """An attention-pool experiment writes only tagged names, so it must not abort."""
    model_dir = tmp_path / "gnn"
    model_dir.mkdir()
    (model_dir / "gnn_config.json").write_text("{}")
    (model_dir / "structural_gnn_v2.pth").write_bytes(b"published")

    guard_train_gnn(str(model_dir), "attention", "v2", allow_overwrite=False)


# ---------------------------------------------------------------------------
# src/train_gnn.py - the guard is actually wired into the entry points
# ---------------------------------------------------------------------------
#
# Every case above calls _guard_output_dir directly and would stay green if the
# call site were deleted. These two go through the real training functions. They
# are cheap because the guard runs before the dataset is read, so a nonexistent
# --data path never costs anything.


@pytest.mark.parametrize(
    ("func_name", "kwargs"),
    [
        ("train_gnn", {}),
        ("train_gnn_v2", {"pooling": "mean"}),
    ],
)
def test_guard_is_wired_into_the_training_entry_points(tmp_path, func_name, kwargs):
    pytest.importorskip("torch_geometric", reason="train_gnn_v2 imports torch_geometric")
    import src.train_gnn as tg

    func = getattr(tg, func_name)
    model_dir = tmp_path / "gnn"
    model_dir.mkdir()
    sentinel = model_dir / "structural_gnn_v2.pth"
    sentinel.write_bytes(b"published")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        func(str(tmp_path / "does_not_exist.csv"), model_dir=str(model_dir), **kwargs)

    assert sentinel.read_bytes() == b"published"


@pytest.mark.parametrize("func_name", ["train_gnn", "train_gnn_v2"])
def test_training_entry_points_require_an_explicit_model_dir(tmp_path, func_name):
    """No default destination: omitting model_dir is a TypeError, not a write to models/gnn."""
    import src.train_gnn as tg

    with pytest.raises(TypeError):
        getattr(tg, func_name)(str(tmp_path / "does_not_exist.csv"))


# ---------------------------------------------------------------------------
# src/gnn_benchmark.py
# ---------------------------------------------------------------------------


def _benchmark_module():
    return pytest.importorskip("src.gnn_benchmark", reason="gnn_benchmark requires the gnn extra")


def test_benchmark_plans_both_result_csvs():
    mod = _benchmark_module()
    names = [os.path.basename(p) for p in mod.planned_benchmark_paths("models", True)]
    assert names == ["gnn_sequence_benchmark.csv", "gnn_bipartite_benchmark.csv"]


def test_benchmark_does_not_plan_bipartite_when_it_will_not_be_written():
    """Guarding a file the run would skip would abort a legitimate run."""
    mod = _benchmark_module()
    names = [os.path.basename(p) for p in mod.planned_benchmark_paths("models", False)]
    assert names == ["gnn_sequence_benchmark.csv"]


def test_writes_bipartite_mirrors_the_real_write_condition(tmp_path):
    mod = _benchmark_module()
    matrix = tmp_path / "binding.csv"

    # Missing matrix -> no bipartite write, whatever the flag says.
    assert mod._writes_bipartite(False, str(matrix)) is False
    matrix.write_text("peptide\n")
    assert mod._writes_bipartite(False, str(matrix)) is True
    assert mod._writes_bipartite(True, str(matrix)) is False


def test_benchmark_guard_refuses_to_clobber_a_tracked_result(tmp_path):
    mod = _benchmark_module()
    sentinel = tmp_path / "gnn_sequence_benchmark.csv"
    sentinel.write_text("model,auc_pr\nGCN,0.7781\n")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        mod._guard_output_dir(str(tmp_path), writes_bipartite=False, allow_overwrite=False)

    assert "0.7781" in sentinel.read_text()


def test_benchmark_guard_passes_on_an_empty_directory(tmp_path):
    mod = _benchmark_module()
    mod._guard_output_dir(str(tmp_path), writes_bipartite=True, allow_overwrite=False)


def test_benchmark_allow_overwrite_disarms_the_guard(tmp_path):
    mod = _benchmark_module()
    (tmp_path / "gnn_sequence_benchmark.csv").write_text("model,auc_pr\n")
    mod._guard_output_dir(str(tmp_path), writes_bipartite=True, allow_overwrite=True)
