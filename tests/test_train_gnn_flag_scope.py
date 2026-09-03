"""Neither GNN entry point may leak a process-global torch flag into the interpreter.

set_seed deliberately leaves cudnn.deterministic, cudnn.benchmark and
deterministic-algorithms mode set, because a caller asking for a seeded run
wants them for the whole run. Confining them to that run is
_restore_torch_global_flags' job, and both train_gnn and train_gnn_v2 carry it.

tests/test_train_gnn_anomaly_scope.py covers the anomaly half on v1. This file
covers the determinism and cudnn flags on BOTH entry points. train_gnn_v2
matters most: it was undecorated entirely, so every v2 run left all three flags
altered even though v2 never touches anomaly mode.

These tests run the real entry points but never train. Both call set_seed one
line before the --model-dir overwrite guard, so a colliding sentinel file
exercises the whole set-and-restore path with no dataset, no ESM-2 embeddings
and no GPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="GNN modules require torch")

import src.train_gnn as tg  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_global_flags():
    """Pin a start state set_seed OVERWRITES, then restore it afterwards.

    The pinned values are chosen so every flag differs from what set_seed
    writes: it sets deterministic-algorithms on, cudnn.deterministic True and
    cudnn.benchmark False, so this pins them off, False and True respectively.
    That is what makes the assertions discriminate. An earlier version pinned
    cudnn.benchmark to False, the same value set_seed writes, and the two
    "restores cudnn flags" assertions on it could not tell a real restore from
    set_seed's own write - they passed with the benchmark restore deleted.
    """
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    anomaly = torch.is_anomaly_enabled()
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_anomaly_enabled(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.set_anomaly_enabled(anomaly)


def _colliding_v1_dir(tmp_path):
    """A model_dir holding one planned v1 artifact, which makes the v1 guard abort."""
    model_dir = tmp_path / "gnn"
    model_dir.mkdir()
    (model_dir / "structural_gnn_v2.pth").write_bytes(b"published")
    return model_dir


def _colliding_v2_dir(tmp_path):
    """A model_dir holding one planned v2 mean-pooling artifact."""
    model_dir = tmp_path / "gnn"
    model_dir.mkdir()
    (model_dir / "structural_gnn_v2_mean.pth").write_bytes(b"published")
    return model_dir


def _abort_v1(tmp_path, model_dir):
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        tg.train_gnn(str(tmp_path / "does_not_exist.csv"), model_dir=str(model_dir))


def _abort_v2(tmp_path, model_dir):
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        tg.train_gnn_v2(str(tmp_path / "does_not_exist.csv"), model_dir=str(model_dir))


def test_v1_run_restores_deterministic_algorithms(tmp_path):
    _abort_v1(tmp_path, _colliding_v1_dir(tmp_path))

    assert torch.are_deterministic_algorithms_enabled() is False, (
        "train_gnn leaked deterministic-algorithms mode; every later torch user in "
        "this process would silently inherit it from a run that did no training"
    )


def test_v2_run_restores_deterministic_algorithms(tmp_path):
    """v2 carried no wrapper at all, so it leaked all three flags on every run."""
    _abort_v2(tmp_path, _colliding_v2_dir(tmp_path))

    assert torch.are_deterministic_algorithms_enabled() is False


def test_v1_run_restores_cudnn_flags(tmp_path):
    """Both assertions discriminate: set_seed writes the OPPOSITE of each pinned value."""
    _abort_v1(tmp_path, _colliding_v1_dir(tmp_path))

    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_v2_run_restores_cudnn_flags(tmp_path):
    _abort_v2(tmp_path, _colliding_v2_dir(tmp_path))

    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_a_callers_deterministic_mode_is_restored_not_cleared(tmp_path):
    """Restore what the caller had; do not hardcode False the way a naive reset would.

    warn_only=False is deliberate and load-bearing. set_seed sets
    (True, warn_only=True), so a caller who already had (True, warn_only=True)
    cannot tell a real restore from set_seed's own write, and the test would pass
    with the wrapper removed. Only a state set_seed never produces discriminates.
    """
    torch.use_deterministic_algorithms(True, warn_only=False)
    _abort_v2(tmp_path, _colliding_v2_dir(tmp_path))

    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.is_deterministic_algorithms_warn_only_enabled() is False


def test_v2_run_does_not_enable_anomaly_detection(tmp_path, monkeypatch):
    """b73ce33 removed anomaly mode from v2 for a 25-30 percent speedup. Keep it removed.

    The wrapper only restores; it must never be read as licence to re-enable.
    """
    seen = {}

    def spy(model_dir, pooling, architecture, allow_overwrite):
        seen["enabled"] = torch.is_anomaly_enabled()
        raise FileExistsError("Refusing to overwrite (stubbed guard)")

    monkeypatch.setattr(tg, "_guard_output_dir", spy)

    with pytest.raises(FileExistsError):
        tg.train_gnn_v2(str(tmp_path / "does_not_exist.csv"), model_dir=str(tmp_path / "gnn"))

    assert seen["enabled"] is False
