"""The v1 GNN trainer must not leak autograd anomaly detection into the process.

torch.autograd.set_detect_anomaly flips a PROCESS-GLOBAL flag, and torch enables
it in the constructor rather than in __enter__, so the bare call in train_gnn
switched anomaly detection on for every later autograd user in the same
interpreter. Commit b73ce33 removed that call from train_gnn_v2 for a 25-30
percent slowdown but deliberately kept it in v1, so the fix scopes the flag
instead of deleting it: v1 still trains with anomaly detection on, and nothing
downstream inherits it.

These tests run the real entry point but never train. train_gnn enables the flag
one line before it calls the --model-dir overwrite guard, so a colliding sentinel
file is enough to exercise the whole enable-and-restore path with no dataset, no
ESM-2 embeddings and no GPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="GNN modules require torch")

import src.train_gnn as tg  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_anomaly_flag():
    """Pin a known start state and restore it, so these tests neither inherit nor leak."""
    previous = torch.is_anomaly_enabled()
    torch.set_anomaly_enabled(False)
    try:
        yield
    finally:
        torch.set_anomaly_enabled(previous)


def _colliding_model_dir(tmp_path):
    """A model_dir holding one planned v1 artifact, which makes the guard abort."""
    model_dir = tmp_path / "gnn"
    model_dir.mkdir()
    (model_dir / "structural_gnn_v2.pth").write_bytes(b"published")
    return model_dir


def test_aborted_v1_run_leaves_anomaly_detection_off(tmp_path):
    """The regression: a run that did no training at all still poisoned the process."""
    model_dir = _colliding_model_dir(tmp_path)
    assert torch.is_anomaly_enabled() is False

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        tg.train_gnn(str(tmp_path / "does_not_exist.csv"), model_dir=str(model_dir))

    assert torch.is_anomaly_enabled() is False, (
        "train_gnn leaked the process-global autograd anomaly flag; a train_gnn_v2 "
        "run later in this process would inherit the slowdown b73ce33 removed"
    )


def test_v1_run_still_enables_anomaly_detection_while_it_runs(tmp_path, monkeypatch):
    """Pins that the flag is SCOPED, not deleted. Deleting it is an owner decision."""
    seen = {}

    def spy(model_dir, pooling, architecture, allow_overwrite):
        seen["enabled"] = torch.is_anomaly_enabled()
        raise FileExistsError("Refusing to overwrite (stubbed guard)")

    monkeypatch.setattr(tg, "_guard_output_dir", spy)

    with pytest.raises(FileExistsError):
        tg.train_gnn(str(tmp_path / "does_not_exist.csv"), model_dir=str(tmp_path / "gnn"))

    assert seen["enabled"] is True, "v1 lost the anomaly detection b73ce33 kept on purpose"


def test_a_callers_enabled_flag_is_restored_not_cleared(tmp_path):
    """Restore the caller's mode on the way out; do not hardcode False."""
    model_dir = _colliding_model_dir(tmp_path)
    torch.set_anomaly_enabled(True)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        tg.train_gnn(str(tmp_path / "does_not_exist.csv"), model_dir=str(model_dir))

    assert torch.is_anomaly_enabled() is True
