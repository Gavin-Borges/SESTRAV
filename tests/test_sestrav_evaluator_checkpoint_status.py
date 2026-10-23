"""run_evaluation_pipeline reports what happened to its GNN checkpoint.

Why this test exists
--------------------
With no checkpoint the evaluator scores with an INITIALIZED, untrained
structural GNN. Those are real model scores, not mock ones, so the report's
`use_mock_fallback` is False - and until this test's change nothing else in
the report said the weights were random (to-do PC-2, finding A7). A reader of
validation_report.json could not tell a trained run from an untrained one, nor
a requested checkpoint that was missing from one that was never asked for.

Each arm is now reported on its own: `checkpoint_status` is "loaded",
"absent", "failed" or "not_attempted"; `checkpoint_requested` is the path the
caller passed (reduced as below), or null; `checkpoint_loaded` is that path only when it loaded;
and `untrained_baseline` is true when the run built an initialized GNN with no
checkpoint to load (a virus that then fell back to mock scores says so in its
own `used_mock_fallback`). Both paths go through
`src.provenance_paths.relative_source` - repo-relative inside the checkout,
`external:<name>` outside it - so the report carries no workstation path
whatever the caller passed. The failure POLICY (warn and fall back, or raise) is
unchanged and is an owner decision (PC-7).
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.verify.sestrav_evaluator import HAS_PYG, run_evaluation_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(not HAS_PYG, reason="requires torch_geometric")

_AAS = "ACDEFGHIKLMNPQRSTVWY"


def _targets(tmp_path):
    peps = ["SLLMWITQ" + _AAS[i] for i in range(8)]
    val_csv = tmp_path / "ebv.csv"
    pd.DataFrame(
        {"peptide": peps, "allele": ["HLA-A*02:01"] * 8, "label": [1, 0] * 4}
    ).to_csv(val_csv, index=False)
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps({"viruses": {"EBV": {"validation_out": str(val_csv)}}}))
    return targets


def _run(tmp_path, **kwargs):
    report = run_evaluation_pipeline(_targets(tmp_path), results_dir=tmp_path / "results", **kwargs)
    written = json.loads((tmp_path / "results" / "validation_report.json").read_text())
    assert written["metadata"] == report["metadata"], "the JSON on disk must carry the same fields"
    recorded = json.dumps(written["metadata"])
    for spelling in {str(tmp_path), tmp_path.as_posix(), json.dumps(str(tmp_path))[1:-1]}:
        assert spelling not in recorded, "a workstation path reached the report"
    return report["metadata"]


def test_no_checkpoint_is_reported_as_an_untrained_baseline(tmp_path, capsys):
    meta = _run(tmp_path, use_mock=False)
    assert "GNN checkpoint: absent (untrained baseline)" in capsys.readouterr().out
    assert meta["checkpoint_status"] == "absent"
    assert meta["checkpoint_requested"] is None
    assert meta["checkpoint_loaded"] is None
    assert meta["untrained_baseline"] is True
    assert meta["use_mock_fallback"] is False  # real (untrained) GNN scores, not mock


def test_a_requested_checkpoint_that_is_missing_is_named(tmp_path):
    missing = tmp_path / "not_there.pth"
    meta = _run(tmp_path, model_checkpoint_path=missing, use_mock=False)
    assert meta["checkpoint_status"] == "absent"
    assert meta["checkpoint_requested"] == "external:not_there.pth"
    assert meta["checkpoint_loaded"] is None
    assert meta["untrained_baseline"] is True


def test_a_requested_checkpoint_inside_the_checkout_is_recorded_repo_relative(tmp_path):
    absent = REPO_ROOT / "models" / "gnn" / "no_such_checkpoint_for_this_test.pth"
    assert not absent.exists()
    meta = _run(tmp_path, model_checkpoint_path=absent, use_mock=False)
    assert meta["checkpoint_status"] == "absent"
    assert meta["checkpoint_requested"] == "models/gnn/no_such_checkpoint_for_this_test.pth"


def test_a_loaded_checkpoint_is_named_and_not_a_baseline(tmp_path):
    from src.verify.structural_gnn import StructuralGNN

    ckpt = tmp_path / "gnn.pth"
    torch.save({"model_state_dict": StructuralGNN().state_dict()}, ckpt)  # nosec B614
    meta = _run(tmp_path, model_checkpoint_path=ckpt, use_mock=False)
    assert meta["checkpoint_status"] == "loaded"
    assert meta["checkpoint_requested"] == "external:gnn.pth"
    assert meta["checkpoint_loaded"] == "external:gnn.pth"
    assert meta["untrained_baseline"] is False
    assert meta["use_mock_fallback"] is False


def test_an_incompatible_checkpoint_is_reported_as_failed(tmp_path):
    ckpt = tmp_path / "wrong_shape.pth"
    torch.save({"model_state_dict": {"not_a_layer.weight": torch.zeros(1)}}, ckpt)  # nosec B614
    meta = _run(tmp_path, model_checkpoint_path=ckpt, use_mock=False)
    assert meta["checkpoint_status"] == "failed"
    assert meta["checkpoint_requested"] == "external:wrong_shape.pth"
    assert meta["checkpoint_loaded"] is None
    assert meta["untrained_baseline"] is False
    assert meta["use_mock_fallback"] is True


def test_a_mock_run_does_not_attempt_a_checkpoint(tmp_path):
    meta = _run(tmp_path, use_mock=True)
    assert meta["checkpoint_status"] == "not_attempted"
    assert meta["checkpoint_loaded"] is None
    assert meta["untrained_baseline"] is False
    assert meta["use_mock_fallback"] is True
