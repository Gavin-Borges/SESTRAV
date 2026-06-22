"""GNN-path coverage for src/verify/sestrav_evaluator.py.

The mock paths, metric math, _load_torch_checkpoint, and main() are already
covered by tests/test_sestrav_evaluator.py and ..._extended.py. This module
exercises the *live* structural-GNN branches that those skip:
  - predict_dataset (PyG DataLoader forward pass)
  - evaluate_single_virus with a real model (main + breakout-mutant prediction)
  - run_evaluation_pipeline instantiating + loading a real checkpoint

The structural GNN derives node/edge features deterministically from sequence
(no ESM model, no network), so these run cheaply on CPU. The whole module is
skipped if torch_geometric is unavailable.
"""
import json

import pandas as pd
import pytest
import torch

from src.verify.sestrav_evaluator import (
    HAS_PYG,
    evaluate_single_virus,
    mutate_anchors,
    mutate_tcr,
    run_evaluation_pipeline,
)

pytestmark = pytest.mark.skipif(not HAS_PYG, reason="requires torch_geometric")

_AAS = "ACDEFGHIKLMNPQRSTVWY"
_CPU = torch.device("cpu")


def _cohort(n=8):
    """Small mixed-label cohort with the columns the dataset requires."""
    peps = ["SLLMWITQ" + _AAS[i % 20] for i in range(n)]  # distinct valid 9-mers
    labels = ([1, 0] * (n // 2 + 1))[:n]
    return pd.DataFrame(
        {"peptide": peps, "allele": ["HLA-A*02:01"] * n, "label": labels}
    )


# ---------------------------------------------------------------------------
# mutate_* — short-peptide (no-op) branches (110->113, 120->123)
# ---------------------------------------------------------------------------


def test_mutate_anchors_too_short_unchanged():
    assert mutate_anchors("A") == "A"


def test_mutate_tcr_too_short_unchanged():
    assert mutate_tcr("ABC") == "ABC"


# ---------------------------------------------------------------------------
# evaluate_single_virus — live GNN path (predict_dataset + breakout)
# ---------------------------------------------------------------------------


def test_evaluate_single_virus_real_gnn():
    from src.verify.structural_gnn import StructuralGNN

    df = _cohort()
    model = StructuralGNN()
    res = evaluate_single_virus("EBV", df, model=model, device=_CPU, use_mock=False)

    assert "roc_auc" in res and "prc_auc" in res
    assert res["sample_count"] == len(df)
    esc = res["escape_mutant_cross_validation"]
    assert esc["positive_count"] == int((df["label"] == 1).sum())
    # Breakout metrics are populated from the live model, not the mock fallback.
    assert "mean_wildtype_score" in esc
    assert 0.0 <= esc["anchor_sensitivity_success_rate"] <= 1.0


# ---------------------------------------------------------------------------
# run_evaluation_pipeline — instantiate + load a real checkpoint (303-311)
# ---------------------------------------------------------------------------


def _write_targets(tmp_path):
    val_csv = tmp_path / "ebv.csv"
    _cohort().to_csv(val_csv, index=False)
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps({"viruses": {"EBV": {"validation_out": str(val_csv)}}}))
    return targets


def test_run_evaluation_pipeline_loads_wrapped_checkpoint(tmp_path):
    from src.verify.structural_gnn import StructuralGNN

    targets = _write_targets(tmp_path)
    ckpt = tmp_path / "gnn_wrapped.pth"
    torch.save({"model_state_dict": StructuralGNN().state_dict()}, ckpt)

    report = run_evaluation_pipeline(
        targets, model_checkpoint_path=ckpt,
        results_dir=tmp_path / "results", use_mock=False,
    )
    assert report["metadata"]["use_mock_fallback"] is False
    assert "EBV" in report["viral_families"]
    assert (tmp_path / "results" / "validation_report.json").exists()


def test_run_evaluation_pipeline_loads_bare_state_dict(tmp_path):
    from src.verify.structural_gnn import StructuralGNN

    targets = _write_targets(tmp_path)
    ckpt = tmp_path / "gnn_bare.pth"
    torch.save(StructuralGNN().state_dict(), ckpt)  # bare state_dict branch

    report = run_evaluation_pipeline(
        targets, model_checkpoint_path=ckpt,
        results_dir=tmp_path / "results", use_mock=False,
    )
    assert "EBV" in report["viral_families"]
