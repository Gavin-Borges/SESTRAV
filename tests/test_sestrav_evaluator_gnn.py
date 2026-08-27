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
    return pd.DataFrame({"peptide": peps, "allele": ["HLA-A*02:01"] * n, "label": labels})


# ---------------------------------------------------------------------------
# mutate_* - short-peptide (no-op) branches (110->113, 120->123)
# ---------------------------------------------------------------------------


def test_mutate_anchors_too_short_unchanged():
    assert mutate_anchors("A") == "A"


def test_mutate_tcr_too_short_unchanged():
    assert mutate_tcr("ABC") == "ABC"


# ---------------------------------------------------------------------------
# evaluate_single_virus - live GNN path (predict_dataset + breakout)
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
    # A clean real-GNN run must report that it never fell back.
    assert res["used_mock_fallback"] is False


def test_evaluate_single_virus_reports_unplanned_fallback(monkeypatch):
    """Regression pin: an exception mid-real-GNN-attempt must be visible in
    the result, not merely swallowed into mock scores that look identical to
    a deliberately-requested mock run.

    Before this fix, evaluate_single_virus's except arms fell back to mock
    predictions silently - the caller had no way to distinguish "asked for
    mock" from "real GNN attempt failed and used mock instead", and
    run_evaluation_pipeline's top-level use_mock_fallback flag stayed False
    in the second case even though every score in the report was mock.
    """
    import src.verify.sestrav_evaluator as ev
    from src.verify.structural_gnn import StructuralGNN

    def _boom(df):
        raise RuntimeError("simulated dataset construction failure")

    monkeypatch.setattr(ev, "StructuralPeptideMHCDataset", _boom)

    df = _cohort()
    model = StructuralGNN()
    res = evaluate_single_virus("EBV", df, model=model, device=_CPU, use_mock=False)

    assert res["used_mock_fallback"] is True
    # The metric machinery still ran, over mock scores - the point is that
    # the caller now KNOWS that, not that scoring stopped.
    assert "roc_auc" in res


def test_evaluate_single_virus_reports_breakout_only_fallback(monkeypatch):
    """Regression pin for the SECOND except arm specifically (the breakout-
    mutant path), independent of the main-cohort arm pinned above.

    The main cohort succeeds normally here - only the three breakout dataset
    constructions (ds_wt/ds_anchor/ds_tcr) fail. Deleting only the breakout
    arm's `used_mock_fallback = True` must fail this test even though the
    main-cohort arm's assignment (and its own test) are both untouched.
    """
    import src.verify.sestrav_evaluator as ev
    from src.verify.structural_gnn import StructuralGNN, StructuralPeptideMHCDataset

    call_count = {"n": 0}

    def _fail_after_first(df):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return StructuralPeptideMHCDataset(df)
        raise RuntimeError("simulated breakout dataset construction failure")

    monkeypatch.setattr(ev, "StructuralPeptideMHCDataset", _fail_after_first)

    df = _cohort()
    model = StructuralGNN()
    res = evaluate_single_virus("EBV", df, model=model, device=_CPU, use_mock=False)

    assert call_count["n"] > 1, "breakout branch was never reached - test setup is broken"
    assert res["used_mock_fallback"] is True
    esc = res["escape_mutant_cross_validation"]
    assert "mean_wildtype_score" in esc


# ---------------------------------------------------------------------------
# run_evaluation_pipeline - instantiate + load a real checkpoint (303-311)
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
    torch.save({"model_state_dict": StructuralGNN().state_dict()}, ckpt)  # nosec B614

    report = run_evaluation_pipeline(
        targets,
        model_checkpoint_path=ckpt,
        results_dir=tmp_path / "results",
        use_mock=False,
    )
    assert report["metadata"]["use_mock_fallback"] is False
    assert "EBV" in report["viral_families"]
    assert (tmp_path / "results" / "validation_report.json").exists()


def test_run_evaluation_pipeline_reports_per_virus_fallback_in_metadata(tmp_path, monkeypatch):
    """Regression pin for the PIPELINE-level recompute, distinct from the
    evaluate_single_virus-level pin above.

    use_mock=False and a real checkpoint loads successfully, so
    report["metadata"]["use_mock_fallback"] is computed as False before the
    per-virus loop runs - but the one virus in this run fails GNN dataset
    construction and falls back to mock scores. Deleting the pipeline's
    post-loop recompute (the any_virus_fallback OR) leaves the top-level flag
    at its pre-loop value, mislabelling every score in the report as real.
    """
    import src.verify.sestrav_evaluator as ev
    from src.verify.structural_gnn import StructuralGNN

    def _boom(df):
        raise RuntimeError("simulated dataset construction failure")

    monkeypatch.setattr(ev, "StructuralPeptideMHCDataset", _boom)

    targets = _write_targets(tmp_path)
    ckpt = tmp_path / "gnn.pth"
    torch.save({"model_state_dict": StructuralGNN().state_dict()}, ckpt)  # nosec B614

    report = run_evaluation_pipeline(
        targets,
        model_checkpoint_path=ckpt,
        results_dir=tmp_path / "results",
        use_mock=False,
    )
    assert report["viral_families"]["EBV"]["used_mock_fallback"] is True
    assert report["metadata"]["use_mock_fallback"] is True


def test_run_evaluation_pipeline_loads_bare_state_dict(tmp_path):
    from src.verify.structural_gnn import StructuralGNN

    targets = _write_targets(tmp_path)
    ckpt = tmp_path / "gnn_bare.pth"
    torch.save(StructuralGNN().state_dict(), ckpt)  # bare state_dict branch  # nosec B614

    report = run_evaluation_pipeline(
        targets,
        model_checkpoint_path=ckpt,
        results_dir=tmp_path / "results",
        use_mock=False,
    )
    assert "EBV" in report["viral_families"]
