"""Extended tests for src/verify/sestrav_evaluator.py.

Covers the uncovered paths not reached by test_sestrav_evaluator.py:
  - calculate_average_precision with n_pos=0 (early return)
  - run_mock_predictions with is_mutated / mutation_type variants
  - evaluate_single_virus with empty DataFrame
  - evaluate_single_virus with all-negative cohort (no positives)
  - run_evaluation_pipeline when val_csv is absent (auto-mock extraction)
  - _load_torch_checkpoint success and fallback paths
  - main() CLI entry point
"""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.verify.sestrav_evaluator import (
    _load_torch_checkpoint,
    calculate_average_precision,
    calculate_roc_auc,
    evaluate_single_virus,
    run_evaluation_pipeline,
    run_mock_predictions,
)


# ---------------------------------------------------------------------------
# calculate_average_precision — n_pos=0 early return (line 76)
# ---------------------------------------------------------------------------

def test_calculate_average_precision_all_negative():
    y_true = np.array([0, 0, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.2, 0.1])
    ap = calculate_average_precision(y_true, y_scores)
    assert ap == 0.0


def test_calculate_roc_auc_all_same_class():
    y_true = np.array([1, 1, 1])
    y_scores = np.array([0.9, 0.8, 0.7])
    auc = calculate_roc_auc(y_true, y_scores)
    assert auc == 0.0


# ---------------------------------------------------------------------------
# run_mock_predictions — mutation type variants (lines 138-143)
# ---------------------------------------------------------------------------

class TestRunMockPredictions:
    def _df(self, n=4):
        return pd.DataFrame({
            "peptide": ["GILGFVFTL", "FMYSDFHFI", "KLVALGINA", "ACDEFGHIK"][:n],
            "label":   [1, 1, 0, 0][:n],
        })

    def test_no_mutation_scores_in_range(self):
        scores = run_mock_predictions(self._df())
        assert scores.shape == (4,)
        assert (scores >= 0.01).all() and (scores <= 0.99).all()

    def test_anchor_mutation_lowers_score(self):
        df = self._df()
        base = run_mock_predictions(df)
        mutated = run_mock_predictions(df, is_mutated=True, mutation_type="anchor")
        assert (mutated < base).all()

    def test_tcr_mutation_lowers_score(self):
        df = self._df()
        base = run_mock_predictions(df)
        mutated = run_mock_predictions(df, is_mutated=True, mutation_type="tcr")
        assert (mutated < base).all()

    def test_is_mutated_no_type_unchanged_penalty(self):
        df = self._df()
        # is_mutated=True but mutation_type=None → no branch taken (score unchanged)
        base = run_mock_predictions(df)
        mutated = run_mock_predictions(df, is_mutated=True, mutation_type=None)
        np.testing.assert_array_equal(base, mutated)


# ---------------------------------------------------------------------------
# evaluate_single_virus — empty DataFrame (lines 174-175)
# ---------------------------------------------------------------------------

def test_evaluate_single_virus_empty_df():
    df = pd.DataFrame(columns=["peptide", "label", "allele", "protein"])
    result = evaluate_single_virus("TestVirus", df, model=None,
                                   device=torch.device("cpu"), use_mock=True)
    assert result == {}


# ---------------------------------------------------------------------------
# evaluate_single_virus — all-negative cohort (no positives for mutant CV)
# ---------------------------------------------------------------------------

def test_evaluate_single_virus_no_positives():
    df = pd.DataFrame({
        "peptide": ["KLVALGINA", "CINGVCWTV", "ACDEFGHIK"],
        "label":   [0, 0, 0],
        "allele":  ["HLA-A*02:01"] * 3,
        "protein": ["Decoy"] * 3,
    })
    result = evaluate_single_virus("TestVirus", df, model=None,
                                   device=torch.device("cpu"), use_mock=True)
    assert "roc_auc" in result
    esc = result["escape_mutant_cross_validation"]
    assert esc["positive_count"] == 0
    assert esc["anchor_sensitivity_success_rate"] == 1.0


# ---------------------------------------------------------------------------
# run_evaluation_pipeline — val_csv absent → auto-mock extraction
# (lines 333-335)
# ---------------------------------------------------------------------------

def test_run_evaluation_pipeline_auto_mock_when_csv_missing(tmp_path):
    targets = {
        "viruses": {
            "InfluenzaA": {
                "taxonomy_id": 11520,
                "mhc_alleles": ["HLA-A*02:01"],
                "proteome_fasta": str(tmp_path / "nonexistent.fasta"),
                "validation_out": str(tmp_path / "flu_verify.csv"),
            }
        }
    }
    targets_json = tmp_path / "targets.json"
    targets_json.write_text(json.dumps(targets))

    # NOTE: val_csv is NOT pre-created — pipeline must auto-generate it
    report = run_evaluation_pipeline(
        targets_json,
        model_checkpoint_path=None,
        results_dir=tmp_path / "results",
        use_mock=True,
    )
    assert "InfluenzaA" in report["viral_families"]
    assert report["global_summary"]["total_cohorts"] == 1


# ---------------------------------------------------------------------------
# run_evaluation_pipeline — empty viruses block (graceful)
# ---------------------------------------------------------------------------

def test_run_evaluation_pipeline_empty_viruses(tmp_path):
    targets = {"viruses": {}}
    targets_json = tmp_path / "targets.json"
    targets_json.write_text(json.dumps(targets))
    report = run_evaluation_pipeline(targets_json, results_dir=tmp_path / "results",
                                     use_mock=True)
    assert report["global_summary"]["total_cohorts"] == 0
    assert report["global_summary"]["mean_roc_auc"] == 0.0


# ---------------------------------------------------------------------------
# _load_torch_checkpoint — success path (lines 268-281)
# ---------------------------------------------------------------------------

def test_load_torch_checkpoint_success(tmp_path):
    chk = tmp_path / "model.pth"
    state = {"weight": torch.tensor([1.0, 2.0])}
    torch.save(state, chk)  # nosec B614 — test fixture, trusted tensor

    loaded = _load_torch_checkpoint(chk, torch.device("cpu"))
    assert "weight" in loaded
    torch.testing.assert_close(loaded["weight"], state["weight"])


def test_load_torch_checkpoint_fallback_on_weights_only_failure(tmp_path):
    chk = tmp_path / "model.pth"
    state = {"w": torch.tensor([3.0])}
    torch.save(state, chk)  # nosec B614 — test fixture, trusted tensor

    with patch("torch.load", side_effect=[RuntimeError("weights_only failed"), state]):
        loaded = _load_torch_checkpoint(chk, torch.device("cpu"))
    assert loaded is state


def test_load_torch_checkpoint_raises_on_complete_failure(tmp_path):
    chk = tmp_path / "model.pth"
    chk.write_bytes(b"not-a-valid-checkpoint")

    with pytest.raises(RuntimeError, match="Failed to load"):
        _load_torch_checkpoint(chk, torch.device("cpu"))


# ---------------------------------------------------------------------------
# main() CLI entry point
# ---------------------------------------------------------------------------

def test_main_runs_with_mock_flag(tmp_path):
    from src.verify.sestrav_evaluator import main

    targets = {
        "viruses": {
            "InfluenzaA": {
                "taxonomy_id": 11520,
                "mhc_alleles": ["HLA-A*02:01"],
                "proteome_fasta": str(tmp_path / "nonexistent.fasta"),
                "validation_out": str(tmp_path / "flu_verify.csv"),
            }
        }
    }
    cfg = tmp_path / "targets.json"
    cfg.write_text(json.dumps(targets))

    with patch("sys.argv", ["sestrav_evaluator.py", str(cfg), "--mock"]):
        main()


def test_main_exits_on_missing_targets(tmp_path):
    from src.verify.sestrav_evaluator import main

    with (
        patch("sys.argv", ["sestrav_evaluator.py",
                            str(tmp_path / "nonexistent.json")]),
        pytest.raises(SystemExit),
    ):
        main()
