"""
Unit tests for the SESTRAV-VERIFY automated benchmarking and cross-validation suite.
"""

import json
import pytest
import numpy as np
import pandas as pd

from src.verify.sestrav_evaluator import (
    calculate_roc_auc,
    calculate_average_precision,
    mutate_anchors,
    mutate_tcr,
    evaluate_single_virus,
    run_evaluation_pipeline,
)


def test_calculate_roc_auc_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.2, 0.1])
    auc = calculate_roc_auc(y_true, y_scores)
    assert auc == pytest.approx(1.0)


def test_calculate_roc_auc_reversed():
    y_true = np.array([1, 1, 0, 0])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    auc = calculate_roc_auc(y_true, y_scores)
    assert auc == pytest.approx(0.0)


def test_calculate_roc_auc_ties():
    # Tie handling test
    y_true = np.array([1, 0])
    y_scores = np.array([0.5, 0.5])
    auc = calculate_roc_auc(y_true, y_scores)
    assert auc == pytest.approx(0.5)


def test_calculate_average_precision_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.2, 0.1])
    ap = calculate_average_precision(y_true, y_scores)
    assert ap == pytest.approx(1.0)


def test_calculate_average_precision_reversed():
    y_true = np.array([1, 1, 0, 0])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    ap = calculate_average_precision(y_true, y_scores)
    # R: 0.0 -> 0.5 (p=1/3), 0.5 -> 1.0 (p=2/4=0.5)
    # AP = 0.5 * (1/3) + 0.5 * 0.5 = 0.16666666... + 0.25 = 0.41666666...
    assert ap == pytest.approx(5.0 / 12.0)


def test_mutate_anchors():
    pep = "GILGFVFTL"
    mutated = mutate_anchors(pep)
    # Check that second and last positions are mutated to D or K
    assert mutated[0] == pep[0]
    assert mutated[1] in ("D", "K")
    assert mutated[-1] in ("D", "K")
    assert mutated[2:-1] == pep[2:-1]


def test_mutate_tcr():
    pep = "GILGFVFTL"
    mutated = mutate_tcr(pep)
    # Check that P4 and P5 are mutated to P
    assert mutated[0:3] == pep[0:3]
    assert mutated[3] == "P"
    assert mutated[4] == "P"
    assert mutated[5:] == pep[5:]


def test_evaluate_single_virus_mocked():
    df = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL", "FMYSDFHFI", "KLVALGINA", "CINGVCWTV"],
            "allele": ["HLA-A*02:01", "HLA-A*02:01", "HLA-A*02:01", "HLA-A*02:01"],
            "label": [1, 1, 0, 0],
            "protein": ["M1", "PA", "Decoy", "Decoy"],
        }
    )

    res = evaluate_single_virus("InfluenzaA", df, model=None, device="cpu", use_mock=True)
    assert "roc_auc" in res
    assert "prc_auc" in res
    assert res["sample_count"] == 4
    assert "escape_mutant_cross_validation" in res

    esc = res["escape_mutant_cross_validation"]
    assert esc["positive_count"] == 2
    assert esc["anchor_degradation_ratio"] < 1.0
    assert esc["tcr_degradation_ratio"] < 1.0


def test_run_evaluation_pipeline_mocked(tmp_path):
    targets_config = {
        "viruses": {
            "TestVirus": {
                "name": "Test Virus",
                "taxonomy_id": 99999,
                "family": "Testviridae",
                "proteome_fasta": "data/verify/proteomes/sars_cov_2.fasta",
                "validation_out": str(tmp_path / "test_virus_verify.csv"),
                "mhc_alleles": ["HLA-A*02:01"],
            }
        }
    }

    targets_json = tmp_path / "targets.json"
    with open(targets_json, "w") as f:
        json.dump(targets_config, f)

    # Create a dummy validation CSV file
    df_val = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL", "FMYSDFHFI", "KLVALGINA", "CINGVCWTV"],
            "allele": ["HLA-A*02:01", "HLA-A*02:01", "HLA-A*02:01", "HLA-A*02:01"],
            "label": [1, 1, 0, 0],
            "protein": ["M1", "PA", "Decoy", "Decoy"],
        }
    )
    df_val.to_csv(tmp_path / "test_virus_verify.csv", index=False)

    results_dir = tmp_path / "results"

    report = run_evaluation_pipeline(
        targets_json, model_checkpoint_path=None, results_dir=results_dir, use_mock=True
    )

    assert "TestVirus" in report["viral_families"]
    assert report["global_summary"]["total_cohorts"] == 1
    assert (results_dir / "validation_report.json").exists()
