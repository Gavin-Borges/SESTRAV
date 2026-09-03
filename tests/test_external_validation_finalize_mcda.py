"""MCDA tests for measured and unmeasured training overlap."""

import pandas as pd

from src.external_validation_finalize import (
    CONTAMINATION_CAP_PCT,
    REFERENCE_NAME,
    assign_mcda_verdict,
)

TOOL_NAME = "PRIME 2.1"


def _verdict(overlap_pct):
    metrics = pd.DataFrame(
        [
            {"tool": REFERENCE_NAME, "auc_pr": 0.80, "issr_10": 0.80},
            {"tool": TOOL_NAME, "auc_pr": 0.90, "issr_10": 0.90},
        ]
    )
    fdr = pd.DataFrame(
        [
            {"tool": TOOL_NAME, "metric": "auc_pr", "fdr_significant": True},
            {"tool": TOOL_NAME, "metric": "issr_10", "fdr_significant": True},
        ]
    )
    bootstrap = pd.DataFrame(
        [
            {
                "comparison": f"{REFERENCE_NAME} vs {TOOL_NAME}",
                "auc_pr_significant": "yes",
                "issr10_significant": "yes",
            }
        ]
    )
    return assign_mcda_verdict(TOOL_NAME, metrics, fdr, bootstrap, overlap_pct)


def test_unmeasured_overlap_is_skipped_not_clean():
    result = _verdict(None)
    assert result["overlap_contaminated"] is None
    assert result["overlap_status"] == "skipped"
    assert result["verdict"] != "Strongly Better"
    assert "overlap unmeasured" in result["rationale"].lower()


def test_measured_clean_overlap_allows_strongly_better():
    result = _verdict(10.0)
    assert result["overlap_contaminated"] is False
    assert result["overlap_status"] == "pass"
    assert result["verdict"] == "Strongly Better"


def test_measured_contaminated_overlap_is_not_clean():
    result = _verdict(CONTAMINATION_CAP_PCT + 1.0)
    assert result["overlap_contaminated"] is True
    assert result["overlap_status"] == "fail"
    assert result["verdict"] != "Strongly Better"
