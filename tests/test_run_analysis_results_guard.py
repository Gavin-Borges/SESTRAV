"""Overwrite-guard tests for scripts/run_analysis.py.

Third confirmed instance of the silent-overwrite defect class for results/
(see tests/test_final_validation_results_guard.py's module docstring for the
full context of this line of PRs). Split into its own file because importing
scripts.run_analysis pulls in src.shap_analysis, which imports the `shap`
library - on at least one Windows dev machine in this project this crashes the
Python process natively (Windows fatal exception 0xc06d007f inside
scipy.linalg.inv, called from shap/plots/colors/_colorconv.py at import time,
confirmed to reproduce identically on unmodified `python -m src.shap_analysis
--help`, i.e. pre-existing and unrelated to this defect-class fix). Isolating
these cases into their own file means that crash cannot block collection of
the sibling file's tests, which do not import shap.

run_analysis.py's guard has to know about SHAP's delegated writes
(src/shap_analysis.py's run_shap_analysis, called from inside main()) as well
as its own 2 direct CSVs - both are exercised through the same
planned_analysis_paths() the real run uses, so the guard can never drift from
what main() actually writes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_analysis import _guard_results_dir as guard_run_analysis, planned_analysis_paths

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# planned path set
# ---------------------------------------------------------------------------


def test_run_analysis_plans_both_direct_csvs_and_all_shap_outputs(tmp_path):
    names = {p.name for p in map(Path, planned_analysis_paths(str(tmp_path)))}
    assert names == {
        "gold_standard_validation.csv",
        "baseline_comparison.csv",
        "shap_values_rf.csv",
        "shap_values_xgb.csv",
        "shap_summary_rf.png",
        "shap_summary_xgb.png",
        "shap_bar_rf.png",
        "shap_bar_xgb.png",
        "shap_waterfall_top_gs.png",
    }


# ---------------------------------------------------------------------------
# guard behavior
# ---------------------------------------------------------------------------


def test_run_analysis_guard_passes_on_an_empty_directory(tmp_path):
    guard_run_analysis(str(tmp_path), allow_overwrite=False)


def test_run_analysis_guard_refuses_to_clobber_a_tracked_result(tmp_path):
    sentinel = tmp_path / "gold_standard_validation.csv"
    sentinel.write_text("peptide,virus,stage1_found\nAAA,EBV,True\n")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        guard_run_analysis(str(tmp_path), allow_overwrite=False)

    assert "stage1_found" in sentinel.read_text()


def test_run_analysis_guard_catches_a_shap_plot_too(tmp_path):
    """The regression a guard scoped to only the 2 direct CSVs would have missed."""
    sentinel = tmp_path / "shap_summary_rf.png"
    sentinel.write_bytes(b"published")

    with pytest.raises(FileExistsError) as excinfo:
        guard_run_analysis(str(tmp_path), allow_overwrite=False)

    assert "shap_summary_rf.png" in str(excinfo.value)
    assert "--allow-overwrite" in str(excinfo.value)


def test_run_analysis_allow_overwrite_disarms_the_guard(tmp_path):
    (tmp_path / "baseline_comparison.csv").write_text("published")
    guard_run_analysis(str(tmp_path), allow_overwrite=True)


# ---------------------------------------------------------------------------
# CLI - these spawn a subprocess that also imports shap, so they are expected
# to fail the same way locally on an affected machine; CI (Linux) is the
# authoritative check for these two.
# ---------------------------------------------------------------------------


def test_run_analysis_cli_requires_results_dir_explicitly():
    """No default destination: omitting --results-dir must fail argparse, not run."""
    result = subprocess.run(
        [sys.executable, "run_analysis.py"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert "--results-dir" in result.stderr


def test_run_analysis_cli_advertises_allow_overwrite():
    result = subprocess.run(
        [sys.executable, "run_analysis.py", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "--allow-overwrite" in result.stdout
    assert "--results-dir RESULTS_DIR" in result.stdout
    assert "[--results-dir" not in result.stdout
