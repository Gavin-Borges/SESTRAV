"""Unit tests for the continuous IEDB benchmark logic (Part 11, obj #9).

Covers the baseline-comparison and regression-alerting logic extracted from the
monthly GitHub Actions job into ``src.continuous_validation`` - including the
>3% relative regression threshold, baseline I/O, input cleaning, and the
orchestration that writes the ``REGRESSION_DETECTED`` marker the workflow uses to
open a GitHub Issue. Scoring is injected so no trained model is required.
"""
import json

import pandas as pd
import pytest

from src import continuous_validation as cv


# ---------------------------------------------------------------------------
# compute_regression - the core threshold logic
# ---------------------------------------------------------------------------

def test_no_baseline_is_not_a_regression():
    result = cv.compute_regression(0.50, None)
    assert result["is_regression"] is False
    assert result["delta"] is None
    assert result["rel_delta"] is None
    assert result["baseline_auc_pr"] is None


def test_improvement_is_not_a_regression():
    result = cv.compute_regression(0.80, 0.76)
    assert result["is_regression"] is False
    assert result["delta"] == pytest.approx(0.04)
    assert result["rel_delta"] > 0


def test_small_drop_below_threshold_is_not_a_regression():
    # 2% relative drop (< 3% threshold) → no alert.
    result = cv.compute_regression(0.7635 * 0.98, 0.7635)
    assert result["rel_delta"] == pytest.approx(-0.02, abs=1e-6)
    assert result["is_regression"] is False


def test_large_drop_above_threshold_is_a_regression():
    # 5% relative drop (> 3% threshold) → alert.
    result = cv.compute_regression(0.7635 * 0.95, 0.7635)
    assert result["rel_delta"] == pytest.approx(-0.05, abs=1e-6)
    assert result["is_regression"] is True


def test_drop_just_below_threshold_is_not_a_regression():
    # 2.9% relative drop is strictly less than the 3% threshold - NOT a regression.
    # (Exact boundary testing avoided: 0.7635 * 0.97 / 0.7635 - 1 is floating-point
    # sensitive and rounds to exactly -0.03 ± machine epsilon.)
    result = cv.compute_regression(0.7635 * (1 - 0.029), 0.7635)
    assert result["rel_delta"] == pytest.approx(-0.029, rel=1e-5)
    assert result["is_regression"] is False


def test_custom_threshold_is_honoured():
    result = cv.compute_regression(0.7635 * 0.98, 0.7635, rel_threshold=0.01)
    assert result["is_regression"] is True


# ---------------------------------------------------------------------------
# baseline I/O
# ---------------------------------------------------------------------------

def test_load_baseline_missing_file_returns_empty(tmp_path):
    assert cv.load_baseline(str(tmp_path / "nope.json")) == {}


def test_load_baseline_reads_json(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"iedb_ebv_hpv16_tcell": {"auc_pr": 0.76}}))
    loaded = cv.load_baseline(str(path))
    assert loaded["iedb_ebv_hpv16_tcell"]["auc_pr"] == 0.76


def test_baseline_auc_pr_nested_flat_null_and_missing():
    assert cv.baseline_auc_pr({"iedb_ebv_hpv16_tcell": {"auc_pr": 0.7}}) == 0.7
    assert cv.baseline_auc_pr({"iedb_ebv_hpv16_tcell": 0.65}) == 0.65
    assert cv.baseline_auc_pr({"iedb_ebv_hpv16_tcell": {"auc_pr": None}}) is None
    assert cv.baseline_auc_pr({}) is None


def test_regression_message_is_formatted():
    result = cv.compute_regression(0.70, 0.80)
    msg = cv.regression_message(result)
    assert "current_auc_pr=0.7000" in msg
    assert "baseline_auc_pr=0.8000" in msg
    assert "rel_delta=-12.50%" in msg


# ---------------------------------------------------------------------------
# load_inputs
# ---------------------------------------------------------------------------

def test_load_inputs_missing_files_returns_none(tmp_path):
    assert cv.load_inputs([str(tmp_path / "a.csv"), str(tmp_path / "b.csv")]) is None


def test_load_inputs_concats_dedups_and_casts_label(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    pd.DataFrame({"peptide": ["AAA", "BBB"], "label": [1, 0]}).to_csv(a, index=False)
    # Duplicate peptide AAA plus a NaN-label row that must be dropped.
    pd.DataFrame({"peptide": ["AAA", "CCC"], "label": [1.0, None]}).to_csv(b, index=False)
    df = cv.load_inputs([str(a), str(b)])
    assert set(df["peptide"]) == {"AAA", "BBB"}  # CCC dropped (NaN label), AAA deduped
    assert df["label"].dtype.kind in ("i", "u")


# ---------------------------------------------------------------------------
# run() orchestration - regression marker + results persistence
# ---------------------------------------------------------------------------

def _write_inputs(tmp_path):
    p = tmp_path / "iedb.csv"
    pd.DataFrame({"peptide": ["AAA", "BBB", "CCC"], "label": [1, 0, 1]}).to_csv(p, index=False)
    return [str(p)]


def test_run_empty_inputs_exits_cleanly(tmp_path):
    code = cv.run([str(tmp_path / "absent.csv")], results_dir=str(tmp_path / "out"))
    assert code == 0
    assert not (tmp_path / "out").exists()


def test_run_writes_marker_on_regression(tmp_path):
    baseline = tmp_path / "baselines.json"
    baseline.write_text(json.dumps({cv.BASELINE_KEY: {"auc_pr": 0.80}}))
    marker = tmp_path / "REGRESSION_DETECTED"
    results = tmp_path / "out"

    code = cv.run(
        _write_inputs(tmp_path),
        baseline_path=str(baseline),
        results_dir=str(results),
        score_fn=lambda df, m, b: {"auc_pr": 0.70, "n_peptides": len(df), "n_positive": 2},
        marker_path=str(marker),
        today="2026-06-22",
    )
    assert code == 0
    assert marker.exists()  # 12.5% relative drop > 3% → regression flagged
    payload = json.loads((results / "benchmark_latest.json").read_text())
    assert payload["regression"]["is_regression"] is True
    assert (results / "benchmark_2026-06-22.json").exists()


def test_run_no_marker_when_within_threshold(tmp_path):
    baseline = tmp_path / "baselines.json"
    baseline.write_text(json.dumps({cv.BASELINE_KEY: {"auc_pr": 0.80}}))
    marker = tmp_path / "REGRESSION_DETECTED"
    results = tmp_path / "out"

    code = cv.run(
        _write_inputs(tmp_path),
        baseline_path=str(baseline),
        results_dir=str(results),
        score_fn=lambda df, m, b: {"auc_pr": 0.79, "n_peptides": len(df), "n_positive": 2},
        marker_path=str(marker),
        today="2026-06-22",
    )
    assert code == 0
    assert not marker.exists()  # 1.25% drop < 3% → no alert
    assert (results / "benchmark_latest.json").exists()


def test_run_seeds_when_no_baseline(tmp_path):
    marker = tmp_path / "REGRESSION_DETECTED"
    results = tmp_path / "out"

    code = cv.run(
        _write_inputs(tmp_path),
        baseline_path=str(tmp_path / "absent.json"),
        results_dir=str(results),
        score_fn=lambda df, m, b: {"auc_pr": 0.50, "n_peptides": len(df), "n_positive": 2},
        marker_path=str(marker),
        today="2026-06-22",
    )
    assert code == 0
    assert not marker.exists()  # no baseline yet → seed, never a regression
    payload = json.loads((results / "benchmark_latest.json").read_text())
    assert payload["regression"]["baseline_auc_pr"] is None
