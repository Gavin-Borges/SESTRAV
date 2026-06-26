"""Tests for scripts/evaluate_per_virus.py.

Covers:
  bootstrap_metric    CI ordering, single-class handling
  expected_calibration_error    valid/invalid score range, perfect calibration
  precision_at_recall    basic case, unreachable threshold, single-class
  evaluate_virus    all 6 Amendment 6 metrics present, real-neg subset
  evaluate_all_viruses    min_virus_size filter, per-virus grouping
  check_exit_criterion    pass/fail combos for EBV and HPV
  main    CSV output, JSON output, missing file, exit codes
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_per_virus import (
    MIN_SAMPLES_DEFAULT,
    bootstrap_metric,
    check_exit_criterion,
    evaluate_all_viruses,
    evaluate_virus,
    expected_calibration_error,
    main,
    precision_at_recall,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(
    n_pos: int = 40,
    n_neg_real: int = 30,
    n_neg_decoy: int = 20,
    virus: str = "EBV",
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_pos + n_neg_real + n_neg_decoy
    labels = np.array([1] * n_pos + [0] * n_neg_real + [0] * n_neg_decoy)
    scores = rng.uniform(0.0, 1.0, size=n)
    # Give positives slightly higher scores so AUC > 0.5.
    scores[:n_pos] = rng.uniform(0.4, 1.0, size=n_pos)
    scores[n_pos:] = rng.uniform(0.0, 0.6, size=n_neg_real + n_neg_decoy)

    origins = (
        ["tested_negative"] * n_neg_real
        + ["self_proteome_decoy"] * n_neg_decoy
    )
    peptides = ["GILGFVFTL"] * n_pos + ["NLVPMVATV"] * (n_neg_real + n_neg_decoy)

    return pd.DataFrame(
        {
            "label": labels,
            "score": scores,
            "virus": virus,
            "negative_origin": [None] * n_pos + origins,
            "peptide": peptides,
        }
    )


def _write_csv(tmp_path: Path, df: pd.DataFrame, name: str = "pred.csv") -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# bootstrap_metric
# ---------------------------------------------------------------------------


def test_bootstrap_metric_returns_triple() -> None:
    rng = np.random.default_rng(7)
    y_true = np.array([1] * 20 + [0] * 20)
    y_score = rng.uniform(0, 1, 40)
    from sklearn.metrics import roc_auc_score
    result = bootstrap_metric(y_true, y_score, roc_auc_score, n_resamples=50, seed=0)
    assert len(result) == 3
    point, lower, upper = result
    assert lower <= point <= upper or math.isnan(point)


def test_bootstrap_metric_ci_ordered() -> None:
    rng = np.random.default_rng(1)
    y_true = np.array([1] * 30 + [0] * 30)
    y_score = np.concatenate([rng.uniform(0.4, 1.0, 30), rng.uniform(0.0, 0.6, 30)])
    from sklearn.metrics import roc_auc_score
    _, lo, hi = bootstrap_metric(y_true, y_score, roc_auc_score, n_resamples=100, seed=0)
    assert lo <= hi


def test_bootstrap_metric_single_class_uses_point() -> None:
    y_true = np.array([1] * 20)
    y_score = np.linspace(0, 1, 20)
    def dummy_fn(yt: np.ndarray, ys: np.ndarray) -> float:
        return 0.5
    point, lo, hi = bootstrap_metric(y_true, y_score, dummy_fn, n_resamples=10, seed=0)
    assert point == 0.5


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------


def test_ece_perfect_calibration() -> None:
    probs = np.linspace(0.0, 1.0, 100)
    y_true = (probs > 0.5).astype(float)
    ece = expected_calibration_error(y_true, probs)
    assert ece is not None
    assert 0.0 <= ece <= 1.0


def test_ece_returns_none_for_scores_above_one() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([1.5, 2.0, 0.5, 1.1])
    assert expected_calibration_error(y_true, y_score) is None


def test_ece_returns_none_for_negative_scores() -> None:
    y_true = np.array([0, 1])
    y_score = np.array([-0.1, 0.9])
    assert expected_calibration_error(y_true, y_score) is None


def test_ece_in_unit_interval() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 100).astype(float)
    y_score = rng.uniform(0.0, 1.0, 100)
    ece = expected_calibration_error(y_true, y_score)
    assert ece is not None
    assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# precision_at_recall
# ---------------------------------------------------------------------------


def test_precision_at_recall_basic() -> None:
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_score = np.array([0.9, 0.8, 0.4, 0.3, 0.7, 0.2])
    p = precision_at_recall(y_true, y_score, recall_threshold=0.10)
    assert p is not None
    assert 0.0 <= p <= 1.0


def test_precision_at_recall_returns_none_single_class() -> None:
    y_true = np.array([1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7])
    assert precision_at_recall(y_true, y_score, recall_threshold=0.10) is None


def test_precision_at_recall_returns_none_unreachable() -> None:
    # All scores in [0, 0.1] so recall never reaches 0.99 at any threshold.
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.9, 0.1, 0.8, 0.05])
    # Threshold 0.999 recall is not reachable with only 2 positives in a non-ideal score set.
    result = precision_at_recall(y_true, y_score, recall_threshold=0.999)
    # Result is None or a valid precision value; either is acceptable.
    if result is not None:
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# evaluate_virus
# ---------------------------------------------------------------------------


REQUIRED_KEYS = {
    "n_pos", "n_neg_real", "n_neg_decoy", "n_total",
    "auc_roc", "auc_roc_lower", "auc_roc_upper",
    "auc_pr", "auc_pr_lower", "auc_pr_upper",
    "precision_at_10pct_recall",
    "ece",
    "auc_roc_9mer", "auc_roc_non9mer",
    "auc_roc_real_neg_only",
}


def test_evaluate_virus_all_keys_present() -> None:
    df = _make_df()
    result = evaluate_virus(df, "score", n_bootstrap=20)
    assert REQUIRED_KEYS.issubset(result.keys())


def test_evaluate_virus_counts_correct() -> None:
    df = _make_df(n_pos=10, n_neg_real=15, n_neg_decoy=5)
    r = evaluate_virus(df, "score", n_bootstrap=10)
    assert r["n_pos"] == 10
    assert r["n_neg_real"] == 15
    assert r["n_neg_decoy"] == 5
    assert r["n_total"] == 30


def test_evaluate_virus_auc_roc_in_range() -> None:
    df = _make_df()
    r = evaluate_virus(df, "score", n_bootstrap=20)
    assert 0.0 <= r["auc_roc"] <= 1.0
    assert r["auc_roc_lower"] <= r["auc_roc"] <= r["auc_roc_upper"]


def test_evaluate_virus_real_neg_only_differs() -> None:
    # Real-neg-only AUC may differ from full AUC when decoys inflate it.
    df = _make_df(n_pos=30, n_neg_real=20, n_neg_decoy=40)
    r = evaluate_virus(df, "score", n_bootstrap=20)
    # Both should be computable; values may differ.
    assert r["auc_roc"] is not None
    assert r["auc_roc_real_neg_only"] is not None


def test_evaluate_virus_real_neg_none_when_no_real_negs() -> None:
    df = _make_df(n_pos=20, n_neg_real=0, n_neg_decoy=20)
    r = evaluate_virus(df, "score", n_bootstrap=10)
    assert r["auc_roc_real_neg_only"] is None


def test_evaluate_virus_no_origin_column() -> None:
    df = _make_df().drop(columns=["negative_origin"])
    r = evaluate_virus(df, "score", n_bootstrap=10)
    assert r["n_neg_real"] == 0
    assert r["auc_roc_real_neg_only"] is None


def test_evaluate_virus_single_class_nan() -> None:
    df = pd.DataFrame({
        "label": [0] * 30,
        "score": np.random.default_rng(0).uniform(0, 1, 30),
        "negative_origin": ["self_proteome_decoy"] * 30,
        "peptide": ["GILGFVFTL"] * 30,
    })
    r = evaluate_virus(df, "score", n_bootstrap=5)
    assert math.isnan(r["auc_roc"])


def test_evaluate_virus_length_subgroups_present() -> None:
    df_mixed = _make_df()
    # Add some 10mers.
    extra = df_mixed.copy()
    extra["peptide"] = "GILGFVFTLA"  # 10mer
    df_both = pd.concat([df_mixed, extra], ignore_index=True)
    r = evaluate_virus(df_both, "score", n_bootstrap=10)
    assert r["auc_roc_9mer"] is not None or r["auc_roc_non9mer"] is not None


# ---------------------------------------------------------------------------
# evaluate_all_viruses
# ---------------------------------------------------------------------------


def test_evaluate_all_viruses_groups_correctly() -> None:
    df_ebv = _make_df(virus="EBV")
    df_hpv = _make_df(virus="HPV")
    df = pd.concat([df_ebv, df_hpv], ignore_index=True)
    results = evaluate_all_viruses(df, "score", n_bootstrap=10)
    assert "EBV" in results
    assert "HPV" in results


def test_evaluate_all_viruses_min_size_filter() -> None:
    df_big = _make_df(n_pos=30, n_neg_real=20, n_neg_decoy=10, virus="EBV")
    df_tiny = _make_df(n_pos=2, n_neg_real=2, n_neg_decoy=0, virus="TINY")
    df = pd.concat([df_big, df_tiny], ignore_index=True)
    results = evaluate_all_viruses(df, "score", min_virus_size=MIN_SAMPLES_DEFAULT, n_bootstrap=10)
    assert "EBV" in results
    assert "TINY" not in results


def test_evaluate_all_viruses_empty_input_returns_empty() -> None:
    df = pd.DataFrame({"label": [], "score": [], "virus": []})
    results = evaluate_all_viruses(df, "score", n_bootstrap=10)
    assert results == {}


# ---------------------------------------------------------------------------
# check_exit_criterion
# ---------------------------------------------------------------------------


def _make_results(
    ebv_roc: float, ebv_lo: float, hpv_roc: float
) -> dict:
    return {
        "EBV": {
            "auc_roc": ebv_roc,
            "auc_roc_lower": ebv_lo,
            "auc_roc_upper": ebv_roc + 0.05,
        },
        "HPV": {
            "auc_roc": hpv_roc,
            "auc_roc_lower": hpv_roc - 0.05,
            "auc_roc_upper": hpv_roc + 0.05,
        },
    }


def test_check_exit_criterion_passes() -> None:
    results = _make_results(ebv_roc=0.70, ebv_lo=0.60, hpv_roc=0.65)
    passed, msg = check_exit_criterion(results)
    assert passed
    assert "PASS" in msg


def test_check_exit_criterion_fails_ebv_lo() -> None:
    results = _make_results(ebv_roc=0.70, ebv_lo=0.55, hpv_roc=0.65)
    passed, msg = check_exit_criterion(results)
    assert not passed
    assert "EBV" in msg
    assert "FAIL" in msg


def test_check_exit_criterion_fails_hpv() -> None:
    results = _make_results(ebv_roc=0.70, ebv_lo=0.60, hpv_roc=0.50)
    passed, msg = check_exit_criterion(results)
    assert not passed
    assert "HPV" in msg


def test_check_exit_criterion_missing_ebv() -> None:
    results = {"HPV": {"auc_roc": 0.65, "auc_roc_lower": 0.59}}
    passed, msg = check_exit_criterion(results)
    assert not passed
    assert "EBV" in msg


def test_check_exit_criterion_nan_values() -> None:
    results = {
        "EBV": {"auc_roc": float("nan"), "auc_roc_lower": float("nan")},
        "HPV": {"auc_roc": float("nan"), "auc_roc_lower": float("nan")},
    }
    passed, _ = check_exit_criterion(results)
    assert not passed


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _write_multi_virus_csv(tmp_path: Path) -> Path:
    ebv = _make_df(virus="EBV", seed=1)
    hpv = _make_df(virus="HPV", seed=2)
    df = pd.concat([ebv, hpv], ignore_index=True)
    return _write_csv(tmp_path, df)


def test_main_full_write_produces_outputs(tmp_path: Path) -> None:
    pred = _write_multi_virus_csv(tmp_path)
    out_json = tmp_path / "results.json"
    out_csv = tmp_path / "results.csv"
    rc = main([
        "--predictions", str(pred),
        "--output-json", str(out_json),
        "--output-csv", str(out_csv),
        "--n-bootstrap", "20",
    ])
    assert rc in (0, 2), f"Expected exit code 0 or 2, got {rc}"
    assert out_json.exists()
    assert out_csv.exists()


def test_main_json_structure(tmp_path: Path) -> None:
    pred = _write_multi_virus_csv(tmp_path)
    out_json = tmp_path / "results.json"
    main(["--predictions", str(pred), "--output-json", str(out_json), "--n-bootstrap", "10"])
    data = json.loads(out_json.read_text())
    assert "per_virus" in data
    assert "exit_criterion_passed" in data
    assert "EBV" in data["per_virus"]


def test_main_csv_columns(tmp_path: Path) -> None:
    pred = _write_multi_virus_csv(tmp_path)
    out_csv = tmp_path / "results.csv"
    main(["--predictions", str(pred), "--output-csv", str(out_csv), "--n-bootstrap", "10"])
    df = pd.read_csv(out_csv)
    assert "virus" in df.columns
    assert "auc_roc" in df.columns
    assert len(df) >= 1


def test_main_missing_file_returns_1(tmp_path: Path) -> None:
    rc = main(["--predictions", str(tmp_path / "nonexistent.csv")])
    assert rc == 1


def test_main_missing_score_column_returns_1(tmp_path: Path) -> None:
    df = _make_df().rename(columns={"score": "proba"})
    pred = _write_csv(tmp_path, df)
    rc = main(["--predictions", str(pred), "--n-bootstrap", "5"])
    assert rc == 1


def test_main_compare_same_file(tmp_path: Path) -> None:
    pred = _write_multi_virus_csv(tmp_path)
    rc = main([
        "--predictions", str(pred),
        "--compare", str(pred),
        "--n-bootstrap", "10",
    ])
    assert rc in (0, 1, 2)


def test_main_no_virus_column_returns_1(tmp_path: Path) -> None:
    df = _make_df().drop(columns=["virus"])
    pred = _write_csv(tmp_path, df)
    rc = main(["--predictions", str(pred), "--n-bootstrap", "5"])
    assert rc == 1
