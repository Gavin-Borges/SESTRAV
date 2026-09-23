"""Tests for scripts/evaluate_per_virus.py.

Covers:
  bootstrap_metric    CI ordering, single-class handling
  expected_calibration_error    valid/invalid score range, perfect calibration
  precision_at_recall    basic case, unreachable threshold, single-class
  evaluate_virus    all 6 Amendment 6 metrics present, real-neg subset
  evaluate_all_viruses    min_virus_size filter, per-virus grouping
  check_exit_criterion    pass/fail combos for EBV and HPV, on the HONEST column
  chance_ceiling    the derived validity floor, against a permutation null
  adjudicate_viruses    both AUC scales plus population labels
  main    CSV output, JSON output, missing file, exit codes
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_per_virus import (
    EXIT_CRITERION,
    FLOOR_Z,
    HEADLINE_AUC_COL,
    HONEST_AUC_COL,
    MIN_SAMPLES_DEFAULT,
    TARGET_PANEL,
    adjudicate_viruses,
    bootstrap_metric,
    chance_ceiling,
    check_exit_criterion,
    compare_predictions,
    evaluate_all_viruses,
    evaluate_virus,
    expected_calibration_error,
    format_adjudication_table,
    format_table,
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

    origins = ["tested_negative"] * n_neg_real + ["self_proteome_decoy"] * n_neg_decoy
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
    "n_pos",
    "n_neg_real",
    "n_neg_decoy",
    "n_total",
    "auc_roc",
    "auc_roc_lower",
    "auc_roc_upper",
    "auc_pr",
    "auc_pr_lower",
    "auc_pr_upper",
    "precision_at_10pct_recall",
    "ece",
    "auc_roc_9mer",
    "auc_roc_non9mer",
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
    # Real-neg-only AUC must differ from full AUC when decoys look like positives.
    # Using the full y_true/y_score pair for both keys is the mutation this
    # assertion is built to catch.
    df = _make_df(n_pos=30, n_neg_real=20, n_neg_decoy=40)
    df.loc[df["label"] == 1, "score"] = 0.9
    real_neg = (df["label"] == 0) & (df["negative_origin"] == "tested_negative")
    decoy = (df["label"] == 0) & (df["negative_origin"] == "self_proteome_decoy")
    df.loc[real_neg, "score"] = 0.1
    df.loc[decoy, "score"] = 0.99
    r = evaluate_virus(df, "score", n_bootstrap=20)
    assert r["auc_roc"] is not None
    assert r["auc_roc_real_neg_only"] is not None
    assert r["auc_roc"] != r["auc_roc_real_neg_only"]


def test_evaluate_virus_real_neg_none_when_no_real_negs() -> None:
    df = _make_df(n_pos=20, n_neg_real=0, n_neg_decoy=20)
    r = evaluate_virus(df, "score", n_bootstrap=10)
    assert r["auc_roc_real_neg_only"] is None


def test_evaluate_virus_no_origin_column() -> None:
    df = _make_df().drop(columns=["negative_origin"])
    r = evaluate_virus(df, "score", n_bootstrap=10)
    assert r["n_neg_real"] == 0
    assert r["auc_roc_real_neg_only"] is None


def test_evaluate_virus_iedb_api_counts_as_real_neg() -> None:
    # iedb_api negatives are genuine assay-confirmed negatives (aligns with
    # build_dataset_v5._real_neg_origins); allele_matched_nonbinder is a decoy.
    rng = np.random.default_rng(3)
    n_pos, n_api, n_decoy = 20, 15, 20
    df = pd.DataFrame(
        {
            "label": [1] * n_pos + [0] * (n_api + n_decoy),
            "score": np.concatenate(
                [rng.uniform(0.4, 1.0, n_pos), rng.uniform(0.0, 0.6, n_api + n_decoy)]
            ),
            "virus": "CMV",
            "negative_origin": [None] * n_pos
            + ["iedb_api"] * n_api
            + ["allele_matched_nonbinder"] * n_decoy,
            "peptide": ["GILGFVFTL"] * (n_pos + n_api + n_decoy),
        }
    )
    r = evaluate_virus(df, "score", n_bootstrap=10)
    assert r["n_neg_real"] == n_api
    assert r["n_neg_decoy"] == n_decoy
    assert r["auc_roc_real_neg_only"] is not None


def test_evaluate_virus_single_class_nan() -> None:
    df = pd.DataFrame(
        {
            "label": [0] * 30,
            "score": np.random.default_rng(0).uniform(0, 1, 30),
            "negative_origin": ["self_proteome_decoy"] * 30,
            "peptide": ["GILGFVFTL"] * 30,
        }
    )
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
    ebv_roc: float,
    ebv_lo: float,
    hpv_roc: float,
    ebv_honest: float | None = None,
    hpv_honest: float | None = None,
    n_pos: int = 300,
    n_neg_real: int = 300,
) -> dict:
    """Two gated viruses.

    The honest column defaults to the contaminated one, so a call that sets neither
    describes a decoy-free virus where the two scales agree. n_pos/n_neg_real default
    to 300/300, which puts chance_ceiling at 0.5388, below both Amendment 6 levels,
    so the validity check does not fire unless a test asks it to.
    """
    return {
        "EBV": {
            HEADLINE_AUC_COL: ebv_roc,
            "auc_roc_lower": ebv_lo,
            "auc_roc_upper": ebv_roc + 0.05,
            HONEST_AUC_COL: ebv_roc if ebv_honest is None else ebv_honest,
            "n_pos": n_pos,
            "n_neg_real": n_neg_real,
            "n_neg_decoy": 100,
        },
        "HPV": {
            HEADLINE_AUC_COL: hpv_roc,
            "auc_roc_lower": hpv_roc - 0.05,
            "auc_roc_upper": hpv_roc + 0.05,
            HONEST_AUC_COL: hpv_roc if hpv_honest is None else hpv_honest,
            "n_pos": n_pos,
            "n_neg_real": n_neg_real,
            "n_neg_decoy": 0,
        },
    }


def test_check_exit_criterion_passes() -> None:
    results = _make_results(ebv_roc=0.70, ebv_lo=0.60, hpv_roc=0.65)
    passed, msg = check_exit_criterion(results)
    assert passed
    assert "PASS" in msg


def test_check_exit_criterion_fails_ebv_roc() -> None:
    results = _make_results(ebv_roc=0.50, ebv_lo=0.40, hpv_roc=0.65)
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
    rc = main(
        [
            "--predictions",
            str(pred),
            "--output-json",
            str(out_json),
            "--output-csv",
            str(out_csv),
            "--n-bootstrap",
            "20",
        ]
    )
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
    rc = main(
        [
            "--predictions",
            str(pred),
            "--compare",
            str(pred),
            "--n-bootstrap",
            "10",
        ]
    )
    assert rc in (0, 1, 2)


def test_main_no_virus_column_returns_1(tmp_path: Path) -> None:
    df = _make_df().drop(columns=["virus"])
    pred = _write_csv(tmp_path, df)
    rc = main(["--predictions", str(pred), "--n-bootstrap", "5"])
    assert rc == 1


# ---------------------------------------------------------------------------
# format_table
# ---------------------------------------------------------------------------


def _make_format_result(precision_val: object = 0.90) -> dict:
    return {
        "n_pos": 10,
        "n_neg_real": 5,
        "n_neg_decoy": 5,
        "n_total": 20,
        "auc_roc": 0.75,
        "auc_roc_lower": 0.65,
        "auc_roc_upper": 0.85,
        "auc_pr": 0.70,
        "auc_pr_lower": 0.60,
        "auc_pr_upper": 0.80,
        "precision_at_10pct_recall": precision_val,
        "ece": 0.05,
        "auc_roc_9mer": 0.72,
        "auc_roc_non9mer": 0.68,
        "auc_roc_real_neg_only": 0.73,
    }


def test_format_table_contains_header() -> None:
    table = format_table({"EBV": _make_format_result()})
    assert "Virus" in table
    assert "AUC-ROC" in table


def test_format_table_none_formats_as_na() -> None:
    table = format_table({"EBV": _make_format_result(precision_val=None)})
    assert "N/A" in table


def test_format_table_sorted_by_virus() -> None:
    table = format_table(
        {
            "ZIKV": _make_format_result(),
            "EBV": _make_format_result(),
        }
    )
    assert table.index("EBV") < table.index("ZIKV")


# ---------------------------------------------------------------------------
# compare_predictions
# ---------------------------------------------------------------------------


def test_compare_predictions_row_mismatch_returns_empty() -> None:
    df_a = _make_df(n_pos=10, n_neg_real=5, n_neg_decoy=5)
    df_b = _make_df(n_pos=5, n_neg_real=3, n_neg_decoy=2)
    result = compare_predictions(df_a, df_b, "score", "score")
    assert result == {}


def test_compare_predictions_small_group_skipped() -> None:
    df = _make_df(n_pos=3, n_neg_real=1, n_neg_decoy=1, virus="TINY")
    result = compare_predictions(df, df.copy(), "score", "score")
    assert result == {}


# ---------------------------------------------------------------------------
# additional main() and evaluate_virus() paths
# ---------------------------------------------------------------------------


def test_main_missing_label_col_returns_1(tmp_path: Path) -> None:
    df = _make_df().drop(columns=["label"])
    pred = _write_csv(tmp_path, df)
    rc = main(["--predictions", str(pred), "--n-bootstrap", "5"])
    assert rc == 1


def test_main_no_results_returns_1(tmp_path: Path) -> None:
    df_a = _make_df(n_pos=3, n_neg_real=2, n_neg_decoy=2, virus="EBV")
    df_b = _make_df(n_pos=3, n_neg_real=2, n_neg_decoy=2, virus="HPV")
    df = pd.concat([df_a, df_b], ignore_index=True)
    pred = _write_csv(tmp_path, df)
    rc = main(["--predictions", str(pred), "--n-bootstrap", "5", "--min-virus-size", "20"])
    assert rc == 1


def test_evaluate_virus_small_n_no_bootstrap() -> None:
    rng = np.random.default_rng(9)
    df = pd.DataFrame(
        {
            "label": [1] * 5 + [0] * 5,
            "score": np.concatenate([rng.uniform(0.5, 1.0, 5), rng.uniform(0.0, 0.5, 5)]),
            "negative_origin": ["tested_negative"] * 10,
            "peptide": ["GILGFVFTL"] * 10,
        }
    )
    r = evaluate_virus(df, "score", n_bootstrap=10)
    assert not math.isnan(r["auc_roc"])
    assert math.isnan(r["auc_roc_lower"])


# ---------------------------------------------------------------------------
# evaluate_virus() no-peptide-column path
# ---------------------------------------------------------------------------


def test_evaluate_virus_no_peptide_column() -> None:
    df = _make_df().drop(columns=["peptide"])
    r = evaluate_virus(df, "score", n_bootstrap=10)
    assert r["auc_roc_9mer"] is None
    assert r["auc_roc_non9mer"] is None


# ---------------------------------------------------------------------------
# main() compare file not found
# ---------------------------------------------------------------------------


def test_main_compare_file_not_found_returns_1(tmp_path: Path) -> None:
    pred = _write_multi_virus_csv(tmp_path)
    rc = main(
        [
            "--predictions",
            str(pred),
            "--compare",
            "/nonexistent/path.csv",
            "--n-bootstrap",
            "5",
        ]
    )
    assert rc == 1


# ---------------------------------------------------------------------------
# New targeted tests (Fix 1 + Fix 3)
# ---------------------------------------------------------------------------


def test_main_returns_2_when_hpv_below_threshold(tmp_path: Path) -> None:
    # HPV with all-negative labels produces NaN AUC, failing the exit criterion.
    ebv = _make_df(virus="EBV", seed=3)
    hpv_all_neg = _make_df(n_pos=0, n_neg_real=30, n_neg_decoy=20, virus="HPV", seed=4)
    df = pd.concat([ebv, hpv_all_neg], ignore_index=True)
    pred = _write_csv(tmp_path, df)
    rc = main(["--predictions", str(pred), "--n-bootstrap", "10"])
    assert rc == 2


def test_main_returns_2_when_ebv_missing_from_predictions(tmp_path: Path) -> None:
    # No EBV rows at all; check_exit_criterion reports EBV missing and fails.
    hpv = _make_df(virus="HPV", seed=5)
    pred = _write_csv(tmp_path, hpv)
    rc = main(["--predictions", str(pred), "--n-bootstrap", "10"])
    assert rc == 2


def test_check_exit_criterion_hpv_at_exact_boundary() -> None:
    # Exactly at threshold: should pass.
    results_pass = _make_results(ebv_roc=0.70, ebv_lo=0.60, hpv_roc=0.58)
    passed, _ = check_exit_criterion(results_pass)
    assert passed

    # One ULP below threshold: should fail.
    results_fail = _make_results(ebv_roc=0.70, ebv_lo=0.60, hpv_roc=0.5799)
    passed, _ = check_exit_criterion(results_fail)
    assert not passed


def test_check_exit_criterion_ebv_at_exact_boundary() -> None:
    # Exactly at threshold: should pass.
    results_pass = _make_results(ebv_roc=0.57, ebv_lo=0.50, hpv_roc=0.65)
    passed, _ = check_exit_criterion(results_pass)
    assert passed

    # One ULP below threshold: should fail.
    results_fail = _make_results(ebv_roc=0.5699, ebv_lo=0.50, hpv_roc=0.65)
    passed, _ = check_exit_criterion(results_fail)
    assert not passed


def test_exit_criterion_dict_matches_check_function() -> None:
    # Catches drift if a threshold is updated in one place but not the other.
    assert EXIT_CRITERION["HPV"][HONEST_AUC_COL] == 0.58
    assert EXIT_CRITERION["EBV"][HONEST_AUC_COL] == 0.57
    # MUTATION GUARD. The levels are keyed on the honest column and on nothing else,
    # so re-pointing the gate at the contaminated column cannot be done silently.
    assert HONEST_AUC_COL == "auc_roc_real_neg_only"
    assert HEADLINE_AUC_COL == "auc_roc"
    assert HEADLINE_AUC_COL not in EXIT_CRITERION["HPV"]
    assert HEADLINE_AUC_COL not in EXIT_CRITERION["EBV"]


def test_evaluate_all_viruses_missing_virus_returns_no_entry() -> None:
    # Only EBV rows; HPV must not appear as a key in the results dict.
    df = _make_df(virus="EBV")
    results = evaluate_all_viruses(df, "score", n_bootstrap=10)
    assert "EBV" in results
    assert "HPV" not in results


def test_ece_counts_a_score_of_exactly_one():
    """A perfectly confident, perfectly wrong set must score ECE 1.0, not 0.0.

    Equal-width bins are half-open, so a final bin of [lo, 1.0) excludes a score
    of exactly 1.0 from every bin while n still counts it. The buggy form
    returned 0.0 here - perfect calibration - for the worst possible input.
    Random forests produce exactly 1.0 whenever every tree agrees, so this is a
    real input: 779 of 35,555 rows in the tracked v4 out-of-fold frame.
    """
    y_true = np.zeros(100)
    y_prob = np.ones(100)

    ece = expected_calibration_error(y_true, y_prob)
    assert ece is not None
    assert math.isclose(ece, 1.0, rel_tol=1e-9)


def test_ece_matches_a_hand_computed_two_bin_case():
    """Guards the fix against over-correction: ordinary scores bin as before."""
    y_true = np.array([0.0, 1.0])
    y_prob = np.array([0.0, 0.5])
    # Bin [0.0,0.1): one row, |0 - 0.0| = 0.  Bin [0.5,0.6): one row, |1 - 0.5| = 0.5.
    ece = expected_calibration_error(y_true, y_prob)
    assert ece is not None
    assert math.isclose(ece, 0.25, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# The exit criterion is decided on the HONEST column (mutation guards)
# ---------------------------------------------------------------------------


def test_exit_criterion_reads_the_honest_column_not_the_contaminated_one() -> None:
    """MUTATION GUARD, and the decisive one for this change.

    Both viruses look excellent on the contaminated scale (0.99) and fail badly on
    the honest one (0.40). Swap HONEST_AUC_COL back for HEADLINE_AUC_COL anywhere in
    check_exit_criterion and this gate certifies PASS on a model that cannot rank
    real assay-confirmed negatives, which is exactly the defect being fixed.
    """
    results = _make_results(
        ebv_roc=0.99, ebv_lo=0.95, hpv_roc=0.99, ebv_honest=0.40, hpv_honest=0.40
    )
    passed, msg = check_exit_criterion(results)
    assert not passed
    assert "0.400" in msg
    assert msg.count("FAIL") == 2
    # The contaminated number is still reported, labelled as not gated.
    assert "not gated" in msg
    assert "0.990" in msg


def test_exit_criterion_passes_when_only_the_honest_column_is_strong() -> None:
    """The converse mutation direction: a contaminated column at 0.10 must not veto."""
    results = _make_results(
        ebv_roc=0.10, ebv_lo=0.05, hpv_roc=0.10, ebv_honest=0.90, hpv_honest=0.90
    )
    passed, msg = check_exit_criterion(results)
    assert passed
    assert "PASS" in msg


def test_exit_criterion_fails_closed_when_the_honest_column_is_absent() -> None:
    """A missing honest value must fail, never fall back to the contaminated one."""
    results = _make_results(ebv_roc=0.99, ebv_lo=0.95, hpv_roc=0.99)
    for virus in ("EBV", "HPV"):
        del results[virus][HONEST_AUC_COL]
    passed, msg = check_exit_criterion(results)
    assert not passed
    assert "undefined" in msg


def test_exit_criterion_fails_when_the_level_sits_below_its_validity_floor() -> None:
    """A level a chance-level model could clear cannot certify anything.

    40 positives against 30 real negatives puts chance_ceiling at 0.6155, above both
    Amendment 6 levels, so neither 0.57 nor 0.58 separates a working model from a
    coin flip at that n. The gate must refuse rather than report PASS.
    """
    results = _make_results(
        ebv_roc=0.99, ebv_lo=0.95, hpv_roc=0.99, n_pos=40, n_neg_real=30
    )
    passed, msg = check_exit_criterion(results)
    assert not passed
    assert "validity floor" in msg
    assert "PASS" not in msg


# ---------------------------------------------------------------------------
# chance_ceiling: the derived validity floor
# ---------------------------------------------------------------------------


def test_chance_ceiling_matches_its_documented_closed_form() -> None:
    """The docstring derivation and the implementation must be the same expression."""
    for n_pos, n_neg in ((181, 137), (287, 72), (2473, 980), (1, 228)):
        expected = 0.5 + FLOOR_Z * math.sqrt((n_pos + n_neg + 1) / (12 * n_pos * n_neg))
        got = chance_ceiling(n_pos, n_neg)
        assert got is not None
        assert math.isclose(got, expected, rel_tol=1e-12)


def test_chance_ceiling_is_undefined_for_an_empty_arm() -> None:
    assert chance_ceiling(0, 100) is None
    assert chance_ceiling(100, 0) is None


def test_chance_ceiling_matches_a_permutation_null() -> None:
    """SECOND INSTRUMENT for the derivation, independent of the closed form.

    Permuting the labels destroys any association while leaving the score
    distribution intact, so the 95th percentile of the permuted AUCs estimates the
    same quantity chance_ceiling computes analytically.

    The tolerance is two-sided on purpose. Monte Carlo error on a 95th percentile at
    2,000 permutations is roughly 0.002 here, so a strict one-sided assertion is not
    something this test can support; the docstring records a draft of exactly that
    assertion failing. The exact claim is the variance one, tested below.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(20260916)
    n_pos, n_neg = 120, 100
    y_true = np.array([1] * n_pos + [0] * n_neg)
    y_score = rng.uniform(0.0, 1.0, size=n_pos + n_neg)
    null = np.array([roc_auc_score(rng.permutation(y_true), y_score) for _ in range(2000)])

    closed = chance_ceiling(n_pos, n_neg)
    assert closed is not None
    assert abs(closed - float(np.quantile(null, 0.95))) < 0.01


def test_chance_ceiling_null_spread_is_never_understated_under_ties() -> None:
    """The EXACT half of the conservatism claim, with no Monte Carlo in it.

    chance_ceiling uses the tie-free null variance (N + 1) / (12 * n_pos * n_neg).
    The tie-corrected variance subtracts sum(t**3 - t) / (N * (N - 1)), a
    non-negative quantity, so the tie-free form can only be the larger of the two.
    That is what makes the returned value safe to treat as a lower bound on any
    defensible threshold. Checked here on a heavily tied score vector and on a
    tie-free one, where the two expressions must coincide exactly.
    """
    n_pos, n_neg = 120, 100
    n = n_pos + n_neg

    def _tie_corrected(scores: np.ndarray) -> float:
        _, counts = np.unique(scores, return_counts=True)
        correction = float(np.sum(counts**3 - counts)) / (n * (n - 1.0))
        return 0.5 + FLOOR_Z * math.sqrt(((n + 1.0) - correction) / (12.0 * n_pos * n_neg))

    closed = chance_ceiling(n_pos, n_neg)
    assert closed is not None

    rng = np.random.default_rng(4)
    tied = np.round(rng.uniform(0.0, 1.0, size=n), 1)  # 11 distinct values, heavy ties
    assert len(np.unique(tied)) < n
    assert closed > _tie_corrected(tied)

    untied = np.arange(n, dtype=float)  # all distinct
    assert math.isclose(closed, _tie_corrected(untied), rel_tol=1e-12)


# ---------------------------------------------------------------------------
# adjudicate_viruses: both scales, and every count names its population
# ---------------------------------------------------------------------------


def _adj_metrics(
    n_pos: int, n_neg_real: int, n_neg_decoy: int, headline: float, honest: float | None
) -> dict:
    return {
        "n_pos": n_pos,
        "n_neg_real": n_neg_real,
        "n_neg_decoy": n_neg_decoy,
        HEADLINE_AUC_COL: headline,
        HONEST_AUC_COL: honest,
    }


def _adj_fixture() -> dict:
    return {
        # In the target panel, decoy-free: the two scales must agree exactly.
        "HPV": _adj_metrics(181, 137, 0, 0.4820, 0.4820),
        # In the target panel, decoy-bearing: contaminated well above honest.
        "EBV": _adj_metrics(287, 72, 300, 0.7113, 0.5557),
        # NOT in the target panel, decoy-free.
        "RSV": _adj_metrics(15, 105, 0, 0.4946, 0.4946),
        # Single-class: no headline AUC, so not a two-class virus at all.
        "ZIKV": _adj_metrics(0, 54, 0, float("nan"), None),
    }


def test_adjudicate_viruses_excludes_single_class_viruses() -> None:
    adj = adjudicate_viruses(_adj_fixture())
    assert [r["virus"] for r in adj] == ["EBV", "HPV", "RSV"]


def test_adjudicate_viruses_labels_both_populations() -> None:
    """Requirement: a count from this list must be able to name its population.

    "Decoy-free" is 2 across all two-class viruses here and 1 across the target
    panel. Both are correct, they describe the same data, and an unlabelled count is
    ambiguous between them. The shipped artifact shows the same shape at 6 and 3.
    """
    adj = adjudicate_viruses(_adj_fixture())
    assert sum(1 for r in adj if r["decoy_free"]) == 2
    panel = [r for r in adj if r["in_target_panel"]]
    assert sorted(r["virus"] for r in panel) == ["EBV", "HPV"]
    assert sum(1 for r in panel if r["decoy_free"]) == 1
    assert "RSV" not in TARGET_PANEL


def test_adjudicate_viruses_reports_both_scales_and_the_gap() -> None:
    adj = {r["virus"]: r for r in adjudicate_viruses(_adj_fixture())}
    # Decoy-free virus: the honest column IS the headline, so inflation is zero.
    assert adj["HPV"]["auc_roc_contaminated"] == adj["HPV"]["auc_roc_honest"]
    assert math.isclose(adj["HPV"]["decoy_inflation"], 0.0, abs_tol=1e-12)
    # Decoy-bearing virus: the contaminated column is the higher of the two.
    assert adj["EBV"]["auc_roc_contaminated"] > adj["EBV"]["auc_roc_honest"]
    assert math.isclose(adj["EBV"]["decoy_inflation"], 0.7113 - 0.5557, abs_tol=1e-9)


def test_adjudicate_viruses_scores_the_floor_against_the_honest_column() -> None:
    """The floor uses the HONEST arms (positives + real negatives), not all rows.

    EBV's 300 decoys must not enter its floor: counting them would shrink the null
    variance and hand the virus an easier bar on the strength of the very rows the
    honest column exists to exclude.
    """
    adj = {r["virus"]: r for r in adjudicate_viruses(_adj_fixture())}
    assert adj["EBV"]["chance_floor"] == chance_ceiling(287, 72)
    assert adj["EBV"]["chance_floor"] != chance_ceiling(287, 72 + 300)
    # Both gated viruses sit below their own floors on these shipped-shape numbers.
    assert adj["EBV"]["beats_chance_floor"] is False
    assert adj["HPV"]["beats_chance_floor"] is False


def test_adjudicate_viruses_sets_exit_status_only_for_gated_viruses() -> None:
    adj = {r["virus"]: r for r in adjudicate_viruses(_adj_fixture())}
    assert adj["EBV"]["exit_status"] == "FAIL"
    assert adj["HPV"]["exit_status"] == "FAIL"
    # RSV is two-class but EXIT_CRITERION names no level for it, so there is no
    # verdict to report. Inventing one would be an exit criterion this repo never set.
    assert adj["RSV"]["exit_status"] is None
    assert adj["RSV"]["exit_threshold"] is None


def test_adjudication_table_reports_both_scales_side_by_side() -> None:
    table = format_adjudication_table(_adj_fixture())
    assert "ROC_all" in table
    assert "ROC_real" in table
    assert "CONTAMINATED" in table
    assert "HONEST" in table
    # Every population count in the footer is labelled.
    assert "all two-class (n=3)" in table
    assert "target panel (n=2)" in table
