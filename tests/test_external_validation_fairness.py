"""The external-validation fairness supplement.

src/external_validation_fairness.py computes the prevalence baseline, the
per-tool metrics, the EBV/HPV16 virus-weighted averages, the length strata, the
gold-standard holdout spotlight, the mean-versus-max allele collapse and the
training-overlap screen that the external-validation report quotes. No test
named the module or any of its functions before this file. Each metric test
compares a function against src.evaluate_metrics.evaluate run on the subset
the function should have selected, so it binds the selection and the
combination, not the metric implementation (tests/test_metrics.py binds that).
The thresholds are tested at their boundaries (10 rows; 30% and 50% overlap).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluate_metrics import evaluate
from src.external_validation_fairness import (
    PREVALENCE_POS_RATE,
    _tool_columns,
    collapse_allele_scores,
    evaluate_tools,
    holdout_spotlight,
    length_stratified,
    parse_raw_predig,
    parse_raw_prime,
    prevalence_baseline_row,
    quantify_overlap_robust,
    run_fairness,
    virus_weighted_metrics,
    write_fairness_report,
)
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES


def _scored(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    label = np.tile([1, 0, 1, 1, 0], n // 5 + 1)[:n]
    return pd.DataFrame(
        {
            "label": label.astype(float),
            "good": label + rng.uniform(0.0, 0.9, n),
            "noisy": rng.uniform(0.0, 1.0, n),
        }
    )


def _metrics(frame: pd.DataFrame, col: str) -> dict:
    valid = frame.dropna(subset=["label", col])
    return evaluate(valid["label"].values, valid[col].values)


def test_tool_columns_maps_only_the_score_columns_present() -> None:
    merged = pd.DataFrame(columns=["peptide", "rf_oof_score", "prime_score", "unrelated"])
    assert _tool_columns(merged) == {
        "SESTRAV RF (31-feat)": "rf_oof_score",
        "PRIME 2.1 (max)": "prime_score",
    }


def test_prevalence_baseline_is_the_frozen_tier_a_class_balance() -> None:
    assert PREVALENCE_POS_RATE == 506 / 720
    row = prevalence_baseline_row(42)
    assert row["n_peptides"] == 42
    assert row["auc_roc"] == 0.5
    assert row["auc_pr"] == row["issr_10"] == row["issr_25"] == PREVALENCE_POS_RATE
    assert row["delta_auc_pr_vs_prev"] == row["delta_issr10_vs_prev"] == 0.0


def test_evaluate_tools_scores_each_tool_on_its_own_valid_rows() -> None:
    df = _scored(30, 0)
    df.loc[3, "label"] = np.nan
    df.loc[[5, 8], "noisy"] = np.nan
    out = evaluate_tools(df, {"Good": "good", "Noisy": "noisy", "Absent": "no_such_column"})

    assert list(out["tool"]) == ["Good", "Noisy", "Random / Prevalence baseline"]
    assert list(out.columns[:2]) == ["tool", "n_peptides"]
    by_tool = out.set_index("tool")
    assert by_tool.loc["Good", "n_peptides"] == 29
    assert by_tool.loc["Noisy", "n_peptides"] == 27
    for tool, col in (("Good", "good"), ("Noisy", "noisy")):
        expected = _metrics(df, col)
        for key in ("auc_roc", "auc_pr", "issr_10"):
            assert by_tool.loc[tool, key] == pytest.approx(expected[key])
        assert by_tool.loc[tool, "delta_auc_pr_vs_prev"] == pytest.approx(
            expected["auc_pr"] - PREVALENCE_POS_RATE
        )
        assert by_tool.loc[tool, "delta_issr10_vs_prev"] == pytest.approx(
            expected["issr_10"] - PREVALENCE_POS_RATE
        )
    # The baseline row takes the FIRST tool's row count.
    assert by_tool.loc["Random / Prevalence baseline", "n_peptides"] == 29


def test_evaluate_tools_baseline_is_optional_and_nothing_scores_to_empty() -> None:
    df = _scored(20, 1)
    out = evaluate_tools(df, {"Good": "good"}, include_prevalence=False)
    assert list(out["tool"]) == ["Good"]
    assert evaluate_tools(df, {"Absent": "no_such_column"}).empty


def test_virus_weighted_metrics_weights_ebv_and_hpv16_equally() -> None:
    ebv, hpv = _scored(20, 2), _scored(20, 3)
    ebv["virus"], hpv["virus"] = "EBV", "HPV16"
    other = _scored(5, 4)
    other["virus"] = "CMV"
    df = pd.concat([ebv, hpv, other], ignore_index=True)
    out = virus_weighted_metrics(df, {"Good": "good"})

    assert len(out) == 1
    row = out.iloc[0]
    e, h = _metrics(ebv, "good"), _metrics(hpv, "good")
    assert row["tool"] == "Good"
    assert row["weighted_auc_pr"] == pytest.approx(0.5 * e["auc_pr"] + 0.5 * h["auc_pr"])
    assert row["weighted_auc_roc"] == pytest.approx(0.5 * e["auc_roc"] + 0.5 * h["auc_roc"])
    assert row["weighted_issr_10"] == pytest.approx(0.5 * e["issr_10"] + 0.5 * h["issr_10"])
    assert (row["ebv_n"], row["hpv16_n"]) == (20, 20)


def test_virus_weighted_metrics_needs_ten_scored_rows_of_both_viruses() -> None:
    ebv, hpv = _scored(20, 5), _scored(10, 6)
    ebv["virus"], hpv["virus"] = "EBV", "HPV16"
    assert len(virus_weighted_metrics(pd.concat([ebv, hpv]), {"Good": "good"})) == 1  # 10 is enough
    assert virus_weighted_metrics(pd.concat([ebv, hpv.iloc[:9]]), {"Good": "good"}).empty
    assert virus_weighted_metrics(ebv, {"Good": "good"}).empty


def test_length_stratified_separates_nine_mers_from_the_rest() -> None:
    nine, other = _scored(12, 7), _scored(15, 8)
    nine["peptide"] = ["A" * 9] * 12
    other["peptide"] = ["A" * 10] * 8 + ["A" * 8] * 7
    out = length_stratified(pd.concat([nine, other], ignore_index=True), {"Good": "good"})

    assert list(out["length_group"]) == ["9-mer", "non-9-mer"]
    for group, frame in (("9-mer", nine), ("non-9-mer", other)):
        row = out[out["length_group"] == group].iloc[0]
        expected = _metrics(frame, "good")
        assert row["n"] == len(frame)
        assert row["auc_pr"] == pytest.approx(expected["auc_pr"])
        assert row["auc_roc"] == pytest.approx(expected["auc_roc"])
        assert row["issr_10"] == pytest.approx(expected["issr_10"])


def test_length_stratified_skips_small_groups_and_needs_peptides() -> None:
    nine, other = _scored(12, 9), _scored(10, 10)
    nine["peptide"] = ["A" * 9] * 12
    other["peptide"] = ["A" * 10] * 10
    both = length_stratified(pd.concat([nine, other], ignore_index=True), {"Good": "good"})
    assert list(both["length_group"]) == ["9-mer", "non-9-mer"]  # 10 rows is enough
    short = length_stratified(
        pd.concat([nine, other.iloc[:9]], ignore_index=True), {"Good": "good"}
    )
    assert list(short["length_group"]) == ["9-mer"]
    assert length_stratified(_scored(20, 11), {"Good": "good"}).empty


def test_holdout_spotlight_keeps_gold_standard_peptides_sorted() -> None:
    gold = sorted(GOLD_STANDARD_EPITOPES)[:3]
    df = pd.DataFrame(
        {
            "peptide": [gold[2], "NOTGOLDXX", gold[0], gold[1], "ALSONOTGD"],
            "label": [1, 0, 1, 0, 1],
            "virus": ["EBV"] * 5,
            "binding_max": [0.1, 0.2, 0.3, 0.4, 0.5],
            "prime_score": [0.5, 0.4, 0.3, 0.2, 0.1],
            "unrelated": list(range(5)),
        }
    )
    out = holdout_spotlight(df)
    assert list(out["peptide"]) == gold
    assert list(out.columns) == ["peptide", "label", "virus", "binding_max", "prime_score"]
    assert holdout_spotlight(df[~df["peptide"].isin(gold)]).empty


def test_collapse_allele_scores_takes_per_peptide_max_and_mean() -> None:
    raw = pd.DataFrame({"pep": ["P1", "P1", "P2"], "allele": ["A", "B", "A"], "s": [0.2, 0.8, 0.5]})
    max_df, mean_df = collapse_allele_scores(raw, "pep", "s", "x")
    assert list(max_df.columns) == ["peptide", "x_max_score"]
    assert list(mean_df.columns) == ["peptide", "x_mean_score"]
    assert dict(zip(max_df["peptide"], max_df["x_max_score"])) == {"P1": 0.8, "P2": 0.5}
    assert dict(zip(mean_df["peptide"], mean_df["x_mean_score"])) == pytest.approx(
        {"P1": 0.5, "P2": 0.5}
    )


def test_parse_raw_predig_resolves_columns_case_insensitively(tmp_path) -> None:
    path = tmp_path / "predig.csv"
    path.write_text("Epitope,HLA,PredIG\nP1,A,0.2\nP1,B,0.6\nP2,A,0.3\n", encoding="utf-8")
    max_df, mean_df = parse_raw_predig(str(path))
    assert dict(zip(max_df["peptide"], max_df["predig_max_score"])) == {"P1": 0.6, "P2": 0.3}
    assert dict(zip(mean_df["peptide"], mean_df["predig_mean_score"])) == pytest.approx(
        {"P1": 0.4, "P2": 0.3}
    )
    bad = tmp_path / "bad.csv"
    bad.write_text("sequence,value\nP1,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot parse PredIG columns"):
        parse_raw_predig(str(bad))


def test_parse_raw_prime_reads_tab_separated_output_with_comment_lines(tmp_path) -> None:
    path = tmp_path / "prime.txt"
    path.write_text(
        "# PRIME output\n"
        "Peptide\tScore_bestAllele\tBestAllele\n"
        "P1\t0.1\tA0201\n"
        "P1\t0.7\tB0702\n"
        "P2\t0.4\tA0201\n",
        encoding="utf-8",
    )
    max_df, mean_df = parse_raw_prime(str(path))
    assert dict(zip(max_df["peptide"], max_df["prime_max_score"])) == {"P1": 0.7, "P2": 0.4}
    assert dict(zip(mean_df["peptide"], mean_df["prime_mean_score"])) == pytest.approx(
        {"P1": 0.4, "P2": 0.4}
    )
    bad = tmp_path / "bad.txt"
    bad.write_text("Sequence\tValue\nP1\t0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot parse PRIME columns"):
        parse_raw_prime(str(bad))


def _train_list(tmp_path, peptides, column="Epitope") -> str:
    path = tmp_path / "train.csv"
    pd.DataFrame({column: peptides}).to_csv(path, index=False)
    return str(path)


def _nine_mers(n: int) -> list[str]:
    # Distinct strings of equal length cannot contain one another, so every overlap
    # in the tests below is exactly the one the test builds.
    return [f"A{i:08d}" for i in range(n)]


def test_quantify_overlap_without_a_training_list_reports_unknown(tmp_path) -> None:
    for train_path in (None, "", str(tmp_path / "absent.csv")):
        meta = quantify_overlap_robust({"AAAAAAAAA"}, train_path)
        assert meta["status"] == "missing_train_list"
        assert meta["overlap_count"] == meta["total_overlap_pct"] == "unknown"


def test_quantify_overlap_counts_exact_matches_then_substrings(tmp_path) -> None:
    train = _train_list(tmp_path, ["AAAAAAAAA", "XCCCCCCCCCX", "DDDDD", "ZZZZ"])
    meta = quantify_overlap_robust({"aaaaaaaaa", "CCCCCCCCC", "DDDDDDDDD", "EEEEEEEEE"}, train)
    assert meta["eval_peptide_count"] == 4
    assert (meta["exact_overlap_count"], meta["substring_overlap_count"]) == (1, 2)
    assert meta["total_overlap_count"] == meta["overlap_count"] == 3
    assert meta["exact_overlap_pct"] == 25.0
    assert meta["total_overlap_pct"] == meta["overlap_pct"] == 75.0
    assert meta["exact_overlap_peptides_sample"] == ["AAAAAAAAA"]
    assert meta["substring_only_peptides_sample"] == ["CCCCCCCCC", "DDDDDDDDD"]
    assert meta["total_overlap_peptides"] == ["AAAAAAAAA", "CCCCCCCCC", "DDDDDDDDD"]
    assert meta["status"] == "contaminated"


@pytest.mark.parametrize(
    ("n_overlap", "cap", "status"),
    [
        (3, 30.0, "acceptable"),  # 30% is not above the 30% cap
        (4, 30.0, "contaminated_cap"),
        (5, 30.0, "contaminated_cap"),  # 50% is not above 50
        (6, 30.0, "contaminated"),
        (4, 45.0, "acceptable"),  # the cap is a parameter
    ],
)
def test_quantify_overlap_status_thresholds_are_strict(tmp_path, n_overlap, cap, status) -> None:
    peptides = _nine_mers(10)
    meta = quantify_overlap_robust(
        set(peptides), _train_list(tmp_path, peptides[:n_overlap]), contamination_cap_pct=cap
    )
    assert meta["total_overlap_pct"] == 10.0 * n_overlap
    assert meta["status"] == status


def test_quantify_overlap_falls_back_to_the_first_column_and_handles_no_peptides(tmp_path) -> None:
    train = _train_list(tmp_path, ["A00000001"], column="sequence")
    assert quantify_overlap_robust({"A00000001", "A00000002"}, train)["exact_overlap_count"] == 1
    empty = quantify_overlap_robust(set(), train)
    assert (empty["total_overlap_pct"], empty["status"]) == (0.0, "acceptable")


def test_write_fairness_report_lists_every_section(tmp_path) -> None:
    path = tmp_path / "report.md"
    metrics = pd.DataFrame({"tool": ["X"], "auc_pr": [0.7]})
    empty = pd.DataFrame()
    meta = {"predig": {"status": "acceptable"}}
    write_fairness_report(str(path), metrics, empty, empty, empty, empty, empty, meta)
    text = path.read_text(encoding="utf-8")
    assert "## Intersection Set Metrics" in text
    assert text.count("*No data*") == 5
    assert '"status": "acceptable"' in text
    assert "Overlap-Excluded" not in text

    write_fairness_report(str(path), metrics, empty, empty, empty, empty, empty, meta, metrics)
    assert "## Overlap-Excluded / Overlap-Only Subsets (intersection)" in path.read_text(
        encoding="utf-8"
    )


def test_run_fairness_writes_every_output_from_a_synthetic_run(tmp_path) -> None:
    rng = np.random.RandomState(12)
    peptides = _nine_mers(34) + [f"B{i:09d}" for i in range(26)]
    label = np.tile([1, 0, 1], 20)
    merged = pd.DataFrame(
        {
            "peptide": peptides,
            "label": label,
            "virus": ["EBV", "HPV16"] * 30,
            "rf_oof_score": label + rng.uniform(0, 0.9, 60),
            "binding_max": rng.uniform(0, 1, 60),
            "predig_max_score": rng.uniform(0, 1, 60),
            "prime_score": rng.uniform(0, 1, 60),
        }
    )
    merged.loc[[0, 1, 2, 3], "rf_oof_score"] = np.nan  # 56 rows in the intersection
    merged_path = tmp_path / "merged.csv"
    merged.to_csv(merged_path, index=False)
    flagged = peptides[10:35]  # 25 peptides, all inside the intersection
    train = _train_list(tmp_path, flagged)
    run_dir = tmp_path / "run"
    (run_dir / "manifests").mkdir(parents=True)  # run_fairness writes there but does not create it

    report = run_fairness(str(merged_path), str(run_dir), predig_train=train)

    processed = run_dir / "processed"
    assert report == str(processed / "fairness_analysis.md")
    inter = pd.read_csv(processed / "fairness_metrics_intersection.csv").set_index("tool")
    assert inter.loc["SESTRAV RF (31-feat)", "n_peptides"] == 56
    rf_rows = merged.dropna(subset=["rf_oof_score"])
    expected = evaluate(rf_rows["label"].values, rf_rows["rf_oof_score"].values)
    assert inter.loc["SESTRAV RF (31-feat)", "auc_pr"] == pytest.approx(expected["auc_pr"])
    full = pd.read_csv(processed / "fairness_metrics_external_full.csv")
    assert "SESTRAV RF (31-feat)" not in set(full["tool"])
    assert full.set_index("tool").loc["Binding-only (max)", "n_peptides"] == 60

    overlap = json.loads((run_dir / "manifests" / "training_overlap.json").read_text())
    assert overlap["predig"]["exact_overlap_count"] == 25
    assert overlap["predig"]["status"] == "contaminated_cap"  # 25 of 60 is 41.67%
    assert overlap["prime"]["status"] == "missing_train_list"

    subsets = pd.read_csv(processed / "overlap_subset_metrics.csv")
    rf = subsets[subsets["tool"] == "SESTRAV RF (31-feat)"].set_index("subset")
    assert rf.loc["overlap-only", "n_peptides"] == 25
    assert rf.loc["overlap-excluded", "n_peptides"] == 31
    for name in (
        "virus_weighted_metrics.csv",
        "length_stratified_auc_pr.csv",
        "holdout_spotlight.csv",
        "mean_collapse_sensitivity.csv",
    ):
        assert (processed / name).exists()
    assert "Overlap-Excluded / Overlap-Only" in (processed / "fairness_analysis.md").read_text(
        encoding="utf-8"
    )


def test_run_fairness_collapses_raw_predig_and_prime_scores_both_ways(tmp_path) -> None:
    rng = np.random.RandomState(13)
    peptides = _nine_mers(30)
    label = np.tile([1, 0, 1], 10)
    merged = pd.DataFrame(
        {
            "peptide": peptides,
            "label": label,
            "virus": ["EBV", "HPV16"] * 15,
            "rf_oof_score": label + rng.uniform(0, 0.9, 30),
            "binding_max": rng.uniform(0, 1, 30),
        }
    )
    merged_path = tmp_path / "merged.csv"
    merged.to_csv(merged_path, index=False)
    predig = pd.DataFrame(
        {
            "peptide": peptides * 2,
            "allele": ["A"] * 30 + ["B"] * 30,
            "predig_score": rng.uniform(0, 1, 60),
        }
    )
    predig_path = tmp_path / "predig.csv"
    predig.to_csv(predig_path, index=False)
    prime = pd.DataFrame(
        {
            "Peptide": peptides * 2,
            "Score_bestAllele": rng.uniform(0, 1, 60),
            "BestAllele": ["A"] * 60,
        }
    )
    prime_path = tmp_path / "prime.txt"
    prime_path.write_text("# PRIME\n" + prime.to_csv(sep="\t", index=False), encoding="utf-8")
    run_dir = tmp_path / "run"
    (run_dir / "manifests").mkdir(parents=True)

    run_fairness(
        str(merged_path), str(run_dir), predig_raw=str(predig_path), prime_raw=str(prime_path)
    )

    collapse = pd.read_csv(run_dir / "processed" / "mean_collapse_sensitivity.csv")
    assert list(zip(collapse["tool"], collapse["collapse"])) == [
        ("PredIG-Path", "max"),
        ("PredIG-Path", "mean"),
        ("PRIME 2.1", "max"),
        ("PRIME 2.1", "mean"),
    ]
    for (tool, how), raw, col in (
        (("PredIG-Path", "max"), predig, "predig_score"),
        (("PredIG-Path", "mean"), predig, "predig_score"),
        (("PRIME 2.1", "max"), prime.rename(columns={"Peptide": "peptide"}), "Score_bestAllele"),
        (("PRIME 2.1", "mean"), prime.rename(columns={"Peptide": "peptide"}), "Score_bestAllele"),
    ):
        per_peptide = raw.groupby("peptide")[col].agg(how)
        expected = evaluate(label, per_peptide.loc[peptides].values)
        got = collapse[(collapse["tool"] == tool) & (collapse["collapse"] == how)].iloc[0]
        assert got["auc_pr"] == pytest.approx(expected["auc_pr"])
        assert got["auc_roc"] == pytest.approx(expected["auc_roc"])
    # No training list, so nothing is flagged and no overlap-subset table is written.
    assert not (run_dir / "processed" / "overlap_subset_metrics.csv").exists()
