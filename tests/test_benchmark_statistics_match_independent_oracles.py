"""The external-benchmark statistics agree with independent oracles.

Why this test exists
--------------------
`src/external_benchmark_comparison.py` produces the head-to-head statistics behind the
external benchmark report: the R10/R25 enrichment ratios, the pairwise Spearman table,
the per-virus breakdown and the bootstrap 95% CIs (with their yes/no significance
flags) for SESTRAV RF against every other tool. `src/external_validation_finalize.py`
reuses `per_virus_metrics` and `run_bootstrap_comparisons` for its appended report.
Before this file no test called any of those five functions, so a flipped comparison,
an off-by-one threshold or a swapped difference would have shipped silently into a
published table.

Pinned here, each against an oracle that does not route through the code under test:

- `compute_issr_ratios`: R10 = ISSR@10(tool) / ISSR@10(binding-only) and R25 likewise
  (the definition `src/h2_tier_a_evaluation.py` prints for its own R10), so the
  binding row's own ratio is exactly 1. A baseline ISSR at or below 0.05 yields NaN
  for every row (a strict `>` floor). No binding row: the table comes back unchanged.
  The input is not mutated.
- `spearman_correlation_matrix`: one row per unordered tool pair, in tool order, with
  no self-pairs. rho is the Pearson correlation of average ranks over the PAIRWISE
  complete rows and p is two-sided from Student's t with n - 2 degrees of freedom,
  both recomputed here without scipy's spearmanr. A pair with fewer than 5 complete
  rows is omitted; 5 is kept. A constant tool gives NaN rho (undefined), not a number.
- `per_virus_metrics`: one row per (virus, tool) with at least 10 rows whose label and
  score are both present, viruses in sorted order, missing viruses excluded; AUC-ROC
  and AUC-PR equal scikit-learn on exactly that group, ISSR and recall at 10%/25% equal
  a hand count of the top max(1, floor(n*k/100)) scores. A single-class group is kept
  with NaN AUCs, as `evaluate()` documents.
- `bootstrap_metric_diff`: a paired nonparametric bootstrap of metric(a) - metric(b).
  Every resample draws n rows with replacement from all n rows and scores BOTH arms on
  the same draw; single-class resamples are skipped, not scored; the returned triple
  is the mean and the 2.5th/97.5th linear-interpolation percentiles of the kept
  differences, recomputed here by hand. The seeded stream is pinned because the
  published CIs are reproducible only under it: defaults are 2000 resamples, seed 42.
- `run_bootstrap_comparisons`: one row per tool other than the reference, in tool
  order, skipping tools with fewer than 20 rows complete on label, reference and
  tool; each row equals direct `bootstrap_metric_diff` calls (reference first) with
  average precision and ISSR@10; "significant" means the 95% CI excludes zero, so a
  bound sitting exactly on zero is "no".

Out of scope by design: nothing here runs `run_comparison`, `run_finalize` or
`baseline_comparison`; those paths belong to a separate mutation campaign. None of
these five functions computes a bootstrap p-value, so the two-sided bootstrap p question
raised for `src/external_validation_finalize.py` has no counterpart to pin here.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, roc_auc_score

from src import external_benchmark_comparison as ebc
from src.evaluate_metrics import issr_at_k

BINDING = "Binding-only (max)"
REFERENCE = "SESTRAV RF (31-feat)"
BOOT_COLUMNS = [
    "comparison",
    "auc_pr_diff_mean",
    "auc_pr_diff_ci_low",
    "auc_pr_diff_ci_high",
    "auc_pr_significant",
    "issr10_diff_mean",
    "issr10_diff_ci_low",
    "issr10_diff_ci_high",
    "issr10_significant",
]


# ---------------------------------------------------------------------------
# Independent oracles
# ---------------------------------------------------------------------------


def _average_ranks(values) -> np.ndarray:
    """1-based ranks; tied values share the mean of the positions they span."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ordered = x[order]
    ranks = np.empty(len(x))
    start = 0
    while start < len(x):
        stop = start
        while stop + 1 < len(x) and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def _spearman_oracle(x, y) -> tuple[float, float]:
    """rho = Pearson r of average ranks; p two-sided from Student's t, n - 2 df."""
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    rho = float(np.sum(rx * ry) / math.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    dof = len(rx) - 2
    if abs(rho) >= 1.0:
        return rho, 0.0
    t_stat = rho * math.sqrt(dof / ((1.0 - rho) * (1.0 + rho)))
    return rho, float(2.0 * student_t.sf(abs(t_stat), dof))


def _top_indices(scores, k: int) -> list[int]:
    count = max(1, (len(scores) * k) // 100)
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:count]


def _issr_oracle(labels, scores, k: int) -> float:
    top = _top_indices(list(scores), k)
    return sum(float(labels[i]) for i in top) / len(top)


def _recall_oracle(labels, scores, k: int) -> float:
    top = _top_indices(list(scores), k)
    return sum(float(labels[i]) for i in top) / max(1, int(sum(labels)))


def _percentile_oracle(values, q: float) -> float:
    """Linear interpolation between the two closest order statistics."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _bootstrap_oracle(y, a, b, metric, n_boot: int, seed: int):
    """Paired bootstrap, written out longhand; returns (triple, kept resample count)."""
    generator = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = generator.integers(0, n, size=n)
        if len(set(y[idx].tolist())) < 2:
            continue
        diffs.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    if not diffs:
        return (math.nan, math.nan, math.nan), 0
    triple = (
        math.fsum(diffs) / len(diffs),
        _percentile_oracle(diffs, 2.5),
        _percentile_oracle(diffs, 97.5),
    )
    return triple, len(diffs)


def _excludes_zero(lo: float, hi: float) -> str:
    return "no" if lo <= 0.0 <= hi else "yes"


# ---------------------------------------------------------------------------
# compute_issr_ratios
# ---------------------------------------------------------------------------


def _ratio_table() -> pd.DataFrame:
    # The binding row sits in the MIDDLE so a lookup that grabbed the wrong row
    # (the first, or every other row) produces a different denominator.
    return pd.DataFrame(
        {
            "tool": ["SESTRAV RF (31-feat)", BINDING, "PRIME 2.1"],
            "issr_10": [0.8, 0.4, 0.2],
            "issr_25": [0.6, 0.5, 0.25],
            "auc_pr": [0.7, 0.6, 0.5],
        }
    )


def test_issr_ratio_is_each_tools_issr_over_the_binding_only_issr():
    table = _ratio_table()
    out = ebc.compute_issr_ratios(table)
    for _, row in out.iterrows():
        assert row["R10"] == pytest.approx(row["issr_10"] / 0.4, rel=1e-12)
        assert row["R25"] == pytest.approx(row["issr_25"] / 0.5, rel=1e-12)
    binding = out.loc[out["tool"] == BINDING].iloc[0]
    assert binding["R10"] == 1.0
    assert binding["R25"] == 1.0
    assert list(out.columns) == list(table.columns) + ["R10", "R25"]
    assert list(out["tool"]) == list(table["tool"])


def test_issr_ratio_divides_by_the_named_binding_tool():
    table = _ratio_table().replace({"tool": {BINDING: "NetMHCpan EL"}})
    out = ebc.compute_issr_ratios(table, binding_tool="NetMHCpan EL")
    assert list(out["R10"]) == pytest.approx([2.0, 1.0, 0.5], rel=1e-12)
    assert list(out["R25"]) == pytest.approx([1.2, 1.0, 0.5], rel=1e-12)


def test_issr_ratio_without_a_binding_row_returns_the_table_unchanged():
    table = _ratio_table()
    table = table.loc[table["tool"] != BINDING].reset_index(drop=True)
    out = ebc.compute_issr_ratios(table)
    pd.testing.assert_frame_equal(out, table)
    assert "R10" not in out.columns and "R25" not in out.columns


@pytest.mark.parametrize("baseline", [0.0, 0.03, 0.05])
def test_issr_ratio_is_undefined_when_the_binding_issr_is_at_or_below_the_floor(baseline):
    table = _ratio_table()
    table.loc[table["tool"] == BINDING, "issr_10"] = baseline
    out = ebc.compute_issr_ratios(table)
    assert out["R10"].isna().all()
    # The two floors are independent: R25's baseline (0.5) is well above it.
    assert list(out["R25"]) == pytest.approx([1.2, 1.0, 0.5], rel=1e-12)


@pytest.mark.parametrize("baseline", [0.0, 0.03, 0.05])
def test_issr_ratio_r25_floor_is_independent_of_r10(baseline):
    table = _ratio_table()
    table.loc[table["tool"] == BINDING, "issr_25"] = baseline
    out = ebc.compute_issr_ratios(table)
    assert out["R25"].isna().all()
    assert list(out["R10"]) == pytest.approx([2.0, 1.0, 0.5], rel=1e-12)


def test_issr_ratio_is_defined_just_above_the_floor():
    table = _ratio_table()
    table.loc[table["tool"] == BINDING, ["issr_10", "issr_25"]] = 0.0625
    out = ebc.compute_issr_ratios(table)
    assert list(out["R10"]) == pytest.approx([12.8, 1.0, 3.2], rel=1e-12)
    assert list(out["R25"]) == pytest.approx([9.6, 1.0, 4.0], rel=1e-12)


def test_issr_ratio_does_not_mutate_its_input():
    table = _ratio_table()
    snapshot = table.copy()
    ebc.compute_issr_ratios(table)
    pd.testing.assert_frame_equal(table, snapshot)


# ---------------------------------------------------------------------------
# spearman_correlation_matrix
# ---------------------------------------------------------------------------


def _scores_with_gaps_and_ties() -> pd.DataFrame:
    generator = np.random.default_rng(3)
    n = 30
    a = np.round(generator.normal(size=n), 1)
    b = np.round(0.6 * a + generator.normal(size=n), 1)
    c = np.round(-0.4 * a + generator.normal(size=n), 1)
    frame = pd.DataFrame({"a": a, "b": b, "c": c, "d": np.round(generator.normal(size=n), 1)})
    # Different gaps per column, so each pair has its own complete-row set.
    frame.loc[[0, 1, 2], "a"] = np.nan
    frame.loc[[2, 3, 4, 5], "b"] = np.nan
    frame.loc[[6], "c"] = np.nan
    return frame


TOOLS_ABCD = {"A": "a", "B": "b", "C": "c", "D": "d"}


def test_spearman_lists_each_unordered_tool_pair_once_in_tool_order():
    out = ebc.spearman_correlation_matrix(_scores_with_gaps_and_ties(), TOOLS_ABCD)
    assert list(out.columns) == ["tool_a", "tool_b", "rho", "p_value"]
    pairs = list(zip(out["tool_a"], out["tool_b"]))
    assert pairs == [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]


def test_spearman_rho_and_p_match_a_rank_pearson_oracle_on_pairwise_complete_rows():
    frame = _scores_with_gaps_and_ties()
    # Precondition: ties exist, so the average-rank path is exercised.
    assert frame["a"].dropna().duplicated().any()
    out = ebc.spearman_correlation_matrix(frame, TOOLS_ABCD)
    assert len(out) == 6
    for _, row in out.iterrows():
        col_a, col_b = TOOLS_ABCD[row["tool_a"]], TOOLS_ABCD[row["tool_b"]]
        pair = frame[[col_a, col_b]].dropna()
        rho, p_value = _spearman_oracle(pair[col_a].to_numpy(), pair[col_b].to_numpy())
        assert row["rho"] == pytest.approx(rho, abs=1e-12)
        assert row["p_value"] == pytest.approx(p_value, abs=1e-12)
        # And the library the module documents itself against agrees too.
        ref = spearmanr(pair[col_a], pair[col_b])
        assert row["rho"] == pytest.approx(float(ref[0]), abs=1e-12)


def test_spearman_keeps_a_pair_with_five_complete_rows_and_drops_one_with_four():
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "b": [2.0, 1.0, 4.0, 3.0, 5.0, np.nan, np.nan, np.nan],
            "c": [np.nan, np.nan, np.nan, 1.0, 3.0, 2.0, 4.0, np.nan],
        }
    )
    # Complete rows: A-B 5 (kept), A-C 4 (dropped), B-C 2 (dropped).
    out = ebc.spearman_correlation_matrix(frame, {"A": "a", "B": "b", "C": "c"})
    assert list(zip(out["tool_a"], out["tool_b"])) == [("A", "B")]
    rho, p_value = _spearman_oracle(frame["a"][:5], frame["b"][:5])
    assert out["rho"].iloc[0] == pytest.approx(rho, abs=1e-12)
    assert out["p_value"].iloc[0] == pytest.approx(p_value, abs=1e-12)


def test_spearman_reports_nan_rho_for_a_constant_tool_rather_than_a_number():
    frame = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "flat": [0.5] * 6, "b": [6.0, 5, 4, 3, 2, 1]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ebc.spearman_correlation_matrix(frame, {"A": "a", "Flat": "flat", "B": "b"})
    assert list(zip(out["tool_a"], out["tool_b"])) == [("A", "Flat"), ("A", "B"), ("Flat", "B")]
    by_pair = {(r.tool_a, r.tool_b): r for r in out.itertuples()}
    for pair in [("A", "Flat"), ("Flat", "B")]:
        assert math.isnan(by_pair[pair].rho)
        assert math.isnan(by_pair[pair].p_value)
    assert by_pair[("A", "B")].rho == pytest.approx(-1.0, abs=1e-12)


def test_spearman_rho_is_symmetric_in_tool_order():
    frame = _scores_with_gaps_and_ties()
    forward = ebc.spearman_correlation_matrix(frame, TOOLS_ABCD)
    reverse = ebc.spearman_correlation_matrix(frame, dict(reversed(list(TOOLS_ABCD.items()))))
    assert len(forward) == len(reverse) == 6
    back = {frozenset((r.tool_a, r.tool_b)): (r.rho, r.p_value) for r in reverse.itertuples()}
    for r in forward.itertuples():
        rho, p_value = back[frozenset((r.tool_a, r.tool_b))]
        assert r.rho == pytest.approx(rho, abs=1e-12)
        assert r.p_value == pytest.approx(p_value, abs=1e-12)


def test_spearman_with_no_qualifying_pair_is_empty():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 1.0, 2.0]})
    assert ebc.spearman_correlation_matrix(frame, {"A": "a", "B": "b"}).empty


# ---------------------------------------------------------------------------
# per_virus_metrics
# ---------------------------------------------------------------------------


def _virus_frame() -> pd.DataFrame:
    generator = np.random.default_rng(11)
    rows = []
    # Inserted out of alphabetical order so sorting is observable.
    for virus, n in (("Zika", 14), ("Dengue", 12), ("HIV", 9)):
        for i in range(n):
            rows.append(
                {
                    "virus": virus,
                    "label": float(i % 3 == 0),
                    "s1": float(generator.random()),
                    "s2": float(generator.random()),
                }
            )
    # Rows without a virus join no group, even though they are high scoring positives.
    rows += [{"virus": np.nan, "label": 1.0, "s1": 0.999, "s2": 0.999} for _ in range(3)]
    # A Zika row with no label is excluded from both tools' Zika groups.
    rows.append({"virus": "Zika", "label": np.nan, "s1": 0.998, "s2": 0.998})
    frame = pd.DataFrame(rows)
    # One Zika positive loses its s2 score, so Zika/T2 has 13 valid rows, T1 has 14.
    first_zika_positive = frame.index[(frame["virus"] == "Zika") & (frame["label"] == 1.0)][0]
    frame.loc[first_zika_positive, "s2"] = np.nan
    return frame


def test_per_virus_without_a_virus_column_is_empty():
    frame = _virus_frame().drop(columns=["virus"])
    assert ebc.per_virus_metrics(frame, {"T1": "s1"}).empty


def test_per_virus_metrics_match_sklearn_and_a_hand_top_k_count_per_group():
    frame = _virus_frame()
    out = ebc.per_virus_metrics(frame, {"T1": "s1", "T2": "s2"})
    assert list(out.columns[:3]) == ["tool", "virus", "n"]
    # HIV (9 rows) is below the floor; viruses sorted; tools in the given order.
    assert list(zip(out["tool"], out["virus"], out["n"])) == [
        ("T1", "Dengue", 12),
        ("T2", "Dengue", 12),
        ("T1", "Zika", 14),
        ("T2", "Zika", 13),
    ]
    for _, row in out.iterrows():
        col = {"T1": "s1", "T2": "s2"}[row["tool"]]
        group = frame.loc[frame["virus"] == row["virus"], ["label", col]].dropna()
        labels = group["label"].to_numpy()
        scores = group[col].to_numpy()
        assert row["auc_roc"] == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)
        assert row["auc_pr"] == pytest.approx(average_precision_score(labels, scores), abs=1e-12)
        for k in (10, 25):
            issr = _issr_oracle(labels, scores, k)
            assert row[f"issr_{k}"] == pytest.approx(issr, abs=1e-12)
            assert row[f"precision_{k}"] == pytest.approx(issr, abs=1e-12)
            assert row[f"recall_{k}"] == pytest.approx(_recall_oracle(labels, scores, k), abs=1e-12)


def test_per_virus_keeps_a_group_with_ten_valid_rows_and_drops_one_with_nine():
    generator = np.random.default_rng(5)
    rows = []
    for virus in ("V10", "V9_score_gap", "V9_label_gap"):
        for i in range(10):
            rows.append({"virus": virus, "label": float(i % 2), "s": float(generator.random())})
    frame = pd.DataFrame(rows)
    frame.loc[frame.index[frame["virus"] == "V9_score_gap"][0], "s"] = np.nan
    frame.loc[frame.index[frame["virus"] == "V9_label_gap"][0], "label"] = np.nan
    out = ebc.per_virus_metrics(frame, {"T": "s"})
    assert list(zip(out["virus"], out["n"])) == [("V10", 10)]


def test_per_virus_single_class_group_is_kept_with_undefined_auc():
    frame = pd.DataFrame(
        {"virus": ["Mono"] * 10, "label": [1.0] * 10, "s": np.linspace(0.1, 1.0, 10)}
    )
    out = ebc.per_virus_metrics(frame, {"T": "s"})
    assert len(out) == 1
    row = out.iloc[0]
    assert (row["tool"], row["virus"], row["n"]) == ("T", "Mono", 10)
    assert math.isnan(row["auc_roc"]) and math.isnan(row["auc_pr"])
    assert row["issr_10"] == 1.0 and row["issr_25"] == 1.0


def test_per_virus_returns_empty_when_no_group_qualifies():
    frame = _virus_frame()
    frame = frame.loc[frame["virus"] == "HIV"]
    out = ebc.per_virus_metrics(frame, {"T1": "s1", "T2": "s2"})
    assert isinstance(out, pd.DataFrame) and out.empty


# ---------------------------------------------------------------------------
# bootstrap_metric_diff
# ---------------------------------------------------------------------------


def _paired_inputs(n: int, seed: int):
    generator = np.random.default_rng(seed)
    y = np.array([i % 2 for i in range(n)])
    a = y * 0.4 + generator.random(n)
    b = generator.random(n)
    return y, a, b


@pytest.mark.parametrize(("n", "n_boot", "seed"), [(16, 60, 0), (9, 45, 7), (25, 30, 123)])
def test_bootstrap_matches_a_longhand_paired_resampling_oracle(n, n_boot, seed):
    y, a, b = _paired_inputs(n, seed + 1)
    got = ebc.bootstrap_metric_diff(y, a, b, average_precision_score, n_bootstrap=n_boot, seed=seed)
    expected, kept = _bootstrap_oracle(y, a, b, average_precision_score, n_boot, seed)
    assert kept > 0
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-15)
    assert got[1] <= got[0] <= got[2]


def test_bootstrap_resamples_all_rows_with_replacement_and_pairs_both_arms():
    n = 12
    y = np.array([i % 2 for i in range(n)])
    ids = np.arange(n, dtype=float)
    calls = []

    def recorder(y_b, s):
        calls.append((np.array(y_b), np.array(s)))
        return 0.0

    ebc.bootstrap_metric_diff(y, ids, ids + 100.0, recorder, n_bootstrap=200, seed=5)
    assert calls and len(calls) % 2 == 0
    seen: set[int] = set()
    any_repeat = False
    for (y_a, s_a), (y_b, s_b) in zip(calls[0::2], calls[1::2]):
        drawn = s_a.astype(int)
        assert len(drawn) == n
        assert set(drawn.tolist()) <= set(range(n))
        assert np.array_equal(s_b, s_a + 100.0), "arm b was scored on a different draw"
        assert np.array_equal(y_a, y[drawn]) and np.array_equal(y_b, y[drawn])
        seen |= set(drawn.tolist())
        any_repeat |= len(set(drawn.tolist())) < n
    assert seen == set(range(n)), "some rows can never be drawn"
    assert any_repeat, "no resample repeated a row, so it was not drawn with replacement"


def test_bootstrap_difference_is_a_minus_b():
    y, a, b = _paired_inputs(20, 2)
    forward = ebc.bootstrap_metric_diff(y, a, b, average_precision_score, n_bootstrap=101, seed=3)
    reverse = ebc.bootstrap_metric_diff(y, b, a, average_precision_score, n_bootstrap=101, seed=3)
    assert reverse[0] == pytest.approx(-forward[0], abs=1e-12)
    assert reverse[1] == pytest.approx(-forward[2], abs=1e-12)
    assert reverse[2] == pytest.approx(-forward[1], abs=1e-12)
    # a ranks every positive above every negative and b inverts it, so on every
    # two-class resample AUC-ROC(a) - AUC-ROC(b) is exactly 1 - 0.
    perfect = y.astype(float) + np.linspace(0.0, 0.5, len(y))
    inverted = -perfect
    exact = ebc.bootstrap_metric_diff(y, perfect, inverted, roc_auc_score, n_bootstrap=40, seed=1)
    assert exact == (1.0, 1.0, 1.0)
    same = ebc.bootstrap_metric_diff(y, a, a, average_precision_score, n_bootstrap=40, seed=1)
    assert same == (0.0, 0.0, 0.0)


def test_bootstrap_is_deterministic_under_a_fixed_seed_and_moves_with_the_seed():
    y, a, b = _paired_inputs(18, 4)
    first = ebc.bootstrap_metric_diff(y, a, b, average_precision_score, n_bootstrap=50, seed=9)
    again = ebc.bootstrap_metric_diff(y, a, b, average_precision_score, n_bootstrap=50, seed=9)
    other = ebc.bootstrap_metric_diff(y, a, b, average_precision_score, n_bootstrap=50, seed=10)
    assert first == again
    assert first != other


def test_bootstrap_defaults_are_2000_resamples_and_seed_42():
    n = 40
    y = np.array([i % 2 for i in range(n)])
    generator = np.random.default_rng(8)
    a = generator.random(n)
    b = generator.random(n)
    calls = []

    def mean_positive_score(y_b, s):
        calls.append(1)
        return float(s[y_b == 1].mean())

    default = ebc.bootstrap_metric_diff(y, a, b, mean_positive_score)
    # 40 balanced rows never resample to one class under this seed, so every one of
    # the resamples is scored, once per arm.
    assert len(calls) == 2 * 2000
    explicit = ebc.bootstrap_metric_diff(y, a, b, mean_positive_score, n_bootstrap=2000, seed=42)
    assert default == explicit


def test_bootstrap_skips_single_class_resamples_instead_of_scoring_them():
    # Four rows, two positive: about 1 in 8 resamples is single-class, and
    # roc_auc_score raises on those, so scoring one would fail loudly.
    y = np.array([0, 1, 0, 1])
    a = np.array([0.1, 0.8, 0.3, 0.6])
    b = np.array([0.7, 0.2, 0.4, 0.5])
    calls = []

    def counted_auc(y_b, s):
        calls.append(1)
        return roc_auc_score(y_b, s)

    got = ebc.bootstrap_metric_diff(y, a, b, counted_auc, n_bootstrap=80, seed=2)
    expected, kept = _bootstrap_oracle(y, a, b, roc_auc_score, 80, 2)
    assert 0 < kept < 80, "precondition: some resamples must be single-class"
    assert len(calls) == 2 * kept
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_bootstrap_with_no_two_class_resample_returns_a_nan_triple():
    y = np.ones(6, dtype=int)
    s = np.linspace(0.0, 1.0, 6)
    got = ebc.bootstrap_metric_diff(y, s, s[::-1], roc_auc_score, n_bootstrap=20, seed=0)
    assert len(got) == 3 and all(math.isnan(v) for v in got)


# ---------------------------------------------------------------------------
# run_bootstrap_comparisons
# ---------------------------------------------------------------------------


def _comparison_frame() -> pd.DataFrame:
    generator = np.random.default_rng(21)
    n = 34
    label = np.array([i % 2 for i in range(n)], dtype=float)
    frame = pd.DataFrame(
        {
            "label": label,
            "rf_oof_score": label * 0.3 + generator.random(n),
            "a": label * 0.1 + generator.random(n),
            "b": generator.random(n),
            "c": generator.random(n),
        }
    )
    frame.loc[0, "label"] = np.nan  # excluded from every comparison
    frame.loc[1, "rf_oof_score"] = np.nan  # excluded from every comparison
    frame.loc[2:5, "a"] = np.nan  # A: 28 complete rows
    frame.loc[2:13, "b"] = np.nan  # B: exactly 20 complete rows, kept
    frame.loc[2:14, "c"] = np.nan  # C: 19 complete rows, skipped
    return frame


# The reference sits in the middle of the mapping: excluded by NAME, order kept.
COMPARISON_TOOLS = {"A": "a", REFERENCE: "rf_oof_score", "B": "b", "C": "c"}


def _issr10(y_true, scores):
    return issr_at_k(y_true, scores, 10)


def test_run_bootstrap_rows_equal_direct_bootstrap_calls_on_the_complete_rows():
    frame = _comparison_frame()
    out = ebc.run_bootstrap_comparisons(frame, COMPARISON_TOOLS, n_bootstrap=40)
    assert list(out.columns) == BOOT_COLUMNS
    assert list(out["comparison"]) == [f"{REFERENCE} vs A", f"{REFERENCE} vs B"]
    for (_, row), col, expected_n in zip(out.iterrows(), ["a", "b"], [28, 20]):
        valid = frame.dropna(subset=["label", "rf_oof_score", col])
        assert len(valid) == expected_n
        y = valid["label"].to_numpy()
        ref = valid["rf_oof_score"].to_numpy()
        tool = valid[col].to_numpy()
        ap = ebc.bootstrap_metric_diff(y, ref, tool, average_precision_score, 40)
        issr = ebc.bootstrap_metric_diff(y, ref, tool, _issr10, 40)
        assert (
            row["auc_pr_diff_mean"],
            row["auc_pr_diff_ci_low"],
            row["auc_pr_diff_ci_high"],
        ) == ap
        assert (
            row["issr10_diff_mean"],
            row["issr10_diff_ci_low"],
            row["issr10_diff_ci_high"],
        ) == issr
        assert row["auc_pr_significant"] == _excludes_zero(ap[1], ap[2])
        assert row["issr10_significant"] == _excludes_zero(issr[1], issr[2])


def test_run_bootstrap_difference_is_reference_minus_tool():
    n = 24
    label = np.array([i % 2 for i in range(n)], dtype=float)
    ref = label + np.linspace(0.0, 0.5, n)
    frame = pd.DataFrame({"label": label, "rf_oof_score": ref, "worse": -ref})
    out = ebc.run_bootstrap_comparisons(frame, {"Worse": "worse"}, n_bootstrap=30)
    row = out.iloc[0]
    assert row["comparison"] == f"{REFERENCE} vs Worse"
    assert row["auc_pr_diff_ci_low"] > 0
    assert row["auc_pr_significant"] == "yes"
    # Top 10% of the reference is all positive and of the inverse all negative.
    assert (row["issr10_diff_mean"], row["issr10_diff_ci_low"]) == (1.0, 1.0)
    assert row["issr10_significant"] == "yes"


class _SpyBootstrap:
    """Stands in for bootstrap_metric_diff: records each call, replays set results."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.results.pop(0)


def _one_tool_frame(n: int = 25) -> pd.DataFrame:
    generator = np.random.default_rng(31)
    return pd.DataFrame(
        {
            "label": [float(i % 2) for i in range(n)],
            "rf_oof_score": generator.random(n),
            "t": generator.random(n),
        }
    )


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (0.1, 0.4, "yes"),
        (-0.4, -0.1, "yes"),
        (1.5, 2.0, "yes"),
        (-0.2, 0.3, "no"),
        (0.0, 0.3, "no"),
        (-0.3, 0.0, "no"),
        (0.0, 0.0, "no"),
    ],
)
def test_run_bootstrap_significance_means_the_ci_excludes_zero(monkeypatch, lo, hi, expected):
    spy = _SpyBootstrap([(0.5, lo, hi), (0.25, lo, hi)])
    monkeypatch.setattr(ebc, "bootstrap_metric_diff", spy)
    out = ebc.run_bootstrap_comparisons(_one_tool_frame(), {"T": "t"}, n_bootstrap=7)
    row = out.iloc[0]
    assert row["auc_pr_significant"] == expected
    assert row["issr10_significant"] == expected


def test_run_bootstrap_reports_each_metric_from_its_own_bootstrap(monkeypatch):
    spy = _SpyBootstrap([(0.15, 0.1, 0.2), (-0.05, -0.1, 0.2)])
    monkeypatch.setattr(ebc, "bootstrap_metric_diff", spy)
    row = ebc.run_bootstrap_comparisons(_one_tool_frame(), {"T": "t"}, n_bootstrap=7).iloc[0]
    assert (row["auc_pr_diff_mean"], row["auc_pr_diff_ci_low"], row["auc_pr_diff_ci_high"]) == (
        0.15,
        0.1,
        0.2,
    )
    assert (row["issr10_diff_mean"], row["issr10_diff_ci_low"], row["issr10_diff_ci_high"]) == (
        -0.05,
        -0.1,
        0.2,
    )
    assert (row["auc_pr_significant"], row["issr10_significant"]) == ("yes", "no")


def test_run_bootstrap_passes_reference_first_and_2000_resamples_by_default(monkeypatch):
    frame = _one_tool_frame()
    spy = _SpyBootstrap([(0.0, -1.0, 1.0), (0.0, -1.0, 1.0)])
    monkeypatch.setattr(ebc, "bootstrap_metric_diff", spy)
    ebc.run_bootstrap_comparisons(frame, {"T": "t"})
    assert len(spy.calls) == 2
    for args, kwargs in spy.calls:
        bound = dict(zip(["y_true", "scores_a", "scores_b", "metric_fn", "n_bootstrap"], args))
        bound.update(kwargs)
        assert np.array_equal(bound["y_true"], frame["label"].to_numpy())
        assert np.array_equal(bound["scores_a"], frame["rf_oof_score"].to_numpy())
        assert np.array_equal(bound["scores_b"], frame["t"].to_numpy())
        assert bound["n_bootstrap"] == 2000
    assert spy.calls[0][0][3] is average_precision_score
    # The second metric is ISSR at the top 10%: with 100 rows whose top 10 are all
    # positive and whose 11th is negative, k=10 gives exactly 1.0 and k=11 would not.
    issr_metric = spy.calls[1][0][3]
    probe_y = np.array([1] * 10 + [0] * 90)
    probe_s = np.linspace(1.0, 0.0, 100)
    assert issr_metric(probe_y, probe_s) == 1.0
    assert issr_metric(probe_y, probe_s) == issr_at_k(probe_y, probe_s, 10)
