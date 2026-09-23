"""tests/test_allele_stratified.py
================================
Unit tests for allele-stratified concordance metrics and bootstrap CI logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_allele_stratified import (
    _adjudicate_across_partitions,
    _adjudicate_partition,
    _stratum_dominance,
    compute_stratum_concordance,
    compute_stratified_metrics,
    stratified_bootstrap_ci,
)


def test_compute_stratum_concordance_perfect():
    pos = np.array([0.9, 0.8])
    neg = np.array([0.2, 0.1, 0.3])
    auc, pairs = compute_stratum_concordance(pos, neg)
    assert pairs == 6
    assert auc == 1.0


def test_compute_stratum_concordance_inverted():
    pos = np.array([0.1, 0.2])
    neg = np.array([0.8, 0.9])
    auc, pairs = compute_stratum_concordance(pos, neg)
    assert pairs == 4
    assert auc == 0.0


def test_compute_stratum_concordance_ties():
    pos = np.array([0.5, 0.8])
    neg = np.array([0.5, 0.2])
    # pairs:
    # (0.5, 0.5) -> tie (0.5)
    # (0.5, 0.2) -> pos > neg (1.0)
    # (0.8, 0.5) -> pos > neg (1.0)
    # (0.8, 0.2) -> pos > neg (1.0)
    # Total: 3.5 / 4 = 0.875
    auc, pairs = compute_stratum_concordance(pos, neg)
    assert pairs == 4
    assert auc == 0.875


def test_compute_stratum_concordance_empty():
    auc, pairs = compute_stratum_concordance(np.array([]), np.array([0.5]))
    assert pairs == 0
    assert np.isnan(auc)


def test_compute_stratified_metrics_multi_strata():
    # Stratum A: 2 pos, 2 neg (4 pairs), perfect (auc = 1.0)
    # Stratum B: 1 pos, 2 neg (2 pairs), inverted (auc = 0.0)
    # Stratum C: 2 pos, 0 neg (0 pairs, single class)
    # Pair-weighted (MH) concordance: (4 * 1.0 + 2 * 0.0) / 6 = 4/6 = 0.6667
    # Row-weighted concordance: (4 samples * 1.0 + 3 samples * 0.0) / 7 = 4/7 = 0.5714
    data = [
        {"allele": "A", "label": 1, "score": 0.9},
        {"allele": "A", "label": 1, "score": 0.8},
        {"allele": "A", "label": 0, "score": 0.2},
        {"allele": "A", "label": 0, "score": 0.1},
        {"allele": "B", "label": 1, "score": 0.1},
        {"allele": "B", "label": 0, "score": 0.8},
        {"allele": "B", "label": 0, "score": 0.9},
        {"allele": "C", "label": 1, "score": 0.5},
        {"allele": "C", "label": 1, "score": 0.6},
    ]
    df = pd.DataFrame(data)
    res = compute_stratified_metrics(df, score_col="score")

    assert res["total_pairs"] == 6
    assert res["two_class_strata_count"] == 2
    assert res["total_strata_count"] == 3
    assert pytest.approx(res["mh_concordance"], abs=1e-4) == 4 / 6
    assert pytest.approx(res["row_weighted_concordance"], abs=1e-4) == 4 / 7


def test_stratified_bootstrap_ci():
    # Verify deterministic reproducibility and valid interval
    data = [
        {"allele": "HLA-A*02:01", "label": 1, "score": 0.8},
        {"allele": "HLA-A*02:01", "label": 1, "score": 0.7},
        {"allele": "HLA-A*02:01", "label": 0, "score": 0.3},
        {"allele": "HLA-A*02:01", "label": 0, "score": 0.2},
    ]
    df = pd.DataFrame(data)
    low, high = stratified_bootstrap_ci(df, score_col="score", n_resamples=500, seed=12345)
    assert 0.0 <= low <= high <= 1.0
    assert low == 1.0
    assert high == 1.0


# ---------------------------------------------------------------------------
# Paired delta CI.
#
# The script reports two MARGINAL intervals, one per score column, drawn from
# two deliberately different RNG seeds. Overlapping marginals are not a test of
# their difference, and the difference is what the within-allele question asks.
# These fixtures are synthetic on purpose: the real cohort lives under results/,
# which is read-denied here, and a test that needs it would not run in CI either.
# ---------------------------------------------------------------------------

ALLELES = ("HLA-A*02:01", "HLA-A*01:01", "HLA-B*07:02", "HLA-B*44:02")


def _synthetic_cohort(kind: str, seed: int = 7, n_per_class: int = 15) -> pd.DataFrame:
    """Four strata, n_per_class positives and negatives each.

    kind='monotone': model is a strictly increasing affine transform of raw, so
                     every pairwise comparison has the same sign in both arms and
                     the paired delta is EXACTLY zero in every resample.
    kind='noise':    model is drawn independently of the label; raw is informative.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for allele in ALLELES:
        for label in (1, 0):
            for _ in range(n_per_class):
                raw = float(rng.normal(0.6 if label == 1 else 0.0, 1.0))
                model = 2.0 * raw + 1.0 if kind == "monotone" else float(rng.normal(0.0, 1.0))
                rows.append(
                    {
                        "allele": allele,
                        "label": label,
                        "immunogenicity_score": model,
                        "presentation_score": raw,
                    }
                )
    return pd.DataFrame(rows)


def test_paired_delta_is_zero_for_monotone_transform():
    """An order-preserving transform cannot change concordance, so delta is 0."""
    res = stratified_bootstrap_ci(
        _synthetic_cohort("monotone"),
        score_col="immunogenicity_score",
        compare_col="presentation_score",
        n_resamples=500,
        seed=20260909,
    )
    assert res["delta_point"] == pytest.approx(0.0, abs=1e-12)
    assert res["delta_ci"][0] == pytest.approx(0.0, abs=1e-12)
    assert res["delta_ci"][1] == pytest.approx(0.0, abs=1e-12)
    assert res["delta_excludes_zero"] is False
    assert res["score_ci"] == pytest.approx(res["compare_ci"], abs=1e-12)


def test_paired_delta_strictly_below_zero_for_noise_model():
    """A model with no signal must lose to an informative raw score, detectably.

    The final assertion is the point of this test: the fixture is required to keep
    the two MARGINAL intervals overlapping, which is the exact condition under
    which the old two-interval reporting was inconclusive and the paired contrast
    earns its keep.
    """
    res = stratified_bootstrap_ci(
        _synthetic_cohort("noise"),
        score_col="immunogenicity_score",
        compare_col="presentation_score",
        n_resamples=1000,
        seed=20260909,
    )
    assert res["delta_point"] < 0.0
    assert res["delta_ci"][1] < 0.0, f"upper bound {res['delta_ci'][1]} is not below 0"
    assert res["delta_excludes_zero"] is True

    lo_m, hi_m = res["score_ci"]
    lo_r, hi_r = res["compare_ci"]
    assert lo_m < hi_r and lo_r < hi_m, (
        "fixture must keep the MARGINAL intervals overlapping - that is the "
        "condition under which the paired test is the only one that can answer"
    )


def test_paired_marginal_matches_unpaired_call():
    """Drawing indices instead of values must not perturb the RNG stream.

    If this fails, every already-published marginal CI has silently moved.
    """
    df = _synthetic_cohort("noise")
    marginal = stratified_bootstrap_ci(
        df, score_col="immunogenicity_score", n_resamples=400, seed=20260909
    )
    paired = stratified_bootstrap_ci(
        df,
        score_col="immunogenicity_score",
        compare_col="presentation_score",
        n_resamples=400,
        seed=20260909,
    )
    assert paired["score_ci"] == pytest.approx(marginal, abs=1e-12)


def test_backwards_compatible_two_tuple():
    """Without compare_col the historical 2-tuple shape is preserved."""
    out = stratified_bootstrap_ci(
        _synthetic_cohort("noise"), score_col="immunogenicity_score", n_resamples=200, seed=1
    )
    assert isinstance(out, tuple) and len(out) == 2
    assert 0.0 <= out[0] <= out[1] <= 1.0


def test_paired_mode_uses_one_shared_row_mask():
    """Per-column dropna would desynchronise the two arms and break the pairing.

    NaNs are injected into ONE column only. Under a shared mask the surviving rows
    are identical in both arms, so a monotone fixture still gives exactly zero.
    """
    df = _synthetic_cohort("monotone")
    df.loc[df.index[:5], "presentation_score"] = np.nan
    res = stratified_bootstrap_ci(
        df,
        score_col="immunogenicity_score",
        compare_col="presentation_score",
        n_resamples=300,
        seed=20260909,
    )
    assert res["delta_point"] == pytest.approx(0.0, abs=1e-12)
    assert res["paired_pairs"] > 0


def test_paired_mode_with_no_two_class_stratum_returns_nan_dict():
    df = _synthetic_cohort("noise")
    df["label"] = 1
    res = stratified_bootstrap_ci(
        df,
        score_col="immunogenicity_score",
        compare_col="presentation_score",
        n_resamples=50,
        seed=1,
    )
    assert np.isnan(res["delta_point"])
    assert res["paired_pairs"] == 0
    assert res["delta_excludes_zero"] is False


def test_stratum_dominance_flags_a_single_dominant_allele():
    strata = [
        {"allele": "HLA-A*02:01", "pairs": 1666},
        {"allele": "HLA-A*01:01", "pairs": 150},
        {"allele": "HLA-B*07:02", "pairs": 100},
        {"allele": "HLA-B*44:02", "pairs": 0},
    ]
    dom = _stratum_dominance(strata, total_pairs=1916)
    assert dom["top_allele"] == "HLA-A*02:01"
    assert dom["top_allele_pairs"] == 1666
    assert dom["top_allele_pair_share"] == pytest.approx(1666 / 1916)
    assert dom["n_strata_contributing"] == 3, "a zero-pair stratum must not count"
    assert dom["dominance_warning"] is True


def test_stratum_dominance_silent_when_evenly_spread():
    strata = [{"allele": f"A{i}", "pairs": 100} for i in range(5)]
    dom = _stratum_dominance(strata, total_pairs=500)
    assert dom["dominance_warning"] is False
    assert dom["top_allele_pair_share"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Section 3 of the generated report must be DERIVED from the run. It was once
# hardcoded, so the report asserted "Statistical Power Restored ... completely
# surpassing" and "Directly resolves" the confound on every run - including runs
# whose own adjudication fell through to the neither-hypothesis branch, which is
# what the shipped cohort actually produces. These pin the derivation.
# ---------------------------------------------------------------------------


def _report_for(adjudication: str, per_partition: dict | None = None) -> str:
    from scripts.evaluate_allele_stratified import _generate_markdown_report

    def partition(pairs, model_c, raw_c):
        return {
            "n_samples": 442,
            "n_pos": 221,
            "n_neg": 221,
            "total_same_allele_pairs": pairs,
            "unstratified_model_auc": 0.3787,
            "model_mh_concordance": model_c,
            "model_mh_ci": (0.4181, 0.6759),
            "raw_mh_concordance": raw_c,
            "raw_mh_ci": (0.5783, 0.7923),
            "paired_delta_model_minus_raw": model_c - raw_c,
            "paired_delta_ci": (-0.2687, -0.0089),
            "paired_delta_excludes_zero": True,
            "strata_detail": [
                {"allele": "HLA-A*02:01", "n_pos": 17, "n_neg": 98,
                 "pairs": 1666, "concordance": 0.5744},
            ],
            "stratum_dominance": {
                "dominance_warning": False,
                "top_allele": "HLA-A*02:01",
                "top_allele_pairs": 1666,
                "top_allele_pair_share": 0.87,
                "n_strata_contributing": 17,
            },
        }

    res = {
        "adjudication": adjudication,
        "verdict_summary": "summary",
        "dataset": "fixture.csv",
        "bootstrap_resamples": 1000,
        "bootstrap_seed": 42,
        "partitions": {
            "all_same_allele": partition(2621, 0.5616, 0.6608),
            "human_hla_only": partition(1916, 0.5475, 0.6863),
        },
    }
    if per_partition is not None:
        res["per_partition_adjudication"] = per_partition
    return _generate_markdown_report(res)


def test_section_three_never_claims_statistical_power():
    """Pair counts are a product of n_pos and n_neg, not a sample size."""
    for adjudication in (
        "HYPOTHESIS_1_SUPPORTED",
        "HYPOTHESIS_2_SUPPORTED",
        "NEITHER_HYPOTHESIS_MATCHED",
    ):
        report = _report_for(adjudication)
        assert "Statistical Power Restored" not in report
        assert "completely surpassing" not in report
        assert "not a sample size" in report


def test_a_fallthrough_adjudication_does_not_claim_the_confound_is_resolved():
    report = _report_for("NEITHER_HYPOTHESIS_MATCHED")
    assert "Confound NOT resolved" in report
    assert "fallthrough, not a" in report
    assert "Directly resolves" not in report


def test_a_matched_hypothesis_does_report_a_resolution():
    for adjudication in ("HYPOTHESIS_1_SUPPORTED", "HYPOTHESIS_2_SUPPORTED"):
        report = _report_for(adjudication)
        assert "Confound resolution" in report
        assert "Confound NOT resolved" not in report
        assert adjudication in report


# ---------------------------------------------------------------------------
# Cross-partition adjudication.
#
# The run-level verdict used to be read off partition_results["human_hla_only"]
# alone, while the comment above it said the primary test was on two partitions.
# A run could therefore print "Hypothesis 1 ... STRONGLY SUPPORTED" off one
# partition while another partition computed in the SAME run contradicted it.
# The dissenting partition never reached the verdict at all.
#
# The fixture below is the SESTRAV influenza-original 179-row external cohort,
# Mantel-Haenszel within-allele concordance, as recorded in
# results/influenza_original_stratified_metrics.json. Values are named by COHORT
# and STATISTIC, never matched by their first four decimals: this repo carries
# unrelated quantities that agree to four places.
# ---------------------------------------------------------------------------

INFLUENZA_ORIGINAL_179_MH = {
    # partition: (model MH concordance, raw presentation MH concordance)
    "all_same_allele": (0.5616883116883117, 0.5097402597402597),
    "human_hla_only": (0.515625, 0.4895833333333333),
    "models_ten_only": (0.45528455284552843, 0.4878048780487805),
}


def _partitions(spec: dict) -> dict:
    """Minimal partition records carrying only what the adjudicator reads."""
    return {
        name: {"model_mh_concordance": model_c, "raw_mh_concordance": raw_c}
        for name, (model_c, raw_c) in spec.items()
    }


def test_adjudicate_partition_pins_each_preregistered_branch():
    """Per-partition thresholds are unchanged; only the non-finite case is new."""
    assert _adjudicate_partition(0.56, 0.51) == "HYPOTHESIS_1_SUPPORTED"
    assert _adjudicate_partition(0.50, 0.45) == "HYPOTHESIS_1_SUPPORTED"
    assert _adjudicate_partition(0.42, 0.62) == "HYPOTHESIS_2_SUPPORTED"
    assert _adjudicate_partition(0.42, 0.49) == "NEITHER_HYPOTHESIS_MATCHED"
    assert _adjudicate_partition(0.62, 0.62) == "NEITHER_HYPOTHESIS_MATCHED"
    # A partition with no two-class stratum has a NaN concordance. Every
    # comparison against NaN is False, so the un-guarded form silently reported
    # an absence of measurement as a fallthrough VERDICT.
    assert _adjudicate_partition(float("nan"), 0.51) == "NOT_ADJUDICABLE"
    assert _adjudicate_partition(0.56, float("nan")) == "NOT_ADJUDICABLE"


def test_a_dissenting_partition_blocks_a_supported_verdict():
    """MUTATION GUARD for the single-partition read.

    These are the measured influenza-original 179-row MH concordances. Two
    partitions adjudicate HYPOTHESIS_1_SUPPORTED and models_ten_only, the
    partition restricted to the ten alleles the model actually has features for,
    adjudicates neither. Restore the old read of human_hla_only alone and this
    test fails: it recovers HYPOTHESIS_1_SUPPORTED and the STRONGLY SUPPORTED
    sentence.
    """
    adjudication, verdict, per_partition = _adjudicate_across_partitions(
        _partitions(INFLUENZA_ORIGINAL_179_MH)
    )
    assert adjudication == "PARTITIONS_DISAGREE"
    assert "STRONGLY SUPPORTED" not in verdict
    assert per_partition["human_hla_only"]["adjudication"] == "HYPOTHESIS_1_SUPPORTED"
    assert per_partition["models_ten_only"]["adjudication"] == "NEITHER_HYPOTHESIS_MATCHED"


def test_disagreement_does_not_flip_to_the_opposite_verdict():
    """Promoting the dissenting partition is the same defect with the sign flipped."""
    adjudication, verdict, _ = _adjudicate_across_partitions(
        _partitions(INFLUENZA_ORIGINAL_179_MH)
    )
    assert adjudication not in (
        "HYPOTHESIS_1_SUPPORTED",
        "HYPOTHESIS_2_SUPPORTED",
        "NEITHER_HYPOTHESIS_MATCHED",
    )
    assert "majority" in verdict


def test_the_verdict_names_every_partition_it_adjudicated():
    """A verdict a reader cannot audit against its own population is not auditable."""
    spec = INFLUENZA_ORIGINAL_179_MH
    _, verdict, per_partition = _adjudicate_across_partitions(_partitions(spec))
    assert set(per_partition) == set(spec)
    for name in spec:
        assert name in verdict, f"{name} is adjudicated but absent from the verdict text"


def test_unanimous_partitions_still_reach_the_preregistered_verdict():
    """The fix must not make a supported verdict unreachable, only unanimous."""
    adjudication, verdict, _ = _adjudicate_across_partitions(
        _partitions(
            {
                "all_same_allele": (0.5617, 0.5097),
                "human_hla_only": (0.5156, 0.5100),
                "models_ten_only": (0.5300, 0.4900),
            }
        )
    )
    assert adjudication == "HYPOTHESIS_1_SUPPORTED"
    assert "STRONGLY SUPPORTED" in verdict
    assert "every adjudicable partition" in verdict
    assert "3 of 3" in verdict


def test_unanimous_hypothesis_two_is_reachable_the_same_way():
    adjudication, verdict, _ = _adjudicate_across_partitions(
        _partitions({"all_same_allele": (0.42, 0.62), "human_hla_only": (0.47, 0.58)})
    )
    assert adjudication == "HYPOTHESIS_2_SUPPORTED"
    assert "2 of 2" in verdict


def test_unanimous_fallthrough_is_labelled_a_fallthrough_not_a_finding():
    adjudication, verdict, _ = _adjudicate_across_partitions(
        _partitions({"all_same_allele": (0.42, 0.49), "human_hla_only": (0.62, 0.62)})
    )
    assert adjudication == "NEITHER_HYPOTHESIS_MATCHED"
    assert "FALLTHROUGH, not a finding" in verdict


def test_a_non_adjudicable_partition_is_named_but_does_not_vote():
    """A NaN partition must not be counted as agreeing, or as dissenting."""
    spec = _partitions({"all_same_allele": (0.5617, 0.5097), "human_hla_only": (0.5156, 0.5100)})
    spec["models_ten_only"] = {
        "model_mh_concordance": float("nan"),
        "raw_mh_concordance": float("nan"),
    }
    adjudication, verdict, per_partition = _adjudicate_across_partitions(spec)
    assert adjudication == "HYPOTHESIS_1_SUPPORTED"
    assert per_partition["models_ten_only"]["adjudication"] == "NOT_ADJUDICABLE"
    assert "2 of 3" in verdict, "the un-votable partition must still be counted in the roster"
    assert "models_ten_only -> NOT_ADJUDICABLE" in verdict


def test_no_adjudicable_partition_reports_absence_of_measurement():
    nan = float("nan")
    adjudication, verdict, _ = _adjudicate_across_partitions(
        _partitions({"all_same_allele": (nan, nan), "human_hla_only": (nan, nan)})
    )
    assert adjudication == "NOT_ADJUDICABLE"
    assert "absence of measurement, not a result" in verdict


def test_report_names_the_disagreement_and_every_partition_verdict():
    """Section 3 must carry the disagreement, not only the run-level label."""
    roster = {
        "all_same_allele": {
            "adjudication": "HYPOTHESIS_1_SUPPORTED",
            "model_mh_concordance": 0.5616883116883117,
            "raw_mh_concordance": 0.5097402597402597,
        },
        "models_ten_only": {
            "adjudication": "NEITHER_HYPOTHESIS_MATCHED",
            "model_mh_concordance": 0.45528455284552843,
            "raw_mh_concordance": 0.4878048780487805,
        },
    }
    report = _report_for("PARTITIONS_DISAGREE", per_partition=roster)
    assert "Confound NOT resolved" in report
    assert "Confound resolution" not in report
    assert "PARTITIONS_DISAGREE" in report
    assert "Per-partition verdicts" in report
    for name in roster:
        assert f"`{name}`: {roster[name]['adjudication']}" in report


def test_report_never_hardcodes_a_verdict_name_the_run_did_not_produce():
    """The fallthrough branch used to assert NEITHER_HYPOTHESIS_MATCHED for any
    unrecognised label, which is a false statement about the run."""
    report = _report_for("PARTITIONS_DISAGREE")
    assert "NEITHER_HYPOTHESIS_MATCHED" not in report
    assert "NOT_ADJUDICABLE" not in _report_for("PARTITIONS_DISAGREE")
    assert "adjudicated NOT_ADJUDICABLE" in _report_for("NOT_ADJUDICABLE")
