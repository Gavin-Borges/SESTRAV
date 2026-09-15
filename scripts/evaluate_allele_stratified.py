"""scripts/evaluate_allele_stratified.py
======================================
Performs allele-stratified evaluation on external validation cohorts (SCI-3 / T2-2).

Calculates:
  1. Mantel-Haenszel (pair-weighted) stratified concordance (AUC) over same-allele
     positive/negative pairs only:
       C_MH = sum_s(W_s * C_s) / sum_s(W_s)
     where W_s = n_pos,s * n_neg,s.
  2. Row-weighted within-allele concordance:
       C_row = sum_s(N_s * C_s) / sum_s(N_s)
     where N_s = n_pos,s + n_neg,s.
  3. Unstratified pooled AUC-ROC and AUC-PR for comparison.
  4. Both metrics evaluated on:
     - Immunogenicity Score (Stage 4 RF model)
     - Raw Presentation Score (Stage 2 best-of-10 MHCflurry)
  5. Stratified bootstrap confidence intervals (N=2,000) using an independent seed
     (default: 20260909) independent of `np.random.default_rng(42)`.
  6. Partitioned analysis across:
     - All same-allele pairs (including non-human MHC strata)
     - Human-HLA only (HLA-A*, HLA-B*, HLA-C*)
     - Model's-Ten Canonical Alleles (the 10 alleles represented by the features)

Evaluates pre-registered hypotheses from `inversion_localisation_synthesis_2026-09-07.md`:
  - Hypothesis 1 (Cohort-composition artifact): stratified model AUC >= 0.50 and
    stratified raw binding ~= 0.50.
  - Hypothesis 2 (Non-transferable learned mapping): stratified model AUC < 0.50 while
    stratified raw binding > 0.50.
  - Hypothesis 3 (Undiscovered corpus mechanism): tracks specific corpus subsets.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CANONICAL_TEN = frozenset(
    [
        "HLA-A*02:01",
        "HLA-A*01:01",
        "HLA-A*03:01",
        "HLA-A*11:01",
        "HLA-A*24:02",
        "HLA-B*07:02",
        "HLA-B*08:01",
        "HLA-B*27:05",
        "HLA-B*35:01",
        "HLA-B*44:02",
    ]
)


def compute_stratum_concordance(pos_scores: np.ndarray, neg_scores: np.ndarray) -> tuple[float, int]:
    """Compute Wilcoxon-Mann-Whitney concordance (AUC) and pair count for a single stratum."""
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    pairs = n_pos * n_neg
    if pairs == 0:
        return float("nan"), 0

    # Vectorized pair comparison: pos > neg + 0.5 * (pos == neg)
    diff = pos_scores[:, np.newaxis] - neg_scores[np.newaxis, :]
    concordant = np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
    auc = float(concordant / pairs)
    return auc, pairs


from dataclasses import asdict, dataclass


@dataclass
class StratumResult:
    allele: str
    n_pos: int
    n_neg: int
    n_total: int
    pairs: int
    concordance: float
    single_class: bool


def compute_stratified_metrics(
    df: pd.DataFrame, score_col: str, allele_col: str = "allele", label_col: str = "label"
) -> dict[str, Any]:
    """Calculate Mantel-Haenszel and row-weighted concordance across all strata with pairs > 0."""
    strata: list[StratumResult] = []
    total_concordant = 0.0
    total_pairs = 0
    total_samples = 0

    for allele, group in df.groupby(allele_col):
        pos = group[group[label_col] == 1][score_col].dropna().values
        neg = group[group[label_col] == 0][score_col].dropna().values
        n_pos = len(pos)
        n_neg = len(neg)
        pairs = n_pos * n_neg

        if pairs == 0:
            strata.append(
                StratumResult(
                    allele=str(allele),
                    n_pos=n_pos,
                    n_neg=n_neg,
                    n_total=len(group),
                    pairs=0,
                    concordance=float("nan"),
                    single_class=True,
                )
            )
            continue

        auc, p_count = compute_stratum_concordance(pos, neg)
        strata.append(
            StratumResult(
                allele=str(allele),
                n_pos=n_pos,
                n_neg=n_neg,
                n_total=len(group),
                pairs=pairs,
                concordance=auc,
                single_class=False,
            )
        )
        total_concordant += auc * pairs
        total_pairs += pairs
        total_samples += (n_pos + n_neg)

    if total_pairs == 0:
        return {
            "mh_concordance": float("nan"),
            "row_weighted_concordance": float("nan"),
            "total_pairs": 0,
            "two_class_strata_count": 0,
            "strata": [asdict(s) for s in strata],
        }

    mh_concordance = total_concordant / total_pairs
    two_class = [s for s in strata if not s.single_class]
    row_weighted = sum(s.concordance * s.n_total for s in two_class) / sum(s.n_total for s in two_class)

    return {
        "mh_concordance": float(mh_concordance),
        "row_weighted_concordance": float(row_weighted),
        "total_pairs": total_pairs,
        "two_class_strata_count": len(two_class),
        "total_strata_count": len(strata),
        "strata": [asdict(s) for s in strata],
    }


def _concordant_count(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Raw concordant-pair count for one stratum, ties counted as 0.5. Unnormalised."""
    diff = pos_scores[:, np.newaxis] - neg_scores[np.newaxis, :]
    return float(np.sum(diff > 0) + 0.5 * np.sum(diff == 0))


def _stratum_dominance(
    strata: list[dict[str, Any]], total_pairs: int, threshold: float = 0.50
) -> dict[str, Any]:
    """Share of same-allele pairs contributed by the single largest stratum.

    A concordance pooled over strata can be dominated by one allele, in which case
    "within allele" means "within THAT allele" and must not be read as a general
    within-allele result. Reuses the per-stratum pair counts compute_stratified_metrics
    already produced, so it costs no extra resampling.
    """
    two_class = [st for st in strata if st.get("pairs", 0) > 0]
    if not two_class or total_pairs <= 0:
        return {
            "top_allele": None,
            "top_allele_pairs": 0,
            "top_allele_pair_share": float("nan"),
            "n_strata_contributing": 0,
            "dominance_warning": False,
        }
    top = max(two_class, key=lambda st: st["pairs"])
    share = top["pairs"] / total_pairs
    return {
        "top_allele": top["allele"],
        "top_allele_pairs": int(top["pairs"]),
        "top_allele_pair_share": float(share),
        "n_strata_contributing": len(two_class),
        "dominance_warning": bool(share >= threshold),
    }


def stratified_bootstrap_ci(
    df: pd.DataFrame,
    score_col: str,
    allele_col: str = "allele",
    label_col: str = "label",
    n_resamples: int = 2000,
    seed: int = 20260909,
    alpha: float = 0.05,
    compare_col: str | None = None,
) -> tuple[float, float] | dict[str, Any]:
    """Compute 95% bootstrap confidence interval for Mantel-Haenszel concordance.

    Resamples independently within each stratum (preserving stratum sample sizes and
    eliminating cross-stratum variance leakage) using an independent RNG seed.

    When ``compare_col`` is given, BOTH score columns are evaluated on the SAME
    resampled row indices, so the per-resample difference
    ``C(score_col) - C(compare_col)`` is a genuine PAIRED contrast. Returns a dict
    carrying both marginal CIs, the paired delta point estimate, its CI, and whether
    that CI excludes zero. With ``compare_col=None`` the historical ``(low, high)``
    2-tuple is returned, numerically unchanged.

    Why the pairing has to be BUILT rather than recovered: the two marginal calls in
    run_stratified_evaluation use DIFFERENT seeds (``bootstrap_seed`` and
    ``bootstrap_seed + 1``) by design, so differencing their resamples after the fact
    is not a paired contrast. Two marginal CIs that overlap are not a test of their
    difference, and that difference is the comparison the within-allele question
    actually asks.
    """
    rng = np.random.default_rng(seed)
    cols = [score_col] if compare_col is None else [score_col, compare_col]

    # Pre-extract pos and neg arrays per stratum
    strata_data = []
    for _, group in df.groupby(allele_col):
        if compare_col is None:
            pos = group[group[label_col] == 1][score_col].dropna().values
            neg = group[group[label_col] == 0][score_col].dropna().values
            if len(pos) > 0 and len(neg) > 0:
                strata_data.append(({score_col: pos}, {score_col: neg}, len(pos) * len(neg)))
        else:
            # ONE shared row mask across both columns. A per-column dropna would give
            # the two arms different row counts, which desynchronises the bootstrap
            # stream from the first divergent stratum onward and destroys the pairing.
            usable = group[group[cols].notna().all(axis=1)]
            pos_g = usable[usable[label_col] == 1]
            neg_g = usable[usable[label_col] == 0]
            if len(pos_g) > 0 and len(neg_g) > 0:
                strata_data.append(
                    (
                        {c: pos_g[c].to_numpy(dtype=float) for c in cols},
                        {c: neg_g[c].to_numpy(dtype=float) for c in cols},
                        len(pos_g) * len(neg_g),
                    )
                )

    if not strata_data:
        if compare_col is None:
            return float("nan"), float("nan")
        nan = float("nan")
        return {
            "score_ci": (nan, nan),
            "compare_ci": (nan, nan),
            "delta_point": nan,
            "delta_ci": (nan, nan),
            "delta_excludes_zero": False,
            "paired_pairs": 0,
            "n_resamples": int(n_resamples),
        }

    total_pairs = sum(p for _, _, p in strata_data)
    low_idx = int(n_resamples * (alpha / 2))
    high_idx = int(n_resamples * (1.0 - alpha / 2))
    boot_estimates: list[float] = []
    compare_estimates: list[float] = []
    delta_estimates: list[float] = []

    for _ in range(n_resamples):
        concordant_sum = 0.0
        compare_sum = 0.0
        for pos, neg, _pairs in strata_data:
            n_p = len(pos[score_col])
            n_n = len(neg[score_col])
            # Draw INDICES, not values, so the second column can reuse the same rows.
            # rng.integers(0, n, n) consumes the identical RNG stream that
            # rng.choice(arr, size=n, replace=True) did, verified on numpy 2.4.6
            # across several seeds and uneven strata, so single-column results are
            # bit-identical to the pre-refactor implementation and no already
            # published confidence interval moves.
            idx_pos = rng.integers(0, n_p, n_p)
            idx_neg = rng.integers(0, n_n, n_n)
            concordant_sum += _concordant_count(pos[score_col][idx_pos], neg[score_col][idx_neg])
            if compare_col is not None:
                compare_sum += _concordant_count(
                    pos[compare_col][idx_pos], neg[compare_col][idx_neg]
                )

        boot_estimates.append(concordant_sum / total_pairs)
        if compare_col is not None:
            compare_estimates.append(compare_sum / total_pairs)
            delta_estimates.append((concordant_sum - compare_sum) / total_pairs)

    boot_estimates.sort()
    if compare_col is None:
        return float(boot_estimates[low_idx]), float(boot_estimates[high_idx])

    compare_estimates.sort()
    sorted_delta = sorted(delta_estimates)
    delta_lo = float(sorted_delta[low_idx])
    delta_hi = float(sorted_delta[high_idx])
    return {
        "score_ci": (float(boot_estimates[low_idx]), float(boot_estimates[high_idx])),
        "compare_ci": (float(compare_estimates[low_idx]), float(compare_estimates[high_idx])),
        "delta_point": float(np.mean(delta_estimates)),
        "delta_ci": (delta_lo, delta_hi),
        "delta_excludes_zero": bool(delta_lo > 0.0 or delta_hi < 0.0),
        "paired_pairs": int(total_pairs),
        "n_resamples": int(n_resamples),
    }


def _guard_output_path(path: str, force: bool, label: str) -> None:
    """Refuse to clobber an existing output, and create its parent directory.

    Two hazards this closes. The report default used to name a dated file under
    _local/notes/, which is gitignored and therefore has NO version history, so an
    unqualified re-run overwrote the only copy of a prior measurement in place and
    irrecoverably. And a bare relative filename made os.path.dirname return "",
    which raised from os.makedirs AFTER the full bootstrap had already run,
    discarding the result; os.path.abspath fixes that.
    """
    if os.path.exists(path) and not force:
        raise FileExistsError(
            f"{label} already exists and would be overwritten in place: {path}\n"
            "If it sits under _local/ it is gitignored, so there is no version "
            "history and no way back.\n"
            "Pass --force to overwrite, or give an explicit path."
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def run_stratified_evaluation(
    scored_csv_path: str,
    output_report_path: str | None = None,
    output_json_path: str | None = None,
    bootstrap_seed: int = 20260909,
    n_bootstrap: int = 2000,
    force: bool = False,
) -> dict[str, Any]:
    """Run full stratified evaluation and pre-registered hypothesis test."""
    # Checked up front: failing after several thousand resamples discards the run.
    if output_json_path:
        _guard_output_path(output_json_path, force, "JSON metrics path")
    if output_report_path:
        _guard_output_path(output_report_path, force, "Markdown report path")

    print("=" * 70)
    print("SESTRAV Allele-Stratified Cohort Concordance Evaluation (SCI-3 / T2-2)")
    print("=" * 70)
    print(f"Loading scored cohort: {scored_csv_path}")

    if not os.path.exists(scored_csv_path):
        raise FileNotFoundError(f"Scored cohort not found at: {scored_csv_path}")

    df = pd.read_csv(scored_csv_path)
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

    if "allele" not in df.columns:
        raise ValueError(f"Cohort at {scored_csv_path} lacks required 'allele' column")

    score_col = "immunogenicity_score" if "immunogenicity_score" in df.columns else "calibrated_score"
    raw_col = "presentation_score"

    # Define partition subsets
    partitions = {
        "all_same_allele": df,
        "human_hla_only": df[df["allele"].astype(str).str.startswith(("HLA-A", "HLA-B", "HLA-C"))],
        "models_ten_only": df[df["allele"].isin(CANONICAL_TEN)],
    }

    partition_results = {}

    for part_name, part_df in partitions.items():
        print(f"\n--- Evaluating Partition: {part_name} ({len(part_df)} rows) ---")
        n_pos = int((part_df["label"] == 1).sum())
        n_neg = int((part_df["label"] == 0).sum())
        print(f"Class distribution: {n_pos} positive, {n_neg} negative")

        # Unstratified pooled metrics
        if n_pos > 0 and n_neg > 0:
            unstrat_model_auc = float(roc_auc_score(part_df["label"], part_df[score_col]))
            unstrat_model_pr = float(average_precision_score(part_df["label"], part_df[score_col]))
            unstrat_raw_auc = float(roc_auc_score(part_df["label"], part_df[raw_col]))
        else:
            unstrat_model_auc = float("nan")
            unstrat_model_pr = float("nan")
            unstrat_raw_auc = float("nan")

        # Stratified Model Metrics
        model_res = compute_stratified_metrics(part_df, score_col=score_col)
        model_ci = stratified_bootstrap_ci(
            part_df, score_col=score_col, n_resamples=n_bootstrap, seed=bootstrap_seed
        )

        # Stratified Raw Presentation Metrics
        raw_res = compute_stratified_metrics(part_df, score_col=raw_col)
        raw_ci = stratified_bootstrap_ci(
            part_df, score_col=raw_col, n_resamples=n_bootstrap, seed=bootstrap_seed + 1
        )

        # PAIRED contrast. A third, independent pass: it must not reuse either
        # marginal stream, and inside it both arms share one index draw per stratum.
        # Reusing bootstrap_seed here makes this run's score_ci bit-identical to
        # model_ci above, which is a free internal consistency check.
        paired = stratified_bootstrap_ci(
            part_df,
            score_col=score_col,
            compare_col=raw_col,
            n_resamples=n_bootstrap,
            seed=bootstrap_seed,
        )
        dominance = _stratum_dominance(model_res["strata"], model_res["total_pairs"])

        print(f"Total Same-Allele Pairs: {model_res['total_pairs']:,}")
        print(
            f"Model Concordance (MH): {model_res['mh_concordance']:.4f} [95% CI: {model_ci[0]:.4f}, {model_ci[1]:.4f}]"
        )
        print(f"Model Concordance (Row-wt): {model_res['row_weighted_concordance']:.4f}")
        print(f"Unstratified Model AUC: {unstrat_model_auc:.4f}")
        print(
            f"Raw Binding Concordance (MH): {raw_res['mh_concordance']:.4f} [95% CI: {raw_ci[0]:.4f}, {raw_ci[1]:.4f}]"
        )
        print(f"Unstratified Raw Binding AUC: {unstrat_raw_auc:.4f}")
        print(
            f"Paired Delta (Model - Raw): {paired['delta_point']:+.4f} "
            f"[95% CI: {paired['delta_ci'][0]:+.4f}, {paired['delta_ci'][1]:+.4f}] "
            f"{'EXCLUDES 0' if paired['delta_excludes_zero'] else '(includes 0)'}"
        )
        if dominance["dominance_warning"]:
            print(
                f"WARNING: {dominance['top_allele']} supplies "
                f"{dominance['top_allele_pairs']:,} of {model_res['total_pairs']:,} pairs "
                f"({dominance['top_allele_pair_share']:.1%}); this is not a general "
                "within-allele result."
            )

        partition_results[part_name] = {
            "n_samples": len(part_df),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "total_same_allele_pairs": model_res["total_pairs"],
            "two_class_strata": model_res["two_class_strata_count"],
            "unstratified_model_auc": unstrat_model_auc,
            "unstratified_model_pr": unstrat_model_pr,
            "unstratified_raw_auc": unstrat_raw_auc,
            "model_mh_concordance": model_res["mh_concordance"],
            "model_mh_ci": model_ci,
            "model_row_weighted": model_res["row_weighted_concordance"],
            "raw_mh_concordance": raw_res["mh_concordance"],
            "raw_mh_ci": raw_ci,
            "paired_delta_model_minus_raw": paired["delta_point"],
            "paired_delta_ci": paired["delta_ci"],
            "paired_delta_excludes_zero": paired["delta_excludes_zero"],
            # If this differs from total_same_allele_pairs, the shared NaN mask
            # dropped rows and the delta is measured on a NARROWER population than
            # the two marginals above it. A reader has to be able to see that.
            "paired_delta_pairs": paired["paired_pairs"],
            "paired_delta_marginal_check": {
                "model_ci_paired_run": paired["score_ci"],
                "raw_ci_paired_run": paired["compare_ci"],
            },
            "stratum_dominance": dominance,
            "strata_detail": model_res["strata"],
        }

    # Adjudicate Pre-registered Hypotheses
    # Primary test is on human_hla_only and all_same_allele
    human_eval = partition_results["human_hla_only"]
    model_c = human_eval["model_mh_concordance"]
    raw_c = human_eval["raw_mh_concordance"]

    if model_c >= 0.50 and (0.45 <= raw_c <= 0.55):
        adjudication = "HYPOTHESIS_1_SUPPORTED"
        verdict_text = (
            "Hypothesis 1 (Cohort-Composition Confound) is STRONGLY SUPPORTED. "
            "Within same-allele pairs, model concordance is >= 0.50 and raw presentation concordance is ~0.50. "
            "The below-chance pooled AUC was a between-allele Simpson composition artifact."
        )
    elif model_c < 0.50 and raw_c > 0.50:
        adjudication = "HYPOTHESIS_2_SUPPORTED"
        verdict_text = (
            "Hypothesis 2 (Non-Transferable Learned Mapping) is SUPPORTED. "
            "Stratified model concordance remains below chance (<0.50) while raw presentation score remains above chance."
        )
    else:
        # Neither pre-registered branch matched. This is a FALLTHROUGH, not a
        # finding, and the label must not read as one: the old value,
        # EMPIRICAL_DISCRIMINATION, was rendered under a "Pre-Registered
        # Verdict" heading and read as "the model empirically discriminates",
        # the opposite of what reaching this branch means.
        adjudication = "NEITHER_HYPOTHESIS_MATCHED"
        verdict_text = (
            f"Within-allele model concordance is {model_c:.4f} [95% CI {human_eval['model_mh_ci'][0]:.4f}, {human_eval['model_mh_ci'][1]:.4f}] "
            f"vs raw presentation concordance {raw_c:.4f} [95% CI {human_eval['raw_mh_ci'][0]:.4f}, {human_eval['raw_mh_ci'][1]:.4f}]."
        )

    _d = human_eval
    _sign = "excludes" if _d["paired_delta_excludes_zero"] else "includes"
    verdict_text += (
        f" Paired within-allele delta (model - raw) is "
        f"{_d['paired_delta_model_minus_raw']:+.4f} "
        f"[95% CI {_d['paired_delta_ci'][0]:+.4f}, {_d['paired_delta_ci'][1]:+.4f}], "
        f"which {_sign} zero. The two marginal intervals above are NOT a test of "
        "their difference; this delta is."
    )

    print("\n" + "=" * 70)
    print(f"ADJUDICATION: {adjudication}")
    print(verdict_text)
    print("=" * 70)

    final_results = {
        "dataset": os.path.basename(scored_csv_path),
        "bootstrap_resamples": n_bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "adjudication": adjudication,
        "verdict_summary": verdict_text,
        "partitions": partition_results,
    }

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2)
        print(f"Saved JSON metrics to: {output_json_path}")

    if output_report_path:
        report_md = _generate_markdown_report(final_results)
        with open(output_report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Saved Markdown report to: {output_report_path}")

    return final_results


def _generate_markdown_report(res: dict[str, Any]) -> str:
    """Format evaluation results into GitHub Flavored Markdown."""
    lines = [
        "# Allele-Stratified Cohort Concordance Evaluation (SCI-3 / T2-2)",
        "",
        f"**Source Dataset:** `{res['dataset']}`  ",
        f"**Bootstrap Resamples:** {res['bootstrap_resamples']:,} (Independent RNG Seed `{res['bootstrap_seed']}`)  ",
        f"**Pre-Registered Verdict:** **{res['adjudication']}**  ",
        "",
        "> [!NOTE]",
        f"> {res['verdict_summary']}",
        "",
        "## 1. Concordance Across Cohort Partitions",
        "",
        "| Partition | Samples (Pos/Neg) | Same-Allele Pairs | Unstratified Model AUC | Stratified Model Concordance (95% CI) | Stratified Raw Binding Concordance (95% CI) | Paired Delta (Model - Raw), 95% CI |",
        "|---|---|---|---|---|---|---|",
    ]

    for name, p in res["partitions"].items():
        lines.append(
            f"| **{name}** | {p['n_samples']} ({p['n_pos']}/{p['n_neg']}) | **{p['total_same_allele_pairs']:,}** | "
            f"{p['unstratified_model_auc']:.4f} | **{p['model_mh_concordance']:.4f}** `[{p['model_mh_ci'][0]:.4f}, {p['model_mh_ci'][1]:.4f}]` | "
            f"{p['raw_mh_concordance']:.4f} `[{p['raw_mh_ci'][0]:.4f}, {p['raw_mh_ci'][1]:.4f}]` | "
            f"{p['paired_delta_model_minus_raw']:+.4f} "
            f"`[{p['paired_delta_ci'][0]:+.4f}, {p['paired_delta_ci'][1]:+.4f}]` "
            f"{'**excludes 0**' if p['paired_delta_excludes_zero'] else '(includes 0)'} |"
        )

    dominant = [
        (name, p["stratum_dominance"])
        for name, p in res["partitions"].items()
        if p.get("stratum_dominance", {}).get("dominance_warning")
    ]
    if dominant:
        lines.extend(["", "> [!WARNING]", "> **Single-stratum dominance.**"])
        for name, d in dominant:
            lines.append(
                f"> In `{name}`, `{d['top_allele']}` supplies {d['top_allele_pairs']:,} "
                f"of the same-allele pairs ({d['top_allele_pair_share']:.1%}) across "
                f"{d['n_strata_contributing']} contributing strata."
            )
        lines.append(
            "> \"Within-allele\" here is in practice \"within that allele\". The paired "
            "delta and its interval are dominated by one stratum and must not be read "
            "as a general within-allele result. Note a TIGHTER interval makes that "
            "misreading easier, not harder."
        )

    lines.extend(
        [
            "",
            "## 2. Per-Stratum Breakdown (Human HLA Partition)",
            "",
            "| Allele | Positives | Negatives | Pairs | Concordance (Model) |",
            "|---|---|---|---|---|",
        ]
    )

    human_strata = res["partitions"]["human_hla_only"]["strata_detail"]
    for s in sorted(human_strata, key=lambda x: x["pairs"], reverse=True):
        c_str = f"{s['concordance']:.4f}" if not np.isnan(s["concordance"]) else "N/A (single-class)"
        lines.append(f"| `{s['allele']}` | {s['n_pos']} | {s['n_neg']} | {s['pairs']:,} | {c_str} |")

    # Section 3 is DERIVED from the run, never asserted. Every line below was once a
    # hardcoded string emitted regardless of the numbers, so the report claimed the
    # confound was resolved even on runs whose own adjudication fell through to the
    # neither-hypothesis branch - which is what the shipped cohort actually does.
    adjudication = res.get("adjudication", "")
    same_allele_pairs = res["partitions"]["all_same_allele"]["total_same_allele_pairs"]

    if adjudication == "HYPOTHESIS_1_SUPPORTED":
        resolution = (
            "- **Confound resolution**: this run adjudicated HYPOTHESIS_1_SUPPORTED, so the "
            "below-chance pooled AUC is attributed to between-allele composition."
        )
    elif adjudication == "HYPOTHESIS_2_SUPPORTED":
        resolution = (
            "- **Confound resolution**: this run adjudicated HYPOTHESIS_2_SUPPORTED, so the "
            "below-chance pooled AUC is NOT explained by composition alone."
        )
    else:
        resolution = (
            "- **Confound NOT resolved**: neither pre-registered hypothesis matched, so this "
            "run adjudicated NEITHER_HYPOTHESIS_MATCHED. That is a fallthrough, not a "
            "finding. Read the concordances and their intervals in section 2 directly."
        )

    lines.extend(
        [
            "",
            "## 3. Scientific Adjudication",
            "",
            f"- **Same-allele pair count**: {same_allele_pairs:,}. Pairs are a PRODUCT "
            "(n_pos * n_neg within each stratum), not independent observations, so this "
            "count is not a sample size and must not be read as statistical power.",
            resolution,
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run allele-stratified concordance evaluation.")
    parser.add_argument("--input", required=True, help="Path to scored cohort CSV.")
    parser.add_argument(
        "--report",
        default=os.path.join(
            PROJECT_ROOT,
            "_local",
            "notes",
            f"allele_stratified_evaluation_{datetime.date.today():%Y-%m-%d}.md",
        ),
        help="Path to output Markdown report. Defaults to a file stamped with TODAY.",
    )
    parser.add_argument(
        "--json-out",
        default=os.path.join(PROJECT_ROOT, "results", "allele_stratified_metrics.json"),
        help="Path to output JSON metrics.",
    )
    parser.add_argument("--seed", type=int, default=20260909, help="Independent bootstrap RNG seed.")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Number of bootstrap resamples.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Permit overwriting an existing --report or --json-out path.",
    )
    args = parser.parse_args()

    try:
        run_stratified_evaluation(
            scored_csv_path=args.input,
            output_report_path=args.report,
            output_json_path=args.json_out,
            bootstrap_seed=args.seed,
            n_bootstrap=args.bootstrap,
            force=args.force,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
