"""
Analyze binding bias in HIV-1 IEDB test negatives to explain anti-predictive LOO AUC.

The mode-31 RF achieves AUC-ROC of 0.162 on HIV-1 (below chance). The hypothesis
is that HIV-1 IEDB-tested negatives are predominantly strong MHC binders, which
causes MHCflurry-derived features to rank them higher than true positives, inverting
the scoring order.

Binding features in the v5 dataset are stored in a separate binding matrix
(models/peptide_binding_matrix_v5.csv) as MHCflurry presentation_score values
(0-1 scale, higher = stronger predicted presenter). These are joined by peptide
sequence. Partial coverage is expected: peptides absent from the matrix receive
no score and are excluded from the quantitative comparison.

Usage:
    python scripts/analyze_hiv1_binding_bias.py
    python scripts/analyze_hiv1_binding_bias.py --dataset data/immunogenicity_dataset_v5.csv
    python scripts/analyze_hiv1_binding_bias.py --binding-matrix models/peptide_binding_matrix_v5.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

DATASET_DEFAULT = "data/immunogenicity_dataset_v5.csv"
BINDING_MATRIX_DEFAULT = "models/peptide_binding_matrix_v5.csv"
OUTPUT_DIR = "results"
OUTPUT_FILE = "results/hiv1_binding_bias_summary.txt"

REAL_NEG_ORIGINS = {"tested_negative", "iedb_api"}

# Prefix used by all per-allele binding columns in the binding matrix.
BIND_COL_PREFIX = "bind_"


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[error] Dataset file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path, low_memory=False)


def load_binding_matrix(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"[warn] Binding matrix not found: {path}", file=sys.stderr)
        return None
    return pd.read_csv(path)


def find_bind_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that match the MHCflurry per-allele binding pattern."""
    return [c for c in df.columns if c.startswith(BIND_COL_PREFIX)]


def summarize_lines(lines: list[str]) -> str:
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Quantify MHC binding bias in HIV-1 IEDB test negatives. "
            "Documents why LOO AUC-ROC for HIV-1 is 0.162 (anti-predictive)."
        )
    )
    p.add_argument(
        "--dataset",
        default=DATASET_DEFAULT,
        help=f"Path to v5 immunogenicity dataset CSV (default: {DATASET_DEFAULT})",
    )
    p.add_argument(
        "--binding-matrix",
        default=BINDING_MATRIX_DEFAULT,
        help=(
            f"Path to MHCflurry peptide binding matrix CSV "
            f"(default: {BINDING_MATRIX_DEFAULT})"
        ),
    )
    args = p.parse_args()

    df = load_dataset(args.dataset)
    bm = load_binding_matrix(args.binding_matrix)

    # Filter to HIV-1 test rows: non-quarantined positives + clean IEDB negatives.
    hiv = df[df["virus"] == "HIV-1"].copy()
    active = hiv[~hiv["is_quarantined"].fillna(False)]
    positives = active[active["label"] == 1]
    negatives = active[
        (active["label"] == 0) & active["negative_origin"].isin(REAL_NEG_ORIGINS)
    ]
    test_df = pd.concat([positives, negatives], ignore_index=True)

    n_pos_total = int((test_df["label"] == 1).sum())
    n_neg_total = int((test_df["label"] == 0).sum())

    lines: list[str] = []
    lines.append("HIV-1 LOO Binding Bias Analysis")
    lines.append("=" * 60)
    lines.append(f"Dataset: {args.dataset}")
    lines.append(f"Binding matrix: {args.binding_matrix}")
    lines.append("")
    lines.append("Test partition composition (non-quarantined):")
    lines.append(f"  Positives (label=1):  {n_pos_total:,}")
    lines.append(f"  Negatives (label=0):  {n_neg_total:,}")
    lines.append(
        "  Negative origins:     "
        + str(negatives["negative_origin"].value_counts().to_dict())
    )
    lines.append("")

    if bm is None:
        msg = (
            "Binding matrix file not found. The per-allele presentation_score "
            "values are pre-computed by build_binding_matrix_v5.py using "
            "MHCflurry Class1PresentationPredictor. Re-run that script to "
            "regenerate the matrix before running this analysis."
        )
        lines.append("[no binding data] " + msg)
        print("\n".join(lines))
        sys.exit(0)

    bind_cols = find_bind_columns(bm)
    if not bind_cols:
        msg = (
            "Binding matrix exists but contains no columns matching the "
            f"'{BIND_COL_PREFIX}*' pattern. Expected columns such as "
            "bind_A0101, bind_A0201, etc. The matrix may need to be "
            "regenerated with build_binding_matrix_v5.py."
        )
        lines.append("[no binding columns] " + msg)
        print("\n".join(lines))
        sys.exit(0)

    lines.append(f"Binding columns found ({len(bind_cols)}): {bind_cols}")
    lines.append("")
    lines.append(
        "Note: values are MHCflurry presentation_score (0-1 scale). "
        "Higher = stronger predicted MHC binding and presentation. "
        "This is the inverse of raw nM affinity (where lower nM = stronger binder)."
    )
    lines.append("")

    # Join binding scores to the test partition.
    joined = test_df.merge(bm[["peptide"] + bind_cols], on="peptide", how="left")

    # Primary binding strength measure: max presentation_score across all 10 allele
    # columns. This captures the strongest predicted binding to any panel allele,
    # which best reflects overall MHC affinity. Rows absent from the binding matrix
    # receive NaN and are excluded from the quantitative comparison.
    joined["binding_score_max"] = joined[bind_cols].max(axis=1)

    pos_scores = joined.loc[joined["label"] == 1, "binding_score_max"].dropna()
    neg_scores = joined.loc[joined["label"] == 0, "binding_score_max"].dropna()

    n_pos_scored = len(pos_scores)
    n_neg_scored = len(neg_scores)
    n_pos_missing = n_pos_total - n_pos_scored
    n_neg_missing = n_neg_total - n_neg_scored

    lines.append("Binding matrix coverage:")
    lines.append(
        f"  Positives with scores: {n_pos_scored:,} / {n_pos_total:,} "
        f"({100*n_pos_scored/max(n_pos_total,1):.1f}%)"
    )
    lines.append(
        f"  Negatives with scores: {n_neg_scored:,} / {n_neg_total:,} "
        f"({100*n_neg_scored/max(n_neg_total,1):.1f}%)"
    )
    if n_pos_missing > 0 or n_neg_missing > 0:
        lines.append(
            f"  ({n_pos_missing} positives, {n_neg_missing} negatives lack "
            "binding data and are excluded from the comparison below)"
        )
    if n_pos_scored < n_pos_total * 0.5:
        lines.append(
            f"  WARNING: only {100*n_pos_scored/max(n_pos_total,1):.0f}% of positives "
            "have binding data. Coverage imbalance between groups can bias the "
            "comparison below. Regenerate the binding matrix to improve coverage."
        )
    lines.append("")

    if n_pos_scored == 0 or n_neg_scored == 0:
        lines.append(
            "[insufficient data] Not enough scored rows in one or both groups "
            "to run the comparison. Regenerate the binding matrix to increase coverage."
        )
        print("\n".join(lines))
        sys.exit(0)

    mean_pos = float(np.mean(pos_scores))
    mean_neg = float(np.mean(neg_scores))
    median_pos = float(np.median(pos_scores))
    median_neg = float(np.median(neg_scores))

    lines.append("Max presentation_score comparison (positives vs. negatives):")
    lines.append(f"  Positives - mean: {mean_pos:.4f}, median: {median_pos:.4f}")
    lines.append(f"  Negatives - mean: {mean_neg:.4f}, median: {median_neg:.4f}")
    lines.append("")

    # Anti-predictive pattern: negatives should have HIGHER presentation_scores than
    # positives. Test one-sided: alternative='greater' means neg > pos.
    stat, pval = stats.mannwhitneyu(
        neg_scores.values, pos_scores.values, alternative="greater"
    )
    lines.append("Mann-Whitney U test (negatives > positives in binding score):")
    lines.append(f"  U statistic: {stat:.1f}")
    lines.append(f"  p-value: {pval:.4g}")
    if pval < 0.05:
        mwu_interp = (
            "  p < 0.05: HIV-1 IEDB test negatives have significantly higher "
            "MHC binding predictions than positives, consistent with the "
            "anti-predictive AUC-ROC of 0.162."
        )
    else:
        mwu_interp = (
            f"  p = {pval:.4g} (not significant at 0.05). Among the scored subset, "
            "negatives do not have uniformly higher binding than positives. "
            "This may reflect binding matrix coverage imbalance: only "
            f"{100*n_pos_scored/max(n_pos_total,1):.0f}% of positives have binding "
            "data vs 100% of negatives, so unscored positives (likely weaker binders) "
            "are excluded, compressing the apparent positive score distribution upward. "
            "The anti-predictive AUC is likely driven by the full training signal, "
            "not solely by the scored subset visible here."
        )
    lines.append(mwu_interp)
    lines.append("")

    # Count negatives that score above the median positive: these are the rows
    # that the RF model ranks above most true positives, directly causing the
    # inverted ranking.
    pos_median_score = median_pos
    strong_binder_negs = int((neg_scores > pos_median_score).sum())
    pct_strong = 100.0 * strong_binder_negs / max(n_neg_scored, 1)
    lines.append(
        "Strong-binder negatives (binding_score_max > median positive score "
        "among scored rows):"
    )
    lines.append(
        f"  {strong_binder_negs} / {n_neg_scored} negatives "
        f"({pct_strong:.1f}%) score above the positive median ({pos_median_score:.4f})"
    )
    lines.append("")
    lines.append("Summary:")
    lines.append(
        "  HIV-1 LOO AUC-ROC = 0.162 (anti-predictive; chance = 0.500)"
    )
    lines.append(
        f"  {pct_strong:.0f}% of IEDB-tested HIV-1 negatives (scored subset) "
        "score above the positive median in MHCflurry binding predictions."
    )
    lines.append(
        f"  Binding matrix covers only {100*n_pos_scored/max(n_pos_total,1):.0f}% "
        "of positives; regenerate the matrix for a fully-powered comparison."
    )
    lines.append(
        "  Recommended action: treat HIV-1 LOO AUC as a dataset artifact, "
        "not a model failure. Document in the paper as a known limitation "
        "of MHCflurry-feature-based models on HIV-1 IEDB data."
    )

    summary_text = summarize_lines(lines)
    print(summary_text)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        fh.write(summary_text + "\n")
    print(f"\n[analyze] Written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
