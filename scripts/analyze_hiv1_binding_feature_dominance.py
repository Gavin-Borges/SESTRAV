"""
HIV-1 binding-feature dominance analysis for SESTRAV mode-31.

Investigates why the RF mode-31 model is anti-predictive for HIV-1 in LOO
evaluation (AUC-ROC 0.162).  Three questions:

  Q1. Do binding features dominate the global RF model?
  Q2. Are HIV-1 OOF scores inverted relative to immunogenicity labels?
  Q3. Are HIV-1 IEDB-confirmed negatives strong MHC binders?

Reads existing artifacts only - no retraining.

Inputs (relative to repo root):
  models/v5/feature_importances.csv
  models/v5/rf_oof_predictions_mode31.csv
  models/peptide_binding_matrix_v5.csv
  data/immunogenicity_dataset_v5.csv

Output:
  results/hiv1_binding_dominance_analysis.txt  (human-readable findings)
  results/hiv1_binding_dominance_analysis.csv  (per-virus summary table)
"""

from pathlib import Path

import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FEAT_IMP_PATH = REPO_ROOT / "models" / "v5" / "feature_importances.csv"
OOF_PATH = REPO_ROOT / "models" / "v5" / "rf_oof_predictions_mode31.csv"
BINDING_MATRIX_PATH = REPO_ROOT / "models" / "peptide_binding_matrix_v5.csv"
DATASET_PATH = REPO_ROOT / "data" / "immunogenicity_dataset_v5.csv"

BINDING_COLS = [
    "bind_A0101", "bind_A0201", "bind_A0301", "bind_A1101", "bind_A2402",
    "bind_B0702", "bind_B0801", "bind_B2705", "bind_B3501", "bind_B4402",
]

REAL_NEG_ORIGINS = {"tested_negative", "iedb_api"}

# Virus names as they appear in the LOO clean results
FOCUS_VIRUSES = ["HIV-1", "CMV"]


# ---------------------------------------------------------------------------
# Q1: Feature importance breakdown
# ---------------------------------------------------------------------------

def analyze_feature_importances(path: Path) -> dict:
    fi = pd.read_csv(path)
    binding_mask = fi["feature"].isin(BINDING_COLS)
    length_mask = fi["feature"] == "peptide_length"
    physico_mask = ~binding_mask & ~length_mask

    binding_total = fi.loc[binding_mask, "rf_importance"].sum()
    physico_total = fi.loc[physico_mask, "rf_importance"].sum()
    length_total = fi.loc[length_mask, "rf_importance"].sum()

    top_binding = fi[binding_mask].sort_values("rf_importance", ascending=False).head(5)
    top_physico = fi[physico_mask].sort_values("rf_importance", ascending=False).head(5)

    return {
        "binding_total": binding_total,
        "physico_total": physico_total,
        "length_total": length_total,
        "top_binding": top_binding,
        "top_physico": top_physico,
        "n_binding": int(binding_mask.sum()),
        "n_physico": int(physico_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Q2: Per-virus OOF score inversion
# ---------------------------------------------------------------------------

def analyze_oof_by_virus(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for virus, grp in oof.groupby("virus"):
        pos = grp[grp["label"] == 1]["score"]
        # restrict negatives to real IEDB assay-confirmed negatives
        neg_mask = (grp["label"] == 0) & (
            grp["negative_origin"].isin(REAL_NEG_ORIGINS)
            | grp["negative_origin"].isna()  # self-decoys have NaN origin
        )
        # for inversion analysis, only count assay-confirmed negatives
        real_neg_mask = (grp["label"] == 0) & grp["negative_origin"].isin(REAL_NEG_ORIGINS)
        neg_real = grp[real_neg_mask]["score"]

        if len(pos) < 5 or len(neg_real) < 5:
            continue

        # Mann-Whitney U: do real negatives have HIGHER scores than positives?
        # one-sided: alternative="greater" means neg > pos
        stat, pval = stats.mannwhitneyu(
            neg_real.values, pos.values, alternative="greater"
        )
        # effect size: rank-biserial correlation
        n1, n2 = len(neg_real), len(pos)
        rbc = 1 - (2 * stat) / (n1 * n2)  # +1 = neg always higher, -1 = pos always higher

        rows.append({
            "virus": virus,
            "n_pos": len(pos),
            "n_real_neg": len(neg_real),
            "mean_score_pos": pos.mean(),
            "mean_score_neg_real": neg_real.mean(),
            "score_inversion": neg_real.mean() > pos.mean(),
            "mw_pval_neg_gt_pos": pval,
            "rank_biserial_corr": rbc,
        })

    return pd.DataFrame(rows).sort_values("mean_score_neg_real", ascending=False)


# ---------------------------------------------------------------------------
# Q3: Binding scores for HIV-1 and CMV negatives vs positives
# ---------------------------------------------------------------------------

def analyze_binding_scores(dataset: pd.DataFrame, binding_matrix: pd.DataFrame) -> pd.DataFrame:
    merged = dataset.merge(binding_matrix, on="peptide", how="left")
    rows = []
    for virus in FOCUS_VIRUSES:
        grp = merged[merged["virus"] == virus].copy()
        grp = grp[~grp["is_quarantined"].fillna(False)]

        pos = grp[grp["label"] == 1]
        neg_real = grp[
            (grp["label"] == 0) & grp["negative_origin"].isin(REAL_NEG_ORIGINS)
        ]

        if len(pos) == 0 or len(neg_real) == 0:
            continue

        for allele_col in BINDING_COLS:
            if allele_col not in grp.columns:
                continue
            pos_scores = pos[allele_col].dropna()
            neg_scores = neg_real[allele_col].dropna()
            if len(pos_scores) < 3 or len(neg_scores) < 3:
                continue
            stat, pval = stats.mannwhitneyu(
                neg_scores.values, pos_scores.values, alternative="greater"
            )
            rows.append({
                "virus": virus,
                "allele": allele_col,
                "n_pos": len(pos_scores),
                "n_neg": len(neg_scores),
                "mean_binding_pos": pos_scores.mean(),
                "mean_binding_neg": neg_scores.mean(),
                "neg_higher_than_pos": neg_scores.mean() > pos_scores.mean(),
                "mw_pval": pval,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("SESTRAV: HIV-1 Binding Feature Dominance Analysis")
    log("Model: RF mode-31 (v5 dataset)  |  Script: analyze_hiv1_binding_feature_dominance.py")
    log("=" * 70)

    # --- Q1 ---
    log("\n--- Q1: Global RF Feature Importance Breakdown ---")
    fi_results = analyze_feature_importances(FEAT_IMP_PATH)
    b = fi_results["binding_total"]
    p = fi_results["physico_total"]
    l = fi_results["length_total"]
    total = b + p + l
    log(f"  Binding features ({fi_results['n_binding']} allele columns): {b:.4f}  ({b/total*100:.1f}%)")
    log(f"  Physicochemical features ({fi_results['n_physico']} columns):  {p:.4f}  ({p/total*100:.1f}%)")
    log(f"  Peptide length:                              {l:.4f}  ({l/total*100:.1f}%)")
    log()
    log("  Top 5 binding features by RF importance:")
    for _, row in fi_results["top_binding"].iterrows():
        log(f"    {row['feature']:20s}  {row['rf_importance']:.4f}")
    log("  Top 5 physicochemical features by RF importance:")
    for _, row in fi_results["top_physico"].iterrows():
        log(f"    {row['feature']:20s}  {row['rf_importance']:.4f}")

    # --- Q2 ---
    log("\n--- Q2: Per-Virus OOF Score Inversion (real IEDB negatives only) ---")
    oof = pd.read_csv(OOF_PATH)
    virus_summary = analyze_oof_by_virus(oof)
    log(f"  {'Virus':<25} {'n_pos':>6} {'n_neg':>6} {'score_pos':>10} {'score_neg':>10} {'inverted':>9} {'MW p-val':>10} {'RBC':>7}")
    log("  " + "-" * 88)
    for _, row in virus_summary.iterrows():
        log(
            f"  {row['virus']:<25} "
            f"{int(row['n_pos']):>6} "
            f"{int(row['n_real_neg']):>6} "
            f"{row['mean_score_pos']:>10.4f} "
            f"{row['mean_score_neg_real']:>10.4f} "
            f"{'YES' if row['score_inversion'] else 'no':>9} "
            f"{row['mw_pval_neg_gt_pos']:>10.4f} "
            f"{row['rank_biserial_corr']:>7.3f}"
        )

    hiv_row = virus_summary[virus_summary["virus"] == "HIV-1"]
    if not hiv_row.empty:
        r = hiv_row.iloc[0]
        log()
        log(f"  HIV-1 finding: score_neg ({r['mean_score_neg_real']:.4f}) "
            f"{'>' if r['score_inversion'] else '<'} score_pos ({r['mean_score_pos']:.4f}), "
            f"inversion = {r['score_inversion']}, MW p = {r['mw_pval_neg_gt_pos']:.4f}, "
            f"rank-biserial = {r['rank_biserial_corr']:.3f}")

    # --- Q3 ---
    log("\n--- Q3: MHCflurry Binding Scores - HIV-1 vs CMV (pos vs real neg) ---")
    try:
        dataset = pd.read_csv(DATASET_PATH)
        binding_matrix = pd.read_csv(BINDING_MATRIX_PATH)
        binding_results = analyze_binding_scores(dataset, binding_matrix)

        if binding_results.empty:
            log("  No binding data found for focus viruses in v5 dataset.")
        else:
            for virus in FOCUS_VIRUSES:
                v = binding_results[binding_results["virus"] == virus]
                if v.empty:
                    log(f"  {virus}: no data")
                    continue
                inverted_count = v["neg_higher_than_pos"].sum()
                sig_inverted = (v["neg_higher_than_pos"] & (v["mw_pval"] < 0.05)).sum()
                log(f"\n  {virus} ({v.iloc[0]['n_pos']} pos, {v.iloc[0]['n_neg']} real neg):")
                log(f"    Alleles where neg binding > pos binding: {inverted_count}/10")
                log(f"    Statistically significant inversions (p<0.05): {sig_inverted}/10")
                log(f"    {'Allele':<15} {'n_pos':>6} {'n_neg':>6} {'mean_pos':>10} {'mean_neg':>10} {'inv':>5} {'p-val':>8}")
                log("    " + "-" * 62)
                for _, row in v.iterrows():
                    log(
                        f"    {row['allele']:<15} "
                        f"{int(row['n_pos']):>6} "
                        f"{int(row['n_neg']):>6} "
                        f"{row['mean_binding_pos']:>10.4f} "
                        f"{row['mean_binding_neg']:>10.4f} "
                        f"{'Y' if row['neg_higher_than_pos'] else 'n':>5} "
                        f"{row['mw_pval']:>8.4f}"
                    )
    except FileNotFoundError as e:
        log(f"  WARNING: Could not load dataset/binding matrix: {e}")

    # --- Summary ---
    log("\n" + "=" * 70)
    log("SUMMARY FOR DISCUSSION SECTION")
    log("=" * 70)
    b_pct = b / total * 100
    log(f"  1. Binding features account for {b_pct:.1f}% of total RF mode-31 importance")
    log(f"     ({fi_results['n_binding']} binding columns vs {fi_results['n_physico']} physicochemical).")

    if not hiv_row.empty:
        r = hiv_row.iloc[0]
        if r["score_inversion"]:
            log(f"  2. HIV-1 OOF: negatives score HIGHER than positives "
                f"(neg={r['mean_score_neg_real']:.3f}, pos={r['mean_score_pos']:.3f}, "
                f"MW p={r['mw_pval_neg_gt_pos']:.4f}). Confirms anti-predictive mechanism.")
        else:
            log("  2. HIV-1 OOF: score inversion NOT confirmed in OOF data.")

    log()
    log("Output saved to:")
    log("  results/hiv1_binding_dominance_analysis.txt")
    log("  results/hiv1_binding_dominance_analysis.csv")

    # Save outputs
    txt_path = RESULTS_DIR / "hiv1_binding_dominance_analysis.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    if "virus_summary" in dir() and not virus_summary.empty:
        csv_path = RESULTS_DIR / "hiv1_binding_dominance_analysis.csv"
        virus_summary.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
