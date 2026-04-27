"""
Gold-standard sensitivity analysis for SESTRAV finalization.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

from src.gold_standard import GOLD_STANDARD, VIRUS_FILE_MAP

# Strain-sensitive epitopes called out in project documentation/comments.
STRAIN_SENSITIVE = {"FLRGRAYGI", "HPVGEADYFEY"}
# Additional conservative exclusions for the "uncertain-excluded" set.
AMBIGUOUS_EXTRA = {"YVLDHLIVV"}


def _sets() -> Dict[str, Set[str]]:
    all_peptides = {row["peptide"] for row in GOLD_STANDARD}
    high_conf = all_peptides - STRAIN_SENSITIVE
    uncertain_excluded = high_conf - AMBIGUOUS_EXTRA
    return {
        "current_standard": all_peptides,
        "high_confidence_only": high_conf,
        "uncertain_excluded": uncertain_excluded,
    }


def _evaluate_set_on_ranked(ranked_df: pd.DataFrame, peptides: Iterable[str], top_k: float) -> Dict[str, float]:
    ranked = ranked_df.copy()
    ranked["rank"] = ranked["immunogenicity_score"].rank(ascending=False, method="min")
    target = ranked[ranked["peptide"].isin(set(peptides))]
    n_target = len(target)
    if n_target == 0:
        return {
            "n_gs": 0,
            "recovery_topk": 0.0,
            "mean_rank_pct": float("nan"),
            "median_rank_pct": float("nan"),
        }
    cutoff = max(1, int(len(ranked) * top_k))
    in_top = int((target["rank"] <= cutoff).sum())
    return {
        "n_gs": int(n_target),
        "recovery_topk": float(in_top / n_target),
        "mean_rank_pct": float((target["rank"] / len(ranked) * 100).mean()),
        "median_rank_pct": float((target["rank"] / len(ranked) * 100).median()),
    }


def run_gold_standard_sensitivity(results_dir: str, output_csv: str, output_md: str) -> pd.DataFrame:
    """Run the 3-way sensitivity analysis and write CSV + markdown report."""
    rows: List[Dict] = []
    gs_sets = _sets()

    for virus, prefix in VIRUS_FILE_MAP.items():
        ranked_path = os.path.join(results_dir, f"{prefix}_ranked.csv")
        if not os.path.isfile(ranked_path):
            continue
        ranked = pd.read_csv(ranked_path)
        virus_gs = {row["peptide"] for row in GOLD_STANDARD if row["virus"] == virus}

        for set_name, peptide_set in gs_sets.items():
            subset = peptide_set & virus_gs
            m10 = _evaluate_set_on_ranked(ranked, subset, 0.10)
            m25 = _evaluate_set_on_ranked(ranked, subset, 0.25)
            rows.append(
                {
                    "virus": virus,
                    "set_name": set_name,
                    "n_gs": m10["n_gs"],
                    "recovery_top10": m10["recovery_topk"],
                    "recovery_top25": m25["recovery_topk"],
                    "mean_rank_pct": m25["mean_rank_pct"],
                    "median_rank_pct": m25["median_rank_pct"],
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        df.to_csv(output_csv, index=False)
        with open(output_md, "w", encoding="utf-8") as f:
            f.write("# Gold Standard Sensitivity\n\nNo ranked files found for sensitivity analysis.\n")
        return df

    base = df[df["set_name"] == "current_standard"].set_index("virus")
    comp_rows = []
    for _, row in df.iterrows():
        if row["set_name"] == "current_standard":
            continue
        if row["virus"] not in base.index:
            continue
        ref = base.loc[row["virus"]]
        comp_rows.append(
            {
                "virus": row["virus"],
                "set_name": row["set_name"],
                "delta_recovery_top10": float(row["recovery_top10"] - ref["recovery_top10"]),
                "delta_recovery_top25": float(row["recovery_top25"] - ref["recovery_top25"]),
                "delta_mean_rank_pct": float(row["mean_rank_pct"] - ref["mean_rank_pct"]),
            }
        )
    comp_df = pd.DataFrame(comp_rows)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    comp_csv = output_csv.replace(".csv", "_deltas.csv")
    comp_df.to_csv(comp_csv, index=False)

    md = f"""# Gold Standard Sensitivity Review

## Set definitions
- `current_standard`: all current gold-standard entries.
- `high_confidence_only`: excludes strain-sensitive epitopes (`FLRGRAYGI`, `HPVGEADYFEY`).
- `uncertain_excluded`: excludes strain-sensitive epitopes plus conservative ambiguity exclusion (`YVLDHLIVV`).

## Purpose
- Check whether conclusions are brittle to plausible revisions in benchmark composition.
- Report top-k recovery and rank shifts relative to current standard.

## Output files
- Metrics by set: `{output_csv}`
- Delta vs current set: `{comp_csv}`
"""
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SESTRAV gold-standard sensitivity analysis")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-csv", default="results/gold_standard_sensitivity.csv")
    parser.add_argument("--output-md", default="results/gold_standard_sensitivity.md")
    args = parser.parse_args()
    run_gold_standard_sensitivity(args.results_dir, args.output_csv, args.output_md)
