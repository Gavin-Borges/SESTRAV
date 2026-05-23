"""
Build Tier A join tables and PredIG/PRIME sidecar inputs before external runs.

Reads:
  - immunogenicity_dataset.csv (peptide, label, virus)
  - models/rf_oof_predictions.csv (OOF RF scores from train_classifier)
  - models/peptide_binding_matrix.csv (bind_* presentation-style columns)
  - config.yaml (alleles for expansion + optional proteome mapping)

Writes under results/ (default):
  - external_validation_input.csv — one row per labeled peptide (left join):
        rf_oof_score (NaN for 16 gold-standard holdout rows), binding_max
        (NaN if peptide missing from binding matrix), tier_a_baseline_complete,
        is_gold_standard_holdout.
  - external_predig_peptide_allele_pairs.csv — peptide × allele for PredIG
  - external_prime_peptides.txt — one peptide per line
  - external_prime_alleles_compact.txt — comma-separated PRIME allele list
"""

from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd
import yaml

from src.iedb_data_loader import GOLD_STANDARD_EPITOPES

DEFAULT_VIRUS_TO_PROTEOME = {
    "EBV": "EBV_B95_8_panel8",
    "HPV16": "HPV16_18_panel8",
    "HPV": "HPV16_18_panel8",
    "HPV18": "HPV16_18_panel8",
}


def hla_to_prime_compact(hla: str) -> str:
    """HLA-A*02:01 -> A0201; HLA-B*08:01 -> B0801."""
    s = hla.replace("HLA-", "").strip()
    if "*" not in s:
        return s.replace(":", "")
    gene, allele = s.split("*", 1)
    allele = allele.replace(":", "")
    return f"{gene}{allele}"


def _binding_max_row(row: pd.Series, bind_cols: List[str]) -> float:
    return float(row[bind_cols].max())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare external validation input CSVs (PredIG/PRIME)"
    )
    parser.add_argument("--repo-root", default=".", help="SESTRAV-Dev root")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Output directory (default: config output_dir or results)",
    )
    args = parser.parse_args()
    root = os.path.abspath(args.repo_root)

    config_path = os.path.join(root, "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    alleles = cfg.get("alleles") or []
    if not alleles:
        raise RuntimeError("config.yaml must define 'alleles'")

    results_dir = args.results_dir or cfg.get("output_dir", "results")
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(root, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    data_path = os.path.join(root, "immunogenicity_dataset.csv")
    oof_path = os.path.join(root, "models", "rf_oof_predictions.csv")
    bind_path = os.path.join(root, "models", "peptide_binding_matrix.csv")

    for path in (data_path, oof_path, bind_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    ds = pd.read_csv(data_path)
    oof = pd.read_csv(oof_path)
    bind = pd.read_csv(bind_path)

    if "peptide" not in ds.columns or "label" not in ds.columns:
        raise ValueError("immunogenicity_dataset.csv needs peptide, label")
    if "score" not in oof.columns:
        raise ValueError("rf_oof_predictions.csv must contain 'score'")

    if "method" in oof.columns:
        oof_rf = oof[oof["method"] == "RandomForest"].copy()
    else:
        oof_rf = oof.copy()
    if oof_rf.empty:
        oof_rf = oof.copy()
    oof_rf = oof_rf.drop_duplicates(subset=["peptide"], keep="first")[
        ["peptide", "score"]
    ].rename(columns={"score": "rf_oof_score"})

    bind_cols = [c for c in bind.columns if c.startswith("bind_")]
    n_alleles = len(alleles)
    if len(bind_cols) != n_alleles:
        import warnings
        warnings.warn(
            f"Binding matrix has {len(bind_cols)} bind_* columns but "
            f"config.yaml defines {n_alleles} alleles; proceeding with "
            f"available columns",
            stacklevel=2,
        )
    if not bind_cols:
        raise ValueError("Binding matrix contains no bind_* columns")

    bind_sub = bind[["peptide"] + bind_cols].drop_duplicates(
        subset=["peptide"], keep="first"
    ).copy()
    bind_sub["binding_max"] = bind_sub.apply(
        lambda r: _binding_max_row(r, bind_cols), axis=1
    )

    merged = (
        ds.merge(oof_rf, on="peptide", how="left")
        .merge(bind_sub[["peptide", "binding_max"]], on="peptide", how="left")
    )

    def map_proteome(v: str) -> str:
        key = str(v).strip()
        if key not in DEFAULT_VIRUS_TO_PROTEOME:
            raise KeyError(
                f"Unknown virus '{key}' - add mapping in DEFAULT_VIRUS_TO_PROTEOME"
            )
        return DEFAULT_VIRUS_TO_PROTEOME[key]

    merged["proteome_id"] = merged["virus"].map(map_proteome)
    merged["is_gold_standard_holdout"] = merged["peptide"].isin(
        GOLD_STANDARD_EPITOPES
    )
    merged["tier_a_baseline_complete"] = merged["rf_oof_score"].notna() & merged[
        "binding_max"
    ].notna()

    out_main = os.path.join(results_dir, "external_validation_input.csv")
    out_cols = [
        "peptide",
        "label",
        "virus",
        "proteome_id",
        "is_gold_standard_holdout",
        "rf_oof_score",
        "binding_max",
        "tier_a_baseline_complete",
    ]
    merged.to_csv(out_main, index=False, columns=out_cols)

    preds = merged["peptide"].unique()
    rows = []
    for pep in preds:
        for hla in alleles:
            rows.append({"peptide": pep, "allele": hla})
    predig_pairs = pd.DataFrame(rows)
    predig_pairs.to_csv(
        os.path.join(results_dir, "external_predig_peptide_allele_pairs.csv"),
        index=False,
    )

    with open(
        os.path.join(results_dir, "external_prime_peptides.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(str(p) for p in preds) + "\n")

    prime_alleles = ",".join(hla_to_prime_compact(h) for h in alleles)
    with open(
        os.path.join(results_dir, "external_prime_alleles_compact.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(prime_alleles + "\n")

    n_oof = int(merged["rf_oof_score"].notna().sum())
    n_bind = int(merged["binding_max"].notna().sum())
    n_complete = int(merged["tier_a_baseline_complete"].sum())
    print(f"[external-prep] Wrote {out_main} ({len(merged)} rows)")
    print(
        f"[external-prep] Rows with rf_oof_score: {n_oof}, "
        f"with binding_max: {n_bind}, both: {n_complete}"
    )
    print(f"[external-prep] PredIG peptide-allele rows: {len(predig_pairs)}")
    print("[external-prep] PRIME peptide list + compact alleles under results/")


if __name__ == "__main__":
    main()
