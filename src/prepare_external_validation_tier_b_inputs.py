"""
Prepare Tier B external validation inputs (proteome prefilter).

Selects top N peptides per virus by binding_max (max of bind_* columns),
union all 15 gold-standard epitopes, and writes PredIG/PRIME sidecars.

Usage:
    python -m src.prepare_external_validation_tier_b_inputs \\
        [--top-n 2000] [--results-dir results]
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Set

import pandas as pd
import yaml

from src.gold_standard import GOLD_STANDARD
from src.prepare_external_validation_inputs import (
    DEFAULT_VIRUS_TO_PROTEOME,
    hla_to_prime_compact,
)

# Tier A used 9 alleles (B4402 unsupported in PRIME)
TIER_B_ALLELES = [
    "HLA-A*01:01",
    "HLA-A*02:01",
    "HLA-A*03:01",
    "HLA-A*11:01",
    "HLA-A*24:02",
    "HLA-B*07:02",
    "HLA-B*08:01",
    "HLA-B*27:05",
    "HLA-B*35:01",
]

VIRUS_FEATURE_FILES = {
    "EBV": "EBV_B95_8_panel8_features.csv",
    "HPV16": "HPV16_18_panel8_features.csv",
}

VIRUS_RANKED_FILES = {
    "EBV": "EBV_B95_8_panel8_ranked.csv",
    "HPV16": "HPV16_18_panel8_ranked.csv",
}


def _bind_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("bind_")]


def binding_max_frame(df: pd.DataFrame) -> pd.DataFrame:
    bind_cols = _bind_cols(df)
    if not bind_cols:
        raise ValueError("No bind_* columns in features file")
    out = df.copy()
    out["binding_max"] = out[bind_cols].max(axis=1)
    return out


def gs_peptides_for_virus(virus_key: str) -> Set[str]:
    if virus_key == "HPV16":
        return {g["peptide"] for g in GOLD_STANDARD if g["virus"] in ("HPV", "HPV16")}
    return {g["peptide"] for g in GOLD_STANDARD if g["virus"] == "EBV"}


def select_tier_b_pool(
    features_path: str,
    virus_key: str,
    top_n: int,
) -> pd.DataFrame:
    df = binding_max_frame(pd.read_csv(features_path))
    df = df.sort_values("binding_max", ascending=False)
    df = df.drop_duplicates(subset=["peptide"], keep="first")

    top = df.head(top_n).copy()
    gs = gs_peptides_for_virus(virus_key)
    gs_rows = df[df["peptide"].isin(gs)].copy()
    gs_rows["forced_gold_standard"] = True
    top["forced_gold_standard"] = top["peptide"].isin(gs)

    combined = pd.concat([top, gs_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["peptide"], keep="first")
    combined["virus"] = virus_key
    combined["proteome_id"] = combined["virus"].map(DEFAULT_VIRUS_TO_PROTEOME)
    return combined


def build_predig_pairs(peptides: List[str], alleles: List[str]) -> pd.DataFrame:
    rows = []
    for pep in peptides:
        for allele in alleles:
            rows.append({"peptide": pep, "allele": allele})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Tier B external validation inputs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--top-n", type=int, default=2000)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    results_dir = args.results_dir
    config_path = args.config
    alleles = TIER_B_ALLELES
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if cfg.get("alleles"):
            alleles = [a for a in cfg["alleles"] if "B*44:02" not in a and "B4402" not in a]

    manifest: Dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "tierB",
        "top_n_per_virus": args.top_n,
        "alleles": alleles,
        "allele_count": len(alleles),
        "viruses": {},
        "gold_standard_audit": {},
    }

    all_peptides: List[str] = []
    all_pairs: List[pd.DataFrame] = []
    meta_rows: List[pd.DataFrame] = []

    for virus_key, feat_file in VIRUS_FEATURE_FILES.items():
        feat_path = os.path.join(results_dir, feat_file)
        if not os.path.isfile(feat_path):
            raise FileNotFoundError(feat_path)

        pool = select_tier_b_pool(feat_path, virus_key, args.top_n)
        gs = gs_peptides_for_virus(virus_key)
        in_pool = gs & set(pool["peptide"])
        cutoff = None
        if len(pool) >= args.top_n:
            ranked = pool.sort_values("binding_max", ascending=False)
            if len(ranked) >= args.top_n:
                cutoff = float(ranked.iloc[args.top_n - 1]["binding_max"])

        pep_path = os.path.join(results_dir, f"external_tier_b_{virus_key}_peptides.txt")
        pool["peptide"].to_csv(pep_path, index=False, header=False)

        manifest["viruses"][virus_key] = {
            "features_file": feat_file,
            "unique_peptides_in_proteome": int(
                pd.read_csv(feat_path)["peptide"].nunique()
            ),
            "tier_b_pool_size": len(pool),
            "top_n_requested": args.top_n,
            "binding_max_at_rank_n": cutoff,
            "gold_standard_total": len(gs),
            "gold_standard_in_pool": len(in_pool),
            "gold_standard_missing": sorted(gs - in_pool),
            "forced_gold_standard_count": int(pool["forced_gold_standard"].sum()),
            "peptide_list": pep_path,
        }
        manifest["gold_standard_audit"][virus_key] = {
            "present": sorted(in_pool),
            "missing": sorted(gs - in_pool),
        }

        meta_rows.append(
            pool[["peptide", "virus", "proteome_id", "binding_max", "forced_gold_standard"]]
        )
        all_peptides.extend(pool["peptide"].tolist())
        all_pairs.append(build_predig_pairs(pool["peptide"].tolist(), alleles))

    meta = pd.concat(meta_rows, ignore_index=True)
    meta_path = os.path.join(results_dir, "external_tier_b_peptide_meta.csv")
    meta.to_csv(meta_path, index=False)

    pairs = pd.concat(all_pairs, ignore_index=True)
    pairs_path = os.path.join(results_dir, "external_tier_b_predig_pairs.csv")
    pairs.to_csv(pairs_path, index=False)

    combined_path = os.path.join(results_dir, "external_tier_b_all_peptides.txt")
    pd.Series(sorted(set(all_peptides))).to_csv(combined_path, index=False, header=False)

    prime_alleles = ",".join(hla_to_prime_compact(a) for a in alleles)
    with open(os.path.join(results_dir, "external_tier_b_prime_alleles.txt"), "w") as f:
        f.write(prime_alleles)

    manifest_path = os.path.join(results_dir, "external_tier_b_prefilter_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    missing_any = any(
        manifest["gold_standard_audit"][v]["missing"] for v in manifest["gold_standard_audit"]
    )
    print(f"[tier-b-prep] Wrote {meta_path} ({len(meta)} peptides)")
    print(f"[tier-b-prep] PredIG pairs: {pairs_path} ({len(pairs)} rows)")
    print(f"[tier-b-prep] Manifest: {manifest_path}")
    if missing_any:
        print("[tier-b-prep] WARNING: some gold-standard epitopes missing from pool", file=__import__("sys").stderr)
    else:
        print("[tier-b-prep] Gold-standard audit: 15/15 epitopes in Tier B pools (per virus subset)")


if __name__ == "__main__":
    main()
