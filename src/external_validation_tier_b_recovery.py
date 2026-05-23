"""
Tier B gold-standard recovery and negative discrimination (proteome face validity).

Ranks prefiltered proteome pools by each tool and reports recovery of the
15 canonical gold-standard epitopes vs binding-only baseline.

Usage:
    python -m src.external_validation_tier_b_recovery \\
        --run-dir results/external_tool_outputs/extval_YYYYMMDD_HHMM_gb_tierB \\
        --meta results/external_tier_b_peptide_meta.csv \\
        [--predig-raw RUN_DIR/raw/predig_path_output.csv] \\
        [--prime-raw RUN_DIR/raw/prime21_output.txt]
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import pandas as pd

from src.baseline_comparison import _rank_and_evaluate
from src.external_validation_fairness import parse_raw_predig, parse_raw_prime
from src.gold_standard import GOLD_STANDARD, GOLD_STANDARD_NEGATIVES
from src.naming import proteome_id_candidates

VIRUS_RANKED = {
    "EBV": "EBV_B95_8_panel8",
    "HPV16": "HPV16_18_panel8",
}


def _gs_by_virus() -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {"EBV": set(), "HPV16": set()}
    for g in GOLD_STANDARD:
        if g["virus"] == "EBV":
            out["EBV"].add(g["peptide"])
        else:
            out["HPV16"].add(g["peptide"])
    return out


def _negatives_by_virus() -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {"EBV": set(), "HPV16": set()}
    for g in GOLD_STANDARD_NEGATIVES[:25]:
        v = "HPV16" if g["virus"] in ("HPV", "HPV16") else "EBV"
        out[v].add(g["peptide"])
    return out


def load_ranked_features(results_dir: str, virus_key: str) -> pd.DataFrame:
    prefix = VIRUS_RANKED[virus_key]
    path = None
    for cand in proteome_id_candidates(prefix):
        p = os.path.join(results_dir, f"{cand}_ranked.csv")
        if os.path.isfile(p):
            path = p
            break
    if path is None:
        path = os.path.join(results_dir, f"{prefix}_ranked.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def merge_external_scores(
    meta: pd.DataFrame,
    predig_raw: Optional[str],
    prime_raw: Optional[str],
) -> pd.DataFrame:
    df = meta.copy()
    if predig_raw and os.path.isfile(predig_raw):
        max_d, mean_d = parse_raw_predig(predig_raw)
        df = df.merge(
            max_d.rename(columns={"predig_max_score": "predig_max_score"}),
            on="peptide",
            how="left",
        )
        df = df.merge(
            mean_d.rename(columns={"predig_mean_score": "predig_mean_score"}),
            on="peptide",
            how="left",
        )
    if prime_raw and os.path.isfile(prime_raw):
        max_d, mean_d = parse_raw_prime(prime_raw)
        if "prime_score" not in df.columns:
            df = df.merge(
                max_d.rename(columns={"prime_max_score": "prime_score"}),
                on="peptide",
                how="left",
            )
        df = df.merge(mean_d, on="peptide", how="left")
    return df


def attach_sestrav_scores(df: pd.DataFrame, results_dir: str) -> pd.DataFrame:
    rows = []
    for virus_key in df["virus"].unique():
        ranked = load_ranked_features(results_dir, virus_key)
        sub = df[df["virus"] == virus_key]
        m = sub.merge(
            ranked[["peptide", "immunogenicity_score", "binding_score"]],
            on="peptide",
            how="left",
        )
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def recovery_table(
    df: pd.DataFrame,
    gs_peptides: Set[str],
    methods: Dict[str, str],
    scope: str,
) -> List[dict]:
    rows = []
    pool = df[df["peptide"].isin(df["peptide"])].copy()
    for label, col in methods.items():
        if col not in pool.columns:
            continue
        valid = pool.dropna(subset=[col])
        if valid.empty:
            continue
        res = _rank_and_evaluate(valid, col, gs_peptides, label)
        if res:
            res["scope"] = scope
            res["score_column"] = col
            rows.append(res)
    return rows


def negative_discrimination(
    df: pd.DataFrame,
    neg_peptides: Set[str],
    tool_col: str,
    binding_col: str = "binding_score",
) -> int:
    sub = df[df["peptide"].isin(neg_peptides)].dropna(subset=[tool_col, binding_col])
    if sub.empty:
        return 0
    sub = sub.copy()
    sub["_rank_tool"] = sub[tool_col].rank(ascending=False)
    sub["_rank_bind"] = sub[binding_col].rank(ascending=False)
    return int((sub["_rank_tool"] > sub["_rank_bind"]).sum())


def run_recovery(
    run_dir: str,
    meta_path: str,
    results_dir: str,
    predig_raw: Optional[str],
    prime_raw: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(meta_path)
    meta = attach_sestrav_scores(meta, results_dir)
    meta = merge_external_scores(meta, predig_raw, prime_raw)

    methods = {
        "SESTRAV RF (immunogenicity_score)": "immunogenicity_score",
        "Binding-only (binding_score)": "binding_score",
        "PredIG-Path (max)": "predig_max_score",
        "PRIME 2.1 (max)": "prime_score",
    }

    gs_map = _gs_by_virus()
    neg_map = _negatives_by_virus()
    recovery_rows: List[dict] = []

    for virus_key, gs in gs_map.items():
        pool = meta[meta["virus"] == virus_key]
        recovery_rows.extend(
            recovery_table(pool, gs, methods, scope=f"tierB_{virus_key}")
        )

    recovery_rows.extend(
        recovery_table(meta, set().union(*gs_map.values()), methods, scope="tierB_combined")
    )
    recovery_df = pd.DataFrame(recovery_rows)

    neg_rows = []
    for virus_key, negs in neg_map.items():
        pool = meta[meta["virus"] == virus_key]
        present = negs & set(pool["peptide"])
        for tool_name, col in methods.items():
            if col not in pool.columns:
                continue
            pushed = negative_discrimination(pool, present, col)
            neg_rows.append(
                {
                    "virus": virus_key,
                    "tool": tool_name,
                    "negatives_in_pool": len(present),
                    "pushed_down_vs_binding": pushed,
                }
            )

    neg_df = pd.DataFrame(neg_rows)
    return recovery_df, neg_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier B gold-standard recovery analysis")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--meta", default="results/external_tier_b_peptide_meta.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--predig-raw", default=None)
    parser.add_argument("--prime-raw", default=None)
    args = parser.parse_args()

    predig_raw = args.predig_raw or os.path.join(args.run_dir, "raw", "predig_path_output.csv")
    prime_raw = args.prime_raw or os.path.join(args.run_dir, "raw", "prime21_output.txt")

    recovery_df, neg_df = run_recovery(
        args.run_dir,
        args.meta,
        args.results_dir,
        predig_raw,
        prime_raw,
    )

    processed = os.path.join(args.run_dir, "processed")
    os.makedirs(processed, exist_ok=True)

    rec_path = os.path.join(processed, "tier_b_gold_standard_recovery.csv")
    neg_path = os.path.join(processed, "tier_b_negative_discrimination.csv")
    recovery_df.to_csv(rec_path, index=False)
    neg_df.to_csv(neg_path, index=False)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_csv": rec_path,
        "negative_csv": neg_path,
        "recovery_rows": len(recovery_df),
    }
    with open(os.path.join(processed, "tier_b_recovery_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[tier-b-recovery] Wrote {rec_path} ({len(recovery_df)} rows)")
    print(f"[tier-b-recovery] Wrote {neg_path} ({len(neg_df)} rows)")


if __name__ == "__main__":
    main()
