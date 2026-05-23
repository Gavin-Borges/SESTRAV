"""
SESTRAV External Validation Fairness Supplement (plan §4h).

Computes analyses not covered by external_benchmark_comparison.py:
  - Prevalence / random baselines
  - Mean vs max allele collapse (PredIG/PRIME)
  - Virus-weighted metric averages
  - Length-stratified AUC-PR
  - 16 gold-standard holdout spotlight
  - Training-set overlap quantification (when lists provided)

Usage:
    python -m src.external_validation_fairness \\
        --merged results/external_validation_merged_scores.csv \\
        --run-dir results/external_tool_outputs/extval_YYYYMMDD_HHMM_gb_tierA \\
        [--predig-raw results/external_tool_outputs/predig_path_output.csv] \\
        [--predig-train path/to/predig_training_peptides.csv] \\
        [--prime-train path/to/prime_training_peptides.csv]
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluate_metrics import evaluate
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES

PREVALENCE_POS_RATE = 506 / 720  # frozen Tier A class balance


def _tool_columns(merged: pd.DataFrame) -> Dict[str, str]:
    cols: Dict[str, str] = {}
    if "rf_oof_score" in merged.columns:
        cols["SESTRAV RF (30-feat)"] = "rf_oof_score"
    if "binding_max" in merged.columns:
        cols["Binding-only (max)"] = "binding_max"
    if "predig_max_score" in merged.columns:
        cols["PredIG-Path (max)"] = "predig_max_score"
    if "predig_mean_score" in merged.columns:
        cols["PredIG-Path (mean)"] = "predig_mean_score"
    if "prime_score" in merged.columns:
        cols["PRIME 2.1 (max)"] = "prime_score"
    if "prime_mean_score" in merged.columns:
        cols["PRIME 2.1 (mean)"] = "prime_mean_score"
    return cols


def prevalence_baseline_row(n_peptides: int) -> dict:
    return {
        "tool": "Random / Prevalence baseline",
        "n_peptides": n_peptides,
        "auc_roc": 0.50,
        "auc_pr": PREVALENCE_POS_RATE,
        "issr_10": PREVALENCE_POS_RATE,
        "issr_25": PREVALENCE_POS_RATE,
        "delta_auc_pr_vs_prev": 0.0,
        "delta_issr10_vs_prev": 0.0,
    }


def evaluate_tools(
    df: pd.DataFrame,
    tool_columns: Dict[str, str],
    include_prevalence: bool = True,
) -> pd.DataFrame:
    rows = []
    for tool_name, score_col in tool_columns.items():
        if score_col not in df.columns:
            continue
        valid = df.dropna(subset=["label", score_col])
        if valid.empty:
            continue
        m = evaluate(valid["label"].values, valid[score_col].values)
        m["tool"] = tool_name
        m["n_peptides"] = len(valid)
        m["delta_auc_pr_vs_prev"] = m["auc_pr"] - PREVALENCE_POS_RATE
        m["delta_issr10_vs_prev"] = m["issr_10"] - PREVALENCE_POS_RATE
        rows.append(m)

    if include_prevalence and rows:
        rows.append(prevalence_baseline_row(int(rows[0]["n_peptides"])))

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    front = ["tool", "n_peptides"]
    return out[front + [c for c in out.columns if c not in front]]


def virus_weighted_metrics(
    df: pd.DataFrame, tool_columns: Dict[str, str]
) -> pd.DataFrame:
    rows = []
    viruses = sorted(df["virus"].dropna().unique()) if "virus" in df.columns else []
    for tool_name, score_col in tool_columns.items():
        if score_col not in df.columns:
            continue
        per_virus = {}
        for virus in viruses:
            sub = df[df["virus"] == virus].dropna(subset=["label", score_col])
            if len(sub) < 10:
                continue
            per_virus[virus] = evaluate(sub["label"].values, sub[score_col].values)

        if "EBV" in per_virus and "HPV16" in per_virus:
            ebv, hpv = per_virus["EBV"], per_virus["HPV16"]
            rows.append(
                {
                    "tool": tool_name,
                    "weighted_auc_pr": 0.5 * ebv["auc_pr"] + 0.5 * hpv["auc_pr"],
                    "weighted_issr_10": 0.5 * ebv["issr_10"] + 0.5 * hpv["issr_10"],
                    "weighted_auc_roc": 0.5 * ebv["auc_roc"] + 0.5 * hpv["auc_roc"],
                    "ebv_n": len(df[(df["virus"] == "EBV")]),
                    "hpv16_n": len(df[(df["virus"] == "HPV16")]),
                }
            )
    return pd.DataFrame(rows)


def length_stratified(df: pd.DataFrame, tool_columns: Dict[str, str]) -> pd.DataFrame:
    rows = []
    if "peptide" not in df.columns:
        return pd.DataFrame()
    for label, mask_fn in [
        ("9-mer", lambda d: d["peptide"].str.len() == 9),
        ("non-9-mer", lambda d: d["peptide"].str.len() != 9),
    ]:
        subset = df[mask_fn(df)]
        for tool_name, score_col in tool_columns.items():
            if score_col not in subset.columns:
                continue
            valid = subset.dropna(subset=["label", score_col])
            if len(valid) < 10:
                continue
            m = evaluate(valid["label"].values, valid[score_col].values)
            rows.append(
                {
                    "length_group": label,
                    "tool": tool_name,
                    "n": len(valid),
                    "auc_pr": m["auc_pr"],
                    "auc_roc": m["auc_roc"],
                    "issr_10": m["issr_10"],
                }
            )
    return pd.DataFrame(rows)


def holdout_spotlight(df: pd.DataFrame) -> pd.DataFrame:
    holdouts = set(GOLD_STANDARD_EPITOPES)
    sub = df[df["peptide"].isin(holdouts)].copy()
    score_cols = [
        c
        for c in [
            "binding_max",
            "predig_max_score",
            "predig_mean_score",
            "prime_score",
            "prime_mean_score",
        ]
        if c in sub.columns
    ]
    if sub.empty:
        return pd.DataFrame()
    cols = ["peptide", "label", "virus", "is_gold_standard_holdout"] + score_cols
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].sort_values("peptide")


def collapse_allele_scores(
    raw: pd.DataFrame,
    peptide_col: str,
    score_col: str,
    prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return max- and mean-collapsed per-peptide tables."""
    max_df = (
        raw.groupby(peptide_col)[score_col]
        .max()
        .reset_index()
        .rename(columns={peptide_col: "peptide", score_col: f"{prefix}_max_score"})
    )
    mean_df = (
        raw.groupby(peptide_col)[score_col]
        .mean()
        .reset_index()
        .rename(columns={peptide_col: "peptide", score_col: f"{prefix}_mean_score"})
    )
    return max_df, mean_df


def _find_col(df: pd.DataFrame, target: str) -> Optional[str]:
    low = target.lower()
    for c in df.columns:
        if c.lower() == low:
            return c
    return None


def parse_raw_predig(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    pep = _find_col(raw, "peptide") or _find_col(raw, "epitope")
    sc = _find_col(raw, "predig_score") or _find_col(raw, "PredIG") or _find_col(raw, "score")
    if pep is None or sc is None:
        raise ValueError(f"Cannot parse PredIG columns from {list(raw.columns)}")
    return collapse_allele_scores(raw, pep, sc, "predig")


def parse_raw_prime(path: str, sep: str = "\t") -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, sep=sep, comment="#")
    pep = _find_col(raw, "peptide") or _find_col(raw, "Peptide")
    sc = (
        _find_col(raw, "PRIME_score")
        or _find_col(raw, "Score_bestAllele")
        or _find_col(raw, "score")
    )
    if pep is None or sc is None:
        raise ValueError(f"Cannot parse PRIME columns from {list(raw.columns)}")
    return collapse_allele_scores(raw, pep, sc, "prime")


def quantify_overlap(
    eval_peptides: set,
    train_path: Optional[str],
    train_col: str = "epitope",
) -> dict:
    """Exact-match overlap only (legacy). Prefer quantify_overlap_robust."""
    meta = quantify_overlap_robust(eval_peptides, train_path, train_col)
    return {
        "overlap_count": meta.get("exact_overlap_count", "unknown"),
        "overlap_total_eval": meta.get("eval_peptide_count", "unknown"),
        "overlap_pct": meta.get("exact_overlap_pct", "unknown"),
        "status": meta.get("status", "missing_train_list"),
        "overlap_peptides_sample": meta.get("exact_overlap_peptides_sample", []),
    }


def quantify_overlap_robust(
    eval_peptides: set,
    train_path: Optional[str],
    train_col: str = "epitope",
    contamination_cap_pct: float = 30.0,
) -> dict:
    """Exact + substring overlap per External_More_Goals v2.0 §4.2."""
    if not train_path or not os.path.isfile(train_path):
        return {
            "overlap_count": "unknown",
            "overlap_pct": "unknown",
            "status": "missing_train_list",
            "exact_overlap_count": "unknown",
            "substring_overlap_count": "unknown",
            "total_overlap_count": "unknown",
            "total_overlap_pct": "unknown",
        }

    train = pd.read_csv(train_path)
    col = _find_col(train, train_col) or train.columns[0]
    train_peps = set(train[col].astype(str).str.upper())
    eval_upper = {p.upper() for p in eval_peptides}

    exact = eval_upper & train_peps
    substring = set()
    for test_pep in eval_upper - exact:
        for train_pep in train_peps:
            if test_pep in train_pep or train_pep in test_pep:
                substring.add(test_pep)
                break

    total = exact | substring
    n_eval = len(eval_upper)
    total_pct = 100.0 * len(total) / n_eval if n_eval else 0.0
    exact_pct = 100.0 * len(exact) / n_eval if n_eval else 0.0

    if total_pct > 50:
        status = "contaminated"
    elif total_pct > contamination_cap_pct:
        status = "contaminated_cap"
    else:
        status = "acceptable"

    return {
        "train_source": train_path,
        "eval_peptide_count": n_eval,
        "exact_overlap_count": len(exact),
        "exact_overlap_pct": round(exact_pct, 2),
        "substring_overlap_count": len(substring),
        "total_overlap_count": len(total),
        "total_overlap_pct": round(total_pct, 2),
        "status": status,
        "exact_overlap_peptides_sample": sorted(exact)[:25],
        "substring_only_peptides_sample": sorted(substring)[:25],
        "total_overlap_peptides": sorted(total),
        "overlap_count": len(total),
        "overlap_pct": round(total_pct, 2),
    }


def write_fairness_report(
    path: str,
    intersection_metrics: pd.DataFrame,
    external_full_metrics: pd.DataFrame,
    virus_weighted: pd.DataFrame,
    length_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    collapse_df: pd.DataFrame,
    overlap_meta: dict,
    overlap_subset_df: Optional[pd.DataFrame] = None,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    lines = [
        "# External Validation Fairness Analysis",
        "",
        f"Generated: {ts}",
        "",
        "## Mandatory Disclosure",
        "",
        "SESTRAV RF uses 5-fold out-of-fold predictions (conservative). PredIG and",
        "PRIME use fully-trained models that may have seen evaluation peptides during",
        "their own IEDB-based training.",
        "",
        f"Dataset composition: {PREVALENCE_POS_RATE:.1%} positive rate; "
        "57% 9-mers; 65% EBV / 35% HPV16.",
        "",
        "## Training Overlap",
        "",
        "```json",
        json.dumps(overlap_meta, indent=2),
        "```",
        "",
    ]

    def _table(title: str, df: pd.DataFrame) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if df.empty:
            lines.append("*No data*")
        else:
            lines.append(df.to_string(index=False))
        lines.append("")

    _table("Intersection Set Metrics (704 rows with RF OOF when available)", intersection_metrics)
    _table("External Full-Set Metrics (720 rows, external tools + binding)", external_full_metrics)
    _table("Virus-Weighted Averages (0.5 EBV + 0.5 HPV16)", virus_weighted)
    _table("Length-Stratified AUC-PR", length_df)
    _table("Mean vs Max Collapse Sensitivity", collapse_df)
    if overlap_subset_df is not None and not overlap_subset_df.empty:
        _table("Overlap-Excluded / Overlap-Only Subsets (intersection)", overlap_subset_df)
    _table("16 Gold-Standard Holdout Spotlight", holdout_df)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_fairness(
    merged_path: str,
    run_dir: str,
    predig_raw: Optional[str] = None,
    prime_raw: Optional[str] = None,
    predig_train: Optional[str] = None,
    prime_train: Optional[str] = None,
) -> str:
    os.makedirs(os.path.join(run_dir, "processed"), exist_ok=True)
    processed = os.path.join(run_dir, "processed")

    merged = pd.read_csv(merged_path)
    eval_peptides = set(merged["peptide"].astype(str))

    collapse_rows = []
    if predig_raw and os.path.isfile(predig_raw):
        max_d, mean_d = parse_raw_predig(predig_raw)
        merged = merged.merge(max_d, on="peptide", how="left", suffixes=("", "_dup"))
        merged = merged.merge(mean_d, on="peptide", how="left")
        for collapse, col in [("max", "predig_max_score"), ("mean", "predig_mean_score")]:
            valid = merged.dropna(subset=["label", col])
            if not valid.empty:
                m = evaluate(valid["label"].values, valid[col].values)
                collapse_rows.append({"tool": "PredIG-Path", "collapse": collapse, **m})

    if prime_raw and os.path.isfile(prime_raw):
        max_d, mean_d = parse_raw_prime(prime_raw)
        if "prime_score" not in merged.columns:
            merged = merged.merge(
                max_d.rename(columns={"prime_max_score": "prime_score"}),
                on="peptide",
                how="left",
            )
        if "prime_mean_score" not in merged.columns:
            merged = merged.merge(mean_d, on="peptide", how="left")
        for collapse, col in [("max", "prime_score"), ("mean", "prime_mean_score")]:
            if col not in merged.columns:
                continue
            valid = merged.dropna(subset=["label", col])
            if not valid.empty:
                m = evaluate(valid["label"].values, valid[col].values)
                collapse_rows.append({"tool": "PRIME 2.1", "collapse": collapse, **m})

    tools_all = _tool_columns(merged)

    rf_cols = {k: v for k, v in tools_all.items() if "RF" in k or "Binding" in k or "PredIG" in k or "PRIME" in k}
    external_cols = {k: v for k, v in tools_all.items() if "RF" not in k}

    intersection = merged.dropna(subset=["rf_oof_score"]) if "rf_oof_score" in merged.columns else merged
    intersection_tools = {k: v for k, v in rf_cols.items() if v in intersection.columns}

    intersection_metrics = evaluate_tools(intersection, intersection_tools)
    external_full_metrics = evaluate_tools(merged, external_cols)

    virus_weighted = virus_weighted_metrics(merged, rf_cols)
    length_df = length_stratified(merged, rf_cols)
    holdout_df = holdout_spotlight(merged)
    collapse_df = pd.DataFrame(collapse_rows)

    overlap_meta = {
        "predig": quantify_overlap_robust(eval_peptides, predig_train, train_col="Epitope"),
        "prime": quantify_overlap_robust(eval_peptides, prime_train, train_col="epitope"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Overlap subset metrics on intersection set
    overlap_subset_rows = []
    if "rf_oof_score" in merged.columns:
        inter = merged.dropna(subset=["rf_oof_score", "predig_max_score", "prime_score"])
        flagged = set()
        for key in ("predig", "prime"):
            peps = overlap_meta[key].get("total_overlap_peptides")
            if peps:
                flagged.update(p.upper() for p in peps)
        if flagged:
            inter = inter.copy()
            inter["_overlap"] = inter["peptide"].astype(str).str.upper().isin(flagged)
            for label, mask in [
                ("overlap-excluded", ~inter["_overlap"]),
                ("overlap-only", inter["_overlap"]),
            ]:
                sub = inter.loc[mask]
                if len(sub) >= 20:
                    m = evaluate_tools(sub, intersection_tools, include_prevalence=False)
                    m["subset"] = label
                    overlap_subset_rows.append(m)
    overlap_subset_df = (
        pd.concat(overlap_subset_rows, ignore_index=True) if overlap_subset_rows else pd.DataFrame()
    )
    if not overlap_subset_df.empty:
        overlap_subset_df.to_csv(
            os.path.join(processed, "overlap_subset_metrics.csv"), index=False
        )

    intersection_metrics.to_csv(
        os.path.join(processed, "fairness_metrics_intersection.csv"), index=False
    )
    external_full_metrics.to_csv(
        os.path.join(processed, "fairness_metrics_external_full.csv"), index=False
    )
    virus_weighted.to_csv(
        os.path.join(processed, "virus_weighted_metrics.csv"), index=False
    )
    length_df.to_csv(
        os.path.join(processed, "length_stratified_auc_pr.csv"), index=False
    )
    holdout_df.to_csv(os.path.join(processed, "holdout_spotlight.csv"), index=False)
    collapse_df.to_csv(
        os.path.join(processed, "mean_collapse_sensitivity.csv"), index=False
    )
    with open(os.path.join(run_dir, "manifests", "training_overlap.json"), "w", encoding="utf-8") as f:
        json.dump(overlap_meta, f, indent=2)

    report_path = os.path.join(processed, "fairness_analysis.md")
    write_fairness_report(
        report_path,
        intersection_metrics,
        external_full_metrics,
        virus_weighted,
        length_df,
        holdout_df,
        collapse_df,
        overlap_meta,
        overlap_subset_df,
    )
    print(f"[fairness] Wrote reports under {processed}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="External validation fairness supplement")
    parser.add_argument("--merged", default="results/external_validation_merged_scores.csv")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--predig-raw", default=None)
    parser.add_argument("--prime-raw", default=None)
    parser.add_argument("--predig-train", default="data/external/predig_train_modf.csv")
    parser.add_argument("--prime-train", default=None)
    args = parser.parse_args()

    run_fairness(
        args.merged,
        args.run_dir,
        predig_raw=args.predig_raw,
        prime_raw=args.prime_raw,
        predig_train=args.predig_train,
        prime_train=args.prime_train,
    )


if __name__ == "__main__":
    main()
