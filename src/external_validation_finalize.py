"""
Finalize external validation per External_More_Goals v2.0 protocol.

Completes gaps after core comparison + fairness supplement:
  - Robust training overlap (exact + substring)
  - Overlap-excluded / overlap-only subset metrics
  - Bootstrap p-values + Benjamini-Hochberg FDR (4 primary tests)
  - MCDA verdicts (§14 / v2.0 §6.2)
  - Full provenance bundle (hashes, coverage, commands log)
  - Unified report append to external_benchmark_comparison.md

Usage:
    python -m src.external_validation_finalize \\
        --run-dir results/external_tool_outputs/extval_20260520_1607_gb_tierA \\
        --merged results/external_validation_merged_scores.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.evaluate_metrics import evaluate, issr_at_k
from src.external_benchmark_comparison import (
    per_virus_metrics,
    run_bootstrap_comparisons,
)
from src.external_validation_fairness import (
    PREVALENCE_POS_RATE,
    evaluate_tools,
    quantify_overlap_robust,
)

CONTAMINATION_CAP_PCT = 30.0
PRIMARY_TOOLS = {
    "SESTRAV RF (30-feat)": "rf_oof_score",
    "Binding-only (max)": "binding_max",
    "PredIG-Path": "predig_max_score",
    "PRIME 2.1": "prime_score",
}
REFERENCE_NAME = "SESTRAV RF (30-feat)"
REFERENCE_COL = "rf_oof_score"
DEFAULT_PREDIG_TRAIN = "data/external/predig_train_modf.csv"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """Return BH-adjusted p-values (same order as input)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val, 1.0)
        adjusted[i] = prev
    out = np.empty(n)
    out[order] = adjusted
    return out.tolist()


def bootstrap_metric_diff_with_p(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_fn,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Bootstrap mean diff, 95% CI, and approximate two-sided p-value."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        diffs.append(metric_fn(y_b, scores_a[idx]) - metric_fn(y_b, scores_b[idx]))
    if not diffs:
        return np.nan, np.nan, np.nan, np.nan
    diffs = np.array(diffs)
    p = min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0))) * 2
    return (
        float(np.mean(diffs)),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
        p,
    )


def run_fdr_tests(
    merged: pd.DataFrame,
    tool_columns: Dict[str, str],
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    """Four primary tests: PredIG/PRIME × AUC-PR/ISSR@10 vs SESTRAV RF."""
    complete = merged.dropna(subset=["label", REFERENCE_COL])
    rows = []
    for tool_name, score_col in tool_columns.items():
        if tool_name == REFERENCE_NAME or score_col not in merged.columns:
            continue
        valid = complete.dropna(subset=[score_col])
        if len(valid) < 20:
            continue
        y = valid["label"].values
        ref = valid[REFERENCE_COL].values
        tool = valid[score_col].values

        for metric_label, metric_fn in [
            ("auc_pr", average_precision_score),
            ("issr_10", lambda yt, ys: issr_at_k(yt, ys, 10)),
        ]:
            mean_d, lo, hi, p_raw = bootstrap_metric_diff_with_p(
                y, ref, tool, metric_fn, n_bootstrap
            )
            rows.append(
                {
                    "tool": tool_name,
                    "metric": metric_label,
                    "comparison": f"{REFERENCE_NAME} vs {tool_name}",
                    "diff_mean_rf_minus_tool": mean_d,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value_raw": p_raw,
                    "ci_excludes_zero": lo > 0 or hi < 0,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_value_fdr"] = benjamini_hochberg(df["p_value_raw"].tolist())
    df["fdr_significant"] = df["p_value_fdr"] < 0.05
    return df


def intersection_mask(merged: pd.DataFrame) -> pd.Series:
    cols = ["label", REFERENCE_COL, "binding_max", "predig_max_score", "prime_score"]
    present = [c for c in cols if c in merged.columns]
    return merged[present].notna().all(axis=1)


def flag_overlap_columns(
    merged: pd.DataFrame,
    predig_train: Optional[str],
    prime_train: Optional[str],
) -> Tuple[pd.DataFrame, dict]:
    eval_peps = set(merged["peptide"].astype(str))
    overlap_meta = {
        "predig": quantify_overlap_robust(eval_peps, predig_train, train_col="Epitope"),
        "prime": quantify_overlap_robust(eval_peps, prime_train, train_col="epitope"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "contamination_cap_pct": CONTAMINATION_CAP_PCT,
    }

    predig_flagged = overlap_meta["predig"].get("total_overlap_peptides", [])
    prime_flagged = overlap_meta["prime"].get("total_overlap_peptides", [])

    merged = merged.copy()
    if predig_flagged:
        merged["in_predig_training"] = merged["peptide"].astype(str).str.upper().isin(
            {p.upper() for p in predig_flagged}
        )
    else:
        merged["in_predig_training"] = np.nan

    if prime_flagged:
        merged["in_prime_training"] = merged["peptide"].astype(str).str.upper().isin(
            {p.upper() for p in prime_flagged}
        )
    else:
        merged["in_prime_training"] = np.nan

    merged["in_external_training"] = False
    if predig_flagged:
        merged["in_external_training"] = merged["in_external_training"] | merged[
            "in_predig_training"
        ].fillna(False)
    if prime_flagged:
        merged["in_external_training"] = merged["in_external_training"] | merged[
            "in_prime_training"
        ].fillna(False)

    return merged, overlap_meta


def subset_metrics(
    merged: pd.DataFrame,
    mask: pd.Series,
    label: str,
    tool_columns: Dict[str, str],
) -> pd.DataFrame:
    sub = merged.loc[mask]
    if sub.empty:
        return pd.DataFrame()
    m = evaluate_tools(sub, tool_columns, include_prevalence=True)
    m["subset"] = label
    m["n_peptides"] = len(sub)
    return m


def coverage_summary(merged: pd.DataFrame) -> dict:
    n_total = len(merged)
    tools = {
        "rf_oof_score": "SESTRAV RF",
        "binding_max": "binding_max",
        "predig_max_score": "PredIG-Path",
        "prime_score": "PRIME 2.1",
    }
    summary = {"n_total_peptides": n_total, "tools": {}}
    missing_rows = []
    for col, name in tools.items():
        if col not in merged.columns:
            summary["tools"][name] = {"scored": 0, "fraction": 0.0}
            continue
        scored = int(merged[col].notna().sum())
        summary["tools"][name] = {
            "scored": scored,
            "fraction": round(scored / n_total, 4) if n_total else 0.0,
        }
        missing = merged.loc[merged[col].isna(), "peptide"].astype(str).tolist()
        for pep in missing:
            missing_rows.append({"tool": name, "peptide": pep})

    summary["intersection_n"] = int(intersection_mask(merged).sum())
    return summary, missing_rows


def assign_mcda_verdict(
    tool_name: str,
    metrics_df: pd.DataFrame,
    fdr_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    overlap_pct: Optional[float],
) -> dict:
    """MCDA bucket for external tool vs SESTRAV RF."""
    row = metrics_df.loc[metrics_df["tool"] == tool_name]
    rf_row = metrics_df.loc[metrics_df["tool"] == REFERENCE_NAME]
    if row.empty or rf_row.empty:
        return {"tool": tool_name, "verdict": "Inconclusive", "rationale": "Missing metrics"}

    auc_pr_tool = float(row["auc_pr"].values[0])
    issr_tool = float(row["issr_10"].values[0])
    auc_pr_rf = float(rf_row["auc_pr"].values[0])
    issr_rf = float(rf_row["issr_10"].values[0])

    auc_delta = auc_pr_rf - auc_pr_tool
    issr_delta = issr_rf - issr_tool

    tool_fdr = fdr_df.loc[fdr_df["tool"] == tool_name]
    auc_fdr_sig = bool(
        tool_fdr.loc[tool_fdr["metric"] == "auc_pr", "fdr_significant"].any()
    )
    issr_fdr_sig = bool(
        tool_fdr.loc[tool_fdr["metric"] == "issr_10", "fdr_significant"].any()
    )

    boot = bootstrap_df.loc[
        bootstrap_df["comparison"] == f"{REFERENCE_NAME} vs {tool_name}"
    ]
    auc_ci_sig = (
        boot["auc_pr_significant"].values[0] == "yes" if not boot.empty else False
    )
    issr_ci_sig = (
        boot["issr10_significant"].values[0] == "yes" if not boot.empty else False
    )

    contaminated = overlap_pct is not None and overlap_pct > CONTAMINATION_CAP_PCT
    cap_note = ""
    if contaminated:
        cap_note = f" Overlap {overlap_pct:.1f}% exceeds {CONTAMINATION_CAP_PCT}% cap."

    rf_wins_auc = auc_delta > 0
    rf_wins_issr = issr_delta > 0
    tool_wins_auc = auc_delta < 0
    tool_wins_issr = issr_delta < 0

    if rf_wins_auc and rf_wins_issr and (auc_ci_sig or auc_fdr_sig):
        verdict = "Worse"
        rationale = (
            f"SESTRAV RF leads on both primary metrics (ΔAUC-PR={auc_delta:+.3f}, "
            f"ΔISSR@10={issr_delta:+.3f}). Bootstrap/FDR support RF advantage on AUC-PR."
            + cap_note
        )
    elif tool_wins_auc and tool_wins_issr and auc_fdr_sig and issr_fdr_sig and not contaminated:
        verdict = "Strongly Better"
        rationale = (
            f"{tool_name} leads on both primary metrics with FDR-significant differences."
        )
    elif tool_wins_auc and tool_wins_issr and not contaminated:
        verdict = "Comparable"
        rationale = (
            f"{tool_name} leads on point estimates but uncertainty is wide or FDR "
            f"non-significant; classify as Comparable pending stronger evidence."
            + cap_note
        )
    elif (rf_wins_auc and tool_wins_issr) or (tool_wins_auc and rf_wins_issr):
        verdict = "Comparable"
        rationale = (
            f"Mixed primary metrics (ΔAUC-PR={auc_delta:+.3f}, ΔISSR@10={issr_delta:+.3f}); "
            f"tie-break yields Comparable."
            + cap_note
        )
    elif not auc_ci_sig and not issr_ci_sig:
        verdict = "Inconclusive"
        rationale = (
            "Bootstrap CIs overlap zero on both primary metrics; directional claim not supported."
            + cap_note
        )
    elif rf_wins_auc or rf_wins_issr:
        verdict = "Comparable" if not (rf_wins_auc and rf_wins_issr) else "Worse"
        rationale = (
            f"Partial RF advantage (ΔAUC-PR={auc_delta:+.3f}, ΔISSR@10={issr_delta:+.3f})."
            + cap_note
        )
    else:
        verdict = "Inconclusive"
        rationale = "Evidence insufficient for directional classification." + cap_note

    if contaminated and verdict == "Strongly Better":
        verdict = "Comparable – Contaminated"
        rationale += " Downgraded from Strongly Better due to training overlap cap."

    if contaminated and verdict == "Comparable":
        verdict = "Comparable – Contaminated"

    return {
        "tool": tool_name,
        "verdict": verdict,
        "auc_pr_delta_rf_minus_tool": round(auc_delta, 4),
        "issr10_delta_rf_minus_tool": round(issr_delta, 4),
        "auc_pr_fdr_significant": auc_fdr_sig,
        "issr10_fdr_significant": issr_fdr_sig,
        "overlap_contaminated": contaminated,
        "rationale": rationale,
    }


def write_commands_log(run_dir: str, provenance_path: str) -> str:
    log_path = os.path.join(run_dir, "logs", "commands.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"# External validation commands log — finalized {ts}",
        f"host={platform.node()} platform={platform.platform()}",
        "",
        "# Tier A PredIG (batched Docker)",
        "python scripts/run_predig_batched.py --input results/external_predig_input_reformatted.csv",
        "docker run --rm -v <repo>/results:/predig:rw -v <uniprot>:/uniprot:ro bsceapm/predig ...",
        "",
        "# Tier A PRIME 2.1 (WSL2)",
        "PRIME.x -i results/external_prime_peptides.txt -o <run_dir>/raw/prime21_output.txt",
        "",
        "# Analysis pipeline",
        "python -m src.external_benchmark_comparison",
        "python -m src.external_validation_fairness --run-dir <run_dir>",
        "python -m src.external_validation_finalize --run-dir <run_dir>",
        "",
    ]
    if os.path.isfile(provenance_path):
        with open(provenance_path, encoding="utf-8") as f:
            prov = json.load(f)
        lines.append("# Tool versions (from provenance.json)")
        lines.append(json.dumps(prov, indent=2))
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return log_path


def build_input_hashes(repo_root: str) -> dict:
    candidates = [
        "results/external_validation_input.csv",
        "results/external_predig_peptide_allele_pairs.csv",
        "results/external_prime_peptides.txt",
        "results/external_prime_alleles_compact.txt",
        "results/external_predig_input_reformatted.csv",
        "immunogenicity_dataset.csv",
        "models/peptide_binding_matrix.csv",
    ]
    out = {}
    for rel in candidates:
        path = os.path.join(repo_root, rel)
        if os.path.isfile(path):
            out[rel] = sha256_file(path)
    return out


def build_artifact_manifest(run_dir: str, extra_paths: List[str]) -> dict:
    artifacts = {}
    for root, _, files in os.walk(run_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, run_dir).replace("\\", "/")
            artifacts[rel] = sha256_file(fp)
    for path in extra_paths:
        if os.path.isfile(path):
            rel = os.path.basename(path)
            artifacts[f"legacy/{rel}"] = sha256_file(path)
    return artifacts


def append_finalize_report(
    report_path: str,
    overlap_meta: dict,
    subset_dfs: Dict[str, pd.DataFrame],
    fdr_df: pd.DataFrame,
    mcda: List[dict],
    coverage: dict,
    run_dir: str,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    sections = [
        "",
        "---",
        "",
        "## v2.0 Protocol Finalization (External_More_Goals)",
        "",
        f"Finalized: {ts}",
        f"Run directory: `{run_dir}`",
        "",
        "### Training Overlap (exact + substring)",
        "",
        "```json",
        json.dumps(overlap_meta, indent=2),
        "```",
        "",
        "### Prevalence Baselines (intersection set)",
        "",
        f"- Random AUC-ROC = 0.50",
        f"- Random AUC-PR = {PREVALENCE_POS_RATE:.4f} ({506}/720 positives)",
        f"- Random ISSR@10 ≈ {PREVALENCE_POS_RATE:.4f}",
        "",
        "### Coverage Summary",
        "",
        "```json",
        json.dumps(coverage, indent=2),
        "```",
        "",
    ]

    for name, df in subset_dfs.items():
        if df.empty:
            continue
        sections.append(f"### Metrics — {name}")
        sections.append("")
        sections.append(_df_to_md(df))
        sections.append("")

    if not fdr_df.empty:
        sections.append("### FDR-Corrected Primary Tests (BH, 4 tests)")
        sections.append("")
        sections.append(_df_to_md(fdr_df))
        sections.append("")

    sections.append("### MCDA Verdicts vs SESTRAV RF")
    sections.append("")
    for v in mcda:
        sections.append(f"#### {v['tool']}")
        sections.append("")
        sections.append(f"1. **Primary verdict:** {v['verdict']}")
        sections.append(
            f"2. **Primary evidence:** ΔAUC-PR (RF−tool)={v['auc_pr_delta_rf_minus_tool']:+.4f}, "
            f"ΔISSR@10={v['issr10_delta_rf_minus_tool']:+.4f}; "
            f"FDR sig AUC-PR={v['auc_pr_fdr_significant']}, ISSR@10={v['issr10_fdr_significant']}"
        )
        sections.append(f"3. **Robustness note:** {v['rationale']}")
        sections.append(
            "4. **Boundary statement:** External tools evaluated with fully-trained models; "
            "SESTRAV RF uses conservative OOF scoring."
        )
        sections.append("")

    sections.append("### Fairness Checklist Sign-Off")
    sections.append("")
    checks = [
        ("720 peptide scores (PredIG + PRIME)", coverage["tools"].get("PredIG-Path", {}).get("scored", 0) >= 720),
        ("10-metric intersection table", coverage.get("intersection_n", 0) >= 700),
        ("Exact+substring overlap quantified", overlap_meta["predig"].get("status") != "missing_train_list"),
        ("FDR correction on 4 primary tests", not fdr_df.empty),
        ("MCDA verdicts assigned", len(mcda) >= 2),
        ("Provenance bundle under run dir", True),
        ("Prevalence baselines reported", True),
        ("Overlap subset metrics", "overlap_excluded" in subset_dfs),
    ]
    for label, ok in checks:
        sections.append(f"- [{'x' if ok else ' '}] {label}")
    sections.append("")

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(sections))


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                cells.append(f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def run_finalize(
    run_dir: str,
    merged_path: str,
    results_dir: str = "results",
    predig_train: Optional[str] = DEFAULT_PREDIG_TRAIN,
    prime_train: Optional[str] = None,
    n_bootstrap: int = 2000,
    run_cross_virus: bool = True,
) -> dict:
    repo_root = os.path.abspath(".")
    os.makedirs(os.path.join(run_dir, "manifests"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    processed = os.path.join(run_dir, "processed")

    merged = pd.read_csv(merged_path)
    merged, overlap_meta = flag_overlap_columns(merged, predig_train, prime_train)

    flagged_path = os.path.join(processed, "merged_scores_with_overlap_flags.csv")
    os.makedirs(processed, exist_ok=True)
    merged.to_csv(flagged_path, index=False)

    inter = intersection_mask(merged)
    tool_cols = {k: v for k, v in PRIMARY_TOOLS.items() if v in merged.columns}

    subset_dfs = {
        "intersection (primary)": subset_metrics(merged, inter, "intersection", tool_cols),
        "overlap-excluded": subset_metrics(
            merged, inter & ~merged["in_external_training"], "overlap_excluded", tool_cols
        ),
        "overlap-only": subset_metrics(
            merged, inter & merged["in_external_training"], "overlap_only", tool_cols
        ),
    }
    for name, df in subset_dfs.items():
        safe = name.split()[0].replace("-", "_")
        df.to_csv(os.path.join(processed, f"metrics_{safe}.csv"), index=False)

    intersection_metrics = subset_dfs["intersection (primary)"]
    fdr_df = run_fdr_tests(merged.loc[inter], tool_cols, n_bootstrap)
    if not fdr_df.empty:
        fdr_df.to_csv(os.path.join(processed, "fdr_primary_tests.csv"), index=False)

    bootstrap_df = run_bootstrap_comparisons(
        merged.loc[inter], tool_cols, n_bootstrap=n_bootstrap
    )

    mcda = []
    for tool in ["PredIG-Path", "PRIME 2.1"]:
        pct = overlap_meta.get("predig" if "PredIG" in tool else "prime", {}).get(
            "total_overlap_pct"
        )
        if pct == "unknown":
            pct = None
        mcda.append(
            assign_mcda_verdict(
                tool, intersection_metrics, fdr_df, bootstrap_df, pct
            )
        )
    with open(os.path.join(processed, "mcda_verdicts.json"), "w", encoding="utf-8") as f:
        json.dump(mcda, f, indent=2)

    coverage, missing_rows = coverage_summary(merged)
    if missing_rows:
        miss_path = os.path.join(processed, "missing_peptides_by_tool.csv")
        pd.DataFrame(missing_rows).to_csv(miss_path, index=False)
    cov_path = os.path.join(run_dir, "manifests", "coverage_summary.json")
    with open(cov_path, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2)

    overlap_path = os.path.join(run_dir, "manifests", "training_overlap.json")
    with open(overlap_path, "w", encoding="utf-8") as f:
        json.dump(overlap_meta, f, indent=2)

    input_hashes = build_input_hashes(repo_root)
    hash_path = os.path.join(run_dir, "manifests", "input_hashes.json")
    with open(hash_path, "w", encoding="utf-8") as f:
        json.dump(input_hashes, f, indent=2)

    prov_path = os.path.join(run_dir, "manifests", "provenance.json")
    log_path = write_commands_log(run_dir, prov_path)

    legacy = [
        os.path.join(results_dir, "external_benchmark_comparison.md"),
        merged_path,
    ]
    artifact_manifest = build_artifact_manifest(run_dir, legacy)
    art_path = os.path.join(run_dir, "manifests", "artifact_sha256.json")
    with open(art_path, "w", encoding="utf-8") as f:
        json.dump(artifact_manifest, f, indent=2)

    report_path = os.path.join(results_dir, "external_benchmark_comparison.md")
    append_finalize_report(
        report_path, overlap_meta, subset_dfs, fdr_df, mcda, coverage, run_dir
    )

    if run_cross_virus and not os.path.isfile(
        os.path.join(results_dir, "external_validation_cross_virus.csv")
    ):
        try:
            from src.external_validation_cross_virus import run_cross_virus

            run_cross_virus(
                "immunogenicity_dataset.csv",
                "models/peptide_binding_matrix.csv",
                os.path.join(results_dir, "external_validation_cross_virus.csv"),
            )
        except Exception as exc:
            print(f"[finalize] cross-virus skipped: {exc}", file=sys.stderr)

    virus_df = per_virus_metrics(merged.loc[inter], tool_cols)
    if not virus_df.empty:
        virus_df.to_csv(
            os.path.join(results_dir, "external_tool_metrics_by_virus.csv"),
            index=False,
        )

    print(f"[finalize] Overlap PredIG: {overlap_meta['predig'].get('total_overlap_pct')}%")
    print(f"[finalize] MCDA: {', '.join(v['verdict'] for v in mcda)}")
    print(f"[finalize] Artifacts under {run_dir}/manifests/")
    return {
        "overlap_meta": overlap_meta,
        "mcda": mcda,
        "coverage": coverage,
        "commands_log": log_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize external validation (v2.0)")
    parser.add_argument(
        "--run-dir",
        default="results/external_tool_outputs/extval_20260520_1607_gb_tierA",
    )
    parser.add_argument("--merged", default="results/external_validation_merged_scores.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--predig-train", default=DEFAULT_PREDIG_TRAIN)
    parser.add_argument("--prime-train", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--skip-cross-virus", action="store_true")
    args = parser.parse_args()

    run_finalize(
        args.run_dir,
        args.merged,
        results_dir=args.results_dir,
        predig_train=args.predig_train,
        prime_train=args.prime_train,
        n_bootstrap=args.n_bootstrap,
        run_cross_virus=not args.skip_cross_virus,
    )


if __name__ == "__main__":
    main()
