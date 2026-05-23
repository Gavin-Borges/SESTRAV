"""
SESTRAV External Benchmark Comparison — PredIG and PRIME

Automates Steps 7-13 of the External Validation Plan:
  1. Parse PredIG and PRIME raw output files
  2. Collapse per-allele scores to per-peptide (max over alleles)
  3. Merge all tool scores into a unified table
  4. Run evaluate() on every tool against shared labels
  5. Compute ISSR R10/R25 ratios, Spearman correlations, negative discrimination
  6. Bootstrap 95% CIs for AUC-PR and ISSR@10 differences vs SESTRAV RF
  7. Per-virus breakdown
  8. Generate ROC/PR overlay, score distribution, and rank scatter figures
  9. Write the comparison report (results/external_benchmark_comparison.md)

Usage:
    python -m src.external_benchmark_comparison \\
        --predig results/external_tool_outputs/predig_path_output.csv \\
        --prime results/external_tool_outputs/prime21_output.txt \\
        --results-dir results

All flag defaults match the paths in the External Validation Plan.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.evaluate_metrics import evaluate
from src.gold_standard import GOLD_STANDARD, GOLD_STANDARD_NEGATIVES

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# PredIG / PRIME parsing helpers
# ---------------------------------------------------------------------------

def parse_predig(
    path: str,
    peptide_col: str = "peptide",
    allele_col: str = "allele",
    score_col: str = "predig_score",
) -> pd.DataFrame:
    """Read PredIG output and collapse to one score per peptide (max over alleles)."""
    df = pd.read_csv(path)
    col_map = _fuzzy_column_map(df, peptide_col=peptide_col, score_col=score_col)
    peptide_key = col_map["peptide"]
    score_key = col_map["score"]

    collapsed = (
        df.groupby(peptide_key)[score_key]
        .max()
        .reset_index()
        .rename(columns={peptide_key: "peptide", score_key: "predig_max_score"})
    )
    return collapsed


def parse_prime(
    path: str,
    peptide_col: str = "peptide",
    score_col: str = "PRIME_score",
    pctrank_col: str = "pctrank",
    sep: str = "\t",
    use_pctrank: bool = False,
) -> pd.DataFrame:
    """Read PRIME output and produce one score per peptide.

    If the output is per-peptide-allele, collapse via max.
    If use_pctrank is True, inverts %Rank (1 - pct/100) so higher = better.
    """
    df = pd.read_csv(path, sep=sep, comment="#")
    col_map = _fuzzy_column_map(df, peptide_col=peptide_col, score_col=score_col)
    peptide_key = col_map["peptide"]
    score_key = col_map["score"]

    if use_pctrank:
        pct_key = _find_col(df, pctrank_col)
        if pct_key is None:
            raise ValueError(
                f"use_pctrank=True but column '{pctrank_col}' not found in PRIME output"
            )
        df["_prime_inverted"] = 1.0 - df[pct_key] / 100.0
        score_key = "_prime_inverted"

    is_per_allele = df.groupby(peptide_key).size().max() > 1
    if is_per_allele:
        collapsed = (
            df.groupby(peptide_key)[score_key]
            .max()
            .reset_index()
            .rename(columns={peptide_key: "peptide", score_key: "prime_score"})
        )
    else:
        collapsed = df[[peptide_key, score_key]].copy()
        collapsed = collapsed.rename(
            columns={peptide_key: "peptide", score_key: "prime_score"}
        )
    return collapsed


def _find_col(df: pd.DataFrame, target: str) -> Optional[str]:
    """Case-insensitive column lookup."""
    low = target.lower()
    for c in df.columns:
        if c.lower() == low:
            return c
    return None


def _fuzzy_column_map(
    df: pd.DataFrame, peptide_col: str, score_col: str
) -> Dict[str, str]:
    """Resolve user-specified column names against actual DataFrame columns."""
    pep = _find_col(df, peptide_col) or _find_col(df, "epitope") or _find_col(df, "Peptide")
    sc = (
        _find_col(df, score_col)
        or _find_col(df, "PredIG")
        or _find_col(df, "predig_score")
        or _find_col(df, "Score_bestAllele")
        or _find_col(df, "PRIME_score")
    )
    if pep is None:
        raise ValueError(
            f"Peptide column '{peptide_col}' not found. Available: {list(df.columns)}"
        )
    if sc is None:
        raise ValueError(
            f"Score column '{score_col}' not found. Available: {list(df.columns)}"
        )
    return {"peptide": pep, "score": sc}


# ---------------------------------------------------------------------------
# Merge and evaluate
# ---------------------------------------------------------------------------

def build_merged_table(
    base_path: str,
    oof_path: str,
    predig_scores: Optional[pd.DataFrame],
    prime_scores: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build the unified score table with all tools' scores aligned by peptide."""
    base = pd.read_csv(base_path)

    oof = pd.read_csv(oof_path)
    xgb_score = _extract_xgb_oof(oof)
    if xgb_score is not None:
        base = base.merge(xgb_score, on="peptide", how="left")

    if predig_scores is not None:
        base = base.merge(predig_scores, on="peptide", how="left")
    if prime_scores is not None:
        base = base.merge(prime_scores, on="peptide", how="left")

    return base


def _extract_xgb_oof(oof: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Extract XGBoost OOF scores if present in rf_oof_predictions.csv."""
    if "method" not in oof.columns:
        return None
    xgb = oof[oof["method"] == "XGBoost"].copy()
    if xgb.empty:
        return None
    xgb = xgb.drop_duplicates(subset=["peptide"], keep="first")
    return xgb[["peptide", "score"]].rename(columns={"score": "xgb_oof_score"})


def evaluate_all_tools(
    merged: pd.DataFrame,
    tool_columns: Dict[str, str],
) -> pd.DataFrame:
    """Run evaluate() for every tool and return a tidy DataFrame."""
    rows = []
    for tool_name, score_col in tool_columns.items():
        subset = merged.dropna(subset=["label", score_col])
        if subset.empty:
            continue
        m = evaluate(subset["label"].values, subset[score_col].values)
        m["tool"] = tool_name
        m["n_peptides"] = len(subset)
        rows.append(m)
    df = pd.DataFrame(rows)
    front = ["tool", "n_peptides"]
    return df[front + [c for c in df.columns if c not in front]]


# ---------------------------------------------------------------------------
# Derived comparisons
# ---------------------------------------------------------------------------

def compute_issr_ratios(
    results_df: pd.DataFrame,
    binding_tool: str = "Binding-only (max)",
) -> pd.DataFrame:
    """Compute R10 and R25 enrichment ratios relative to binding-only baseline."""
    bind_row = results_df.loc[results_df["tool"] == binding_tool]
    if bind_row.empty:
        return results_df
    b10 = float(bind_row["issr_10"].values[0])
    b25 = float(bind_row["issr_25"].values[0])
    results_df = results_df.copy()
    results_df["R10"] = results_df["issr_10"].apply(
        lambda x: x / b10 if b10 > 0.05 else np.nan
    )
    results_df["R25"] = results_df["issr_25"].apply(
        lambda x: x / b25 if b25 > 0.05 else np.nan
    )
    return results_df


def spearman_correlation_matrix(
    merged: pd.DataFrame,
    tool_columns: Dict[str, str],
) -> pd.DataFrame:
    """Pairwise Spearman rho between all tools."""
    rows = []
    tools = list(tool_columns.items())
    for i, (name_a, col_a) in enumerate(tools):
        for name_b, col_b in tools[i + 1 :]:
            valid = merged.dropna(subset=[col_a, col_b])
            if len(valid) < 5:
                continue
            rho, pval = spearmanr(valid[col_a], valid[col_b])
            rows.append(
                {"tool_a": name_a, "tool_b": name_b, "rho": rho, "p_value": pval}
            )
    return pd.DataFrame(rows)


def negative_discrimination(
    merged: pd.DataFrame,
    tool_columns: Dict[str, str],
    n_core: int = 10,
) -> pd.DataFrame:
    """Check how many gold-standard negatives each tool pushes down vs binding."""
    neg_peptides = [gs["peptide"] for gs in GOLD_STANDARD_NEGATIVES[:n_core]]
    neg_mask = merged["peptide"].isin(neg_peptides)
    neg_df = merged.loc[neg_mask].copy()
    if neg_df.empty:
        return pd.DataFrame()

    bind_ranks = merged["binding_max"].rank(ascending=False)
    rows = []
    for tool_name, score_col in tool_columns.items():
        tool_ranks = merged[score_col].rank(ascending=False)
        neg_tool_ranks = tool_ranks.loc[neg_mask]
        neg_bind_ranks = bind_ranks.loc[neg_mask]
        pushed = int((neg_tool_ranks > neg_bind_ranks).sum())
        rows.append(
            {
                "tool": tool_name,
                "negatives_evaluated": int(neg_mask.sum()),
                "pushed_down_vs_binding": pushed,
            }
        )
    return pd.DataFrame(rows)


def per_virus_metrics(
    merged: pd.DataFrame,
    tool_columns: Dict[str, str],
) -> pd.DataFrame:
    """Evaluate metrics per-virus for each tool."""
    virus_col = "virus"
    if virus_col not in merged.columns:
        return pd.DataFrame()
    rows = []
    for virus in sorted(merged[virus_col].dropna().unique()):
        subset = merged[merged[virus_col] == virus]
        for tool_name, score_col in tool_columns.items():
            valid = subset.dropna(subset=["label", score_col])
            if len(valid) < 10:
                continue
            m = evaluate(valid["label"].values, valid[score_col].values)
            m["tool"] = tool_name
            m["virus"] = virus
            m["n"] = len(valid)
            rows.append(m)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    front = ["tool", "virus", "n"]
    return df[front + [c for c in df.columns if c not in front]]


# ---------------------------------------------------------------------------
# Bootstrap statistical testing
# ---------------------------------------------------------------------------

def bootstrap_metric_diff(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_fn,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap 95% CI for metric(scores_a) - metric(scores_b)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        ma = metric_fn(y_b, scores_a[idx])
        mb = metric_fn(y_b, scores_b[idx])
        diffs.append(ma - mb)
    if not diffs:
        return np.nan, np.nan, np.nan
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(
        np.percentile(diffs, 97.5)
    )


def run_bootstrap_comparisons(
    merged: pd.DataFrame,
    tool_columns: Dict[str, str],
    reference_col: str = "rf_oof_score",
    reference_name: str = "SESTRAV RF (30-feat)",
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    """Bootstrap CIs for AUC-PR and ISSR@10 differences: reference vs each tool."""
    complete = merged.dropna(subset=["label", reference_col])
    y = complete["label"].values
    ref_scores = complete[reference_col].values

    rows = []
    for tool_name, score_col in tool_columns.items():
        if tool_name == reference_name:
            continue
        valid = complete.dropna(subset=[score_col])
        if len(valid) < 20:
            continue
        y_v = valid["label"].values
        ref_v = valid[reference_col].values
        tool_v = valid[score_col].values

        mean_ap, lo_ap, hi_ap = bootstrap_metric_diff(
            y_v, ref_v, tool_v, average_precision_score, n_bootstrap
        )
        from src.evaluate_metrics import issr_at_k

        def issr10(yt, ys):
            return issr_at_k(yt, ys, 10)

        mean_issr, lo_issr, hi_issr = bootstrap_metric_diff(
            y_v, ref_v, tool_v, issr10, n_bootstrap
        )

        rows.append(
            {
                "comparison": f"{reference_name} vs {tool_name}",
                "auc_pr_diff_mean": mean_ap,
                "auc_pr_diff_ci_low": lo_ap,
                "auc_pr_diff_ci_high": hi_ap,
                "auc_pr_significant": "yes" if lo_ap > 0 or hi_ap < 0 else "no",
                "issr10_diff_mean": mean_issr,
                "issr10_diff_ci_low": lo_issr,
                "issr10_diff_ci_high": hi_issr,
                "issr10_significant": "yes"
                if lo_issr > 0 or hi_issr < 0
                else "no",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def generate_figures(
    merged: pd.DataFrame,
    tool_columns: Dict[str, str],
    output_dir: str,
) -> List[str]:
    """Generate ROC/PR overlay, score distributions, and rank scatter plots."""
    if not _HAS_MPL:
        print("[comparison] matplotlib not available; skipping figures")
        return []

    complete = merged.dropna(subset=["label"] + list(tool_columns.values()))
    paths = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for tool_name, score_col in tool_columns.items():
        fpr, tpr, _ = roc_curve(complete["label"], complete[score_col])
        ax1.plot(fpr, tpr, label=tool_name)
        prec, rec, _ = precision_recall_curve(complete["label"], complete[score_col])
        ax2.plot(rec, prec, label=tool_name)
    ax1.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curves")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    roc_pr_path = os.path.join(output_dir, "external_benchmark_roc_pr_curves.png")
    fig.savefig(roc_pr_path, dpi=150)
    plt.close(fig)
    paths.append(roc_pr_path)

    n_tools = len(tool_columns)
    fig, axes = plt.subplots(1, n_tools, figsize=(4 * n_tools, 5), squeeze=False)
    for i, (tool_name, score_col) in enumerate(tool_columns.items()):
        ax = axes[0, i]
        pos = complete.loc[complete["label"] == 1, score_col]
        neg = complete.loc[complete["label"] == 0, score_col]
        ax.hist(neg, bins=30, alpha=0.6, label="Negative", density=True)
        ax.hist(pos, bins=30, alpha=0.6, label="Positive", density=True)
        ax.set_title(tool_name, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("Score")
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "external_benchmark_score_distributions.png")
    fig.savefig(dist_path, dpi=150)
    plt.close(fig)
    paths.append(dist_path)

    if "rf_oof_score" in merged.columns:
        ext_tools = {k: v for k, v in tool_columns.items() if v not in ("rf_oof_score", "binding_max")}
        if ext_tools:
            fig, axes = plt.subplots(
                1, len(ext_tools), figsize=(6 * len(ext_tools), 5), squeeze=False
            )
            gs_pos = {gs["peptide"] for gs in GOLD_STANDARD}
            gs_neg = {gs["peptide"] for gs in GOLD_STANDARD_NEGATIVES[:10]}
            for i, (tool_name, score_col) in enumerate(ext_tools.items()):
                ax = axes[0, i]
                rf_ranks = complete["rf_oof_score"].rank(ascending=False)
                tool_ranks = complete[score_col].rank(ascending=False)
                ax.scatter(rf_ranks, tool_ranks, s=8, alpha=0.3, color="gray")
                for pep_set, color, label in [
                    (gs_pos, "green", "GS positive"),
                    (gs_neg, "red", "GS negative"),
                ]:
                    mask = complete["peptide"].isin(pep_set)
                    ax.scatter(
                        rf_ranks[mask], tool_ranks[mask],
                        s=30, color=color, label=label, zorder=5,
                    )
                ax.set_xlabel("SESTRAV RF rank")
                ax.set_ylabel(f"{tool_name} rank")
                ax.set_title(f"SESTRAV RF vs {tool_name}")
                ax.legend(fontsize=7)
            plt.tight_layout()
            scatter_path = os.path.join(
                output_dir, "external_benchmark_rank_scatter.png"
            )
            fig.savefig(scatter_path, dpi=150)
            plt.close(fig)
            paths.append(scatter_path)

    return paths


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_comparison_report(
    output_dir: str,
    metrics_df: pd.DataFrame,
    ratios_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    neg_disc_df: pd.DataFrame,
    virus_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    figure_paths: List[str],
    merged: pd.DataFrame,
    provenance: Dict,
) -> str:
    """Write results/external_benchmark_comparison.md."""
    path = os.path.join(output_dir, "external_benchmark_comparison.md")
    ts = datetime.now(timezone.utc).isoformat()

    lines = [
        "# External Benchmark Comparison Report",
        "",
        f"Generated: {ts}",
        "",
        "## Provenance",
        "",
    ]
    for key, val in provenance.items():
        lines.append(f"- **{key}:** {val}")
    lines.append("")

    lines.append("## Head-to-Head Metric Table")
    lines.append("")
    lines.append(_df_to_md_table(ratios_df))
    lines.append("")

    if not corr_df.empty:
        lines.append("## Rank Correlation Matrix (Spearman rho)")
        lines.append("")
        lines.append(_df_to_md_table(corr_df))
        lines.append("")

    if not neg_disc_df.empty:
        lines.append("## Gold-Standard Negative Discrimination")
        lines.append("")
        lines.append(_df_to_md_table(neg_disc_df))
        lines.append("")

    if not virus_df.empty:
        lines.append("## Per-Virus Breakdown")
        lines.append("")
        lines.append(_df_to_md_table(virus_df))
        lines.append("")

    if not bootstrap_df.empty:
        lines.append("## Statistical Significance (Bootstrap 95% CI)")
        lines.append("")
        lines.append(_df_to_md_table(bootstrap_df))
        lines.append("")

    if figure_paths:
        lines.append("## Figures")
        lines.append("")
        for fp in figure_paths:
            fname = os.path.basename(fp)
            lines.append(f"- `{fname}`")
        lines.append("")

    n_total = len(merged)
    n_complete = len(
        merged.dropna(
            subset=[c for c in merged.columns if c.endswith("_score") or c == "binding_max"]
        )
    )
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append(f"- Total labeled peptides: {n_total}")
    lines.append(f"- Complete rows (all tools scored): {n_complete}")
    if "label" in merged.columns:
        lines.append(
            f"- Class balance: {int(merged['label'].sum())} positive / "
            f"{int((1 - merged['label']).sum())} negative"
        )
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def _df_to_md_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a markdown table string."""
    if df.empty:
        return "*No data*"
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 100 else f"{v:.2f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_comparison(
    predig_path: Optional[str] = None,
    prime_path: Optional[str] = None,
    results_dir: str = "results",
    oof_path: str = "models/rf_oof_predictions.csv",
    base_path: Optional[str] = None,
    predig_peptide_col: str = "peptide",
    predig_score_col: str = "predig_score",
    prime_peptide_col: str = "peptide",
    prime_score_col: str = "PRIME_score",
    prime_sep: str = "\t",
    prime_use_pctrank: bool = False,
    n_bootstrap: int = 2000,
) -> str:
    """End-to-end comparison pipeline. Returns path to the comparison report."""
    if base_path is None:
        base_path = os.path.join(results_dir, "external_validation_input.csv")

    provenance: Dict[str, str] = {
        "base_input": base_path,
        "oof_predictions": oof_path,
    }

    predig_scores = None
    if predig_path and os.path.isfile(predig_path):
        predig_scores = parse_predig(
            predig_path,
            peptide_col=predig_peptide_col,
            score_col=predig_score_col,
        )
        provenance["predig_file"] = predig_path
        provenance["predig_peptides_parsed"] = str(len(predig_scores))
        print(f"[comparison] PredIG: {len(predig_scores)} peptides parsed")

    prime_scores = None
    if prime_path and os.path.isfile(prime_path):
        prime_scores = parse_prime(
            prime_path,
            peptide_col=prime_peptide_col,
            score_col=prime_score_col,
            sep=prime_sep,
            use_pctrank=prime_use_pctrank,
        )
        provenance["prime_file"] = prime_path
        provenance["prime_peptides_parsed"] = str(len(prime_scores))
        provenance["prime_score_column"] = (
            "pctrank (inverted)" if prime_use_pctrank else prime_score_col
        )
        print(f"[comparison] PRIME: {len(prime_scores)} peptides parsed")

    provenance_path = os.path.join(
        results_dir, "external_tool_outputs", "provenance.json"
    )
    if os.path.isfile(provenance_path):
        with open(provenance_path, encoding="utf-8") as f:
            ext_prov = json.load(f)
        for key in ("predig", "prime"):
            if key in ext_prov:
                sub = ext_prov[key]
                provenance[f"{key}_version"] = sub.get("version", "unknown")
                provenance[f"{key}_run_date"] = sub.get("run_date", "unknown")

    merged = build_merged_table(base_path, oof_path, predig_scores, prime_scores)
    merged_path = os.path.join(results_dir, "external_validation_merged_scores.csv")
    merged.to_csv(merged_path, index=False)
    print(f"[comparison] Merged table: {len(merged)} rows -> {merged_path}")

    tool_columns = _discover_tool_columns(merged)
    print(f"[comparison] Tools discovered: {list(tool_columns.keys())}")

    for col_name, col_key in tool_columns.items():
        n_valid = merged[col_key].notna().sum()
        print(f"  {col_name} ({col_key}): {n_valid}/{len(merged)} non-null")

    metrics_df = evaluate_all_tools(merged, tool_columns)
    ratios_df = compute_issr_ratios(metrics_df)
    print("\n[comparison] Metrics:")
    print(ratios_df.to_string(index=False))

    corr_df = spearman_correlation_matrix(merged, tool_columns)
    neg_disc_df = negative_discrimination(merged, tool_columns)
    virus_df = per_virus_metrics(merged, tool_columns)
    bootstrap_df = run_bootstrap_comparisons(
        merged, tool_columns, n_bootstrap=n_bootstrap
    )

    figure_paths = generate_figures(merged, tool_columns, results_dir)

    report_path = write_comparison_report(
        results_dir,
        metrics_df,
        ratios_df,
        corr_df,
        neg_disc_df,
        virus_df,
        bootstrap_df,
        figure_paths,
        merged,
        provenance,
    )
    print(f"\n[comparison] Report written: {report_path}")

    metrics_csv = os.path.join(results_dir, "external_tool_metrics_by_virus.csv")
    if not virus_df.empty:
        virus_df.to_csv(metrics_csv, index=False)
        print(f"[comparison] Per-virus metrics: {metrics_csv}")

    return report_path


def _discover_tool_columns(merged: pd.DataFrame) -> Dict[str, str]:
    """Discover which tool score columns are present in the merged table."""
    candidates = [
        ("SESTRAV RF (30-feat)", "rf_oof_score"),
        ("SESTRAV XGBoost", "xgb_oof_score"),
        ("Binding-only (max)", "binding_max"),
        ("PredIG-Path", "predig_max_score"),
        ("PRIME 2.1", "prime_score"),
    ]
    found = {}
    for name, col in candidates:
        if col in merged.columns and merged[col].notna().any():
            found[name] = col
    return found


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SESTRAV External Benchmark Comparison (PredIG / PRIME)"
    )
    parser.add_argument(
        "--predig",
        default=None,
        help="Path to PredIG raw output CSV (default: not provided)",
    )
    parser.add_argument(
        "--prime",
        default=None,
        help="Path to PRIME raw output file (default: not provided)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for inputs and outputs (default: results)",
    )
    parser.add_argument(
        "--oof-path",
        default="models/rf_oof_predictions.csv",
        help="Path to OOF predictions CSV",
    )
    parser.add_argument(
        "--base-path",
        default=None,
        help="Path to external_validation_input.csv (default: results-dir/external_validation_input.csv)",
    )
    parser.add_argument(
        "--predig-peptide-col", default="peptide",
        help="PredIG peptide column name",
    )
    parser.add_argument(
        "--predig-score-col", default="predig_score",
        help="PredIG score column name",
    )
    parser.add_argument(
        "--prime-peptide-col", default="peptide",
        help="PRIME peptide column name",
    )
    parser.add_argument(
        "--prime-score-col", default="PRIME_score",
        help="PRIME score column name",
    )
    parser.add_argument(
        "--prime-sep", default="\t",
        help="PRIME file delimiter (default: tab)",
    )
    parser.add_argument(
        "--prime-use-pctrank", action="store_true",
        help="Use inverted %%Rank instead of PRIME Score (fallback only)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=2000,
        help="Number of bootstrap resamples (default: 2000)",
    )
    args = parser.parse_args()

    run_comparison(
        predig_path=args.predig,
        prime_path=args.prime,
        results_dir=args.results_dir,
        oof_path=args.oof_path,
        base_path=args.base_path,
        predig_peptide_col=args.predig_peptide_col,
        predig_score_col=args.predig_score_col,
        prime_peptide_col=args.prime_peptide_col,
        prime_score_col=args.prime_score_col,
        prime_sep=args.prime_sep,
        prime_use_pctrank=args.prime_use_pctrank,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
