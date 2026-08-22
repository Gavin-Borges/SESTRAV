"""
SESTRAV Benchmark Runner - Stage 1 Core
========================================
Converts the one-time May 2026 external benchmark into a reproducible,
CLI-invocable protocol.

Usage:
    python scripts/benchmark_runner.py --tier A --run-id 20260606_v1
    python scripts/benchmark_runner.py --tier B --run-id 20260606_tier_b

Outputs (written to results/external_tool_outputs/<run-id>/):
    merged_scores.csv      - filtered + annotated score table
    metrics_summary.csv    - 10 core metrics per tool (AUC-PR, AUC-ROC, ISSR@10/25, ...)
    run_manifest.json      - metadata: run_id, tier, n_total, n_clean, contamination_rate
    length_stratified.csv  - 9-mer vs. non-9-mer breakdown per tool

Contamination gate:
    Flags runs where the exact + substring overlap between training set and
    eval set exceeds 30% (hard cap per master plan).

Freeze check:
    Reads results/freeze_status.json; aborts if valid != true.
"""

import argparse
import json
import sys
import datetime
from pathlib import Path
import ahocorasick

import pandas as pd
import numpy as np

# ── Add project root to sys.path so src.evaluate_metrics is importable ──────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_metrics import evaluate  # noqa: E402


# ── Constants ────────────────────────────────────────────────────────────────

FREEZE_STATUS_PATH = PROJECT_ROOT / "results" / "freeze_status.json"
MERGED_SCORES_PATH = PROJECT_ROOT / "results" / "external_validation_merged_scores.csv"
TRAINING_DATA_PATH = PROJECT_ROOT / "immunogenicity_dataset.csv"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "external_tool_outputs"

# Contamination cap: if overlap_rate > this, gate fails
CONTAMINATION_CAP = 0.30

# Tools present in the merged scores CSV (column names for scores)
TOOL_SCORE_COLUMNS = {
    "rf": "rf_oof_score",
    "binding": "binding_max",
}

# Optional external tool columns that may exist
OPTIONAL_TOOL_COLUMNS = {
    "prime": "prime_score",
    "predig": "predig_score",
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def get_git_commit_hash() -> str:
    """Retrieve the current Git commit hash of the repository."""
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def get_model_checkpoint_name() -> str:
    """Retrieve the model checkpoint filename from config.yaml."""
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                model_path = config.get("model_path")
                if model_path:
                    return Path(model_path).name
        except Exception:
            pass
    return "unknown"


def load_freeze_status() -> dict:
    """Load and return freeze_status.json. Abort if not valid."""
    if not FREEZE_STATUS_PATH.exists():
        print(f"[ERROR] freeze_status.json not found at {FREEZE_STATUS_PATH}")
        sys.exit(1)
    with open(FREEZE_STATUS_PATH) as f:
        status = json.load(f)
    if not status.get("valid", False):
        print("[ERROR] freeze_status.json reports valid=false. Run full_validation_report first.")
        sys.exit(1)
    return status


def load_merged_scores(tier: str) -> pd.DataFrame:
    """Load the canonical merged scores CSV and filter by tier."""
    if not MERGED_SCORES_PATH.exists():
        print(f"[ERROR] Merged scores not found: {MERGED_SCORES_PATH}")
        sys.exit(1)

    df = pd.read_csv(MERGED_SCORES_PATH)
    required = {
        "peptide",
        "label",
        "virus",
        "rf_oof_score",
        "binding_max",
        "tier_a_baseline_complete",
    }
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] merged_scores.csv missing columns: {missing}")
        sys.exit(1)

    if tier == "A":
        df = df[df["tier_a_baseline_complete"] == True].copy()  # noqa: E712
    # Tier B: use entire file (no additional filter needed for MVP)

    return df


def filter_contaminated_peptides(training_sequences: list, evaluation_sequences: list) -> list:
    """
    Highly efficient contamination gate using Aho-Corasick.
    Builds the trie using the evaluation sequences, and parses the training sequences
    in a single pass to flag any substring matches.
    Returns a clean list of evaluation sequences that are 100% free of substring overlap.
    """
    eval_list = [str(x) for x in evaluation_sequences if pd.notna(x)]
    train_list = [str(x) for x in training_sequences if pd.notna(x)]

    if not eval_list or not train_list:
        return eval_list

    # Build Aho-Corasick automaton using unique uppercase evaluation sequences as keys
    A = ahocorasick.Automaton()
    for seq in set(s.upper() for s in eval_list):
        A.add_word(seq, seq)
    A.make_automaton()

    # Parse training sequences in a single pass
    contaminated = set()
    for train_seq in train_list:
        train_seq_up = train_seq.upper()
        for end_idx, eval_seq_up in A.iter(train_seq_up):
            contaminated.add(eval_seq_up)

    # Return a clean list of evaluation sequences preserving original order/casing
    return [seq for seq in eval_list if seq.upper() not in contaminated]


def compute_contamination(eval_df: pd.DataFrame, training_path: Path) -> dict:
    """
    Compute exact + substring overlap between eval set and training set.
    Returns dict with overlap_count, total_eval, overlap_rate, gate_pass.
    """
    if not training_path.exists():
        return {
            "overlap_count": "N/A",
            "total_eval": len(eval_df),
            "overlap_rate": "N/A",
            "gate_pass": True,
            "note": "Training data not found; contamination check skipped.",
        }

    train_df = pd.read_csv(training_path)
    if "peptide" not in train_df.columns:
        return {
            "overlap_count": "N/A",
            "total_eval": len(eval_df),
            "overlap_rate": "N/A",
            "gate_pass": True,
            "note": "Training data has no 'peptide' column; contamination check skipped.",
        }

    train_peptides = list(train_df["peptide"].dropna())
    eval_peptides = list(eval_df["peptide"].dropna())
    eval_peptides_up = [p.upper() for p in eval_peptides]

    # Build Aho-Corasick automaton using unique evaluation sequences
    A = ahocorasick.Automaton()
    for seq in set(eval_peptides_up):
        A.add_word(seq, seq)
    A.make_automaton()

    # Parse training sequences in a single pass to find all overlap
    matched_eval = set()
    for train_seq in train_peptides:
        train_seq_up = train_seq.upper()
        for end_idx, eval_seq_up in A.iter(train_seq_up):
            matched_eval.add(eval_seq_up)

    train_peptides_set = set(t.upper() for t in train_peptides)
    exact_overlap = {p for p in matched_eval if p in train_peptides_set}
    substring_overlap = matched_eval - exact_overlap

    exact_count = sum(1 for p in eval_peptides_up if p in exact_overlap)
    substring_count = sum(1 for p in eval_peptides_up if p in substring_overlap)

    total_overlap = exact_count + substring_count
    total_eval = len(eval_peptides)
    rate = total_overlap / total_eval if total_eval > 0 else 0.0

    return {
        "overlap_count": total_overlap,
        "exact_count": exact_count,
        "substring_count": substring_count,
        "total_eval": total_eval,
        "overlap_rate": round(rate, 4),
        "gate_pass": rate <= CONTAMINATION_CAP,
        "cap_threshold": CONTAMINATION_CAP,
    }


def compute_metrics_for_tool(
    df: pd.DataFrame, score_col: str, label_col: str = "label"
) -> dict | None:
    """
    Compute all 10 SESTRAV metrics plus scikit-learn metrics (including Brier Score) for a given score column.
    Returns None if the column doesn't exist or has no valid scores.
    """
    if score_col not in df.columns:
        return None

    valid_mask = df[score_col].notna() & df[label_col].notna()
    df_valid = df[valid_mask]

    if len(df_valid) == 0:
        return None

    y_true = df_valid[label_col].values
    y_scores = df_valid[score_col].values

    if len(np.unique(y_true)) < 2:
        return None

    # Compute baseline SESTRAV metrics
    metrics = evaluate(y_true, y_scores)

    # Calculate additional metrics using scikit-learn explicitly
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    metrics["auc_roc"] = float(roc_auc_score(y_true, y_scores))
    metrics["auc_pr"] = float(average_precision_score(y_true, y_scores))
    metrics["brier_score"] = float(brier_score_loss(y_true, y_scores))

    metrics["n_samples"] = int(len(df_valid))
    metrics["n_positive"] = int(y_true.sum())
    metrics["n_negative"] = int(len(y_true) - y_true.sum())
    return metrics


def compute_length_stratified(df: pd.DataFrame, score_col: str, label_col: str = "label") -> dict:
    """
    Compute AUC-PR for 9-mer vs non-9-mer subsets.
    Returns dict with keys '9mer' and 'non_9mer'.
    """
    if score_col not in df.columns:
        return {}

    df = df[df[score_col].notna() & df[label_col].notna()].copy()
    df["pep_len"] = df["peptide"].str.len()

    results = {}
    for label, mask in [("9mer", df["pep_len"] == 9), ("non_9mer", df["pep_len"] != 9)]:
        subset = df[mask]
        if len(subset) == 0 or len(np.unique(subset[label_col])) < 2:
            results[label] = {"n": int(len(subset)), "auc_pr": float("nan")}
        else:
            from sklearn.metrics import average_precision_score

            auc_pr = average_precision_score(subset[label_col].values, subset[score_col].values)
            results[label] = {"n": int(len(subset)), "auc_pr": round(float(auc_pr), 4)}

    return results


def build_clean_subset(df: pd.DataFrame, training_path: Path) -> pd.DataFrame:
    """
    Return a copy of df with training-overlapping peptides removed.
    Used for the 'clean holdout' metrics (contamination-excluded).
    """
    if not training_path.exists():
        return df

    train_df = pd.read_csv(training_path)
    if "peptide" not in train_df.columns:
        return df

    train_peptides = list(train_df["peptide"].dropna())
    df_copy = df.copy()
    eval_peptides = list(df_copy["peptide"].dropna())

    clean_eval = set(filter_contaminated_peptides(train_peptides, eval_peptides))
    clean_eval_up = {p.upper() for p in clean_eval}

    mask = df_copy["peptide"].apply(lambda p: str(p).upper() in clean_eval_up)
    return df_copy[mask].copy()


# ── Main ─────────────────────────────────────────────────────────────────────


def run_benchmark(tier: str, run_id: str, skip_freeze_check: bool = False):
    print(f"\n{'=' * 60}")
    print("  SESTRAV Benchmark Runner")
    print(f"  Tier: {tier}  |  Run ID: {run_id}")
    print(f"  {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"{'=' * 60}\n")

    # ── 1. Freeze check ──────────────────────────────────────────────────────
    if not skip_freeze_check:
        freeze = load_freeze_status()
        print(
            f"[OK] Freeze status valid. Dataset version: {freeze.get('dataset_version', 'unknown')}"
        )
    else:
        print("[WARN] Freeze check skipped (--skip-freeze-check flag).")
        freeze = {}

    # ── 2. Load data ──────────────────────────────────────────────────────────
    df = load_merged_scores(tier)
    print(f"[OK] Loaded merged scores: {len(df)} rows (Tier {tier} filter applied)")

    # ── 3. Create output directory ────────────────────────────────────────────
    # -- 3. Create output directory --------------------------------------------
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Output directory: {out_dir}")

    # -- 4. Contamination gate -------------------------------------------------
    print("\n-- Contamination Gate -----------------------------------------")
    contamination = compute_contamination(df, TRAINING_DATA_PATH)
    print(f"    Eval set:        {contamination['total_eval']} peptides")
    if contamination["overlap_rate"] != "N/A":
        print(
            f"    Training overlap: {contamination['overlap_count']} peptides "
            f"({contamination['overlap_rate'] * 100:.1f}%)"
        )
        if contamination["gate_pass"]:
            print(f"    Gate: PASS (<= {CONTAMINATION_CAP * 100:.0f}% cap)")
        else:
            print(
                f"    Gate: FAIL (> {CONTAMINATION_CAP * 100:.0f}% cap - "
                "results are valid but flagged)"
            )
    else:
        print(f"    {contamination.get('note', '')}")

    # -- 5. Compute metrics: all tools -----------------------------------------
    print("\n-- Metric Computation -----------------------------------------")
    all_tools = {**TOOL_SCORE_COLUMNS, **OPTIONAL_TOOL_COLUMNS}
    metrics_rows = []

    for tool_name, score_col in all_tools.items():
        metrics = compute_metrics_for_tool(df, score_col)
        if metrics is None:
            print(
                f"    {tool_name:10s}: [SKIP] column '{score_col}' not found or insufficient data"
            )
            continue

        row = {"tool": tool_name, "score_column": score_col, **metrics}
        metrics_rows.append(row)
        print(
            f"    {tool_name:10s}: AUC-PR={metrics['auc_pr']:.4f}  "
            f"AUC-ROC={metrics['auc_roc']:.4f}  "
            f"ISSR@10={metrics['issr_10']:.4f}  "
            f"ISSR@25={metrics['issr_25']:.4f}  "
            f"N={metrics['n_samples']}"
        )

    # -- 6. Clean subset metrics (contamination-excluded) ----------------------
    df_clean = build_clean_subset(df, TRAINING_DATA_PATH)
    print(f"\n-- Clean Holdout Metrics (N={len(df_clean)}) -------------------")

    clean_metrics_rows = []
    for tool_name, score_col in all_tools.items():
        metrics = compute_metrics_for_tool(df_clean, score_col)
        if metrics is None:
            continue
        row = {"tool": tool_name, "score_column": score_col, "subset": "clean_holdout", **metrics}
        clean_metrics_rows.append(row)
        print(f"    {tool_name:10s}: AUC-PR={metrics['auc_pr']:.4f}  N={metrics['n_samples']}")

    # -- 7. Length-stratified metrics ------------------------------------------
    print("\n-- Length-Stratified Metrics ----------------------------------")
    length_rows = []
    for tool_name, score_col in all_tools.items():
        if score_col not in df.columns:
            continue
        strat = compute_length_stratified(df, score_col)
        for length_group, values in strat.items():
            length_rows.append(
                {
                    "tool": tool_name,
                    "length_group": length_group,
                    **values,
                }
            )
            auc_str = f"{values['auc_pr']:.4f}" if values.get("auc_pr") else "N/A"
            print(f"    {tool_name:10s} {length_group:9s}: AUC-PR={auc_str}  N={values['n']}")

    # -- 8. Write outputs ------------------------------------------------------
    print("\n-- Writing Outputs --------------------------------------------")

    # merged_scores.csv
    merged_out = out_dir / "merged_scores.csv"
    df.to_csv(merged_out, index=False)
    print(f"    [WRITE] {merged_out.name}")

    # metrics_summary.csv
    if metrics_rows or clean_metrics_rows:
        all_rows = [{"subset": "all", **r} for r in metrics_rows] + clean_metrics_rows
        metrics_df = pd.DataFrame(all_rows)
        metrics_out = out_dir / "metrics_summary.csv"
        metrics_df.to_csv(metrics_out, index=False)
        print(f"    [WRITE] {metrics_out.name}")
    else:
        print("    [SKIP] No metrics computed - metrics_summary.csv not written.")

    # length_stratified.csv
    if length_rows:
        length_df = pd.DataFrame(length_rows)
        length_out = out_dir / "length_stratified.csv"
        length_df.to_csv(length_out, index=False)
        print(f"    [WRITE] {length_out.name}")

    # run_manifest.json
    manifest = {
        "run_id": run_id,
        "tier": tier,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit_hash": get_git_commit_hash(),
        "dataset_file_name_used": MERGED_SCORES_PATH.name,
        "model_checkpoint_name": get_model_checkpoint_name(),
        "freeze_status": freeze,
        "contamination": contamination,
        "n_total": len(df),
        "n_clean": len(df_clean),
        "tools_evaluated": [r["tool"] for r in metrics_rows],
        "output_dir": out_dir.relative_to(PROJECT_ROOT).as_posix(),
        "source_merged_scores": MERGED_SCORES_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "source_training_data": TRAINING_DATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
    }
    manifest_out = out_dir / "run_manifest.json"
    with open(manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"    [WRITE] {manifest_out.name}")

    print(f"\n[DONE] Run '{run_id}' complete. Results in: {out_dir}\n")
    return manifest


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="SESTRAV Benchmark Runner - reproducible external validation protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tier",
        choices=["A", "B"],
        default="A",
        help="Benchmark tier to run (A = 720-peptide intersection set, B = 4000-peptide full set)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=datetime.datetime.now(datetime.timezone.utc).strftime("run_%Y%m%d_%H%M%S"),
        help="Unique run identifier used as output subdirectory name",
    )
    parser.add_argument(
        "--skip-freeze-check",
        action="store_true",
        help="Skip reading freeze_status.json (use for development/testing only)",
    )
    args = parser.parse_args()

    run_benchmark(tier=args.tier, run_id=args.run_id, skip_freeze_check=args.skip_freeze_check)


if __name__ == "__main__":
    main()
