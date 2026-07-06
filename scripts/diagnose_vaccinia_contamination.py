"""
diagnose_vaccinia_contamination.py

Diagnostic: retrain RF mode-31 with Orthopoxvirus vaccinia rows excluded
and compare per-virus AUC-ROC to the v5 production baseline.

Hypothesis: vaccinia (21,494 rows, 100% negative, 78% of active training
data) overwhelms RF with a negative signal, causing the HPV AUC-ROC
inversion (baseline = 0.318). class_weight='balanced' adjusts for global
class imbalance but not per-virus volume dominance.

Output: comparison table printed to stdout; OOF written to _local/diag/.
No production files are modified.

Usage:
    conda activate sestrav
    python scripts/diagnose_vaccinia_contamination.py
    python scripts/diagnose_vaccinia_contamination.py --n-bootstrap 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.train_classifier import (  # noqa: E402
    _filter_quarantined,
    prepare_features_31,
    _cross_validate,
)
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES  # noqa: E402
from evaluate_per_virus import (  # noqa: E402
    evaluate_all_viruses,
    check_exit_criterion,
    format_table,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VACCINIA_NAME = "Orthopoxvirus vaccinia"
DATA_PATH = REPO_ROOT / "data" / "immunogenicity_dataset_v5.csv"
BINDING_MATRIX = REPO_ROOT / "models" / "peptide_binding_matrix_v5.csv"
BASELINE_OOF = REPO_ROOT / "models" / "v5" / "rf_oof_predictions_mode31.csv"
DIAG_OUT_DIR = REPO_ROOT / "_local" / "diag"

N_FOLDS = 5
RANDOM_STATE = 42

RF_KWARGS = dict(
    n_estimators=200,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_and_filter(exclude_vaccinia: bool) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = _filter_quarantined(df)
    if exclude_vaccinia:
        n_before = len(df)
        df = df[df["virus"] != VACCINIA_NAME].copy().reset_index(drop=True)
        dropped = n_before - len(df)
        print(f"  Vaccinia exclusion: {n_before} -> {len(df)} rows (dropped {dropped})")
    return df


def run_cv(df: pd.DataFrame, label: str, n_bootstrap: int) -> pd.DataFrame:
    """5-fold RF mode-31 CV; returns OOF predictions DataFrame."""
    gs_mask = df["peptide"].isin(GOLD_STANDARD_EPITOPES)
    train_pool = df[~gs_mask].copy().reset_index(drop=True)
    n_pos = int(train_pool["label"].sum())
    n_neg = len(train_pool) - n_pos
    print(
        f"  [{label}] pool={len(train_pool)} rows  pos={n_pos}  neg={n_neg}  "
        f"({train_pool['label'].mean():.2%} positive)"
    )

    X = prepare_features_31(train_pool, str(BINDING_MATRIX))
    y = train_pool["label"].values

    meta_cols = [
        c
        for c in ["peptide", "virus", "strain", "protein", "negative_origin", "hla_allele"]
        if c in train_pool.columns
    ]
    metadata = train_pool[meta_cols].copy().reset_index(drop=True)

    _, _, _, oof_df = _cross_validate(
        X,
        y,
        metadata,
        RandomForestClassifier,
        RF_KWARGS,
        n_splits=N_FOLDS,
        random_state=RANDOM_STATE,
        subgroup_columns=[c for c in ["virus", "strain"] if c in metadata.columns],
    )
    return oof_df


def _fmt_val(val: object) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return "  N/A  "
    return f"{val:7.3f}"


def print_delta_table(
    baseline: dict[str, dict],
    diagnostic: dict[str, dict],
    diag_label: str,
) -> None:
    print(f"\n=== DELTA TABLE: {diag_label} vs baseline ===")
    header = f"{'Virus':<32}  {'Baseline':>8}  {'Diagnostic':>10}  {'Delta':>8}  {'Note'}"
    print(header)
    print("-" * len(header))

    all_viruses = sorted(set(list(baseline.keys()) + list(diagnostic.keys())))
    critical = {"HPV", "EBV", "HCV", "HIV-1"}

    for virus in all_viruses:
        b = baseline.get(virus, {}).get("auc_roc", float("nan"))
        d = diagnostic.get(virus, {}).get("auc_roc", float("nan"))
        delta = d - b if (b == b and d == d) else float("nan")
        note = ""
        if virus in critical:
            note = "<-- TARGET"
        if virus == "HPV":
            if d == d and d > 0.5:
                note = "<-- HYPOTHESIS CONFIRMED (flipped above 0.5)"
            elif d == d and d < 0.5:
                note = "<-- still inverted"
        delta_str = f"{delta:+8.3f}" if delta == delta else "     N/A"
        print(f"{virus:<32}  {_fmt_val(b)}  {_fmt_val(d)}  {delta_str}  {note}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=500,
        help="Bootstrap resamples for per-virus CIs (default: 500)",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        format="%(levelname)s %(name)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.WARNING,
    )
    np.random.seed(RANDOM_STATE)
    DIAG_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- BASELINE (already computed, just re-evaluate) ----
    print("\n" + "=" * 70)
    print("BASELINE: v5 production OOF (models/v5/rf_oof_predictions_mode31.csv)")
    print("=" * 70)
    if not BASELINE_OOF.exists():
        print(f"ERROR: baseline OOF not found at {BASELINE_OOF}")
        return 1
    baseline_oof = pd.read_csv(BASELINE_OOF)
    baseline_results = evaluate_all_viruses(
        baseline_oof, score_col="score", n_bootstrap=args.n_bootstrap
    )
    print(format_table(baseline_results))
    passed_b, msg_b = check_exit_criterion(baseline_results)
    print(f"\nAmendment 6: {'PASS' if passed_b else 'FAIL'} -- {msg_b}")

    # ---- DIAGNOSTIC: exclude vaccinia ----
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: RF mode-31 retrain -- Orthopoxvirus vaccinia EXCLUDED")
    print("=" * 70)
    df_no_vacc = load_and_filter(exclude_vaccinia=True)
    oof_no_vacc = run_cv(df_no_vacc, "no-vaccinia", args.n_bootstrap)
    out_path = DIAG_OUT_DIR / "novaccinia_oof_mode31.csv"
    oof_no_vacc.to_csv(out_path, index=False)
    print(f"  OOF written -> {out_path}")

    results_no_vacc = evaluate_all_viruses(
        oof_no_vacc, score_col="score", n_bootstrap=args.n_bootstrap
    )
    print(format_table(results_no_vacc))
    passed_nv, msg_nv = check_exit_criterion(results_no_vacc)
    print(f"\nAmendment 6: {'PASS' if passed_nv else 'FAIL'} -- {msg_nv}")

    # ---- DELTA TABLE ----
    print_delta_table(baseline_results, results_no_vacc, "no-vaccinia")

    # ---- CONCLUSION ----
    hpv_base = baseline_results.get("HPV", {}).get("auc_roc", float("nan"))
    hpv_nv = results_no_vacc.get("HPV", {}).get("auc_roc", float("nan"))
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if hpv_nv == hpv_nv and hpv_nv > 0.5:
        print(f"  HYPOTHESIS CONFIRMED: HPV AUC-ROC {hpv_base:.3f} -> {hpv_nv:.3f}")
        print("  Vaccinia contamination is the root cause of the HPV inversion.")
        print("  Recommended fix: exclude vaccinia from v5 training OR implement")
        print("  per-virus stratified sampling to cap single-virus contribution.")
    elif hpv_nv == hpv_nv and hpv_nv < 0.5:
        print(f"  HYPOTHESIS NOT CONFIRMED: HPV AUC-ROC {hpv_base:.3f} -> {hpv_nv:.3f}")
        print("  HPV inversion persists without vaccinia.")
        print("  Redirect investigation to HPV negative quality / feature confounds.")
    else:
        print("  HPV result: insufficient data after vaccinia exclusion to evaluate.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
