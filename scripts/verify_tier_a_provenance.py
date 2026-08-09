#!/usr/bin/env python3
"""Verify which dataset generation produced the certified Tier A SESTRAV arm.

Background (docs/claims_register.md D16): `results/table3_tier_a_metrics.csv` reports
SESTRAV RF AUC-PR 0.828 / AUC-ROC 0.7255 / ISSR@10 0.8429 over an n=704 field, and
`README.md` presents that as the canonical v5 `mode_31` result. It is not - it is a v3-era
measurement. This script is the evidence for that claim, so the register cites a
reproducible check rather than an assertion.

What it establishes, by direct measurement:
  1. Field membership. All 704 Tier A peptides resolve against the TRACKED v3 corpus
     (`data/immunogenicity_dataset_v3.csv`, 1,004 peptides). Only ~414 exist in v5, and
     ~236 exist in neither v4 nor v5 - so v5 cannot be the source and v3 can.
  2. Score agreement. Mean absolute deviation of a v3 re-run from the stored
     `rf_oof_score` column is roughly 0.084, versus roughly 0.371 for the current v5 OOF.
  3. Metric recovery. A v3 re-run at 500 estimators recovers the certified ISSR@10
     exactly (0.8429), with AUC-ROC within ~0.004 and AUC-PR within ~0.009 - residuals
     consistent with fold-shuffle variance between independent runs, not with a
     different measurement.

The consequence worth carrying: the v5 mode-31 production model has never been evaluated
on the full Tier A field and cannot be, because 290 of its 704 peptides are absent from v5.
Re-running Tier A under v5 produces a different, smaller field - a replacement, not a refresh.

This is a read-only verifier: it prints a comparison and writes nothing. It is deliberately
NOT an artifact generator, so it does not widen the tracked `results/` surface.

Reproduce:  python scripts/verify_tier_a_provenance.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_metrics import evaluate  # noqa: E402
from src.train_classifier import prepare_features_31  # noqa: E402

RANDOM_STATE = 42
N_SPLITS = 5
# The v3-era production model card documents 500 estimators; the current v5 pipeline uses
# 200. Both are reported so the better-fitting configuration is visible rather than asserted.
ESTIMATOR_SETTINGS = (200, 500)

# Certified cells from results/table3_tier_a_metrics.csv (SESTRAV RF row).
CERTIFIED = {"auc_pr": 0.827767, "auc_roc": 0.725505, "issr_10": 0.842857}

V3_PATH = PROJECT_ROOT / "data" / "immunogenicity_dataset_v3.csv"
V4_PATH = PROJECT_ROOT / "data" / "immunogenicity_dataset_v4.csv"
V5_PATH = PROJECT_ROOT / "data" / "immunogenicity_dataset_v5.csv"
BINDING_MATRIX_PATH = PROJECT_ROOT / "models" / "peptide_binding_matrix.csv"
TIER_A_PATH = PROJECT_ROOT / "results" / "external_validation_input.csv"
V5_OOF_PATH = PROJECT_ROOT / "models" / "rf_oof_predictions.csv"


def _peptides(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return set(pd.read_csv(path, low_memory=False)["peptide"])


def _tier_a_field() -> pd.DataFrame:
    field = pd.read_csv(TIER_A_PATH)
    if "tier_a_baseline_complete" in field.columns:
        field = field[field["tier_a_baseline_complete"].astype(bool)]
    return field.reset_index(drop=True)


def report_membership(field: pd.DataFrame) -> None:
    peptides = set(field["peptide"])
    v3, v4, v5 = _peptides(V3_PATH), _peptides(V4_PATH), _peptides(V5_PATH)
    print(f"Tier A certified field: {len(field)} rows, {len(peptides)} unique peptides")
    print(f"  resolvable against v3 : {len(peptides & v3):>4}  (tracked)")
    print(f"  resolvable against v4 : {len(peptides & v4):>4}  (gitignored)")
    print(f"  resolvable against v5 : {len(peptides & v5):>4}")
    print(f"  in NEITHER v4 nor v5  : {len(peptides - v4 - v5):>4}")
    if v3 and peptides <= v3:
        print("  -> every Tier A peptide is present in v3; v3 is the only generation that covers the field.")


def report_v5_disagreement(field: pd.DataFrame) -> None:
    if not V5_OOF_PATH.is_file():
        print("\n[v5 OOF comparison skipped: models/rf_oof_predictions.csv absent]")
        return
    oof = pd.read_csv(V5_OOF_PATH)
    if "method" in oof.columns and (oof["method"] == "RandomForest").any():
        oof = oof[oof["method"] == "RandomForest"]
    current = oof.drop_duplicates(subset=["peptide"], keep="first")[["peptide", "score"]]
    merged = field[["peptide", "rf_oof_score"]].merge(current, on="peptide", how="inner")
    if merged.empty:
        print("\nv5 OOF overlap with the Tier A field: none")
        return
    delta = (merged["rf_oof_score"] - merged["score"]).abs()
    print(f"\nStored column vs CURRENT v5 OOF: matched {len(merged)}/{len(field)} peptides")
    print(f"  identical (<1e-9): {int((delta < 1e-9).sum())}/{len(merged)}   mean abs diff: {delta.mean():.4f}")


def report_v3_reproduction(field: pd.DataFrame) -> None:
    df = pd.read_csv(V3_PATH, low_memory=False)
    X = prepare_features_31(df, str(BINDING_MATRIX_PATH))
    y = df["label"].astype(int).to_numpy()

    for n_estimators in ESTIMATOR_SETTINGS:
        oof = np.full(len(y), np.nan)
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        for train_idx, test_idx in splitter.split(X, y):
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            )
            clf.fit(X.iloc[train_idx], y[train_idx])
            oof[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]

        scored = pd.DataFrame(
            {"peptide": df["peptide"].to_numpy(), "score": oof}
        ).drop_duplicates(subset=["peptide"], keep="first")
        merged = field[["peptide", "label", "rf_oof_score"]].merge(scored, on="peptide", how="left")
        resolved = merged.dropna(subset=["score"])
        if resolved.empty:
            print(f"\nv3 re-run (n_estimators={n_estimators}): no Tier A peptide resolved")
            continue

        metrics = evaluate(resolved["label"].to_numpy(), resolved["score"].to_numpy())
        delta = (resolved["rf_oof_score"] - resolved["score"]).abs()
        print(f"\nv3 re-run, n_estimators={n_estimators}: coverage {len(resolved)}/{len(merged)}")
        for key in ("auc_pr", "auc_roc", "issr_10"):
            print(
                f"  {key:8} {metrics[key]:.4f}   certified {CERTIFIED[key]:.4f}   "
                f"delta {metrics[key] - CERTIFIED[key]:+.4f}"
            )
        print(f"  mean abs diff vs stored column: {delta.mean():.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-reproduction",
        action="store_true",
        help="Report field membership and the v5 disagreement only, without retraining on v3.",
    )
    args = parser.parse_args(argv)

    for required in (V3_PATH, BINDING_MATRIX_PATH, TIER_A_PATH):
        if not required.is_file():
            print(f"Missing required tracked input: {required}", file=sys.stderr)
            return 1

    report_membership(_tier_a_field())
    report_v5_disagreement(_tier_a_field())
    if not args.skip_reproduction:
        report_v3_reproduction(_tier_a_field())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
