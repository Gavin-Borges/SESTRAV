"""
Leave-one-virus-out (LOO) cross-virus transfer benchmark - v4 dataset, mode-31 RF.

For each held-out virus:
  - Train RF (mode-31, n_estimators=200, class_weight='balanced') on all other
    viral rows PLUS the 5,000 self-proteome hard decoys.
  - Evaluate on the held-out virus rows only.

Viruses with <10 positives or <10 negatives are skipped (undefined AUC-PR).
Gold-standard epitopes are excluded from training but retained in test sets.

Output:
  results/loo_cross_virus_v4.json   - machine-readable per-virus metrics + provenance
  results/loo_cross_virus_v4.csv    - flat CSV for quick inspection

Usage:
  python scripts/run_loo_cross_virus_v4.py
  python scripts/run_loo_cross_virus_v4.py --min-pos 20 --min-neg 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Allow running from repo root without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate_metrics import evaluate
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES
from src.train_classifier import prepare_features_31

DATASET_PATH = "data/immunogenicity_dataset_v4.csv"
BINDING_MATRIX_PATH = "models/peptide_binding_matrix_v4.csv"
OUTPUT_JSON = "results/loo_cross_virus_v4.json"
OUTPUT_CSV = "results/loo_cross_virus_v4.csv"

RF_PARAMS = dict(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)


def _fit_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    return clf


def run_loo(
    dataset_path: str = DATASET_PATH,
    binding_matrix_path: str = BINDING_MATRIX_PATH,
    output_json: str = OUTPUT_JSON,
    output_csv: str = OUTPUT_CSV,
    min_pos: int = 10,
    min_neg: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    print(f"[loo] Loaded {len(df)} rows from {dataset_path}")

    viral = df[df["source_type"] == "Virus"].copy()
    decoys = df[df["source_type"] == "Self"].copy()
    print(f"[loo] Viral rows: {len(viral)} | Self-decoy rows: {len(decoys)}")

    # Precompute full feature matrix - build once, slice per fold
    full = pd.concat([viral, decoys], ignore_index=True)
    print(f"[loo] Computing mode-31 features for {len(full)} rows …")
    t0 = time.time()
    X_full = prepare_features_31(full, binding_matrix_path)
    y_full = full["label"].values
    virus_full = full["virus"].fillna("Self").values
    source_type_full = full["source_type"].values
    print(f"[loo] Features ready in {time.time()-t0:.1f}s  shape={X_full.shape}")

    # Index gold-standard peptides for exclusion from training
    gs_mask = full["peptide"].isin(GOLD_STANDARD_EPITOPES).values

    # Enumerate eligible viruses
    virus_counts = (
        viral.groupby("virus")["label"]
        .agg(n_pos="sum", n_neg=lambda x: (x == 0).sum())
        .reset_index()
    )
    eligible = virus_counts[
        (virus_counts["n_pos"] >= min_pos) & (virus_counts["n_neg"] >= min_neg)
    ]["virus"].tolist()
    print(f"\n[loo] Eligible viruses ({len(eligible)}): {eligible}")

    rows = []
    for test_virus in sorted(eligible):
        is_test = (virus_full == test_virus)
        is_decoy = (source_type_full == "Self")
        is_train_viral = (source_type_full == "Virus") & (~is_test) & (~gs_mask)
        is_train = is_train_viral | is_decoy

        X_train = X_full.values[is_train]
        y_train = y_full[is_train]
        X_test = X_full.values[is_test]
        y_test = y_full[is_test]

        n_train_viral = int(is_train_viral.sum())
        n_train_decoy = int(is_decoy.sum())
        n_train_pos = int(y_train.sum())
        n_train_neg = int((y_train == 0).sum())
        n_test_pos = int(y_test.sum())
        n_test_neg = int((y_test == 0).sum())

        if len(np.unique(y_test)) < 2:
            print(f"  [{test_virus}] SKIP - test set has only one class")
            continue

        print(
            f"  [{test_virus}] train={len(y_train)} (viral={n_train_viral} + decoy={n_train_decoy}) "
            f"| test={len(y_test)} (pos={n_test_pos}, neg={n_test_neg}) …",
            end="",
            flush=True,
        )
        t1 = time.time()
        clf = _fit_rf(X_train, y_train)
        scores = clf.predict_proba(X_test)[:, 1]
        m = evaluate(y_test, scores)
        elapsed = time.time() - t1
        print(f"  AUC-PR={m['auc_pr']:.4f}  AUC-ROC={m['auc_roc']:.4f}  ({elapsed:.1f}s)")

        rows.append(
            {
                "test_virus": test_virus,
                "auc_pr": round(m["auc_pr"], 4),
                "auc_roc": round(m["auc_roc"], 4),
                "issr_10": round(m.get("issr_10", float("nan")), 4),
                "issr_25": round(m.get("issr_25", float("nan")), 4),
                "n_train_viral": n_train_viral,
                "n_train_decoy": n_train_decoy,
                "n_train_pos": n_train_pos,
                "n_train_neg": n_train_neg,
                "n_test_total": len(y_test),
                "n_test_pos": n_test_pos,
                "n_test_neg": n_test_neg,
                "pos_rate_test": round(n_test_pos / len(y_test), 3),
            }
        )

    result_df = pd.DataFrame(rows).sort_values("auc_pr", ascending=False)

    os.makedirs("results", exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    provenance = {
        "benchmark": "loo_cross_virus_v4",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_path,
        "binding_matrix": binding_matrix_path,
        "feature_mode": 31,
        "rf_params": RF_PARAMS,
        "min_pos_threshold": min_pos,
        "min_neg_threshold": min_neg,
        "train_includes_hard_decoys": True,
        "gold_standard_excluded_from_train": True,
        "n_gold_standard": int(gs_mask.sum()),
        "n_eligible_viruses": len(eligible),
        "results": rows,
    }
    with open(output_json, "w") as fh:
        json.dump(provenance, fh, indent=2)

    print(f"\n[loo] Wrote {output_csv} ({len(result_df)} rows)")
    print(f"[loo] Wrote {output_json}")
    print("\nSummary table (sorted by AUC-PR):")
    print(
        result_df[["test_virus", "auc_pr", "auc_roc", "issr_10", "n_test_total", "pos_rate_test"]]
        .to_string(index=False)
    )
    return result_df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DATASET_PATH)
    p.add_argument("--binding-matrix", default=BINDING_MATRIX_PATH)
    p.add_argument("--output-json", default=OUTPUT_JSON)
    p.add_argument("--output-csv", default=OUTPUT_CSV)
    p.add_argument("--min-pos", type=int, default=10)
    p.add_argument("--min-neg", type=int, default=10)
    args = p.parse_args()

    run_loo(
        dataset_path=args.dataset,
        binding_matrix_path=args.binding_matrix,
        output_json=args.output_json,
        output_csv=args.output_csv,
        min_pos=args.min_pos,
        min_neg=args.min_neg,
    )


if __name__ == "__main__":
    main()
