"""
Cross-virus transfer replication (Workstream A1, Phase 2).

Trains virus-specific RF models and evaluates on held-out virus peptides using
OOF-style scoring, producing results/external_validation_cross_virus.csv.

Usage:
    python -m src.external_validation_cross_virus --data immunogenicity_dataset.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from src.evaluate_metrics import evaluate
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES
from src.train_classifier import prepare_features_30


def _oof_scores(X: np.ndarray, y: np.ndarray, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    if hasattr(X, "values"):
        X = X.values
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in skf.split(X, y):
        clf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
        clf.fit(X[train_idx], y[train_idx])
        oof[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return oof


def run_cross_virus(
    data_path: str,
    binding_matrix_path: str,
    output_path: str = "results/external_validation_cross_virus.csv",
    n_folds: int = 5,
) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = df[~df["peptide"].isin(GOLD_STANDARD_EPITOPES)].copy()

    X_all = prepare_features_30(df, binding_matrix_path)
    y_all = df["label"].values
    viruses = df["virus"].values

    rows = []

    # Pooled baseline
    oof_all = _oof_scores(X_all, y_all, n_folds=n_folds)
    rows.append(
        {"train": "All", "test": "All", **evaluate(y_all, oof_all), "n_test": len(y_all)}
    )

    for train_virus in sorted(df["virus"].unique()):
        for test_virus in sorted(df["virus"].unique()):
            if train_virus == test_virus:
                continue

            train_mask = viruses == train_virus
            test_mask = viruses == test_virus

            X_train = X_all[train_mask].values if hasattr(X_all, "values") else X_all[train_mask]
            y_train = y_all[train_mask]
            X_test = X_all[test_mask].values if hasattr(X_all, "values") else X_all[test_mask]
            y_test = y_all[test_mask]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            clf = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            )
            clf.fit(X_train, y_train)
            scores = clf.predict_proba(X_test)[:, 1]
            m = evaluate(y_test, scores)
            rows.append(
                {
                    "train": train_virus,
                    "test": test_virus,
                    **m,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                }
            )

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[cross-virus] Wrote {output_path} ({len(out)} rows)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-virus transfer table (A1)")
    parser.add_argument("--data", default="immunogenicity_dataset.csv")
    parser.add_argument(
        "--binding-matrix", default="models/peptide_binding_matrix.csv"
    )
    parser.add_argument(
        "--output", default="results/external_validation_cross_virus.csv"
    )
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    run_cross_virus(
        args.data,
        args.binding_matrix,
        output_path=args.output,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()
