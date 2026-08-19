"""
Cross-virus transfer replication (Workstream A1, Phase 2).

Trains virus-specific RF models and evaluates on held-out virus peptides using
OOF-style scoring, producing results/external_validation_cross_virus.csv.

Usage:
    python -m src.external_validation_cross_virus \\
        --data data/immunogenicity_dataset_v4.csv \\
        --output results/external_validation_cross_virus.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from src.artifact_guard import guard_planned_paths, planned_paths_under
from src.evaluate_metrics import evaluate
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES
from src.ml_utils import pin_serial_scoring
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
            n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        pin_serial_scoring(clf)
        oof[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return oof


def planned_cross_virus_paths(output_path: str) -> list[str]:
    """The single tracked artifact run_cross_virus writes.

    output_path is a literal file path (--output), not a directory, so this
    always resolves to exactly one entry.
    """
    return planned_paths_under(os.path.dirname(output_path) or ".", [os.path.basename(output_path)])


def _guard_output_path(output_path: str, allow_overwrite: bool) -> None:
    """Refuse to clobber the tracked artifact already on disk.

    --output names a file, not a directory, so scope/remedy override the
    default "under '{output_dir}'" / "Point {flag} at a fresh directory"
    clauses, which would be wrong advice here - same shape as
    src/data_bias_audit.py's single-file guards.
    """
    guard_planned_paths(
        os.path.dirname(output_path) or ".",
        planned_cross_virus_paths(output_path),
        allow_overwrite,
        flag="--output",
        api_hint="run_cross_virus(..., allow_overwrite=True)",
        scope="among this run's planned artifacts",
        remedy="Point --output at a fresh path, ",
    )


def run_cross_virus(
    data_path: str,
    binding_matrix_path: str,
    output_path: str,
    n_folds: int = 5,
    allow_overwrite: bool = False,
) -> pd.DataFrame:
    _guard_output_path(output_path, allow_overwrite)
    df = pd.read_csv(data_path)
    df = df[~df["peptide"].isin(GOLD_STANDARD_EPITOPES)].copy()

    X_all = prepare_features_30(df, binding_matrix_path)
    y_all = df["label"].values
    viruses = df["virus"].values

    rows = []

    # Pooled baseline
    oof_all = _oof_scores(X_all, y_all, n_folds=n_folds)
    rows.append({"train": "All", "test": "All", **evaluate(y_all, oof_all), "n_test": len(y_all)})

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
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            pin_serial_scoring(clf)
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
    parser.add_argument("--data", default="data/immunogenicity_dataset_v4.csv")
    parser.add_argument("--binding-matrix", default="models/peptide_binding_matrix_v4.csv")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the cross-virus transfer table. No default: it "
        "refuses to guess a destination.",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace the cross-virus artifact at --output if it already "
        "exists. Without this flag the run aborts before any work if it "
        "would be overwritten.",
    )
    args = parser.parse_args()

    run_cross_virus(
        args.data,
        args.binding_matrix,
        output_path=args.output,
        n_folds=args.n_folds,
        allow_overwrite=args.allow_overwrite,
    )


if __name__ == "__main__":
    main()
