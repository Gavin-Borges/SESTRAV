"""Fit a Cross Venn-Abers conformal calibrator for the production RF mode-31 model.

This script loads out-of-fold (OOF) predictions from the certified mode-31
RandomForest model, fits a group-disjoint Cross Venn-Abers predictor (CVAP),
evaluates out-of-fold calibration coverage metrics, outputs both the fitted
calibrator artifact (conformal_calibrator.joblib) and the coverage validation
artifact (conformal_coverage_mode31.json), and updates the SHA256 checksum
manifest.

Usage:
    python scripts/fit_conformal_calibrator.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.artifact_integrity import (
    default_manifest_path_for,
    sha256_file,
    update_checksum_manifest,
)
from src.conformal import (
    CrossVennAbers,
    fit_venn_abers,
    oof_intervals,
)

RANDOM_SEED = 42

TARGET_VIRUSES: tuple[str, ...] = (
    "CMV",
    "DENV",
    "EBV",
    "HBV",
    "HCV",
    "HIV-1",
    "HPV",
    "IAV",
    "SARS-CoV-2",
)

DEFAULT_OOF_PATH = Path("models/v5/rf_oof_predictions.csv")
DEFAULT_MODEL_PATH = Path("models/v5/rf_31feature_integrated.joblib")
DEFAULT_OUTPUT_PATH = Path("models/v5/conformal_calibrator.joblib")
DEFAULT_COVERAGE_PATH = Path("models/v5/conformal_coverage_mode31.json")


def brier_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mean squared error between predicted probability and the 0/1 label."""
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if len(s) == 0:
        return float("nan")
    return float(np.mean((s - y) ** 2))


def _load_rf_oof(oof_path: Path) -> pd.DataFrame:
    """Load OOF predictions, keeping only RandomForest rows with valid scores."""
    df = pd.read_csv(oof_path)
    df = df[df["method"] == "RandomForest"].copy()
    df = df.dropna(subset=["score", "label", "peptide"])
    df["score"] = df["score"].astype(np.float64).clip(0.0, 1.0)
    df["label"] = df["label"].astype(int)
    return df


def compute_coverage_report(
    df: pd.DataFrame,
    model: CrossVennAbers,
    lower: np.ndarray,
    upper: np.ndarray,
    model_path: Path,
    oof_path: Path,
) -> dict[str, Any]:
    """Compute empirical calibration, bracket width, and provenance metrics."""
    scores = df["score"].to_numpy(dtype=np.float64)
    labels = df["label"].to_numpy(dtype=np.float64)
    mid = 0.5 * (lower + upper)
    width = upper - lower

    report: dict[str, Any] = {
        "method": "RandomForest",
        "feature_mode": 31,
        "n_samples": int(len(df)),
        "n_groups": int(df["peptide"].nunique()),
        "n_folds": int(model.n_folds),
        "random_state": model.random_state,
        "source_model": {
            "path": str(model_path).replace("\\", "/"),
            "sha256": sha256_file(model_path) if model_path.is_file() else None,
        },
        "source_oof": {
            "path": str(oof_path).replace("\\", "/"),
            "sha256": sha256_file(oof_path) if oof_path.is_file() else None,
        },
        "metrics": {
            "mean_raw_score": float(np.mean(scores)),
            "mean_calibrated_mid": float(np.mean(mid)),
            "mean_label": float(np.mean(labels)),
            "calibration_in_large_gap": float(abs(np.mean(mid) - np.mean(labels))),
            "mean_interval_width": float(np.mean(width)),
            "median_interval_width": float(np.median(width)),
            "q25_interval_width": float(np.percentile(width, 25)),
            "q75_interval_width": float(np.percentile(width, 75)),
            "min_lower": float(np.min(lower)),
            "max_upper": float(np.max(upper)),
            "ordering_violations": int(np.sum(lower > upper)),
            "brier_score_raw": brier_score(labels, scores),
            "brier_score_calibrated_mid": brier_score(labels, mid),
        },
        "validity_scope": (
            "Cross Venn-Abers Predictor (CVAP) produces probability intervals [lower, upper] "
            "via multi-fold geometric mean merging (Vovk et al. 2015). CVAP intervals reflect "
            "multi-fold calibration uncertainty and shrink with local density rather than holding "
            "a nominal coverage level. Peptide groups are kept disjoint across folds to eliminate "
            "group leakage."
        ),
    }

    if "virus" in df.columns:
        per_virus: dict[str, dict[str, float]] = {}
        for virus in TARGET_VIRUSES:
            mask = (df["virus"] == virus).to_numpy()
            if not np.any(mask):
                continue
            v_scores = scores[mask]
            v_labels = labels[mask]
            v_mid = mid[mask]
            v_width = width[mask]
            per_virus[virus] = {
                "n_samples": int(np.sum(mask)),
                "mean_raw_score": float(np.mean(v_scores)),
                "mean_calibrated_mid": float(np.mean(v_mid)),
                "mean_label": float(np.mean(v_labels)),
                "mean_interval_width": float(np.mean(v_width)),
                "median_interval_width": float(np.median(v_width)),
                "brier_raw": brier_score(v_labels, v_scores),
                "brier_mid": brier_score(v_labels, v_mid),
            }
        report["per_virus"] = per_virus

    return report


def main() -> None:
    """Fit conformal predictor, compute coverage, and save artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oof",
        type=Path,
        default=DEFAULT_OOF_PATH,
        help="Path to the RF OOF predictions CSV.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained RF model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the conformal calibrator joblib.",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=DEFAULT_COVERAGE_PATH,
        help="Path to write the conformal coverage JSON artifact.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of group-disjoint folds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random state seed.",
    )
    args = parser.parse_args()

    print(f"[conformal] Loading OOF predictions from {args.oof}...")
    df = _load_rf_oof(args.oof)
    scores = df["score"].to_numpy(dtype=np.float64)
    labels = df["label"].to_numpy(dtype=np.float64)
    groups = df["peptide"].to_numpy()
    print(f"[conformal] Loaded {len(df)} rows across {df['peptide'].nunique()} unique peptides")

    print(f"[conformal] Fitting Cross Venn-Abers predictor (n_folds={args.n_folds}, seed={args.seed})...")
    model = fit_venn_abers(
        scores=scores,
        labels=labels,
        groups=groups,
        n_folds=args.n_folds,
        random_state=args.seed,
    )

    print("[conformal] Computing out-of-fold intervals for validation...")
    lower, upper = oof_intervals(model)

    print(f"[conformal] Generating coverage report for {args.coverage_output}...")
    coverage_report = compute_coverage_report(
        df=df,
        model=model,
        lower=lower,
        upper=upper,
        model_path=args.model,
        oof_path=args.oof,
    )

    # Save artifacts
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[conformal] Saving calibrator artifact to {args.output}...")
    joblib.dump(model, args.output)

    print(f"[conformal] Saving coverage report to {args.coverage_output}...")
    with open(args.coverage_output, "w", encoding="utf-8") as f:
        json.dump(coverage_report, f, indent=2)

    # Update checksum manifest
    manifest_path = default_manifest_path_for(args.output)
    updated_manifest = update_checksum_manifest(
        manifest_path, [args.output, args.coverage_output]
    )
    print(f"[conformal] Updated checksum manifest at {updated_manifest}")

    m = coverage_report["metrics"]
    print("\n[conformal] OOF Metrics Summary:")
    print(f"  Samples               : {coverage_report['n_samples']}")
    print(f"  Peptides (groups)     : {coverage_report['n_groups']}")
    print(f"  Ordering violations   : {m['ordering_violations']}")
    print(f"  Mean interval width   : {m['mean_interval_width']:.6f}")
    print(f"  Median interval width : {m['median_interval_width']:.6f}")
    print(f"  Cal in large gap      : {m['calibration_in_large_gap']:.6f}")
    print(f"  Brier score (raw)     : {m['brier_score_raw']:.6f}")
    print(f"  Brier score (mid)     : {m['brier_score_calibrated_mid']:.6f}")


if __name__ == "__main__":
    main()
