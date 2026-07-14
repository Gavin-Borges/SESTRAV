"""Fit a global isotonic calibrator for the production RF mode-31 model.

This script reproduces the deployed calibration layer from out-of-fold (OOF)
predictions. It fits a single global ``sklearn.isotonic.IsotonicRegression``
that maps a raw RandomForest score in [0, 1] to a calibrated probability, saves
the calibrator as a joblib artifact next to the production model, and writes the
matching SHA256 checksum sidecar manifest that ``load_verified_joblib`` requires
under ``required_checksum=True``.

It also reports honest evaluation numbers computed from the actual OOF data:
global Expected Calibration Error (ECE, 10 uniform bins) and Brier score before
and after calibration, plus per-virus ECE for the nine target viruses (decoy
only viruses are excluded from the per-virus breakdown).

Reproduce with:

    python scripts/fit_calibrator.py

The isotonic method exposes ``.predict(x)`` operating directly on the raw score
in [0, 1]. This differs from the Platt path, which calls ``.predict_proba`` on
logits. The Stage 4 apply hook dispatches on the calibrator type.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.artifact_integrity import default_manifest_path_for, update_checksum_manifest

# Deterministic seed for any stochastic dependency (isotonic itself is
# deterministic, but we set this so the whole run is reproducible).
RANDOM_SEED = 42

# The nine target viruses with genuine positive labels. Decoy only viruses
# (e.g. Orthopoxvirus vaccinia) are excluded from the per-virus breakdown.
TARGET_VIRUSES = (
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

DEFAULT_OOF_PATH = Path("models/v5/rf_oof_predictions_mode31.csv")
DEFAULT_OUTPUT_PATH = Path("models/isotonic_calibrator.joblib")


def expected_calibration_error(
    labels: np.ndarray, scores: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error with uniform-width bins in [0, 1].

    ECE is the sample-weighted mean absolute gap between mean predicted
    probability and observed positive fraction across ``n_bins`` bins.
    """
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(scores)
    if total == 0:
        return float("nan")
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Include the right edge only in the last bin so 1.0 lands somewhere.
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_conf = float(scores[mask].mean())
        bin_acc = float(labels[mask].mean())
        ece += (count / total) * abs(bin_conf - bin_acc)
    return float(ece)


def brier_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mean squared error between predicted probability and the 0/1 label."""
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return float("nan")
    return float(np.mean((scores - labels) ** 2))


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    """Fit a global isotonic calibrator mapping raw score -> probability."""
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(scores, labels)
    return calibrator


def _load_rf_oof(oof_path: Path) -> pd.DataFrame:
    """Load OOF predictions, keeping only RandomForest rows with valid scores."""
    df = pd.read_csv(oof_path)
    df = df[df["method"] == "RandomForest"].copy()
    df = df.dropna(subset=["score", "label"])
    df["score"] = df["score"].astype(np.float64).clip(0.0, 1.0)
    df["label"] = df["label"].astype(int)
    return df


def main() -> None:
    """Fit the calibrator, save it with a checksum sidecar, and report metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oof",
        type=Path,
        default=DEFAULT_OOF_PATH,
        help="Path to the RF OOF predictions CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the isotonic calibrator joblib.",
    )
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    df = _load_rf_oof(args.oof)
    scores = df["score"].to_numpy(dtype=np.float64)
    labels = df["label"].to_numpy(dtype=np.float64)

    print(f"[fit] Loaded {len(df)} RandomForest OOF rows from {args.oof}")

    calibrator = fit_isotonic(scores, labels)
    calibrated = np.asarray(calibrator.predict(scores), dtype=np.float64)

    # Global metrics on the full OOF set (decoys included, matching the
    # assessment's global figure).
    ece_before = expected_calibration_error(labels, scores)
    ece_after = expected_calibration_error(labels, calibrated)
    brier_before = brier_score(labels, scores)
    brier_after = brier_score(labels, calibrated)

    print("\n[global] Metrics on full OOF set (n=%d)" % len(df))
    print(f"  ECE   before: {ece_before:.5f}  after: {ece_after:.5f}")
    print(f"  Brier before: {brier_before:.5f}  after: {brier_after:.5f}")

    print("\n[per-virus] ECE for the 9 target viruses (10-bin uniform)")
    print(f"  {'virus':<12} {'n':>6} {'ece_raw':>10} {'ece_cal':>10} {'delta':>10}")
    for virus in TARGET_VIRUSES:
        sub = df[df["virus"] == virus]
        if sub.empty:
            continue
        v_scores = sub["score"].to_numpy(dtype=np.float64)
        v_labels = sub["label"].to_numpy(dtype=np.float64)
        v_cal = np.asarray(calibrator.predict(v_scores), dtype=np.float64)
        e_raw = expected_calibration_error(v_labels, v_scores)
        e_cal = expected_calibration_error(v_labels, v_cal)
        print(
            f"  {virus:<12} {len(sub):>6} {e_raw:>10.5f} "
            f"{e_cal:>10.5f} {e_cal - e_raw:>+10.5f}"
        )

    # Save the calibrator and write the checksum sidecar manifest.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from joblib import dump as joblib_dump

    joblib_dump(calibrator, args.output)
    manifest_path = update_checksum_manifest(
        default_manifest_path_for(args.output), [args.output]
    )

    print(f"\n[save] Calibrator written to {args.output}")
    print(f"[save] Checksum sidecar updated at {manifest_path}")


if __name__ == "__main__":
    main()
