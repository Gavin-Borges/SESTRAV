"""
SESTRAV Calibration Analysis

Computes Brier scores and reliability diagrams for OOF predictions,
comparing v1 and v2 datasets to assess probability calibration quality.

Outputs (all into --output-dir):
  - calibration_reliability_diagram.png
  - calibration_score_distribution.png
  - calibration_metrics.csv (git-tracked)

Usage:
    python -m src.calibration_analysis --output-dir results/scratch/calibration
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.artifact_guard import guard_planned_paths, planned_paths_under


def _load_oof(path, method="RandomForest"):
    """Load OOF predictions for a given method from the standard CSV format."""
    df = pd.read_csv(path)
    if "method" in df.columns:
        df = df[df["method"] == method]
    return df


def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    """Compute Brier score and binned calibration curve."""
    brier = brier_score_loss(y_true, y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    trivial_brier = brier_score_loss(y_true, np.full_like(y_prob, y_true.mean()))
    return {
        "brier_score": brier,
        "trivial_brier": trivial_brier,
        "brier_skill": 1.0 - brier / trivial_brier if trivial_brier > 0 else 0.0,
        "prob_true": prob_true,
        "prob_pred": prob_pred,
    }


def planned_calibration_paths(output_dir: str) -> list[str]:
    """Every path run_calibration_analysis writes into output_dir.

    All three names are literals written directly by run_calibration_analysis.
    This entry point delegates no output to any other module and derives no
    filenames, so there is nothing here of the shape that the `.replace()`- and
    f-string-built names elsewhere in this defect-class line hid.
    """
    names = [
        "calibration_reliability_diagram.png",
        "calibration_score_distribution.png",
        "calibration_metrics.csv",  # git-tracked
    ]
    return planned_paths_under(output_dir, names)


def _guard_output_dir(output_dir: str, allow_overwrite: bool) -> None:
    """Refuse to clobber artifacts already on disk unless overwrite is explicit."""
    guard_planned_paths(
        output_dir,
        planned_calibration_paths(output_dir),
        allow_overwrite,
        flag="--output-dir",
        api_hint="run_calibration_analysis(..., allow_overwrite=True)",
        detail=(
            ": calibration_metrics.csv is the one git-tracked artifact this module writes"
        ),
    )


def run_calibration_analysis(
    v2_oof_path,
    output_dir,
    v1_oof_path=None,
    method="RandomForest",
    allow_overwrite=False,
):
    """Run calibration analysis and produce reliability diagram + Brier scores.

    output_dir has no default and sits ahead of the defaulted parameters
    because Python does not allow a no-default parameter after a defaulted one.
    """
    _guard_output_dir(output_dir, allow_overwrite)
    os.makedirs(output_dir, exist_ok=True)

    v2_df = _load_oof(v2_oof_path, method)
    v2_metrics = compute_calibration_metrics(v2_df["label"].values, v2_df["score"].values)

    results = [
        {
            "dataset": "v2",
            "method": method,
            "n_samples": len(v2_df),
            "positive_rate": v2_df["label"].mean(),
            "brier_score": v2_metrics["brier_score"],
            "trivial_brier": v2_metrics["trivial_brier"],
            "brier_skill_score": v2_metrics["brier_skill"],
        }
    ]

    print("=" * 60)
    print("CALIBRATION ANALYSIS")
    print("=" * 60)
    print(
        f"  v2: Brier={v2_metrics['brier_score']:.4f}, "
        f"Trivial={v2_metrics['trivial_brier']:.4f}, "
        f"Skill={v2_metrics['brier_skill']:.4f}"
    )

    v1_metrics = None
    if v1_oof_path and os.path.isfile(v1_oof_path):
        v1_df = _load_oof(v1_oof_path, method)
        v1_metrics = compute_calibration_metrics(v1_df["label"].values, v1_df["score"].values)
        results.append(
            {
                "dataset": "v1",
                "method": method,
                "n_samples": len(v1_df),
                "positive_rate": v1_df["label"].mean(),
                "brier_score": v1_metrics["brier_score"],
                "trivial_brier": v1_metrics["trivial_brier"],
                "brier_skill_score": v1_metrics["brier_skill"],
            }
        )
        print(
            f"  v1: Brier={v1_metrics['brier_score']:.4f}, "
            f"Trivial={v1_metrics['trivial_brier']:.4f}, "
            f"Skill={v1_metrics['brier_skill']:.4f}"
        )
    print("=" * 60)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.5)
    ax.plot(
        v2_metrics["prob_pred"],
        v2_metrics["prob_true"],
        "s-",
        label=f"v2 (Brier={v2_metrics['brier_score']:.4f})",
        linewidth=2,
    )
    if v1_metrics:
        ax.plot(
            v1_metrics["prob_pred"],
            v1_metrics["prob_true"],
            "o-",
            label=f"v1 (Brier={v1_metrics['brier_score']:.4f})",
            linewidth=2,
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"{method} - Reliability Diagram (OOF)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "calibration_reliability_diagram.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  Reliability diagram saved to {output_dir}/calibration_reliability_diagram.png")

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(v2_df["score"].values, bins=bins, alpha=0.6, label="v2", edgecolor="black")
    if v1_oof_path and os.path.isfile(v1_oof_path):
        v1_df_full = _load_oof(v1_oof_path, method)
        ax.hist(v1_df_full["score"].values, bins=bins, alpha=0.4, label="v1", edgecolor="black")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title(f"{method} - OOF Score Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "calibration_score_distribution.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"  Score distribution saved to {output_dir}/calibration_score_distribution.png")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "calibration_metrics.csv"), index=False)
    print(f"  Metrics CSV saved to {output_dir}/calibration_metrics.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SESTRAV calibration analysis")
    parser.add_argument("--v2-oof", default="models/rf_oof_predictions.csv")
    parser.add_argument("--v1-oof", default="models/v1_backup/rf_oof_predictions.csv")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for calibration outputs. No default: this analysis writes 3 "
        "artifacts into it, one of them the git-tracked calibration_metrics.csv, so it "
        "refuses to guess a destination.",
    )
    parser.add_argument("--method", default="RandomForest")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace calibration artifacts that already exist in --output-dir. "
        "Without this flag the run aborts before any work if any would be overwritten.",
    )
    args = parser.parse_args()
    run_calibration_analysis(
        v2_oof_path=args.v2_oof,
        output_dir=args.output_dir,
        v1_oof_path=args.v1_oof,
        method=args.method,
        allow_overwrite=args.allow_overwrite,
    )
