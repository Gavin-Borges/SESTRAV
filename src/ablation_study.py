"""
SESTRAV Ablation Study - Feature Group Contribution Analysis

Evaluates the contribution of different feature groups to immunogenicity
prediction by training the best ANN architecture on each subset.

Feature group definitions (from CMB 523 Project 2):
  physico_20  : 20 TCR-contact physicochemical features only
  binding_10  : 10 per-allele MHC binding features only
  sestrav_21  : physico + peptide_length (legacy training track)
  combined_30 : physico + binding (legacy comparator 30-feature track)
  full_31     : physico + binding + peptide_length

RETRACTED 2026-08-31 (ruling R3). Every one of the ten Project 2 figures below is
an UNBOUND course-export value. Their only cited source is a CMB 523 Colab export
absent from git, from disk, and from the local workspace, so none can ever acquire
provenance here. They are retained rather than repaired or deleted: substituting
models/ann_cv_summary.csv would trade an unbound number for a mis-attributed one
(it measures the legacy 64-32 d0.3 network over the 704 peptides left after 16
gold-standard epitopes are held out of 720, and carries no per-feature-group
breakdown). Do not cite, quote, or compare these numbers.

SCOPE, stated because this retraction is narrower than it reads. The same ten
values, rounded to 3 decimals, are LIVE and unmarked on eight other tracked
surfaces, and this change amends none of them: the identical ablation table in
docs/model_evaluation_summary.md; README.md (0.864 in the headline paragraph and
in the model comparison table, 0.825 as the legacy comparator);
docs/data_registry.md AD-1, a LOCKED row in which "full_31 0.864 vs combined_30
0.825" is the stated rationale for the canonical feature_mode=31;
docs/claims_register.md D8, recording the same pair for the same decision;
docs/feature_glossary.md; and the rf_31feature_integrated and rf_30feature model
cards. The retracted-token sweep cannot see any of them - it is keyed on the
4-decimal literals and the 3-decimal restatements are outside its vocabulary by
construction - so a green sweep here is not evidence that the retraction is
complete.

It also stands against docs/claims_register.md D17, a CLOSED row which states
that "those ablation numbers are therefore not void and are not retracted" and
corroborates them as a real run against a real binding matrix (its binding_10 row
scores AUC-ROC 0.727 on the ten binding features alone, impossible against the
all-zeros matrix D17 concerns). The two are not strictly contradictory: D17
argues the numbers are a genuine measurement, this header argues no artifact in
this repo can re-derive them, and both can hold at once. But a reader will meet
both, and reconciling them touches a LOCKED row and the README headline. That is
an owner decision and it has not been made.

Project 2 results (5-fold CV, best ANN: 256-128-64 ReLU d0.2) - ALL RETRACTED:
  physico_20:  AUC-ROC=0.5766, AUC-PR=0.7719
  binding_10:  AUC-ROC=0.7273, AUC-PR=0.8509
  sestrav_21:  AUC-ROC=0.6215, AUC-PR=0.7844
  combined_30: AUC-ROC=0.6699, AUC-PR=0.8252
  full_31:     AUC-ROC=0.7433, AUC-PR=0.8639
RETRACTED: the "(best overall)" superlative formerly carried on the full_31 line is
WITHDRAWN - it rested on two of these unbound values.

--output-dir is required and has no default. The existing
models/ablation_study_results.csv there is never replaced unless --allow-overwrite
is passed.

Usage:
    python -m src.ablation_study --data immunogenicity_dataset.csv \\
        --binding-matrix models/peptide_binding_matrix.csv \\
        --output-dir models/scratch/ablation
"""

import argparse
import os
import numpy as np
import pandas as pd

from src.artifact_guard import guard_planned_paths
from src.features import PHYSICO_COLUMNS, BINDING_ALLELE_COLUMNS
from src.evaluate_metrics import summarize_fold_metrics
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES
from src.model import (
    set_seeds,
    get_device,
    compute_pos_weight,
    run_cv,
    SEED,
    N_FOLDS,
)
from src.train_classifier import _filter_quarantined, prepare_features_30


# Feature group definitions
FEATURE_GROUPS = {
    "physico_20": PHYSICO_COLUMNS,
    "binding_10": BINDING_ALLELE_COLUMNS,
    "sestrav_21": PHYSICO_COLUMNS + ["peptide_length"],
    "combined_30": PHYSICO_COLUMNS + BINDING_ALLELE_COLUMNS,
    "full_31": PHYSICO_COLUMNS + BINDING_ALLELE_COLUMNS + ["peptide_length"],
}


def _guard_output_dir(output_dir: str, allow_overwrite: bool) -> None:
    """Refuse to clobber an existing ablation_study_results.csv unless overwrite is explicit."""
    out_path = os.path.join(output_dir, "ablation_study_results.csv")
    guard_planned_paths(
        output_dir,
        [out_path],
        allow_overwrite,
        flag="--output-dir",
        api_hint="",
        remedy="Point --output-dir at a fresh directory (for example models/scratch/<run-name>), ",
        single_path=True,
    )


def run_ablation(
    data_path,
    binding_matrix_path,
    output_dir: str,
    config=None,
    n_folds=N_FOLDS,
    allow_overwrite=False,
):
    """Run ablation study across all feature groups.

    Args:
        data_path:           Path to immunogenicity_dataset.csv.
        binding_matrix_path: Path to peptide_binding_matrix.csv.
        output_dir:          Directory for result CSVs. No default: pass "models"
                              only when you intend to replace the published
                              artifact, otherwise use a scratch directory.
        config:              Architecture config dict.
        n_folds:             CV folds.
        allow_overwrite:     Replace an existing ablation_study_results.csv in
                              output_dir instead of aborting before any work.

    Returns:
        pd.DataFrame of ablation results sorted by AUC-PR.
    """
    _guard_output_dir(output_dir, allow_overwrite)

    if config is None:
        config = {"hidden": [256, 128, 64], "dropout": 0.2, "activation": "relu"}

    set_seeds(SEED)
    device = get_device()
    print(f"Device: {device}")

    # Load data
    df = pd.read_csv(data_path)
    df = _filter_quarantined(df)
    gs_mask = df["peptide"].isin(GOLD_STANDARD_EPITOPES)
    pool = df[~gs_mask].copy()
    y = pool["label"].values
    virus = pool["virus"].values if "virus" in pool.columns else np.zeros(len(pool))
    strat_key = np.array([f"{l}_{v}" for l, v in zip(y, virus)])
    pos_weight = compute_pos_weight(y)

    print(f"Training pool: {len(pool)} records")
    print(f"Class balance: {np.mean(y):.2%} positive")

    X_30 = prepare_features_30(pool, binding_matrix_path)

    # Add peptide_length column
    lengths = pool["peptide"].str.len().values
    X_full = X_30.copy()
    X_full["peptide_length"] = lengths

    # Run ablation
    config_name = "-".join(str(h) for h in config["hidden"])
    print(f"\nArchitecture: {config_name} {config['activation']} d{config['dropout']}")

    results = []
    for group_name, group_cols in FEATURE_GROUPS.items():
        available = [c for c in group_cols if c in X_full.columns]
        if len(available) < len(group_cols):
            missing = set(group_cols) - set(X_full.columns)
            print(f"\n  WARNING: {group_name} missing {len(missing)} columns: {missing}")
            continue

        X_group = X_full[available].values
        n_features = X_group.shape[1]
        print(f"\nAblation: {group_name} ({n_features} features)...")

        # Adjust architecture input dim
        group_config = config.copy()

        fold_metrics = run_cv(
            X_group, y, strat_key, group_config, pos_weight, n_folds=n_folds, device=device
        )
        avg, std = summarize_fold_metrics(fold_metrics)

        print(
            f"  AUC-ROC={avg['auc_roc']:.4f} +/- {std['auc_roc']:.4f}  "
            f"AUC-PR={avg['auc_pr']:.4f} +/- {std['auc_pr']:.4f}"
        )

        results.append(
            {
                "feature_set": group_name,
                "n_features": n_features,
                "auc_roc_mean": avg["auc_roc"],
                "auc_roc_std": std["auc_roc"],
                "auc_pr_mean": avg["auc_pr"],
                "auc_pr_std": std["auc_pr"],
                "issr_10_mean": avg["issr_10"],
                "issr_25_mean": avg["issr_25"],
            }
        )

    df_results = pd.DataFrame(results)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ablation_study_results.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\n{'=' * 60}")
    print("Ablation Study Results:")
    print(f"{'=' * 60}")
    print(df_results.to_string(index=False))
    print(f"\nResults saved to {out_path}")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SESTRAV Ablation Study - Feature group contribution analysis"
    )
    parser.add_argument("--data", required=True, help="Path to immunogenicity_dataset.csv")
    parser.add_argument(
        "--binding-matrix", required=True, help="Path to peptide_binding_matrix.csv"
    )
    parser.add_argument(
        "--architecture", default="256-128-64", help="Hidden layer sizes (default: 256-128-64)"
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for result CSVs. No default: pass models/ only when you "
        "intend to replace the published artifact, otherwise use a scratch "
        "directory",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace an existing ablation_study_results.csv in --output-dir "
        "instead of aborting before any work",
    )
    args = parser.parse_args()

    hidden = [int(x) for x in args.architecture.split("-")]
    config = {"hidden": hidden, "dropout": args.dropout, "activation": args.activation}

    run_ablation(
        args.data,
        args.binding_matrix,
        args.output_dir,
        config=config,
        n_folds=args.cv_folds,
        allow_overwrite=args.allow_overwrite,
    )
