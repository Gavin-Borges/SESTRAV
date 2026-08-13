"""
scripts/score_validation_cohorts.py
===================================
Scores the clean validation cohorts (SARS-CoV-2 and Influenza A) using the
canonical pre-trained 31-feature Random Forest model.

Performs:
  1. MHCflurry binding prediction for the 10 target alleles (Stage 2).
  2. Physicochemical TCR feature extraction at positions p4-p8 (Stage 3).
  3. Scoring using models/rf_31feature_integrated.joblib (Stage 4).
  4. Pre-registered metric evaluation (AUC-PR, AUC-ROC, ISSR@10) with 95% bootstrap CI (N=2,000).

Outputs:
  - results/sars2_scored.csv
  - results/influenza_scored.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from mhcflurry import Class1PresentationPredictor
from sklearn.metrics import roc_auc_score, average_precision_score

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import compute_features_for_dataset, FEATURE_COLUMNS_31
from src.evaluate_metrics import evaluate, issr_at_k
from src.artifact_integrity import (
    load_verified_joblib,
    model_provenance_fields,
    write_provenance_sidecar,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_31feature_integrated.joblib")
CALIBRATOR_PATH = os.path.join(
    PROJECT_ROOT, "models", "platt_calibrator.joblib"
)  # legacy; skipped if absent
ALLELES = [
    "HLA-A*02:01",
    "HLA-A*01:01",
    "HLA-A*03:01",
    "HLA-A*24:02",
    "HLA-A*11:01",
    "HLA-B*07:02",
    "HLA-B*08:01",
    "HLA-B*27:05",
    "HLA-B*35:01",
    "HLA-B*44:02",
]

SARS2_CLEAN = os.path.join(PROJECT_ROOT, "data", "external", "sars2_clean.csv")
INFLUENZA_CLEAN = os.path.join(PROJECT_ROOT, "data", "external", "influenza_clean.csv")

SARS2_SCORED = os.path.join(PROJECT_ROOT, "results", "sars2_scored.csv")
INFLUENZA_SCORED = os.path.join(PROJECT_ROOT, "results", "influenza_scored.csv")


def predict_binding_matrix(peptides):
    """Predict MHC-I presentation using MHCflurry for the 10 target alleles."""
    print(f"Running MHCflurry predictions for {len(peptides)} peptides against 10 alleles...")
    predictor = Class1PresentationPredictor.load()

    per_allele_cols = {}
    for allele in ALLELES:
        print(f"  Predicting for {allele}...")
        pred_df = predictor.predict(peptides=peptides, alleles=[allele], verbose=0)
        # Map allele to column name bind_A0201, etc.
        clean_name = allele.replace("HLA-", "").replace("*", "").replace(":", "")
        col_name = f"bind_{clean_name}"
        per_allele_cols[col_name] = pred_df.set_index("peptide")["presentation_score"].to_dict()

    # Build pivoted dataframe
    pivoted = pd.DataFrame({"peptide": peptides})
    for col_name, score_dict in per_allele_cols.items():
        pivoted[col_name] = pivoted["peptide"].map(score_dict)

    # Compute presentation_score as best presentation across alleles
    pivoted["presentation_score"] = pivoted[
        [f"bind_{a.replace('HLA-', '').replace('*', '').replace(':', '')}" for a in ALLELES]
    ].max(axis=1)
    return pivoted


def bootstrap_ci(y_true, y_scores, metric_fn, n_resamples=2000, alpha=0.05):
    """Calculate 95% bootstrap confidence intervals for a metric."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n = len(y_true)
    rng = np.random.default_rng(42)
    values = []
    for _ in range(n_resamples):
        indices = rng.choice(n, size=n, replace=True)
        sample_true = y_true[indices]
        sample_scores = y_scores[indices]
        if len(np.unique(sample_true)) < 2:
            continue
        values.append(metric_fn(sample_true, sample_scores))
    if not values:
        return float("nan"), float("nan")
    values = sorted(values)
    lower = values[int(len(values) * (alpha / 2))]
    upper = values[int(len(values) * (1.0 - alpha / 2))]
    return lower, upper


def run_evaluation(cohort_path, scored_output, name):
    print(f"\nEvaluating cohort: {name}")
    print("=" * 60)

    # 1. Load clean cohort
    if not os.path.exists(cohort_path):
        print(f"ERROR: Cohort file not found: {cohort_path}", file=sys.stderr)
        return None
    df = pd.read_csv(cohort_path)
    peptides = df["peptide"].tolist()

    # 2. Predict binding matrix
    binding_df = predict_binding_matrix(peptides)

    # 3. Compute TCR Features
    print("Extracting TCR contact features...")
    features_df = compute_features_for_dataset(
        binding_df, peptide_col="peptide", binding_col="presentation_score"
    )

    # Keep the original labels and virus info
    features_df = features_df.merge(df[["peptide", "label", "virus"]], on="peptide", how="left")

    # 4. Score immunogenicity
    print(f"Loading trained RF model: {os.path.basename(MODEL_PATH)}")
    model = load_verified_joblib(MODEL_PATH, required_checksum=True)
    X = features_df[FEATURE_COLUMNS_31].copy()

    raw_scores = model.predict_proba(X)[:, 1]
    features_df["immunogenicity_score"] = raw_scores

    # Apply Platt calibrator if available
    calibrated = False
    if os.path.exists(CALIBRATOR_PATH):
        print("Applying Platt calibration...")
        calibrator = load_verified_joblib(CALIBRATOR_PATH, required_checksum=True)
        logits = np.log((raw_scores + 1e-10) / (1.0 - raw_scores + 1e-10)).reshape(-1, 1)
        features_df["calibrated_score"] = calibrator.predict_proba(logits)[:, 1]
        calibrated = True

    # Determine rank
    rank_col = "calibrated_score" if calibrated else "immunogenicity_score"
    features_df = features_df.sort_values(
        by=[rank_col, "peptide"], ascending=[False, True]
    ).reset_index(drop=True)
    features_df["rank"] = features_df.index + 1

    # Save scored output
    os.makedirs(os.path.dirname(scored_output), exist_ok=True)
    # lineterminator is pinned to LF rather than left to the platform (pandas
    # writes CRLF on Windows) so the sidecar's recorded sha256 is reproducible
    # across machines - same reasoning as scripts/assess_calibration.py.
    features_df.to_csv(scored_output, index=False, lineterminator="\n")
    print(f"Scored results saved to: {scored_output}")

    # Record which model/calibrator artifacts produced this file. Both are
    # untracked and gitignored, so without this, a later overwrite of either
    # makes the scored output permanently unverifiable - the same gap that
    # produced the retracted 0.99 tsnadb_crossdomain figure (D-series,
    # 2026-08-12). Calibrator fields are recorded even when calibration was
    # not applied (calibrator_sha256 is then None) so the sidecar always shows
    # what was and wasn't used, distinguished by key prefix so neither
    # artifact's fields can silently overwrite the other's.
    calibrator_fields = model_provenance_fields(CALIBRATOR_PATH)
    provenance_sidecar = write_provenance_sidecar(
        scored_output,
        script="scripts/score_validation_cohorts.py",
        extra={
            **model_provenance_fields(MODEL_PATH),
            "calibrator_path": calibrator_fields["model_path"],
            "calibrator_sha256": calibrator_fields["model_sha256"],
            "calibrated": calibrated,
        },
    )
    print(f"Provenance written to {provenance_sidecar}")

    # 5. Calculate Metrics and Bootstrap CIs
    y_true = features_df["label"].values
    y_scores = features_df[rank_col].values

    metrics = evaluate(y_true, y_scores)

    # Estimate Bootstrap CIs
    ci_roc = bootstrap_ci(y_true, y_scores, lambda yt, ys: roc_auc_score(yt, ys))
    ci_pr = bootstrap_ci(y_true, y_scores, lambda yt, ys: average_precision_score(yt, ys))
    ci_issr = bootstrap_ci(y_true, y_scores, lambda yt, ys: issr_at_k(yt, ys, 10))

    print("\nResults Summary:")
    print(f"  Peptides Count: {len(y_true)} ({y_true.sum()} Pos, {(y_true == 0).sum()} Neg)")
    print(f"  AUC-ROC:  {metrics['auc_roc']:.4f} (95% CI: [{ci_roc[0]:.4f}, {ci_roc[1]:.4f}])")
    print(f"  AUC-PR:   {metrics['auc_pr']:.4f} (95% CI: [{ci_pr[0]:.4f}, {ci_pr[1]:.4f}])")
    print(f"  ISSR@10:  {metrics['issr_10']:.4f} (95% CI: [{ci_issr[0]:.4f}, {ci_issr[1]:.4f}])")
    print("=" * 60)

    return {
        "virus": name,
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "auc_roc": metrics["auc_roc"],
        "auc_roc_ci": ci_roc,
        "auc_pr": metrics["auc_pr"],
        "auc_pr_ci": ci_pr,
        "issr_10": metrics["issr_10"],
        "issr_10_ci": ci_issr,
    }


def main():
    print("SESTRAV Out-of-Distribution Cohorts Scoring and Validation")
    print("=" * 60)

    results = []

    # Score SARS-CoV-2
    sars_res = run_evaluation(SARS2_CLEAN, SARS2_SCORED, "SARS-CoV-2")
    if sars_res:
        results.append(sars_res)

    # Score Influenza A
    flu_res = run_evaluation(INFLUENZA_CLEAN, INFLUENZA_SCORED, "Influenza A")
    if flu_res:
        results.append(flu_res)

    # Write pre-registration results log in docs or results
    if results:
        log_path = os.path.join(PROJECT_ROOT, "docs", "stage3_results_log.md")
        with open(log_path, "w") as f:
            f.write("# Stage 3 Computational Validation Results Log\n\n")
            f.write(
                "This log records the out-of-distribution evaluation results for the SESTRAV model on independent viral cohorts.\n\n"
            )
            f.write(
                "| Cohort | Peptides | Pos / Neg | AUC-PR (95% CI) | AUC-ROC (95% CI) | ISSR@10 (95% CI) | Target Met? |\n"
            )
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            for r in results:
                target_met = "YES" if r["auc_pr"] >= 0.75 else "NO"
                f.write(
                    f"| {r['virus']} | {r['n_samples']} | {r['n_positive']} / {r['n_samples'] - r['n_positive']} | "
                    f"{r['auc_pr']:.4f} `[{r['auc_pr_ci'][0]:.4f}, {r['auc_pr_ci'][1]:.4f}]` | "
                    f"{r['auc_roc']:.4f} `[{r['auc_roc_ci'][0]:.4f}, {r['auc_roc_ci'][1]:.4f}]` | "
                    f"{r['issr_10']:.4f} `[{r['issr_10_ci'][0]:.4f}, {r['issr_10_ci'][1]:.4f}]` | {target_met} |\n"
                )
            f.write("\n*Note: Bootstrap intervals estimated via N=2,000 resamples.*\n")
            model_fields = model_provenance_fields(MODEL_PATH)
            calibrator_fields = model_provenance_fields(CALIBRATOR_PATH)
            model_sha_prefix = (model_fields["model_sha256"] or "MISSING")[:12]
            calibrator_sha = calibrator_fields["model_sha256"]
            calibrator_sha_prefix = calibrator_sha[:12] if calibrator_sha else "not applied"
            f.write(
                f"\n*Scoring model: `{model_fields['model_path']}` (sha256 `{model_sha_prefix}...`). "
                f"Calibrator: `{calibrator_fields['model_path']}` (sha256 `{calibrator_sha_prefix}"
                f"{'...' if calibrator_sha else ''}`). Full artifact provenance, including complete "
                "hashes, is recorded in the `.provenance.json` sidecar next to each per-cohort scored "
                "CSV in `results/`.*\n"
            )
        print(f"\n[SUCCESS] Wrote Stage 3 results log to: {log_path}")


if __name__ == "__main__":
    main()
