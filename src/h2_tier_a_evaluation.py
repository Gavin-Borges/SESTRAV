"""
SESTRAV H2 Tier A Evaluation

Runs the proposal-aligned Tier A test on the labeled IEDB benchmark:
  - binding-only baseline (max of 10 per-allele binding scores)
  - integrated model (out-of-fold predicted probabilities)

Outputs:
  - h2_tier_a_fold_metrics.csv
  - h2_tier_a_summary.csv
  - h2_tier_a_summary.md
  - h2_tier_a_oof_scores.csv (per-row OOF peptide/label/integrated_score/
    binding_score, one row per validation-fold member; lets any published
    ISSR/recall figure derived from this run be checked for top-K tie
    sensitivity without a retrain)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from scipy.stats import false_discovery_control

from src.artifact_guard import guard_planned_paths, planned_paths_under
from src.artifact_integrity import (
    load_verified_joblib,
    model_provenance_fields,
    write_provenance_sidecar,
)
from src.evaluate_metrics import evaluate
from src.features import BINDING_ALLELE_COLUMNS
from src.train_classifier import prepare_features, prepare_features_30, prepare_features_50
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES
from src.subgroup_eval import evaluate_subgroups
from src.naming import resolve_model_path


def _bootstrap_ratio_ci(
    y_true: np.ndarray,
    integrated_scores: np.ndarray,
    binding_scores: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Bootstrap CI for ISSR@10 ratio using out-of-fold predictions."""
    n = len(y_true)
    if n == 0:
        return np.nan, np.nan

    ratios = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        int_b = integrated_scores[idx]
        bind_b = binding_scores[idx]
        m_int = evaluate(y_b, int_b)
        m_bind = evaluate(y_b, bind_b)
        denom = float(m_bind["issr_10"])
        if denom > 0:
            ratios.append(float(m_int["issr_10"] / denom))

    if not ratios:
        return np.nan, np.nan

    lo = float(np.percentile(ratios, 100 * (alpha / 2)))
    hi = float(np.percentile(ratios, 100 * (1 - alpha / 2)))
    return lo, hi


def _paired_sign_flip_pvalue(deltas: np.ndarray) -> float:
    """Exact one-sided sign-flip test p-value for paired fold deltas > 0."""
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    if n == 0:
        return np.nan
    observed = float(np.mean(deltas))
    if observed <= 0:
        return 1.0

    # Exact enumeration is tractable here (default n=5 folds).
    means = []
    for mask in range(1 << n):
        signs = np.ones(n)
        for i in range(n):
            if (mask >> i) & 1:
                signs[i] = -1.0
        means.append(float(np.mean(deltas * signs)))
    means_arr = np.asarray(means)
    return float(np.mean(means_arr >= observed))


def _build_features(
    df: pd.DataFrame,
    model_n_features: int,
    binding_matrix_path: str,
) -> pd.DataFrame:
    """Build model input matrix matching the serialized model feature count."""
    if model_n_features == 21:
        return prepare_features(df, include_binding=False)
    if model_n_features == 30:
        return prepare_features_30(df, binding_matrix_path)
    if model_n_features == 50:
        return prepare_features_50(df, binding_matrix_path)
    raise ValueError(
        f"Unsupported model feature count: {model_n_features}. Expected 21, 30, or 50. "
        "H2 Tier A is a v3-corpus evaluation and its certified result comes from the "
        "30-feature model; a 31-feature model is NOT simply a newer drop-in here, since "
        "scoring the v3 corpus with it would produce numbers that diverge from the "
        "certified H2 ledger (docs/claims_register.md D17). Pass --model-path explicitly "
        "if you intend a non-certified configuration."
    )


def _attach_binding_max(df: pd.DataFrame, binding_matrix_path: str) -> pd.Series:
    """Attach binding-only scalar b_i = max over 10 allele presentation scores."""
    binding_df = pd.read_csv(binding_matrix_path)
    missing_cols = [c for c in BINDING_ALLELE_COLUMNS if c not in binding_df.columns]
    if missing_cols:
        raise ValueError("Binding matrix is missing required columns: " + ", ".join(missing_cols))

    binding_max = (
        binding_df[["peptide"] + BINDING_ALLELE_COLUMNS]
        .copy()
        .assign(binding_max=lambda d: d[BINDING_ALLELE_COLUMNS].max(axis=1))
        .drop_duplicates(subset=["peptide"])
        .loc[:, ["peptide", "binding_max"]]
    )

    merged = df[["peptide"]].merge(binding_max, on="peptide", how="left")
    return merged["binding_max"].fillna(0.0)


def _aggregate(metrics_rows: List[Dict]) -> pd.DataFrame:
    """Aggregate fold-level metrics into mean and std by method."""
    fold_df = pd.DataFrame(metrics_rows)
    if "subgroup_key" in fold_df.columns:
        fold_df = fold_df[fold_df["subgroup_key"].fillna("overall") == "overall"]
    numeric_cols = ["auc_roc", "auc_pr", "issr_10", "issr_25"]
    agg = fold_df.groupby("method")[numeric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "method",
        "auc_roc_mean",
        "auc_roc_std",
        "auc_pr_mean",
        "auc_pr_std",
        "issr_10_mean",
        "issr_10_std",
        "issr_25_mean",
        "issr_25_std",
    ]
    return agg


def planned_h2_tier_a_paths(output_dir: str) -> list[str]:
    """Every path run_h2_tier_a writes into output_dir.

    All four are written directly by run_h2_tier_a; this entry point delegates
    no output to any other module. `evaluate_subgroups` is the only helper it
    calls that could plausibly write, and it returns DataFrames without touching
    disk. There are no derived filenames here, unlike the `.replace()`-built
    names in data_bias_audit and gold_standard_sensitivity.
    """
    names = [
        "h2_tier_a_fold_metrics.csv",  # fold_df
        "h2_tier_a_summary.csv",  # full_summary_df (summary + h2_decision rows)
        "h2_tier_a_summary.md",  # markdown decision report
        "h2_tier_a_oof_scores.csv",  # oof_df: per-row fold/peptide/label/scores
    ]
    return planned_paths_under(output_dir, names)


def _guard_output_dir(output_dir: str, allow_overwrite: bool) -> None:
    """Refuse to clobber artifacts already on disk unless overwrite is explicit."""
    guard_planned_paths(
        output_dir,
        planned_h2_tier_a_paths(output_dir),
        allow_overwrite,
        flag="--output-dir",
        api_hint="run_h2_tier_a(..., allow_overwrite=True)",
        detail=(
            ": h2_tier_a_summary.md is the source behind the H2 Tier A primary-hypothesis "
            "result reported in README.md. The previously certified R10 = 0.9494 is RETRACTED "
            "as void (docs/claims_register.md D17): it was computed against an all-zeros "
            "binding matrix, so the binding-only arm was a constant"
        ),
    )


def run_h2_tier_a(
    data_path: str,
    model_path: str,
    binding_matrix_path: str,
    output_dir: str,
    n_splits: int = 5,
    random_state: int = 42,
    n_bootstrap: int = 5000,
    allow_overwrite: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run Tier A evaluation and write CSV + markdown artifacts."""
    _guard_output_dir(output_dir, allow_overwrite)
    os.makedirs(output_dir, exist_ok=True)

    model_path = resolve_model_path(model_path)
    df = pd.read_csv(data_path)

    gs_mask = df["peptide"].isin(GOLD_STANDARD_EPITOPES)
    train_pool = df.loc[~gs_mask].copy().reset_index(drop=True)

    # EXACT/SUBSTRING OVERLAP CHECK: Remove train pool items that are exact/substrings of Gold Standard
    gs_peptides = set(GOLD_STANDARD_EPITOPES)
    overlap_mask = train_pool["peptide"].apply(
        lambda p: p in gs_peptides or any(p in gs or gs in p for gs in gs_peptides)
    )
    if overlap_mask.any():
        print(
            f"Removed {overlap_mask.sum()} peptides from train pool due to exact/substring overlap with Gold Standard."
        )
    train_pool = train_pool.loc[~overlap_mask].copy().reset_index(drop=True)

    y = train_pool["label"].to_numpy()
    subgroup_columns = [c for c in ["virus", "strain"] if c in train_pool.columns]

    template_model = load_verified_joblib(model_path, required_checksum=True)
    model_prov = model_provenance_fields(model_path)
    model_n_features = getattr(template_model, "n_features_in_", None)
    if model_n_features is None:
        raise ValueError(f"Model at {model_path} does not expose n_features_in_.")

    X = _build_features(train_pool, model_n_features, binding_matrix_path)
    binding_scores = _attach_binding_max(train_pool, binding_matrix_path).to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rng = np.random.default_rng(random_state)

    rows: List[Dict] = []
    fold_issr10_delta: List[float] = []
    oof_y: List[np.ndarray] = []
    oof_int: List[np.ndarray] = []
    oof_bind: List[np.ndarray] = []
    oof_rows: List[pd.DataFrame] = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = clone(template_model)
        model.fit(X_tr, y_tr)
        integrated_scores = model.predict_proba(X_val)[:, 1]
        binding_val = binding_scores[val_idx]
        oof_y.append(y_val)
        oof_int.append(integrated_scores)
        oof_bind.append(binding_val)

        m_int = evaluate(y_val, integrated_scores)
        m_bind = evaluate(y_val, binding_val)
        fold_issr10_delta.append(float(m_int["issr_10"] - m_bind["issr_10"]))

        rows.append(
            {
                "fold": fold_idx,
                "method": "integrated_model",
                "n_val": len(val_idx),
                "subgroup_key": "overall",
                "subgroup_value": "all",
                **m_int,
            }
        )
        rows.append(
            {
                "fold": fold_idx,
                "method": "binding_only_max",
                "n_val": len(val_idx),
                "subgroup_key": "overall",
                "subgroup_value": "all",
                **m_bind,
            }
        )

        fold_meta = train_pool.iloc[val_idx].copy().reset_index(drop=True)
        fold_meta["label"] = y_val
        fold_meta["integrated_score"] = integrated_scores
        fold_meta["binding_score"] = binding_val

        oof_rows.append(
            pd.DataFrame(
                {
                    "fold": fold_idx,
                    "peptide": fold_meta["peptide"].to_numpy(),
                    **{col: fold_meta[col].to_numpy() for col in subgroup_columns},
                    "label": y_val,
                    "integrated_score": integrated_scores,
                    "binding_score": binding_val,
                }
            )
        )

        for row in evaluate_subgroups(
            fold_meta,
            score_col="integrated_score",
            label_col="label",
            group_columns=subgroup_columns,
            min_group_size=15,
        ):
            if row["subgroup_key"] == "overall":
                continue
            rows.append({"fold": fold_idx, "method": "integrated_model", **row})
        for row in evaluate_subgroups(
            fold_meta,
            score_col="binding_score",
            label_col="label",
            group_columns=subgroup_columns,
            min_group_size=15,
        ):
            if row["subgroup_key"] == "overall":
                continue
            rows.append({"fold": fold_idx, "method": "binding_only_max", **row})

    fold_df = pd.DataFrame(rows)
    summary_df = _aggregate(rows)
    oof_df = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame(
        columns=["fold", "peptide", *subgroup_columns, "label", "integrated_score", "binding_score"]
    )

    bind_row = summary_df.loc[summary_df["method"] == "binding_only_max"].iloc[0]
    int_row = summary_df.loc[summary_df["method"] == "integrated_model"].iloc[0]

    denom_10 = float(bind_row["issr_10_mean"])
    denom_25 = float(bind_row["issr_25_mean"])
    ratio_10 = float(int_row["issr_10_mean"] / denom_10) if denom_10 > 0 else np.nan
    ratio_25 = float(int_row["issr_25_mean"] / denom_25) if denom_25 > 0 else np.nan
    fold_delta_p = _paired_sign_flip_pvalue(np.asarray(fold_issr10_delta, dtype=float))

    # FDR Correction (Benjamini-Hochberg) on the single p-value, or multiple if extended
    fdr_corrected_p = float(false_discovery_control([fold_delta_p])[0])

    y_oof = np.concatenate(oof_y) if oof_y else np.array([])
    int_oof = np.concatenate(oof_int) if oof_int else np.array([])
    bind_oof = np.concatenate(oof_bind) if oof_bind else np.array([])
    ratio_10_ci_lo, ratio_10_ci_hi = _bootstrap_ratio_ci(
        y_oof,
        int_oof,
        bind_oof,
        rng=rng,
        n_bootstrap=n_bootstrap,
    )

    stability_threshold = 0.08
    stable_ratio = denom_10 >= stability_threshold
    ci_supports_h2 = bool(np.isfinite(ratio_10_ci_lo) and ratio_10_ci_lo >= 2.0)
    h2_supported = (ratio_10 >= 2.0) and stable_ratio and ci_supports_h2

    decision_df = pd.DataFrame(
        [
            {
                "method": "h2_decision",
                "auc_roc_mean": np.nan,
                "auc_roc_std": np.nan,
                "auc_pr_mean": np.nan,
                "auc_pr_std": np.nan,
                "issr_10_mean": np.nan,
                "issr_10_std": np.nan,
                "issr_25_mean": np.nan,
                "issr_25_std": np.nan,
                "issr_10_ratio_integrated_over_binding": ratio_10,
                "issr_25_ratio_integrated_over_binding": ratio_25,
                "issr_10_ratio_bootstrap_ci_low": ratio_10_ci_lo,
                "issr_10_ratio_bootstrap_ci_high": ratio_10_ci_hi,
                "issr_10_delta_fold_signflip_p_greater": fold_delta_p,
                "issr_10_delta_fdr_corrected_p": fdr_corrected_p,
                "binding_issr_10_mean": denom_10,
                "ratio_stable_binding_issr10_gte_0_08": stable_ratio,
                "ratio_ci_low_gte_2": ci_supports_h2,
                "h2_supported_ratio_gte_2_and_stable": h2_supported,
                "model_path": model_path,
                "model_sha256": model_prov["model_sha256"],
                "feature_count": model_n_features,
                "n_total_labeled": len(df),
                "n_train_pool_after_gold_standard_holdout": len(train_pool),
                "n_gold_standard_held_out": int(gs_mask.sum()),
            }
        ]
    )

    summary_df = summary_df.copy()
    summary_df["issr_10_ratio_integrated_over_binding"] = np.nan
    summary_df["issr_25_ratio_integrated_over_binding"] = np.nan
    summary_df["issr_10_ratio_bootstrap_ci_low"] = np.nan
    summary_df["issr_10_ratio_bootstrap_ci_high"] = np.nan
    summary_df["issr_10_delta_fold_signflip_p_greater"] = np.nan
    summary_df["issr_10_delta_fdr_corrected_p"] = np.nan
    summary_df["binding_issr_10_mean"] = np.nan
    summary_df["ratio_stable_binding_issr10_gte_0_08"] = np.nan
    summary_df["ratio_ci_low_gte_2"] = np.nan
    summary_df["h2_supported_ratio_gte_2_and_stable"] = np.nan
    summary_df["model_path"] = model_path
    summary_df["model_sha256"] = model_prov["model_sha256"]
    summary_df["feature_count"] = model_n_features
    summary_df["n_total_labeled"] = len(df)
    summary_df["n_train_pool_after_gold_standard_holdout"] = len(train_pool)
    summary_df["n_gold_standard_held_out"] = int(gs_mask.sum())

    full_summary_df = pd.concat([summary_df, decision_df], ignore_index=True)

    fold_csv = os.path.join(output_dir, "h2_tier_a_fold_metrics.csv")
    summary_csv = os.path.join(output_dir, "h2_tier_a_summary.csv")
    summary_md = os.path.join(output_dir, "h2_tier_a_summary.md")
    oof_csv = os.path.join(output_dir, "h2_tier_a_oof_scores.csv")

    fold_df.to_csv(fold_csv, index=False, lineterminator="\n")
    full_summary_df.to_csv(summary_csv, index=False, lineterminator="\n")
    oof_df.to_csv(oof_csv, index=False, lineterminator="\n")

    write_provenance_sidecar(
        fold_csv,
        script="src/h2_tier_a_evaluation.py",
        extra=model_prov,
    )
    write_provenance_sidecar(
        summary_csv,
        script="src/h2_tier_a_evaluation.py",
        extra=model_prov,
    )
    write_provenance_sidecar(
        oof_csv,
        script="src/h2_tier_a_evaluation.py",
        extra=model_prov,
    )

    decision_line = "SUPPORTED" if h2_supported else "NOT SUPPORTED"
    stability_line = (
        "stable" if stable_ratio else "potentially unstable (low binding ISSR@10 denominator)"
    )

    md = f"""# H2 Tier A Evaluation Summary

## Inputs
- Dataset: `{data_path}`
- Integrated model template: `{model_path}` (sha256: `{model_prov["model_sha256"]}`)
- Binding matrix: `{binding_matrix_path}`
- CV: StratifiedKFold(n_splits={n_splits}, shuffle=True, random_state={random_state})
- Splitter note: this is a label-only (ungrouped) splitter, but it is **peptide-disjoint
  on this corpus by construction** - the v3 dataset is 1,004 rows over 1,004 unique
  peptides, so no peptide can straddle a fold boundary and a peptide-grouped re-run
  would be a no-op. This figure is therefore NOT exposed to the D15 leakage defect
  (`docs/claims_register.md` D15, D17); the disclosure is stated here rather than left
  for a reader to infer from the splitter name.
- Gold-standard peptides held out before CV: `{int(gs_mask.sum())}`

## Fold-aggregated metrics (mean +/- std)
- Integrated model ISSR@10: `{float(int_row["issr_10_mean"]):.4f} +/- {float(int_row["issr_10_std"]):.4f}`
- Binding-only ISSR@10: `{denom_10:.4f} +/- {float(bind_row["issr_10_std"]):.4f}`
- Integrated model ISSR@25: `{float(int_row["issr_25_mean"]):.4f} +/- {float(int_row["issr_25_std"]):.4f}`
- Binding-only ISSR@25: `{denom_25:.4f} +/- {float(bind_row["issr_25_std"]):.4f}`

## Enrichment ratios
- R10 = ISSR@10(integrated) / ISSR@10(binding-only): `{ratio_10:.4f}`
- R25 = ISSR@25(integrated) / ISSR@25(binding-only): `{ratio_25:.4f}`
- Bootstrap 95% CI for R10 (OOF): `[{ratio_10_ci_lo:.4f}, {ratio_10_ci_hi:.4f}]`
- Fold-level paired sign-flip p-value for ISSR@10 delta > 0: `{fold_delta_p:.4f}` (FDR Corrected: `{fdr_corrected_p:.4f}`)
- Binding ISSR@10 denominator quality: `{stability_line}`

## H2 Decision
- Rule used: `R10 >= 2.0`, `binding ISSR@10 >= 0.08`, and `lower 95% CI(R10) >= 2.0`
- Result: **{decision_line}**

## Output files
- Fold metrics CSV: `h2_tier_a_fold_metrics.csv`
- Summary CSV: `h2_tier_a_summary.csv`
- Per-row OOF scores CSV: `h2_tier_a_oof_scores.csv`
"""
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(md)

    return fold_df, full_summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SESTRAV H2 Tier A evaluation")
    # These three defaults are the V3 TRIPLE, and that is deliberate - do not
    # "modernise" them to v4/v5 or to the canonical mode-31 model. H2 Tier A is
    # a v3-corpus evaluation by design: v3 is 1,004 rows over 1,004 UNIQUE
    # peptides, which is exactly why peptide-grouping is a no-op here and why
    # the D15 leakage defect never touched this result. Two mechanical
    # stale-reference sweeps previously bumped them anyway - 906643d moved
    # --data to v4, 710d311 moved --model-path to the 31-feature model and
    # --binding-matrix to v4 - and the result was an entry point that could not
    # run at all: _build_features supports 21/30/50 features and raises on 31,
    # so a bare invocation died before scoring anything. The v4 dataset is also
    # untracked, so that default could never resolve in a fresh clone. Restored
    # 2026-08-13 to the configuration that produced, and reproduces, the
    # certified artifacts in results/ (see h2_tier_a_summary.md's Inputs block,
    # which records these same three paths). Pinned by
    # tests/test_h2_tier_a_results_guard.py::test_cli_defaults_are_the_certified_v3_triple.
    parser.add_argument(
        "--data",
        default="data/immunogenicity_dataset_v3.csv",
        help="Path to labeled immunogenicity dataset CSV",
    )
    parser.add_argument(
        "--model-path",
        default="models/rf_30feature_integrated.joblib",
        help="Path to integrated model .joblib used as CV template",
    )
    parser.add_argument(
        "--binding-matrix",
        default="models/peptide_binding_matrix_v3.csv",
        help="Path to per-allele binding matrix CSV",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for H2 Tier A outputs. No default: this evaluation writes 4 "
        "tracked release artifacts into it, including h2_tier_a_summary.md, the source "
        "behind the H2 Tier A primary-hypothesis result in README.md, so it refuses to "
        "guess a destination. Note the previously certified R10 = 0.9494 is RETRACTED as "
        "void (docs/claims_register.md D17): it was computed against an all-zeros binding "
        "matrix.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified CV folds (default: 5)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for CV shuffling",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Number of bootstrap resamples for R10 CI (default: 5000)",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace H2 Tier A artifacts that already exist in --output-dir. "
        "Without this flag the run aborts before any work if any would be overwritten.",
    )
    args = parser.parse_args()

    run_h2_tier_a(
        data_path=args.data,
        model_path=args.model_path,
        binding_matrix_path=args.binding_matrix,
        output_dir=args.output_dir,
        n_splits=args.cv_folds,
        random_state=args.random_state,
        n_bootstrap=args.bootstrap_samples,
        allow_overwrite=args.allow_overwrite,
    )
