"""SESTRAV command-line interface.

Four subcommands:
  sestrav predict    -- score all peptides from a FASTA proteome
  sestrav validate   -- cross-validate a model on a labeled dataset
  sestrav benchmark  -- compare SESTRAV predictions to gold-standard labels
  sestrav info       -- print environment / provenance summary
"""

from __future__ import annotations

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _require_file(path: str, parser: argparse.ArgumentParser, label: str) -> None:
    if not os.path.isfile(path):
        parser.error(f"{label} not found: '{path}'")


def _read_config() -> dict:
    try:
        import yaml

        cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        if os.path.isfile(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except ImportError:
        pass
    return {}


# ---------------------------------------------------------------------------
# sestrav info
# ---------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    """Print installed versions, active config, and model provenance."""
    import importlib.metadata

    print("=" * 60)
    print("SESTRAV Environment Info")
    print("=" * 60)

    # Package version
    try:
        version = importlib.metadata.version("sestrav")
        print(f"  sestrav version : {version}")
    except importlib.metadata.PackageNotFoundError:
        print("  sestrav version : (editable install - see pyproject.toml)")

    # MHCflurry version
    try:
        import mhcflurry

        print(f"  mhcflurry       : {mhcflurry.__version__}")
    except ImportError:
        print("  mhcflurry       : not installed")

    # PyTorch version
    try:
        import torch

        print(f"  torch           : {torch.__version__}")
        cuda = "available" if torch.cuda.is_available() else "not available"
        print(f"  CUDA            : {cuda}")
    except ImportError:
        print("  torch           : not installed")

    # Active config
    cfg = _read_config()
    if cfg:
        print(f"  feature_mode    : {cfg.get('feature_mode', 'not set')}")
        print(f"  model_path      : {cfg.get('model_path', 'not set')}")
        mhcflurry_pin = cfg.get("mhcflurry_model_version", "not set")
        print(f"  mhcflurry_pin   : {mhcflurry_pin}")
        alleles = cfg.get("alleles", [])
        print(
            f"  allele panel    : {len(alleles)} alleles ({', '.join(alleles[:3])}{'...' if len(alleles) > 3 else ''})"
        )
        viruses = cfg.get("antigens", [])
        print(f"  viruses/panels  : {', '.join(viruses)}")
    else:
        print("  config          : config.yaml not found")

    # MHCflurry pin verification
    if cfg:
        pinned = cfg.get("mhcflurry_model_version")
        try:
            import mhcflurry

            installed = mhcflurry.__version__
            if pinned and installed != pinned:
                print("\n  [WARNING] MHCflurry version mismatch:")
                print(f"    config pins   : {pinned}")
                print(f"    installed     : {installed}")
                print("    Update mhcflurry_model_version in config.yaml to silence this.")
        except ImportError:
            pass

    print("=" * 60)
    return 0


# ---------------------------------------------------------------------------
# sestrav predict
# ---------------------------------------------------------------------------


def cmd_predict(args: argparse.Namespace) -> int:
    """Score all peptides from a FASTA proteome and output ranked CSV."""
    import re

    _require_file(args.fasta, _build_predict_parser(), "--fasta")
    _require_file(args.model, _build_predict_parser(), "--model")

    os.makedirs(args.output, exist_ok=True)

    # Derive a proteome_id from the FASTA filename
    proteome_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", os.path.splitext(os.path.basename(args.fasta))[0])
    lengths = args.lengths if args.lengths else [8, 9, 10, 11]
    alleles = args.alleles if args.alleles else None  # None -> stage2 default panel

    print(f"[sestrav predict] Proteome: {proteome_id}")
    print(f"[sestrav predict] Lengths: {lengths}")
    print(f"[sestrav predict] Output: {args.output}")

    # Stage 1 - peptide generation
    print("[Stage 1] Generating peptides...")
    from functions.stage1_peptide_generation import generate_peptides

    peptides_df = generate_peptides(args.fasta, proteome_id, peptide_lengths=lengths)
    print(f"          {len(peptides_df)} peptides generated")

    # Stage 2 - MHC binding prediction
    print("[Stage 2] Predicting MHC binding...")
    from functions.stage2_mhc_binding_prediction import predict_binding

    binding_df = predict_binding(peptides_df, proteome_id, alleles=alleles)
    print(f"          {len(binding_df)} binding predictions")

    # Stage 3 - TCR feature extraction
    print("[Stage 3] Extracting TCR features...")
    from functions.stage3_tcr_feature_extraction import extract_tcr_features

    features_df = extract_tcr_features(binding_df, proteome_id)
    print(f"          {features_df.shape[1]} features per peptide")

    # Stage 4 - immunogenicity scoring
    print("[Stage 4] Scoring immunogenicity...")
    from functions.stage4_immunogenicity_scoring import score_immunogenicity

    ranked_df, _ = score_immunogenicity(
        features_df,
        proteome_id,
        model_path=args.model,
        freeze_mode=False,
        virus=args.virus,
        per_virus_calibration_dir=args.per_virus_calibration_dir,
    )
    print(f"          {len(ranked_df)} peptides scored")

    # Write final output
    out_path = os.path.join(args.output, f"{proteome_id}_ranked.csv")
    ranked_df.to_csv(out_path, index=False)
    print(f"[sestrav predict] Ranked output -> {out_path}")

    # Top 10 preview
    score_col = (
        "immunogenicity_score"
        if "immunogenicity_score" in ranked_df.columns
        else ranked_df.columns[-1]
    )
    print(f"\nTop 10 candidates (by {score_col}):")
    preview_cols = [c for c in ["peptide", "protein_id", score_col] if c in ranked_df.columns]
    print(ranked_df[preview_cols].head(10).to_string(index=False))
    return 0


# ---------------------------------------------------------------------------
# sestrav validate
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    """Cross-validate a model configuration on a labeled dataset."""

    _require_file(args.dataset, _build_validate_parser(), "--dataset")
    if args.binding_matrix:
        _require_file(args.binding_matrix, _build_validate_parser(), "--binding-matrix")

    cfg = _read_config()
    feature_mode = args.feature_mode or str(cfg.get("feature_mode", "21"))
    binding_matrix = args.binding_matrix or cfg.get("binding_matrix_path")

    print(f"[sestrav validate] Dataset     : {args.dataset}")
    print(f"[sestrav validate] Feature mode: {feature_mode}")
    print(f"[sestrav validate] CV folds    : {args.folds}")

    from src.train_classifier import train_models

    rf_final, xgb_final, rf_avg, xgb_avg = train_models(
        data_path=args.dataset,
        model_dir=args.model_dir,
        n_cv_folds=args.folds,
        feature_mode=int(feature_mode) if feature_mode.isdigit() else feature_mode,
        binding_matrix_path=binding_matrix,
        use_sample_weights=args.sample_weights,
        allow_overwrite=args.allow_overwrite,
    )

    report = {
        "dataset": args.dataset,
        "feature_mode": feature_mode,
        "cv_folds": args.folds,
        "rf": {
            "auc_pr": rf_avg.get("auc_pr"),
            "auc_roc": rf_avg.get("auc_roc"),
            "issr_10": rf_avg.get("issr_10"),
        },
        "xgb": {
            "auc_pr": xgb_avg.get("auc_pr"),
            "auc_roc": xgb_avg.get("auc_roc"),
            "issr_10": xgb_avg.get("issr_10"),
        },
    }

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("# SESTRAV Validation Report\n\n")
            f.write(f"**Dataset:** `{args.dataset}`  \n")
            f.write(f"**Feature mode:** {feature_mode}  \n")
            f.write(f"**CV folds:** {args.folds}  \n\n")
            f.write("## Results\n\n")
            f.write("| Model | AUC-PR | AUC-ROC | ISSR@10 |\n")
            f.write("|-------|--------|---------|--------|\n")
            f.write(
                f"| RF    | {report['rf']['auc_pr']:.4f} | {report['rf']['auc_roc']:.4f} | {report['rf']['issr_10']:.4f} |\n"
            )
            f.write(
                f"| XGB   | {report['xgb']['auc_pr']:.4f} | {report['xgb']['auc_roc']:.4f} | {report['xgb']['issr_10']:.4f} |\n"
            )
        print(f"[sestrav validate] Report -> {args.report}")

    return 0


# ---------------------------------------------------------------------------
# sestrav benchmark
# ---------------------------------------------------------------------------


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Compare SESTRAV predictions to gold-standard immunogenicity labels."""
    import pandas as pd
    from src.evaluate_metrics import evaluate
    from src.iedb_data_loader import GOLD_STANDARD_EPITOPES

    _require_file(args.predictions, _build_benchmark_parser(), "--predictions")

    pred_df = pd.read_csv(args.predictions)

    # Auto-detect score column
    score_col = args.score_column
    if not score_col:
        candidates = ["immunogenicity_score", "score", "probability"]
        score_col = next((c for c in candidates if c in pred_df.columns), pred_df.columns[-1])

    if score_col not in pred_df.columns:
        print(f"[sestrav benchmark] ERROR: score column '{score_col}' not found in predictions")
        print(f"  Available columns: {list(pred_df.columns)}")
        return 1

    # Build gold-standard labels
    gs_set = set(GOLD_STANDARD_EPITOPES)
    peptide_col = "peptide" if "peptide" in pred_df.columns else pred_df.columns[0]
    pred_df["_gs_label"] = pred_df[peptide_col].isin(gs_set).astype(int)

    positives = int(pred_df["_gs_label"].sum())
    print(f"[sestrav benchmark] Predictions : {len(pred_df)} peptides")
    print(
        f"[sestrav benchmark] Gold std    : {positives} positives ({positives / len(pred_df):.1%})"
    )
    print(f"[sestrav benchmark] Score column: {score_col}")

    scores = pred_df[score_col].values
    labels = pred_df["_gs_label"].values

    metrics = evaluate(labels, scores)
    print("\nBenchmark Results:")
    print(f"  AUC-PR   : {metrics['auc_pr']:.4f}")
    print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")
    print(f"  ISSR@10  : {metrics['issr_10']:.4f}")
    print(f"  ISSR@25  : {metrics['issr_25']:.4f}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("# SESTRAV Benchmark Report\n\n")
            f.write(f"**Predictions:** `{args.predictions}`  \n")
            f.write(f"**Score column:** `{score_col}`  \n")
            f.write(f"**Gold-standard positives:** {positives} / {len(pred_df)}\n\n")
            f.write("## Metrics\n\n")
            f.write("| Metric | Value |\n|--------|-------|\n")
            for k, v in metrics.items():
                f.write(f"| {k} | {v:.4f} |\n")
        print(f"[sestrav benchmark] Report -> {args.output}")

    return 0


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------


def _build_predict_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--fasta", required=True, help="Path to viral proteome FASTA file")
    p.add_argument("--model", required=True, help="Path to trained .joblib model file")
    p.add_argument(
        "--output",
        default="results/cli_predict/",
        help="Output directory (default: results/cli_predict/)",
    )
    p.add_argument(
        "--alleles",
        nargs="+",
        default=None,
        help="HLA alleles (default: 10-allele canonical panel)",
    )
    p.add_argument(
        "--lengths", type=int, nargs="+", default=None, help="Peptide lengths (default: 8 9 10 11)"
    )
    p.add_argument(
        "--feature-mode", type=str, default=None, help="Feature mode (default: from config.yaml)"
    )
    # Added 2026-08-15 to stop this parser drifting from main()'s p_predict.
    # This function exists only to render usage text for _require_file's error
    # messages, so a missing flag here does not break dispatch - it silently
    # prints a usage line that omits real options. It had already drifted:
    # --virus and --per-virus-calibration-dir shipped in main() and were never
    # mirrored here, so `sestrav predict` with a bad --fasta advertised a
    # narrower interface than the one it accepts. Keep the two definitions in
    # sync, or collapse them to one source; do not let them diverge again.
    p.add_argument(
        "--virus",
        default=None,
        help="Virus label selecting a per-virus calibrator (default: global calibrator)",
    )
    p.add_argument(
        "--per-virus-calibration-dir",
        default=None,
        help="Directory of per-virus calibrators (default: models/calibration/per_virus)",
    )
    return p


def _build_validate_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--dataset", required=True, help="Path to labeled immunogenicity CSV")
    p.add_argument(
        "--binding-matrix",
        default=None,
        help="Path to peptide binding matrix CSV (required for mode 30+)",
    )
    p.add_argument(
        "--model-dir",
        required=True,
        help="Directory to write trained models and metrics. No default: pass "
        "models/ only when you intend to replace the published artifacts, otherwise "
        "use a scratch directory such as models/local",
    )
    p.add_argument("--folds", type=int, default=5, help="CV folds (default: 5)")
    p.add_argument("--feature-mode", type=str, default=None, help="Feature mode override")
    p.add_argument(
        "--sample-weights", action="store_true", help="Apply bias-correction sample weights"
    )
    p.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace training artifacts that already exist in --model-dir",
    )
    p.add_argument("--report", default=None, help="Path to write markdown validation report")
    return p


def _build_benchmark_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--predictions", required=True, help="Path to ranked predictions CSV")
    p.add_argument(
        "--score-column",
        default=None,
        help="Column name for prediction scores (auto-detected if omitted)",
    )
    p.add_argument("--output", default=None, help="Path to write markdown benchmark report")
    return p


def _build_info_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(add_help=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sestrav",
        description="SESTRAV: Structural Epitope Scoring via TCR Recognition And Vaccinology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  predict    Score all peptides from a FASTA proteome file
  validate   Cross-validate a model configuration on a labeled dataset
  benchmark  Compare SESTRAV predictions to gold-standard labels
  info       Print environment and configuration summary

Examples:
  sestrav info
  sestrav predict --fasta data/proteomes/EBV_B95_8_panel8.fasta --model models/rf_31feature_integrated.joblib --output results/ebv_run/
  sestrav validate --dataset data/immunogenicity_dataset_v5.csv --model-dir models/local --feature-mode 31 --binding-matrix models/peptide_binding_matrix_v5.csv
  sestrav benchmark --predictions results/ebv_run/EBV_B95_8_panel8_ranked.csv
""",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # predict
    p_predict = subparsers.add_parser("predict", help="Score peptides from a FASTA proteome")
    p_predict.add_argument("--fasta", required=True, help="Path to viral proteome FASTA file")
    p_predict.add_argument("--model", required=True, help="Path to trained .joblib model file")
    p_predict.add_argument("--output", default="results/cli_predict/", help="Output directory")
    p_predict.add_argument(
        "--alleles",
        nargs="+",
        default=None,
        metavar="ALLELE",
        help="HLA alleles (default: 10-allele canonical panel from config.yaml)",
    )
    p_predict.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Peptide lengths to generate (default: 8 9 10 11)",
    )
    p_predict.add_argument(
        "--feature-mode",
        type=str,
        default=None,
        help="Feature mode override (default: from config.yaml)",
    )
    p_predict.add_argument(
        "--virus",
        type=str,
        default=None,
        help="Which virus this proteome represents (e.g. SARS-CoV-2), if known. "
        "Selects a per-virus calibrator when one has been promoted for it. "
        "Unset, unrecognised, or off-panel values all fall back to the global "
        "calibrator identically - there is no separate error path.",
    )
    p_predict.add_argument(
        "--per-virus-calibration-dir",
        type=str,
        default=None,
        help="Directory to search for <virus>.joblib calibrators (default: "
        "models/calibration/per_virus, which does not exist until a per-virus "
        "calibrator is promoted).",
    )

    # validate
    p_validate = subparsers.add_parser("validate", help="Cross-validate on a labeled dataset")
    p_validate.add_argument("--dataset", required=True, help="Path to labeled immunogenicity CSV")
    p_validate.add_argument(
        "--binding-matrix", default=None, help="Binding matrix CSV (required for mode 30+)"
    )
    p_validate.add_argument(
        "--model-dir",
        required=True,
        help="Output directory for trained models and metrics. No default: pass "
        "models/ only when you intend to replace the published artifacts, otherwise "
        "use a scratch directory such as models/local",
    )
    p_validate.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    p_validate.add_argument("--feature-mode", type=str, default=None, help="Feature mode override")
    p_validate.add_argument(
        "--sample-weights",
        action="store_true",
        help="Apply EBV/HPV16 and 9-mer bias-correction weights",
    )
    p_validate.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace training artifacts that already exist in --model-dir",
    )
    p_validate.add_argument(
        "--report", default=None, help="Path to write markdown validation report"
    )

    # benchmark
    p_benchmark = subparsers.add_parser(
        "benchmark", help="Benchmark predictions against gold standard"
    )
    p_benchmark.add_argument("--predictions", required=True, help="Path to ranked predictions CSV")
    p_benchmark.add_argument(
        "--score-column", default=None, help="Score column name (auto-detected if omitted)"
    )
    p_benchmark.add_argument("--output", default=None, help="Path to write markdown report")

    # info
    subparsers.add_parser("info", help="Print environment and configuration summary")

    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_help()
        return 0

    dispatch = {
        "predict": cmd_predict,
        "validate": cmd_validate,
        "benchmark": cmd_benchmark,
        "info": cmd_info,
    }
    return dispatch[args.subcommand](args)


if __name__ == "__main__":
    sys.exit(main())
