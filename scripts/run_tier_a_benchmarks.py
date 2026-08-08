"""Run the Tier-A external benchmark comparison (DeepImmuno/BigMHC/MixMHCpred).

Scores results/external_validation_input.csv (n=720) against three external
tools and aggregates by MAX per peptide across alleles, then evaluates all
five candidates (SESTRAV RF, Binding-only, DeepImmuno, BigMHC, MixMHCpred).

This writes two independent git-tracked artifacts:
  data/tier_a_external_benchmarks.csv     per-peptide external tool scores
  results/table3_tier_a_metrics.csv       the certified Tier-A headline table

Neither --scores-output nor --metrics-output has a default: both paths above
are git-tracked, so a bare invocation runs the benchmark and prints results
without writing anything rather than silently rewriting either one.

Reproduce:
  python scripts/run_tier_a_benchmarks.py \\
      --scores-output data/tier_a_external_benchmarks.csv \\
      --metrics-output results/table3_tier_a_metrics.csv
"""
import os
import sys
import subprocess
import pandas as pd
import numpy as np
import argparse
import shutil

TRACKED_SCORES_OUTPUT = "data/tier_a_external_benchmarks.csv"
TRACKED_METRICS_OUTPUT = "results/table3_tier_a_metrics.csv"


def get_data():
    # 1. Load peptides and labels
    peptides_df = pd.read_csv("results/external_validation_input.csv")
    # This has 720 rows. We only care about peptide and label

    # 2. Load peptide-allele pairs
    pairs_df = pd.read_csv("results/external_predig_peptide_allele_pairs.csv")
    # Columns: peptide, allele

    # Merge label into pairs
    df = pd.merge(pairs_df, peptides_df[["peptide", "label"]], on="peptide", how="left")

    # standardize columns for existing wrappers
    df = df.rename(columns={"allele": "hla_allele"})

    return df, peptides_df


def run_deepimmuno(df, tmp_dir):
    print("Running DeepImmuno...")
    input_file = os.path.join(tmp_dir, "deepimmuno_in.csv")

    # DeepImmuno only supports 9 and 10-mers
    valid_mask = df["peptide"].str.len().isin([9, 10])
    di_df = df.loc[valid_mask, ["peptide", "hla_allele"]].copy()
    di_df["hla_allele"] = di_df["hla_allele"].str.replace(":", "")

    if len(di_df) == 0:
        return np.full(len(df), np.nan)

    di_df.to_csv(input_file, index=False, header=False)

    conda_exe = shutil.which("conda") or "conda"
    cmd = [
        conda_exe,
        "run",
        "-n",
        "di",
        "python",
        "deepimmuno-cnn.py",
        "--mode",
        "multiple",
        "--intdir",
        os.path.abspath(input_file),
        "--outdir",
        os.path.abspath(tmp_dir),
    ]
    env = os.environ.copy()
    env["TF_USE_LEGACY_KERAS"] = "1"
    subprocess.run(cmd, cwd=r"_local\tools\DeepImmuno-main", check=True, env=env)

    output_file = os.path.join(tmp_dir, "deepimmuno-cnn-result.txt")
    res = pd.read_csv(output_file, sep="\t")

    # Map back to full dataframe
    scores = np.full(len(df), np.nan)
    scores[valid_mask] = res["immunogenicity"].values
    return scores


def run_bigmhc(df, tmp_dir):
    print("Running BigMHC...")
    input_file = os.path.abspath(os.path.join(tmp_dir, "bigmhc_in.csv"))
    df[["peptide", "hla_allele"]].to_csv(input_file, index=False)

    cmd = [
        os.path.abspath(r"_local\venv_bigmhc\Scripts\python.exe"),
        "predict.py",
        "-i",
        input_file,
        "-m",
        "im",  # BigMHC-IM for immunogenicity
        "-a",
        "1",  # hla is in column 1 (0-indexed)
        "-p",
        "0",  # peptide in col 0
        "-c",
        "1",  # skip header (1 row)
        "-d",
        "cpu",
    ]
    subprocess.run(cmd, cwd=r"_local\tools\bigmhc\src", check=True)

    output_file = input_file + ".prd"
    res = pd.read_csv(output_file)
    return res["BigMHC_IM"].values


def run_mixmhcpred(df, tmp_dir):
    print("Running MixMHCpred 2.2...")
    results = pd.Series(index=df.index, dtype=float)

    perl_exe = r"C:\Program Files\Git\usr\bin\perl.exe"
    script = os.path.abspath(r"_local\tools\MixMHCpred2.2\lib\run_MixMHCpred.pl").replace("\\", "/")
    lib_path = os.path.abspath(r"_local\tools\MixMHCpred2.2\lib").replace("\\", "/")

    for allele, group in df.groupby("hla_allele"):
        peptides_file = os.path.abspath(os.path.join(tmp_dir, "mix_peptides.txt"))
        out_file = os.path.abspath(os.path.join(tmp_dir, "mix_out.txt"))

        group["peptide"].to_csv(peptides_file, index=False, header=False)

        in_path = peptides_file.replace("\\", "/")
        out_path = out_file.replace("\\", "/")

        cmd = [
            perl_exe,
            script,
            "--alleles",
            allele,
            "--input",
            in_path,
            "--dir",
            os.path.abspath(tmp_dir).replace("\\", "/"),
            "--output",
            out_path,
            "--lib",
            lib_path,
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            out_df = pd.read_csv(out_file, sep="\t", comment="#")
            score_col = "Score_bestAllele"
            results.loc[group.index] = out_df[score_col].values
        except Exception as e:
            print(f"Failed for allele {allele}: {e}")
            results.loc[group.index] = np.nan

    return results.values


sys.path.insert(0, os.path.abspath("."))
from src.evaluate_metrics import evaluate


def maybe_write_csv(df: pd.DataFrame, output_path: str | None, columns: list[str]) -> None:
    """Write df[columns] to output_path, or do nothing if output_path is falsy.

    Shared by both write sites below so the write-or-skip decision is
    identical (and independently testable) for both tracked artifacts.
    """
    if not output_path:
        return
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df[columns].to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Tier-A external benchmark comparison."
    )
    parser.add_argument("--smoke", action="store_true", help="Run on a small subset")
    parser.add_argument(
        "--scores-output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            f"Per-peptide external-tool scores CSV path (optional). No "
            f"default: {TRACKED_SCORES_OUTPUT} is a git-tracked artifact, so "
            "this script refuses to guess a destination - omit this flag to "
            "skip writing it."
        ),
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            f"Tier-A metrics table CSV path (optional). No default: "
            f"{TRACKED_METRICS_OUTPUT} is a git-tracked artifact (the source "
            "of the certified headline AUC-PR figure), so this script "
            "refuses to guess a destination - omit this flag to skip "
            "writing it."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    pairs_df, peptides_df = get_data()

    if args.smoke:
        pairs_df = pairs_df.head(20).copy()

    tmp_dir = "_local/bench_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"Running benchmarks on {len(pairs_df)} peptide-allele pairs...")

    pairs_df["deepimmuno_score"] = run_deepimmuno(pairs_df, tmp_dir)
    pairs_df["bigmhc_score"] = run_bigmhc(pairs_df, tmp_dir)
    pairs_df["mixmhcpred_score"] = run_mixmhcpred(pairs_df, tmp_dir)

    # Aggregate by MAX across alleles for each peptide
    print("Aggregating scores by MAX per peptide...")
    agg_df = (
        pairs_df.groupby("peptide")
        .agg(
            {
                "deepimmuno_score": "max",
                "bigmhc_score": "max",
                "mixmhcpred_score": "max",
                "label": "first",  # label is the same for all alleles of a peptide
            }
        )
        .reset_index()
    )

    # We must ensure all 720 peptides are in agg_df, even if some failed.
    agg_df = pd.merge(
        peptides_df[["peptide", "label", "rf_oof_score", "binding_max"]],
        agg_df.drop("label", axis=1),
        on="peptide",
        how="left",
    )

    # Save the per-peptide scores
    maybe_write_csv(
        agg_df,
        args.scores_output,
        ["peptide", "label", "deepimmuno_score", "bigmhc_score", "mixmhcpred_score"],
    )

    print("\n--- Benchmark Results (Tier A) ---")

    metrics_list = []

    # Evaluate sanity anchors first
    for model, col in [
        ("SESTRAV RF", "rf_oof_score"),
        ("Binding-only", "binding_max"),
        ("DeepImmuno", "deepimmuno_score"),
        ("BigMHC", "bigmhc_score"),
        ("MixMHCpred 2.2", "mixmhcpred_score"),
    ]:
        y_score = agg_df[col].values
        y_true = agg_df["label"].values
        valid = ~np.isnan(y_score)

        n_scored = valid.sum()
        coverage_pct = n_scored / len(agg_df) * 100

        if n_scored > 0:
            metrics = evaluate(y_true[valid], y_score[valid])
            auc = metrics["auc_roc"]
            pr = metrics["auc_pr"]
            issr = metrics["issr_10"]
        else:
            auc = np.nan
            pr = np.nan
            issr = np.nan

        metrics_list.append(
            {
                "tool": model,
                "auc_pr": pr,
                "auc_roc": auc,
                "issr_10": issr,
                "n_scored": n_scored,
                "coverage_pct": coverage_pct,
            }
        )

        print(f"{model}:")
        print(f"  AUC-PR:  {pr:.4f}")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  ISSR@10: {issr:.4f}")
        print(f"  Coverage: {n_scored}/{len(agg_df)} ({coverage_pct:.1f}%)")

    metrics_df = pd.DataFrame(metrics_list)
    maybe_write_csv(
        metrics_df, args.metrics_output, ["tool", "auc_pr", "auc_roc", "issr_10", "n_scored", "coverage_pct"]
    )


if __name__ == "__main__":
    main()
