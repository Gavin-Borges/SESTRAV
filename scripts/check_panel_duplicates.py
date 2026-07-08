"""
Check a published validation panel for overlap with the v5 training dataset.

Before ingesting a new panel (e.g., Webster HBV, Riemer HPV), use this script
to identify which panel peptides already exist in the v5 dataset. Rows that
appear in both files cannot be treated as held-out validation.

Output files written to --output-dir (default: same directory as the panel file):
  {panel_stem}_duplicates.tsv  - panel rows that ARE already in the dataset
  {panel_stem}_new_only.tsv   - panel rows that are NOT in the dataset (safe to use)

Usage:
    python scripts/check_panel_duplicates.py --panel-file data/webster_hbv_panel.csv
    python scripts/check_panel_duplicates.py --panel-file data/panel.tsv --hla-check
    python scripts/check_panel_duplicates.py \\
        --panel-file data/panel.csv \\
        --dataset-file data/immunogenicity_dataset_v5.csv \\
        --output-dir results/panel_checks/
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

DATASET_DEFAULT = "data/immunogenicity_dataset_v5.csv"

# Column names compared after normalization (strip + uppercase).
PEPTIDE_COL = "peptide"
HLA_COL = "hla_allele"


def _detect_delimiter(path: str) -> str:
    """Return ',' for .csv files and '\\t' for everything else."""
    _, ext = os.path.splitext(path.lower())
    if ext == ".csv":
        return ","
    if ext in {".tsv", ".txt"}:
        return "\t"
    return ","


def _load_file(path: str, label: str) -> pd.DataFrame:
    """Load a CSV or TSV, trying the extension-inferred delimiter first."""
    if not os.path.exists(path):
        print(f"[error] {label} file not found: {path}", file=sys.stderr)
        sys.exit(1)
    primary_sep = _detect_delimiter(path)
    df = pd.read_csv(path, sep=primary_sep, low_memory=False)
    if df.shape[1] == 1 and primary_sep == ",":
        df = pd.read_csv(path, sep="\t", low_memory=False)
    print(f"[check] Loaded {len(df):,} rows from {path} ({label})")
    return df


def _normalize_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _find_col(df: pd.DataFrame, name: str) -> str | None:
    """Case-insensitive column lookup; returns the actual column name or None."""
    for col in df.columns:
        if col.strip().lower() == name.lower():
            return col
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Check which peptides in a new validation panel already exist in "
            "the v5 training dataset to prevent held-out contamination."
        )
    )
    p.add_argument(
        "--panel-file",
        required=True,
        help="Path to the panel CSV or TSV file (must contain a 'peptide' column)",
    )
    p.add_argument(
        "--dataset-file",
        default=DATASET_DEFAULT,
        help=f"Path to v5 immunogenicity dataset CSV (default: {DATASET_DEFAULT})",
    )
    p.add_argument(
        "--hla-check",
        action="store_true",
        help=(
            "Also report (peptide, hla_allele) pair duplicates in addition to "
            "peptide-only duplicates"
        ),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for output TSV files. "
            "Defaults to the directory containing the panel file."
        ),
    )
    args = p.parse_args()

    panel_path = args.panel_file
    dataset_path = args.dataset_file
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(panel_path))
    panel_stem = os.path.splitext(os.path.basename(panel_path))[0]

    panel = _load_file(panel_path, "panel")
    dataset = _load_file(dataset_path, "dataset")

    panel_pep_col = _find_col(panel, PEPTIDE_COL)
    if panel_pep_col is None:
        print(
            f"[error] Panel file has no 'peptide' column. "
            f"Found columns: {list(panel.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset_pep_col = _find_col(dataset, PEPTIDE_COL)
    if dataset_pep_col is None:
        print(
            f"[error] Dataset file has no 'peptide' column. "
            f"Found columns: {list(dataset.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    panel["_pep_norm"] = _normalize_col(panel[panel_pep_col])
    dataset["_pep_norm"] = _normalize_col(dataset[dataset_pep_col])

    dataset_peptides = set(dataset["_pep_norm"])

    peptide_dup_mask = panel["_pep_norm"].isin(dataset_peptides)
    n_total = len(panel)
    n_dup_peptide = int(peptide_dup_mask.sum())
    n_new_peptide = n_total - n_dup_peptide

    print()
    print("Peptide-level duplicate check:")
    print(f"  Total in panel:          {n_total:,}")
    print(f"  Already in dataset:      {n_dup_peptide:,}")
    print(f"  New (not in dataset):    {n_new_peptide:,}")
    print(f"  Pct new:                 {100.0 * n_new_peptide / max(n_total, 1):.1f}%")

    if args.hla_check:
        panel_hla_col = _find_col(panel, HLA_COL)
        dataset_hla_col = _find_col(dataset, HLA_COL)
        if panel_hla_col is None or dataset_hla_col is None:
            missing = []
            if panel_hla_col is None:
                missing.append("panel")
            if dataset_hla_col is None:
                missing.append("dataset")
            print(
                f"\n[warn] --hla-check requested but hla_allele column not found in: "
                f"{', '.join(missing)}. Skipping pair check."
            )
        else:
            panel["_hla_norm"] = _normalize_col(panel[panel_hla_col])
            dataset["_hla_norm"] = _normalize_col(dataset[dataset_hla_col])
            dataset_pairs = set(
                zip(dataset["_pep_norm"], dataset["_hla_norm"])
            )
            pair_dup_mask = panel.apply(
                lambda row: (row["_pep_norm"], row["_hla_norm"]) in dataset_pairs,
                axis=1,
            )
            n_dup_pair = int(pair_dup_mask.sum())
            print()
            print("(peptide, hla_allele) pair duplicate check:")
            print(f"  Pair duplicates in dataset: {n_dup_pair:,}")
            print(
                f"  Pair-new (not in dataset):  {n_total - n_dup_pair:,} "
                f"({100.0 * (n_total - n_dup_pair) / max(n_total, 1):.1f}%)"
            )

    panel_out = panel.drop(
        columns=[c for c in panel.columns if c.startswith("_")],
        errors="ignore",
    )

    os.makedirs(output_dir, exist_ok=True)

    dup_path = os.path.join(output_dir, f"{panel_stem}_duplicates.tsv")
    new_path = os.path.join(output_dir, f"{panel_stem}_new_only.tsv")

    panel_out[peptide_dup_mask].to_csv(dup_path, sep="\t", index=False)
    panel_out[~peptide_dup_mask].to_csv(new_path, sep="\t", index=False)

    print()
    print(f"[check] Wrote {dup_path} ({n_dup_peptide:,} rows - already in dataset)")
    print(f"[check] Wrote {new_path} ({n_new_peptide:,} rows - safe for held-out use)")


if __name__ == "__main__":
    main()
