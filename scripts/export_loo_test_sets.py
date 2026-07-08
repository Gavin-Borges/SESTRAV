"""
Export per-virus held-out test partitions as TSV files for external benchmarking.

For each of the 9 LOO viruses, writes a TSV containing all non-quarantined
positives plus real IEDB-tested negatives (negative_origin in
{tested_negative, iedb_api}). Synthetic decoys (allele_matched_nonbinder,
viral_proteome_decoy) are excluded from every exported partition.

Output TSV columns: peptide, hla_allele, label, negative_origin, reference_pmid

Usage:
    python scripts/export_loo_test_sets.py
    python scripts/export_loo_test_sets.py --output-dir results/loo_test_sets/
    python scripts/export_loo_test_sets.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

DATASET_DEFAULT = "data/immunogenicity_dataset_v5.csv"
OUTPUT_DIR_DEFAULT = "results/loo_test_sets/"

# Canonical short labels used in the v5 dataset for the 9 LOO viruses.
LOO_VIRUSES = [
    "CMV",
    "HBV",
    "HCV",
    "EBV",
    "IAV",
    "HPV",
    "SARS-CoV-2",
    "DENV",
    "HIV-1",
]

# Amendment 7: only these negative_origin values qualify for the test partition.
REAL_NEG_ORIGINS = {"tested_negative", "iedb_api"}

OUTPUT_COLUMNS = ["peptide", "hla_allele", "label", "negative_origin", "reference_pmid"]


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[error] Dataset file not found: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False)
    print(f"[export] Loaded {len(df):,} rows from {path}")
    return df


def build_test_partition(df: pd.DataFrame, virus: str) -> pd.DataFrame:
    """Return held-out test rows for one virus.

    Includes all non-quarantined positives plus non-quarantined real IEDB
    negatives (tested_negative or iedb_api origin). allele_matched_nonbinder
    and viral_proteome_decoy rows are intentionally omitted.
    """
    virus_rows = df[df["virus"] == virus]
    active = virus_rows[~virus_rows["is_quarantined"].fillna(False)]
    positives = active[active["label"] == 1]
    negatives = active[
        (active["label"] == 0) & active["negative_origin"].isin(REAL_NEG_ORIGINS)
    ]
    partition = pd.concat([positives, negatives], ignore_index=True)
    return partition[OUTPUT_COLUMNS]


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Export per-virus LOO held-out test partitions as TSV files for "
            "external benchmarking (e.g., NetMHCpan, BigMHC)."
        )
    )
    p.add_argument(
        "--dataset",
        default=DATASET_DEFAULT,
        help=f"Path to v5 immunogenicity dataset CSV (default: {DATASET_DEFAULT})",
    )
    p.add_argument(
        "--output-dir",
        default=OUTPUT_DIR_DEFAULT,
        help=f"Directory for output TSV files (default: {OUTPUT_DIR_DEFAULT})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary table without writing any files",
    )
    args = p.parse_args()

    df = load_dataset(args.dataset)

    partitions = []
    for virus in LOO_VIRUSES:
        partition = build_test_partition(df, virus)
        n_pos = int((partition["label"] == 1).sum())
        n_neg = int((partition["label"] == 0).sum())
        partitions.append(
            {
                "virus": virus,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_total": n_pos + n_neg,
                "partition": partition,
            }
        )

    print()
    print(f"{'virus':<14} {'n_positive':>11} {'n_negative':>11} {'n_total':>9}")
    print("-" * 50)
    for r in partitions:
        print(
            f"{r['virus']:<14} {r['n_positive']:>11,} {r['n_negative']:>11,} "
            f"{r['n_total']:>9,}"
        )

    if args.dry_run:
        print("\n[export] Dry run - no files written.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for r in partitions:
        safe_name = r["virus"].replace("/", "_").replace(" ", "_")
        out_path = os.path.join(args.output_dir, f"{safe_name}_held_out.tsv")
        r["partition"].to_csv(out_path, sep="\t", index=False)
        print(f"[export] Wrote {out_path} ({r['n_total']:,} rows)")

    print(f"\n[export] Done. {len(partitions)} files written to {args.output_dir}")


if __name__ == "__main__":
    main()
