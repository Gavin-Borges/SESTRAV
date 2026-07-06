"""Sample a high-confidence subset of the TSNAdb v2 SNV cohort for cross-domain evaluation.

Reads data/SNV-derived.txt (raw download, confidence columns intact), applies
allele and confidence filters, and draws a seeded 5,000-peptide sample for use
as the positive arm of the tumor cross-domain benchmark.

Filters (applied in order):
  1. Canonical-10 HLA alleles only (SESTRAV panel).
  2. Peptide length 8-11 (MHC Class I canonical range).
  3. Valid standard amino acids only.
  4. DeepImmuno immunogenicity score >= 0.5  (Deep_imm column).
  5. MHCflurry presentation rank <= 2.0%     (MHCf_rank (%) column).
  6. Deduplicate on (peptide, hla_allele).
  7. Seeded random sample of 5,000 (seed=42).

Output: data/tsnadb_crossdomain_cohort.csv (v4 schema) + _provenance.json sidecar.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dataset_utils import normalize_peptides, validate_against_schema, write_provenance

CANONICAL_10 = [
    "HLA-A*01:01",
    "HLA-A*02:01",
    "HLA-A*03:01",
    "HLA-A*11:01",
    "HLA-A*24:02",
    "HLA-B*07:02",
    "HLA-B*08:01",
    "HLA-B*27:05",
    "HLA-B*35:01",
    "HLA-B*44:02",
]

DEEP_IMM_MIN = 0.5
MHCF_RANK_MAX = 2.0
SAMPLE_N = 5_000
SEED = 42


def _normalize_hla(series: pd.Series) -> pd.Series:
    return series.str.replace(r"(HLA-[A-Z])(\d)", r"\1*\2", regex=True)


def build_cohort(raw_path: str, sample_n: int = SAMPLE_N, seed: int = SEED) -> pd.DataFrame:
    """Load raw TSNAdb file, filter, deduplicate, and sample.

    Returns a DataFrame conforming to the v4 schema with label=1 for all rows.
    """
    df = pd.read_csv(raw_path, sep="\t")

    # Normalize HLA format: HLA-A02:01 → HLA-A*02:01
    df["HLA_norm"] = _normalize_hla(df["HLA"].astype(str))

    # Filter 1: canonical-10 alleles only
    df = df[df["HLA_norm"].isin(CANONICAL_10)].copy()
    print(f"After allele filter ({len(CANONICAL_10)} alleles): {len(df):,} rows")

    # Filter 2+3: confidence thresholds
    conf_mask = (df["Deep_imm"] >= DEEP_IMM_MIN) & (df["MHCf_rank (%)"] <= MHCF_RANK_MAX)
    df = df[conf_mask].copy()
    print(
        f"After confidence filter (Deep_imm>={DEEP_IMM_MIN}, MHCf_rank<={MHCF_RANK_MAX}%): "
        f"{len(df):,} rows"
    )

    # Build v4 schema, then normalize_peptides enforces length + amino acid validity
    df_v4 = pd.DataFrame(
        {
            "peptide": df["Peptide"].astype(str).str.strip().str.upper(),
            "label": 1,
            "virus": "None",
            "protein": df["Mutation"].fillna("Unknown"),
            "strain": df["Tissue"].fillna("Unknown"),
            "hla_allele": df["HLA_norm"],
            "source_type": "Tumor",
            "database_source": "TSNAdb",
        }
    )
    df_v4 = normalize_peptides(df_v4)
    df_v4 = df_v4.drop_duplicates(subset=["peptide", "hla_allele"])
    # Deterministic sort before sampling so seed=42 gives the same result on any OS
    df_v4 = df_v4.sort_values(["peptide", "hla_allele"]).reset_index(drop=True)
    print(f"After deduplication on (peptide, hla_allele): {len(df_v4):,} rows")

    if len(df_v4) > sample_n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(df_v4), size=sample_n, replace=False))
        df_v4 = df_v4.iloc[idx].reset_index(drop=True)

    print(f"Final sample: {len(df_v4):,} rows")
    return df_v4


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample TSNAdb cross-domain cohort")
    parser.add_argument(
        "--input", default="data/SNV-derived.txt", help="Raw TSNAdb SNV-derived file (TSV)"
    )
    parser.add_argument("--output", default="data/tsnadb_crossdomain_cohort.csv")
    parser.add_argument("--schema", default="data/immunogenicity_dataset_v4_schema.json")
    parser.add_argument("--n", type=int, default=SAMPLE_N, help="Sample size")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    df = build_cohort(args.input, sample_n=args.n, seed=args.seed)
    validate_against_schema(df, args.schema)
    df.to_csv(args.output, index=False)
    write_provenance(
        args.output,
        sources=[args.input],
        row_count=len(df),
        extra={
            "database": "TSNAdb",
            "filters": {
                "canonical_alleles": CANONICAL_10,
                "deep_imm_min": DEEP_IMM_MIN,
                "mhcf_rank_max_pct": MHCF_RANK_MAX,
            },
            "sample_n": args.n,
            "seed": args.seed,
        },
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
