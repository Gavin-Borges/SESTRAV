"""
ingest_iedb_negatives.py - Ingest IEDB bulk export CSV files filtering for
negative-outcome T cell assays.

Reads a standard IEDB T cell assay bulk export CSV and outputs a v5-schema-
compatible CSV of filtered, deduplicated negative rows suitable for inclusion
in the v5 dataset build.

Usage:
    python scripts/ingest_iedb_negatives.py \
        --input data/iedb_tcell_export.csv \
        --output data/iedb_negatives_v5.csv \
        --existing-dataset data/immunogenicity_dataset_v4.csv \
        [--dry-run]
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard amino acid alphabet (single-letter codes)
AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

# HLA allele prefix filter - only Class I A/B/C alleles
HLA_PREFIX = ("HLA-A", "HLA-B", "HLA-C")

# IEDB qualitative measures considered negative
NEGATIVE_MEASURES = {"negative", "negative-low"}

# Assay groups considered direct functional assays (quality weight 1.0)
DIRECT_FUNCTIONAL_ASSAY_GROUPS = {
    "T cell",
    "IFN-gamma",
    "ELISpot",
    "IFN-gamma ELISpot",
    "MHC multimer",
    "intracellular cytokine staining",
    "proliferation",
    "cytotoxicity",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


def normalize_hla_allele(raw: str) -> str | None:
    """Normalize an HLA allele string.

    Returns the normalized allele string or None if it cannot be parsed as a
    Class I HLA-A/B/C allele.
    """
    if not isinstance(raw, str):
        return None
    allele = raw.strip()
    if not allele.startswith(HLA_PREFIX):
        return None
    return allele


def is_valid_peptide(seq: str) -> bool:
    """Return True if seq is a standard AA string of length 8-11."""
    if not isinstance(seq, str):
        return False
    seq = seq.strip()
    if len(seq) < 8 or len(seq) > 11:
        return False
    return bool(AA_PATTERN.match(seq))


def assign_quality_weight(assay_group: str | None) -> float:
    """Return quality weight for the given assay group string."""
    if not isinstance(assay_group, str):
        return 0.7
    # Check for keywords indicating direct functional measurement
    ag_lower = assay_group.lower()
    for marker in (
        "ifn-gamma",
        "elispot",
        "multimer",
        "cytotoxicity",
        "intracellular cytokine",
        "proliferation",
    ):
        if marker in ag_lower:
            return 1.0
    return 0.7


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_iedb_export(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load an IEDB bulk export CSV.

    IEDB exports sometimes have a multi-row header; handle the common case
    where the real column header is on row 0 or row 1.
    """
    logger.info("Loading IEDB export from %s", path)
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        sys.exit(1)
    logger.info("Raw input: %d rows x %d columns", len(df), len(df.columns))
    return df


def filter_rows(
    df: pd.DataFrame, logger: logging.Logger
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply all inclusion filters and return (filtered_df, stats_dict)."""
    stats: dict[str, int] = {}
    stats["raw_input"] = len(df)

    # Normalise column names: strip whitespace, lower case for lookup
    col_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=col_map)

    # Identify required columns using flexible matching
    col_lower = {c.lower(): c for c in df.columns}

    def find_col(*candidates: str) -> str | None:
        for c in candidates:
            if c.lower() in col_lower:
                return col_lower[c.lower()]
        return None

    col_host = find_col("Host Organism Name", "host organism name", "host")
    col_measure = find_col(
        "Qualitative Measure", "qualitative measure", "qualitative_measure"
    )
    col_epitope = find_col("Epitope Name", "epitope name", "name", "epitope")
    col_allele = find_col("Allele Name", "allele name", "allele")
    col_assay_group = find_col("Assay Group", "assay group", "assay_group")
    col_antigen = find_col(
        "Antigen Name", "antigen name", "antigen", "description"
    )
    col_pmid = find_col(
        "Reference PubMed ID",
        "reference pubmed id",
        "pubmed id",
        "pubmed_id",
        "pmid",
    )
    col_description = find_col(
        "Description", "description", "antigen description"
    )

    missing = [
        name
        for name, col in (
            ("Host Organism Name", col_host),
            ("Qualitative Measure", col_measure),
            ("Epitope Name", col_epitope),
            ("Allele Name", col_allele),
        )
        if col is None
    ]
    if missing:
        logger.error(
            "Required columns not found in input: %s. Available: %s",
            missing,
            list(df.columns[:20]),
        )
        sys.exit(1)

    # --- Filter 1: Human host ---
    mask_host = (
        df[col_host]
        .fillna("")
        .str.contains("Homo sapiens", case=False, na=False)
    )
    df = df[mask_host].copy()
    stats["after_human_host_filter"] = len(df)
    logger.info(
        "After human host filter: %d rows (removed %d)",
        len(df),
        stats["raw_input"] - len(df),
    )

    # --- Filter 2: Negative qualitative measure ---
    mask_neg = (
        df[col_measure]
        .fillna("")
        .str.strip()
        .str.lower()
        .isin(NEGATIVE_MEASURES)
    )
    df = df[mask_neg].copy()
    stats["after_negative_measure_filter"] = len(df)
    logger.info(
        "After negative measure filter: %d rows (removed %d)",
        len(df),
        stats["after_human_host_filter"] - len(df),
    )

    # --- Filter 3 & 4: Peptide length 8-11 and standard AA ---
    peptides = df[col_epitope].fillna("").str.strip()
    mask_peptide = peptides.apply(is_valid_peptide)
    df = df[mask_peptide].copy()
    stats["after_peptide_filter"] = len(df)
    logger.info(
        "After peptide validity filter (length 8-11, standard AA): %d rows (removed %d)",
        len(df),
        stats["after_negative_measure_filter"] - len(df),
    )

    # --- Filter 5: HLA allele parseable as Class I A/B/C ---
    alleles_raw = df[col_allele].fillna("").str.strip()
    mask_hla = alleles_raw.apply(lambda x: normalize_hla_allele(x) is not None)
    df = df[mask_hla].copy()
    stats["after_hla_filter"] = len(df)
    logger.info(
        "After HLA Class I A/B/C filter: %d rows (removed %d)",
        len(df),
        stats["after_peptide_filter"] - len(df),
    )

    return df, stats, col_epitope, col_allele, col_assay_group, col_antigen, col_pmid, col_description


def build_output(
    df: pd.DataFrame,
    col_epitope: str,
    col_allele: str,
    col_assay_group: str | None,
    col_antigen: str | None,
    col_pmid: str | None,
    col_description: str | None,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Construct the v5-schema output DataFrame from filtered IEDB rows."""
    out: dict[str, object] = {}

    out["peptide"] = df[col_epitope].str.strip()
    out["label"] = 0
    out["hla_allele"] = (
        df[col_allele].fillna("").str.strip().apply(normalize_hla_allele)
    )

    # virus: attempt to extract from description or antigen name
    if col_description is not None:
        out["virus"] = df[col_description].fillna("Unknown").str.strip()
    elif col_antigen is not None:
        out["virus"] = df[col_antigen].fillna("Unknown").str.strip()
    else:
        out["virus"] = "Unknown"

    # protein
    if col_antigen is not None:
        out["protein"] = df[col_antigen].fillna("").str.strip()
    else:
        out["protein"] = ""

    out["strain"] = None
    out["source_type"] = "Virus"
    out["database_source"] = "IEDB"
    out["tcr_alpha_cdr3"] = None
    out["tcr_beta_cdr3"] = None
    out["virus_family"] = None  # filled by build_dataset_v5.py
    out["negative_origin"] = "tested_negative"

    if col_assay_group is not None:
        out["assay_type"] = df[col_assay_group].fillna("").str.strip()
        out["assay_quality_weight"] = (
            df[col_assay_group].fillna("").apply(assign_quality_weight)
        )
    else:
        out["assay_type"] = None
        out["assay_quality_weight"] = 0.7
        logger.warning(
            "Assay Group column not found; defaulting quality weight to 0.7"
        )

    if col_pmid is not None:
        out["reference_pmid"] = (
            df[col_pmid]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", None)
            .replace("nan", None)
        )
    else:
        out["reference_pmid"] = None
        logger.warning("Reference PubMed ID column not found; setting to null")

    out["is_quarantined"] = False  # determined by build_dataset_v5.py

    result = pd.DataFrame(out, index=df.index)
    return result


def deduplicate_against_existing(
    new_df: pd.DataFrame,
    existing_path: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, int]:
    """Drop rows from new_df that match (peptide, hla_allele, virus) in existing."""
    logger.info("Loading existing dataset from %s for deduplication", existing_path)
    try:
        existing = pd.read_csv(existing_path, low_memory=False)
    except Exception as exc:
        logger.error("Failed to read existing dataset %s: %s", existing_path, exc)
        sys.exit(1)

    logger.info("Existing dataset: %d rows", len(existing))

    key_cols = ["peptide", "hla_allele", "virus"]
    # Only deduplicate on columns that exist in both dataframes
    available = [c for c in key_cols if c in existing.columns and c in new_df.columns]
    if not available:
        logger.warning(
            "No common key columns found for deduplication; skipping"
        )
        return new_df, 0

    existing_keys = set(
        existing[available].fillna("").apply(tuple, axis=1)
    )
    new_keys = new_df[available].fillna("").apply(tuple, axis=1)
    mask_dup = new_keys.isin(existing_keys)
    n_dup = int(mask_dup.sum())

    result = new_df[~mask_dup].copy()
    logger.info(
        "Deduplication: %d duplicate rows removed; %d rows remain",
        n_dup,
        len(result),
    )
    return result, n_dup


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ingest_iedb_negatives.py",
        description=(
            "Ingest IEDB bulk export CSV and output a v5-schema-compatible "
            "CSV of negative-outcome T cell assay rows."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to IEDB T cell assay bulk export CSV.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path for output CSV (v5 schema).",
    )
    parser.add_argument(
        "--existing-dataset",
        required=True,
        type=Path,
        metavar="PATH",
        help=(
            "Path to existing dataset CSV (e.g., immunogenicity_dataset_v4.csv). "
            "Rows matching (peptide, hla_allele, virus) in this file are dropped."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print stats (total input rows, rows after each filter stage, "
            "deduplication hits) without writing output."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger("ingest_iedb_negatives")

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1
    if not args.existing_dataset.exists():
        logger.error("Existing dataset not found: %s", args.existing_dataset)
        return 1

    # Load and filter
    raw_df = load_iedb_export(args.input, logger)
    result = filter_rows(raw_df, logger)
    filtered_df, stats, col_epitope, col_allele, col_assay_group, col_antigen, col_pmid, col_description = result

    if len(filtered_df) == 0:
        logger.warning("No rows survived filtering; output will be empty.")

    # Build v5-schema output
    out_df = build_output(
        filtered_df,
        col_epitope,
        col_allele,
        col_assay_group,
        col_antigen,
        col_pmid,
        col_description,
        logger,
    )

    # Deduplication
    out_df, n_dup = deduplicate_against_existing(
        out_df, args.existing_dataset, logger
    )
    stats["deduplication_hits"] = n_dup
    stats["final_output_rows"] = len(out_df)

    # Report
    logger.info("--- Filter stage summary ---")
    for stage, count in stats.items():
        logger.info("  %-45s %d", stage, count)

    if args.dry_run:
        print("\n=== DRY RUN - no output written ===")
        print(f"  raw_input:                          {stats['raw_input']}")
        print(f"  after_human_host_filter:            {stats['after_human_host_filter']}")
        print(f"  after_negative_measure_filter:      {stats['after_negative_measure_filter']}")
        print(f"  after_peptide_filter:               {stats['after_peptide_filter']}")
        print(f"  after_hla_filter:                   {stats['after_hla_filter']}")
        print(f"  deduplication_hits:                 {stats['deduplication_hits']}")
        print(f"  final_output_rows:                  {stats['final_output_rows']}")
        return 0

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    logger.info("Wrote %d rows to %s", len(out_df), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
