"""
ingest_lanl_hiv.py - Parse a manually-downloaded LANL HIV Molecular Immunology
Database export into the v5 dataset schema.

The LANL HIV Molecular Immunology Database (hiv.lanl.gov/content/immunology/)
does not support programmatic bulk download; the file must be downloaded manually
from the web interface:

  1. Navigate to hiv.lanl.gov > Immunology > CTL/CD8+ T Cell Responses
  2. Search: Organism=HIV-1, Assay type=T cell, export all records as
     tab-delimited or CSV; save to data/lanl_hiv_tcell_export.tsv

IMPORTANT: LANL HIV is primarily a curated POSITIVE-epitope repository.
The "Best Defined CTL Epitopes" table contains only confirmed immunogenic
peptides. For training negatives the IEDB API is the primary source
(fetch_iedb_tcell.py --virus HIV). This script supplements IEDB with:
  1. Any tested-negative records from comprehensive per-protein assay studies
  2. Confirmed HIV-1 CTL epitopes not yet represented in IEDB (positives)

By default the script ingests ALL assay records (positive and negative).
Use --negatives-only to restrict to tested-negative rows only.

Run --inspect immediately after download to confirm actual column names, then
re-run with --col-* overrides if the defaults do not match your export.

  python scripts/ingest_lanl_hiv.py --input data/lanl_hiv_tcell_export.tsv --inspect

  python scripts/ingest_lanl_hiv.py \\
      --input data/lanl_hiv_tcell_export.tsv \\
      --output data/lanl_hiv_v5.csv \\
      [--negatives-only] \\
      [--col-epitope "Sequence"] \\
      [--col-assay-result "Assay Response"] \\
      [--col-hla "HLA"] \\
      [--col-pmid "PMID"] \\
      [--dry-run]
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from scripts.ingest_iedb_negatives import (
    compute_file_sha256,
    get_git_sha,
    is_valid_peptide,
    normalize_hla_allele,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NCBI taxonomy ID for Human immunodeficiency virus 1
HIV1_TAXON_ID = 11676

# Qualitative measure strings recognized as confirmed-negative.
# LANL HIV uses "Negative" / "No response" in comprehensive assay studies.
NEGATIVE_MEASURES = {
    "negative",
    "neg",
    "no response",
    "non-responder",
    "not detected",
    "not immunogenic",
    "no t cell response",
}

# Qualitative measure strings recognized as confirmed-positive.
POSITIVE_MEASURES = {
    "positive",
    "pos",
    "yes",
    "immunogenic",
    "response",
    "t cell response",
    "recognized",
    "confirmed",
    "reactive",
}

# Column name candidates tried in order (case-insensitive).
# LANL HIV exports use varying column headers across search result types.
CANDIDATE_EPITOPE = (
    "Optimal Sequence",
    "Sequence",
    "Epitope",
    "Epitope Name",
    "Peptide",
    "Linear Sequence",
    "optimal sequence",
    "sequence",
    "peptide",
)
CANDIDATE_HLA = (
    "HLA",
    "HLA Type",
    "HLA Restriction",
    "MHC Restriction",
    "MHC",
    "Allele",
    "hla",
    "hla type",
    "mhc restriction",
)
CANDIDATE_RESULT = (
    "Qualitative Measure",
    "Assay Result",
    "Assay Response",
    "Response",
    "Result",
    "Immunogenic",
    "qualitative measure",
    "assay result",
    "response",
)
CANDIDATE_PMID = (
    "PMID",
    "PubMed ID",
    "Reference PMID",
    "Ref PMID",
    "PubMed",
    "pmid",
    "pubmed id",
    "reference",
)
CANDIDATE_PROTEIN = (
    "Protein",
    "Gene",
    "Subprotein",
    "Antigen",
    "protein",
    "gene",
)
CANDIDATE_ASSAY = (
    "Assay",
    "Assay Type",
    "Assay Group",
    "Method",
    "assay",
    "assay type",
    "method",
)
CANDIDATE_SUBTYPE = (
    "Subtype",
    "Clade",
    "HIV Subtype",
    "Strain",
    "subtype",
    "clade",
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------


def _find_col(columns: list[str], *candidates: str) -> str | None:
    """Case-insensitive column lookup; tries candidates in order."""
    col_lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in col_lower:
            return col_lower[candidate.lower()]
    return None


def resolve_columns(
    df: pd.DataFrame,
    overrides: dict[str, str],
    logger: logging.Logger,
) -> dict[str, str | None]:
    """Resolve logical field names to actual column names.

    ``overrides`` maps logical name -> explicit column name (from CLI).
    Falls back to CANDIDATE_* lists when override is not provided.
    """
    cols = list(df.columns)

    def _resolve(logical: str, candidates: tuple[str, ...]) -> str | None:
        if logical in overrides and overrides[logical]:
            if overrides[logical] in cols:
                return overrides[logical]
            logger.warning(
                "Override column %r for %s not found in file; trying defaults",
                overrides[logical],
                logical,
            )
        return _find_col(cols, *candidates)

    mapping = {
        "epitope": _resolve("epitope", CANDIDATE_EPITOPE),
        "hla": _resolve("hla", CANDIDATE_HLA),
        "result": _resolve("result", CANDIDATE_RESULT),
        "pmid": _resolve("pmid", CANDIDATE_PMID),
        "protein": _resolve("protein", CANDIDATE_PROTEIN),
        "assay": _resolve("assay", CANDIDATE_ASSAY),
        "subtype": _resolve("subtype", CANDIDATE_SUBTYPE),
    }

    # Required fields: epitope and hla are always required.
    # result is required unless ingesting from a Best-Defined-Epitopes table
    # (all-positive, no qualitative measure column).
    missing_required = [k for k in ("epitope", "hla") if mapping[k] is None]
    if missing_required:
        logger.error(
            "Could not resolve required columns %s. Run --inspect to see available "
            "column names, then use --col-* overrides.",
            missing_required,
        )
        logger.error("Available columns: %s", cols)
        sys.exit(1)

    if mapping["result"] is None:
        logger.warning(
            "No qualitative-measure column found. Will treat all records as "
            "POSITIVE (Best-Defined-Epitopes table mode). "
            "Use --col-assay-result to specify a result column if one exists."
        )

    logger.info("Column mapping: %s", mapping)
    return mapping


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_lanl_export(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load a LANL HIV export file (auto-detects tab vs comma separation)."""
    logger.info("Loading LANL HIV export from %s", path)
    suffix = path.suffix.lower()
    sep = "\t" if suffix in (".tsv", ".txt") else ","
    try:
        df = pd.read_csv(path, sep=sep, low_memory=False)
        # If comma-split produced a single column, retry with tab.
        if len(df.columns) == 1 and sep == ",":
            logger.info("Single-column result with comma sep; retrying with tab")
            df = pd.read_csv(path, sep="\t", low_memory=False)
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        sys.exit(1)

    df.columns = [str(c).strip() for c in df.columns]
    logger.info("Raw input: %d rows x %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def _parse_label(result_str: str) -> int | None:
    """Map a qualitative measure string to binary label.

    Returns:
        1 for positive / immunogenic
        0 for negative / non-responding
        None for ambiguous / unknown (row will be skipped)
    """
    s = result_str.strip().lower()
    if s in POSITIVE_MEASURES or s.startswith("positive"):
        return 1
    if s in NEGATIVE_MEASURES or s.startswith("negative"):
        return 0
    return None


def filter_rows(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    negatives_only: bool,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter to HIV-1 assay rows with valid peptides, Class I HLA, and label."""
    stats: dict[str, int] = {"raw_input": len(df)}

    result_col = mapping.get("result")
    epitope_col = cast(str, mapping["epitope"])
    hla_col = cast(str, mapping["hla"])

    # --- Filter 1: Label assignment ---
    if result_col is not None:
        labels = df[result_col].fillna("").astype(str).str.strip().apply(_parse_label)
        if negatives_only:
            mask_label = labels == 0
        else:
            mask_label = labels.notna()
        df = df[mask_label].copy()
        df["_label"] = labels[mask_label].astype(int)
    else:
        # No result column: treat all as positive (Best-Defined-Epitopes table)
        df = df.copy()
        df["_label"] = 1

    stats["after_label_filter"] = len(df)
    logger.info(
        "After label filter (%s): %d rows (removed %d)",
        "negatives only" if negatives_only else "all labeled",
        len(df),
        stats["raw_input"] - len(df),
    )
    if len(df) > 0:
        pos = int((df["_label"] == 1).sum())
        neg = int((df["_label"] == 0).sum())
        logger.info("  Label breakdown: %d positive, %d negative", pos, neg)

    # --- Filter 2: Peptide validity (length 8-11, standard AA) ---
    peptides = df[epitope_col].fillna("").str.strip()
    mask_pep = peptides.apply(is_valid_peptide)
    df = df[mask_pep].copy()
    df["_peptide"] = df[epitope_col].str.strip().str.upper()
    stats["after_peptide_filter"] = len(df)
    logger.info(
        "After peptide validity filter (8-11 aa, standard AA): %d rows (removed %d)",
        len(df),
        stats["after_label_filter"] - len(df),
    )

    # --- Filter 3: Class I HLA-A/B/C 4-digit normalization via mhcgnomes ---
    raw_alleles = df[hla_col].fillna("").str.strip()
    norm_alleles = raw_alleles.apply(normalize_hla_allele)
    mask_hla = norm_alleles.notna()
    df = df[mask_hla].copy()
    df["_hla_norm"] = norm_alleles[mask_hla]
    stats["after_hla_filter"] = len(df)
    logger.info(
        "After Class I HLA-A/B/C filter: %d rows (removed %d)",
        len(df),
        stats["after_peptide_filter"] - len(df),
    )

    # --- Intra-export deduplication on (peptide, hla, pmid) ---
    pmid_col = mapping.get("pmid")
    dedup_keys = pd.DataFrame(
        {
            "peptide": df["_peptide"],
            "hla": df["_hla_norm"],
            "pmid": (
                df[pmid_col].fillna("").astype(str).str.strip() if pmid_col is not None else ""
            ),
        }
    )
    before_dedup = len(df)
    keep = ~dedup_keys.duplicated(keep="first")
    df = df[keep.to_numpy()].copy()
    stats["intra_export_duplicates_removed"] = before_dedup - len(df)
    stats["after_dedup"] = len(df)
    logger.info(
        "Dedup: removed %d duplicates; %d rows remain",
        stats["intra_export_duplicates_removed"],
        len(df),
    )

    stats["final_rows"] = len(df)
    return df, stats


# ---------------------------------------------------------------------------
# Output construction
# ---------------------------------------------------------------------------


def build_output(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Construct the v5-schema output DataFrame from filtered LANL HIV rows."""
    epitope_col = cast(str, mapping["epitope"])
    hla_col = cast(str, mapping["hla"])
    pmid_col = mapping.get("pmid")
    protein_col = mapping.get("protein")
    assay_col = mapping.get("assay")
    subtype_col = mapping.get("subtype")

    out: dict[str, object] = {}
    out["peptide"] = df["_peptide"]
    out["label"] = df["_label"]
    out["virus"] = "HIV-1"
    out["protein"] = df[protein_col].fillna("").str.strip() if protein_col else ""
    # Use subtype as the strain field (HIV-1 A/B/C/D, etc.)
    out["strain"] = df[subtype_col].fillna("").str.strip() if subtype_col else None
    out["hla_allele"] = df["_hla_norm"]
    out["source_type"] = "Virus"
    out["database_source"] = "LANL-HIV"
    out["tcr_alpha_cdr3"] = None
    out["tcr_beta_cdr3"] = None
    out["virus_family"] = "Retroviridae"
    out["negative_origin"] = df["_label"].map({0: "tested_negative", 1: "iedb_positive"})

    if assay_col is not None:
        assay_series = df[assay_col].fillna("").str.strip()
        out["assay_type"] = assay_series
        # LANL assay descriptions are free text; default to tier 2.
        out["assay_quality_weight"] = 0.7
        out["assay_quality_tier"] = 2
    else:
        out["assay_type"] = None
        out["assay_quality_weight"] = 0.7
        out["assay_quality_tier"] = 2

    out["reference_pmid"] = (
        df[pmid_col].fillna("").astype(str).str.strip().replace("", None)
        if pmid_col is not None
        else None
    )
    out["iedb_assay_id"] = None
    out["infection_phase"] = None
    out["antigen_latency_program"] = None
    out["assay_context"] = "natural_infection"
    out["cross_reactivity_tested"] = None
    out["virus_taxon_id"] = HIV1_TAXON_ID
    out["is_quarantined"] = False

    result = pd.DataFrame(out, index=df.index)
    logger.info(
        "Built output: %d rows (%d positive, %d negative)",
        len(result),
        int((result["label"] == 1).sum()),
        int((result["label"] == 0).sum()),
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ingest_lanl_hiv.py",
        description=(
            "Parse a manually-downloaded LANL HIV Molecular Immunology Database "
            "export into v5 schema. Run --inspect first to verify column names."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to downloaded LANL HIV T-cell export (TSV or CSV).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output CSV path (default: data/lanl_hiv_v5.csv).",
    )
    parser.add_argument(
        "--negatives-only",
        action="store_true",
        help=(
            "Restrict output to tested-negative records only. "
            "By default, all labeled records (positive and negative) are emitted."
        ),
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help=(
            "Print column names and sample rows from the input file, then exit. "
            "Use this to identify the correct column names before running the "
            "full ingest."
        ),
    )
    # Column override arguments
    parser.add_argument(
        "--col-epitope",
        default=None,
        metavar="COL",
        help="Column name for the peptide/epitope sequence.",
    )
    parser.add_argument(
        "--col-hla", default=None, metavar="COL", help="Column name for the MHC/HLA restriction."
    )
    parser.add_argument(
        "--col-assay-result",
        default=None,
        metavar="COL",
        help="Column name for the assay qualitative result.",
    )
    parser.add_argument(
        "--col-pmid", default=None, metavar="COL", help="Column name for the PubMed ID / reference."
    )
    parser.add_argument(
        "--col-protein",
        default=None,
        metavar="COL",
        help="Column name for the viral protein/gene name.",
    )
    parser.add_argument(
        "--col-assay-type",
        default=None,
        metavar="COL",
        help="Column name for the assay type/group.",
    )
    parser.add_argument(
        "--col-subtype",
        default=None,
        metavar="COL",
        help="Column name for the HIV-1 subtype/clade.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print filter stats without writing output."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    random.seed(42)
    np.random.seed(42)

    args = parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger("ingest_lanl_hiv")

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1

    df = load_lanl_export(args.input, logger)

    # --inspect mode: show columns and sample then exit.
    if args.inspect:
        print("\n=== LANL HIV Export - Column Inspector ===")
        print(f"Rows: {len(df)}")
        print(f"Columns ({len(df.columns)}):")
        for i, col in enumerate(df.columns):
            sample_vals = df[col].dropna().head(3).tolist()
            print(f"  [{i:>3}] {col!r:<40} sample: {sample_vals}")
        print("\nRe-run without --inspect (or with --col-* overrides) to proceed.")
        return 0

    # Build column override mapping
    overrides: dict[str, str] = {}
    if args.col_epitope:
        overrides["epitope"] = args.col_epitope
    if args.col_hla:
        overrides["hla"] = args.col_hla
    if args.col_assay_result:
        overrides["result"] = args.col_assay_result
    if args.col_pmid:
        overrides["pmid"] = args.col_pmid
    if args.col_protein:
        overrides["protein"] = args.col_protein
    if args.col_assay_type:
        overrides["assay"] = args.col_assay_type
    if args.col_subtype:
        overrides["subtype"] = args.col_subtype

    mapping = resolve_columns(df, overrides, logger)
    filtered_df, stats = filter_rows(df, mapping, args.negatives_only, logger)

    if len(filtered_df) == 0:
        logger.warning(
            "No rows survived filtering. If this is a Best-Defined-Epitopes "
            "table, all entries are positive; do not use --negatives-only. "
            "Run --inspect to verify column names and qualitative-measure values."
        )

    out_df = build_output(filtered_df, mapping, logger)

    # Sort deterministically (data-ingest rule 1)
    sort_cols = [c for c in ("peptide", "hla_allele", "reference_pmid") if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols).reset_index(drop=True)

    logger.info("--- Filter summary ---")
    for key, val in stats.items():
        logger.info("  %-45s %d", key, val)

    if args.dry_run:
        print("\n=== DRY RUN - no output written ===")
        for key, val in stats.items():
            print(f"  {key + ':':<45} {val}")
        print(
            f"\n  Positive rows: {int((out_df['label'] == 1).sum())}"
            f"  Negative rows: {int((out_df['label'] == 0).sum())}"
        )
        return 0

    output_path = args.output or Path("data/lanl_hiv_v5.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(out_df), output_path)

    # Provenance sidecar (data-ingest rule 2)
    provenance_path = output_path.with_name(output_path.stem + "_provenance.json")
    provenance = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "input_file": str(args.input),
        "input_sha256": compute_file_sha256(args.input),
        "source": "LANL HIV Molecular Immunology Database (hiv.lanl.gov)",
        "negatives_only": args.negatives_only,
        "column_mapping": mapping,
        "filter_stats": stats,
        "output_file": str(output_path),
        "positive_count": int((out_df["label"] == 1).sum()),
        "negative_count": int((out_df["label"] == 0).sum()),
        "output_checksum_sha256": compute_file_sha256(output_path),
    }
    with open(provenance_path, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)
    logger.info("Wrote provenance sidecar: %s", provenance_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
