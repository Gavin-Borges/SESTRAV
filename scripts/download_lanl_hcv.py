"""
download_lanl_hcv.py - Parse a manually-downloaded LANL HCV Immunology Database
export into the v5 dataset schema.

The LANL HCV Immunology Database (hcv.lanl.gov) does not support programmatic
bulk download; the file must be downloaded manually from the web interface:

  1. Navigate to hcv.lanl.gov > Immunology > T Cell Epitopes > Search
  2. Export all T-cell assay records as tab-delimited or CSV;
     save to data/lanl_hcv_tcell_export.tsv

IMPORTANT: The LANL HCV database is primarily a curated POSITIVE-epitope
repository. Third-party analyses (e.g. Repitope, PMC6477061) treat all LANL
HCV entries as immunogenic a priori, and an independent extract of ~195 LANL
HCV entries showed zero negative qualitative-measure values.

Run --inspect immediately after download. If the export has no negative
qualitative measure column (or all values are positive), the output will be
empty and IEDB is the correct source for HCV negatives (see
ingest_iedb_negatives.py). The --col-* override flags remain available in
case the actual export format differs from this expectation.

Run --inspect on the downloaded file first to confirm actual column names, then
re-run with the correct column-map overrides if the defaults do not match:

  python scripts/download_lanl_hcv.py --input data/lanl_hcv_tcell_export.tsv --inspect

  python scripts/download_lanl_hcv.py \\
      --input data/lanl_hcv_tcell_export.tsv \\
      --output data/lanl_hcv_negatives_v5.csv \\
      [--col-epitope "Sequence"] \\
      [--col-assay-result "Assay Response"] \\
      [--col-hla "MHC Restriction"] \\
      [--col-pmid "PMID"] \\
      [--col-assay-type "Assay Type"] \\
      [--dry-run]

If the export yields 0 rows, use IEDB bulk export for HCV negatives instead.
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

# Known-negative qualitative measure strings for LANL HCV exports.
# LANL uses "Negative" / "negative" for non-responding assays.
NEGATIVE_MEASURES = {"negative", "neg", "no response", "non-responder"}

# Candidate column name variants tried in order (case-insensitive).
# Add additional variants from --inspect output if the default set misses.
CANDIDATE_EPITOPE = (
    "Sequence",
    "Epitope",
    "Epitope Name",
    "Peptide",
    "peptide",
    "sequence",
)
CANDIDATE_HLA = (
    "MHC Restriction",
    "HLA",
    "MHC",
    "Restriction",
    "Allele",
    "mhc restriction",
    "hla allele",
)
CANDIDATE_RESULT = (
    "Qualitative Measure",
    "Assay Response",
    "Response",
    "Result",
    "assay response",
    "qualitative measure",
    "result",
)
CANDIDATE_PMID = (
    "PMID",
    "PubMed ID",
    "Reference PMID",
    "Ref PMID",
    "pmid",
    "pubmed id",
    "reference",
)
CANDIDATE_PROTEIN = (
    "Protein",
    "Antigen",
    "protein",
    "antigen",
)
CANDIDATE_ASSAY = (
    "Assay",
    "Assay Type",
    "Assay Group",
    "assay",
    "assay type",
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
    }

    # Required fields
    missing = [k for k in ("epitope", "hla", "result") if mapping[k] is None]
    if missing:
        logger.error(
            "Could not resolve required columns %s. Run --inspect to see available "
            "column names, then use --col-* overrides.",
            missing,
        )
        logger.error("Available columns: %s", cols)
        sys.exit(1)

    logger.info("Column mapping: %s", mapping)
    return mapping


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_lanl_export(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load a LANL HCV export file (auto-detects tab vs comma separation)."""
    logger.info("Loading LANL HCV export from %s", path)
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


def filter_rows(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter to HCV negative assay rows with valid peptides and Class I HLA."""
    stats: dict[str, int] = {"raw_input": len(df)}

    # --- Filter 1: Negative assay result ---
    result_col = cast(str, mapping["result"])
    result_series = df[result_col].fillna("").str.strip().str.lower()
    mask_neg = result_series.isin(NEGATIVE_MEASURES)
    df = df[mask_neg].copy()
    stats["after_negative_filter"] = len(df)
    logger.info(
        "After negative assay filter: %d rows (removed %d)",
        len(df),
        stats["raw_input"] - len(df),
    )

    # --- Filter 2: Peptide validity (length 8-11, standard AA) ---
    epitope_col = cast(str, mapping["epitope"])
    peptides = df[epitope_col].fillna("").str.strip()
    mask_pep = peptides.apply(is_valid_peptide)
    df = df[mask_pep].copy()
    df["_peptide"] = df[epitope_col].str.strip()
    stats["after_peptide_filter"] = len(df)
    logger.info(
        "After peptide validity filter (8-11 aa, standard AA): %d rows (removed %d)",
        len(df),
        stats["after_negative_filter"] - len(df),
    )

    # --- Filter 3: Class I HLA-A/B/C 4-digit normalization via mhcgnomes ---
    hla_col = cast(str, mapping["hla"])
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
            "pmid": df[pmid_col].fillna("").astype(str).str.strip() if pmid_col is not None else "",
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
    """Construct the v5-schema output DataFrame from filtered LANL rows."""
    epitope_col = cast(str, mapping["epitope"])
    hla_col = cast(str, mapping["hla"])
    pmid_col = mapping.get("pmid")
    protein_col = mapping.get("protein")
    assay_col = mapping.get("assay")

    out: dict[str, object] = {}
    out["peptide"] = df["_peptide"]
    out["label"] = 0
    out["virus"] = "HCV"
    out["protein"] = df[protein_col].fillna("").str.strip() if protein_col else ""
    out["strain"] = None
    out["hla_allele"] = df["_hla_norm"]
    out["source_type"] = "Virus"
    out["database_source"] = "LANL-HCV"
    out["tcr_alpha_cdr3"] = None
    out["tcr_beta_cdr3"] = None
    out["virus_family"] = "Flaviviridae"
    out["negative_origin"] = "tested_negative"

    if assay_col is not None:
        assay_series = df[assay_col].fillna("").str.strip()
        out["assay_type"] = assay_series
        # LANL assay descriptions are free text; default to tier 2.
        out["assay_quality_tier"] = 2
        out["assay_quality_weight"] = 0.7
    else:
        out["assay_type"] = None
        out["assay_quality_tier"] = 2
        out["assay_quality_weight"] = 0.7

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
    out["virus_taxon_id"] = 11103  # NCBI:txid11103 - Hepatitis C virus
    out["is_quarantined"] = False

    result = pd.DataFrame(out, index=df.index)
    logger.info("Built output: %d rows", len(result))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download_lanl_hcv.py",
        description=(
            "Parse a manually-downloaded LANL HCV Immunology Database export "
            "into v5 schema. Run --inspect first to verify column names."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to downloaded LANL HCV T-cell export (TSV or CSV).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output CSV path (default: data/lanl_hcv_negatives_v5.csv).",
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
        help="Column name for the assay qualitative result (positive/negative).",
    )
    parser.add_argument(
        "--col-pmid", default=None, metavar="COL", help="Column name for the PubMed ID / reference."
    )
    parser.add_argument(
        "--col-protein", default=None, metavar="COL", help="Column name for the viral protein name."
    )
    parser.add_argument(
        "--col-assay-type",
        default=None,
        metavar="COL",
        help="Column name for the assay type/group.",
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
    logger = logging.getLogger("download_lanl_hcv")

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1

    df = load_lanl_export(args.input, logger)

    # --inspect mode: show columns and sample then exit.
    if args.inspect:
        print("\n=== LANL HCV Export - Column Inspector ===")
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

    mapping = resolve_columns(df, overrides, logger)
    filtered_df, stats = filter_rows(df, mapping, logger)

    if len(filtered_df) == 0:
        logger.warning("No rows survived filtering; check --inspect output.")

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
        return 0

    output_path = args.output or Path("data/lanl_hcv_negatives_v5.csv")
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
        "source": "LANL HCV Immunology Database (hcv.lanl.gov)",
        "column_mapping": mapping,
        "filter_stats": stats,
        "output_file": str(output_path),
        "output_checksum_sha256": compute_file_sha256(output_path),
    }
    with open(provenance_path, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)
    logger.info("Wrote provenance sidecar: %s", provenance_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
