"""
ingest_published_panels.py - Ingest manually-curated published peptide panel CSVs
into the v5 dataset schema.

Handles per-study immunogenicity panels from publications (e.g., Bertoletti HBV,
Thimme HCV, Khanna EBV, Stevanovic HPV). Biology-context columns (infection_phase,
antigen_latency_program, assay_context, cross_reactivity_tested, virus_taxon_id)
are set uniformly from CLI arguments since they are panel-level metadata.

Input CSV required columns:
    peptide    8-11 standard amino acid residues
    label      integer 0 (negative) or 1 (positive)

Input CSV optional columns:
    hla_allele   normalized via mhcgnomes; rows with unparseable alleles are dropped
    protein      protein name
    strain       virus strain

Usage:
    python scripts/ingest_published_panels.py \
        --input data/published_panels/bertoletti_hbv.csv \
        --virus HBV \
        --pmid 12520014 \
        --assay-context natural_infection \
        --infection-phase chronic \
        --output data/published_panel_HBV_12520014_v5.csv
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

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

# Valid enum values for CLI-settable biology-context columns.
VALID_ASSAY_CONTEXT = ("vaccine_induced", "natural_infection", "unknown")
VALID_INFECTION_PHASE = ("acute", "chronic", "unknown")
VALID_LATENCY_PROGRAM = ("lytic", "latent-I", "latent-II", "latent-III", "unknown")

# Quality tier to sample-weight mapping (mirrors ingest_iedb_negatives.py).
TIER_WEIGHT: dict[int, float] = {1: 1.0, 2: 0.7, 3: 0.5}


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
# Loading
# ---------------------------------------------------------------------------


def load_panel_csv(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load and minimally validate a published panel CSV."""
    logger.info("Loading panel from %s", path)
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        sys.exit(1)

    df.columns = [str(c).strip() for c in df.columns]

    required = {"peptide", "label"}
    missing = required - set(df.columns)
    if missing:
        logger.error(
            "Input CSV missing required columns: %s. Got: %s",
            sorted(missing),
            list(df.columns),
        )
        sys.exit(1)

    logger.info("Raw input: %d rows x %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_rows(
    df: pd.DataFrame,
    hla_override: str | None,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply validity filters and HLA normalization. Returns (filtered_df, stats)."""
    stats: dict[str, int] = {}
    stats["raw_input"] = len(df)

    df = df.copy()

    # --- Label validation ---
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    mask_bad_label = df["label"].isna() | ~df["label"].isin([0, 1])
    n_bad = int(mask_bad_label.sum())
    if n_bad:
        logger.warning("Dropping %d rows with invalid label values (not 0 or 1)", n_bad)
        df = df[~mask_bad_label].copy()
    df["label"] = df["label"].astype(int)
    stats["after_label_filter"] = len(df)

    # --- Peptide validity: length 8-11, standard AA ---
    mask_pep = df["peptide"].fillna("").str.strip().apply(is_valid_peptide).astype(bool)
    n_bad_pep = int((~mask_pep).sum())
    if n_bad_pep:
        logger.warning(
            "Dropping %d rows with invalid peptide sequences (length or AA)",
            n_bad_pep,
        )
    df = df[mask_pep].copy()
    df["peptide"] = df["peptide"].str.strip()
    stats["after_peptide_filter"] = len(df)

    # --- HLA normalization ---
    if hla_override is not None:
        norm = normalize_hla_allele(hla_override)
        if norm is None:
            logger.error(
                "--hla-allele %r is not a parseable Class I HLA-A/B/C 4-digit allele",
                hla_override,
            )
            sys.exit(1)
        df["hla_allele"] = norm
        logger.info("HLA override: all rows set to %s", norm)
    elif "hla_allele" in df.columns:
        raw_alleles = df["hla_allele"].fillna("").str.strip()
        norm_alleles = raw_alleles.apply(normalize_hla_allele)
        mask_hla = norm_alleles.notna()
        n_bad_hla = int((~mask_hla).sum())
        if n_bad_hla:
            logger.warning(
                "Dropping %d rows with unparseable HLA alleles (not Class I A/B/C 4-digit)",
                n_bad_hla,
            )
        df = df[mask_hla].copy()
        df["hla_allele"] = norm_alleles[mask_hla]
        stats["after_hla_filter"] = len(df)
    else:
        logger.warning(
            "No hla_allele column and no --hla-allele provided; hla_allele will be null"
        )
        df["hla_allele"] = None

    stats["final_rows"] = len(df)
    logger.info(
        "Filter result: %d/%d rows retained",
        stats["final_rows"],
        stats["raw_input"],
    )
    return df, stats


# ---------------------------------------------------------------------------
# Output construction
# ---------------------------------------------------------------------------


def build_output(
    df: pd.DataFrame,
    virus: str,
    pmid: str,
    assay_context: str | None,
    infection_phase: str | None,
    latency_program: str | None,
    cross_reactivity_tested: bool | None,
    virus_taxon_id: int | None,
    assay_quality_tier: int,
    assay_type: str | None,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Construct the v5-schema output DataFrame from a filtered panel."""
    out: dict[str, object] = {}

    out["peptide"] = df["peptide"]
    out["label"] = df["label"].astype(int)
    out["virus"] = virus
    out["protein"] = df["protein"].str.strip() if "protein" in df.columns else ""
    out["strain"] = df["strain"] if "strain" in df.columns else None
    out["hla_allele"] = df["hla_allele"]
    out["source_type"] = "Virus"
    out["database_source"] = "Published"
    out["tcr_alpha_cdr3"] = None
    out["tcr_beta_cdr3"] = None
    out["virus_family"] = None  # assigned downstream by build_dataset_v5.py
    # label=0 rows are tested negatives; label=1 rows have no origin tag.
    out["negative_origin"] = df["label"].apply(
        lambda x: "tested_negative" if int(x) == 0 else None
    )
    out["assay_type"] = assay_type
    out["assay_quality_tier"] = assay_quality_tier
    out["assay_quality_weight"] = TIER_WEIGHT[assay_quality_tier]
    out["reference_pmid"] = pmid
    out["iedb_assay_id"] = None
    out["infection_phase"] = infection_phase
    out["antigen_latency_program"] = latency_program
    out["assay_context"] = assay_context
    out["cross_reactivity_tested"] = cross_reactivity_tested
    out["virus_taxon_id"] = virus_taxon_id
    out["is_quarantined"] = False  # final determination by build_dataset_v5.py

    result = pd.DataFrame(out, index=df.index)
    logger.info(
        "Built v5 output: %d rows (%d pos, %d neg)",
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
        prog="ingest_published_panels.py",
        description=(
            "Ingest manually-curated published peptide panel CSVs into v5 schema. "
            "Biology-context columns are set uniformly from CLI arguments."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to published panel CSV.",
    )
    parser.add_argument(
        "--virus",
        required=True,
        metavar="VIRUS",
        help="Canonical virus name (e.g., HBV, HCV, EBV, HPV).",
    )
    parser.add_argument(
        "--pmid",
        required=True,
        metavar="PMID",
        help="PubMed ID of the source publication.",
    )
    parser.add_argument(
        "--assay-context",
        choices=list(VALID_ASSAY_CONTEXT) + ["null"],
        default=None,
        metavar="CONTEXT",
        help=(
            "Immunization context: vaccine_induced, natural_infection, unknown, "
            "or null (default: null)."
        ),
    )
    parser.add_argument(
        "--infection-phase",
        choices=list(VALID_INFECTION_PHASE) + ["null"],
        default=None,
        metavar="PHASE",
        help="Infection phase: acute, chronic, unknown, or null (default: null).",
    )
    parser.add_argument(
        "--latency-program",
        choices=list(VALID_LATENCY_PROGRAM) + ["null"],
        default=None,
        metavar="PROGRAM",
        help=(
            "EBV antigen latency program: lytic, latent-I, latent-II, latent-III, "
            "unknown, or null (default: null)."
        ),
    )
    cr_group = parser.add_mutually_exclusive_group()
    cr_group.add_argument(
        "--cross-reactivity-tested",
        dest="cross_reactivity_tested",
        action="store_const",
        const=True,
        default=None,
        help="HPV only: cross-reactivity between HPV16/18 was explicitly tested.",
    )
    cr_group.add_argument(
        "--no-cross-reactivity-tested",
        dest="cross_reactivity_tested",
        action="store_const",
        const=False,
        help="HPV only: cross-reactivity between HPV16/18 was assumed, not tested.",
    )
    parser.add_argument(
        "--virus-taxon-id",
        type=int,
        default=None,
        metavar="TAXON_ID",
        help="NCBI Taxonomy ID for the source virus (e.g., EBV=10376).",
    )
    parser.add_argument(
        "--hla-allele",
        default=None,
        metavar="HLA",
        help=(
            "Single HLA allele for panels where all rows share one allele. "
            "Overrides the hla_allele column if present in the input CSV."
        ),
    )
    parser.add_argument(
        "--assay-quality-tier",
        type=int,
        choices=[1, 2, 3],
        default=1,
        metavar="TIER",
        help=(
            "Assay quality tier: 1=direct ex vivo functional (default, weight 1.0), "
            "2=indirect (0.7), 3=expanded-culture (0.5)."
        ),
    )
    parser.add_argument(
        "--assay-type",
        default=None,
        metavar="TYPE",
        help="Free-text assay type description (e.g., 'T cell IFN-gamma ELISpot').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Output CSV path. Default: data/published_panel_{virus}_{pmid}_v5.csv"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print filter stats without writing any output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Deterministic output: seed both RNGs at entry (data-ingest rule 1).
    random.seed(42)
    np.random.seed(42)

    args = parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger("ingest_published_panels")

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1

    # Resolve "null" string to Python None for enum-valued args.
    assay_context = (
        None if args.assay_context in (None, "null") else args.assay_context
    )
    infection_phase = (
        None if args.infection_phase in (None, "null") else args.infection_phase
    )
    latency_program = (
        None if args.latency_program in (None, "null") else args.latency_program
    )

    raw_df = load_panel_csv(args.input, logger)
    filtered_df, stats = filter_rows(raw_df, args.hla_allele, logger)

    if len(filtered_df) == 0:
        logger.warning(
            "0 rows survived filtering (raw_input=%d). "
            "--dry-run will show stats; live run will abort.",
            stats.get("raw_input", 0),
        )

    out_df = build_output(
        filtered_df,
        virus=args.virus,
        pmid=args.pmid,
        assay_context=assay_context,
        infection_phase=infection_phase,
        latency_program=latency_program,
        cross_reactivity_tested=args.cross_reactivity_tested,
        virus_taxon_id=args.virus_taxon_id,
        assay_quality_tier=args.assay_quality_tier,
        assay_type=args.assay_type,
        logger=logger,
    )

    # Sort deterministically so re-runs on the same input are byte-stable.
    sort_cols = [c for c in ("peptide", "hla_allele", "label") if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols).reset_index(drop=True)

    n_pos = int((out_df["label"] == 1).sum())
    n_neg = int((out_df["label"] == 0).sum())

    logger.info("--- Filter summary ---")
    for key, val in stats.items():
        logger.info("  %-35s %d", key, val)

    if args.dry_run:
        print("\n=== DRY RUN - no output written ===")
        for key, val in stats.items():
            print(f"  {key + ':':<35} {val}")
        print(f"  {'positives:':<35} {n_pos}")
        print(f"  {'negatives:':<35} {n_neg}")
        return 0

    if len(out_df) == 0:
        logger.error(
            "Aborting: 0 rows survived filtering (raw_input=%d). "
            "Verify peptide and label columns are populated in %s "
            "and that all peptides are 8-11 AA standard residues.",
            stats.get("raw_input", 0),
            args.input,
        )
        return 1

    # Resolve output path.
    output_path = args.output
    if output_path is None:
        output_path = Path(f"data/published_panel_{args.virus}_{args.pmid}_v5.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(out_df), output_path)

    # Provenance sidecar (data-ingest rule 2).
    provenance_path = output_path.with_name(output_path.stem + "_provenance.json")
    provenance = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "input_file": str(args.input),
        "input_sha256": compute_file_sha256(args.input),
        "virus": args.virus,
        "pmid": args.pmid,
        "assay_context": assay_context,
        "infection_phase": infection_phase,
        "antigen_latency_program": latency_program,
        "cross_reactivity_tested": args.cross_reactivity_tested,
        "virus_taxon_id": args.virus_taxon_id,
        "assay_quality_tier": args.assay_quality_tier,
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
