"""merge_iedb_api_negatives.py

Bridge the two IEDB data pipelines into a single negatives file for
build_dataset_v5.py.

Background
----------
Two separate IEDB pipelines exist but are not connected:

  Pipeline A: fetch_iedb_tcell.py -> data/iedb/*_tcell.csv
    - REST API, paginates all viruses
    - Contains label=1 (positives) and label=0 (negatives)
    - 11 columns, assay_quality_weight as float

  Pipeline B: ingest_iedb_negatives.py -> data/iedb_negatives_v5.csv
    - IEDB bulk export CSV download, negatives only
    - 23-column schema (richer, with assay_quality_tier, iedb_assay_id, etc.)

build_dataset_v5.py reads only Pipeline B via --iedb-negatives.  This script:
  1. Reads all data/iedb/*_tcell.csv, filters to label=0 rows.
  2. Normalizes to the 23-column Pipeline B schema.
  3. Deduplicates within the API set on (peptide, hla_allele), keeping the
     highest-quality assay per pair.
  4. Removes rows whose (peptide, hla_allele) already appear in the existing
     Pipeline B file.
  5. Concatenates existing + net-new and writes the merged output.

Usage
-----
    python scripts/merge_iedb_api_negatives.py \\
        --api-dir data/iedb \\
        --existing data/iedb_negatives_v5.csv \\
        --output data/iedb_negatives_v5_merged.csv

    # Dry run: print stats without writing
    python scripts/merge_iedb_api_negatives.py \\
        --api-dir data/iedb \\
        --existing data/iedb_negatives_v5.csv \\
        --output data/iedb_negatives_v5_merged.csv \\
        --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ssl_fix  # noqa: F401, E402 - patch SSL before any network calls
from _dataset_utils import git_sha, write_provenance

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEDUP_KEY = ["peptide", "hla_allele"]

# Columns expected by build_dataset_v5.py / ingest_iedb_negatives.py output
SCHEMA_COLUMNS = [
    "peptide",
    "label",
    "hla_allele",
    "virus",
    "protein",
    "strain",
    "source_type",
    "database_source",
    "tcr_alpha_cdr3",
    "tcr_beta_cdr3",
    "virus_family",
    "negative_origin",
    "assay_type",
    "assay_quality_tier",
    "assay_quality_weight",
    "reference_pmid",
    "iedb_assay_id",
    "infection_phase",
    "antigen_latency_program",
    "assay_context",
    "cross_reactivity_tested",
    "virus_taxon_id",
    "is_quarantined",
]

# Viruses represented in our 6-target scope (informational only; all viruses pass through)
TARGET_VIRUSES = {"EBV", "HPV", "HPV16", "HPV18", "HBV", "HCV", "HIV-1", "SARS-CoV-2"}

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(levelname)s  %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality tier mapping
# ---------------------------------------------------------------------------


def _quality_tier(weight: float) -> int:
    """Map assay_quality_weight float to integer quality tier (1=best, 3=weakest).

    Thresholds align with the three-tier system in ingest_iedb_negatives.py:
      Tier 1 (>=0.9): direct ex vivo functional (cytotoxicity, IFN-g ELISpot, tetramer).
      Tier 2 (>=0.6): indirect functional (proliferation, activation, degranulation).
      Tier 3 (<0.6): qualitative binding or other indirect readouts.
    """
    if weight >= 0.9:
        return 1
    if weight >= 0.6:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Load and normalize API negatives
# ---------------------------------------------------------------------------


def _load_api_negatives(api_dir: Path) -> pd.DataFrame:
    """Read all *_tcell.csv in api_dir, return label=0 rows only."""
    frames: list[pd.DataFrame] = []
    csv_files = sorted(api_dir.glob("*_tcell.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_tcell.csv files found in {api_dir}")

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as exc:
            logger.warning("Skipping %s: %s", csv_path.name, exc)
            continue
        if "label" not in df.columns:
            logger.warning("Skipping %s: no 'label' column", csv_path.name)
            continue
        negs = df[df["label"] == 0].copy()
        if len(negs) > 0:
            logger.info("  %s: %d negatives (of %d rows)", csv_path.name, len(negs), len(df))
            frames.append(negs)
        else:
            logger.info("  %s: 0 negatives - skipping", csv_path.name)

    if not frames:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _normalize_to_schema(raw: pd.DataFrame) -> pd.DataFrame:
    """Map 11-column API frame to the 23-column negatives schema."""
    weights = pd.to_numeric(
        raw.get("assay_quality_weight", pd.Series(0.5, index=raw.index)),
        errors="coerce",
    ).fillna(0.5)

    out = pd.DataFrame(index=raw.index)
    out["peptide"] = raw["peptide"].astype(str).str.strip().str.upper()
    out["label"] = 0
    out["hla_allele"] = raw.get("hla_allele", pd.Series("", index=raw.index)).fillna("").astype(str)
    out["virus"] = raw.get("virus", pd.Series("", index=raw.index)).fillna("").astype(str)
    out["protein"] = raw.get("protein", pd.Series("", index=raw.index)).fillna("").astype(str)
    out["strain"] = raw.get("strain", pd.Series("", index=raw.index)).fillna("").astype(str)
    out["source_type"] = (
        raw.get("source_type", pd.Series("Virus", index=raw.index)).fillna("Virus").astype(str)
    )
    out["database_source"] = "IEDB"
    out["tcr_alpha_cdr3"] = np.nan
    out["tcr_beta_cdr3"] = np.nan
    out["virus_family"] = np.nan
    out["negative_origin"] = "iedb_api"
    out["assay_type"] = raw.get("assay_type", pd.Series("", index=raw.index)).fillna("").astype(str)
    out["assay_quality_tier"] = weights.map(_quality_tier).astype(int)
    out["assay_quality_weight"] = weights
    pmid = raw.get("reference_pmid", pd.Series("", index=raw.index)).fillna("").astype(str)
    out["reference_pmid"] = pmid
    out["iedb_assay_id"] = np.nan
    out["infection_phase"] = np.nan
    out["antigen_latency_program"] = np.nan
    out["assay_context"] = np.nan
    out["cross_reactivity_tested"] = np.nan
    out["virus_taxon_id"] = np.nan
    out["is_quarantined"] = False

    return out[SCHEMA_COLUMNS]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _dedup_within_api(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate API negatives on (peptide, hla_allele).

    Among duplicates keeps the row with the highest assay_quality_weight.
    Deterministic: ties broken by sort order of remaining columns.
    """
    n_before = len(df)
    df_sorted = df.sort_values("assay_quality_weight", ascending=False)
    df_deduped = df_sorted.drop_duplicates(subset=DEDUP_KEY, keep="first").reset_index(drop=True)
    n_dropped = n_before - len(df_deduped)
    if n_dropped > 0:
        logger.info(
            "  Within-API dedup: removed %d duplicate (peptide, hla_allele) pairs", n_dropped
        )
    return df_deduped


def _dedup_against_existing(new: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Remove rows from new whose (peptide, hla_allele) already appear in existing."""
    if existing.empty:
        return new

    existing_keys = set(
        zip(
            existing["peptide"].astype(str).str.upper(),
            existing["hla_allele"].astype(str).str.upper(),
        )
    )
    new_keys = list(
        zip(
            new["peptide"].astype(str).str.upper(),
            new["hla_allele"].astype(str).str.upper(),
        )
    )
    mask_new = pd.Series([k not in existing_keys for k in new_keys], index=new.index)
    n_overlap = (~mask_new).sum()
    if n_overlap > 0:
        logger.info(
            "  Cross-dedup against existing: removed %d rows already present "
            "in Pipeline B (by peptide+hla_allele)",
            n_overlap,
        )
    return new[mask_new].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------


def _summarize(df: pd.DataFrame, label: str) -> None:
    logger.info("--- %s: %d rows ---", label, len(df))
    if df.empty:
        return
    target_counts = df[df["virus"].isin(TARGET_VIRUSES)].groupby("virus").size()
    if not target_counts.empty:
        for virus, cnt in target_counts.sort_values(ascending=False).items():
            logger.info("    %-15s %d", virus, cnt)
    other = len(df) - target_counts.sum()
    if other > 0:
        logger.info("    %-15s %d", "(other viruses)", other)


# ---------------------------------------------------------------------------
# Main merge logic
# ---------------------------------------------------------------------------


def merge(
    api_dir: Path,
    existing_path: Path,
    output_path: Path,
    dry_run: bool = False,
) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # -- Load existing Pipeline B negatives
    logger.info("Loading existing negatives from %s", existing_path)
    existing = pd.read_csv(existing_path, low_memory=False)
    # Ensure peptides are uppercase for consistent dedup
    existing["peptide"] = existing["peptide"].astype(str).str.strip().str.upper()
    _summarize(existing, "Existing Pipeline B negatives")

    # -- Load and normalize API negatives
    logger.info("\nLoading Pipeline A API negatives from %s/", api_dir)
    raw_api = _load_api_negatives(api_dir)
    logger.info("Raw API negative rows: %d", len(raw_api))

    if raw_api.empty:
        logger.warning("No API negatives found - output will be identical to existing file.")
        merged = existing.copy()
    else:
        normalized = _normalize_to_schema(raw_api)
        logger.info("After schema normalization: %d rows", len(normalized))
        _summarize(normalized, "API negatives (pre-dedup)")

        # -- Deduplicate
        logger.info("\nDeduplication:")
        deduped_api = _dedup_within_api(normalized)
        net_new = _dedup_against_existing(deduped_api, existing)
        _summarize(net_new, "Net-new API negatives (after full dedup)")

        # -- Merge
        merged = pd.concat([existing, net_new], ignore_index=True)

    # -- Final sort for determinism
    sort_cols = [c for c in ["peptide", "hla_allele", "assay_type"] if c in merged.columns]
    merged = merged.sort_values(sort_cols).reset_index(drop=True)
    merged["is_quarantined"] = merged["is_quarantined"].fillna(False)

    logger.info("\n=== MERGE SUMMARY ===")
    logger.info("Existing Pipeline B rows : %d", len(existing))
    logger.info("Net-new from API         : %d", len(merged) - len(existing))
    logger.info("Merged output total      : %d", len(merged))
    for virus in sorted(TARGET_VIRUSES):
        old = int(
            (
                existing["virus"].isin({virus, "HPV16", "HPV18"})
                if virus == "HPV"
                else existing["virus"] == virus
            ).sum()
        )
        new_total = int(
            (
                merged["virus"].isin({"HPV", "HPV16", "HPV18"})
                if virus == "HPV"
                else merged["virus"] == virus
            ).sum()
        )
        if old > 0 or new_total > 0:
            logger.info("  %-15s %4d -> %4d (net +%d)", virus, old, new_total, new_total - old)

    if dry_run:
        logger.info("\nDRY RUN - no file written.")
        return merged

    # -- Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info("\nWritten: %s", output_path)

    write_provenance(
        str(output_path),
        sources=[str(api_dir), str(existing_path)],
        row_count=len(merged),
        extra={
            "existing_rows": len(existing),
            "net_new_api_rows": len(merged) - len(existing),
            "git_sha": git_sha(),
            "per_virus_counts": {
                v: int((merged["virus"] == v).sum()) for v in sorted(TARGET_VIRUSES)
            },
        },
    )
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge IEDB API negatives (Pipeline A) into the iedb_negatives schema (Pipeline B).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--api-dir",
        type=Path,
        default=Path("data/iedb"),
        metavar="DIR",
        help="Directory containing *_tcell.csv files from fetch_iedb_tcell.py. Default: data/iedb/",
    )
    p.add_argument(
        "--existing",
        type=Path,
        default=Path("data/iedb_negatives_v5.csv"),
        metavar="FILE",
        help="Existing Pipeline B negatives CSV (ingest_iedb_negatives.py output). "
        "Default: data/iedb_negatives_v5.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/iedb_negatives_v5_merged.csv"),
        metavar="FILE",
        help="Output path for merged negatives. Default: data/iedb_negatives_v5_merged.csv",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing the output file.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.api_dir.exists():
        logger.error("--api-dir not found: %s", args.api_dir)
        return 1
    if not args.existing.exists():
        logger.error("--existing not found: %s", args.existing)
        return 1
    merge(args.api_dir, args.existing, args.output, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
