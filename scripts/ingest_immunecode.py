"""Ingest ImmuneCODE MIRA SARS-CoV-2 data and extract non-immunogenic peptide
negatives for the SESTRAV immunogenicity prediction pipeline.

ImmuneCODE MIRA (Adaptive Biotechnologies) maps SARS-CoV-2 peptides to TCR
beta sequences from COVID-19 patient blood samples. Peptide-allele pairs that
appear in the assay data but accumulate zero TCR template matches are treated
as experimentally confirmed negatives for MHC Class I immunogenicity.

Negative definition (MIRA-specific)
-------------------------------------
A (peptide, HLA allele) pair is labelled 0 (non-immunogenic) when the raw
peptide entry (which may be a semicolon-separated pool) appears in the assay
data with zero TCR template rows - meaning no clonal T-cell expansion was
observed. Individual peptides within a matched pool are excluded because it
is ambiguous which pool member drove the response.

Source: Nolan S, et al. A large-scale database of T-cell receptor beta
  sequences and binding associations from natural and synthetic exposure to
  SARS-CoV-2. Research Square (2020). PMID 32665443.
Download:
  https://clients.adaptivebiotech.com/pub/covid-2020/immuneCODE-MIRA-release002.1.zip

Usage
-----
  Auto-download (requires network access):
    python scripts/ingest_immunecode.py

  Pre-downloaded ZIP:
    python scripts/ingest_immunecode.py --input path/to/immuneCODE-MIRA-release002.1.zip

  Pre-extracted CSV:
    python scripts/ingest_immunecode.py --input path/to/peptide-detail-cov2.csv

  Dry run - print stats, no file write:
    python scripts/ingest_immunecode.py --dry-run

  Custom output path:
    python scripts/ingest_immunecode.py --output data/my_negatives.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ssl_fix  # noqa: F401, E402 - patch SSL before any network calls
from _dataset_utils import write_provenance  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIRA_DOWNLOAD_URL = (
    "https://clients.adaptivebiotech.com/pub/covid-2020/immuneCODE-MIRA-release002.1.zip"
)

# Preferred CSV filenames inside the ZIP archive, tried in order.
_CANDIDATE_CSVS: list[str] = [
    "peptide-detail-cov2.csv",
    "peptide-detail-ci.csv",
]

# NCBI taxon ID for SARS-CoV-2 (Wuhan-Hu-1 reference strain).
_VIRUS_TAXON_ID: int = 2697049

_REFERENCE_PMID: str = "32665443"

# Standard 20 amino acids (MHC Class I canonical set).
_VALID_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")

# MHC Class I canonical peptide length bounds.
_MIN_LEN: int = 8
_MAX_LEN: int = 11

# Exactly 23 schema columns required by SESTRAV negatives pipeline.
SCHEMA_COLUMNS: list[str] = [
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

# Fixed values applied to every output row.
_FIXED_FIELDS: dict[str, object] = {
    "label": 0,
    "virus": "SARS-CoV-2",
    "protein": np.nan,
    "strain": "SARS-CoV-2 (Wuhan-Hu-1)",
    "source_type": "Virus",
    "database_source": "ImmuneCODE",
    "tcr_alpha_cdr3": np.nan,
    "tcr_beta_cdr3": np.nan,
    "virus_family": "Coronaviridae",
    "negative_origin": "immunecode_mira",
    "assay_type": "MIRA_TCR_sequencing",
    "assay_quality_tier": 1,
    "assay_quality_weight": 0.9,
    "reference_pmid": _REFERENCE_PMID,
    "iedb_assay_id": np.nan,
    "infection_phase": np.nan,
    "antigen_latency_program": np.nan,
    "assay_context": np.nan,
    "cross_reactivity_tested": np.nan,
    "virus_taxon_id": _VIRUS_TAXON_ID,
    "is_quarantined": False,
}

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ZIP utilities
# ---------------------------------------------------------------------------


def _safe_extract_zip(zip_ref: zipfile.ZipFile, dest_dir: str) -> None:
    """Extract a ZIP archive, rejecting path-traversal members (zip-slip guard)."""
    dest_root = os.path.realpath(dest_dir)
    for member in zip_ref.infolist():
        target = os.path.realpath(os.path.join(dest_dir, member.filename))
        if target != dest_root and not target.startswith(dest_root + os.sep):
            raise RuntimeError(f"Unsafe zip member path: {member.filename}")
    zip_ref.extractall(dest_dir)


def _find_csv_in_dir(search_dir: str) -> str | None:
    """Return the first MIRA peptide-detail CSV path found under search_dir."""
    for candidate in _CANDIDATE_CSVS:
        for root, _dirs, files in os.walk(search_dir):
            if candidate in files:
                return os.path.join(root, candidate)
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _download_mira(dest_dir: str) -> str:
    """Download the MIRA ZIP from Adaptive Biotechnologies.

    Returns the local path of the downloaded file.
    Raises RuntimeError with an actionable message if the download fails.
    """
    os.makedirs(dest_dir, exist_ok=True)
    zip_name = "immuneCODE-MIRA-release002.1.zip"
    zip_path = os.path.join(dest_dir, zip_name)
    log.info("Downloading ImmuneCODE MIRA from %s", MIRA_DOWNLOAD_URL)
    try:
        urllib.request.urlretrieve(MIRA_DOWNLOAD_URL, zip_path)  # nosec B310
    except Exception as exc:
        raise RuntimeError(
            f"Download failed: {exc}\n"
            "Please download the ImmuneCODE MIRA archive manually from:\n"
            f"  {MIRA_DOWNLOAD_URL}\n"
            "Then pass it via --input PATH."
        ) from exc
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"Downloaded file is not a valid ZIP archive: {zip_path}")
    return zip_path


def _load_from_zip(zip_path: str) -> tuple[pd.DataFrame, str]:
    """Extract a MIRA ZIP, locate the peptide-detail CSV, and load it.

    Returns (DataFrame, zip_path_for_provenance).
    """
    log.info("Extracting ZIP: %s", zip_path)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extract_zip(zf, tmp)
        csv_path = _find_csv_in_dir(tmp)
        if csv_path is None:
            raise FileNotFoundError(
                f"No MIRA peptide-detail CSV found in '{zip_path}'. "
                f"Expected one of: {_CANDIDATE_CSVS}"
            )
        log.info("Reading CSV from archive: %s", os.path.basename(csv_path))
        # Read into memory before temp dir is deleted.
        df = pd.read_csv(csv_path, low_memory=False)
    return df, zip_path


def _load_csv(input_path: str | None) -> tuple[pd.DataFrame, str]:
    """Load a MIRA peptide-detail CSV from a ZIP, a CSV, or by auto-download.

    Dispatch order:
      1. input_path is None -> attempt download from MIRA_DOWNLOAD_URL.
      2. input_path is a ZIP (detected by zipfile.is_zipfile) -> extract then read.
      3. input_path is a CSV -> read directly.

    Returns (DataFrame, source_description) where source_description is used
    only for the provenance sidecar.
    """
    if input_path is None:
        log.info("No --input given; attempting auto-download.")
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = _download_mira(tmp)
            return _load_from_zip(zip_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"--input path not found: {input_path}")

    if zipfile.is_zipfile(input_path):
        return _load_from_zip(input_path)

    # Direct CSV path.
    log.info("Loading CSV: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    return df, input_path


# ---------------------------------------------------------------------------
# Allele normalization
# ---------------------------------------------------------------------------


def _normalize_allele(raw: str) -> str | None:
    """Normalize an HLA allele string to HLA-X*GG:PP (Class I only).

    Handles:
      HLA-A*02:01  -> HLA-A*02:01  (already canonical)
      HLA-A*0201   -> HLA-A*02:01  (8-digit, no colon)
      A*02:01      -> HLA-A*02:01  (missing HLA prefix)
      A*0201       -> HLA-A*02:01  (missing prefix, 8-digit)

    Returns None for Class II alleles (HLA-D*) or unrecognized formats.
    Only HLA-A, HLA-B, and HLA-C (Class I) are accepted.
    """
    if not isinstance(raw, str):
        return None
    allele = raw.strip()
    # Already canonical HLA-[ABC]*DD:DD
    if re.match(r"^HLA-[ABC]\*\d{2}:\d{2}$", allele):
        return allele
    # 8-digit no colon: HLA-A*0201 -> HLA-A*02:01
    m = re.match(r"^HLA-([ABC])\*(\d{2})(\d{2})$", allele)
    if m:
        return f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}"
    # Missing prefix, canonical digits: A*02:01 -> HLA-A*02:01
    m = re.match(r"^([ABC])\*(\d{2}):(\d{2})$", allele)
    if m:
        return f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}"
    # Missing prefix, 8-digit: A*0201 -> HLA-A*02:01
    m = re.match(r"^([ABC])\*(\d{2})(\d{2})$", allele)
    if m:
        return f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}"
    # Cannot normalize or not Class I.
    return None


# ---------------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------------


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name present in df.columns, or None."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _require_column(df: pd.DataFrame, candidates: list[str], role: str) -> str:
    """Return the first matching column, or raise ValueError with details."""
    col = _detect_column(df, candidates)
    if col is None:
        raise ValueError(
            f"Cannot find '{role}' column in DataFrame. "
            f"Tried: {candidates}. "
            f"Available columns: {df.columns.tolist()}"
        )
    return col


# ---------------------------------------------------------------------------
# Negative extraction
# ---------------------------------------------------------------------------


def _extract_negatives(df: pd.DataFrame) -> list[dict[str, object]]:
    """Extract non-immunogenic (peptide, hla_allele) pairs from MIRA data.

    Strategy
    --------
    1. Detect the peptide, allele, and TCR template columns by name.
    2. Mark rows where the TCR column is non-null and non-empty as hits.
    3. Group by (peptide_raw, allele_raw) and sum hit counts per group.
    4. Groups with sum = 0 indicate pools or peptides that were assayed but
       showed no T-cell expansion -> candidate negatives.
    5. Expand semicolon-separated peptide pools into individual sequences.
    6. Expand semicolon-separated allele fields and normalize each allele.
    7. Apply length (8-11 aa) and amino acid (standard 20 aa) filters.

    Pools that DO have TCR matches are intentionally excluded: it is ambiguous
    which pool member drove the response, so no individual peptide within a
    matched pool can be safely labelled as a negative.
    """
    pep_col = _require_column(
        df,
        ["Peptide", "peptide", "Antigen Sequence", "antigen_sequence"],
        "peptide",
    )
    allele_col = _require_column(
        df,
        ["HLA Restrictions", "HLA_Restrictions", "MHC", "hla", "Allele"],
        "HLA allele",
    )
    tcr_col = _detect_column(
        df,
        ["Amino Acids", "amino_acids", "CDR3b", "cdr3b", "TCR Beta CDR3"],
    )

    log.info(
        "Columns detected - peptide: '%s', allele: '%s', TCR: '%s'",
        pep_col,
        allele_col,
        tcr_col if tcr_col else "(not found)",
    )

    if tcr_col is None:
        log.warning(
            "No TCR template column found (tried: Amino Acids, CDR3b, ...). "
            "Cannot identify zero-match rows. "
            "Provide a file with an 'Amino Acids' column that includes null "
            "entries for tested but non-expanding peptides."
        )
        return []

    # Boolean mask: True where a TCR template entry is present.
    has_tcr: pd.Series = df[tcr_col].notna() & (df[tcr_col].astype(str).str.strip() != "")
    df = df.copy()
    df["_has_tcr"] = has_tcr.astype(int)

    # Aggregate TCR hit count per (raw peptide, raw allele) pair.
    grp = df.groupby([pep_col, allele_col], sort=False)["_has_tcr"].sum().reset_index()

    n_total = len(grp)
    neg_mask = grp["_has_tcr"] == 0
    n_neg_pools = int(neg_mask.sum())
    n_pos_pools = n_total - n_neg_pools

    log.info(
        "Grouped %d unique (peptide, allele) entries: "
        "%d with TCR hits, %d with zero hits (candidate negatives).",
        n_total,
        n_pos_pools,
        n_neg_pools,
    )

    if n_neg_pools == 0:
        log.warning(
            "No zero-TCR-count (peptide, allele) pairs found. "
            "The input file may contain only enriched TCR-match rows. "
            "Consider providing the full MIRA catalog that includes "
            "zero-template-count entries for tested but non-expanding pools."
        )
        return []

    neg_pools = grp[neg_mask]

    # Expand each zero-count pool into individual (peptide, allele) records.
    records: list[dict[str, object]] = []
    skipped_len = 0
    skipped_aa = 0
    skipped_allele = 0

    for _, row in neg_pools.iterrows():
        raw_pep_str = str(row[pep_col])
        raw_allele_str = str(row[allele_col])

        peptides_raw = [p.strip().upper() for p in raw_pep_str.split(";") if p.strip()]
        alleles_raw = [a.strip() for a in raw_allele_str.split(";") if a.strip()]

        for pep in peptides_raw:
            if not (_MIN_LEN <= len(pep) <= _MAX_LEN):
                skipped_len += 1
                continue
            if not all(c in _VALID_AA for c in pep):
                skipped_aa += 1
                continue
            for allele_raw in alleles_raw:
                allele = _normalize_allele(allele_raw)
                if allele is None:
                    skipped_allele += 1
                    continue
                records.append({"peptide": pep, "hla_allele": allele})

    log.info(
        "Expanded to %d raw (peptide, allele) records. "
        "Skipped: %d wrong length, %d invalid AA, %d non-Class-I allele.",
        len(records),
        skipped_len,
        skipped_aa,
        skipped_allele,
    )
    return records


# ---------------------------------------------------------------------------
# Output construction
# ---------------------------------------------------------------------------


def _build_output_df(records: list[dict[str, object]]) -> pd.DataFrame:
    """Assemble the 23-column SESTRAV schema DataFrame from extracted records.

    Applies deduplication on (peptide, hla_allele) keeping the highest
    assay_quality_weight (all ImmuneCODE rows share the same weight, so this
    effectively removes exact duplicates that arise from pool expansion).
    """
    if not records:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    df = pd.DataFrame(records)

    # Apply fixed metadata fields.
    for col, val in _FIXED_FIELDS.items():
        df[col] = val

    # Dedup on (peptide, hla_allele), keep highest quality weight row.
    df = (
        df.sort_values("assay_quality_weight", ascending=False)
        .drop_duplicates(subset=["peptide", "hla_allele"])
        .reset_index(drop=True)
    )

    # Deterministic sort for byte-stable output across re-runs.
    df = df.sort_values(["peptide", "hla_allele"]).reset_index(drop=True)

    # Enforce column order and return exactly the 23 schema columns.
    return df[SCHEMA_COLUMNS]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract SARS-CoV-2 non-immunogenic negatives from ImmuneCODE MIRA "
            "and produce a SESTRAV 23-column negatives CSV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input",
        metavar="PATH",
        help=(
            "Path to the ImmuneCODE MIRA ZIP archive or a pre-extracted "
            "peptide-detail-cov2.csv / peptide-detail-ci.csv file. "
            "If omitted, the script attempts to download automatically from "
            f"{MIRA_DOWNLOAD_URL}"
        ),
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        default="data/immunecode_sars_negatives.csv",
        help=("Output CSV path (default: data/immunecode_sars_negatives.csv)."),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics and a sample of output rows; do not write files.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ingest_immunecode."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    # Deterministic seeds for reproducibility.
    random.seed(42)
    np.random.seed(42)

    # Load raw data.
    try:
        df_raw, source_path = _load_csv(args.input)
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1

    log.info("Loaded %d rows from source.", len(df_raw))

    # Extract zero-TCR-count negatives.
    records = _extract_negatives(df_raw)
    if not records:
        log.error(
            "No negative records extracted. "
            "Check the --input file and the log messages above for details."
        )
        return 1

    # Build the 23-column output DataFrame.
    df_out = _build_output_df(records)

    unique_peptides = int(df_out["peptide"].nunique())
    unique_alleles = int(df_out["hla_allele"].nunique())
    log.info(
        "Final output: %d unique (peptide, hla_allele) negative pairs "
        "(%d unique peptides, %d unique alleles).",
        len(df_out),
        unique_peptides,
        unique_alleles,
    )

    allele_counts = df_out["hla_allele"].value_counts().head(10)
    log.info("Allele distribution (top 10):\n%s", allele_counts.to_string())

    if args.dry_run:
        log.info("--dry-run: no files written.")
        print(df_out.head(20).to_string(index=False))
        return 0

    # Write output CSV.
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    log.info("Wrote %d rows -> %s", len(df_out), args.output)

    # Write provenance sidecar.
    write_provenance(
        args.output,
        sources=[source_path],
        row_count=len(df_out),
        extra={
            "database": "ImmuneCODE MIRA",
            "release": "002.1",
            "reference_pmid": _REFERENCE_PMID,
            "label": 0,
            "virus": "SARS-CoV-2",
            "negative_origin": "immunecode_mira",
            "unique_peptides": unique_peptides,
            "unique_alleles": unique_alleles,
            "allele_distribution": allele_counts.to_dict(),
        },
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
