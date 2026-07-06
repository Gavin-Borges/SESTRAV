"""
ingest_iedb_negatives.py - Ingest IEDB bulk export CSV files filtering for
negative-outcome T cell assays.

Reads a standard IEDB T cell assay bulk export CSV and outputs a v5-schema-
compatible CSV of filtered, deduplicated negative rows suitable for inclusion
in the v5 dataset build.

Hardening (MASTER_STRATEGIC_PLAN Part 16, Amendment 1):
  1. Two-row IEDB header detection (post-2023 exports carry a group-name row).
  2. Secondary negativity signal via response_frequency < 0.1 when the
     qualitative measure is blank.
  3. HLA normalization through mhcgnomes (4-digit; Class I HLA-A/B/C only;
     supertypes and non-human alleles rejected).
  4. Intra-export deduplication on (peptide, hla_allele, reference_pmid).
  5. Persistent IEDB assay/reference id preserved as iedb_assay_id.
Plus a 3-tier assay quality weighting and an input_sha256 provenance pin.

Usage:
    python scripts/ingest_iedb_negatives.py \
        --input data/iedb_tcell_export.csv \
        --output data/iedb_negatives_v5.csv \
        --existing-dataset data/immunogenicity_dataset_v4.csv \
        [--dry-run]
"""

import argparse
import hashlib
import json
import logging
import random
import re
import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mhcgnomes
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard amino acid alphabet (single-letter codes)
AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

# Class I HLA genes accepted after mhcgnomes normalization.
CLASS_I_GENES = {"A", "B", "C"}

# IEDB qualitative measures considered negative
NEGATIVE_MEASURES = {"negative", "negative-low"}

# Secondary negativity signal: a blank qualitative measure with a reported
# response frequency below this threshold is treated as a tested negative.
RESPONSE_FREQUENCY_NEG_THRESHOLD = 0.1

# Assay-group keyword markers indicating a direct ex vivo functional readout
# (quality tier 1, weight 1.0).
DIRECT_FUNCTIONAL_MARKERS = (
    "ifn-gamma",
    "elispot",
    "multimer",
    "cytotoxicity",
    "intracellular cytokine",
    "proliferation",
)

# Assay-group keyword markers indicating an expanded-culture protocol
# (quality tier 3, weight 0.5) - less direct than ex vivo functional assays.
EXPANDED_CULTURE_MARKERS = (
    "stimulated",
    "expanded",
    "bulk culture",
    "cultured",
)

# Quality tier -> sample weight.
TIER_WEIGHT = {1: 1.0, 2: 0.7, 3: 0.5}

# Assay types that represent biochemical / non-T-cell-functional readouts.
# These must not be used as immunogenicity negatives in training data because
# a "negative" biological-activity outcome (e.g., a failed binding assay) does
# not imply the peptide is non-immunogenic in a T cell context. Confirmed
# root-cause of HPV AUC-ROC inversion in v5 (PMID 38608022 contributed 66
# mislabelled negatives; OOF score mean 0.71 vs HPV positives 0.37).
EXCLUDED_ASSAY_TYPES: frozenset[str] = frozenset({"biological activity"})

# Lower-cased IEDB top-level group tokens. When pandas reads a post-2023 export
# with header=0, the first column name is one of these group labels rather than
# a field name, signalling that the real header is on row 1.
IEDB_GROUP_TOKENS = {
    "reference",
    "epitope",
    "related object",
    "host",
    "1st in vivo process",
    "2nd in vivo process",
    "in vitro process",
    "adoptive transfer",
    "immunization comments",
    "assay",
    "assay id",
    "effector cell",
    "antigen presenting cell",
    "mhc restriction",
    "assay antigen",
    "assay comments",
}

# Mapping from (group.lower(), field.lower()) pairs in the June 2026 IEDB v3
# CSV two-row header to the legacy column names that resolve_columns expects.
# Older exports already use the legacy names directly in the field row so this
# map produces no-op renames for those files.
_TWO_ROW_V3_LEGACY_MAP: dict[tuple[str, str], str] = {
    ("epitope", "name"): "Epitope Name",
    ("host", "name"): "Host Organism Name",
    ("mhc restriction", "name"): "Allele Name",
    ("assay antigen", "name"): "Antigen Name",
    ("assay", "qualitative measurement"): "Qualitative Measure",
    ("reference", "pmid"): "Reference PubMed ID",
    ("assay", "response frequency (%)"): "Response Frequency As Reported",
    ("epitope", "species"): "Description",
    ("assay", "method"): "Assay Group",
    ("assay id", "iedb iri"): "Assay ID",
}

# NCBI taxonomy names used by post-2023 IEDB exports that differ from the
# common abbreviations in the SESTRAV v4 dataset. Keys are lowercased for
# case-insensitive lookup. Only maps names that would otherwise mismatch the
# v4 virus column and break per-virus deduplication and evaluation grouping.
_VIRUS_TAXONOMY_TO_COMMON: dict[str, str] = {
    "human gammaherpesvirus 4": "EBV",
    "orthohepadnavirus hominoidei": "HBV",
    "hepacivirus hominis": "HCV",
    "alphapapillomavirus 9": "HPV16",
    "alphapapillomavirus 7": "HPV18",
    "alphapapillomavirus 10": "HPV",
    "human betaherpesvirus 5": "CMV",
    "lentivirus humimdef1": "HIV-1",
    "human alphaherpesvirus 1": "HSV-1",
    "human alphaherpesvirus 2": "HSV-2",
    "human alphaherpesvirus 3": "VZV",
    "human respiratory syncytial virus": "RSV",
    "deltaretrovirus priTlym1".lower(): "HTLV-1",
    "betapolyomavirus hominis": "MCPyV",
    "homo sapiens": "Self",
    "enterovirus betacoxsackie": "CoxsackievirusB",
    "severe acute respiratory syndrome coronavirus 2": "SARS-CoV-2",
    "yellow fever virus": "YFV",
    "dengue virus": "DENV",
    "rotavirus alphagastroenteritidis": "RotavirusA",
    "mammarenavirus lassaense": "LassaVirus",
    "alphavirus chikungunya": "CHIKV",
    "zika virus": "ZIKV",
}

# Repairs the "HLA-A 02:01" space form to the "HLA-A*02:01" form mhcgnomes
# parses. Only touches a gene-then-space-then-digit prefix.
_ALLELE_SPACE_RE = re.compile(r"^(HLA-[A-Z]+[0-9]*)\s+(\d)")

# ---------------------------------------------------------------------------
# Resolved-column container
# ---------------------------------------------------------------------------


@dataclass
class ResolvedColumns:
    """Actual column names resolved from a (header-normalized) IEDB export."""

    host: str
    measure: str
    epitope: str
    allele: str
    assay_group: str | None
    antigen: str | None
    pmid: str | None
    description: str | None
    response_frequency: str | None
    assay_id: str | None


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


def get_git_sha() -> str:
    """Return the short HEAD git SHA, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(  # nosec
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # nosec B110
        pass
    return "unknown"


def compute_file_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _repair_allele_spacing(allele: str) -> str:
    """Repair the 'HLA-A 02:01' space form to 'HLA-A*02:01'."""
    return _ALLELE_SPACE_RE.sub(r"\1*\2", allele)


def normalize_hla_allele(raw: str) -> str | None:
    """Normalize an HLA allele string to 4-digit Class I HLA-A/B/C resolution.

    Uses mhcgnomes to parse and normalize. Returns the normalized allele string
    (e.g. ``HLA-A*02:01``) or ``None`` if the value is not a parseable human
    Class I HLA-A/B/C allele. Supertypes (``HLA-A2``), non-human alleles
    (``H-2-Kb``), Class II alleles, and bare class strings (``Class I``) are all
    rejected.
    """
    if not isinstance(raw, str):
        return None
    allele = _repair_allele_spacing(raw.strip())
    if not allele:
        return None
    try:
        parsed = mhcgnomes.parse(allele)
    except Exception:
        return None
    # Reject Serotype (supertype), MhcClass, Gene and anything that is not a
    # fully specified Allele.
    if not isinstance(parsed, mhcgnomes.Allele):
        return None
    if not (
        getattr(parsed, "is_human", False)
        and getattr(parsed, "is_class1", False)
        and getattr(parsed, "gene_name", None) in CLASS_I_GENES
    ):
        return None
    # Restrict to 4-digit (2-field) resolution where the allele is more specific.
    try:
        if parsed.num_allele_fields >= 2:
            parsed = parsed.restrict_allele_fields(2)
    except Exception:  # nosec B110
        pass
    return parsed.to_string()


def is_valid_peptide(seq: str) -> bool:
    """Return True if seq is a standard AA string of length 8-11."""
    if not isinstance(seq, str):
        return False
    seq = seq.strip()
    if len(seq) < 8 or len(seq) > 11:
        return False
    return bool(AA_PATTERN.match(seq))


def assign_quality_tier(assay_group: str | None) -> int:
    """Return the assay quality tier (1/2/3) for an assay-group string.

    Tier 1: direct ex vivo functional assay (IFN-gamma ELISpot, multimer, ...).
    Tier 3: expanded-culture protocol (stimulated / expanded / cultured / bulk
            culture) - downweighted as less direct, even with a functional
            readout.
    Tier 2: everything else (default).
    """
    if not isinstance(assay_group, str):
        return 2
    ag = assay_group.lower()
    if any(marker in ag for marker in EXPANDED_CULTURE_MARKERS):
        return 3
    if any(marker in ag for marker in DIRECT_FUNCTIONAL_MARKERS):
        return 1
    return 2


def assign_quality_weight(assay_group: str | None) -> float:
    """Return the sample weight for an assay-group string (tier -> weight)."""
    return TIER_WEIGHT[assign_quality_tier(assay_group)]


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------


def _find_col(columns: list[str], *candidates: str) -> str | None:
    col_lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in col_lower:
            return col_lower[candidate.lower()]
    return None


def strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding whitespace from column names."""
    return df.rename(columns={c: str(c).strip() for c in df.columns})


def resolve_columns(df: pd.DataFrame, logger: logging.Logger) -> ResolvedColumns:
    """Resolve logical IEDB fields to actual column names; exit if any required
    column is missing."""
    cols = list(df.columns)
    host = _find_col(cols, "Host Organism Name", "host organism name", "host")
    measure = _find_col(cols, "Qualitative Measure", "qualitative measure", "qualitative_measure")
    epitope = _find_col(cols, "Epitope Name", "epitope name", "name", "epitope")
    allele = _find_col(cols, "Allele Name", "allele name", "allele")

    missing = [
        name
        for name, col in (
            ("Host Organism Name", host),
            ("Qualitative Measure", measure),
            ("Epitope Name", epitope),
            ("Allele Name", allele),
        )
        if col is None
    ]
    if missing:
        logger.error(
            "Required columns not found in input: %s. Available: %s",
            missing,
            cols[:20],
        )
        sys.exit(1)

    return ResolvedColumns(
        host=host,  # type: ignore[arg-type]
        measure=measure,  # type: ignore[arg-type]
        epitope=epitope,  # type: ignore[arg-type]
        allele=allele,  # type: ignore[arg-type]
        assay_group=_find_col(cols, "Assay Group", "assay group", "assay_group"),
        antigen=_find_col(cols, "Antigen Name", "antigen name", "antigen", "description"),
        pmid=_find_col(
            cols,
            "Reference PubMed ID",
            "reference pubmed id",
            "pubmed id",
            "pubmed_id",
            "pmid",
        ),
        description=_find_col(cols, "Description", "description", "antigen description"),
        response_frequency=_find_col(
            cols,
            "Response Frequency As Reported",
            "response frequency as reported",
            "response frequency",
            "response_frequency",
        ),
        assay_id=_find_col(
            cols,
            "Assay ID",
            "assay id",
            "Assay ID - IEDB IRI",
            "Reference ID",
            "reference id",
            "Reference IRI",
        ),
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_iedb_export(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load an IEDB bulk export CSV, handling the post-2023 two-row header.

    Post-2023 IEDB exports carry a group-name row (Reference, Epitope, Assay,
    ...) as row 0 and the real field names on row 1. Detection uses a fast
    2-row peek so large files are not read twice just for header sniffing.

    When the two-row format is detected, column names are normalized using
    ``_TWO_ROW_V3_LEGACY_MAP`` so that the downstream ``resolve_columns``
    function sees the same logical names regardless of which IEDB export
    vintage produced the file.
    """
    logger.info("Loading IEDB export from %s", path)
    try:
        df_peek = pd.read_csv(path, header=None, nrows=2, low_memory=False)
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        sys.exit(1)

    first_val = str(df_peek.iloc[0, 0]).strip().lower()
    if first_val in IEDB_GROUP_TOKENS or first_val.startswith("unnamed"):
        logger.info(
            "Detected IEDB two-row header (row 0 group token %r); re-reading with header=1",
            df_peek.iloc[0, 0],
        )
        try:
            groups_row = df_peek.iloc[0].fillna("").astype(str).str.strip().str.lower().tolist()
            fields_row = df_peek.iloc[1].fillna("").astype(str).str.strip().str.lower().tolist()
            df = pd.read_csv(path, header=1, low_memory=False)
            pandas_cols = list(df.columns)
            rename: dict[str, str] = {}
            for i, (g, f) in enumerate(zip(groups_row, fields_row)):
                if i >= len(pandas_cols):
                    break
                legacy = _TWO_ROW_V3_LEGACY_MAP.get((g, f))
                if legacy and pandas_cols[i] not in rename:
                    rename[pandas_cols[i]] = legacy
            if rename:
                df = df.rename(columns=rename)
                logger.info(
                    "Normalized %d column names from two-row header format",
                    len(rename),
                )
        except Exception as exc:
            logger.error("Failed to re-read %s with header=1: %s", path, exc)
            sys.exit(1)
    else:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            logger.error("Failed to read %s: %s", path, exc)
            sys.exit(1)

    logger.info("Raw input: %d rows x %d columns", len(df), len(df.columns))
    return df


def filter_rows(
    df: pd.DataFrame, cols: ResolvedColumns, logger: logging.Logger
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply all inclusion filters and return (filtered_df, stats_dict)."""
    stats: dict[str, int] = {}
    stats["raw_input"] = len(df)

    # --- Filter 1: Human host ---
    mask_host = df[cols.host].fillna("").str.contains("Homo sapiens", case=False, na=False)
    df = df[mask_host].copy()
    stats["after_human_host_filter"] = len(df)
    logger.info(
        "After human host filter: %d rows (removed %d)",
        len(df),
        stats["raw_input"] - len(df),
    )

    # --- Filter 2: Negative outcome ---
    # Primary signal: explicit negative qualitative measure.
    measure_series = df[cols.measure].fillna("").str.strip().str.lower()
    mask_neg_qual = measure_series.isin(NEGATIVE_MEASURES)
    # Secondary signal: blank qualitative measure but a reported response
    # frequency below threshold (Amendment 1, Filter 2).
    mask_neg_rf = pd.Series(False, index=df.index)
    if cols.response_frequency is not None:
        rf = pd.to_numeric(df[cols.response_frequency], errors="coerce")
        mask_neg_rf = (measure_series == "") & rf.notna() & (rf < RESPONSE_FREQUENCY_NEG_THRESHOLD)
    mask_neg = mask_neg_qual | mask_neg_rf
    stats["negatives_by_qualitative_measure"] = int(mask_neg_qual.sum())
    stats["negatives_by_response_frequency"] = int((mask_neg_rf & ~mask_neg_qual).sum())
    df = df[mask_neg].copy()
    stats["after_negative_measure_filter"] = len(df)
    logger.info(
        "After negative filter: %d rows (%d by qualitative measure, %d by response frequency)",
        len(df),
        stats["negatives_by_qualitative_measure"],
        stats["negatives_by_response_frequency"],
    )

    # --- Filter 3 & 4: Peptide length 8-11 and standard AA ---
    peptides = df[cols.epitope].fillna("").str.strip()
    mask_peptide = peptides.apply(is_valid_peptide)
    df = df[mask_peptide].copy()
    stats["after_peptide_filter"] = len(df)
    logger.info(
        "After peptide validity filter (length 8-11, standard AA): %d rows (removed %d)",
        len(df),
        stats["after_negative_measure_filter"] - len(df),
    )

    # --- Filter 5: HLA allele resolvable as Class I A/B/C (mhcgnomes) ---
    alleles_raw = df[cols.allele].fillna("").str.strip()
    mask_hla = alleles_raw.apply(lambda x: normalize_hla_allele(x) is not None)
    df = df[mask_hla].copy()
    stats["after_hla_filter"] = len(df)
    logger.info(
        "After HLA Class I A/B/C filter: %d rows (removed %d)",
        len(df),
        stats["after_peptide_filter"] - len(df),
    )

    # --- Filter 6: Exclude non-T-cell-functional assay types ---
    if cols.assay_group is not None:
        assay_group_series = df[cols.assay_group].fillna("").str.strip()
        mask_excluded = assay_group_series.isin(EXCLUDED_ASSAY_TYPES)
        n_excluded = int(mask_excluded.sum())
        df = df[~mask_excluded].copy()
        stats["biological_activity_rows_excluded"] = n_excluded
        stats["after_assay_type_filter"] = len(df)
        if n_excluded:
            logger.info(
                "Assay type filter: excluded %d rows (%s); %d rows remain",
                n_excluded,
                ", ".join(sorted(EXCLUDED_ASSAY_TYPES)),
                len(df),
            )

    # --- Intra-export dedup on (peptide, hla_allele, reference_pmid) ---
    # Collapse the same epitope reported across multiple assay-format rows of a
    # single publication before the v4 cross-dedup runs in the caller.
    dedup_keys = pd.DataFrame(
        {
            "peptide": df[cols.epitope].fillna("").str.strip(),
            "allele": df[cols.allele].fillna("").str.strip().apply(normalize_hla_allele),
            "pmid": (
                df[cols.pmid].fillna("").astype(str).str.strip() if cols.pmid is not None else ""
            ),
        }
    )
    before_dedup = len(df)
    keep_mask = ~dedup_keys.duplicated(keep="first")
    df = df[keep_mask.to_numpy()].copy()
    stats["intra_export_duplicates_removed"] = before_dedup - len(df)
    stats["after_intra_export_dedup"] = len(df)
    logger.info(
        "Intra-export dedup: %d duplicate rows removed; %d rows remain",
        stats["intra_export_duplicates_removed"],
        len(df),
    )

    return df, stats


def build_output(
    df: pd.DataFrame,
    cols: ResolvedColumns,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Construct the v5-schema output DataFrame from filtered IEDB rows."""
    out: dict[str, object] = {}

    out["peptide"] = df[cols.epitope].str.strip()
    out["label"] = 0
    out["hla_allele"] = df[cols.allele].fillna("").str.strip().apply(normalize_hla_allele)

    if cols.description is not None:
        raw_virus = df[cols.description].fillna("Unknown").str.strip()
    elif cols.antigen is not None:
        raw_virus = df[cols.antigen].fillna("Unknown").str.strip()
    else:
        raw_virus = pd.Series(["Unknown"] * len(df), index=df.index)
    out["virus"] = raw_virus.apply(lambda v: _VIRUS_TAXONOMY_TO_COMMON.get(str(v).lower(), v))

    # protein
    if cols.antigen is not None:
        out["protein"] = df[cols.antigen].fillna("").str.strip()
    else:
        out["protein"] = ""

    out["strain"] = None
    out["source_type"] = "Virus"
    out["database_source"] = "IEDB"
    out["tcr_alpha_cdr3"] = None
    out["tcr_beta_cdr3"] = None
    out["virus_family"] = None  # filled by build_dataset_v5.py
    out["negative_origin"] = "tested_negative"

    if cols.assay_group is not None:
        assay_group = df[cols.assay_group].fillna("").str.strip()
        out["assay_type"] = assay_group
        out["assay_quality_tier"] = assay_group.apply(assign_quality_tier)
        out["assay_quality_weight"] = assay_group.apply(assign_quality_weight)
    else:
        out["assay_type"] = None
        out["assay_quality_tier"] = 2
        out["assay_quality_weight"] = TIER_WEIGHT[2]
        logger.warning("Assay Group column not found; defaulting quality tier 2 (0.7)")

    if cols.pmid is not None:
        out["reference_pmid"] = (
            df[cols.pmid].fillna("").astype(str).str.strip().replace("", None).replace("nan", None)
        )
    else:
        out["reference_pmid"] = None
        logger.warning("Reference PubMed ID column not found; setting to null")

    # Row-level provenance: persistent IEDB assay/reference identifier.
    if cols.assay_id is not None:
        out["iedb_assay_id"] = (
            df[cols.assay_id]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", None)
            .replace("nan", None)
        )
    else:
        out["iedb_assay_id"] = None
        logger.warning("Assay/Reference ID column not found; iedb_assay_id set to null")

    # Per-virus biology context columns - populated by panel/build scripts.
    out["infection_phase"] = None
    out["antigen_latency_program"] = None
    out["assay_context"] = None
    out["cross_reactivity_tested"] = None
    out["virus_taxon_id"] = None

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
        logger.warning("No common key columns found for deduplication; skipping")
        return new_df, 0

    existing_keys = set(existing[available].fillna("").apply(tuple, axis=1))
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
    # Deterministic output: seed both RNGs at entry (data-ingest rule 1).
    random.seed(42)
    np.random.seed(42)

    args = parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger("ingest_iedb_negatives")

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1
    if not args.existing_dataset.exists():
        logger.error("Existing dataset not found: %s", args.existing_dataset)
        return 1

    # Load (with two-row header handling), normalize column names, resolve cols
    raw_df = load_iedb_export(args.input, logger)
    raw_df = strip_column_names(raw_df)
    cols = resolve_columns(raw_df, logger)

    # Filter
    filtered_df, stats = filter_rows(raw_df, cols, logger)

    if len(filtered_df) == 0:
        logger.warning("No rows survived filtering; output will be empty.")

    # Build v5-schema output
    out_df = build_output(filtered_df, cols, logger)

    # Deduplication against the existing (frozen) dataset
    out_df, n_dup = deduplicate_against_existing(out_df, args.existing_dataset, logger)
    stats["deduplication_hits"] = n_dup
    stats["final_output_rows"] = len(out_df)

    # Sort deterministically so re-runs on the same input are byte-stable
    sort_cols = [c for c in ("peptide", "hla_allele", "virus") if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols).reset_index(drop=True)

    # Report
    logger.info("--- Filter stage summary ---")
    for stage, count in stats.items():
        logger.info("  %-45s %d", stage, count)

    if args.dry_run:
        print("\n=== DRY RUN - no output written ===")
        for key in (
            "raw_input",
            "after_human_host_filter",
            "negatives_by_qualitative_measure",
            "negatives_by_response_frequency",
            "after_negative_measure_filter",
            "after_peptide_filter",
            "after_hla_filter",
            "biological_activity_rows_excluded",
            "after_assay_type_filter",
            "intra_export_duplicates_removed",
            "after_intra_export_dedup",
            "deduplication_hits",
            "final_output_rows",
        ):
            print(f"  {key + ':':<38} {stats.get(key, 0)}")
        return 0

    # Write output CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    logger.info("Wrote %d rows to %s", len(out_df), args.output)

    # Write provenance sidecar JSON (data-ingest rule 2). input_sha256 pins the
    # exact IEDB export snapshot (IEDB updates weekly; the same URL can return a
    # different file the following month).
    provenance_path = args.output.with_name(args.output.stem + "_provenance.json")
    provenance = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "input_file": str(args.input),
        "input_sha256": compute_file_sha256(args.input),
        "existing_dataset": str(args.existing_dataset),
        "filter_stats": stats,
        "output_file": str(args.output),
        "output_checksum_sha256": compute_file_sha256(args.output),
    }
    with open(provenance_path, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)
    logger.info("Wrote provenance sidecar: %s", provenance_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
