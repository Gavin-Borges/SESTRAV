"""
build_dataset_v5.py - Assemble the SESTRAV v5 immunogenicity dataset.

Merges v4 positives, IEDB negatives, hard decoys, and optional published
panels. Applies HPV normalization, virus_family assignment, and singleton
quarantine. Writes the v5 CSV and a provenance sidecar JSON.

v4 is frozen and is NEVER modified by this script.

Usage:
    python scripts/build_dataset_v5.py \
        --base-dataset data/immunogenicity_dataset_v4.csv \
        --iedb-negatives data/iedb_negatives_v5.csv \
        --output data/immunogenicity_dataset_v5.csv \
        [--published-panels data/published_panels_v5_combined.csv] \
        [--dry-run]
"""

import argparse
import hashlib
import json
import logging
import random
import subprocess  # nosec B404
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dataset_utils import (  # noqa: E402
    normalize_hla_alleles,
    normalize_virus_names,
    _HLA_AMBIGUOUS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VIRUS_FAMILY_MAP: dict[str, str] = {
    "EBV": "Herpesviridae",
    "CMV": "Herpesviridae",
    "HSV-1": "Herpesviridae",
    "HSV-2": "Herpesviridae",
    "VZV": "Herpesviridae",
    "Human betaherpesvirus 6B": "Herpesviridae",
    "Rhadinovirus humangamma8": "Herpesviridae",
    "SARS-CoV-2": "Coronaviridae",
    "HPV": "Papillomaviridae",
    "HBV": "Hepadnaviridae",
    "HCV": "Flaviviridae",
    "DENV": "Flaviviridae",
    "YFV": "Flaviviridae",
    "ZIKV": "Flaviviridae",
    "IAV": "Orthomyxoviridae",
    "Influenza A virus": "Orthomyxoviridae",
    "Influenza B virus": "Orthomyxoviridae",
    "HIV-1": "Retroviridae",
    "HTLV-1": "Retroviridae",
    "RSV": "Pneumoviridae",
    "MCPyV": "Polyomaviridae",
    "Orthopoxvirus vaccinia": "Poxviridae",
    "CHIKV": "Togaviridae",
    "LassaVirus": "Arenaviridae",
    "CoxsackievirusB": "Picornaviridae",
}

# Quarantine thresholds
MIN_ROWS_PER_VIRUS = 50
MIN_REAL_NEGATIVES_PER_VIRUS = 10

# V5 output column order
V5_COLUMNS = [
    "peptide",
    "label",
    "virus",
    "protein",
    "strain",
    "hla_allele",
    "source_type",
    "database_source",
    "tcr_alpha_cdr3",
    "tcr_beta_cdr3",
    "virus_family",
    "negative_origin",
    "assay_type",
    "assay_quality_weight",
    "assay_quality_tier",
    "reference_pmid",
    "iedb_assay_id",
    "infection_phase",
    "antigen_latency_program",
    "assay_context",
    "cross_reactivity_tested",
    "virus_taxon_id",
    "is_quarantined",
]

# The four Phase 2 target viruses (used for PMID-depth warnings).
TARGET_VIRUSES = ("EBV", "HPV", "HBV", "HCV")

# Minimum distinct PMIDs a target virus should be backed by before a v5 build.
MIN_DISTINCT_PMIDS_PER_TARGET = 3

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
# Git and checksum utilities
# ---------------------------------------------------------------------------


def get_git_sha() -> str:
    try:
        result = subprocess.run(  # nosec
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_csv(path: Path, label: str, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading %s from %s", label, path)
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        logger.error("Failed to read %s (%s): %s", label, path, exc)
        sys.exit(1)
    logger.info("  %s: %d rows", label, len(df))
    return df


# ---------------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------------


def normalize_hpv(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Normalize HPV16/HPV18 virus labels to HPV with appropriate strain."""
    df = df.copy()

    mask_hpv16 = df["virus"].fillna("") == "HPV16"
    mask_hpv18 = df["virus"].fillna("") == "HPV18"
    # HPV with null strain
    mask_hpv_generic = (df["virus"].fillna("") == "HPV") & (
        df["strain"].isna() | (df["strain"].fillna("") == "")
    )

    n_hpv16 = int(mask_hpv16.sum())
    n_hpv18 = int(mask_hpv18.sum())
    n_hpv_generic = int(mask_hpv_generic.sum())

    df.loc[mask_hpv16, "virus"] = "HPV"
    df.loc[mask_hpv16, "strain"] = "HPV16"

    df.loc[mask_hpv18, "virus"] = "HPV"
    df.loc[mask_hpv18, "strain"] = "HPV18"

    df.loc[mask_hpv_generic, "strain"] = "HPV_generic"

    logger.info(
        "HPV normalization: %d HPV16 rows -> virus=HPV/strain=HPV16, "
        "%d HPV18 rows -> virus=HPV/strain=HPV18, "
        "%d HPV rows -> strain=HPV_generic",
        n_hpv16,
        n_hpv18,
        n_hpv_generic,
    )
    return df


def assign_virus_family(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Map virus column to virus_family using VIRUS_FAMILY_MAP."""
    df = df.copy()
    if "virus_family" not in df.columns:
        df["virus_family"] = None

    # Only assign where virus_family is null and virus is known
    mask_needs_family = df["virus_family"].isna() & df["virus"].notna()
    mapped = df.loc[mask_needs_family, "virus"].map(VIRUS_FAMILY_MAP)
    df.loc[mask_needs_family, "virus_family"] = mapped

    unmapped_viruses = (
        df.loc[mask_needs_family & mapped.isna(), "virus"]
        .dropna()
        .unique()
        .tolist()
    )
    if unmapped_viruses:
        logger.warning(
            "Viruses not in VIRUS_FAMILY_MAP (virus_family left null): %s",
            sorted(unmapped_viruses),
        )

    family_counts = df["virus_family"].value_counts(dropna=False).to_dict()
    logger.info("virus_family distribution: %s", family_counts)
    return df


def backfill_negative_origin(
    df: pd.DataFrame, logger: logging.Logger
) -> pd.DataFrame:
    """Set negative_origin for v4 Self rows that lack it."""
    df = df.copy()
    if "negative_origin" not in df.columns:
        df["negative_origin"] = None

    mask = (
        (df["source_type"].fillna("") == "Self")
        & (df["label"] == 0)
        & df["negative_origin"].isna()
    )
    n = int(mask.sum())
    df.loc[mask, "negative_origin"] = "self_proteome_decoy"
    logger.info(
        "negative_origin backfill: set 'self_proteome_decoy' on %d Self rows", n
    )
    return df


def apply_quarantine(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, list[str]]:
    """Mark is_quarantined for viruses below depth thresholds."""
    df = df.copy()
    # Reset virus-level quarantine state (clears stale flags from source CSVs).
    # Row-level allele quarantines (e.g. "HLA class I" set by normalize_hla_alleles)
    # are re-applied immediately below so they are not lost by this reset.
    df["is_quarantined"] = False
    if "hla_allele" in df.columns:
        ambiguous_allele_mask = df["hla_allele"].isin(_HLA_AMBIGUOUS)
        if ambiguous_allele_mask.any():
            df.loc[ambiguous_allele_mask, "is_quarantined"] = True
            logger.info(
                "Re-quarantined %d rows with ambiguous HLA alleles after state reset",
                int(ambiguous_allele_mask.sum()),
            )

    # Compute per-virus stats (only for viral rows - not Self/Tumor)
    viral_mask = df["source_type"].fillna("") == "Virus"
    virus_counts = (
        df.loc[viral_mask, "virus"].value_counts().to_dict()
    )
    # Real tested negatives per virus: both bulk-export ("tested_negative") and
    # API-bridge ("iedb_api") origins represent genuine assay-confirmed negatives.
    _real_neg_origins = {"tested_negative", "iedb_api"}
    real_neg_mask = viral_mask & (
        df["negative_origin"].fillna("").isin(_real_neg_origins)
    )
    real_neg_counts = (
        df.loc[real_neg_mask, "virus"].value_counts().to_dict()
    )

    quarantined_viruses: list[str] = []
    for virus, n_rows in virus_counts.items():
        n_real_neg = real_neg_counts.get(virus, 0)
        if n_rows < MIN_ROWS_PER_VIRUS or n_real_neg < MIN_REAL_NEGATIVES_PER_VIRUS:
            quarantined_viruses.append(virus)

    # Quarantine viruses with null virus_family: non-canonical or non-virus entries.
    if "virus_family" in df.columns:
        null_fam_mask = viral_mask & df["virus_family"].isna()
        null_fam_viruses = (
            df.loc[null_fam_mask, "virus"].dropna().unique().tolist()
        )
        for v in null_fam_viruses:
            if v not in quarantined_viruses:
                quarantined_viruses.append(v)
                logger.warning(
                    "Quarantined %s: virus_family is null (not in VIRUS_FAMILY_MAP)",
                    v,
                )

    if quarantined_viruses:
        mask_quar = df["virus"].isin(quarantined_viruses)
        df.loc[mask_quar, "is_quarantined"] = True
        logger.warning(
            "Quarantined %d viruses (< %d rows OR < %d real negatives): %s",
            len(quarantined_viruses),
            MIN_ROWS_PER_VIRUS,
            MIN_REAL_NEGATIVES_PER_VIRUS,
            sorted(quarantined_viruses),
        )
    else:
        logger.info("No viruses quarantined; all meet depth thresholds.")

    # Print per-virus quarantine details
    for virus in sorted(quarantined_viruses):
        n_rows = virus_counts.get(virus, 0)
        n_real_neg = real_neg_counts.get(virus, 0)
        logger.info(
            "  QUARANTINED %s: %d total rows, %d real negatives",
            virus,
            n_rows,
            n_real_neg,
        )

    return df, quarantined_viruses


# ---------------------------------------------------------------------------
# Pre-build conflict audit (Part 16, Amendment 3)
# ---------------------------------------------------------------------------


def audit_label_conflicts(
    v4_positives: pd.DataFrame,
    iedb_negatives: pd.DataFrame,
    out_path: Path,
    logger: logging.Logger,
) -> int:
    """Detect v4-positive / IEDB-negative label collisions and write them out.

    A collision on the same (peptide, hla_allele, virus) key is otherwise
    silently suppressed by deduplication. The conflicting rows are written side
    by side (sorted so the two versions are adjacent) to ``out_path`` for manual
    review; Gavin confirms resolution before the first non-dry-run build.

    Returns the number of conflicting (peptide, hla_allele, virus) keys.
    """
    key_cols = ["peptide", "hla_allele", "virus"]
    if not all(c in v4_positives.columns for c in key_cols) or not all(
        c in iedb_negatives.columns for c in key_cols
    ):
        logger.warning(
            "Conflict audit skipped: key columns %s missing from a source",
            key_cols,
        )
        return 0

    def _key(df: pd.DataFrame) -> pd.Series:
        return df[key_cols].fillna("").apply(tuple, axis=1)

    pos_keys = set(_key(v4_positives))
    neg_keys = _key(iedb_negatives)
    conflict_keys = pos_keys.intersection(set(neg_keys))
    n_conflicts = len(conflict_keys)

    # Secondary, more permissive signal: same (peptide, hla_allele) across any
    # virus. IEDB virus names are not yet normalized to the canonical vocabulary
    # at this stage, so the strict triple key under-counts; this surfaces the
    # rest.
    pa_cols = ["peptide", "hla_allele"]
    pos_pa = set(v4_positives[pa_cols].fillna("").apply(tuple, axis=1))
    neg_pa = iedb_negatives[pa_cols].fillna("").apply(tuple, axis=1)
    pa_conflicts = int(neg_pa.isin(pos_pa).sum())

    logger.warning(
        "Label conflict audit: %d (peptide, hla_allele, virus) keys are "
        "v4-positive AND IEDB-negative; %d (peptide, hla_allele) collisions "
        "across any virus",
        n_conflicts,
        pa_conflicts,
    )

    if conflict_keys:
        pos_rows = v4_positives[_key(v4_positives).isin(conflict_keys)].copy()
        neg_rows = iedb_negatives[neg_keys.isin(conflict_keys)].copy()
        pos_rows["conflict_source"] = "v4_positive"
        neg_rows["conflict_source"] = "iedb_negative"
        combined = pd.concat([pos_rows, neg_rows], ignore_index=True, sort=False)
        sort_keys = [c for c in key_cols if c in combined.columns] + [
            "conflict_source"
        ]
        combined = combined.sort_values(sort_keys).reset_index(drop=True)
    else:
        combined = pd.DataFrame(
            columns=[*list(v4_positives.columns), "conflict_source"]
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    logger.info(
        "Wrote conflict pre-audit file: %s (%d rows)", out_path, len(combined)
    )
    return n_conflicts


def warn_low_pmid_depth(
    df: pd.DataFrame, logger: logging.Logger
) -> dict[str, int]:
    """Warn if any target virus is backed by fewer than the minimum distinct
    PMIDs among its non-quarantined rows. Returns per-target distinct PMID
    counts. Not a quarantine trigger - a visibility check (Amendment 3)."""
    counts: dict[str, int] = {}
    if "reference_pmid" not in df.columns or "virus" not in df.columns:
        logger.warning("PMID depth check skipped: required columns missing")
        return counts

    quar = (
        df["is_quarantined"]
        if "is_quarantined" in df.columns
        else pd.Series(False, index=df.index)
    ).fillna(False).astype(bool)

    for virus in TARGET_VIRUSES:
        mask = (df["virus"] == virus) & (~quar)
        pmids = df.loc[mask, "reference_pmid"].dropna().astype(str).str.strip()
        pmids = pmids[pmids != ""]
        n_distinct = int(pmids.nunique())
        counts[virus] = n_distinct
        if n_distinct < MIN_DISTINCT_PMIDS_PER_TARGET:
            logger.warning(
                "PMID depth: target virus %s backed by only %d distinct "
                "PMID(s) (< %d). A virus backed by few publications is fragile.",
                virus,
                n_distinct,
                MIN_DISTINCT_PMIDS_PER_TARGET,
            )
    logger.info("Target-virus distinct PMID counts: %s", counts)
    return counts


# ---------------------------------------------------------------------------
# Schema validation (data-ingest rule 3)
# ---------------------------------------------------------------------------


def validate_output_schema(
    df: pd.DataFrame, schema_path: Path, logger: logging.Logger
) -> None:
    """Validate required columns against the v5 JSON Schema before write."""
    if not schema_path.exists():
        logger.warning("Schema file not found: %s - skipping validation", schema_path)
        return
    try:
        with open(schema_path, encoding="utf-8") as fh:
            schema = json.load(fh)
    except Exception as exc:
        logger.warning("Could not load schema %s: %s - skipping validation", schema_path, exc)
        return
    required_cols = schema.get("items", {}).get("required", [])
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Output dataset missing required schema columns: {missing}. "
            f"Check V5_COLUMNS and transformation pipeline."
        )
    logger.info(
        "Schema validation passed: all %d required columns present", len(required_cols)
    )


# ---------------------------------------------------------------------------
# Source composition helpers
# ---------------------------------------------------------------------------


def extract_v4_sources(
    base_df: pd.DataFrame, logger: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Split base v4 dataset into positives and hard decoys."""
    source_counts: dict[str, int] = {}

    # Positives: viral rows with label=1
    if "source_type" in base_df.columns:
        pos_mask = (
            base_df["label"] == 1
        ) & base_df["source_type"].isin(["Virus", "VDJdb"])
        # Also include rows without source_type if label=1
        pos_mask_no_type = (base_df["label"] == 1) & ~base_df["source_type"].isin(
            ["Virus", "VDJdb", "Self", "Tumor", "Unknown"]
        )
        positives = base_df[pos_mask | pos_mask_no_type].copy()
    else:
        positives = base_df[base_df["label"] == 1].copy()

    source_counts["v4_positives"] = len(positives)
    logger.info("v4 positives (Virus/VDJdb, label=1): %d rows", len(positives))

    # Hard decoys: Self source_type or label=0 non-IEDB-tested
    if "source_type" in base_df.columns:
        decoy_mask = base_df["source_type"] == "Self"
        # Also include label=0 rows not already selected as positives
        other_neg_mask = (base_df["label"] == 0) & ~decoy_mask
        decoys = base_df[decoy_mask].copy()
        source_counts["v4_hard_decoys"] = len(decoys)
        source_counts["v4_other_negatives"] = int(other_neg_mask.sum())
        logger.info("v4 hard decoys (source_type=Self): %d rows", len(decoys))
    else:
        decoys = base_df[base_df["label"] == 0].copy()
        source_counts["v4_hard_decoys"] = len(decoys)

    return positives, decoys, source_counts


def ensure_v5_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing v5 columns with null defaults."""
    defaults: dict[str, object] = {
        "virus_family": None,
        "negative_origin": None,
        "assay_type": None,
        "assay_quality_weight": None,
        "assay_quality_tier": None,
        "reference_pmid": None,
        "iedb_assay_id": None,
        "infection_phase": None,
        "antigen_latency_program": None,
        "assay_context": None,
        "cross_reactivity_tested": None,
        "virus_taxon_id": None,
        "is_quarantined": False,
        "strain": None,
        "protein": "",
        "tcr_alpha_cdr3": None,
        "tcr_beta_cdr3": None,
    }
    df = df.copy()
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def build_virus_composition_table(
    df: pd.DataFrame,
) -> list[dict[str, object]]:
    rows = []
    for virus, group in df.groupby("virus", dropna=False):
        n = len(group)
        n_pos = int((group["label"] == 1).sum())
        n_neg = int((group["label"] == 0).sum())
        n_real_neg = int(
            group.get("negative_origin", pd.Series(dtype=str))
            .fillna("")
            .isin({"tested_negative", "iedb_api"})
            .sum()
        )
        pos_rate = round(n_pos / n, 4) if n > 0 else 0.0
        quarantined = bool(group["is_quarantined"].any()) if "is_quarantined" in group.columns else False
        rows.append(
            {
                "virus": str(virus) if virus is not None else "null",
                "n": n,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_real_neg": n_real_neg,
                "pos_rate": pos_rate,
                "is_quarantined": quarantined,
            }
        )
    rows.sort(key=lambda r: cast(int, r["n"]), reverse=True)
    return rows


def print_composition_table(
    df: pd.DataFrame,
    quarantine_list: list[str],
    logger: logging.Logger,
) -> None:
    """Print a human-readable composition summary."""
    print("\n=== v5 Dataset Composition ===")
    print(f"Total rows: {len(df)}")
    print(f"Positives:  {int((df['label'] == 1).sum())}")
    print(f"Negatives:  {int((df['label'] == 0).sum())}")
    print("\nPer-virus breakdown (viral rows only):")
    print(f"  {'Virus':<20} {'N':>6} {'Pos%':>6} {'RealNeg':>8} {'Quarantined':>12}")
    print("  " + "-" * 56)

    viral_df = df[df.get("source_type", pd.Series("Virus")) == "Virus"] if "source_type" in df.columns else df
    for _, row in pd.DataFrame(build_virus_composition_table(viral_df)).iterrows():
        qflag = "YES" if row["is_quarantined"] else "-"
        pos_pct = f"{row['pos_rate'] * 100:.1f}%"
        print(
            f"  {str(row['virus']):<20} {row['n']:>6} {pos_pct:>6} "
            f"{row['n_real_neg']:>8} {qflag:>12}"
        )

    if quarantine_list:
        print(f"\nQuarantined viruses ({len(quarantine_list)}): {sorted(quarantine_list)}")
    else:
        print("\nNo viruses quarantined.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_dataset_v5.py",
        description=(
            "Assemble the SESTRAV v5 immunogenicity dataset from v4 positives, "
            "IEDB negatives, hard decoys, and optional published panels. "
            "Applies HPV normalization, virus_family labeling, and singleton "
            "quarantine. v4 is never modified."
        ),
    )
    parser.add_argument(
        "--base-dataset",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to frozen v4 dataset CSV (read-only).",
    )
    parser.add_argument(
        "--iedb-negatives",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to IEDB negatives CSV (output of ingest_iedb_negatives.py).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path for v5 output CSV.",
    )
    parser.add_argument(
        "--published-panels",
        type=Path,
        metavar="PATH",
        default=None,
        help="Optional path to additional published panels CSV in v5 schema format.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        metavar="PATH",
        default=Path("data/immunogenicity_dataset_v5_schema.json"),
        help="Path to v5 JSON Schema for output validation (default: %(default)s).",
    )
    parser.add_argument(
        "--conflict-audit-path",
        type=Path,
        metavar="PATH",
        default=Path("data/holding/conflicts_v5_preaudit.csv"),
        help=(
            "Path for the v4-positive / IEDB-negative conflict pre-audit CSV "
            "(default: %(default)s). Review and sign off before a non-dry-run "
            "build."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print composition table and quarantine list without writing files."
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
    logger = logging.getLogger("build_dataset_v5")

    # Validate inputs
    for path, name in [
        (args.base_dataset, "--base-dataset"),
        (args.iedb_negatives, "--iedb-negatives"),
    ]:
        if not path.exists():
            logger.error("%s not found: %s", name, path)
            return 1

    if args.published_panels and not args.published_panels.exists():
        logger.error("--published-panels not found: %s", args.published_panels)
        return 1

    # -----------------------------------------------------------------------
    # 1. Load base v4 dataset and split into positives and hard decoys
    # -----------------------------------------------------------------------
    base_df = load_csv(args.base_dataset, "v4 base dataset", logger)
    v4_positives, v4_decoys, source_counts = extract_v4_sources(base_df, logger)

    # -----------------------------------------------------------------------
    # 2. Load IEDB negatives
    # -----------------------------------------------------------------------
    iedb_neg_df = load_csv(args.iedb_negatives, "IEDB negatives", logger)
    source_counts["iedb_negatives"] = len(iedb_neg_df)

    # -----------------------------------------------------------------------
    # 2b. Pre-build conflict audit (Amendment 3): v4-positive vs IEDB-negative
    #     label collisions. Always run so the file exists for review even on a
    #     dry-run; sign-off is required before a non-dry-run build.
    # -----------------------------------------------------------------------
    conflict_count = audit_label_conflicts(
        v4_positives, iedb_neg_df, args.conflict_audit_path, logger
    )
    source_counts["label_conflicts"] = conflict_count

    # -----------------------------------------------------------------------
    # 3. Load optional published panels
    # -----------------------------------------------------------------------
    panels_df: pd.DataFrame | None = None
    if args.published_panels:
        panels_df = load_csv(
            args.published_panels, "published panels", logger
        )
        source_counts["published_panels"] = len(panels_df)

        # Warn when a panel-positive (peptide, hla_allele) is also an IEDB
        # negative. Dedup resolves correctly (panels win by priority), but the
        # conflict is worth surfacing for audit: it means IEDB called the
        # peptide non-immunogenic while the panel says otherwise.
        _pa = ["peptide", "hla_allele"]
        if (
            "label" in panels_df.columns
            and all(c in panels_df.columns for c in _pa)
            and all(c in iedb_neg_df.columns for c in _pa)
        ):
            _panel_pos_keys = set(
                panels_df.loc[panels_df["label"] == 1, _pa]
                .fillna("")
                .apply(tuple, axis=1)
            )
            _n_conflict = int(
                iedb_neg_df[_pa].fillna("").apply(tuple, axis=1)
                .isin(_panel_pos_keys)
                .sum()
            )
            if _n_conflict:
                logger.warning(
                    "Panel/IEDB conflict: %d panel-positive (peptide, hla_allele) "
                    "pairs are also IEDB-negative; dedup keeps panel version (higher priority)",
                    _n_conflict,
                )
                source_counts["panel_iedb_conflicts"] = _n_conflict

    # -----------------------------------------------------------------------
    # 4. Ensure all parts have v5 columns before concat
    # -----------------------------------------------------------------------
    # Priority order for keep-first dedup below:
    #   curated v4 positives > published panels > self-proteome decoys > IEDB negatives
    parts: list[pd.DataFrame] = [v4_positives]
    if panels_df is not None:
        parts.append(panels_df)
    parts.extend([v4_decoys, iedb_neg_df])

    parts = [ensure_v5_columns(p) for p in parts]

    merged = pd.concat(parts, ignore_index=True, sort=False)
    logger.info("Merged pre-dedup: %d rows", len(merged))

    # Normalize virus names before dedup: IEDB negatives use full taxonomy names
    # ("Influenza A virus") while IEDB positives use short codes ("IAV"). Without
    # this step the same epitope from different sources never merges on virus label.
    merged, n_virus_resolved = normalize_virus_names(merged)
    if n_virus_resolved:
        logger.info("Virus name normalization: %d rows remapped", n_virus_resolved)

    # Resolve low-resolution HLA aliases (e.g. HLA-A2 -> HLA-A*02:01) before
    # dedup so that alias rows correctly merge with canonical-format rows from
    # other sources. Ambiguous entries ("HLA class I") are quarantined.
    merged, hla_norm_summary = normalize_hla_alleles(merged)
    if hla_norm_summary["aliases_resolved"]:
        logger.info("HLA normalization: %s", hla_norm_summary)

    # Deduplicate on (peptide, hla_allele), keeping the highest-priority source.
    # audit_label_conflicts above writes a CSV for v4-positive / IEDB-negative
    # cross-label conflicts but does not abort; any surviving cross-label pairs
    # are resolved here by keep-first priority order.
    pre_dedup_len = len(merged)
    merged = merged.drop_duplicates(subset=["peptide", "hla_allele"], keep="first")
    n_dedup_dropped = pre_dedup_len - len(merged)
    if n_dedup_dropped:
        logger.info("Dedup dropped %d overlapping-source (peptide, hla_allele) rows (same- or cross-label)", n_dedup_dropped)
    source_counts["merged_total"] = len(merged)
    source_counts["dedup_dropped"] = n_dedup_dropped

    # -----------------------------------------------------------------------
    # 5. Transformations
    # -----------------------------------------------------------------------
    # 5a. HPV normalization
    merged = normalize_hpv(merged, logger)

    # 5b. virus_family assignment
    merged = assign_virus_family(merged, logger)

    # 5c. negative_origin backfill for v4 Self rows
    merged = backfill_negative_origin(merged, logger)

    # -----------------------------------------------------------------------
    # 6. Singleton quarantine
    # -----------------------------------------------------------------------
    merged, quarantine_list = apply_quarantine(merged, logger)

    # 6b. PMID-depth visibility check for the target viruses (Amendment 3).
    target_pmid_counts = warn_low_pmid_depth(merged, logger)

    # -----------------------------------------------------------------------
    # 7. Column alignment and ordering
    # -----------------------------------------------------------------------
    for col in V5_COLUMNS:
        if col not in merged.columns:
            merged[col] = None

    extra_cols = [c for c in merged.columns if c not in V5_COLUMNS]
    if extra_cols:
        logger.info("Extra columns not in v5 spec (retained): %s", extra_cols)

    ordered_cols = V5_COLUMNS + extra_cols
    merged = merged[ordered_cols]

    # -----------------------------------------------------------------------
    # 8. Schema validation before write (data-ingest rule 3)
    # -----------------------------------------------------------------------
    validate_output_schema(merged, args.schema, logger)

    # -----------------------------------------------------------------------
    # 9. Report / dry-run
    # -----------------------------------------------------------------------
    composition_table = build_virus_composition_table(merged)
    print_composition_table(merged, quarantine_list, logger)

    if args.dry_run:
        print("\n=== DRY RUN - no files written ===")
        return 0

    # -----------------------------------------------------------------------
    # 10. Write output CSV
    # -----------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    logger.info("Wrote v5 dataset: %d rows to %s", len(merged), args.output)

    # -----------------------------------------------------------------------
    # 11. Write provenance sidecar JSON
    # -----------------------------------------------------------------------
    provenance_path = args.output.with_name(
        args.output.stem + "_provenance.json"
    )
    checksum = compute_file_sha256(args.output)
    provenance = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "source_counts": source_counts,
        "quarantine_list": sorted(quarantine_list),
        "quarantine_thresholds": {
            "min_rows": MIN_ROWS_PER_VIRUS,
            "min_real_negatives": MIN_REAL_NEGATIVES_PER_VIRUS,
        },
        "label_conflicts": conflict_count,
        "target_virus_pmid_counts": target_pmid_counts,
        "virus_composition_table": composition_table,
        "output_file": str(args.output),
        "output_checksum_sha256": checksum,
    }

    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    logger.info("Wrote provenance sidecar: %s", provenance_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
