"""
SESTRAV Automated Data QC Admissibility Gate (Stage 2.4)
=========================================================

Checks datasets for strict adherence to biological and dataset governance policies:
1. Completeness & Metadata: No null/empty peptide sequences or label values.
2. Canonical Amino Acids: Peptide sequences must consist of the 20 standard residues.
3. Peptide Lengths: Peptide lengths must be strictly within the 8-11mer range.
4. Duplicate Label Conflicts: Groups by peptide (+ allele if present) to ensure conflict ratio is <= 15%.
5. Null Allele Fraction: Checks that the fraction of missing allele data is within
   dataset_governance.qc_thresholds.max_null_allele_fraction. Fails closed when no
   allele column can be resolved, since an unmeasurable fraction is not a passing one.
6. Class Ratio Governance: Checks that the Positive:Negative ratio is within
   dataset_governance.qc_thresholds.class_ratio_bounds. The bound is corpus-specific
   and is derived in docs/data_qc_criteria.md; do not hardcode a window here.
7. Minimum Yield: Post-curation unique peptide count must be >= min_peptide_yield (default 500).

Usage:
    python scripts/data_qc_gate.py --dataset data/immunogenicity_dataset_v4.csv
    python scripts/data_qc_gate.py --dataset data/raw/iedb_tcell_assay/ebv_20260606.csv --quarantine results/qc/quarantined_ebv.csv
"""

import argparse
import hashlib
import logging
import os
import sys
import yaml

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("data_qc_gate")

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Canonical allele column names, most specific first. "hla_allele" is the
# repo-wide convention (scripts/_dataset_utils.py, scripts/check_panel_duplicates.py,
# scripts/build_dataset_v5.py); this gate previously recognised only a bare
# "allele", which no shipped corpus uses, so the null-allele check had never
# measured anything on a real dataset.
ALLELE_COL_PRIORITY = ("hla_allele", "allele", "mhc_allele")

# Values that mean "no allele recorded" but are not NaN. Builders write string
# sentinels rather than nulls, so an isna() count alone under-reports.
NULL_ALLELE_TOKENS = {"", "nan", "none", "null", "na", "n/a", "unknown", "-"}


def _is_null_allele(value) -> bool:
    """True when an allele cell carries no usable allele, NaN or sentinel."""
    if pd.isna(value):
        return True
    return str(value).strip().lower() in NULL_ALLELE_TOKENS


def load_config(config_path: str) -> dict:
    """Load config.yaml and return thresholds with defaults."""
    defaults = {
        "min_peptide_yield": 500,
        "max_conflict_ratio": 0.15,
        "max_null_allele_fraction": 0.50,
        "class_ratio_bounds": [0.18, 0.29],
        "freeze_mode": False,
    }
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}. Using standard defaults.")
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            gov = config.get("dataset_governance", {})
            thresholds = gov.get("qc_thresholds", {})

            # Extract values with safe fallbacks
            return {
                "min_peptide_yield": thresholds.get(
                    "min_peptide_yield", defaults["min_peptide_yield"]
                ),
                "max_conflict_ratio": thresholds.get(
                    "max_conflict_ratio", defaults["max_conflict_ratio"]
                ),
                "max_null_allele_fraction": thresholds.get(
                    "max_null_allele_fraction", defaults["max_null_allele_fraction"]
                ),
                "class_ratio_bounds": thresholds.get(
                    "class_ratio_bounds", defaults["class_ratio_bounds"]
                ),
                "freeze_mode": config.get("freeze_mode", defaults["freeze_mode"]),
                "expected_checksum": gov.get("provenance", {}).get("checksum", "pending"),
                "require_checksum": gov.get("require_checksum_match_in_freeze_mode", False),
            }
    except Exception as e:
        logger.error(f"Failed to parse config: {e}. Refusing defaults that disable freeze.")
        raise


def check_dataset_qc(
    dataset_path: str, config_path: str, quarantine_path: str | None = None
) -> bool:
    """
    Load dataset and execute all QC checks.
    Quarantined rows are written to quarantine_path if provided.
    Returns True if all checks pass, False on any strict check failure.
    """
    logger.info(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        return False

    # Compute dataset checksum for MLOps tracking.
    # Normalize CRLF -> LF before hashing so Windows checkouts produce the
    # same digest as Linux CI (git autocrlf expands LF to CRLF on Windows).
    sha256 = hashlib.sha256()
    with open(dataset_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk.replace(b"\r\n", b"\n"))
    checksum = sha256.hexdigest()
    logger.info(f"Computed dataset SHA-256 Checksum: {checksum}")

    # Load configuration thresholds
    cfg = load_config(config_path)
    logger.info(f"Active QC thresholds: {cfg}")

    # Verify checksum matches if freeze mode is active
    if cfg["freeze_mode"] and cfg.get("require_checksum", False):
        expected = cfg.get("expected_checksum", "pending")
        if expected != "pending" and expected != checksum:
            logger.error(
                f"Freeze mode violation! Dataset checksum {checksum} does not match expected {expected}."
            )
            return False

    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        logger.error(f"Failed to read CSV dataset: {e}")
        return False

    logger.info(f"Initial row count: {len(df)}")

    # fuzzy search and normalize columns
    col_map = {}
    for col in df.columns:
        cl = str(col).lower().strip()
        if cl == "peptide" or ("peptide" in cl and "sequence" in cl) or cl == "description":
            col_map.setdefault("peptide", col)
        elif cl == "label" or "qualitative" in cl:
            col_map.setdefault("label", col)
        elif cl == "allele" or "mhc present" in cl:
            col_map.setdefault("allele", col)

    # An exact canonical name wins over whatever column order happened to reach
    # setdefault() first above. The "mhc present" heuristic stays as a fallback.
    lower_to_orig = {str(c).lower().strip(): c for c in df.columns}
    for _canonical in ALLELE_COL_PRIORITY:
        if _canonical in lower_to_orig:
            col_map["allele"] = lower_to_orig[_canonical]
            break

    if "peptide" not in col_map or "label" not in col_map:
        logger.error(
            f"Required columns (Peptide and Label/Qualitative Measure) not found. Columns: {list(df.columns)}"
        )
        return False

    pep_col = col_map["peptide"]
    lbl_col = col_map["label"]
    allele_col = col_map.get("allele")
    logger.info(f"Mapped columns: peptide->'{pep_col}', label->'{lbl_col}', allele->'{allele_col}'")

    # Keep track of quarantined rows with reasons
    quarantined_rows = []

    def add_to_quarantine(idx, row, reason):
        row_dict = row.to_dict()
        row_dict["qc_failure_reason"] = reason
        row_dict["original_index"] = idx
        quarantined_rows.append(row_dict)

    # List of indices to drop from the valid dataset
    indices_to_drop = set()

    # Iterate through records to perform row-level filtering
    for idx, row in df.iterrows():
        pep = row[pep_col]
        lbl = row[lbl_col]

        # 1. Check completeness
        if pd.isna(pep) or pd.isna(lbl):
            add_to_quarantine(idx, row, "Missing peptide sequence or label outcome")
            indices_to_drop.add(idx)
            continue

        pep_str = str(pep).strip().upper()

        # 2. Check canonical amino acids
        if not all(c in STANDARD_AA for c in pep_str):
            add_to_quarantine(idx, row, "Non-canonical amino acids present in sequence")
            indices_to_drop.add(idx)
            continue

        # 3. Check peptide length window (8-11mer)
        if not (8 <= len(pep_str) <= 11):
            add_to_quarantine(
                idx, row, f"Peptide length {len(pep_str)} outside valid 8-11mer window"
            )
            indices_to_drop.add(idx)
            continue

    # Create clean dataset subset
    df_clean = df.drop(index=list(indices_to_drop)).copy()

    # Rename columns to standard ones
    df_clean = df_clean.rename(columns={pep_col: "peptide", lbl_col: "label"})
    if allele_col:
        df_clean = df_clean.rename(columns={allele_col: "allele"})

    # Parse labels to binary integers
    def parse_binary_label(val):
        if pd.isna(val):
            return np.nan
        vs = str(val).strip().lower()
        if vs in ("1", "1.0", "positive", "positive-high", "positive-low", "positive-intermediate"):
            return 1
        if vs in ("0", "0.0", "negative"):
            return 0
        return np.nan

    df_clean["label"] = df_clean["label"].apply(parse_binary_label)

    # Drop rows where labels couldn't be parsed
    unparseable_labels = df_clean[df_clean["label"].isna()]
    for idx, row in unparseable_labels.iterrows():
        add_to_quarantine(idx, row, "Unparseable label outcome (not positive/negative)")
        df_clean = df_clean.drop(index=idx)

    df_clean["label"] = df_clean["label"].astype(int)

    logger.info(f"Row count after row-level QC filters: {len(df_clean)}")

    # 4. Duplicate conflicts check
    # We check duplicate conflicts on the cleaned dataset.
    # Group by peptide (+ allele if present) and count unique labels
    # Normalise the allele into an explicit key: pandas groupby drops NaN keys by
    # default, which would silently exclude every null-allele row from the
    # conflict check entirely.
    if "allele" in df_clean.columns:
        df_clean["_allele_key"] = df_clean["allele"].map(
            lambda v: "<MISSING>" if _is_null_allele(v) else str(v).strip()
        )
        group_cols = ["peptide", "_allele_key"]
    else:
        group_cols = ["peptide"]
    label_counts = df_clean.groupby(group_cols)["label"].nunique()
    conflicting_mask = label_counts > 1

    total_unique_groups = len(label_counts)
    conflicting_groups = conflicting_mask.sum()
    conflict_ratio = conflicting_groups / total_unique_groups if total_unique_groups > 0 else 0.0

    logger.info(
        f"Deduplication Groups: {total_unique_groups} | Conflicting Groups: {conflicting_groups}"
    )
    logger.info(
        f"Conflict Ratio: {conflict_ratio:.4f} (Max threshold: {cfg['max_conflict_ratio']})"
    )

    # Quarantine conflicting rows
    if conflicting_groups > 0:
        # Get the keys of conflicting groups
        conflicting_keys = label_counts[conflicting_mask].index
        if "_allele_key" in df_clean.columns:
            conflict_set = set(conflicting_keys)

            def is_conflict(r):
                return (r["peptide"], r["_allele_key"]) in conflict_set
        else:
            conflict_set = set(conflicting_keys)

            def is_conflict(r):
                return r["peptide"] in conflict_set

        conflict_rows_mask = df_clean.apply(is_conflict, axis=1)
        conflict_df = df_clean[conflict_rows_mask]
        for idx, row in conflict_df.iterrows():
            add_to_quarantine(idx, row, "Conflicting immunogenicity assays for the same group")
            df_clean = df_clean.drop(index=idx)

    # 5. Null allele fraction check
    if "_allele_key" in df_clean.columns:
        df_clean = df_clean.drop(columns=["_allele_key"])
    total_records = len(df_clean)
    allele_column_resolved = "allele" in df_clean.columns
    if allele_column_resolved:
        nan_only_count = int(df_clean["allele"].isna().sum())
        null_allele_count = int(df_clean["allele"].map(_is_null_allele).sum())
        logger.info(
            f"Null allele rows: {null_allele_count} "
            f"(NaN-only would report {nan_only_count})"
        )
        null_allele_fraction = null_allele_count / total_records if total_records > 0 else 1.0
    else:
        null_allele_count = total_records
        null_allele_fraction = 1.0
        logger.error(
            f"No allele column resolved (looked for {ALLELE_COL_PRIORITY}). "
            "The null-allele fraction is UNMEASURABLE, not 0; this check fails closed."
        )

    logger.info(
        f"Null Allele Fraction: {null_allele_fraction:.4f} (Max threshold: {cfg['max_null_allele_fraction']})"
    )

    # 6. Class ratio check
    positives = (df_clean["label"] == 1).sum()
    negatives = (df_clean["label"] == 0).sum()
    class_ratio = positives / negatives if negatives > 0 else float("inf")
    logger.info(f"Class counts: Positive={positives}, Negative={negatives}")
    logger.info(f"Class Ratio (Pos:Neg): {class_ratio:.4f} (Bounds: {cfg['class_ratio_bounds']})")

    # 7. Minimum yield check
    unique_peptides = df_clean["peptide"].nunique()
    logger.info(
        f"Unique peptide yield: {unique_peptides} (Min yield expected: {cfg['min_peptide_yield']})"
    )

    # Write quarantine output if path is provided
    if quarantine_path and quarantined_rows:
        q_dir = os.path.dirname(quarantine_path)
        if q_dir:
            os.makedirs(q_dir, exist_ok=True)
        pd.DataFrame(quarantined_rows).to_csv(quarantine_path, index=False)
        logger.info(f"Wrote {len(quarantined_rows)} quarantined rows to: {quarantine_path}")

    # Evaluate all checks
    checks = {
        "length_and_composition_valid": len(indices_to_drop) == 0,
        "conflict_ratio_passed": conflict_ratio <= cfg["max_conflict_ratio"],
        "allele_column_resolved": allele_column_resolved,
        "null_allele_fraction_passed": allele_column_resolved
        and null_allele_fraction <= cfg["max_null_allele_fraction"],
        "class_ratio_passed": cfg["class_ratio_bounds"][0]
        <= class_ratio
        <= cfg["class_ratio_bounds"][1],
        "peptide_yield_passed": unique_peptides >= cfg["min_peptide_yield"],
    }

    # Log results summary
    print("\n" + "=" * 50)
    print("       DATASET QC ADMISSIBILITY REPORT")
    print("=" * 50)
    for check_name, status in checks.items():
        symbol = "PASS" if status else "FAIL"
        print(f"  {check_name:32s} : {symbol}")
    print("=" * 50 + "\n")

    # Fail gate if any strict check fails
    success = all(checks.values())
    if success:
        logger.info("All dataset QC gates passed successfully.")
    else:
        logger.error("Dataset QC gate FAILED on one or more admissibility checks.")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="SESTRAV automated dataset admissibility QC checker."
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to the immunogenicity CSV dataset to check."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the config.yaml parameters file."
    )
    parser.add_argument("--quarantine", help="Optional path to output quarantined invalid records.")
    args = parser.parse_args()

    success = check_dataset_qc(
        dataset_path=args.dataset, config_path=args.config, quarantine_path=args.quarantine
    )

    if not success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
