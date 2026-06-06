"""
scripts/iedb_data_loader.py
===========================
Robust IEDB CSV export processor for the SESTRAV immunogenicity pipeline.

Pipeline stages
---------------
1. **Filtering**
   - Drop rows whose peptide length falls outside [8, 11] amino acids.
   - Drop rows containing any character outside the 20 standard amino acids
     (ACDEFGHIKLMNPQRSTVWY).

2. **Allele extraction**
   - Isolate the "MHC Allele Name" column (case/whitespace-normalised lookup).
   - Drop rows with null or empty allele data.
   - Normalise allele strings to canonical HLA notation (HLA-A*02:01).

3. **Deduplication & conflict resolution** (grouped by peptide + allele)
   - Compute mean and variance of binary assay labels per group.
   - mean > 0.5  → label = 1
   - mean < 0.5  → label = 0
   - mean == 0.5 (exact tie) → DROP the peptide-allele pair
   - variance > CONFLICT_VAR_THRESHOLD → DROP (highly conflicting assays)

4. **Output**
   - Write the clean dataset to ``immunogenicity_dataset.csv`` (configurable).

Usage
-----
    # Minimal — uses defaults
    python scripts/iedb_data_loader.py path/to/iedb_export.csv

    # Full options
    python scripts/iedb_data_loader.py path/to/iedb_export.csv \\
        --output results/clean_dataset.csv \\
        --conflict-threshold 0.20 \\
        --verbose

Column name detection
---------------------
The script performs a case-insensitive, whitespace-normalised search for:
  * Peptide sequence : columns whose normalised name contains "peptide" and
                       "sequence", OR equals "description", OR equals "name".
  * MHC allele       : columns whose normalised name contains "allele" OR
                       "mhc present" OR "mhc restriction - name".
  * Assay outcome    : columns whose normalised name contains "qualitative".

If the IEDB export uses a multi-level header (two header rows), pass
``--multiheader`` and the script will automatically flatten it.
"""

import argparse
import hashlib
import logging
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The 20 canonical amino acid single-letter codes (IUPAC).
STANDARD_AA: frozenset = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: Peptide length window accepted for MHC class-I binding prediction.
PEPTIDE_MIN_LEN: int = 8
PEPTIDE_MAX_LEN: int = 11

#: Default variance threshold above which a peptide-allele pair is considered
#: "highly conflicting" and is dropped.  For Bernoulli-distributed binary
#: labels the maximum possible variance is 0.25 (p=0.5).  A value of 0.20
#: keeps pairs where ≥80 % of assays agree.
DEFAULT_CONFLICT_VAR_THRESHOLD: float = 0.20

#: Default output path (relative to CWD when invoked from the project root).
DEFAULT_OUTPUT_PATH: str = "immunogenicity_dataset.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _sha256_df(df: pd.DataFrame) -> str:
    """Compute a stable SHA-256 fingerprint of a DataFrame's CSV representation."""
    buf = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(buf).hexdigest()


def _normalise_col_name(col: str) -> str:
    """Lower-case and collapse whitespace for fuzzy column matching."""
    return " ".join(str(col).lower().split())


def _find_column(df: pd.DataFrame, *patterns: str) -> str | None:
    """Return the first column whose normalised name contains any of *patterns*.

    Parameters
    ----------
    df:
        The DataFrame to search.
    *patterns:
        Substrings (already lower-cased) to match against normalised names.

    Returns
    -------
    str | None
        The original column name, or ``None`` if no match is found.
    """
    for col in df.columns:
        norm = _normalise_col_name(col)
        if any(p in norm for p in patterns):
            return col
    return None


def _find_peptide_column(df: pd.DataFrame) -> str | None:
    """Locate the peptide-sequence column using IEDB naming conventions."""
    # Priority 1: explicit "Peptide Sequence" / "Linear Sequence" style
    for col in df.columns:
        norm = _normalise_col_name(col)
        if ("peptide" in norm and "sequence" in norm) or "linear sequence" in norm:
            return col
    # Priority 2: "Description" (T-cell Assay export legacy name)
    for col in df.columns:
        if _normalise_col_name(col) in ("description", "name"):
            return col
    # Priority 3: any column containing "epitope" + ("name" or "linear")
    for col in df.columns:
        norm = _normalise_col_name(col)
        if "epitope" in norm and ("name" in norm or "linear" in norm or "sequence" in norm):
            return col
    return None


def _find_allele_column(df: pd.DataFrame) -> str | None:
    """Locate the MHC allele column using IEDB naming conventions."""
    for col in df.columns:
        norm = _normalise_col_name(col)
        if (
            "allele" in norm
            or "mhc present" in norm
            or "mhc restriction - name" in norm
            or ("mhc" in norm and "name" in norm)
        ):
            return col
    return None


def _find_label_column(df: pd.DataFrame) -> str | None:
    """Locate the assay outcome column (Qualitative Measure / Measurement)."""
    return _find_column(df, "qualitative")


# ---------------------------------------------------------------------------
# Allele normalisation
# ---------------------------------------------------------------------------

MHC_CLASS_II_PREFIXES: tuple = ("HLA-DR", "HLA-DP", "HLA-DQ")


def normalise_allele(raw: str) -> str | None:
    """Canonicalise an HLA allele string to the form ``HLA-A*02:01``.

    Handles missing "HLA-" prefix, missing "*", and missing ":" separator.

    Parameters
    ----------
    raw:
        Raw allele string from the IEDB export.

    Returns
    -------
    str | None
        Canonicalised allele string, or ``None`` if the input is null/empty.
    """
    if pd.isna(raw) or not str(raw).strip():
        return None
    s = str(raw).strip()

    # Ensure "HLA-" prefix
    if not s.upper().startswith("HLA-"):
        s = "HLA-" + s

    # Insert "*" if absent, e.g. HLA-A0201 → HLA-A*0201
    if "*" not in s:
        for prefix in ("HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F", "HLA-G"):
            if s.upper().startswith(prefix):
                s = s[:len(prefix)] + "*" + s[len(prefix):]
                break

    # Insert ":" if absent in the allele group, e.g. HLA-A*0201 → HLA-A*02:01
    if "*" in s and ":" not in s:
        star_pos = s.index("*")
        suffix = s[star_pos + 1 :]
        if len(suffix) >= 4:
            s = s[: star_pos + 1] + suffix[:2] + ":" + suffix[2:]

    return s


def is_mhc_class_i(allele: str) -> bool:
    """Return ``True`` if *allele* is an MHC class-I allele (not DR/DP/DQ)."""
    return not allele.startswith(MHC_CLASS_II_PREFIXES)


# ---------------------------------------------------------------------------
# Core pipeline stages
# ---------------------------------------------------------------------------


def load_raw(filepath: str, multiheader: bool = False) -> pd.DataFrame:
    """Load a raw IEDB CSV export into a flat DataFrame.

    Parameters
    ----------
    filepath:
        Path to the CSV file.
    multiheader:
        If ``True``, read with ``header=[0, 1]`` and flatten the resulting
        MultiIndex columns by joining level strings with " - ".

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    if multiheader:
        df = pd.read_csv(filepath, header=[0, 1], low_memory=False)
        # Flatten MultiIndex: ('Epitope', 'Name') → 'Epitope - Name'
        df.columns = [
            " - ".join(str(lvl).strip() for lvl in col if str(lvl).strip()).strip(" -")
            for col in df.columns
        ]
        logger.info("Loaded multi-header CSV → flattened to %d columns", len(df.columns))
    else:
        # Auto-detect: peek at row 1 to see if there is a sub-header
        peek = pd.read_csv(filepath, nrows=2, header=None)
        has_subheader = len(peek) > 1 and any(
            "qualitative" in str(v).lower() for v in peek.iloc[1]
        )
        if has_subheader:
            df = pd.read_csv(filepath, header=[0, 1], low_memory=False)
            df.columns = [
                " - ".join(str(lvl).strip() for lvl in col if str(lvl).strip()).strip(" -")
                for col in df.columns
            ]
            logger.info(
                "Auto-detected multi-header CSV → flattened to %d columns", len(df.columns)
            )
        else:
            df = pd.read_csv(filepath, low_memory=False)

    logger.info("Raw shape: %d rows × %d columns", *df.shape)
    return df


def parse_label(value) -> int | None:
    """Map an IEDB Qualitative Measure string to a binary integer label.

    Returns
    -------
    int | None
        1 for any "Positive" variant, 0 for "Negative", ``None`` otherwise.
    """
    if pd.isna(value):
        return None
    v = str(value).strip().lower()
    if v.startswith("positive"):
        return 1
    if v == "negative":
        return 0
    return None


# ---------------------------------------------------------------------------
# Stage 1: Filtering
# ---------------------------------------------------------------------------


def stage_filter(df: pd.DataFrame, peptide_col: str) -> pd.DataFrame:
    """Apply length and amino-acid composition filters to *df*.

    Rows are retained when the peptide:
    - has length in [PEPTIDE_MIN_LEN, PEPTIDE_MAX_LEN], AND
    - consists exclusively of the 20 standard amino acids.

    Parameters
    ----------
    df:
        DataFrame containing at least *peptide_col*.
    peptide_col:
        Name of the column holding peptide sequences.

    Returns
    -------
    pd.DataFrame
        Filtered copy with peptides upper-cased and stripped.
    """
    before = len(df)
    df = df.copy()

    # Normalise: strip whitespace and upper-case
    df[peptide_col] = df[peptide_col].astype(str).str.strip().str.upper()

    # Compute length mask
    lengths = df[peptide_col].str.len()
    length_mask = lengths.between(PEPTIDE_MIN_LEN, PEPTIDE_MAX_LEN)

    # Compute amino-acid mask (vectorised via regex complement)
    non_standard_pattern = "[^ACDEFGHIKLMNPQRSTVWY]"
    aa_mask = ~df[peptide_col].str.contains(non_standard_pattern, regex=True, na=True)

    df = df[length_mask & aa_mask].reset_index(drop=True)
    dropped = before - len(df)
    logger.info(
        "Stage 1 (filter): %d → %d rows  (dropped %d — length/AA violations)",
        before,
        len(df),
        dropped,
    )
    return df


# ---------------------------------------------------------------------------
# Stage 2: Allele extraction
# ---------------------------------------------------------------------------


def stage_allele(df: pd.DataFrame, allele_col: str) -> pd.DataFrame:
    """Extract, normalise, and filter on the MHC allele column.

    Steps:
    1. Normalise raw allele strings to canonical HLA notation.
    2. Drop rows where the allele is null/empty after normalisation.
    3. Drop MHC class-II alleles (HLA-DR/DP/DQ).

    Parameters
    ----------
    df:
        Input DataFrame with *allele_col* present.
    allele_col:
        Name of the raw MHC allele column.

    Returns
    -------
    pd.DataFrame
        DataFrame with a clean ``allele`` column added.
    """
    before = len(df)
    df = df.copy()

    df["allele"] = df[allele_col].apply(normalise_allele)

    # Drop null alleles
    null_mask = df["allele"].isna()
    df = df[~null_mask]

    # Drop MHC class-II alleles
    class_ii_mask = ~df["allele"].apply(is_mhc_class_i)
    df = df[~class_ii_mask]

    dropped = before - len(df)
    logger.info(
        "Stage 2 (allele): %d → %d rows  (dropped %d — null/class-II alleles)",
        before,
        len(df),
        dropped,
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 3: Deduplication & conflict resolution
# ---------------------------------------------------------------------------


def stage_deduplicate(
    df: pd.DataFrame,
    label_col: str,
    conflict_var_threshold: float = DEFAULT_CONFLICT_VAR_THRESHOLD,
) -> pd.DataFrame:
    """Resolve duplicate (Peptide Sequence, allele) pairs.

    For each group the binary label mean and variance are computed across all
    contributing assay rows.

    Resolution rules
    ~~~~~~~~~~~~~~~~
    - ``mean > 0.5``  → label = **1** (immunogenic)
    - ``mean < 0.5``  → label = **0** (non-immunogenic)
    - ``mean == 0.5`` (exact tie) → **DROP**
    - ``variance > conflict_var_threshold`` → **DROP** (high conflict)

    Parameters
    ----------
    df:
        DataFrame with ``Peptide Sequence`` (or the detected peptide column,
        stored as ``peptide``), ``allele``, and the assay label column.
    label_col:
        Column name holding the binary (0/1) assay outcome.
    conflict_var_threshold:
        Variance above which a group is considered too conflicting to use.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame with columns: peptide, allele, label, n_assays,
        mean_label, var_label.
    """
    # Ensure the label column is numeric (int → float for aggregation)
    df = df.copy()
    df["_label_num"] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=["_label_num"])
    df["_label_num"] = df["_label_num"].astype(float)

    GROUP_KEYS = ["peptide", "allele"]
    before_groups = df.groupby(GROUP_KEYS).ngroups
    before_rows = len(df)

    agg = (
        df.groupby(GROUP_KEYS)["_label_num"]
        .agg(
            n_assays="count",
            mean_label="mean",
            var_label=lambda x: float(np.var(x, ddof=0)),  # population variance
        )
        .reset_index()
    )

    # --- Tie detection (exact 0.5 mean) ---
    tie_mask = agg["mean_label"] == 0.5
    n_ties = tie_mask.sum()

    # --- High-conflict detection (variance exceeds threshold) ---
    conflict_mask = agg["var_label"] > conflict_var_threshold
    n_conflicts = conflict_mask.sum()

    # Combined drop mask: ties OR high-conflict
    drop_mask = tie_mask | conflict_mask
    n_dropped = drop_mask.sum()

    # Retain clean groups
    clean = agg[~drop_mask].copy()

    # Assign final label
    clean["label"] = (clean["mean_label"] > 0.5).astype(int)

    # Drop internal helpers
    clean = clean.drop(columns=[], errors="ignore")

    logger.info(
        "Stage 3 (dedup): %d assay rows / %d groups → %d clean groups  "
        "(dropped %d: %d exact ties + %d high-conflict; var threshold=%.3f)",
        before_rows,
        before_groups,
        len(clean),
        n_dropped,
        n_ties,
        n_conflicts,
        conflict_var_threshold,
    )

    # Final column order
    out_cols = ["peptide", "allele", "label", "n_assays", "mean_label", "var_label"]
    return clean[out_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def process_iedb_export(
    filepath: str,
    output_path: str = DEFAULT_OUTPUT_PATH,
    conflict_var_threshold: float = DEFAULT_CONFLICT_VAR_THRESHOLD,
    multiheader: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """End-to-end IEDB export processing pipeline.

    Parameters
    ----------
    filepath:
        Path to the raw IEDB CSV export.
    output_path:
        Destination CSV path for the cleaned dataset.
    conflict_var_threshold:
        Variance threshold for dropping high-conflict peptide-allele pairs.
        Set to ``0.0`` to drop *any* pair with disagreeing assays.
        Set to ``0.25`` to keep all non-tie pairs regardless of conflict.
    multiheader:
        Force multi-level header parsing (``header=[0, 1]``).
    verbose:
        If ``True``, set logging level to DEBUG.

    Returns
    -------
    pd.DataFrame
        The final cleaned dataset (also written to *output_path*).
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        for h in logging.root.handlers:
            h.setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("SESTRAV IEDB Data Loader")
    logger.info("Input  : %s", filepath)
    logger.info("Output : %s", output_path)
    logger.info("Conflict variance threshold : %.3f", conflict_var_threshold)
    logger.info("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    raw_df = load_raw(filepath, multiheader=multiheader)

    # ── Detect required columns ───────────────────────────────────────────
    peptide_col = _find_peptide_column(raw_df)
    allele_col = _find_allele_column(raw_df)
    label_col = _find_label_column(raw_df)

    missing = []
    if peptide_col is None:
        missing.append("Peptide Sequence")
    if allele_col is None:
        missing.append("MHC Allele Name")
    if label_col is None:
        missing.append("Qualitative Measure/Measurement")

    if missing:
        detected = list(raw_df.columns)
        raise ValueError(
            f"Could not locate required column(s): {missing}\n"
            f"Detected columns: {detected}\n"
            "Tip: if this is a multi-level header export, retry with --multiheader."
        )

    logger.info(
        "Column mapping → peptide=%r  allele=%r  label=%r",
        peptide_col,
        allele_col,
        label_col,
    )

    # ── Stage 1: Filter ───────────────────────────────────────────────────
    stage1_df = stage_filter(raw_df, peptide_col)

    # Rename peptide column to canonical "peptide"
    if peptide_col != "peptide":
        stage1_df = stage1_df.rename(columns={peptide_col: "peptide"})

    # ── Map label column to binary integers ───────────────────────────────
    original_label_col = label_col if label_col != "peptide" else label_col + "_raw"
    stage1_df["_bin_label"] = stage1_df[label_col].apply(parse_label)

    # Drop rows where label could not be mapped (neither positive nor negative)
    unmapped = stage1_df["_bin_label"].isna().sum()
    if unmapped:
        logger.info("  Dropping %d rows with unmappable label (not positive/negative)", unmapped)
    stage1_df = stage1_df.dropna(subset=["_bin_label"]).reset_index(drop=True)

    # ── Stage 2: Allele extraction ────────────────────────────────────────
    stage2_df = stage_allele(stage1_df, allele_col)

    # ── Stage 3: Deduplication & conflict resolution ──────────────────────
    # Pass the binary label column we just created
    stage3_df = stage_deduplicate(
        stage2_df.rename(columns={"_bin_label": "_label_for_dedup"}),
        label_col="_label_for_dedup",
        conflict_var_threshold=conflict_var_threshold,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("─" * 60)
    logger.info("  Final dataset summary")
    logger.info("─" * 60)
    logger.info("  Rows (peptide-allele pairs) : %d", len(stage3_df))
    logger.info("  Unique peptides             : %d", stage3_df["peptide"].nunique())
    logger.info("  Unique alleles              : %d", stage3_df["allele"].nunique())
    logger.info(
        "  Class balance (positive)    : %.2f %%",
        stage3_df["label"].mean() * 100,
    )
    logger.info(
        "  Single-assay pairs          : %d",
        (stage3_df["n_assays"] == 1).sum(),
    )
    logger.info(
        "  Multi-assay pairs           : %d",
        (stage3_df["n_assays"] > 1).sum(),
    )

    # ── Write output ──────────────────────────────────────────────────────
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    stage3_df.to_csv(output_path, index=False)

    sha = _sha256_df(stage3_df)
    logger.info("")
    logger.info("  Output written to : %s", output_path)
    logger.info("  SHA-256           : %s", sha)
    logger.info("─" * 60)

    return stage3_df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iedb_data_loader",
        description=(
            "Process a raw IEDB CSV export into a clean immunogenicity dataset.\n\n"
            "Steps: (1) length/AA filter, (2) allele extraction, "
            "(3) dedup + conflict resolution."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        metavar="IEDB_CSV",
        help="Path to the raw IEDB CSV export file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        default=DEFAULT_OUTPUT_PATH,
        help=(
            f"Output CSV path for the cleaned dataset. "
            f"(default: {DEFAULT_OUTPUT_PATH})"
        ),
    )
    parser.add_argument(
        "--conflict-threshold",
        "-t",
        metavar="FLOAT",
        type=float,
        default=DEFAULT_CONFLICT_VAR_THRESHOLD,
        help=(
            "Population variance threshold for dropping high-conflict "
            "peptide-allele pairs.  Range [0.0, 0.25].  "
            f"(default: {DEFAULT_CONFLICT_VAR_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--multiheader",
        action="store_true",
        default=False,
        help=(
            "Force two-level header parsing (header=[0, 1]).  "
            "Use when the IEDB export has two header rows."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for ``scripts/iedb_data_loader.py``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate conflict threshold
    if not (0.0 <= args.conflict_threshold <= 0.25):
        parser.error(
            f"--conflict-threshold must be in [0.0, 0.25]; got {args.conflict_threshold}"
        )

    try:
        process_iedb_export(
            filepath=args.input,
            output_path=args.output,
            conflict_var_threshold=args.conflict_threshold,
            multiheader=args.multiheader,
            verbose=args.verbose,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Column detection failed:\n%s", exc)
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error: %s", exc)
        sys.exit(3)


if __name__ == "__main__":
    main()
