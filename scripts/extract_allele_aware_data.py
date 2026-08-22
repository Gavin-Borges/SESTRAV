"""
SESTRAV Allele-Aware Data Migration Script
==========================================

Reads IEDB T-cell Assay export files (EBV + HPV16), applies the existing
iedb_data_loader T-cell Assay parsing path, and produces an allele-aware
training dataset where each row represents a (peptide, allele) pair with
a majority-voted immunogenicity label.

Includes HLA pseudo-sequence mapping: each of the 10 target alleles is mapped
to its 34-residue MHC binding pocket pseudo-sequence (following the NetMHCpan
convention), expressed as 4 physicochemical features per position (136 dims).

Output dataset: data/allele_aware/IEDB-<date>-EBV_HPV16_ALLELE_AWARE-v1.csv

Usage:
    python scripts/extract_allele_aware_data.py --data-dir data/iedb
    python scripts/extract_allele_aware_data.py --data-dir data/iedb \\
        --output data/allele_aware/custom_output.csv --verbose
"""

import argparse
import os
import sys
import json
import hashlib
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES
from src.features import KD_HYDRO, VDW_VOL, AROMATIC, CHARGE

# ---------------------------------------------------------------------------
# Target 10-allele panel (compact -> standard)
# ---------------------------------------------------------------------------

TARGET_ALLELES = {
    "HLA-A*01:01",
    "HLA-A*02:01",
    "HLA-A*03:01",
    "HLA-A*11:01",
    "HLA-A*24:02",
    "HLA-B*07:02",
    "HLA-B*08:01",
    "HLA-B*27:05",
    "HLA-B*35:01",
    "HLA-B*44:02",
}

# ---------------------------------------------------------------------------
# HLA pocket pseudo-sequences (34 variable positions per NetMHCpan 4.1)
#
# CORRECTED 2026-08-20 (docs/claims_register.md D30). The previous table here
# was fabricated, and the features it produced were degenerate. Both, not one
# or the other: all five HLA-A entries shared one identical
# 35-char string and all five HLA-B entries shared a second identical 35-char
# string (two distinct values across ten alleles, both trimmed to 34 on load),
# neither matching any real allele's sequence, despite a comment claiming
# "Source: NetMHCpan 4.1 MHC_pseudo.dat". That comment named a real, publicly
# fetchable file whose contents refuted it - it was never actually checked
# against the source it cited.
#
# These ten values ARE now sourced from that file, verified directly:
#   URL:    https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/NetMHCpan_train.tar.gz
#   Member: NetMHCpan_train/MHC_pseudo.dat (11,373 non-blank lines, "ALLELE SEQUENCE" per line)
#   Tarball sha256: 06f2c9f20bb959238bf5d601fca0489a0ed3f17648b952f30640205afca8f9b4
#   Fetched and extracted 2026-08-20; each value below copied verbatim, not
#   retyped, and each is exactly 34 characters (verified programmatically).
# ---------------------------------------------------------------------------

HLA_PSEUDOSEQ = {
    "HLA-A*01:01": "YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY",
    "HLA-A*02:01": "YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY",
    "HLA-A*03:01": "YFAMYQENVAQTDVDTLYIIYRDYTWAELAYTWY",
    "HLA-A*11:01": "YYAMYQENVAQTDVDTLYIIYRDYTWAAQAYRWY",
    "HLA-A*24:02": "YSAMYEEKVAHTDENIAYLMFHYYTWAVQAYTGY",
    "HLA-B*07:02": "YYSEYRNIYAQTDESNLYLSYDYYTWAERAYEWY",
    "HLA-B*08:01": "YDSEYRNIFTNTDESNLYLSYNYYTWAVDAYTWY",
    "HLA-B*27:05": "YHTEYREICAKTDEDTLYLNYHDYTWAVLAYEWY",
    "HLA-B*35:01": "YYATYRNIFTNTYESNLYIRYDSYTWAVLAYLWY",
    "HLA-B*44:02": "YYTKYREISTNTYENTAYIRYDDYTWAVDAYLSY",
}
PSEUDO_LEN = 34
for _allele, _seq in HLA_PSEUDOSEQ.items():
    if len(_seq) != PSEUDO_LEN:
        raise ValueError(
            f"HLA_PSEUDOSEQ[{_allele!r}] is {len(_seq)} chars, expected {PSEUDO_LEN}. "
            "A silently trimmed or padded pseudo-sequence is a wrong feature vector "
            "with no error, not a smaller correct one - see docs/claims_register.md D30."
        )

ALLELE_PSEUDO_COLS = [
    f"hla_p{i + 1}_{prop}"
    for i in range(PSEUDO_LEN)
    for prop in ("hydrophobicity", "aromaticity", "vdw_volume", "charge")
]


def pseudoseq_to_features(seq):
    """Convert a 34-residue pseudo-sequence to 136 physicochemical features."""
    feats = []
    for aa in seq[:PSEUDO_LEN]:
        feats.append(KD_HYDRO.get(aa, 0.0))
        feats.append(float(AROMATIC.get(aa, 0)))
        feats.append(float(VDW_VOL.get(aa, 0.0)))
        feats.append(float(CHARGE.get(aa, 0)))
    return feats


# Pre-compute pseudo-sequence feature rows for each target allele
ALLELE_PSEUDO_FEATURES = {
    allele: pseudoseq_to_features(seq) for allele, seq in HLA_PSEUDOSEQ.items()
}


def standardize_allele_name(name):
    """Normalise allele strings to 'HLA-A*02:01' canonical form."""
    if pd.isna(name):
        return None
    s = str(name).strip()
    if not s.startswith("HLA-"):
        s = "HLA-" + s
    # Insert * if missing (e.g. HLA-A0201 -> HLA-A*0201)
    if "*" not in s:
        for prefix in ("HLA-A", "HLA-B", "HLA-C"):
            if s.startswith(prefix):
                s = prefix + "*" + s[len(prefix) :]
                break
    # Insert colon if missing (HLA-A*0201 -> HLA-A*02:01)
    if ":" not in s and "*" in s:
        star_pos = s.index("*")
        suffix = s[star_pos + 1 :]
        if len(suffix) == 4:
            s = s[: star_pos + 1] + suffix[:2] + ":" + suffix[2:]
    return s


def _parse_iedb_multiheader_csv(fpath, verbose=False):
    """Parse an IEDB T-cell Assay CSV that uses a 2-row multi-level header.

    IEDB exports with header=[0,1] produce tuple column keys like:
        ('Epitope', 'Name')               -> peptide sequence
        ('Assay', 'Qualitative Measurement') -> pos/neg label
        ('MHC Restriction', 'Class')      -> I or II filter
        ('Host', 'MHC Present')           -> precise allele (e.g. HLA-A*02:01)
        ('MHC Restriction', 'Name')       -> coarse name fallback
        ('Epitope', 'Source Organism')    -> EBV / HPV16 organism
    Returns a flat DataFrame with: peptide, label, allele, organism_str
    """
    from src.iedb_data_loader import STANDARD_AA

    df = pd.read_csv(fpath, header=[0, 1], low_memory=False)

    # Build lookup: (top_level_lower, sub_level_lower) -> column key
    col_lookup = {}
    for col in df.columns:
        top = str(col[0]).lower().strip()
        sub = str(col[1]).lower().strip()
        col_lookup[(top, sub)] = col

    # Required columns
    pep_col = col_lookup.get(("epitope", "name"))
    label_col = col_lookup.get(("assay", "qualitative measurement")) or col_lookup.get(
        ("assay", "qualitative measure")
    )
    mhc_class_col = col_lookup.get(("mhc restriction", "class"))
    # Prefer precise allele from host column, fall back to MHC Restriction Name
    allele_col_precise = col_lookup.get(("host", "mhc present"))
    allele_col_coarse = col_lookup.get(("mhc restriction", "name"))
    org_col = col_lookup.get(("epitope", "source organism"))

    if pep_col is None or label_col is None:
        if verbose:
            print(
                f"    [multi-header] missing required columns; pep_col={pep_col}, label_col={label_col}"
            )
        return pd.DataFrame()

    records = []
    for _, row in df.iterrows():
        pep_raw = row.get(pep_col)
        label_raw = row.get(label_col)
        if pd.isna(pep_raw) or pd.isna(label_raw):
            continue

        peptide = str(pep_raw).strip().upper()
        if not (8 <= len(peptide) <= 11):
            continue
        if not all(aa in STANDARD_AA for aa in peptide):
            continue

        lv = str(label_raw).strip().lower()
        if lv.startswith("positive"):
            label = 1
        elif lv == "negative":
            label = 0
        else:
            continue

        # MHC class filter - skip class II
        if mhc_class_col is not None:
            mhc_class = str(row.get(mhc_class_col, "")).strip()
            if mhc_class == "II":
                continue

        # Allele: try precise host column first, then coarse MHC Restriction Name
        allele_raw = None
        if allele_col_precise is not None:
            v = row.get(allele_col_precise)
            if not pd.isna(v) and str(v).strip():
                allele_raw = str(v).strip()
        if allele_raw is None and allele_col_coarse is not None:
            v = row.get(allele_col_coarse)
            if not pd.isna(v) and str(v).strip():
                allele_raw = str(v).strip()

        allele = standardize_allele_name(allele_raw) if allele_raw else None
        if allele and any(allele.startswith(p) for p in ("HLA-DR", "HLA-DP", "HLA-DQ")):
            continue

        org_str = ""
        if org_col is not None:
            org_str = str(row.get(org_col, "")).strip().lower()

        records.append(
            {
                "peptide": peptide,
                "label": label,
                "allele": allele,
                "organism_str": org_str,
            }
        )

    return pd.DataFrame(records)


def _virus_from_organism(org_str):
    """Map IEDB organism string to EBV or HPV16 (or None)."""
    s = org_str.lower()
    if "herpesvirus 4" in s or "gammaherpesvirus 4" in s or "epstein" in s or "ebv" in s:
        return "EBV"
    if "papillomavirus 16" in s or "hpv16" in s or "hpv-16" in s:
        return "HPV16"
    return None


def _is_processed_tcell_csv(df: pd.DataFrame) -> bool:
    """Return True if df looks like a fetch_iedb_tcell.py output (flat binary-label format).

    The processed format has a binary 'label' column (0/1) and 'hla_allele' column,
    produced by fetch_iedb_tcell.py. Raw IEDB exports use 'Qualitative Measure'.
    """
    cols = {c.lower() for c in df.columns}
    return "label" in cols and "hla_allele" in cols and "virus" in cols


def _load_processed_tcell_csv(fpath: str, verbose: bool = False) -> list[dict]:
    """Load a fetch_iedb_tcell.py output CSV into the canonical record format.

    The processed format has columns: peptide, label (0/1), virus, hla_allele,
    protein, strain, assay_type, assay_quality_weight, reference_pmid.
    Each row is already one (peptide, assay, allele) observation; majority voting
    across duplicate (peptide, allele) pairs happens downstream in
    build_allele_aware_dataset().
    """
    from src.iedb_data_loader import STANDARD_AA

    try:
        df = pd.read_csv(fpath, low_memory=False)
    except Exception as e:
        if verbose:
            print(f"    [skip] {fpath}: read error ({e})")
        return []

    if not _is_processed_tcell_csv(df):
        return []

    records = []
    for _, row in df.iterrows():
        pep = str(row.get("peptide", "")).strip().upper()
        if not (8 <= len(pep) <= 11):
            continue
        if not all(aa in STANDARD_AA for aa in pep):
            continue

        try:
            lbl = int(row["label"])
        except (ValueError, TypeError):
            continue
        if lbl not in (0, 1):
            continue

        allele_raw = row.get("hla_allele", None)
        if pd.isna(allele_raw) or not str(allele_raw).strip():
            continue
        allele = standardize_allele_name(str(allele_raw).strip())
        if allele and any(allele.startswith(p) for p in ("HLA-DR", "HLA-DP", "HLA-DQ")):
            continue

        virus = str(row.get("virus", "")).strip()
        if not virus:
            continue

        records.append(
            {
                "peptide": pep,
                "label": lbl,
                "allele": allele,
                "virus": virus,
            }
        )

    return records


def load_tcell_assay_files(data_dir, verbose=False):
    """Load IEDB T-cell Assay format files from data_dir.

    Handles three formats:
    1. 2-level multi-header CSV (standard IEDB T-cell Assay export) - parsed
       with header=[0,1] using _parse_iedb_multiheader_csv().
    2. Single-level header CSV/XLSX (older format) - detected by presence of
       a 'Qualitative Measure' column in the first header row.
    3. Processed fetch_iedb_tcell.py output CSV - detected by presence of
       'label' (binary 0/1), 'hla_allele', and 'virus' columns.

    Virus assignment priority (formats 1 and 2):
    1. Filename (if it contains 'ebv', 'herpesvirus', 'hpv16', 'papillomavirus').
    2. Organism column in the data (per-row assignment, for combined exports).
    Format 3 uses the 'virus' column directly (already in SESTRAV display names).

    Returns a DataFrame with columns: peptide, label, allele, virus
    """
    records = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith((".xlsx", ".csv")):
            continue
        fpath = os.path.join(data_dir, fname)

        # --- Determine virus from filename (may be None for generic exports) ---
        fl = fname.lower()
        if "ebv" in fl or "herpesvirus" in fl:
            filename_virus = "EBV"
        elif "hpv16" in fl or "hvp16" in fl or "papillomavirus" in fl:
            filename_virus = "HPV16"
        else:
            filename_virus = None  # will resolve per-row from organism column

        # --- Detect whether this is a 2-level IEDB multi-header file ---
        try:
            peek_raw = (
                pd.read_csv(fpath, header=None, nrows=2)
                if fname.endswith(".csv")
                else pd.read_excel(fpath, header=None, nrows=2)
            )
            row0 = [str(v).lower().strip() for v in peek_raw.iloc[0]]
            row1 = [str(v).lower().strip() for v in peek_raw.iloc[1]] if len(peek_raw) > 1 else []
        except Exception as e:
            if verbose:
                print(f"  [skip] {fname}: read error ({e})")
            continue

        # --- Format 3 (highest priority): processed fetch_iedb_tcell.py output ---
        # Check the header row (row0) for the flat binary-label signature BEFORE
        # running the multiheader detection, because the first DATA row of these
        # files may contain "qualitative binding" in assay_type, which would
        # otherwise falsely trigger the multiheader detector.
        is_processed_flat = "label" in row0 and "hla_allele" in row0 and "virus" in row0

        if not is_processed_flat:
            is_flat_header_csv = any(" - " in c for c in row0)
            if is_flat_header_csv:
                is_multiheader = False
                is_flat_qualitative = True
            else:
                is_multiheader = (
                    "qualitative measurement" in row1
                    or "qualitative measure" in row1
                    or any("qualitative" in v for v in row1)
                )
                is_flat_qualitative = any("qualitative" in v for v in row0)
        else:
            is_multiheader = False
            is_flat_qualitative = False

        if not (is_multiheader or is_flat_qualitative or is_processed_flat):
            if verbose:
                print(
                    f"  [skip] {fname}: unrecognized format (no Qualitative Measure or processed-flat columns)"
                )
            continue

        if is_processed_flat:
            if verbose:
                print(
                    f"  [load] {fname}: format=processed-flat (fetch_iedb_tcell.py output), virus=from-column"
                )
            new_records = _load_processed_tcell_csv(fpath, verbose=verbose)
            if verbose:
                print(f"    -> {len(new_records)} valid records added")
            records.extend(new_records)
            continue

        if verbose:
            fmt = "multi-header" if is_multiheader else "flat-header"
            virus_note = filename_virus or "per-row organism"
            print(f"  [load] {fname}: format={fmt}, virus={virus_note}")

        # --- Parse ---
        if is_multiheader and fname.endswith(".csv"):
            parsed = _parse_iedb_multiheader_csv(fpath, verbose=verbose)
            if parsed.empty:
                if verbose:
                    print("    -> no valid records extracted")
                continue

            # Assign virus per-row using filename override or organism column
            added = 0
            for _, row in parsed.iterrows():
                virus = filename_virus
                if virus is None:
                    virus = _virus_from_organism(str(row.get("organism_str", "")))
                if virus is None:
                    continue  # can't assign virus, skip
                records.append(
                    {
                        "peptide": row["peptide"],
                        "label": row["label"],
                        "allele": row["allele"],
                        "virus": virus,
                    }
                )
                added += 1
            if verbose:
                print(f"    -> {added} valid records added")

        else:
            # Flat single-level header (original path)
            try:
                if fname.endswith(".csv"):
                    df = pd.read_csv(fpath, low_memory=False)
                else:
                    import openpyxl  # noqa

                    df = pd.read_excel(fpath)
            except Exception as e:
                if verbose:
                    print(f"    -> read error: {e}")
                continue

            col_map = {}
            for col in df.columns:
                cl = str(col).lower().strip()
                if "description" in cl or (
                    "epitope" in cl and ("linear" in cl or "name" in cl or "sequence" in cl)
                ):
                    col_map.setdefault("peptide", col)
                elif "qualitative" in cl:
                    col_map.setdefault("label", col)
                elif "allele" in cl or "mhc present" in cl:
                    col_map.setdefault("allele", col)
                elif "organism" in cl or "species" in cl:
                    col_map.setdefault("organism", col)

            if "peptide" not in col_map or "label" not in col_map:
                if verbose:
                    print(
                        f"    -> missing peptide or label column in flat header (found: {list(df.columns[:5])})"
                    )
                continue

            from src.iedb_data_loader import STANDARD_AA

            added = 0
            for _, row in df.iterrows():
                pep_raw = row.get(col_map["peptide"])
                label_raw = row.get(col_map["label"])
                if pd.isna(pep_raw) or pd.isna(label_raw):
                    continue
                peptide = str(pep_raw).strip().upper()
                if not (8 <= len(peptide) <= 11):
                    continue
                if not all(aa in STANDARD_AA for aa in peptide):
                    continue
                lv = str(label_raw).strip().lower()
                if lv.startswith("positive"):
                    label = 1
                elif lv == "negative":
                    label = 0
                else:
                    continue
                allele_raw = row.get(col_map.get("allele", ""), None)
                allele = standardize_allele_name(allele_raw) if allele_raw is not None else None
                if allele and any(allele.startswith(p) for p in ("HLA-DR", "HLA-DP", "HLA-DQ")):
                    continue
                virus = filename_virus
                if virus is None:
                    org_str = str(row.get(col_map.get("organism", ""), ""))
                    virus = _virus_from_organism(org_str)
                if virus is None:
                    continue
                records.append(
                    {"peptide": peptide, "label": label, "allele": allele, "virus": virus}
                )
                added += 1
            if verbose:
                print(f"    -> {added} valid records added")

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def build_allele_aware_dataset(raw_df, target_alleles=None, verbose=False):
    """Filter, deduplicate by (peptide, allele), and compute majority-vote labels.

    Args:
        raw_df: DataFrame from load_tcell_assay_files().
        target_alleles: set of canonical allele strings to keep (None = keep all).
        verbose: print progress.

    Returns:
        DataFrame with columns: peptide, allele, label, virus, n_assays,
        plus 136 HLA pseudo-sequence feature columns.
    """
    df = raw_df.copy()

    if target_alleles:
        before = len(df)
        df = df[df["allele"].isin(target_alleles)]
        if verbose:
            print(f"  Allele filter: {before} -> {len(df)} rows (kept target 10-allele panel)")

    # Deduplicate per (peptide, allele) with majority vote
    def majority_label(series):
        return int(series.mean() >= 0.5)

    agg = (
        df.groupby(["peptide", "allele"])
        .agg(
            label=("label", majority_label),
            n_assays=("label", "count"),
            virus=("virus", "first"),
        )
        .reset_index()
    )

    # Append pseudo-sequence features
    pseudo_rows = []
    for allele in agg["allele"]:
        feats = ALLELE_PSEUDO_FEATURES.get(allele)
        if feats is None:
            feats = [0.0] * (PSEUDO_LEN * 4)
        pseudo_rows.append(feats)
    pseudo_df = pd.DataFrame(pseudo_rows, columns=ALLELE_PSEUDO_COLS)
    agg = pd.concat([agg.reset_index(drop=True), pseudo_df], axis=1)

    # Add peptide_length
    agg["peptide_length"] = agg["peptide"].str.len()

    if verbose:
        print("\nAllele-aware dataset summary:")
        print(f"  Total (peptide, allele) pairs: {len(agg)}")
        print(f"  Unique peptides: {agg['peptide'].nunique()}")
        print(f"  Unique alleles: {agg['allele'].nunique()}")
        print(f"  Class balance: {agg['label'].mean():.2%} positive")
        print("\n  Per-virus breakdown:")
        for virus, grp in agg.groupby("virus"):
            print(f"    {virus}: {len(grp)} pairs, {grp['label'].mean():.1%} positive")
        print("\n  Per-allele breakdown:")
        for allele, grp in agg.groupby("allele"):
            print(f"    {allele}: {len(grp)} pairs, {grp['label'].mean():.1%} positive")

    return agg


def sha256_df(df):
    """Compute SHA256 hash of DataFrame content (stable across runs)."""
    buf = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(buf).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build SESTRAV allele-aware training dataset from IEDB T-cell Assay exports "
            "or processed fetch_iedb_tcell.py output CSVs (auto-detected). "
            "Supports raw IEDB multi-header exports, flat-header exports, and "
            "the binary-label format produced by fetch_iedb_tcell.py."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="data/iedb",
        help="Directory containing IEDB xlsx/csv files (default: data/iedb)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path (default: data/allele_aware/IEDB-<date>-MULTI_VIRUS_ALLELE_AWARE-v2.csv)"
        ),
    )
    parser.add_argument(
        "--all-alleles",
        action="store_true",
        help="Keep all MHC-I alleles (not just the 10-allele panel)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-file loading details",
    )
    args = parser.parse_args()

    today_str = date.today().strftime("%Y%m%d")
    if args.output is None:
        out_dir = "data/allele_aware"
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, f"IEDB-{today_str}-MULTI_VIRUS_ALLELE_AWARE-v2.csv")

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: data directory not found: {args.data_dir}")
        print("  Place IEDB T-cell Assay export files (.xlsx or .csv) there and retry.")
        print("  Files should contain a 'Qualitative Measure' column.")
        print("\nFalling back to checking existing IEDB data in data/ ...")
        # Try the standard data directory used by the pipeline
        for candidate in ["data", "data/raw", "data/iedb_raw"]:
            if os.path.isdir(candidate):
                files = [f for f in os.listdir(candidate) if f.endswith((".xlsx", ".csv"))]
                if files:
                    print(f"  Found {len(files)} file(s) in {candidate}/ - trying that path.")
                    args.data_dir = candidate
                    break
        else:
            print("\nNo IEDB data files found. Please download T-cell Assay exports from:")
            print("  https://www.iedb.org -> search EBV/HPV16 -> Export -> T-cell Assay format")
            print("  Place files in data/iedb/ and re-run this script.")
            sys.exit(0)

    print("\nSESTRAV Allele-Aware Data Migration")
    print(f"{'=' * 60}")
    print(f"Data directory : {args.data_dir}")
    print(f"Output file    : {args.output}")
    print(f"Allele filter  : {'All MHC-I' if args.all_alleles else '10-allele panel'}")

    # Load T-cell Assay format files
    print("\nLoading T-cell Assay exports...")
    raw_df = load_tcell_assay_files(args.data_dir, verbose=args.verbose)

    if raw_df.empty:
        print("\nNo T-cell Assay format records loaded.")
        print("The data directory may contain only Epitope Table format files.")
        print("\nAction required:")
        print("  Download T-cell Assay exports from IEDB for EBV and HPV16.")
        print("  Query parameters:")
        print("    - Organism: Human herpesvirus 4 (EBV) / Human papillomavirus type 16")
        print("    - Assay: T-cell assay")
        print("    - MHC restriction: MHC class I only")
        print("    - Export format: T-cell Assay (NOT Epitope Table)")
        print("\nScript scaffold is ready. Re-run after downloading T-cell Assay files.")
        # Write a provenance stub so the pipeline knows this step was attempted
        stub = {
            "status": "awaiting_input_files",
            "message": "T-cell Assay exports not yet downloaded",
            "data_dir": args.data_dir,
            "output_path": args.output,
            "alleles": sorted(TARGET_ALLELES),
            "iedb_query": {
                "EBV": "Organism=Human herpesvirus 4, Assay=T-cell, MHC=Class I",
                "HPV16": "Organism=Human papillomavirus type 16, Assay=T-cell, MHC=Class I",
            },
        }
        stub_path = args.output.replace(".csv", "_stub.json")
        with open(stub_path, "w") as f:
            json.dump(stub, f, indent=2)
        print(f"\nProvenance stub written to: {stub_path}")
        sys.exit(0)

    # Filter to target allele panel
    target = None if args.all_alleles else TARGET_ALLELES
    agg = build_allele_aware_dataset(raw_df, target_alleles=target, verbose=True)

    # Exclude gold-standard holdouts from training data (same policy as main pipeline)
    gs_mask = agg["peptide"].isin(GOLD_STANDARD_EPITOPES)
    train_df = agg[~gs_mask].copy()
    holdout_df = agg[gs_mask].copy()
    print(f"\n  Gold-standard holdouts excluded from training: {gs_mask.sum()} rows")
    print(f"  Training pairs: {len(train_df)}")

    # Save
    train_df.to_csv(args.output, index=False)
    holdout_path = args.output.replace(".csv", "_holdouts.csv")
    holdout_df.to_csv(holdout_path, index=False)

    # Provenance
    prov = {
        "status": "complete",
        "output_path": args.output,
        "holdout_path": holdout_path,
        "training_pairs": len(train_df),
        "holdout_pairs": len(holdout_df),
        "unique_peptides": int(train_df["peptide"].nunique()),
        "unique_alleles": int(train_df["allele"].nunique()),
        "class_balance_positive": float(train_df["label"].mean()),
        "pseudo_seq_dims": PSEUDO_LEN * 4,
        "sha256": sha256_df(train_df),
        "allele_panel": sorted(TARGET_ALLELES),
    }
    prov_path = args.output.replace(".csv", "_provenance.json")
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)

    print("\nOutputs:")
    print(f"  Training dataset : {args.output}")
    print(f"  Holdout dataset  : {holdout_path}")
    print(f"  Provenance       : {prov_path}")
    print(f"  SHA256           : {prov['sha256']}")


if __name__ == "__main__":
    main()
