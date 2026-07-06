"""
download_tcr3d_structures.py - Download TCR-pMHC-I crystal structures from
TCR3d 2.0 and RCSB for SESTRAV M9 contact-frequency analysis.

Data source:
  TCR3d 2.0 metadata TSV:
    https://tcr3d.ibbr.umd.edu/static/download/tcr_complexes_data.tsv
  Individual PDB files (RCSB):
    https://files.rcsb.org/download/{PDB_ID}.pdb

This script:
  1. Downloads the TCR3d metadata TSV (stdlib urllib only).
  2. Filters rows to Human Class I complexes for the 10 canonical SESTRAV
     alleles, resolution <= cutoff, peptide length 8-11.
  3. Writes data/tcr3d_allele_map.csv mapping pdb_id -> 4-digit canonical allele.
  4. Downloads the filtered PDB files from RCSB into data/tcr3d_structures/.
  5. Prints a summary and the next-step command for compute_contact_freqs.py.

Usage example:
  python scripts/download_tcr3d_structures.py
  python scripts/download_tcr3d_structures.py --dry-run
  python scripts/download_tcr3d_structures.py --resolution-cutoff 3.0 --out-dir data/tcr3d_structures
"""

import argparse
import csv
import io
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TCR3D_TSV_URL = "https://tcr3d.ibbr.umd.edu/static/download/tcr_complexes_data.tsv"
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

# Map from TCR3d 2-field MHC_allele prefix to 4-digit canonical allele.
ALLELE_MAP: dict[str, str] = {
    "HLA-A*01": "HLA-A*01:01",
    "HLA-A*02": "HLA-A*02:01",
    "HLA-A*03": "HLA-A*03:01",
    "HLA-A*11": "HLA-A*11:01",
    "HLA-A*24": "HLA-A*24:02",
    "HLA-B*07": "HLA-B*07:02",
    "HLA-B*08": "HLA-B*08:01",
    "HLA-B*27": "HLA-B*27:05",
    "HLA-B*35": "HLA-B*35:01",
    "HLA-B*44": "HLA-B*44:02",
}

# TCR3d complex-type value for Human Class I.
CLASS_I_VALUE = "CLASSI"

# Download settings.
DOWNLOAD_TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 2

# ---------------------------------------------------------------------------
# TSV fetch and parse
# ---------------------------------------------------------------------------


def fetch_tsv(url: str) -> list[dict[str, str]]:
    """Fetch the TCR3d metadata TSV and return it as a list of row dicts."""
    if not url.startswith("https://"):
        raise ValueError(f"Only HTTPS URLs are permitted; got: {url!r}")
    print(f"Fetching metadata TSV: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:  # nosec B310
            raw_bytes = resp.read()
    except Exception as exc:
        print(f"ERROR: Failed to download TSV: {exc}", file=sys.stderr)
        sys.exit(1)

    # Decode with error replacement in case of stray non-ASCII bytes.
    text = raw_bytes.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    rows = [dict(row) for row in reader]
    print(f"  -> {len(rows)} total rows in TSV")
    return rows


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def _canonical_allele(mhc_allele: str) -> str | None:
    """Return 4-digit canonical allele string or None if not in target set."""
    allele_stripped = mhc_allele.strip()
    for prefix, canonical in ALLELE_MAP.items():
        if allele_stripped.startswith(prefix):
            return canonical
    return None


def _valid_resolution(resolution_str: str, cutoff: float) -> bool:
    """Return True if resolution field is non-empty and <= cutoff Angstroms."""
    val = resolution_str.strip()
    if not val:
        return False
    try:
        return float(val) <= cutoff
    except ValueError:
        return False


def _valid_peptide_length(epitope: str, min_len: int = 8, max_len: int = 11) -> bool:
    """Return True if peptide length is in [min_len, max_len]."""
    pep = epitope.strip()
    return min_len <= len(pep) <= max_len


def filter_rows(
    rows: list[dict[str, str]],
    resolution_cutoff: float,
) -> list[tuple[str, str]]:
    """Apply all four inclusion filters; return list of (pdb_id, canonical_allele).

    Filters applied (all must pass):
      1. MHC_allele starts with one of the 10 canonical 2-field prefixes.
      2. TCR_complex == CLASSI (Human Class I only).
      3. Epitope length is 8-11 inclusive.
      4. Resolution is present and <= resolution_cutoff.
    """
    kept: list[tuple[str, str]] = []
    seen_pdb: set[str] = set()

    counts = {
        "total": len(rows),
        "fail_allele": 0,
        "fail_class": 0,
        "fail_peptide": 0,
        "fail_resolution": 0,
        "duplicate_pdb": 0,
        "kept": 0,
    }

    for row in rows:
        pdb_raw = row.get("PDB_ID", "").strip()
        mhc = row.get("MHC_allele", "")
        tcr_complex = row.get("TCR_complex", "").strip()
        epitope = row.get("Epitope", "")
        resolution = row.get("Resolution", "")

        canonical = _canonical_allele(mhc)
        if canonical is None:
            counts["fail_allele"] += 1
            continue

        if tcr_complex != CLASS_I_VALUE:
            counts["fail_class"] += 1
            continue

        if not _valid_peptide_length(epitope):
            counts["fail_peptide"] += 1
            continue

        if not _valid_resolution(resolution, resolution_cutoff):
            counts["fail_resolution"] += 1
            continue

        pdb_id = pdb_raw.lower()
        if not re.fullmatch(r"[a-z0-9]{4}", pdb_id):
            counts["invalid_format"] = counts.get("invalid_format", 0) + 1
            continue
        if pdb_id in seen_pdb:
            counts["duplicate_pdb"] += 1
            continue

        seen_pdb.add(pdb_id)
        kept.append((pdb_id, canonical))
        counts["kept"] += 1

    print(f"\nFilter summary (resolution cutoff: {resolution_cutoff} A):")
    print(f"  Total rows in TSV:             {counts['total']}")
    print(f"  Removed - allele not in set:   {counts['fail_allele']}")
    print(f"  Removed - not Class I:         {counts['fail_class']}")
    print(f"  Removed - peptide length:      {counts['fail_peptide']}")
    print(f"  Removed - resolution:          {counts['fail_resolution']}")
    print(f"  Removed - duplicate PDB ID:    {counts['duplicate_pdb']}")
    print(f"  Kept:                          {counts['kept']}")
    return kept


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_allele_summary(entries: list[tuple[str, str]]) -> None:
    """Print per-allele structure counts before downloading."""
    from collections import Counter

    allele_counts: Counter[str] = Counter(canonical for _, canonical in entries)
    print("\nStructures per allele:")
    print(f"  {'Allele':<20}  {'N':>6}")
    print(f"  {'-' * 20}  {'------':>6}")
    for allele in sorted(allele_counts):
        print(f"  {allele:<20}  {allele_counts[allele]:>6}")
    print(f"  {'TOTAL':<20}  {len(entries):>6}")


# ---------------------------------------------------------------------------
# Allele map CSV
# ---------------------------------------------------------------------------


def write_allele_map(
    entries: list[tuple[str, str]],
    out_path: Path,
    dry_run: bool,
) -> None:
    """Write pdb_id,hla_allele CSV."""
    if dry_run:
        print(f"\n[dry-run] Would write allele map to: {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="ascii") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pdb_id", "hla_allele"])
        for pdb_id, canonical in sorted(entries):
            writer.writerow([pdb_id, canonical])
    print(f"\nAllele map written: {out_path} ({len(entries)} rows)")


# ---------------------------------------------------------------------------
# PDB download
# ---------------------------------------------------------------------------


def download_pdb(pdb_id: str, dest: Path) -> bool:
    """Download a single PDB file from RCSB with up to MAX_ATTEMPTS tries.

    Returns True on success, False on network/IO failure.
    Raises ValueError if the resolved URL scheme is not https:// (programming
    error - indicates RCSB_PDB_URL constant was changed to a non-HTTPS value).
    """
    url = RCSB_PDB_URL.format(pdb_id=pdb_id.upper())
    if not url.startswith("https://"):
        raise ValueError(f"Only HTTPS URLs are permitted; got: {url!r}")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as resp:  # nosec B310
                content_bytes = resp.read()
            content = content_bytes.decode("ascii", errors="ignore")
            dest.write_text(content, encoding="ascii")
            return True
        except urllib.error.URLError as exc:
            if attempt < MAX_ATTEMPTS:
                print(f"  WARNING: {pdb_id} attempt {attempt} failed ({exc}); retrying...")
                time.sleep(1)
            else:
                print(
                    f"  WARNING: {pdb_id} failed after {MAX_ATTEMPTS} attempts: {exc}",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(
                f"  WARNING: {pdb_id} unexpected error: {exc}",
                file=sys.stderr,
            )
            return False
    return False


def download_all_pdbs(
    entries: list[tuple[str, str]],
    out_dir: Path,
    skip_existing: bool,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Download PDB files; return (n_downloaded, n_skipped, n_failed)."""
    if dry_run:
        print(f"\n[dry-run] Would download {len(entries)} PDB files to: {out_dir}")
        for pdb_id, canonical in entries:
            dest = out_dir / f"{pdb_id}.pdb"
            status = "exists" if dest.exists() else "would download"
            print(f"  {pdb_id}  ({canonical})  [{status}]")
        return 0, 0, 0

    out_dir.mkdir(parents=True, exist_ok=True)
    n_downloaded = 0
    n_skipped = 0
    n_failed = 0
    total = len(entries)

    print(f"\nDownloading {total} PDB files to {out_dir} ...")

    for i, (pdb_id, _canonical) in enumerate(entries, start=1):
        dest = out_dir / f"{pdb_id}.pdb"
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            print(f"  Skipped  {pdb_id} ({i}/{total}) [already on disk]")
            n_skipped += 1
            continue

        success = download_pdb(pdb_id, dest)
        if success:
            print(f"  Downloaded {pdb_id} ({i}/{total})")
            n_downloaded += 1
        else:
            n_failed += 1

    return n_downloaded, n_skipped, n_failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download_tcr3d_structures.py",
        description=(
            "Download TCR-pMHC-I crystal structures from TCR3d 2.0 and RCSB "
            "for SESTRAV M9 contact-frequency analysis."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/tcr3d_structures"),
        metavar="DIR",
        help="Directory to write downloaded PDB files (default: data/tcr3d_structures).",
    )
    parser.add_argument(
        "--allele-map-out",
        type=Path,
        default=Path("data/tcr3d_allele_map.csv"),
        metavar="PATH",
        help="Output CSV for pdb_id -> canonical allele map (default: data/tcr3d_allele_map.csv).",
    )
    parser.add_argument(
        "--resolution-cutoff",
        type=float,
        default=3.5,
        metavar="ANGSTROMS",
        help="Maximum resolution in Angstroms to include (default: 3.5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without writing any files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip PDB files already present on disk (default: True).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-download PDB files even if already on disk.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Step 1: fetch and parse TSV.
    rows = fetch_tsv(TCR3D_TSV_URL)

    # Step 2: apply filters.
    entries = filter_rows(rows, args.resolution_cutoff)
    if not entries:
        print("WARNING: No structures passed all filters. Exiting.", file=sys.stderr)
        return 1

    # Step 3: print per-allele summary table.
    print_allele_summary(entries)

    # Step 4: write allele map CSV.
    write_allele_map(entries, args.allele_map_out, args.dry_run)

    # Step 5: download PDB files.
    n_downloaded, n_skipped, n_failed = download_all_pdbs(
        entries,
        args.out_dir,
        args.skip_existing,
        args.dry_run,
    )

    # Step 6: final summary.
    if not args.dry_run:
        print("\n--- Download complete ---")
        print(f"  Downloaded: {n_downloaded}")
        print(f"  Skipped:    {n_skipped}")
        print(f"  Failed:     {n_failed}")
        print()
        print("Next step:")
        print(
            "  python scripts/compute_contact_freqs.py"
            " --pdb-dir data/tcr3d_structures"
            " --allele-map data/tcr3d_allele_map.csv"
            " --output data/contact_freqs/per_allele_contact_freq.csv"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
