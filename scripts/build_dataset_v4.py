"""Build the SESTRAV v4 immunogenicity dataset.

Sources merged (in priority order):
  1. --iedb-dir   Per-virus IEDB T-cell CSVs (data/iedb/*_tcell.csv); preferred
                  over --v3 when present. Rich 11-column schema with assay metadata.
  2. --v3         Legacy v3 IEDB CSV (5-column; used only if --iedb-dir absent/empty).
  3. --vdjdb      VDJdb viral epitopes (data/vdjdb_v4.csv).
  4. --tsnadb     TSNAdb neoantigen data (optional; skipped if missing).
  5. --decoys     Hard-decoy MHC-I binders from self-proteome (data/hard_decoys.csv).

Deduplication key: (peptide, hla_allele). Deterministic sort for byte-stable re-runs.
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dataset_utils import normalize_peptides, validate_against_schema, write_provenance

# Optional descriptive columns; defaulted to "Unknown" when a source omits them.
STRING_COLS = ["virus", "protein", "strain", "hla_allele", "source_type", "database_source"]


def _load_iedb_dir(iedb_dir: str) -> tuple[list[pd.DataFrame], list[str]]:
    """Load all *_tcell.csv files from iedb_dir. Returns (frames, source_paths)."""
    pattern = os.path.join(iedb_dir, "*_tcell.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return [], []
    frames, sources = [], []
    for p in paths:
        df = pd.read_csv(p)
        frames.append(df)
        sources.append(p)
        pos = int((df["label"] == 1).sum())
        neg = int((df["label"] == 0).sum())
        print(f"  {os.path.basename(p)}: {len(df)} rows ({pos} pos / {neg} neg)")
    return frames, sources


def build_dataset_v4(
    v3_path, vdjdb_path, tsnadb_path, decoys_path, schema_path, output_path,
    iedb_dir=None,
):
    print("Building Dataset v4...")
    dfs = []
    used_sources = []

    # Primary IEDB source: per-virus CSVs from data/iedb/
    iedb_frames: list[pd.DataFrame] = []
    if iedb_dir and os.path.isdir(iedb_dir):
        print(f"Loading per-virus IEDB data from {iedb_dir}/")
        iedb_frames, iedb_sources = _load_iedb_dir(iedb_dir)
        if iedb_frames:
            dfs.extend(iedb_frames)
            used_sources.extend(iedb_sources)
            total = sum(len(f) for f in iedb_frames)
            print(f"  IEDB total: {total} rows across {len(iedb_frames)} virus files")

    # Fallback: legacy v3 only when no per-virus IEDB files loaded
    if not iedb_frames:
        if v3_path and os.path.exists(v3_path):
            print(f"Loading legacy v3 data from {v3_path}")
            df_v3 = pd.read_csv(v3_path)
            df_v3["source_type"] = "Virus"
            df_v3["database_source"] = "IEDB"
            if "hla_allele" not in df_v3.columns:
                df_v3["hla_allele"] = "Unknown"
            dfs.append(df_v3)
            used_sources.append(v3_path)
        else:
            print(f"Warning: No IEDB data found (iedb_dir={iedb_dir!r}, v3={v3_path!r})")

    for name, path in [("VDJdb", vdjdb_path), ("TSNAdb", tsnadb_path), ("Hard Decoys", decoys_path)]:
        if path and os.path.exists(path):
            print(f"Loading {name} data from {path}")
            df_src = pd.read_csv(path)
            dfs.append(df_src)
            used_sources.append(path)
            pos = int((df_src["label"] == 1).sum())
            neg = int((df_src["label"] == 0).sum())
            print(f"  {name}: {len(df_src)} rows ({pos} pos / {neg} neg)")
        else:
            print(f"Warning: {name} data not found at {path!r}")

    if not dfs:
        raise ValueError("No data sources found to build dataset v4.")

    df_merged = pd.concat(dfs, ignore_index=True)
    print(f"\nPre-dedup total: {len(df_merged)} rows")

    # Guard required columns, then default/clean the optional descriptive ones.
    if "peptide" not in df_merged.columns or "label" not in df_merged.columns:
        raise ValueError("Merged data missing required 'peptide'/'label' columns.")
    for col in STRING_COLS:
        if col not in df_merged.columns:
            df_merged[col] = "Unknown"
        df_merged[col] = df_merged[col].fillna("Unknown")

    # Coerce labels and normalize peptides (drops non-standard residues).
    df_merged = df_merged.dropna(subset=["label"])
    df_merged["label"] = df_merged["label"].astype(int)
    df_merged = normalize_peptides(df_merged)

    # De-duplicate and impose a deterministic order for byte-stable re-runs.
    initial_len = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset=["peptide", "hla_allele"])
    n_dropped = initial_len - len(df_merged)
    print(f"Dropped {n_dropped} duplicates ({n_dropped / initial_len * 100:.1f}%).")
    df_merged = df_merged.sort_values(["peptide", "hla_allele"]).reset_index(drop=True)

    pos = int((df_merged["label"] == 1).sum())
    neg = int((df_merged["label"] == 0).sum())
    print(f"Final: {len(df_merged)} rows | {pos} pos ({pos/len(df_merged)*100:.1f}%) | {neg} neg")

    validate_against_schema(df_merged, schema_path)
    df_merged.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df_merged)} records to {output_path}")
    write_provenance(
        output_path,
        sources=used_sources,
        row_count=len(df_merged),
        extra={
            "schema": os.path.basename(schema_path),
            "positive_count": pos,
            "negative_count": neg,
            "positive_fraction": round(pos / len(df_merged), 4),
            "n_sources": len(used_sources),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge datasets and validate against v4 schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--iedb-dir", default="data/iedb",
        help="Directory of per-virus IEDB CSVs (*_tcell.csv). Default: data/iedb/",
    )
    parser.add_argument("--v3", default="data/immunogenicity_dataset_v3.csv",
                        help="Legacy v3 fallback (used only if --iedb-dir is empty/absent)")
    parser.add_argument("--vdjdb", default="data/vdjdb_v4.csv", help="Processed VDJdb path")
    parser.add_argument("--tsnadb", default="data/tsnadb_v4.csv", help="Processed TSNAdb path")
    parser.add_argument("--decoys", default="data/hard_decoys.csv", help="Hard decoys path")
    parser.add_argument("--schema", default="data/immunogenicity_dataset_v4_schema.json",
                        help="Schema path")
    parser.add_argument("--output", default="data/immunogenicity_dataset_v4.csv",
                        help="Output path")
    args = parser.parse_args()

    build_dataset_v4(
        args.v3, args.vdjdb, args.tsnadb, args.decoys, args.schema, args.output,
        iedb_dir=args.iedb_dir,
    )
