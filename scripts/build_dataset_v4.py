import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dataset_utils import normalize_peptides, validate_against_schema, write_provenance

# Optional descriptive columns; defaulted to "Unknown" when a source omits them.
STRING_COLS = ["virus", "protein", "strain", "hla_allele", "source_type", "database_source"]


def build_dataset_v4(v3_path, vdjdb_path, tsnadb_path, decoys_path, schema_path, output_path):
    print("Building Dataset v4...")
    dfs = []
    used_sources = []

    # Legacy v3 (IEDB) — adapt to the v4 column set.
    if os.path.exists(v3_path):
        print(f"Loading legacy v3 data from {v3_path}")
        df_v3 = pd.read_csv(v3_path)
        df_v3['source_type'] = 'Virus'
        df_v3['database_source'] = 'IEDB'
        if 'hla_allele' not in df_v3.columns:
            df_v3['hla_allele'] = 'Unknown'
        dfs.append(df_v3)
        used_sources.append(v3_path)
    else:
        print(f"Warning: Legacy v3 data not found at {v3_path}")

    for name, path in [("VDJdb", vdjdb_path), ("TSNAdb", tsnadb_path), ("Hard Decoys", decoys_path)]:
        if path and os.path.exists(path):
            print(f"Loading {name} data from {path}")
            dfs.append(pd.read_csv(path))
            used_sources.append(path)
        else:
            print(f"Warning: {name} data not found at {path}")

    if not dfs:
        raise ValueError("No data sources found to build dataset v4.")

    df_merged = pd.concat(dfs, ignore_index=True)

    # Guard required columns, then default/clean the optional descriptive ones.
    if 'peptide' not in df_merged.columns or 'label' not in df_merged.columns:
        raise ValueError("Merged data missing required 'peptide'/'label' columns.")
    for col in STRING_COLS:
        if col not in df_merged.columns:
            df_merged[col] = 'Unknown'
        df_merged[col] = df_merged[col].fillna('Unknown')

    # Coerce labels and normalize peptides (drops non-standard residues).
    df_merged = df_merged.dropna(subset=['label'])
    df_merged['label'] = df_merged['label'].astype(int)
    df_merged = normalize_peptides(df_merged)

    # De-duplicate and impose a deterministic order for byte-stable re-runs.
    initial_len = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset=['peptide', 'hla_allele'])
    print(f"Dropped {initial_len - len(df_merged)} duplicates.")
    df_merged = df_merged.sort_values(['peptide', 'hla_allele']).reset_index(drop=True)

    validate_against_schema(df_merged, schema_path)
    df_merged.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df_merged)} records to {output_path}")
    write_provenance(output_path, sources=used_sources, row_count=len(df_merged),
                     extra={"schema": os.path.basename(schema_path)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge datasets and validate against v4 schema")
    parser.add_argument("--v3", default="data/immunogenicity_dataset_v3.csv", help="Legacy v3 dataset path")
    parser.add_argument("--vdjdb", default="data/vdjdb_v4.csv", help="Processed VDJdb path")
    parser.add_argument("--tsnadb", default="data/tsnadb_v4.csv", help="Processed TSNAdb path")
    parser.add_argument("--decoys", default="data/hard_decoys.csv", help="Hard decoys path")
    parser.add_argument("--schema", default="data/immunogenicity_dataset_v4_schema.json", help="Schema path")
    parser.add_argument("--output", default="data/immunogenicity_dataset_v4.csv", help="Output path")
    args = parser.parse_args()

    build_dataset_v4(args.v3, args.vdjdb, args.tsnadb, args.decoys, args.schema, args.output)
