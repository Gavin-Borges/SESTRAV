import argparse
import os
import sys
import urllib.request
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dataset_utils import normalize_peptides, validate_against_schema, write_provenance

VDJDB_URL = "https://github.com/antigenomics/vdjdb-db/releases/latest/download/vdjdb.zip"


def fetch_vdjdb(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "vdjdb.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading VDJdb from {VDJDB_URL}...")
        try:
            urllib.request.urlretrieve(VDJDB_URL, zip_path)  # nosec B310
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print("Downloaded and extracted VDJdb.")
        except Exception as e:
            print(f"Failed to fetch VDJdb: {e}")
            print("Please manually download VDJdb and provide the TSV path.")
            return None
    return os.path.join(output_dir, "vdjdb.txt")


def ingest_vdjdb(input_path, output_path, schema_path):
    fetched = False
    if not input_path or not os.path.exists(input_path):
        input_path = fetch_vdjdb()
        fetched = True
        if not input_path or not os.path.exists(input_path):
            raise FileNotFoundError(f"Could not find or fetch VDJdb input: {input_path}")

    print(f"Ingesting VDJdb from {input_path}")
    df = pd.read_csv(input_path, sep='\t')
    if 'Epitope' not in df.columns or 'MHC A' not in df.columns:
        raise ValueError("VDJdb input missing expected columns 'Epitope' and 'MHC A'")

    # Keep unique epitope/allele pairs (many TCRs bind the same epitope).
    unique_epitopes = df.drop_duplicates(subset=['Epitope', 'MHC A']).copy()
    unique_epitopes = unique_epitopes[unique_epitopes['MHC A'].str.contains('HLA', na=False)].copy()

    df_v4 = pd.DataFrame({
        'peptide': unique_epitopes['Epitope'],
        'label': 1,  # VDJdb contains known positive binders
        'virus': unique_epitopes.get('Epitope species', pd.Series(dtype='object')).fillna('Unknown'),
        'protein': unique_epitopes.get('Epitope gene', pd.Series(dtype='object')).fillna('Unknown'),
        'strain': 'Unknown',
        'hla_allele': unique_epitopes['MHC A'],
        'source_type': 'Virus',
        'database_source': 'VDJdb'
    })
    df_v4 = df_v4.dropna(subset=['peptide', 'hla_allele'])

    # Keep only high-resolution Class I alleles (e.g. HLA-A*02:01).
    valid_allele_mask = df_v4['hla_allele'].str.match(r'^[A-Z]+-[A-Z]\*\d{2}:\d{2}$').fillna(False)
    print(f"Dropped {int((~valid_allele_mask).sum())} entries due to low-resolution HLA alleles.")
    df_v4 = df_v4[valid_allele_mask]

    df_v4 = normalize_peptides(df_v4)
    df_v4 = df_v4.drop_duplicates(subset=['peptide', 'hla_allele'])
    df_v4 = df_v4.sort_values(['peptide', 'hla_allele']).reset_index(drop=True)

    validate_against_schema(df_v4, schema_path)
    print(f"Extracted {len(df_v4)} unique positive epitopes from VDJdb.")
    df_v4.to_csv(output_path, index=False)
    write_provenance(
        output_path, sources=[input_path], row_count=len(df_v4),
        extra={"database": "VDJdb", "fetched_latest_release": fetched,
               "url": VDJDB_URL if fetched else None}
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest VDJdb TSV into SESTRAV v4 Schema")
    parser.add_argument("--input", required=False, help="Path to VDJdb TSV file (will fetch if omitted)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--schema", default="data/immunogenicity_dataset_v4_schema.json",
                        help="v4 schema path")
    args = parser.parse_args()

    ingest_vdjdb(args.input, args.output, args.schema)
