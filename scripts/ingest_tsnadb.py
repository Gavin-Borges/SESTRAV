import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dataset_utils import normalize_peptides, validate_against_schema, write_provenance


def ingest_tsnadb(input_path, output_path, schema_path):
    print(f"Ingesting TSNAdb from {input_path}")
    try:
        df = pd.read_csv(input_path, sep='\t')  # Often TSV
    except Exception:
        df = pd.read_csv(input_path)  # Fallback to CSV

    # TSNAdb column names vary; map the common neoantigen columns.
    peptide_col = ('Peptide' if 'Peptide' in df.columns
                   else 'Mutant_Peptide' if 'Mutant_Peptide' in df.columns else None)
    hla_col = 'HLA' if 'HLA' in df.columns else 'HLA_type' if 'HLA_type' in df.columns else None
    if not peptide_col or not hla_col:
        print("Columns available:", df.columns.tolist())
        raise ValueError(
            f"TSNAdb input missing expected peptide ({peptide_col}) or HLA ({hla_col}) columns"
        )

    protein_col = next((c for c in ('Gene', 'Mutation') if c in df.columns), None)
    tissue_col = next((c for c in ('Cancer_type', 'Tissue') if c in df.columns), None)

    df_v4 = pd.DataFrame({
        'peptide': df[peptide_col],
        'label': 1,  # Neoantigens are positive
        'virus': 'None',
        'protein': df[protein_col] if protein_col else 'Unknown',
        'strain': df[tissue_col] if tissue_col else 'Unknown',
        'hla_allele': df[hla_col],
        'source_type': 'Tumor',
        'database_source': 'TSNAdb'
    })
    df_v4 = df_v4.dropna(subset=['peptide', 'hla_allele'])
    # TSNAdb v2 omits the asterisk: HLA-A02:01 → HLA-A*02:01
    df_v4['hla_allele'] = df_v4['hla_allele'].str.replace(
        r'(HLA-[A-Z])(\d)', r'\1*\2', regex=True
    )
    df_v4 = normalize_peptides(df_v4)
    df_v4 = df_v4.drop_duplicates(subset=['peptide', 'hla_allele'])
    df_v4 = df_v4.sort_values(['peptide', 'hla_allele']).reset_index(drop=True)

    validate_against_schema(df_v4, schema_path)
    print(f"Extracted {len(df_v4)} unique neoantigens.")
    df_v4.to_csv(output_path, index=False)
    write_provenance(output_path, sources=[input_path], row_count=len(df_v4),
                     extra={"database": "TSNAdb"})
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest TSNAdb data into SESTRAV v4 Schema")
    parser.add_argument("--input", required=True, help="Path to TSNAdb data file")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--schema", default="data/immunogenicity_dataset_v4_schema.json",
                        help="v4 schema path")
    args = parser.parse_args()

    ingest_tsnadb(args.input, args.output, args.schema)
