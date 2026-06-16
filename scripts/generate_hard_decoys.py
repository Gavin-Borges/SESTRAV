import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
from mhcflurry import Class1AffinityPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dataset_utils import normalize_peptides, validate_against_schema, write_provenance


def load_fasta(fasta_path):
    """Simple FASTA parser"""
    sequences = []
    current_seq = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line.strip())
        if current_seq:
            sequences.append("".join(current_seq))
    return sequences


def extract_kmers(sequences, k=9):
    """Extract all valid k-mers from a list of sequences"""
    kmers = set()
    valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if all(c in valid_chars for c in kmer):
                kmers.add(kmer)
    # Sort before any RNG use so the seeded shuffle is reproducible across runs.
    return sorted(kmers)


def generate_decoys(fasta_path, allele, num_decoys, output_path, schema_path, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    print(f"Loading reference proteome from {fasta_path}...")
    sequences = load_fasta(fasta_path)
    print(f"Loaded {len(sequences)} sequences.")

    print("Extracting 9-mers...")
    kmers = extract_kmers(sequences, k=9)
    print(f"Extracted {len(kmers)} unique valid 9-mers.")

    print(f"Loading MHCflurry predictor for allele {allele}...")
    try:
        predictor = Class1AffinityPredictor.load()
    except Exception as e:
        print(f"Error loading MHCflurry models: {e}")
        print("Please run 'mhcflurry-downloads fetch models_class1_presentation' first.")
        sys.exit(1)

    # Seeded shuffle -> reproducible decoy selection.
    random.shuffle(kmers)

    batch_size = 100000
    strong_binders_list = []
    total_found = 0

    print("Predicting binding affinities in batches...")
    for i in range(0, len(kmers), batch_size):
        batch = kmers[i:i + batch_size]
        predictions = predictor.predict_to_dataframe(peptides=batch, allele=allele)
        batch_strong = predictions[predictions['affinity'] < 50.0].copy()
        strong_binders_list.append(batch_strong)
        total_found += len(batch_strong)
        print(f"Batch {i // batch_size + 1}: Found {len(batch_strong)} strong binders. "
              f"Total: {total_found}/{num_decoys}")
        if total_found >= num_decoys:
            break

    if not strong_binders_list:
        print("No strong binders found!")
        sys.exit(1)

    strong_binders = pd.concat(strong_binders_list, ignore_index=True)
    if len(strong_binders) > num_decoys:
        strong_binders = strong_binders.sample(n=num_decoys, random_state=seed)

    # Format according to v4 schema
    df_out = pd.DataFrame({
        'peptide': strong_binders['peptide'],
        'label': 0,  # Decoys are negative
        'virus': 'Self',  # Legacy field
        'protein': 'Unknown',
        'strain': 'Unknown',
        'hla_allele': allele,
        'source_type': 'Self',
        'database_source': 'UniProt_HardDecoys'
    })
    df_out = normalize_peptides(df_out)
    df_out = df_out.drop_duplicates(subset=['peptide', 'hla_allele'])
    df_out = df_out.sort_values(['peptide', 'hla_allele']).reset_index(drop=True)

    validate_against_schema(df_out, schema_path)
    print(f"Saving {len(df_out)} decoys to {output_path}")
    df_out.to_csv(output_path, index=False)
    write_provenance(
        output_path, sources=[fasta_path], row_count=len(df_out),
        extra={"allele": allele, "num_decoys_requested": num_decoys,
               "binding_threshold_nM": 50.0, "seed": seed}
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Hard Decoy peptides")
    parser.add_argument("--fasta", required=True, help="Path to self-proteome FASTA")
    parser.add_argument("--allele", default="HLA-A*02:01", help="Target HLA allele")
    parser.add_argument("--num_decoys", type=int, default=1000, help="Number of decoys to generate")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--schema", default="data/immunogenicity_dataset_v4_schema.json",
                        help="v4 schema path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible runs")
    args = parser.parse_args()

    generate_decoys(args.fasta, args.allele, args.num_decoys, args.output, args.schema, seed=args.seed)
