"""
SESTRAV Candidate Epitope Target Generator
==========================================

Generates immunogenic 9-mer candidate epitopes from viral FASTA proteome files
using the trained SESTRAV PyTorch ANN model.

Usage:
    python scripts/generate_targets.py --fasta data/proteomes/EBV_B95_8_panel8.fasta data/proteomes/HPV16_18_panel8.fasta
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from Bio import SeqIO

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def slice_fasta_sequences(fasta_path: str, pathogen_name: str) -> list:
    """
    Load a FASTA file and slice sequences into a sliding window of 9-mers.
    Returns a list of dicts with keys: peptide, virus, source_id.
    """
    if not os.path.exists(fasta_path):
        logging.error(f"FASTA file not found: {fasta_path}")
        return []

    logging.info(f"Loading sequence records from FASTA: {fasta_path}")
    records = list(SeqIO.parse(fasta_path, "fasta"))
    logging.info(f"Loaded {len(records)} sequence(s) from {fasta_path}")

    extracted = []
    for record in records:
        seq = str(record.seq).upper()
        # Slide window of length 9
        for i in range(len(seq) - 8):
            pep = seq[i:i+9]
            # Verify only standard amino acids
            if all(c in "ACDEFGHIKLMNPQRSTVWY" for c in pep):
                extracted.append({
                    "peptide": pep,
                    "virus": pathogen_name,
                    "source_id": record.id
                })

    logging.info(f"Extracted {len(extracted)} candidate 9-mers from {fasta_path}")
    return extracted


def main():
    parser = argparse.ArgumentParser(
        description="Predict immunogenic epitopes from FASTA sequences using SESTRAV ANN model."
    )
    parser.add_argument(
        "--fasta",
        nargs="+",
        default=[
            "data/proteomes/EBV_B95_8_panel8.fasta",
            "data/proteomes/HPV16_18_panel8.fasta"
        ],
        help="Path to one or more FASTA files."
    )
    parser.add_argument(
        "--model",
        default="models/ann_30feature_integrated.pt",
        help="Path to trained PyTorch model checkpoint."
    )
    parser.add_argument(
        "--binding-matrix",
        default="models/peptide_binding_matrix_v3.csv",
        help="Path to peptide binding matrix CSV."
    )
    parser.add_argument(
        "--output",
        default="top_20_candidates.csv",
        help="Path to output CSV file."
    )

    args = parser.parse_args()

    # 1. Slide window extraction over all input FASTA files
    all_extracted = []
    for fasta_file in args.fasta:
        # Infer pathogen name from the filename
        pathogen = "Unknown"
        filename_upper = os.path.basename(fasta_file).upper()
        if "EBV" in filename_upper:
            pathogen = "EBV"
        elif "HPV" in filename_upper:
            pathogen = "HPV16"

        extracted = slice_fasta_sequences(fasta_file, pathogen)
        all_extracted.extend(extracted)

    if not all_extracted:
        logging.error("No candidate peptides extracted. Aborting.")
        sys.exit(1)

    df_candidates = pd.DataFrame(all_extracted)
    logging.info(f"Total extracted 9-mer records: {len(df_candidates)}")

    # Deduplicate candidate peptides to optimize feature extraction/inference
    df_unique = df_candidates.drop_duplicates(subset=["peptide", "virus"]).copy()
    logging.info(f"Total unique (peptide, virus) candidates: {len(df_unique)}")

    # 2. Run feature extractor (30-feature schema)
    logging.info(f"Extracting features using binding matrix: {args.binding_matrix}")
    from src.train_classifier import prepare_features_30
    try:
        X_df = prepare_features_30(df_unique, args.binding_matrix)
    except Exception as e:
        logging.error(f"Failed to prepare features: {e}")
        sys.exit(1)

    # 3. Model inference using PyTorch ANN model
    logging.info(f"Loading PyTorch ANN model checkpoint: {args.model}")
    from src.baseline_comparison import _score_with_ann
    try:
        df_unique["score"] = _score_with_ann(X_df, args.model)
    except Exception as e:
        logging.error(f"Failed during model scoring: {e}")
        sys.exit(1)

    # 4. Sort and export top 20 candidate targets
    top_df = df_unique.sort_values(by="score", ascending=False).head(20).copy()
    top_df = top_df.rename(columns={"virus": "pathogen", "score": "confidence"})

    # Order columns and output final targets CSV
    output_cols = ["peptide", "pathogen", "confidence"]
    top_df[output_cols].to_csv(args.output, index=False)
    logging.info(f"Successfully generated top 20 targets. Saved to: {args.output}")

    # Display candidates in logs
    print("\n" + "="*50)
    print("  TOP 20 IMMUNOGENIC EPITOPE CANDIDATES")
    print("="*50)
    for idx, (_, row) in enumerate(top_df.iterrows(), 1):
        print(f"  {idx:2d}. {row['peptide']:10s} | {row['pathogen']:6s} | Score: {row['confidence']:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
