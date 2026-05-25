"""
SESTRAV Stage 1 — Peptide Generation
Parses viral proteome FASTA files and generates all overlapping k-mer peptides
(8-11 amino acids) via a sliding window. Non-standard amino acids are rejected.
"""

from Bio import SeqIO
import pandas as pd

STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
DEFAULT_LENGTHS = [8, 9, 10, 11]


import re

def _sanitize_name(name):
    """Allow only alphanumeric, underscores, and hyphens."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)


def generate_peptides(fasta_path, proteome_id, peptide_lengths=None):
    proteome_id = _sanitize_name(proteome_id)
    """
    Generate all overlapping peptides from a multi-protein FASTA file.

    Args:
        fasta_path: path to a FASTA file containing one or more protein sequences
        proteome_id: label used in output filename (e.g., 'HPV16_18_panel8')
        peptide_lengths: list of k-mer sizes (default: [8, 9, 10, 11])

    Returns:
        DataFrame with columns: protein_id, peptide, length, start, end
    """
    if peptide_lengths is None:
        peptide_lengths = DEFAULT_LENGTHS

    peptides = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        protein_id = record.id
        seq = str(record.seq).upper()

        for length in peptide_lengths:
            for i in range(len(seq) - length + 1):
                kmer = seq[i:i + length]
                if all(aa in STANDARD_AA for aa in kmer):
                    peptides.append({
                        "protein_id": protein_id,
                        "peptide": kmer,
                        "length": length,
                        "start": i + 1,
                        "end": i + length
                    })

    df = pd.DataFrame(peptides)
    output_path = f"results/{proteome_id}_peptides.csv"
    df.to_csv(output_path, index=False)
    print(f"[Stage 1] Generated {len(df)} peptides for {proteome_id}")
    return df
