from Bio import SeqIO
import pandas as pd

PEPTIDE_LENGTHS = [8, 9, 10, 11]

def generate_peptides(fasta_path, virus_name):
    peptides = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        protein_id = record.id
        seq = str(record.seq)

        for length in PEPTIDE_LENGTHS:
            for i in range(len(seq) - length + 1):
                peptides.append({
                    "protein_id": protein_id,
                    "peptide": seq[i:i+length],
                    "length": length,
                    "start": i + 1,
                    "end": i + length
                })

    df = pd.DataFrame(peptides)
    df.to_csv(f"results/{virus_name}_peptides.csv", index=False)
    print(f"[Stage 1] Generated {len(df)} peptides for {virus_name}")
    return df