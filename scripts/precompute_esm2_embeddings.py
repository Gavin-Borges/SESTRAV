"""Pre-compute ESM-2 per-residue embeddings for all unique peptides in the v4 dataset.

Saves data/esm2_embeddings.pt: dict[str, Tensor] mapping peptide sequence to
a per-residue embedding tensor of shape (max_len, esm_dim) padded with zeros.

Usage:
    python scripts/precompute_esm2_embeddings.py \
        --data data/immunogenicity_dataset_v4.csv \
        --output data/esm2_embeddings.pt
"""
import argparse
import sys
from pathlib import Path

import torch
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
ESM_DIM = 320
MAX_LEN = 11


def precompute_esm2(data_path: str, output_path: str, batch_size: int = 64) -> None:
    from transformers import EsmModel, EsmTokenizer

    print(f"Loading {ESM_MODEL_NAME} ...")
    tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME)
    model = EsmModel.from_pretrained(ESM_MODEL_NAME)
    model.eval()

    df = pd.read_csv(data_path)
    peptides = sorted(df["peptide"].unique().tolist())
    print(f"Computing embeddings for {len(peptides)} unique peptides (batch_size={batch_size}) ...")

    embeddings: dict = {}
    total_batches = (len(peptides) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_seqs = peptides[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)

        # last_hidden_state: (B, padded_len, ESM_DIM)
        # Token layout: [CLS] aa_1 aa_2 ... aa_L [EOS] [PAD ...]
        # Residue embeddings are at positions 1 .. L (inclusive).
        for j, seq in enumerate(batch_seqs):
            L = len(seq)
            residue_emb = outputs.last_hidden_state[j, 1 : 1 + L, :]  # (L, ESM_DIM)
            padded = torch.zeros(MAX_LEN, ESM_DIM)
            take = min(L, MAX_LEN)
            padded[:take, :] = residue_emb[:take, :]
            embeddings[seq] = padded

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == total_batches:
            done = min((batch_idx + 1) * batch_size, len(peptides))
            print(f"  {done}/{len(peptides)} peptides processed")

    print(f"Saving {len(embeddings)} embeddings to {output_path}")
    torch.save(embeddings, output_path)  # nosec B614
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute ESM-2 peptide embeddings")
    parser.add_argument("--data", default="data/immunogenicity_dataset_v4.csv",
                        help="Path to immunogenicity dataset CSV")
    parser.add_argument("--output", default="data/esm2_embeddings.pt",
                        help="Output path for the embeddings .pt file")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Peptides per ESM-2 forward pass")
    args = parser.parse_args()
    precompute_esm2(args.data, args.output, args.batch_size)
