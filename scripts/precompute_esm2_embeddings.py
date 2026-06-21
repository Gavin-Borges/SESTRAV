"""Pre-compute ESM-2 per-residue embeddings for all unique peptides in the v4 dataset.

Saves a dict[str, Tensor] mapping peptide sequence to a per-residue embedding
tensor of shape (max_len, esm_dim) padded with zeros.

Usage — default (t6, 320-dim, v2.1):
    python scripts/precompute_esm2_embeddings.py

Usage — larger model (t12, 480-dim, v2.2):
    python scripts/precompute_esm2_embeddings.py \
        --model facebook/esm2_t12_35M_UR50D \
        --output data/esm2_embeddings_t12.pt
"""
import argparse
import sys
from pathlib import Path

import torch
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MAX_LEN = 11

# ESM-2 model registry: name → embedding dimension
ESM_MODEL_DIMS: dict[str, int] = {
    "facebook/esm2_t6_8M_UR50D":    320,
    "facebook/esm2_t12_35M_UR50D":  480,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
}


def precompute_esm2(
    data_path: str,
    output_path: str,
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    batch_size: int = 64,
) -> None:
    from transformers import EsmModel, EsmTokenizer

    if model_name not in ESM_MODEL_DIMS:
        raise ValueError(
            f"Unknown model '{model_name}'. Known models: {list(ESM_MODEL_DIMS)}"
        )
    esm_dim = ESM_MODEL_DIMS[model_name]

    print(f"Loading {model_name} (dim={esm_dim}) ...")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
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

        # last_hidden_state: (B, padded_len, esm_dim)
        # Token layout: [CLS] aa_1 ... aa_L [EOS] [PAD ...]
        # Residue embeddings are at positions 1..L (inclusive).
        for j, seq in enumerate(batch_seqs):
            L = len(seq)
            residue_emb = outputs.last_hidden_state[j, 1 : 1 + L, :]
            padded = torch.zeros(MAX_LEN, esm_dim)
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
    parser.add_argument("--model", default="facebook/esm2_t6_8M_UR50D",
                        choices=list(ESM_MODEL_DIMS),
                        help="ESM-2 model variant to use")
    parser.add_argument("--output", default=None,
                        help="Output .pt path (default: data/esm2_embeddings_<variant>.pt)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Peptides per ESM-2 forward pass")
    args = parser.parse_args()

    if args.output is None:
        variant = args.model.split("/")[-1].lower().replace("_ur50d", "")
        args.output = f"data/esm2_embeddings_{variant}.pt"

    precompute_esm2(args.data, args.output, model_name=args.model, batch_size=args.batch_size)
