"""Compute self-similarity scores between training peptides and the human proteome.

Self-similarity captures the degree to which a viral peptide resembles a self-peptide
in the human proteome. High self-similarity is a mechanistic basis for non-immunogenicity:
peptides that closely resemble self-peptides are likely targets of central tolerance
(thymic deletion) or peripheral anergy, so even high-affinity MHC binders will fail
to mount a T-cell response.

This feature provides information orthogonal to MHC binding scores. It is a primary
reason why hard decoys (high-binding, non-immunogenic self-peptides) are expected to
improve model discrimination - they occupy the same binding-affinity space as viral
positives but differ on self-similarity.

Algorithm
---------
1. Parse the human proteome FASTA (UP000005640, Swiss-Prot reviewed, ~20k proteins).
2. Build a k-mer hash set of all unique 9-mer windows from the proteome.
3. For each training peptide of length L (8-11):
   - Generate all 9-mer sub-windows of the peptide (1 window for 9-mers,
     0 for 8-mers -> use the full 8-mer as the query, padding logic below).
   - For each window, compute the fraction of positions identical to the
     best-matching human 9-mer (exact match = 1.0; no match = 0.0).
   - Record: self_similarity_max_identity (float), self_similarity_exact_match (bool).

For 8-mer peptides (no full 9-mer window):
   The 8-mer itself is checked against all 8-mer windows of the human proteome
   (a separate 8-mer hash set built in the same pass). Exact match = 1.0.

For 10-mer and 11-mer peptides:
   All 9-mer sub-windows are checked; the maximum identity across windows is recorded.

Exact match (1.0) in the hash set is O(1). Partial identity would require pairwise
alignment (too slow for 20k proteins × training set). Instead we use a compromise:
for peptides with no exact 9-mer match, we record identity as 0.0. This is
conservative but correctly identifies the most-tolerated (= worst-for-training) peptides.

Output
------
data/self_similarity_cache.csv:
  peptide | self_similarity_max_identity | self_similarity_exact_match

Usage
-----
    # Prerequisite: human proteome FASTA must exist
    python scripts/fetch_human_proteome.py          # downloads ~100 MB, run once

    python scripts/precompute_self_similarity.py \\
        --fasta data/proteomes/human_uniprot_UP000005640.fasta \\
        --peptides data/immunogenicity_dataset_v4.csv \\
        --output data/self_similarity_cache.csv

    # Dry-run: estimate coverage without writing
    python scripts/precompute_self_similarity.py --dry-run

    # Resume: skip peptides already in output cache
    python scripts/precompute_self_similarity.py --resume

References
----------
Mason D. A very high level of crossreactivity is an essential feature of the T-cell
  receptor. Immunol Today. 1998;19(9):395-404. doi:10.1016/S0167-5699(98)01299-7

Sette A, Vitiello A, Reherman B, et al. The relationship between class I binding
  affinity and immunogenicity of potential cytotoxic T lymphocyte epitopes.
  J Immunol. 1994;153(12):5586-5592.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dataset_utils import write_provenance

_VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

def _iter_sequences(fasta_path: str):
    """Yield (header, sequence) pairs from a FASTA file."""
    header = None
    chunks: list[str] = []
    with open(fasta_path, "r") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            elif line:
                chunks.append(line.upper())
    if header is not None:
        yield header, "".join(chunks)


# ---------------------------------------------------------------------------
# K-mer index builder
# ---------------------------------------------------------------------------

def build_kmer_sets(fasta_path: str, kmer_lengths: tuple[int, ...] = (8, 9)) -> dict[int, set[str]]:
    """Build hash sets of all k-mers found in the human proteome.

    Only inserts k-mers composed entirely of standard amino acids to avoid
    spurious matches caused by ambiguous residues (X, B, Z, U).

    Args:
        fasta_path: Path to the human proteome FASTA.
        kmer_lengths: Which k-mer sizes to index (8 for 8-mer peptides, 9 for 9–11mers).

    Returns:
        Dict mapping k -> frozenset of all k-mers of that length.
    """
    t0 = time.time()
    kmer_sets: dict[int, set[str]] = {k: set() for k in kmer_lengths}
    n_proteins = 0
    n_kmers_total = 0

    with open(fasta_path, "r") as fh:
        header = None
        chunks: list[str] = []
        for line in fh:
            line = line.rstrip()
            if line.startswith(">") or not line:
                if header and chunks:
                    seq = "".join(chunks)
                    for k in kmer_lengths:
                        for i in range(len(seq) - k + 1):
                            kmer = seq[i: i + k]
                            if all(c in _VALID_AA for c in kmer):
                                kmer_sets[k].add(kmer)
                                n_kmers_total += 1
                    n_proteins += 1
                header = line[1:] if line.startswith(">") else header
                chunks = []
            else:
                chunks.append(line.upper())
        # Process last protein
        if header and chunks:
            seq = "".join(chunks)
            for k in kmer_lengths:
                for i in range(len(seq) - k + 1):
                    kmer = seq[i: i + k]
                    if all(c in _VALID_AA for c in kmer):
                        kmer_sets[k].add(kmer)
                        n_kmers_total += 1
            n_proteins += 1

    elapsed = time.time() - t0
    for k in kmer_lengths:
        print(f"  {k}-mer index: {len(kmer_sets[k]):,} unique k-mers")
    print(f"  Built from {n_proteins:,} proteins in {elapsed:.1f}s")
    return kmer_sets


# ---------------------------------------------------------------------------
# Self-similarity computation (pure, testable)
# ---------------------------------------------------------------------------

def compute_self_similarity(peptide: str, kmer_sets: dict[int, set[str]]) -> dict[str, float | bool]:
    """Compute self-similarity metrics for a single peptide.

    Strategy:
    - 8-mers: check exact match in 8-mer human proteome set.
    - 9-mers: check exact match in 9-mer set.
    - 10-mers/11-mers: check all 9-mer sub-windows; any exact match -> similarity = 1.0.

    Returns:
        dict with keys:
          self_similarity_max_identity : float in [0, 1]
          self_similarity_exact_match  : bool
    """
    n = len(peptide)
    if n < 8 or n > 11:
        return {"self_similarity_max_identity": 0.0, "self_similarity_exact_match": False}

    if n == 8:
        exact = peptide in kmer_sets.get(8, set())
        return {
            "self_similarity_max_identity": 1.0 if exact else 0.0,
            "self_similarity_exact_match": exact,
        }

    # For 9-11mers: slide 9-mer windows
    nine_set = kmer_sets.get(9, set())
    for i in range(n - 9 + 1):
        window = peptide[i: i + 9]
        if window in nine_set:
            return {
                "self_similarity_max_identity": 1.0,
                "self_similarity_exact_match": True,
            }

    return {"self_similarity_max_identity": 0.0, "self_similarity_exact_match": False}


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_peptides(
    peptides: list[str],
    kmer_sets: dict[int, set[str]],
    existing: set[str] | None = None,
) -> pd.DataFrame:
    """Compute self-similarity for a list of peptides.

    Args:
        peptides: Unique peptide sequences (8-11mers, standard AA).
        kmer_sets: Pre-built k-mer sets from build_kmer_sets().
        existing: Set of peptides already in the cache (for --resume).

    Returns:
        DataFrame with columns: peptide, self_similarity_max_identity,
        self_similarity_exact_match.
    """
    to_process = [p for p in peptides if p not in (existing or set())]
    print(f"  Computing self-similarity for {len(to_process):,} peptides "
          f"({len(peptides) - len(to_process):,} already cached)...")

    rows = []
    for p in to_process:
        result = compute_self_similarity(p, kmer_sets)
        rows.append({"peptide": p, **result})

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Precompute human-proteome self-similarity scores for training peptides.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--fasta",
        default="data/proteomes/human_uniprot_UP000005640.fasta",
        help="Human proteome FASTA (default: data/proteomes/human_uniprot_UP000005640.fasta)",
    )
    p.add_argument(
        "--peptides",
        default="data/immunogenicity_dataset_v4.csv",
        help="Training CSV containing a 'peptide' column (default: v4 dataset)",
    )
    p.add_argument(
        "--output",
        default="data/self_similarity_cache.csv",
        help="Output CSV path (default: data/self_similarity_cache.csv)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print unique peptide count and estimated time; do not write output.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip peptides already present in --output.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Validate inputs
    if not os.path.exists(args.fasta):
        print(f"Error: human proteome FASTA not found at {args.fasta}", file=sys.stderr)
        print("Run: python scripts/fetch_human_proteome.py", file=sys.stderr)
        return 1

    if not os.path.exists(args.peptides):
        print(f"Error: peptide source not found at {args.peptides}", file=sys.stderr)
        return 1

    # Load peptides
    df_train = pd.read_csv(args.peptides)
    if "peptide" not in df_train.columns:
        print("Error: peptide source CSV must have a 'peptide' column.", file=sys.stderr)
        return 1

    peptides = sorted({
        p.strip().upper()
        for p in df_train["peptide"].dropna()
        if isinstance(p, str)
        and 8 <= len(p.strip()) <= 11
        and all(c in _VALID_AA for c in p.strip().upper())
    })
    print(f"Unique valid 8-11mer peptides: {len(peptides):,}")

    if args.dry_run:
        # Rough estimate: ~0.01 ms per peptide (hash lookup, no I/O)
        est_seconds = len(peptides) * 0.00001
        print(f"Estimated wall time (after index build): {est_seconds:.1f}s")
        print("Human proteome index build takes ~10-20s on first run.")
        print("Dry-run complete -- no files written.")
        return 0

    # Load existing cache for --resume
    existing: set[str] = set()
    if args.resume and os.path.exists(args.output):
        cached = pd.read_csv(args.output)
        existing = set(cached["peptide"].dropna())
        print(f"Resuming: {len(existing):,} peptides already in cache.")

    # Build k-mer index
    print(f"Building k-mer index from {args.fasta} ...")
    kmer_sets = build_kmer_sets(args.fasta, kmer_lengths=(8, 9))

    # Compute similarities
    df_new = process_peptides(peptides, kmer_sets, existing=existing)

    # Merge with existing if resuming
    if args.resume and existing:
        cached = pd.read_csv(args.output)
        df_out = pd.concat([cached, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out = df_out.sort_values("peptide").reset_index(drop=True)

    # Summary
    n_exact = int(df_out["self_similarity_exact_match"].sum())
    n_total = len(df_out)
    print("\nSelf-similarity summary:")
    print(f"  Total peptides in cache: {n_total:,}")
    print(f"  Exact human proteome matches: {n_exact:,} ({100 * n_exact / max(n_total, 1):.1f}%)")
    print(f"  Non-matches (score=0.0): {n_total - n_exact:,}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"Saved -> {args.output}")

    write_provenance(
        args.output,
        sources=[args.fasta, args.peptides],
        row_count=n_total,
        extra={
            "exact_match_count": n_exact,
            "exact_match_fraction": round(n_exact / max(n_total, 1), 4),
            "kmer_lengths_indexed": [8, 9],
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
