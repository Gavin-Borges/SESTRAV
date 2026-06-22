"""Generate hard-decoy negative peptides for SESTRAV v4 dataset.

Hard decoys are high-affinity MHC Class I binders from the human self-proteome
that are non-immunogenic due to central tolerance. Including them in training
breaks the binding-only negative selection confound present in v3: v3 negatives
are non-immunogenic primarily because they bind MHC poorly, not because they fail
TCR recognition. Hard decoys force the model to learn TCR-contact discrimination
rather than relying on MHC binding as a proxy.

Algorithm
---------
1. Extract all valid 8–11-mer peptides from the human proteome FASTA.
2. Screen with MHCflurry Class1PresentationPredictor for all 10 canonical alleles.
3. Keep peptides with presentation_score ≥ PRESENTATION_THRESHOLD for ≥1 allele.
4. Exclude any peptide present as a positive (label=1) in the supplied training set.
5. Sample deterministically to reach --num-decoys total, balanced across alleles.
6. Validate against v4 schema; write CSV + provenance sidecar.

Usage
-----
    # Prerequisite: download human proteome FASTA first
    python scripts/fetch_human_proteome.py

    # Then generate decoys
    python scripts/generate_hard_decoys.py \\
        --fasta data/proteomes/human_uniprot_UP000005640.fasta \\
        --training-data data/immunogenicity_dataset_v3.csv \\
        --output data/hard_decoys.csv \\
        --num-decoys 10000

References
----------
Rock KL, Goldberg AL. Degradation of cell proteins and the generation of
  MHC class I-presented peptides. Annu Rev Immunol. 1999;17:739-79.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dataset_utils import normalize_peptides, validate_against_schema, write_provenance

# 10 canonical HLA supertypes used throughout SESTRAV
CANONICAL_ALLELES = [
    "HLA-A*02:01", "HLA-A*01:01", "HLA-A*03:01", "HLA-A*11:01", "HLA-A*24:02",
    "HLA-B*07:02", "HLA-B*08:01", "HLA-B*27:05", "HLA-B*35:01", "HLA-B*44:02",
]

# Minimum MHCflurry presentation_score to qualify as a strong binder
PRESENTATION_THRESHOLD = 0.5

# Valid standard amino acids
_VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _load_fasta(fasta_path: str | Path) -> list[str]:
    """Return all protein sequences from a FASTA file."""
    sequences: list[str] = []
    current: list[str] = []
    with open(fasta_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def _extract_kmers(sequences: list[str], lengths: tuple[int, ...] = (8, 9, 10, 11)) -> list[str]:
    """Extract all valid k-mers for all requested peptide lengths.

    Sort before any RNG use so the seeded shuffle is byte-stable across runs.
    """
    kmers: set[str] = set()
    for seq in sequences:
        seq_upper = seq.upper()
        for k in lengths:
            for i in range(len(seq_upper) - k + 1):
                frag = seq_upper[i: i + k]
                if all(c in _VALID_AA for c in frag):
                    kmers.add(frag)
    return sorted(kmers)


def _score_batched(
    kmers: list[str],
    allele: str,
    predictor,
    batch_size: int = 50_000,
) -> pd.DataFrame:
    """Return a DataFrame with (peptide, allele, presentation_score) for kmers."""
    frames: list[pd.DataFrame] = []
    for i in range(0, len(kmers), batch_size):
        batch = kmers[i: i + batch_size]
        result = predictor.predict(peptides=batch, alleles=[allele], verbose=0)
        # Column name differs between MHCflurry versions
        score_col = next(
            (c for c in ["presentation_score", "affinity", "score"] if c in result.columns),
            result.columns[-1],
        )
        strong = result[result[score_col] >= PRESENTATION_THRESHOLD].copy()
        strong = strong.rename(columns={score_col: "presentation_score"})
        strong["allele"] = allele
        frames.append(strong[["peptide", "allele", "presentation_score"]])
        found = sum(len(f) for f in frames)
        print(
            f"  {allele}: batch {i // batch_size + 1}/{(len(kmers) - 1) // batch_size + 1} "
            f"- {len(strong)} strong binders (total {found})"
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["peptide", "allele", "presentation_score"]
    )


def generate_decoys(
    fasta_path: str | Path,
    training_data_path: str | Path,
    output_path: str | Path,
    schema_path: str | Path,
    num_decoys: int = 10_000,
    alleles: list[str] | None = None,
    seed: int = 42,
    lengths: tuple[int, ...] = (8, 9, 10, 11),
    max_candidates: int | None = None,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    output_path = Path(output_path)
    alleles = alleles or CANONICAL_ALLELES

    # Load MHCflurry presentation predictor
    print("Loading MHCflurry Class1PresentationPredictor...")
    try:
        from mhcflurry import Class1PresentationPredictor
        predictor = Class1PresentationPredictor.load()
    except Exception as exc:
        print(f"Error loading MHCflurry: {exc}")
        print("Run: mhcflurry-downloads fetch models_class1_presentation")
        sys.exit(1)

    # Load training positives to exclude from decoys
    print(f"Loading training positives from {training_data_path}...")
    train_df = pd.read_csv(training_data_path)
    positive_peptides: frozenset[str] = frozenset(
        train_df.loc[train_df["label"] == 1, "peptide"].str.upper()
    )
    print(f"  Exclusion set: {len(positive_peptides)} positive peptides.")

    # Extract k-mers from self-proteome
    print(f"Extracting {lengths}-mers from {fasta_path}...")
    sequences = _load_fasta(fasta_path)
    print(f"  Loaded {len(sequences)} protein sequences.")
    kmers = _extract_kmers(sequences, lengths=lengths)
    print(f"  Extracted {len(kmers):,} unique valid k-mers.")

    # Seeded shuffle → reproducible decoy selection
    random.shuffle(kmers)

    # Pre-sample to bound MHCflurry scoring time while keeping random coverage.
    # A 1M seeded sample from ~20M unique k-mers yields ~5k-50k strong binders
    # per allele - far more than the per-allele quota - so decoy quality is
    # unaffected. Recorded in provenance for full reproducibility.
    if max_candidates is not None and len(kmers) > max_candidates:
        kmers = kmers[:max_candidates]
        print(f"  Pre-sampled to {len(kmers):,} candidates (--max-candidates).")

    # Screen each allele
    all_hits: list[pd.DataFrame] = []
    for allele in alleles:
        print(f"\nScoring {len(kmers):,} peptides for {allele}...")
        hits = _score_batched(kmers, allele, predictor)
        # Exclude training positives
        hits = hits[~hits["peptide"].isin(positive_peptides)].copy()
        print(f"  {len(hits)} strong binders after exclusion.")
        all_hits.append(hits)

    if not any(len(h) > 0 for h in all_hits):
        print("ERROR: No strong binders found across any allele.")
        sys.exit(1)

    combined = pd.concat(all_hits, ignore_index=True)
    print(f"\nTotal high-affinity self-peptide candidates: {len(combined):,}")

    # Deduplicate by (peptide, allele); keep highest presentation_score
    combined = (
        combined.sort_values("presentation_score", ascending=False)
        .drop_duplicates(subset=["peptide", "allele"])
        .reset_index(drop=True)
    )

    # Sample up to num_decoys, stratified by allele where possible
    per_allele = max(1, num_decoys // len(alleles))
    sampled_parts: list[pd.DataFrame] = []
    for allele in alleles:
        subset = combined[combined["allele"] == allele]
        n = min(per_allele, len(subset))
        if n > 0:
            sampled_parts.append(subset.sample(n=n, random_state=int(rng.integers(2**31))))
    sampled = pd.concat(sampled_parts, ignore_index=True)

    # If we're still under quota, fill from remaining high-scoring candidates
    already_selected = frozenset(zip(sampled["peptide"], sampled["allele"]))
    remaining = combined[
        ~combined.apply(lambda r: (r["peptide"], r["allele"]) in already_selected, axis=1)
    ]
    shortfall = num_decoys - len(sampled)
    if shortfall > 0 and len(remaining) > 0:
        extra = remaining.sample(n=min(shortfall, len(remaining)), random_state=int(rng.integers(2**31)))
        sampled = pd.concat([sampled, extra], ignore_index=True)

    # Format to v4 schema
    df_out = pd.DataFrame({
        "peptide": sampled["peptide"],
        "label": 0,
        "virus": "Self",
        "protein": "HumanProteome",
        "strain": "GRCh38",
        "hla_allele": sampled["allele"],
        "source_type": "Self",
        "database_source": "UniProt_UP000005640",
        "tcr_alpha_cdr3": None,
        "tcr_beta_cdr3": None,
    })
    df_out = normalize_peptides(df_out)
    df_out = df_out.drop_duplicates(subset=["peptide", "hla_allele"]).reset_index(drop=True)
    df_out = df_out.sort_values(["peptide", "hla_allele"]).reset_index(drop=True)

    print(f"\nFinal decoy set: {len(df_out)} rows")
    print("Allele distribution:")
    print(df_out["hla_allele"].value_counts().to_string())

    validate_against_schema(df_out, schema_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    write_provenance(
        str(output_path),
        sources=[str(fasta_path), str(training_data_path)],
        row_count=len(df_out),
        extra={
            "alleles": alleles,
            "num_decoys_requested": num_decoys,
            "presentation_threshold": PRESENTATION_THRESHOLD,
            "peptide_lengths": list(lengths),
            "seed": seed,
            "n_positive_excluded": len(positive_peptides),
            "mhcflurry_version": _get_mhcflurry_version(),
            "max_candidates": max_candidates,
        },
    )
    print(f"\nWrote {len(df_out)} hard decoys -> {output_path}")
    return df_out


def _get_mhcflurry_version() -> str:
    try:
        import mhcflurry
        return mhcflurry.__version__
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate hard-decoy negatives from the human self-proteome for SESTRAV v4.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Prerequisite: download human proteome FASTA first:\n"
            "  python scripts/fetch_human_proteome.py\n"
            "  mhcflurry-downloads fetch models_class1_presentation\n"
        ),
    )
    parser.add_argument(
        "--fasta", required=True,
        help="Path to human reference proteome FASTA (UniProt UP000005640, see fetch_human_proteome.py)",
    )
    parser.add_argument(
        "--training-data",
        default="data/immunogenicity_dataset_v3.csv",
        help="Training CSV to extract positives for exclusion (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="data/hard_decoys.csv",
        help="Output CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--schema",
        default="data/immunogenicity_dataset_v4_schema.json",
        help="v4 schema path for validation (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoys", type=int, default=10_000,
        help="Target number of hard decoys (default: %(default)s)",
    )
    parser.add_argument(
        "--alleles", nargs="+", default=CANONICAL_ALLELES,
        help="HLA alleles to screen (default: all 10 canonical alleles)",
    )
    parser.add_argument(
        "--lengths", nargs="+", type=int, default=[8, 9, 10, 11],
        help="Peptide lengths to extract (default: 8 9 10 11)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible runs (default: %(default)s)",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=None,
        help=(
            "Pre-sample this many k-mers (after seeded shuffle) before MHCflurry "
            "scoring to bound runtime. None = score all k-mers (default). "
            "1_000_000 is sufficient for a 5k decoy target."
        ),
    )
    args = parser.parse_args(argv)

    if not Path(args.fasta).exists():
        print(
            f"ERROR: FASTA not found: {args.fasta}\n"
            "Run: python scripts/fetch_human_proteome.py",
            file=sys.stderr,
        )
        return 1

    generate_decoys(
        fasta_path=args.fasta,
        training_data_path=args.training_data,
        output_path=args.output,
        schema_path=args.schema,
        num_decoys=args.num_decoys,
        alleles=args.alleles,
        seed=args.seed,
        lengths=tuple(args.lengths),
        max_candidates=args.max_candidates,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
