"""
Build PredIG CSV-Recombinant input from SESTRAV external validation sidecars.

Maps each peptide-HLA pair to parent protein sequence from panel8 proteome FASTA
files for PredIG --type recombinant --model path.

Usage:
    python -m src.build_predig_recombinant_input [--repo-root .] [--results-dir results]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.prepare_external_validation_inputs import DEFAULT_VIRUS_TO_PROTEOME

PROTEOME_FASTA = {
    "EBV_B95_8_panel8": "EBV_B95_8_panel8.fasta",
    "HPV16_18_panel8": "HPV16_18_panel8.fasta",
}

VIRUS_FULL_FASTA = {
    "EBV": "EBV_B95_8_uniprot_reviewed.fasta",
    "HPV16": "HPV16_uniprot_reviewed.fasta",
    "HPV": "HPV16_uniprot_reviewed.fasta",
    "HPV18": "HPV16_uniprot_reviewed.fasta",
}


def synthetic_protein_context(peptide: str, min_len: int = 60) -> Tuple[str, str]:
    """Minimal recombinant context when no parent protein is found in FASTA."""
    pep = peptide.upper()
    pad = max(0, min_len - len(pep))
    left = "G" * (pad // 2)
    right = "G" * (pad - pad // 2)
    seq = left + pep + right
    return f"SYNTH_{pep}", seq


def parse_fasta(path: str) -> Dict[str, Tuple[str, str]]:
    """Return {protein_name: (header, sequence)}."""
    proteins: Dict[str, Tuple[str, str]] = {}
    header: Optional[str] = None
    seq_parts: List[str] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    name = _protein_name_from_header(header)
                    proteins[name] = (header, "".join(seq_parts))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line.upper())

    if header is not None:
        name = _protein_name_from_header(header)
        proteins[name] = (header, "".join(seq_parts))

    return proteins


def _protein_name_from_header(header: str) -> str:
    """Use UniProt-style ID token when present, else first whitespace token."""
    parts = header.split()
    if len(parts) >= 2 and parts[0].startswith("sp|"):
        return parts[0].split("|")[-1] if "|" in parts[0] else parts[1]
    return parts[0]


def find_parent_protein(
    peptide: str, proteins: Dict[str, Tuple[str, str]]
) -> Optional[Tuple[str, str]]:
    """First protein containing peptide (case-insensitive search)."""
    pep = peptide.upper()
    for name, (_hdr, seq) in proteins.items():
        if pep in seq:
            return name, seq
    return None


def build_recombinant_table(
    pairs_path: str,
    validation_input_path: str,
    proteome_dir: str,
    output_path: str,
) -> Tuple[pd.DataFrame, int]:
    pairs = pd.read_csv(pairs_path)
    meta = pd.read_csv(validation_input_path)[["peptide", "proteome_id", "virus"]].drop_duplicates(
        subset=["peptide"]
    )
    merged = pairs.merge(meta, on="peptide", how="left")

    fasta_cache: Dict[str, Dict[str, Tuple[str, str]]] = {}
    rows = []
    unmapped: List[str] = []
    synthetic_count = 0

    def _load_fasta(filename: str) -> Dict[str, Tuple[str, str]]:
        if filename not in fasta_cache:
            fasta_path = os.path.join(proteome_dir, filename)
            if not os.path.isfile(fasta_path):
                return {}
            fasta_cache[filename] = parse_fasta(fasta_path)
        return fasta_cache[filename]

    for _, row in merged.iterrows():
        peptide = str(row["peptide"])
        allele = str(row["allele"])
        proteome_id = str(row["proteome_id"])
        virus = str(row.get("virus", ""))

        hit: Optional[Tuple[str, str]] = None

        if proteome_id in PROTEOME_FASTA:
            proteins = _load_fasta(PROTEOME_FASTA[proteome_id])
            if proteins:
                hit = find_parent_protein(peptide, proteins)

        if hit is None and virus in VIRUS_FULL_FASTA:
            proteins = _load_fasta(VIRUS_FULL_FASTA[virus])
            if proteins:
                hit = find_parent_protein(peptide, proteins)

        if hit is None:
            protein_name, protein_seq = synthetic_protein_context(peptide)
            synthetic_count += 1
            unmapped.append(peptide)
        else:
            protein_name, protein_seq = hit

        rows.append(
            {
                "epitope": peptide,
                "HLA_allele": allele,
                "protein_seq": protein_seq,
                "protein_name": protein_name,
            }
        )

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"[predig-input] Wrote {output_path} ({len(out)} rows)")
    if synthetic_count:
        unique_unmapped = sorted(set(unmapped))
        print(
            f"[predig-input] NOTE: {synthetic_count} rows used synthetic protein context "
            f"({len(unique_unmapped)} unique peptides not found in FASTA)",
            file=sys.stderr,
        )

    return out, synthetic_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PredIG CSV-Recombinant input")
    parser.add_argument("--repo-root", default=".", help="SESTRAV-Dev root")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV (default: results/external_predig_input_recombinant.csv)",
    )
    parser.add_argument(
        "--pairs",
        default=None,
        help="Peptide-allele pairs CSV (default: external_predig_peptide_allele_pairs.csv)",
    )
    parser.add_argument(
        "--meta",
        default=None,
        help="Peptide metadata CSV with peptide, proteome_id, virus (default: external_validation_input.csv)",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.repo_root)
    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(root, results_dir)

    pairs_path = args.pairs or os.path.join(
        results_dir, "external_predig_peptide_allele_pairs.csv"
    )
    validation_path = args.meta or os.path.join(
        results_dir, "external_validation_input.csv"
    )
    proteome_dir = os.path.join(root, "data", "proteomes")
    output_path = args.output or os.path.join(
        results_dir, "external_predig_input_recombinant.csv"
    )

    for path in (pairs_path, validation_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    out, synthetic_count = build_recombinant_table(
        pairs_path, validation_path, proteome_dir, output_path
    )

    expected_pairs = len(pd.read_csv(pairs_path))
    if len(out) != expected_pairs:
        print(
            f"[predig-input] ERROR: expected {expected_pairs} rows, got {len(out)}",
            file=sys.stderr,
        )
        sys.exit(1)

    if synthetic_count:
        print(
            f"[predig-input] Synthetic context rows: {synthetic_count} "
            f"(see stderr for peptide list sample)"
        )


if __name__ == "__main__":
    main()
