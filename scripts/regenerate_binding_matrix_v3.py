"""
Regenerate peptide_binding_matrix_v3.csv with real MHCflurry presentation scores.

The v3 matrix was committed as an all-zeros placeholder (commit f360b90).
This script populates it with actual MHCflurry 2.x Class1PresentationPredictor
scores - the same scorer used to generate peptide_binding_matrix.csv (v2).

Usage:
    python scripts/regenerate_binding_matrix_v3.py
    python scripts/regenerate_binding_matrix_v3.py --dataset data/immunogenicity_dataset_v3.csv --output models/peptide_binding_matrix_v3.csv
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ALLELE_MAP = {
    "bind_A0101": "HLA-A*01:01",
    "bind_A0201": "HLA-A*02:01",
    "bind_A0301": "HLA-A*03:01",
    "bind_A1101": "HLA-A*11:01",
    "bind_A2402": "HLA-A*24:02",
    "bind_B0702": "HLA-B*07:02",
    "bind_B0801": "HLA-B*08:01",
    "bind_B2705": "HLA-B*27:05",
    "bind_B3501": "HLA-B*35:01",
    "bind_B4402": "HLA-B*44:02",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate v3 peptide binding matrix")
    parser.add_argument(
        "--dataset",
        default="data/immunogenicity_dataset_v3.csv",
        help="Input immunogenicity dataset (default: data/immunogenicity_dataset_v3.csv)",
    )
    parser.add_argument(
        "--output",
        default="models/peptide_binding_matrix_v3.csv",
        help="Output binding matrix path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing files",
    )
    args = parser.parse_args()

    dataset_path = PROJECT_ROOT / args.dataset
    output_path = PROJECT_ROOT / args.output

    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    peptides = df["peptide"].unique().tolist()
    print(f"Loaded {len(df)} rows, {len(peptides)} unique peptides from {dataset_path.name}")
    print(
        f"Length distribution: {dict(pd.Series([len(p) for p in peptides]).value_counts().sort_index())}"
    )

    allele_cols = list(ALLELE_MAP.keys())
    alleles = list(ALLELE_MAP.values())

    print(
        f"\nRunning MHCflurry Class1PresentationPredictor on {len(peptides)} peptides x {len(alleles)} alleles..."
    )
    from mhcflurry import Class1PresentationPredictor

    predictor = Class1PresentationPredictor.load()

    # Predict per allele (Class1PresentationPredictor treats a list as a genotype,
    # so we iterate one allele at a time to get per-allele presentation scores).
    per_allele_scores: dict[str, list[float]] = {}
    for col, allele in zip(allele_cols, alleles):
        print(f"  Predicting {allele} ...", end="", flush=True)
        res = predictor.predict(
            peptides=peptides,
            alleles=[allele],
            verbose=False,
        )
        # presentation_score is keyed by the allele name in sample_name column
        score_map = dict(zip(res["peptide"], res["presentation_score"]))
        per_allele_scores[col] = [score_map.get(p, 0.0) for p in peptides]
        nonzero = sum(1 for s in per_allele_scores[col] if s != 0.0)
        print(f" {nonzero}/{len(peptides)} non-zero")

    pivot = pd.DataFrame({"peptide": peptides, **per_allele_scores})

    print(f"\nBinding matrix summary (non-zero values): {(pivot[allele_cols] != 0).sum().sum()}")
    print("Score stats per allele:")
    print(pivot[allele_cols].describe().loc[["mean", "min", "max"]].to_string())

    if args.dry_run:
        print("\n[dry-run] Would write to:", output_path)
        return

    pivot.to_csv(output_path, index=False)
    print(f"\nSaved binding matrix: {output_path} ({len(pivot)} rows)")

    # Write provenance sidecar
    git_sha = ""
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        pass

    provenance = {
        "source_dataset": str(dataset_path.name),
        "mhcflurry_version": __import__("mhcflurry").__version__,
        "alleles": alleles,
        "peptide_count": len(pivot),
        "non_zero_values": int((pivot[allele_cols] != 0).sum().sum()),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "git_sha": git_sha,
        "note": "Regenerated from all-zeros placeholder (commit f360b90). Uses presentation_score from Class1PresentationPredictor.",
    }
    prov_path = output_path.with_suffix(".provenance.json")
    prov_path.write_text(json.dumps(provenance, indent=2))
    print(f"Provenance sidecar: {prov_path}")


if __name__ == "__main__":
    main()
