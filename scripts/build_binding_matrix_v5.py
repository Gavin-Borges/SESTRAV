"""
Build the v5 MHC peptide binding matrix incrementally.

Computes MHCflurry presentation scores for peptides not already in the v4
matrix, then merges and writes the combined v5 matrix.

Usage:
    python scripts/build_binding_matrix_v5.py \\
        --dataset data/immunogenicity_dataset_v5.csv \\
        --existing-matrix models/peptide_binding_matrix_v4.csv \\
        --output models/peptide_binding_matrix_v5.csv \\
        [--dry-run]
"""

from __future__ import annotations

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

ALLELE_MAP: dict[str, str] = {
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
ALLELE_COLS = list(ALLELE_MAP.keys())


def find_new_peptides(active_peps: set[str], existing_peps: set[str]) -> set[str]:
    """Return peptides in active_peps not yet covered by existing_peps."""
    return active_peps - existing_peps


def merge_matrices(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate two binding matrices, deduplicate on peptide, sort by peptide."""
    merged = pd.concat([existing_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset="peptide", keep="first")
    merged = merged.sort_values("peptide").reset_index(drop=True)
    return merged


def _write_provenance(
    output_path: Path,
    dataset_path: Path,
    existing_matrix_path: Path,
    new_peptide_count: int,
    total_peptide_count: int,
) -> None:
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
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_sha": git_sha,
        "source_dataset": str(dataset_path.name),
        "existing_matrix": str(existing_matrix_path.name),
        "new_peptide_count": new_peptide_count,
        "total_peptide_count": total_peptide_count,
        "alleles": list(ALLELE_MAP.values()),
    }
    prov_path = output_path.with_suffix(".provenance.json")
    prov_path.write_text(json.dumps(provenance, indent=2))
    print(f"Provenance sidecar: {prov_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build v5 peptide binding matrix incrementally"
    )
    parser.add_argument(
        "--dataset",
        default="data/immunogenicity_dataset_v5.csv",
        help="Input v5 immunogenicity dataset",
    )
    parser.add_argument(
        "--existing-matrix",
        default="models/peptide_binding_matrix_v4.csv",
        help="Existing v4 binding matrix to extend",
    )
    parser.add_argument(
        "--output",
        default="models/peptide_binding_matrix_v5.csv",
        help="Output v5 binding matrix path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing files or calling MHCflurry",
    )
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / args.dataset

    existing_matrix_path = Path(args.existing_matrix)
    if not existing_matrix_path.is_absolute():
        existing_matrix_path = PROJECT_ROOT / args.existing_matrix

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / args.output

    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    if not existing_matrix_path.exists():
        print(f"ERROR: existing matrix not found: {existing_matrix_path}", file=sys.stderr)
        return 1

    ds = pd.read_csv(dataset_path)
    if "is_quarantined" in ds.columns:
        active_df = ds[~ds["is_quarantined"].astype(bool)]
    else:
        active_df = ds
    active_peps: set[str] = set(active_df["peptide"].dropna().unique())
    print(f"Loaded {len(active_peps)} unique active peptides from {dataset_path.name}")

    existing_df = pd.read_csv(existing_matrix_path)
    existing_peps: set[str] = set(existing_df["peptide"].dropna().unique())
    print(f"Existing matrix covers {len(existing_peps)} peptides")

    new_peps_set = find_new_peptides(active_peps, existing_peps)
    new_peptides = sorted(new_peps_set)
    already_covered = len(active_peps) - len(new_peptides)
    print(f"Already covered: {already_covered}")
    print(f"New peptides to compute: {len(new_peptides)}")

    if args.dry_run:
        print("\n[dry-run] Would compute MHCflurry scores for the above peptides.")
        if new_peptides:
            sample = new_peptides[:5]
            print(f"[dry-run] Sample new peptides: {sample}")
        print(f"[dry-run] Would write to: {output_path}")
        return 0

    if not new_peptides:
        print("No new peptides - copying existing matrix to output.")
        existing_df.to_csv(output_path, index=False)
        _write_provenance(output_path, dataset_path, existing_matrix_path, 0, len(existing_df))
        return 0

    from mhcflurry import Class1PresentationPredictor

    predictor = Class1PresentationPredictor.load()

    per_allele_scores: dict[str, list[float]] = {}
    for col, allele in zip(ALLELE_COLS, list(ALLELE_MAP.values())):
        res = predictor.predict(
            peptides=new_peptides,
            alleles=[allele],
            verbose=False,
        )
        score_map = dict(zip(res["peptide"], res["presentation_score"]))
        per_allele_scores[col] = [score_map.get(p, 0.0) for p in new_peptides]
        nonzero = sum(1 for s in per_allele_scores[col] if s != 0.0)
        print(f"  Predicting {allele} ... {nonzero}/{len(new_peptides)} non-zero")

    new_rows = pd.DataFrame({"peptide": new_peptides, **per_allele_scores})
    merged = merge_matrices(existing_df, new_rows)

    merged.to_csv(output_path, index=False)
    print(f"Saved binding matrix: {output_path} ({len(merged)} rows)")

    _write_provenance(
        output_path,
        dataset_path,
        existing_matrix_path,
        len(new_peptides),
        len(merged),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
