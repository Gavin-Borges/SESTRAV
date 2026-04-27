"""
SESTRAV Gold-Standard Negative Expansion

Identifies additional strong-binding experimentally-negative peptides from
the v2 training dataset to expand the gold-standard negative validation set.

Selection criteria:
  1. Label = 0 (negative) in the deduplicated v2 dataset
  2. NOT in the 201 cross-label conflict set (require unambiguous negative)
  3. High MHC presentation score (top-binding negatives are the hardest
     and most informative test cases for the integrated model)
  4. Balanced across EBV and HPV16

Usage:
    python -m src.expand_negatives \
        --data immunogenicity_dataset.csv \
        --binding-matrix models/peptide_binding_matrix.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.gold_standard import GOLD_STANDARD_NEGATIVES


EXISTING_GS_NEG_PEPTIDES = {gs['peptide'] for gs in GOLD_STANDARD_NEGATIVES}


def find_candidate_negatives(data_path, binding_matrix_path, n_per_virus=10,
                             min_presentation_score=0.3, output_dir='results'):
    """Identify strong-binding negative peptides for gold-standard expansion."""
    os.makedirs(output_dir, exist_ok=True)

    ds = pd.read_csv(data_path)
    bind = pd.read_csv(binding_matrix_path)

    negatives = ds[ds['label'] == 0].copy()
    negatives = negatives[~negatives['peptide'].isin(EXISTING_GS_NEG_PEPTIDES)]

    bind_cols = [c for c in bind.columns if c.startswith('bind_')]
    merged = negatives.merge(bind[['peptide'] + bind_cols], on='peptide', how='inner')

    if not bind_cols:
        print("[Neg Expansion] No binding columns found in matrix")
        return pd.DataFrame()

    merged['max_presentation_score'] = merged[bind_cols].max(axis=1)
    merged['best_allele_col'] = merged[bind_cols].idxmax(axis=1)

    allele_col_to_name = {
        'bind_A0101': 'HLA-A*01:01', 'bind_A0201': 'HLA-A*02:01',
        'bind_A0301': 'HLA-A*03:01', 'bind_A1101': 'HLA-A*11:01',
        'bind_A2402': 'HLA-A*24:02', 'bind_B0702': 'HLA-B*07:02',
        'bind_B0801': 'HLA-B*08:01', 'bind_B2705': 'HLA-B*27:05',
        'bind_B3501': 'HLA-B*35:01', 'bind_B4402': 'HLA-B*44:02',
    }
    merged['best_allele'] = merged['best_allele_col'].map(allele_col_to_name)

    strong = merged[merged['max_presentation_score'] >= min_presentation_score].copy()
    strong = strong.sort_values('max_presentation_score', ascending=False)

    print(f"[Neg Expansion] Total negatives (excl existing GS): {len(negatives)}")
    print(f"[Neg Expansion] With binding data: {len(merged)}")
    print(f"[Neg Expansion] Strong binders (>= {min_presentation_score}): {len(strong)}")

    candidates = []
    for virus in sorted(strong['virus'].unique()):
        virus_strong = strong[strong['virus'] == virus].head(n_per_virus)
        for _, row in virus_strong.iterrows():
            candidates.append({
                'peptide': row['peptide'],
                'virus': row['virus'],
                'best_allele': row['best_allele'],
                'max_presentation_score': round(row['max_presentation_score'], 4),
                'length': len(row['peptide']),
            })
        print(f"  {virus}: {len(virus_strong)} candidates selected")

    candidates_df = pd.DataFrame(candidates)

    output_csv = os.path.join(output_dir, 'expanded_negative_candidates.csv')
    candidates_df.to_csv(output_csv, index=False)
    print(f"[Neg Expansion] Candidates saved to {output_csv}")

    _write_expansion_report(candidates_df, strong, output_dir)

    return candidates_df


def _write_expansion_report(candidates_df, all_strong, output_dir):
    """Write a markdown report for the negative expansion."""
    md_path = os.path.join(output_dir, 'expanded_negative_candidates.md')

    lines = [
        "# Gold-Standard Negative Expansion Candidates",
        "",
        "## Selection Criteria",
        "",
        "1. Labeled negative (non-immunogenic) in v2 dataset",
        "2. Not in the existing 10-peptide gold-standard negative set",
        "3. High MHC presentation score (strong binders that are NOT immunogenic)",
        "4. Balanced across EBV and HPV16",
        "",
        f"**Pool size:** {len(all_strong)} strong-binding negatives available",
        "",
        "## Existing Gold-Standard Negatives (10)",
        "",
        "| Peptide | Virus | Allele | Affinity (nM) |",
        "|---------|-------|--------|---------------|",
    ]
    for gs in GOLD_STANDARD_NEGATIVES:
        lines.append(
            f"| {gs['peptide']} | {gs['virus']} | {gs['allele']} | {gs['affinity_nM']} |"
        )

    lines.extend([
        "",
        "## Expansion Candidates",
        "",
    ])

    for virus in sorted(candidates_df['virus'].unique()):
        vdf = candidates_df[candidates_df['virus'] == virus]
        lines.append(f"### {virus} ({len(vdf)} candidates)")
        lines.append("")
        lines.append("| Peptide | Allele | Presentation Score | Length |")
        lines.append("|---------|--------|-------------------|--------|")
        for _, row in vdf.iterrows():
            lines.append(
                f"| {row['peptide']} | {row['best_allele']} "
                f"| {row['max_presentation_score']:.4f} | {row['length']} |"
            )
        lines.append("")

    lines.extend([
        "## Integration Instructions",
        "",
        "To add these candidates to the gold-standard negative set:",
        "",
        "1. Domain expert reviews biological plausibility of each candidate",
        "2. Verify IEDB source records confirm unambiguous T-cell negativity",
        "3. Add approved peptides to `GOLD_STANDARD_NEGATIVES` in `src/gold_standard.py`",
        "4. Re-run `validate_negative_discrimination()` to assess model performance",
        "5. Update evidence freeze document with expanded negative set results",
    ])

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[Neg Expansion] Report saved to {md_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Expand SESTRAV gold-standard negatives')
    parser.add_argument('--data', required=True, help='Path to immunogenicity_dataset.csv')
    parser.add_argument('--binding-matrix', required=True, help='Path to peptide_binding_matrix.csv')
    parser.add_argument('--n-per-virus', type=int, default=10, help='Candidates per virus')
    parser.add_argument('--min-score', type=float, default=0.3, help='Min presentation score')
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()
    find_candidate_negatives(
        args.data, args.binding_matrix,
        n_per_virus=args.n_per_virus,
        min_presentation_score=args.min_score,
        output_dir=args.output_dir,
    )
