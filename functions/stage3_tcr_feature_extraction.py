"""
SESTRAV Stage 3 — TCR Feature Extraction
Computes physicochemical properties at TCR contact positions p4-p8 for
each peptide, using the shared feature hub in src/features.py.
Canonical 30-feature mode adds 10 per-allele binding columns from Stage 2.
"""

from src.features import compute_features_for_dataset
import re

def _sanitize_name(name):
    """Allow only alphanumeric, underscores, and hyphens."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)


def extract_tcr_features(binding_df, proteome_id):
    proteome_id = _sanitize_name(proteome_id)
    """
    Extract 22 per-position physicochemical features for all peptide rows.

    Uses presentation_score from Stage 2 as the binding_score feature.
    Zero-imputes positions that fall outside the TCR core for short peptides.

    Args:
        binding_df: DataFrame from Stage 2 with 'peptide' and 'presentation_score'
        proteome_id: label used in output filename

    Returns:
        DataFrame with original columns + 22 new feature columns
    """
    binding_col = 'presentation_score'
    if binding_col not in binding_df.columns:
        binding_col = 'affinity'

    features_df = compute_features_for_dataset(
        binding_df,
        peptide_col='peptide',
        binding_col=binding_col
    )

    output_path = f"results/{proteome_id}_features.csv"
    features_df.to_csv(output_path, index=False)
    n_feat = len([c for c in features_df.columns if c.startswith(('p4_', 'p5_', 'p6_', 'p7_', 'p8_', 'bind_', 'binding_', 'peptide_length'))])
    print(f"[Stage 3] Extracted {n_feat} features for {len(features_df)} peptide-allele pairs")
    return features_df
