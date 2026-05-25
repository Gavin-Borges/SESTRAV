"""
SESTRAV Feature Extraction Module — The Shared Hub

Computes 22 per-position physicochemical features at TCR contact positions p4-p8.
This module feeds BOTH the RF/XGBoost classifiers and the CMB 523 ANN benchmark.

TCR contact position mapping (Chowell et al. 2015, PredIG convention):
  p4-p6 are N-terminal-anchored at fixed 0-based indices 3, 4, 5.
  p7 is C-terminal-relative at index (length - 3).
  p8 is C-terminal-relative at index (length - 2).

  For 8-mers:  p4=3, p5=4, p6=5, p7=5 (=p6)→zero, p8=6→zero (both excluded)
  For 9-mers:  p4=3, p5=4, p6=5, p7=6,             p8=7
  For 10-mers: p4=3, p5=4, p6=5, p7=7,             p8=8
  For 11-mers: p4=3, p5=4, p6=5, p7=8,             p8=9

  Positions that overlap with an earlier position (p7 == p6 for 8-mers)
  or fall at or beyond the C-terminal anchor (index >= length - 1) are
  zero-imputed to avoid double-counting and anchor contamination.

Feature units and ranges:
  p{4-8}_hydrophobicity : Kyte-Doolittle scale, unitless, range [-4.5, +4.5]
  p{4-8}_aromaticity    : binary indicator (1 = F/W/Y/H, 0 = other), unitless
  p{4-8}_vdw_volume     : Zamyatnin 1972, Angstrom^3, range [48, 163]
  p{4-8}_charge          : formal charge at pH 7 (K/R = +1, D/E = -1, else 0)
  binding_score          : MHCflurry presentation_score, unitless [0, 1]
                           (set to 0 during training; real values during inference)
  peptide_length         : integer amino acid count, range [8, 11]

Training uses 21 features (binding_score excluded).  The full 22-feature set
is available at inference in the pipeline where MHCflurry scores are present.

Multi-allele 30-feature mode (CMB 523 Project 2):
  Replaces the single binding_score with 10 per-allele MHCflurry presentation
  scores (bind_A0101 ... bind_B4402) for a total of 20 physico + 10 binding = 30
  features.  An optional 31st feature (peptide_length) is also defined.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Published amino acid property lookup tables
# ---------------------------------------------------------------------------

KD_HYDRO = {
    'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C':  2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
    'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2
}

VDW_VOL = {
    'A':  67, 'R': 148, 'N':  96, 'D':  91, 'C':  86,
    'E': 109, 'Q': 114, 'G':  48, 'H': 118, 'I': 124,
    'L': 124, 'K': 135, 'M': 124, 'F': 135, 'P':  90,
    'S':  73, 'T':  93, 'W': 163, 'Y': 141, 'V': 105
}

AROMATIC = {
    'F': 1, 'W': 1, 'Y': 1, 'H': 1
}

CHARGE = {
    'K':  1, 'R':  1, 'D': -1, 'E': -1
}

# Vihinen et al. 1994 Flexibility Scale
FLEXIBILITY = {
    'A': 0.984, 'C': 0.906, 'D': 1.068, 'E': 1.094, 'F': 0.915,
    'G': 1.031, 'H': 0.950, 'I': 0.927, 'K': 1.102, 'L': 0.935,
    'M': 0.952, 'N': 1.048, 'P': 1.049, 'Q': 1.037, 'R': 1.008,
    'S': 1.046, 'T': 0.997, 'V': 0.931, 'W': 0.904, 'Y': 0.929
}

# Zimmerman et al. 1968 Bulkiness
BULKINESS = {
    'A': 11.5, 'C': 13.46, 'D': 11.68, 'E': 13.57, 'F': 19.8,
    'G': 3.4,  'H': 13.69, 'I': 21.4,  'K': 15.71, 'L': 21.4,
    'M': 16.25, 'N': 12.82, 'P': 17.43, 'Q': 14.45, 'R': 14.28,
    'S': 9.47,  'T': 15.77, 'V': 21.57, 'W': 21.67, 'Y': 18.03
}

# Hopp & Woods 1981 Hydrophilicity
HYDROPHILICITY = {
    'A': -0.5, 'C': -1.0, 'D': 3.0, 'E': 3.0, 'F': -2.5,
    'G': 0.0,  'H': -0.5, 'I': -1.8, 'K': 3.0, 'L': -1.8,
    'M': -1.3, 'N': 0.2,  'P': 0.0,  'Q': 0.2, 'R': 3.0,
    'S': 0.3,  'T': -0.4, 'V': -1.5, 'W': -3.4, 'Y': -2.3
}

# TCR Contact Upward-Facing Probabilities (proxy metric based on peptide length and position)
# Estimates the likelihood that a residue is prominently exposed to the TCR (derived from structural alignments)
UPWARD_PROBABILITY = {
    8:  {'p4': 0.8, 'p5': 0.9, 'p6': 0.8, 'p7': 0.0, 'p8': 0.0},
    9:  {'p4': 0.6, 'p5': 0.9, 'p6': 0.9, 'p7': 0.7, 'p8': 0.6},
    10: {'p4': 0.5, 'p5': 0.8, 'p6': 0.9, 'p7': 0.8, 'p8': 0.5},
    11: {'p4': 0.4, 'p5': 0.7, 'p6': 0.8, 'p7': 0.8, 'p8': 0.4},
}

FEATURE_COLUMNS = [
    'p4_hydrophobicity', 'p5_hydrophobicity', 'p6_hydrophobicity',
    'p7_hydrophobicity', 'p8_hydrophobicity',
    'p4_aromaticity', 'p5_aromaticity', 'p6_aromaticity',
    'p7_aromaticity', 'p8_aromaticity',
    'p4_vdw_volume', 'p5_vdw_volume', 'p6_vdw_volume',
    'p7_vdw_volume', 'p8_vdw_volume',
    'p4_charge', 'p5_charge', 'p6_charge',
    'p7_charge', 'p8_charge',
    'binding_score', 'peptide_length'
]

EXPANDED_FEATURE_COLUMNS = FEATURE_COLUMNS[:-2] + [
    'p4_flexibility', 'p5_flexibility', 'p6_flexibility',
    'p7_flexibility', 'p8_flexibility',
    'p4_bulkiness', 'p5_bulkiness', 'p6_bulkiness',
    'p7_bulkiness', 'p8_bulkiness',
    'p4_hydrophilicity', 'p5_hydrophilicity', 'p6_hydrophilicity',
    'p7_hydrophilicity', 'p8_hydrophilicity',
    'p4_upward_prob', 'p5_upward_prob', 'p6_upward_prob',
    'p7_upward_prob', 'p8_upward_prob',
    'binding_score', 'peptide_length'
]

TRAIN_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c != 'binding_score']

PHYSICO_COLUMNS = [c for c in FEATURE_COLUMNS
                   if c not in ('binding_score', 'peptide_length')]

EXPANDED_PHYSICO_COLUMNS = [c for c in EXPANDED_FEATURE_COLUMNS
                            if c not in ('binding_score', 'peptide_length')]

BINDING_ALLELE_COLUMNS = [
    'bind_A0101', 'bind_A0201', 'bind_A0301', 'bind_A1101', 'bind_A2402',
    'bind_B0702', 'bind_B0801', 'bind_B2705', 'bind_B3501', 'bind_B4402',
]

FEATURE_COLUMNS_30 = PHYSICO_COLUMNS + BINDING_ALLELE_COLUMNS
FEATURE_COLUMNS_31 = FEATURE_COLUMNS_30 + ['peptide_length']

FEATURE_COLUMNS_50 = EXPANDED_PHYSICO_COLUMNS + BINDING_ALLELE_COLUMNS

# ---------------------------------------------------------------------------
# HLA Pocket Pseudo-sequence feature columns (allele-aware extension)
# ---------------------------------------------------------------------------
# NetMHCpan 4.1 convention: 34 variable MHC pocket residues, each encoded
# with 4 physicochemical properties → 136 features total.

HLA_PSEUDO_LEN = 34  # canonical pocket length per NetMHCpan

HLA_PSEUDO_COLS = [
    f"hla_p{i+1}_{prop}"
    for i in range(HLA_PSEUDO_LEN)
    for prop in ("hydrophobicity", "aromaticity", "vdw_volume", "charge")
]
# 34 positions × 4 properties = 136 allele features

# Allele-aware 166-feature set: 20 physico + 10 binding + 136 HLA pocket
FEATURE_COLUMNS_ALLELE = FEATURE_COLUMNS_30 + HLA_PSEUDO_COLS

# ---------------------------------------------------------------------------
# Sample weight helpers for bias correction
# ---------------------------------------------------------------------------

def compute_sample_weights(df, virus_col='virus', length_col=None,
                            virus_weight=0.5, length_weight=0.5):
    """Compute per-sample training weights to correct EBV/HPV16 and 9-mer bias.

    Strategy:
      1. Virus balance weight: if EBV is 65% of data, up-weight HPV16 to equalize.
      2. Length balance weight: if 9-mers are 57% of data, up-weight non-9-mers.
      3. Final weight = geometric mean of both correction factors.

    Args:
        df:            DataFrame with at minimum a virus column.
        virus_col:     Column name for virus label (default 'virus').
        length_col:    Column name for peptide sequence (default None; uses 'peptide').
        virus_weight:  Weight applied to virus correction (0-1).
        length_weight: Weight applied to length correction (0-1).

    Returns:
        np.ndarray of per-sample weights, shape (len(df),).
    """
    import numpy as np
    n = len(df)
    weights = np.ones(n, dtype=float)

    # Virus correction
    if virus_col in df.columns:
        virus_vals = df[virus_col].values
        # Dynamically extract all unique viruses
        import pandas as pd
        unique_viruses = pd.Series(virus_vals).dropna().unique().tolist()
        if len(unique_viruses) > 1:
            target_freq = 1.0 / len(unique_viruses)  # Equal weight for all taxa
            for virus in unique_viruses:
                mask = (virus_vals == virus)
                actual_freq = mask.sum() / n
                correction = target_freq / max(actual_freq, 1e-6)
                weights[mask] *= (1.0 - virus_weight) + virus_weight * correction

    # Length correction
    pep_col = length_col if length_col and length_col in df.columns else 'peptide'
    if pep_col in df.columns:
        lengths = df[pep_col].str.len().values
        is_9mer = (lengths == 9)
        freq_9mer = is_9mer.mean()
        freq_non9 = 1.0 - freq_9mer
        if freq_9mer > 0 and freq_non9 > 0:
            target_len_freq = 0.5
            corr_9 = target_len_freq / freq_9mer
            corr_non9 = target_len_freq / freq_non9
            weights[is_9mer] *= (1.0 - length_weight) + length_weight * corr_9
            weights[~is_9mer] *= (1.0 - length_weight) + length_weight * corr_non9

    # Normalize so mean weight = 1 (keeps loss scale stable)
    weights = weights / weights.mean()
    return weights

ALL_POSITION_LABELS = ('p4', 'p5', 'p6', 'p7', 'p8')

PROPERTY_TABLES = {
    'hydrophobicity': (KD_HYDRO, 0.0),
    'aromaticity':    (AROMATIC, 0),
    'vdw_volume':     (VDW_VOL, 0.0),
    'charge':         (CHARGE, 0),
    'flexibility':    (FLEXIBILITY, 0.0),
    'bulkiness':      (BULKINESS, 0.0),
    'hydrophilicity': (HYDROPHILICITY, 0.0),
}


def get_tcr_positions(length):
    """Return TCR contact position (label, 0-based index) pairs for a peptide.

    p4-p6 are N-terminal-anchored (fixed indices 3, 4, 5).
    p7 and p8 are C-terminal-relative (length-3 and length-2).

    A position is valid when its index:
      - falls within the peptide (0 <= idx < length),
      - does not overlap with an earlier fixed position (p7 > p6),
      - stays below the C-terminal anchor residue (idx < length - 1).
    Invalid positions are returned as (label, None) so callers can
    zero-impute them.
    """
    fixed = [('p4', 3), ('p5', 4), ('p6', 5)]
    p7_idx = length - 3
    p8_idx = length - 2

    p7_valid = p7_idx > 5 and p7_idx < length - 1
    p8_valid = p7_valid and p8_idx > p7_idx and p8_idx < length - 1

    return fixed + [
        ('p7', p7_idx if p7_valid else None),
        ('p8', p8_idx if p8_valid else None),
    ]


def compute_features(peptide, binding_score=0.0):
    """Compute 22 physicochemical features for a single peptide.

    Returns a dict with 22 feature values keyed by FEATURE_COLUMNS names.
    Positions outside the valid TCR core are zero-imputed.
    """
    features = {}
    length = len(peptide)
    features['peptide_length'] = length
    features['binding_score'] = binding_score

    positions = get_tcr_positions(length)

    for pos_label, idx in positions:
        # structural proxy probability based on length and pos_label
        upward_prob = UPWARD_PROBABILITY.get(length, UPWARD_PROBABILITY[9]).get(pos_label, 0.0)
        
        if idx is not None:
            aa = peptide[idx]
            features[f'{pos_label}_hydrophobicity'] = KD_HYDRO.get(aa, 0.0)
            features[f'{pos_label}_aromaticity'] = AROMATIC.get(aa, 0)
            features[f'{pos_label}_vdw_volume'] = VDW_VOL.get(aa, 0.0)
            features[f'{pos_label}_charge'] = CHARGE.get(aa, 0)
            features[f'{pos_label}_flexibility'] = FLEXIBILITY.get(aa, 0.0)
            features[f'{pos_label}_bulkiness'] = BULKINESS.get(aa, 0.0)
            features[f'{pos_label}_hydrophilicity'] = HYDROPHILICITY.get(aa, 0.0)
            features[f'{pos_label}_upward_prob'] = upward_prob
        else:
            features[f'{pos_label}_hydrophobicity'] = 0.0
            features[f'{pos_label}_aromaticity'] = 0
            features[f'{pos_label}_vdw_volume'] = 0.0
            features[f'{pos_label}_charge'] = 0
            features[f'{pos_label}_flexibility'] = 0.0
            features[f'{pos_label}_bulkiness'] = 0.0
            features[f'{pos_label}_hydrophilicity'] = 0.0
            features[f'{pos_label}_upward_prob'] = upward_prob

    return features


def compute_features_for_dataset(df, peptide_col='peptide',
                                 binding_col='presentation_score'):
    """
    Apply compute_features() to every row in a DataFrame.
    Returns the original DataFrame with 22 new feature columns appended.
    """
    feature_records = []
    for _, row in df.iterrows():
        peptide = row[peptide_col]
        binding = row.get(binding_col, 0.0)
        if pd.isna(binding):
            binding = 0.0
        feats = compute_features(peptide, binding_score=binding)
        feature_records.append(feats)
    features_df = pd.DataFrame(feature_records)
    return pd.concat([df.reset_index(drop=True),
                      features_df.reset_index(drop=True)], axis=1)
