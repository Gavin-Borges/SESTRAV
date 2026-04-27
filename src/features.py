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

TRAIN_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c != 'binding_score']

PHYSICO_COLUMNS = [c for c in FEATURE_COLUMNS
                   if c not in ('binding_score', 'peptide_length')]

BINDING_ALLELE_COLUMNS = [
    'bind_A0101', 'bind_A0201', 'bind_A0301', 'bind_A1101', 'bind_A2402',
    'bind_B0702', 'bind_B0801', 'bind_B2705', 'bind_B3501', 'bind_B4402',
]

FEATURE_COLUMNS_30 = PHYSICO_COLUMNS + BINDING_ALLELE_COLUMNS

FEATURE_COLUMNS_31 = FEATURE_COLUMNS_30 + ['peptide_length']

ALL_POSITION_LABELS = ('p4', 'p5', 'p6', 'p7', 'p8')

PROPERTY_TABLES = {
    'hydrophobicity': (KD_HYDRO, 0.0),
    'aromaticity':    (AROMATIC, 0),
    'vdw_volume':     (VDW_VOL, 0.0),
    'charge':         (CHARGE, 0),
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
        if idx is not None:
            aa = peptide[idx]
            features[f'{pos_label}_hydrophobicity'] = KD_HYDRO.get(aa, 0.0)
            features[f'{pos_label}_aromaticity'] = AROMATIC.get(aa, 0)
            features[f'{pos_label}_vdw_volume'] = VDW_VOL.get(aa, 0.0)
            features[f'{pos_label}_charge'] = CHARGE.get(aa, 0)
        else:
            features[f'{pos_label}_hydrophobicity'] = 0.0
            features[f'{pos_label}_aromaticity'] = 0
            features[f'{pos_label}_vdw_volume'] = 0.0
            features[f'{pos_label}_charge'] = 0

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
