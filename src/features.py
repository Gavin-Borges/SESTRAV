"""
SESTRAV Feature Extraction Module - The Shared Hub

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

from typing import Optional

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Published amino acid property lookup tables
# ---------------------------------------------------------------------------

KD_HYDRO = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "E": -3.5,
    "Q": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

VDW_VOL = {
    "A": 67,
    "R": 148,
    "N": 96,
    "D": 91,
    "C": 86,
    "E": 109,
    "Q": 114,
    "G": 48,
    "H": 118,
    "I": 124,
    "L": 124,
    "K": 135,
    "M": 124,
    "F": 135,
    "P": 90,
    "S": 73,
    "T": 93,
    "W": 163,
    "Y": 141,
    "V": 105,
}

AROMATIC = {"F": 1, "W": 1, "Y": 1, "H": 1}

CHARGE = {"K": 1, "R": 1, "D": -1, "E": -1}

# Vihinen et al. 1994 Flexibility Scale
FLEXIBILITY = {
    "A": 0.984,
    "C": 0.906,
    "D": 1.068,
    "E": 1.094,
    "F": 0.915,
    "G": 1.031,
    "H": 0.950,
    "I": 0.927,
    "K": 1.102,
    "L": 0.935,
    "M": 0.952,
    "N": 1.048,
    "P": 1.049,
    "Q": 1.037,
    "R": 1.008,
    "S": 1.046,
    "T": 0.997,
    "V": 0.931,
    "W": 0.904,
    "Y": 0.929,
}

# Zimmerman et al. 1968 Bulkiness
BULKINESS = {
    "A": 11.5,
    "C": 13.46,
    "D": 11.68,
    "E": 13.57,
    "F": 19.8,
    "G": 3.4,
    "H": 13.69,
    "I": 21.4,
    "K": 15.71,
    "L": 21.4,
    "M": 16.25,
    "N": 12.82,
    "P": 17.43,
    "Q": 14.45,
    "R": 14.28,
    "S": 9.47,
    "T": 15.77,
    "V": 21.57,
    "W": 21.67,
    "Y": 18.03,
}

# Hopp & Woods 1981 Hydrophilicity
HYDROPHILICITY = {
    "A": -0.5,
    "C": -1.0,
    "D": 3.0,
    "E": 3.0,
    "F": -2.5,
    "G": 0.0,
    "H": -0.5,
    "I": -1.8,
    "K": 3.0,
    "L": -1.8,
    "M": -1.3,
    "N": 0.2,
    "P": 0.0,
    "Q": 0.2,
    "R": 3.0,
    "S": 0.3,
    "T": -0.4,
    "V": -1.5,
    "W": -3.4,
    "Y": -2.3,
}

# TCR Contact Upward-Facing Probabilities (proxy metric based on peptide length and position)
# Estimates the likelihood that a residue is prominently exposed to the TCR (derived from structural alignments)
UPWARD_PROBABILITY = {
    8: {"p4": 0.8, "p5": 0.9, "p6": 0.8, "p7": 0.0, "p8": 0.0},
    9: {"p4": 0.6, "p5": 0.9, "p6": 0.9, "p7": 0.7, "p8": 0.6},
    10: {"p4": 0.5, "p5": 0.8, "p6": 0.9, "p7": 0.8, "p8": 0.5},
    11: {"p4": 0.4, "p5": 0.7, "p6": 0.8, "p7": 0.8, "p8": 0.4},
}

FEATURE_COLUMNS = [
    "p4_hydrophobicity",
    "p5_hydrophobicity",
    "p6_hydrophobicity",
    "p7_hydrophobicity",
    "p8_hydrophobicity",
    "p4_aromaticity",
    "p5_aromaticity",
    "p6_aromaticity",
    "p7_aromaticity",
    "p8_aromaticity",
    "p4_vdw_volume",
    "p5_vdw_volume",
    "p6_vdw_volume",
    "p7_vdw_volume",
    "p8_vdw_volume",
    "p4_charge",
    "p5_charge",
    "p6_charge",
    "p7_charge",
    "p8_charge",
    "binding_score",
    "peptide_length",
]

EXPANDED_FEATURE_COLUMNS = FEATURE_COLUMNS[:-2] + [
    "p4_flexibility",
    "p5_flexibility",
    "p6_flexibility",
    "p7_flexibility",
    "p8_flexibility",
    "p4_bulkiness",
    "p5_bulkiness",
    "p6_bulkiness",
    "p7_bulkiness",
    "p8_bulkiness",
    "p4_hydrophilicity",
    "p5_hydrophilicity",
    "p6_hydrophilicity",
    "p7_hydrophilicity",
    "p8_hydrophilicity",
    "p4_upward_prob",
    "p5_upward_prob",
    "p6_upward_prob",
    "p7_upward_prob",
    "p8_upward_prob",
    "binding_score",
    "peptide_length",
]

TRAIN_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c != "binding_score"]

PHYSICO_COLUMNS = [c for c in FEATURE_COLUMNS if c not in ("binding_score", "peptide_length")]

EXPANDED_PHYSICO_COLUMNS = [
    c for c in EXPANDED_FEATURE_COLUMNS if c not in ("binding_score", "peptide_length")
]

BINDING_ALLELE_COLUMNS = [
    "bind_A0101",
    "bind_A0201",
    "bind_A0301",
    "bind_A1101",
    "bind_A2402",
    "bind_B0702",
    "bind_B0801",
    "bind_B2705",
    "bind_B3501",
    "bind_B4402",
]

FEATURE_COLUMNS_30 = PHYSICO_COLUMNS + BINDING_ALLELE_COLUMNS
FEATURE_COLUMNS_31 = FEATURE_COLUMNS_30 + ["peptide_length"]
# 33-feature: canonical 31 + two antigen-processing columns.
# WARNING: netchop_score / tap_score are MOCK, sequence-derived scores from
# src/external_predictors.py - NOT NetChop 3.1 or TAPreg output, and NOT
# reproducible (the generators use hash(), which CPython salts per process,
# and the seed behind the shipped cache was never recorded). They are also not
# "orthogonal": the generator assumes hydrophobic/basic C-terminal cleavage
# preference by construction, so their feature importance cannot be evidence
# for that mechanism. See docs/claims_register.md D18.
FEATURE_COLUMNS_33 = FEATURE_COLUMNS_31 + ["netchop_score", "tap_score"]
# 35-feature: extended 33 + human-proteome self-similarity (tolerance signal)
FEATURE_COLUMNS_35 = FEATURE_COLUMNS_33 + [
    "self_similarity_max_identity",
    "self_similarity_exact_match",
]

FEATURE_COLUMNS_50 = EXPANDED_PHYSICO_COLUMNS + BINDING_ALLELE_COLUMNS

# ---------------------------------------------------------------------------
# Per-allele TCR contact frequency priors at positions p4-p8 (mode 51)
# ---------------------------------------------------------------------------
# Each list is [p4_weight, p5_weight, p6_weight, p7_weight, p8_weight].
# Values are literature-informed priors (Chowell et al. 2015 PNAS; Baker et al.
# 2012 J Immunol) and will be replaced by compute_contact_freqs.py output once
# the ATLAS / STCRDab PDB corpus has been downloaded and parsed.
ALLELE_CONTACT_WEIGHTS: dict[str, list[float]] = {
    "HLA-A*01:01": [0.60, 0.48, 0.80, 0.48, 0.32],
    "HLA-A*02:01": [0.65, 0.45, 0.82, 0.55, 0.35],
    "HLA-A*03:01": [0.58, 0.52, 0.76, 0.50, 0.30],
    "HLA-A*11:01": [0.58, 0.52, 0.76, 0.50, 0.30],  # A*03 supertype proxy
    "HLA-A*24:02": [0.62, 0.47, 0.79, 0.49, 0.33],
    "HLA-B*07:02": [0.58, 0.60, 0.75, 0.50, 0.40],
    "HLA-B*08:01": [0.62, 0.50, 0.78, 0.52, 0.38],
    "HLA-B*27:05": [0.55, 0.65, 0.70, 0.45, 0.35],
    "HLA-B*35:01": [0.60, 0.55, 0.72, 0.48, 0.36],
    "HLA-B*44:02": [0.61, 0.53, 0.76, 0.51, 0.37],
}

# Population-average fallback (allele absent or not in ALLELE_CONTACT_WEIGHTS)
POPULATION_AVG_CONTACT_WEIGHTS: list[float] = [0.60, 0.53, 0.77, 0.50, 0.36]

CONTACT_WEIGHT_COLUMNS: list[str] = [
    "tcr_contact_weight_p4",
    "tcr_contact_weight_p5",
    "tcr_contact_weight_p6",
    "tcr_contact_weight_p7",
    "tcr_contact_weight_p8",
]

FEATURE_COLUMNS_51 = FEATURE_COLUMNS_50 + CONTACT_WEIGHT_COLUMNS

# ---------------------------------------------------------------------------
# HLA Pocket Pseudo-sequence feature columns (allele-aware extension)
# ---------------------------------------------------------------------------
# NetMHCpan 4.1 convention: 34 variable MHC pocket residues, each encoded
# with 4 physicochemical properties → 136 features total.

HLA_PSEUDO_LEN = 34  # canonical pocket length per NetMHCpan

HLA_PSEUDO_COLS = [
    f"hla_p{i + 1}_{prop}"
    for i in range(HLA_PSEUDO_LEN)
    for prop in ("hydrophobicity", "aromaticity", "vdw_volume", "charge")
]
# 34 positions × 4 properties = 136 allele features

# Allele-aware 166-feature set: 20 physico + 10 binding + 136 HLA pocket
FEATURE_COLUMNS_ALLELE = FEATURE_COLUMNS_30 + HLA_PSEUDO_COLS

# ---------------------------------------------------------------------------
# Sample weight helpers for bias correction
# ---------------------------------------------------------------------------


def compute_sample_weights(
    df, virus_col="virus", length_col=None, virus_weight=0.5, length_weight=0.5
):
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
                mask = virus_vals == virus
                actual_freq = mask.sum() / n
                correction = target_freq / max(actual_freq, 1e-6)
                weights[mask] *= (1.0 - virus_weight) + virus_weight * correction

    # Length correction
    pep_col = length_col if length_col and length_col in df.columns else "peptide"
    if pep_col in df.columns:
        lengths = df[pep_col].str.len().values
        is_9mer = lengths == 9
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


ALL_POSITION_LABELS = ("p4", "p5", "p6", "p7", "p8")

PROPERTY_TABLES = {
    "hydrophobicity": (KD_HYDRO, 0.0),
    "aromaticity": (AROMATIC, 0),
    "vdw_volume": (VDW_VOL, 0.0),
    "charge": (CHARGE, 0),
    "flexibility": (FLEXIBILITY, 0.0),
    "bulkiness": (BULKINESS, 0.0),
    "hydrophilicity": (HYDROPHILICITY, 0.0),
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
    fixed = [
        ("p4", 3 if length > 3 else None),
        ("p5", 4 if length > 4 else None),
        ("p6", 5 if length > 5 else None),
    ]
    p7_idx = length - 3
    p8_idx = length - 2

    p7_valid = p7_idx > 5 and p7_idx < length - 1
    p8_valid = p7_valid and p8_idx > p7_idx and p8_idx < length - 1

    return fixed + [
        ("p7", p7_idx if p7_valid else None),
        ("p8", p8_idx if p8_valid else None),
    ]


def compute_features(peptide, binding_score=0.0):
    """Compute 22 physicochemical features for a single peptide.

    Returns a dict with 22 feature values keyed by FEATURE_COLUMNS names.
    Positions outside the valid TCR core are zero-imputed.
    """
    features = {}
    length = len(peptide)
    features["peptide_length"] = length
    features["binding_score"] = binding_score

    positions = get_tcr_positions(length)

    for pos_label, idx in positions:
        # structural proxy probability based on length and pos_label
        upward_prob = UPWARD_PROBABILITY.get(length, UPWARD_PROBABILITY[9]).get(pos_label, 0.0)

        if idx is not None:
            aa = peptide[idx]
            features[f"{pos_label}_hydrophobicity"] = KD_HYDRO.get(aa, 0.0)
            features[f"{pos_label}_aromaticity"] = AROMATIC.get(aa, 0)
            features[f"{pos_label}_vdw_volume"] = VDW_VOL.get(aa, 0.0)
            features[f"{pos_label}_charge"] = CHARGE.get(aa, 0)
            features[f"{pos_label}_flexibility"] = FLEXIBILITY.get(aa, 0.0)
            features[f"{pos_label}_bulkiness"] = BULKINESS.get(aa, 0.0)
            features[f"{pos_label}_hydrophilicity"] = HYDROPHILICITY.get(aa, 0.0)
            features[f"{pos_label}_upward_prob"] = upward_prob
        else:
            features[f"{pos_label}_hydrophobicity"] = 0.0
            features[f"{pos_label}_aromaticity"] = 0
            features[f"{pos_label}_vdw_volume"] = 0.0
            features[f"{pos_label}_charge"] = 0
            features[f"{pos_label}_flexibility"] = 0.0
            features[f"{pos_label}_bulkiness"] = 0.0
            features[f"{pos_label}_hydrophilicity"] = 0.0
            features[f"{pos_label}_upward_prob"] = upward_prob

    return features


def compute_features_for_dataset(
    df: pd.DataFrame,
    peptide_col: str = "peptide",
    binding_col: str = "presentation_score",
) -> pd.DataFrame:
    """Vectorized batch feature extraction for an entire DataFrame.

    Replaces the previous ``iterrows()`` loop with column-wise numpy/pandas
    operations that are ~20-100x faster on datasets ≥ 10 k peptides.

    The output schema is identical to the old implementation:
    the input DataFrame is returned with all FEATURE_COLUMNS appended
    (plus the full expanded-mode columns when present).  Positions that
    are invalid for a given length are zero-imputed exactly as before.

    Args:
        df:          Source DataFrame; must contain *peptide_col*.
        peptide_col: Name of the column holding peptide sequences.
        binding_col: Name of the MHCflurry presentation_score column;
                     missing or NaN values are filled with 0.0.

    Returns:
        A new DataFrame with the original columns prepended to the
        computed feature columns (index reset).
    """
    df = df.reset_index(drop=True)
    peptides: pd.Series = df[peptide_col].astype(str).str.strip().str.upper()
    lengths: pd.Series = peptides.str.len()

    # ------------------------------------------------------------------
    # binding_score - fill NaN with 0.0
    # ------------------------------------------------------------------
    if binding_col in df.columns:
        binding = df[binding_col].fillna(0.0).astype(float)
    else:
        binding = pd.Series(0.0, index=df.index)

    # ------------------------------------------------------------------
    # Property tables keyed by feature suffix
    # ------------------------------------------------------------------
    prop_tables: dict[str, tuple[dict, object]] = {
        "hydrophobicity": (KD_HYDRO, 0.0),
        "aromaticity": (AROMATIC, 0),
        "vdw_volume": (VDW_VOL, 0.0),
        "charge": (CHARGE, 0),
        "flexibility": (FLEXIBILITY, 0.0),
        "bulkiness": (BULKINESS, 0.0),
        "hydrophilicity": (HYDROPHILICITY, 0.0),
    }

    # ------------------------------------------------------------------
    # TCR position indices vary by length - compute once per-row as arrays
    # ------------------------------------------------------------------
    # Each of the 5 position labels maps to a nullable integer index.
    pos_labels = ("p4", "p5", "p6", "p7", "p8")

    # Fixed N-terminal indices
    idx_p4 = np.where(lengths > 3, 3, -1).astype(object)
    idx_p5 = np.where(lengths > 4, 4, -1).astype(object)
    idx_p6 = np.where(lengths > 5, 5, -1).astype(object)

    # C-terminal relative indices
    p7_raw = (lengths - 3).values
    p8_raw = (lengths - 2).values
    len_arr = lengths.values

    p7_valid = (p7_raw > 5) & (p7_raw < len_arr - 1)
    p8_valid = p7_valid & (p8_raw > p7_raw) & (p8_raw < len_arr - 1)

    idx_p7 = np.where(p7_valid, p7_raw, -1).astype(object)
    idx_p8 = np.where(p8_valid, p8_raw, -1).astype(object)

    # -1 sentinel → None for clarity in column building
    pos_idx_arrays = {
        "p4": idx_p4,
        "p5": idx_p5,
        "p6": idx_p6,
        "p7": idx_p7,
        "p8": idx_p8,
    }

    # ------------------------------------------------------------------
    # Vectorized amino-acid extraction per position
    # ------------------------------------------------------------------
    # Build a (n_samples × 5) array of amino-acid characters; use empty
    # string for invalid positions (maps to default via fillna).
    n = len(df)
    aa_matrix: dict[str, np.ndarray] = {}
    for label in pos_labels:
        idx_arr = pos_idx_arrays[label]
        chars = np.empty(n, dtype=object)
        for i in range(n):
            raw_idx = idx_arr[i]
            if raw_idx == -1 or raw_idx is None:
                chars[i] = None
            else:
                pep = peptides.iat[i]
                chars[i] = pep[raw_idx] if raw_idx < len(pep) else None
        aa_matrix[label] = chars

    # ------------------------------------------------------------------
    # Apply property lookups
    # ------------------------------------------------------------------
    feat_cols: dict[str, np.ndarray] = {}
    for label in pos_labels:
        chars_series = pd.Series(aa_matrix[label], index=df.index)
        for prop, (table, default) in prop_tables.items():
            col_name = f"{label}_{prop}"
            mapped = chars_series.map(table).fillna(default)
            feat_cols[col_name] = mapped.values

    # ------------------------------------------------------------------
    # Upward probability (structural proxy) - lookup by (length, label)
    # ------------------------------------------------------------------
    for label in pos_labels:
        up_vals = np.array(
            [UPWARD_PROBABILITY.get(l, UPWARD_PROBABILITY[9]).get(label, 0.0) for l in len_arr]
        )
        feat_cols[f"{label}_upward_prob"] = up_vals

    # ------------------------------------------------------------------
    # Scalar features
    # ------------------------------------------------------------------
    feat_cols["binding_score"] = binding.values
    feat_cols["peptide_length"] = len_arr

    features_df = pd.DataFrame(feat_cols, index=df.index)
    return pd.concat([df, features_df], axis=1)


def compute_erap_trimming_score(peptide: str, flanking_seq: Optional[str] = None) -> float:
    """
    Compute an ERAP1/2 N-terminal trimming likelihood score.

    Parameters
    ----------
    peptide : str
        The mature peptide sequence.
    flanking_seq : str, optional
        Flanking residues upstream of the peptide's N-terminus.

    Returns
    -------
    float
        Trimming likelihood score bounded in [0.0, 10.0].
    """
    if not peptide:
        return 0.0

    # Standardize inputs
    peptide = str(peptide).strip().upper()
    if flanking_seq is not None:
        flanking_seq = str(flanking_seq).strip().upper()

    # Get N-terminal 3 residues of the precursor/mature sequence to analyze
    if flanking_seq:
        if len(flanking_seq) >= 3:
            seq = flanking_seq[-3:]
        else:
            needed = 3 - len(flanking_seq)
            seq = flanking_seq + peptide[:needed]
    else:
        seq = peptide[:3]

    # Pad to length 3 if necessary
    if len(seq) < 3:
        seq = (seq + "XXX")[:3]

    score = 5.0  # Baseline offset

    # Position 1: ERAP1/2 favors hydrophobic or basic residues
    aa1 = seq[0]
    if aa1 in {"L", "F", "I", "V", "M", "W", "Y"}:
        score += 2.0
    elif aa1 in {"R", "K"}:
        score += 1.5
    elif aa1 == "P":
        score -= 2.0

    # Position 2: ERAP1/2 strongly penalizes proline at P2 (stops trimming)
    aa2 = seq[1]
    if aa2 == "P":
        score -= 3.0
    elif aa2 in {"L", "F", "I", "V", "M", "W", "Y"}:
        score += 1.0

    # Position 3: ERAP1 favors hydrophobic residue at P3
    aa3 = seq[2]
    if aa3 in {"L", "F", "I", "V", "M", "W", "Y"}:
        score += 0.5

    return max(0.0, min(10.0, score))


# ---------------------------------------------------------------------------
# Topological Feature Engineering: ESM-2 CLS Token and Graph Descriptors
# ---------------------------------------------------------------------------

_ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
_esm_model = None
_esm_tokenizer = None

FEATURE_COLUMNS_30_ESM = FEATURE_COLUMNS_30 + [f"esm_{i}" for i in range(320)]
FEATURE_COLUMNS_30_GRAPH = FEATURE_COLUMNS_30 + [f"graph_wl_{i}" for i in range(32)]


def get_esm_cls_token(peptide: str) -> np.ndarray:
    """
    Extract CLS token from ESM-2 t6 model for the peptide.
    Uses deterministic settings. Falls back to a deterministic mock vector if model fails to load.
    """
    import numpy as np

    try:
        global _esm_model, _esm_tokenizer
        if _esm_model is None or _esm_tokenizer is None:
            import torch

            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            torch.use_deterministic_algorithms(True, warn_only=True)
            from transformers import AutoTokenizer, EsmModel

            _esm_tokenizer = AutoTokenizer.from_pretrained(
                _ESM_MODEL_NAME, revision="8c576d2aba1b27317e9321c8491d72f00d1b110a"
            )
            _esm_model = EsmModel.from_pretrained(
                _ESM_MODEL_NAME, revision="8c576d2aba1b27317e9321c8491d72f00d1b110a"
            )
            _esm_model.eval()

        import torch

        seq = str(peptide).strip().upper()
        inputs = _esm_tokenizer(seq, return_tensors="pt")
        with torch.no_grad():
            outputs = _esm_model(**inputs)
            cls_repr = outputs.last_hidden_state[0, 0].numpy()
        return cls_repr
    except Exception as e:
        print(
            f"[ESM Feature] WARNING: Failed to compute ESM-2 CLS token: {e}. Falling back to deterministic mock vector."
        )
        import hashlib

        h = hashlib.sha256(peptide.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:4], "big"))
        return rng.normal(0, 1, 320)


def compute_weisfeiler_lehman_features(G, n_iter=2) -> np.ndarray:
    """
    Compute Weisfeiler-Lehman (WL) graph kernel features for Weisfeiler-Lehman test.
    """
    current_features = {node: str(G.nodes[node].get("x", "0")) for node in G.nodes()}
    all_colors = []

    for _ in range(n_iter):
        new_features = {}
        for node in G.nodes():
            neigh_feats = sorted([current_features[neigh] for neigh in G.neighbors(node)])
            feat_str = f"{current_features[node]}-" + "-".join(neigh_feats)
            new_features[node] = str(hash(feat_str))
            all_colors.append(new_features[node])
        current_features = new_features

    wl_vector = np.zeros(32)
    for color in all_colors:
        import hashlib

        idx = int(hashlib.md5(color.encode("utf-8"), usedforsecurity=False).hexdigest(), 16) % 32
        wl_vector[idx] += 1.0

    return wl_vector


def get_cb_cb_edges(length: int) -> list:
    """
    Returns standard CB-CB contacts (edges as 0-indexed residue pairs) for peptide of given length
    based on standard MHC-I groove structural templates (distances < 6A).
    """
    edges = []
    # Adjacent residues always contact (distance ~ 3.8 A)
    for i in range(length - 1):
        edges.append((i, i + 1))
    # Residues separated by 1 position (i to i+2) contact in extended beta-like sheet conformations (distance ~ 5.5 A)
    for i in range(length - 2):
        edges.append((i, i + 2))

    # Standard bulge contacts for 9-11mers in MHC grooved conformations:
    if length == 9:
        edges.append((3, 5))
        edges.append((4, 6))
    elif length == 10:
        edges.append((3, 6))
        edges.append((4, 7))
    elif length == 11:
        edges.append((3, 7))
        edges.append((4, 8))
        edges.append((5, 9))

    unique_edges = list(set((min(u, v), max(u, v)) for u, v in edges))
    return unique_edges


def compute_wl_features(peptide: str, edges: list, num_iterations: int = 2) -> np.ndarray:
    """
    Build NetworkX graph and compute a fixed-size WL subtree feature vector mapping to size 32.
    """
    import networkx as nx
    import numpy as np

    G = nx.Graph()
    length = len(peptide)
    G.add_nodes_from(range(length))
    G.add_edges_from(edges)

    node_features = {}
    for i in range(length):
        aa = peptide[i]
        hydro = KD_HYDRO.get(aa, 0.0)
        chg = CHARGE.get(aa, 0)
        h_cat = "pos" if hydro > 1.0 else ("neg" if hydro < -1.0 else "neu")
        node_features[i] = f"{h_cat}_{chg}"

    nx.set_node_attributes(G, node_features, name="init_feat")

    current_features = node_features.copy()
    all_colors = []
    for i in range(length):
        all_colors.append(current_features[i])

    for _ in range(num_iterations):
        new_features = {}
        for node in G.nodes():
            neigh_feats = sorted([current_features[neigh] for neigh in G.neighbors(node)])
            feat_str = f"{current_features[node]}-" + "-".join(neigh_feats)
            new_features[node] = str(hash(feat_str))
            all_colors.append(new_features[node])
        current_features = new_features

    wl_vector = np.zeros(32)
    for color in all_colors:
        import hashlib

        idx = int(hashlib.md5(color.encode("utf-8"), usedforsecurity=False).hexdigest(), 16) % 32
        wl_vector[idx] += 1.0

    return wl_vector


# ---------------------------------------------------------------------------
# Antigen processing cache loader (feature_mode=33)
# ---------------------------------------------------------------------------


def load_self_similarity_cache(cache_path: str, df: pd.DataFrame) -> pd.DataFrame:
    """Join precomputed human-proteome self-similarity scores onto a peptide DataFrame.

    Missing peptides receive 0.0 / False (conservative: no known self-match).
    The cache CSV must have columns: peptide, self_similarity_max_identity,
    self_similarity_exact_match.

    Args:
        cache_path: Path to self_similarity_cache.csv.
        df:         DataFrame containing a 'peptide' column.

    Returns:
        df with 'self_similarity_max_identity' (float) and
        'self_similarity_exact_match' (float 0/1) columns appended.
    """
    cache = pd.read_csv(
        cache_path,
        usecols=["peptide", "self_similarity_max_identity", "self_similarity_exact_match"],
    )
    cache = cache.drop_duplicates(subset="peptide").set_index("peptide")
    result = df.copy()
    result["self_similarity_max_identity"] = (
        result["peptide"].map(cache["self_similarity_max_identity"]).fillna(0.0).astype(float)
    )
    # Store as float (0.0 / 1.0) so the feature matrix stays homogeneous
    result["self_similarity_exact_match"] = (
        result["peptide"].map(cache["self_similarity_exact_match"]).fillna(0.0).astype(float)
    )
    missing = int(
        (result["self_similarity_max_identity"] == 0.0).sum()
        - (cache["self_similarity_max_identity"] == 0.0)
        .reindex(result["peptide"])
        .fillna(True)
        .sum()
    )
    if missing > 0:
        print(
            f"[features] {missing} peptides not found in self-similarity cache - defaulting to 0.0"
        )
    return result


ANTIGEN_PROCESSING_COLUMNS: tuple[str, str] = ("netchop_score", "tap_score")


def _read_antigen_processing_cache(cache_path: str) -> pd.DataFrame:
    cache = pd.read_csv(cache_path, usecols=["peptide", *ANTIGEN_PROCESSING_COLUMNS])
    return cache.drop_duplicates(subset="peptide").set_index("peptide")


def antigen_processing_cache_medians(cache_path: str) -> dict[str, float]:
    """Whole-cache medians: the imputation values used outside cross-validation.

    Used for the final full-pool refit, where the whole training pool IS the
    model's training set and a pool-wide statistic is not leakage - only a
    per-CV-fold statistic computed over all rows (including the held-out
    fold) would be.
    """
    cache = _read_antigen_processing_cache(cache_path)
    return {col: float(cache[col].median()) for col in ANTIGEN_PROCESSING_COLUMNS}


def load_antigen_processing_cache(
    cache_path: str, df: pd.DataFrame, *, impute: bool = True
) -> pd.DataFrame:
    """Join precomputed antigen-processing scores onto a peptide DataFrame.

    WARNING: despite the column names, the shipped
    data/antigen_processing_cache.csv holds MOCK sequence-derived scores, not
    NetChop 3.1 / TAPreg output, and they cannot be reproduced from the
    generators (docs/claims_register.md D18). Treat feature_mode 33/35 results
    as a structural ablation, not validated antigen-processing prediction.

    The cache CSV must have columns: peptide, netchop_score, tap_score.

    Args:
        cache_path: Path to the antigen processing cache CSV.
        df:         DataFrame containing a 'peptide' column.
        impute:     When True (default - the behavior every pre-Phase-0 caller
                    relies on), missing peptides receive median imputation from
                    the whole cache distribution. When False the scores are
                    left as NaN, so a caller doing cross-validation can fit the
                    median inside each fold on training rows only; a
                    whole-cache median otherwise leaks the held-out fold's
                    peptides into the training features (docs/claims_register.md D15).

    Returns:
        df with 'netchop_score' and 'tap_score' columns appended.
    """
    cache = _read_antigen_processing_cache(cache_path)
    result = df.copy()
    for col in ANTIGEN_PROCESSING_COLUMNS:
        result[col] = result["peptide"].map(cache[col])
    missing = int(result["netchop_score"].isna().sum())
    if missing and impute:
        medians = {col: float(cache[col].median()) for col in ANTIGEN_PROCESSING_COLUMNS}
        result = result.fillna(medians)
        print(
            f"[features] Imputed {missing} missing antigen processing scores with cache medians "
            f"(netchop={medians['netchop_score']:.4f}, tap={medians['tap_score']:.4f})"
        )
    elif missing:
        print(
            f"[features] {missing} antigen processing scores left as NaN "
            "for in-fold median imputation"
        )
    return result
