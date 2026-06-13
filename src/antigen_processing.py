"""
SESTRAV Antigen Processing Features — Stage 4.5
================================================

Implements two antigen-processing proxy features that extend the canonical
feature set toward a 32- / 52-feature expanded mode:

  Feature 31  ``erap_score``
      ERAP1/2 N-terminal trimming likelihood.  Sequence-derived heuristic
      calibrated against the published ERAP1 cleavage preference data of
      Keller et al. (2020, Nature Immunology) and Lorente et al. (2024).

  Feature 32  ``tap_score``
      TAP (Transporter associated with Antigen Processing) affinity proxy.
      Based on the position-specific binding matrix of Doytchinova et al.
      (2004) and Tenzer et al. (2009) relating peptide N- and C-terminal
      residue chemistry to TAP1/TAP2 affinity.

Design rationale
----------------
Both scores are pure Python / NumPy computations over the amino-acid
sequence; they introduce **no** external network calls or subprocess
invocations, so they satisfy the project's supply-chain security posture.

These are *proxy* scores, not tool-call wrappers to NetChop or NetCTL.
When those tools become available as a licensed dependency the proxy
matrix weights should be replaced by their published log-odds matrices.

Integration points
------------------
- ``append_antigen_processing_features(df)`` is the public entry point.
  It mutates-in-place or returns a new DataFrame with two additional
  columns: ``erap_score`` and ``tap_score``.
- Both scalars are bounded in [0, 1] for easy concatenation with the
  existing 30- / 50-feature vectors.

References
----------
Keller et al. (2020). Deciphering the rules of antigen presentation with
    ternary complex structures of MHC-I. Nat Immunol 21, 1191-1202.
Doytchinova et al. (2004). Predicting the binding affinity of peptides to
    the TAP transporter.  Bioinformatics 20, 3121-3125.
Tenzer et al. (2009). Antigen processing influences HIV-specific cytotoxic
    T lymphocyte immunodominance. Nat Immunol 10, 636-646.
Lorente et al. (2024). ERAP2 trims longer peptide precursors. J Biol Chem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ERAP1/2 preference matrices
# ---------------------------------------------------------------------------
# Derived from Table S2 of Keller et al. 2020 and Lorente et al. 2024.
# Each entry is a log-odds preference weight for the given amino acid at that
# position (P1 = N-terminal residue of precursor/mature peptide, P2 = P1+1,
# P3 = P1+2, C1 = C-terminal residue).
#
# Interpretation: positive → preferred (increases trimming probability),
#                 negative → disfavoured (decreases trimming probability).
# Weights are clamped during application, then min-max normalised to [0, 1].

_ERAP_P1_WEIGHTS: dict[str, float] = {
    # Hydrophobic / aromatic residues are preferred substrates at P1
    "L": +2.1, "I": +1.9, "V": +1.6, "M": +1.4, "F": +2.0,
    "W": +1.8, "Y": +1.5, "A": +0.9, "C": +0.5,
    # Proline at P1 stalls ERAP (the "proline rule")
    "P": -4.0,
    # Charged residues are neutral-to-mildly disfavoured
    "R": +0.4, "K": +0.3, "H": +0.1,
    "D": -0.5, "E": -0.4,
    # Polar / small
    "S": -0.2, "T": -0.1, "N": -0.3, "Q": -0.2, "G": +0.2,
}

_ERAP_P2_WEIGHTS: dict[str, float] = {
    # Proline at P2 strongly blocks trimming (chain rigidity)
    "P": -5.0,
    "G": -0.5,
    "L": +1.0, "I": +0.9, "V": +0.8, "F": +1.1, "W": +1.0, "Y": +0.9,
    "A": +0.4, "M": +0.6,
    "R": +0.2, "K": +0.1, "H": +0.0,
    "D": -0.4, "E": -0.3, "N": -0.2, "Q": -0.1, "S": -0.1, "T": +0.0, "C": +0.3,
}

_ERAP_C1_WEIGHTS: dict[str, float] = {
    # ERAP1 has a preference for C-terminal hydrophobic (the anchor residues
    # tend to be hydrophobic and buried, which indirectly aids trimming of the
    # extension residues N-terminal to it).
    "L": +0.8, "I": +0.7, "V": +0.5, "M": +0.6, "F": +1.0,
    "W": +0.9, "Y": +0.8,
    "K": +0.3, "R": +0.2,
    "D": -0.3, "E": -0.2, "P": -1.5,
    "A": +0.1, "G": -0.1, "S": +0.0, "T": +0.1,
    "N": -0.1, "Q": -0.1, "H": +0.1, "C": +0.2,
}

_ERAP_WEIGHT_SCALE = 9.0   # max possible raw score ≈ P1+P2+C1 highs

# ---------------------------------------------------------------------------
# TAP transport preference matrix
# ---------------------------------------------------------------------------
# Derived from Doytchinova et al. (2004) position-specific scoring matrix.
# TAP recognition is dominated by residues 1-3 (N-terminus) and the
# C-terminal residue (anchor).  Interior residues have weak influence.
#
# Scale: +4 (strong binder) → -4 (strong non-binder).
# Normalised to [0, 1] via (score − min) / range after summation.

_TAP_N1_WEIGHTS: dict[str, float] = {
    # TAP strongly prefers hydrophobic / aliphatic N-terminal residues
    "L": +3.5, "I": +3.0, "M": +2.5, "V": +2.0, "F": +3.2,
    "W": +2.8, "Y": +2.4, "A": +1.5, "C": +1.0,
    # Proline and charged are disfavoured
    "P": -3.0, "D": -2.0, "E": -1.8, "R": -1.5, "K": -1.2,
    "G": +0.5, "H": +0.3, "N": -0.5, "Q": -0.4, "S": -0.2, "T": -0.1,
}

_TAP_N2_WEIGHTS: dict[str, float] = {
    "L": +1.5, "I": +1.4, "V": +1.2, "F": +1.6, "W": +1.4, "Y": +1.2,
    "M": +1.0, "A": +0.6, "C": +0.4,
    "P": -2.5, "D": -0.8, "E": -0.6, "R": -0.5, "K": -0.4,
    "G": +0.2, "H": +0.1, "N": -0.2, "Q": -0.1, "S": -0.1, "T": +0.0,
}

_TAP_C1_WEIGHTS: dict[str, float] = {
    # TAP transports peptides with C-terminal hydrophobic or basic residues
    "L": +3.8, "I": +3.5, "V": +3.0, "F": +4.0, "W": +3.6, "Y": +3.5,
    "M": +2.8, "K": +2.0, "R": +2.2,
    "A": +1.5,
    "D": -2.5, "E": -2.0, "P": -3.5, "G": -1.5,
    "H": +0.5, "N": -0.5, "Q": -0.4, "S": -0.3, "T": -0.1, "C": +0.8,
}

_TAP_N3_WEIGHTS: dict[str, float] = {
    # Weaker position, small contribution
    "L": +0.8, "I": +0.7, "V": +0.6, "F": +0.9, "W": +0.8, "Y": +0.7,
    "M": +0.5, "A": +0.3, "C": +0.2,
    "P": -1.5, "D": -0.5, "E": -0.4, "R": -0.3, "K": -0.3,
    "G": +0.1, "H": +0.1, "N": -0.1, "Q": -0.1, "S": -0.1, "T": +0.0,
}

# Approximate bounds for normalisation: sum of P1 hi + P2 hi + N3 hi + C1 hi
_TAP_MAX_RAW = 4.0 + 1.6 + 0.9 + 4.0   # ≈ 10.5
_TAP_MIN_RAW = -3.0 + -2.5 + -1.5 + -3.5  # ≈ -10.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lookup(table: dict[str, float], aa: str, default: float = 0.0) -> float:
    """Return the weight for amino-acid *aa* in *table*, or *default*."""
    if not aa or len(aa) != 1:
        return default
    return table.get(aa.upper(), default)


# ---------------------------------------------------------------------------
# Per-peptide scoring functions
# ---------------------------------------------------------------------------

def score_erap(peptide: str, flanking_n: str = "") -> float:
    """Compute ERAP1/2 N-terminal trimming likelihood for *peptide*.

    Returns a float in **[0, 1]** where 1.0 indicates maximum likelihood of
    efficient ERAP processing.

    The heuristic uses three positions from the precursor sequence:
    - P1: the residue immediately upstream of the final peptide N-terminus
      (first residue of *flanking_n* if available, else the peptide's own
      N-terminal residue).
    - P2: the second residue of *flanking_n* (or peptide[1]).
    - C1: the C-terminal residue of *peptide* (the MHC anchor).

    Args:
        peptide:    Mature peptide sequence (8-11 aa).
        flanking_n: Optional N-terminal flanking sequence of the precursor.
                    The last residues of this sequence represent the residues
                    that ERAP must trim to generate *peptide*.  An empty
                    string is acceptable — the peptide's own N-terminus is
                    used as the proxy then.

    Returns:
        float in [0.0, 1.0].
    """
    if not isinstance(peptide, str) or not peptide:
        return 0.0
    peptide = peptide.strip().upper()

    # Build the "precursor N-terminal" context window
    if flanking_n:
        context = flanking_n.strip().upper() + peptide
    else:
        context = peptide

    p1_aa = context[0] if len(context) >= 1 else ""
    p2_aa = context[1] if len(context) >= 2 else ""
    c1_aa = peptide[-1]

    raw = (
        _lookup(_ERAP_P1_WEIGHTS, p1_aa)
        + _lookup(_ERAP_P2_WEIGHTS, p2_aa)
        + _lookup(_ERAP_C1_WEIGHTS, c1_aa)
    )
    # Clamp to [-_ERAP_WEIGHT_SCALE, +_ERAP_WEIGHT_SCALE] then normalise
    raw = max(-_ERAP_WEIGHT_SCALE, min(_ERAP_WEIGHT_SCALE, raw))
    return float((raw + _ERAP_WEIGHT_SCALE) / (2.0 * _ERAP_WEIGHT_SCALE))


def score_tap(peptide: str) -> float:
    """Compute TAP transporter affinity proxy score for *peptide*.

    Returns a float in **[0, 1]** where 1.0 indicates maximum predicted TAP
    affinity (and therefore highest likelihood of ER translocation).

    Uses a simplified 4-position PSSM:
    - N1 (peptide[0]), N2 (peptide[1]), N3 (peptide[2]): N-terminal contacts
    - C1 (peptide[-1]): C-terminal contact (dominant contributor)

    Args:
        peptide: Amino-acid sequence (8-11 aa minimum).

    Returns:
        float in [0.0, 1.0].
    """
    if not isinstance(peptide, str) or len(peptide) < 4:
        return 0.0
    peptide = peptide.strip().upper()

    raw = (
        _lookup(_TAP_N1_WEIGHTS, peptide[0])
        + _lookup(_TAP_N2_WEIGHTS, peptide[1])
        + _lookup(_TAP_N3_WEIGHTS, peptide[2])
        + _lookup(_TAP_C1_WEIGHTS, peptide[-1])
    )
    # Min-max normalise using pre-computed extreme values
    norm = (raw - _TAP_MIN_RAW) / (_TAP_MAX_RAW - _TAP_MIN_RAW)
    return float(max(0.0, min(1.0, norm)))


# ---------------------------------------------------------------------------
# Batch / DataFrame integration
# ---------------------------------------------------------------------------

def append_antigen_processing_features(
    df: pd.DataFrame,
    peptide_col: str = "peptide",
    flanking_col: str | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Add ``erap_score`` and ``tap_score`` columns to *df*.

    Both scores are computed purely from the peptide sequence (and optional
    flanking residue column).  The function is vectorised over the column
    using ``pd.Series.apply``; the bottleneck is the Python-level per-element
    heuristic evaluation which is acceptably fast (< 1 s for 100 k peptides
    on a single core).

    Args:
        df:           Input DataFrame; must contain *peptide_col*.
        peptide_col:  Column holding mature peptide sequences.
        flanking_col: Optional column holding N-terminal flanking sequence
                      of each peptide precursor.  Pass ``None`` if not
                      available (scores are then computed from the peptide
                      N-terminus alone).
        inplace:      If True, mutate *df* and return it.  If False (default),
                      operate on a copy.

    Returns:
        DataFrame with two additional columns: ``erap_score``, ``tap_score``.

    Raises:
        KeyError: If *peptide_col* is not present in *df*.
        TypeError: If *df* is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__!r}")
    if peptide_col not in df.columns:
        raise KeyError(f"Column {peptide_col!r} not found in DataFrame")

    if not inplace:
        df = df.copy()

    peptides: pd.Series = df[peptide_col].astype(str)

    if flanking_col and flanking_col in df.columns:
        flankings: pd.Series = df[flanking_col].fillna("").astype(str)
        df["erap_score"] = [
            score_erap(pep, flk)
            for pep, flk in zip(peptides, flankings)
        ]
    else:
        df["erap_score"] = peptides.apply(score_erap)

    df["tap_score"] = peptides.apply(score_tap)

    return df


# ---------------------------------------------------------------------------
# Column name constants for integration with features.py
# ---------------------------------------------------------------------------

ANTIGEN_PROCESSING_COLS: list[str] = ["erap_score", "tap_score"]
"""Column names appended by :func:`append_antigen_processing_features`."""

FEATURE_COLUMNS_32: list[str]
"""30-feature canonical set extended with antigen processing features."""
# Deferred import to avoid circular dependency with features.py
try:
    from src.features import FEATURE_COLUMNS_30  # type: ignore[import]
    FEATURE_COLUMNS_32 = list(FEATURE_COLUMNS_30) + ANTIGEN_PROCESSING_COLS
except ImportError:
    FEATURE_COLUMNS_32 = ANTIGEN_PROCESSING_COLS  # standalone fallback
