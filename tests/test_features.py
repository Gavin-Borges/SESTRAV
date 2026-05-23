"""
Unit tests for the SESTRAV 22-feature extraction module.

Tests verify hand-calculated values using the length-relative TCR contact
position mapping (Chowell et al. 2015 / PredIG convention):
  p4 = index 3          (fixed, N-terminal)
  p5 = index 4          (fixed)
  p6 = index 5          (fixed)
  p7 = index length - 3 (C-terminal-relative)
  p8 = index length - 2 (C-terminal-relative)

Positions that overlap a prior position or reach the C-terminal anchor
residue (index >= length - 1) are zero-imputed.

Run from repo root:
    python -m pytest tests/test_features.py -v
    python -m tests.test_features          (standalone)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import compute_features, get_tcr_positions, FEATURE_COLUMNS


def test_clgglltmv_9mer():
    """CLGGLLTMV (9-mer, LMP2A, HLA-A*02:01) — canonical test.

    C(0) L(1) G(2) G(3) L(4) L(5) T(6) M(7) V(8)
    p4=G(3), p5=L(4), p6=L(5), p7=T(idx 6, L-3=6), p8=M(idx 7, L-2=7)
    p7 is valid: 6 > 5 and 6 < 8.  p8: 7 > 5 and 7 < 8 — valid.
    """
    f = compute_features("CLGGLLTMV", binding_score=0.5)

    assert f['peptide_length'] == 9
    assert f['binding_score'] == 0.5

    assert f['p4_hydrophobicity'] == -0.4   # G
    assert f['p5_hydrophobicity'] == 3.8    # L
    assert f['p6_hydrophobicity'] == 3.8    # L
    assert f['p7_hydrophobicity'] == -0.7   # T at idx 6
    assert f['p8_hydrophobicity'] == 1.9    # M at idx 7

    assert f['p4_aromaticity'] == 0  # G
    assert f['p5_aromaticity'] == 0  # L
    assert f['p6_aromaticity'] == 0  # L
    assert f['p7_aromaticity'] == 0  # T
    assert f['p8_aromaticity'] == 0  # M

    assert f['p4_vdw_volume'] == 48    # G
    assert f['p5_vdw_volume'] == 124   # L
    assert f['p6_vdw_volume'] == 124   # L
    assert f['p7_vdw_volume'] == 93    # T
    assert f['p8_vdw_volume'] == 124   # M

    assert f['p4_charge'] == 0  # G
    assert f['p5_charge'] == 0  # L
    assert f['p6_charge'] == 0  # L
    assert f['p7_charge'] == 0  # T
    assert f['p8_charge'] == 0  # M


def test_rakfkqll_8mer():
    """RAKFKQLL (8-mer, BZLF1, HLA-B*08:01) — p7 and p8 must be zero-filled.

    R(0) A(1) K(2) F(3) K(4) Q(5) L(6) L(7)
    p4=F(3), p5=K(4), p6=Q(5)
    p7=idx L-3=5, but 5 is NOT > 5 (overlaps p6) → zero-imputed
    p8=idx L-2=6, but 6 is NOT > 5 → zero-imputed
    """
    f = compute_features("RAKFKQLL", binding_score=0.0)

    assert f['peptide_length'] == 8

    assert f['p4_hydrophobicity'] == 2.8    # F
    assert f['p5_hydrophobicity'] == -3.9   # K
    assert f['p6_hydrophobicity'] == -3.5   # Q
    assert f['p7_hydrophobicity'] == 0.0    # zero-imputed
    assert f['p8_hydrophobicity'] == 0.0    # zero-imputed

    assert f['p4_aromaticity'] == 1   # F is aromatic
    assert f['p5_aromaticity'] == 0   # K
    assert f['p6_aromaticity'] == 0   # Q
    assert f['p7_aromaticity'] == 0
    assert f['p8_aromaticity'] == 0

    assert f['p5_charge'] == 1   # K = +1
    assert f['p6_charge'] == 0   # Q = 0


def test_tihdiilecv_10mer():
    """TIHDIILECV (10-mer, HPV16 E6, HLA-A*02:01) — all p4-p8 populated.

    T(0) I(1) H(2) D(3) I(4) I(5) L(6) E(7) C(8) V(9)
    p4=D(3), p5=I(4), p6=I(5)
    p7=E(idx L-3=7): 7 > 5 and 7 < 9 → valid
    p8=C(idx L-2=8): 8 > 5 and 8 < 9 → valid
    """
    f = compute_features("TIHDIILECV", binding_score=0.3)

    assert f['peptide_length'] == 10

    assert f['p4_hydrophobicity'] == -3.5   # D
    assert f['p5_hydrophobicity'] == 4.5    # I
    assert f['p6_hydrophobicity'] == 4.5    # I
    assert f['p7_hydrophobicity'] == -3.5   # E at idx 7 (not L at idx 6)
    assert f['p8_hydrophobicity'] == 2.5    # C at idx 8 (not E at idx 7)

    assert f['p4_charge'] == -1  # D = -1
    assert f['p7_charge'] == -1  # E = -1
    assert f['p5_charge'] == 0   # I = 0
    assert f['p8_charge'] == 0   # C = 0

    assert f['p7_aromaticity'] == 0  # E
    assert f['p8_aromaticity'] == 0  # C

    assert f['p7_vdw_volume'] == 109  # E
    assert f['p8_vdw_volume'] == 86   # C


def test_hpvgeadyfey_11mer():
    """HPVGEADYFEY (11-mer, EBNA1, HLA-B*35:01) — all p4-p8 populated.

    H(0) P(1) V(2) G(3) E(4) A(5) D(6) Y(7) F(8) E(9) Y(10)
    p4=G(3), p5=E(4), p6=A(5)
    p7=F(idx L-3=8): 8 > 5 and 8 < 10 → valid
    p8=E(idx L-2=9): 9 > 5 and 9 < 10 → valid
    """
    f = compute_features("HPVGEADYFEY", binding_score=0.7)

    assert f['peptide_length'] == 11

    assert f['p4_hydrophobicity'] == -0.4   # G
    assert f['p5_hydrophobicity'] == -3.5   # E
    assert f['p6_hydrophobicity'] == 1.8    # A
    assert f['p7_hydrophobicity'] == 2.8    # F at idx 8 (not D at idx 6)
    assert f['p8_hydrophobicity'] == -3.5   # E at idx 9 (not Y at idx 7)

    assert f['p4_aromaticity'] == 0   # G
    assert f['p5_aromaticity'] == 0   # E
    assert f['p6_aromaticity'] == 0   # A
    assert f['p7_aromaticity'] == 1   # F is aromatic
    assert f['p8_aromaticity'] == 0   # E

    assert f['p4_charge'] == 0    # G
    assert f['p5_charge'] == -1   # E = -1
    assert f['p6_charge'] == 0    # A
    assert f['p7_charge'] == 0    # F
    assert f['p8_charge'] == -1   # E = -1

    assert f['p7_vdw_volume'] == 135  # F
    assert f['p8_vdw_volume'] == 109  # E


def test_get_tcr_positions_length_relative():
    """Verify position indices are correct for each supported peptide length."""
    pos_8 = get_tcr_positions(8)
    assert pos_8 == [('p4', 3), ('p5', 4), ('p6', 5), ('p7', None), ('p8', None)]

    pos_9 = get_tcr_positions(9)
    assert pos_9 == [('p4', 3), ('p5', 4), ('p6', 5), ('p7', 6), ('p8', 7)]

    pos_10 = get_tcr_positions(10)
    assert pos_10 == [('p4', 3), ('p5', 4), ('p6', 5), ('p7', 7), ('p8', 8)]

    pos_11 = get_tcr_positions(11)
    assert pos_11 == [('p4', 3), ('p5', 4), ('p6', 5), ('p7', 8), ('p8', 9)]


def test_feature_count():
    """Every call must return exactly 42 features (22 canonical + 20 expanded)."""
    from src.features import EXPANDED_FEATURE_COLUMNS
    for pep in ["CLGGLLTMV", "RAKFKQLL", "TIHDIILECV", "HPVGEADYFEY"]:
        f = compute_features(pep)
        assert len(f) == 42, f"{pep}: expected 42 features, got {len(f)}"
        for col in EXPANDED_FEATURE_COLUMNS:
            assert col in f, f"{pep}: missing feature column '{col}'"


if __name__ == "__main__":
    test_clgglltmv_9mer()
    test_rakfkqll_8mer()
    test_tihdiilecv_10mer()
    test_hpvgeadyfey_11mer()
    test_get_tcr_positions_length_relative()
    test_feature_count()
    print("All feature extraction tests passed.")
