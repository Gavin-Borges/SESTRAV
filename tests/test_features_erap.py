# tests/test_features_erap.py
import pytest
from src.features import compute_erap_trimming_score

def test_erap_empty_or_invalid():
    # Empty or null sequence should return 0.0 without crashing
    assert compute_erap_trimming_score("") == 0.0
    assert compute_erap_trimming_score(None) == 0.0


def test_erap_basic_preferences():
    # Peptides with preferred N-terminal residues (hydrophobic or basic)
    # should score higher than neutral or disfavored ones.
    
    score_hydrophobic = compute_erap_trimming_score("LAAALLLL") # L at P1 (+2.0)
    score_basic = compute_erap_trimming_score("RAAALLLL")       # R at P1 (+1.5)
    score_proline = compute_erap_trimming_score("PAAALLLL")     # P at P1 (-2.0)
    score_neutral = compute_erap_trimming_score("SAAALLLL")     # S at P1 (neutral, +0)
    
    assert score_hydrophobic > score_neutral
    assert score_basic > score_neutral
    assert score_neutral > score_proline


def test_erap_proline_p2_penalty():
    # Proline at position P2 is a major trimming blocker and should lower the score.
    score_with_p2_proline = compute_erap_trimming_score("APAALLLL") # P at P2 (-3.0)
    score_with_p2_hydrophobic = compute_erap_trimming_score("ALAALLLL") # L at P2 (+1.0)
    score_neutral = compute_erap_trimming_score("AAAALLLL") # A at P2 (neutral, +0)
    
    assert score_with_p2_hydrophobic > score_neutral
    assert score_neutral > score_with_p2_proline


def test_erap_with_flanking_sequences():
    # Test when flanking sequences are provided
    
    # 1. Flanking sequence >= 3 residues: should score the last 3 flanking residues
    # flanking_seq = "GLS", analyzed = "GLS" -> G (neutral), L at P2 (+1.0), S (neutral) -> score = 6.0
    score_long_flanking = compute_erap_trimming_score("CLGGLLTMV", flanking_seq="GLS")
    # flanking_seq = "GPL", analyzed = "GPL" -> G (neutral), P at P2 (-3.0), L (+0.5) -> score = 2.5
    score_long_flanking_proline = compute_erap_trimming_score("CLGGLLTMV", flanking_seq="GPL")
    
    assert score_long_flanking > score_long_flanking_proline
    
    # 2. Short flanking sequence (< 3 residues): should prepend flanking and take start of peptide
    # flanking_seq = "G", peptide = "CLGGLLTMV" -> analyzed = "GCL"
    # G (neutral), C (neutral), L (+0.5) -> score = 5.5
    score_short_flanking = compute_erap_trimming_score("CLGGLLTMV", flanking_seq="G")
    assert score_short_flanking == 5.5


def test_erap_antigens_smoke():
    # Verify standard antigens are scored within bounds [0.0, 10.0]
    ebv_lmp2 = "CLGGLLTMV"
    hpv16_e6 = "TIHDIILECV"
    
    score_ebv = compute_erap_trimming_score(ebv_lmp2)
    score_hpv = compute_erap_trimming_score(hpv16_e6)
    
    assert 0.0 <= score_ebv <= 10.0
    assert 0.0 <= score_hpv <= 10.0
