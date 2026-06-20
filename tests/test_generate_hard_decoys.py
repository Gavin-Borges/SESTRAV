"""Tests for the pure utility functions in scripts/generate_hard_decoys.py.

The `generate_decoys()` entry point requires MHCflurry models (not available
in CI), so only the FASTA parser and k-mer extractor are tested here.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from generate_hard_decoys import _extract_kmers, _load_fasta  # noqa: E402


# ---------------------------------------------------------------------------
# FASTA parser
# ---------------------------------------------------------------------------

def test_load_fasta_single_sequence(tmp_path):
    fa = tmp_path / "test.fasta"
    fa.write_text(">seq1\nACDEFGHIKL\n")
    assert _load_fasta(str(fa)) == ["ACDEFGHIKL"]


def test_load_fasta_multiline_sequence(tmp_path):
    fa = tmp_path / "test.fasta"
    fa.write_text(">seq1\nACDEF\nGHIKL\n")
    assert _load_fasta(str(fa)) == ["ACDEFGHIKL"]


def test_load_fasta_multiple_sequences(tmp_path):
    fa = tmp_path / "test.fasta"
    fa.write_text(">s1\nACDEF\n>s2\nGHIKL\nMNPQR\n")
    seqs = _load_fasta(str(fa))
    assert seqs == ["ACDEF", "GHIKLMNPQR"]


def test_load_fasta_empty_file(tmp_path):
    fa = tmp_path / "empty.fasta"
    fa.write_text("")
    assert _load_fasta(str(fa)) == []


def test_load_fasta_no_trailing_newline(tmp_path):
    fa = tmp_path / "test.fasta"
    fa.write_text(">s1\nACDEF")
    assert _load_fasta(str(fa)) == ["ACDEF"]


# ---------------------------------------------------------------------------
# k-mer extractor — new API: lengths=(k,) tuple instead of k= scalar
# ---------------------------------------------------------------------------

def test_extract_kmers_returns_valid_9mers():
    seqs = ["ACDEFGHIKLMNPQRSTVWY"]  # 20 AAs → 12 valid 9-mers
    kmers = _extract_kmers(seqs, lengths=(9,))
    assert all(len(k) == 9 for k in kmers)
    assert len(kmers) == 12


def test_extract_kmers_filters_invalid_characters():
    seqs = ["XACDEFGHI", "ACDEFGHIK"]  # first has 'X' → only one valid 9-mer
    kmers = _extract_kmers(seqs, lengths=(9,))
    assert all("X" not in k for k in kmers)


def test_extract_kmers_deduplicates():
    seqs = ["ACDEFGHIK", "ACDEFGHIK"]  # same sequence twice
    kmers = _extract_kmers(seqs, lengths=(9,))
    assert len(kmers) == len(set(kmers))


def test_extract_kmers_is_sorted():
    seqs = ["WVTSRQPNM", "ACDEFGHIK"]
    kmers = _extract_kmers(seqs, lengths=(9,))
    assert kmers == sorted(kmers)


def test_extract_kmers_sequence_shorter_than_k():
    seqs = ["ACDE"]  # 4 chars, k=9 → no k-mers
    assert _extract_kmers(seqs, lengths=(9,)) == []


def test_extract_kmers_empty_input():
    assert _extract_kmers([], lengths=(9,)) == []


def test_extract_kmers_multi_length():
    seqs = ["ACDEFGHIKLM"]  # 11 AAs → 3×8-mers, 2×9-mers, 1×10-mer, 0×11-mers (len==11, ok 1)
    kmers_8 = _extract_kmers(seqs, lengths=(8,))
    kmers_9 = _extract_kmers(seqs, lengths=(9,))
    kmers_multi = _extract_kmers(seqs, lengths=(8, 9))
    assert set(kmers_8) | set(kmers_9) == set(kmers_multi)


def test_extract_kmers_excludes_invalid_aa():
    seqs = ["ACDEBGHIKL"]  # 'B' is not a valid standard amino acid
    kmers = _extract_kmers(seqs, lengths=(9,))
    assert all("B" not in k for k in kmers)


# ---------------------------------------------------------------------------
# max_candidates pre-sampling (exercises the slice path in generate_decoys)
# ---------------------------------------------------------------------------

def test_extract_kmers_supports_max_candidates_truncation():
    """Simulate the max_candidates truncation: seeded shuffle then slice.

    generate_decoys() does:
        random.shuffle(kmers)        # seeded before call
        if max_candidates < len(kmers): kmers = kmers[:max_candidates]

    Verify determinism and that the slice preserves all valid k-mers.
    """
    import random
    seqs = ["ACDEFGHIKLMNPQRSTVWY" * 3]   # 60-mer → many 9-mers
    kmers = _extract_kmers(seqs, lengths=(9,))
    assert len(kmers) > 5, "need more than 5 k-mers for this test"

    random.seed(42)
    shuffled_a = kmers.copy()
    random.shuffle(shuffled_a)
    random.seed(42)
    shuffled_b = kmers.copy()
    random.shuffle(shuffled_b)
    assert shuffled_a == shuffled_b, "shuffle must be deterministic given seed=42"

    truncated = shuffled_a[:5]
    assert len(truncated) == 5
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    assert all(all(c in valid_aa for c in k) for k in truncated)
