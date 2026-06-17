"""Tests for the pure utility functions in scripts/generate_hard_decoys.py.

The `generate_decoys()` entry point requires MHCflurry models (not available
in CI), so only the FASTA parser and k-mer extractor are tested here.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from generate_hard_decoys import extract_kmers, load_fasta  # noqa: E402


# ---------------------------------------------------------------------------
# FASTA parser
# ---------------------------------------------------------------------------

def test_load_fasta_single_sequence(tmp_path):
    fa = tmp_path / "test.fasta"
    fa.write_text(">seq1\nACDEFGHIKL\n")
    assert load_fasta(str(fa)) == ["ACDEFGHIKL"]


def test_load_fasta_multiline_sequence(tmp_path):
    fa = tmp_path / "test.fasta"
    fa.write_text(">seq1\nACDEF\nGHIKL\n")
    assert load_fasta(str(fa)) == ["ACDEFGHIKL"]


def test_load_fasta_multiple_sequences(tmp_path):
    fa = tmp_path / "test.fasta"
    fa.write_text(">s1\nACDEF\n>s2\nGHIKL\nMNPQR\n")
    seqs = load_fasta(str(fa))
    assert seqs == ["ACDEF", "GHIKLMNPQR"]


def test_load_fasta_empty_file(tmp_path):
    fa = tmp_path / "empty.fasta"
    fa.write_text("")
    assert load_fasta(str(fa)) == []


def test_load_fasta_no_trailing_newline(tmp_path):
    fa = tmp_path / "test.fasta"
    fa.write_text(">s1\nACDEF")
    assert load_fasta(str(fa)) == ["ACDEF"]


# ---------------------------------------------------------------------------
# k-mer extractor
# ---------------------------------------------------------------------------

def test_extract_kmers_returns_valid_9mers():
    seqs = ["ACDEFGHIKLMNPQRSTVWY"]  # 20 AAs → 12 valid 9-mers
    kmers = extract_kmers(seqs, k=9)
    assert all(len(k) == 9 for k in kmers)
    assert len(kmers) == 12


def test_extract_kmers_filters_invalid_characters():
    seqs = ["XACDEFGHI", "ACDEFGHIK"]  # first has 'X' → only one valid 9-mer
    kmers = extract_kmers(seqs, k=9)
    assert all("X" not in k for k in kmers)


def test_extract_kmers_deduplicates():
    seqs = ["ACDEFGHIK", "ACDEFGHIK"]  # same sequence twice
    kmers = extract_kmers(seqs, k=9)
    assert len(kmers) == len(set(kmers))


def test_extract_kmers_is_sorted():
    seqs = ["WVTSRQPNM", "ACDEFGHIK"]
    kmers = extract_kmers(seqs, k=9)
    assert kmers == sorted(kmers)


def test_extract_kmers_sequence_shorter_than_k():
    seqs = ["ACDE"]  # 4 chars, k=9 → no k-mers
    assert extract_kmers(seqs, k=9) == []


def test_extract_kmers_empty_input():
    assert extract_kmers([], k=9) == []
