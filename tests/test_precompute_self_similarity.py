"""Tests for scripts/precompute_self_similarity.py - pure logic, no FASTA I/O."""

import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from precompute_self_similarity import (
    compute_self_similarity,
    build_kmer_sets,
    process_peptides,
)

# ---------------------------------------------------------------------------
# Fixtures: in-memory k-mer sets so no FASTA file is needed
# ---------------------------------------------------------------------------

HUMAN_PEPTIDE_9 = "GILGFVFTL"  # canonical flu 9-mer (in influenza NP, also in human data)
HUMAN_PEPTIDE_8 = "ACDEFGHI"  # 8-mer present in the kmer_sets_simple fixture
DECOY_9 = "QQQQQQQQQ"  # Q-repeat; not in any real proteome
DECOY_8 = "QQQQQQQQ"


@pytest.fixture()
def kmer_sets_simple():
    """Minimal k-mer sets containing a handful of known human peptides."""
    return {
        8: {HUMAN_PEPTIDE_8, "ACDEFGHI", "KLMNPQRS"},
        9: {HUMAN_PEPTIDE_9, "ACDEFGHIK", "LMNPQRSTVW"[:9]},
    }


# ---------------------------------------------------------------------------
# compute_self_similarity
# ---------------------------------------------------------------------------


class TestComputeSelfSimilarity:
    def test_9mer_exact_match(self, kmer_sets_simple):
        result = compute_self_similarity(HUMAN_PEPTIDE_9, kmer_sets_simple)
        assert result["self_similarity_exact_match"] is True
        assert result["self_similarity_max_identity"] == 1.0

    def test_9mer_no_match(self, kmer_sets_simple):
        result = compute_self_similarity(DECOY_9, kmer_sets_simple)
        assert result["self_similarity_exact_match"] is False
        assert result["self_similarity_max_identity"] == 0.0

    def test_8mer_exact_match(self, kmer_sets_simple):
        result = compute_self_similarity(HUMAN_PEPTIDE_8, kmer_sets_simple)
        assert result["self_similarity_exact_match"] is True
        assert result["self_similarity_max_identity"] == 1.0

    def test_8mer_no_match(self, kmer_sets_simple):
        result = compute_self_similarity(DECOY_8, kmer_sets_simple)
        assert result["self_similarity_exact_match"] is False
        assert result["self_similarity_max_identity"] == 0.0

    def test_10mer_containing_human_9mer_window(self, kmer_sets_simple):
        """A 10-mer that contains a human 9-mer as a sub-window → match."""
        # HUMAN_PEPTIDE_9 = "GILGFVFTL" (9 AA); prepend one AA
        ten_mer = "A" + HUMAN_PEPTIDE_9  # "AGILGFVFTL"
        result = compute_self_similarity(ten_mer, kmer_sets_simple)
        assert result["self_similarity_exact_match"] is True
        assert result["self_similarity_max_identity"] == 1.0

    def test_10mer_no_human_window(self, kmer_sets_simple):
        ten_mer = DECOY_9 + "A"  # 10-mer, no 9-mer window matches
        result = compute_self_similarity(ten_mer, kmer_sets_simple)
        assert result["self_similarity_exact_match"] is False

    def test_11mer_containing_human_9mer(self, kmer_sets_simple):
        """11-mer with human 9-mer embedded in the middle."""
        eleven_mer = "A" + HUMAN_PEPTIDE_9 + "V"  # "AGILGFVFTLV"
        result = compute_self_similarity(eleven_mer, kmer_sets_simple)
        assert result["self_similarity_exact_match"] is True

    def test_7mer_returns_no_match(self, kmer_sets_simple):
        """Peptides outside 8-11mer range return 0.0 / False."""
        result = compute_self_similarity("ACDEFGH", kmer_sets_simple)
        assert result["self_similarity_exact_match"] is False
        assert result["self_similarity_max_identity"] == 0.0

    def test_12mer_returns_no_match(self, kmer_sets_simple):
        result = compute_self_similarity("ACDEFGHIKLMN", kmer_sets_simple)
        assert result["self_similarity_max_identity"] == 0.0

    def test_output_keys_present(self, kmer_sets_simple):
        result = compute_self_similarity(DECOY_9, kmer_sets_simple)
        assert "self_similarity_max_identity" in result
        assert "self_similarity_exact_match" in result

    def test_max_identity_is_float(self, kmer_sets_simple):
        result = compute_self_similarity(HUMAN_PEPTIDE_9, kmer_sets_simple)
        assert isinstance(result["self_similarity_max_identity"], float)

    def test_exact_match_is_bool(self, kmer_sets_simple):
        result = compute_self_similarity(DECOY_9, kmer_sets_simple)
        assert isinstance(result["self_similarity_exact_match"], bool)

    def test_identity_bounded_0_to_1(self, kmer_sets_simple):
        for pep in [HUMAN_PEPTIDE_9, DECOY_9, HUMAN_PEPTIDE_8, DECOY_8]:
            r = compute_self_similarity(pep, kmer_sets_simple)
            assert 0.0 <= r["self_similarity_max_identity"] <= 1.0


# ---------------------------------------------------------------------------
# process_peptides
# ---------------------------------------------------------------------------


class TestProcessPeptides:
    def test_returns_dataframe(self, kmer_sets_simple):
        df = process_peptides([HUMAN_PEPTIDE_9, DECOY_9], kmer_sets_simple)
        assert isinstance(df, pd.DataFrame)

    def test_output_columns(self, kmer_sets_simple):
        df = process_peptides([HUMAN_PEPTIDE_9], kmer_sets_simple)
        assert "peptide" in df.columns
        assert "self_similarity_max_identity" in df.columns
        assert "self_similarity_exact_match" in df.columns

    def test_known_match_in_output(self, kmer_sets_simple):
        df = process_peptides([HUMAN_PEPTIDE_9, DECOY_9], kmer_sets_simple)
        row = df[df["peptide"] == HUMAN_PEPTIDE_9].iloc[0]
        assert bool(row["self_similarity_exact_match"]) is True
        assert row["self_similarity_max_identity"] == 1.0

    def test_known_non_match_in_output(self, kmer_sets_simple):
        df = process_peptides([HUMAN_PEPTIDE_9, DECOY_9], kmer_sets_simple)
        row = df[df["peptide"] == DECOY_9].iloc[0]
        assert bool(row["self_similarity_exact_match"]) is False

    def test_resume_skips_existing(self, kmer_sets_simple):
        """--resume mode: peptides in existing cache are skipped."""
        df = process_peptides(
            [HUMAN_PEPTIDE_9, DECOY_9],
            kmer_sets_simple,
            existing={HUMAN_PEPTIDE_9},
        )
        assert len(df) == 1
        assert df.iloc[0]["peptide"] == DECOY_9

    def test_empty_input(self, kmer_sets_simple):
        df = process_peptides([], kmer_sets_simple)
        assert len(df) == 0

    def test_all_existing_returns_empty(self, kmer_sets_simple):
        df = process_peptides(
            [DECOY_9],
            kmer_sets_simple,
            existing={DECOY_9},
        )
        assert len(df) == 0


# ---------------------------------------------------------------------------
# build_kmer_sets - integration test using an in-memory FASTA
# ---------------------------------------------------------------------------


class TestBuildKmerSets:
    def test_single_protein_9mers(self, tmp_path):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">sp|TEST\nGILGFVFTLACDEFGHIK\n")
        ksets = build_kmer_sets(str(fasta), kmer_lengths=(9,))
        # Protein is 18 AA → 10 unique 9-mers
        assert len(ksets[9]) == 10
        assert "GILGFVFTL" in ksets[9]
        assert "LACDEFGHI" in ksets[9]

    def test_single_protein_8mers(self, tmp_path):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">sp|TEST\nACDEFGHIKL\n")
        ksets = build_kmer_sets(str(fasta), kmer_lengths=(8,))
        # 10 AA → 3 unique 8-mers
        assert len(ksets[8]) == 3
        assert "ACDEFGHI" in ksets[8]

    def test_non_standard_aa_excluded(self, tmp_path):
        """X residues (ambiguous) must not appear in any k-mer."""
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">sp|TEST\nACDEFGHXKL\n")
        ksets = build_kmer_sets(str(fasta), kmer_lengths=(9,))
        for kmer in ksets[9]:
            assert "X" not in kmer

    def test_multiple_proteins_dedup(self, tmp_path):
        """Identical k-mers from two proteins appear only once."""
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">sp|P1\nGILGFVFTLAAAA\n>sp|P2\nGILGFVFTLBBBB\n")
        ksets = build_kmer_sets(str(fasta), kmer_lengths=(9,))
        # GILGFVFTL appears in both but should only be stored once
        count = sum(1 for k in ksets[9] if k == "GILGFVFTL")
        assert count == 1

    def test_returns_both_lengths(self, tmp_path):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">sp|TEST\nGILGFVFTLKKK\n")
        ksets = build_kmer_sets(str(fasta), kmer_lengths=(8, 9))
        assert 8 in ksets
        assert 9 in ksets

    def test_empty_fasta(self, tmp_path):
        fasta = tmp_path / "empty.fasta"
        fasta.write_text("")
        ksets = build_kmer_sets(str(fasta), kmer_lengths=(9,))
        assert len(ksets[9]) == 0
