"""Tests for features.py graph, ERAP, and weight-computation gaps.

Targets:
  - get_cb_cb_edges: 10-mer (614-615) and 11-mer (616-619) bulge contacts
  - compute_erap_trimming_score: short flanking_seq padding path (line 497)
  - compute_sample_weights: DataFrame without 'peptide' col (line 206->219 False branch)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    compute_erap_trimming_score,
    compute_sample_weights,
    get_cb_cb_edges,
)


# ---------------------------------------------------------------------------
# get_cb_cb_edges - 10-mer and 11-mer bulge contacts
# ---------------------------------------------------------------------------

class TestGetCbCbEdges:
    def _edge_set(self, length: int) -> set[tuple[int, int]]:
        return set(get_cb_cb_edges(length))

    def test_9mer_has_bulge_contacts(self):
        edges = self._edge_set(9)
        assert (3, 5) in edges
        assert (4, 6) in edges

    def test_10mer_has_correct_bulge_contacts(self):
        edges = self._edge_set(10)
        assert (3, 6) in edges
        assert (4, 7) in edges
        assert (3, 8) not in edges, "10-mer should not have 9-mer (3,7) or 11-mer (3,7)-style contacts"

    def test_11mer_has_correct_bulge_contacts(self):
        edges = self._edge_set(11)
        assert (3, 7) in edges
        assert (4, 8) in edges
        assert (5, 9) in edges

    def test_all_edges_are_canonical_ordered(self):
        for length in (9, 10, 11):
            for u, v in get_cb_cb_edges(length):
                assert u <= v, f"Edge ({u},{v}) is not canonically ordered"

    def test_no_duplicate_edges(self):
        for length in (9, 10, 11):
            edges = get_cb_cb_edges(length)
            assert len(edges) == len(set(edges)), f"Duplicate edges for length {length}"

    def test_8mer_no_bulge_contacts(self):
        edges = self._edge_set(8)
        for u, v in edges:
            assert v - u <= 2, "8-mer has only adjacent and i+2 contacts"


# ---------------------------------------------------------------------------
# compute_erap_trimming_score - short flanking_seq triggers padding (line 497)
# ---------------------------------------------------------------------------

class TestComputeErapTrimmingScore:
    def test_empty_peptide_returns_zero(self):
        assert compute_erap_trimming_score("") == 0.0

    def test_short_flanking_seq_padded(self):
        score = compute_erap_trimming_score("A", flanking_seq="B")
        assert 0.0 <= score <= 10.0, "Score must be in [0, 10] after padding"

    def test_single_char_flanking_single_char_peptide(self):
        score = compute_erap_trimming_score("G", flanking_seq="L")
        assert isinstance(score, float)

    def test_normal_9mer_no_flanking(self):
        score = compute_erap_trimming_score("GLFYTRTGL")
        assert 0.0 <= score <= 10.0

    def test_flanking_seq_3plus_uses_last_3(self):
        score_long = compute_erap_trimming_score("GLFYTRTGL", flanking_seq="AAAGLL")
        score_short = compute_erap_trimming_score("GLFYTRTGL", flanking_seq="GLL")
        assert score_long == pytest.approx(score_short), (
            "Only last 3 chars of flanking_seq matter"
        )


# ---------------------------------------------------------------------------
# compute_sample_weights - DataFrame without 'peptide' col (line 206->219 False)
# ---------------------------------------------------------------------------

class TestComputeSampleWeightsNoPeptideCol:
    def test_df_without_peptide_col_skips_length_correction(self):
        df = pd.DataFrame({
            "virus": ["EBV"] * 5 + ["HPV16"] * 5,
            "seq": ["CLGGLLTMV"] * 10,
        })
        w = compute_sample_weights(df, length_col=None)
        assert w.shape == (10,)
        np.testing.assert_allclose(w.mean(), 1.0, atol=1e-9)

    def test_custom_length_col_not_in_df_falls_back_gracefully(self):
        df = pd.DataFrame({
            "virus": ["EBV"] * 4 + ["CMV"] * 4,
            "seq": ["GILGFVFTL"] * 8,
        })
        w = compute_sample_weights(df, length_col="nonexistent_col")
        assert w.shape == (8,)
        np.testing.assert_allclose(w.mean(), 1.0, atol=1e-9)
