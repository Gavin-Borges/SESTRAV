"""Pin manuscript Section 3.3's decoy-vs-binding-matrix figures to their source.

Section 3.3 asserts fourteen quantities about the allele_matched_nonbinder decoy
population and its coverage in the tracked binding matrix. Until
scripts/compute_section33_decoy_binding_join.py existed, no tracked code
performed that join, so the figures were prose nothing could check.

This test closes that loop from the other side: it recomputes the join and
asserts each figure still equals what docs/paper.md prints. If the corpus, the
binding matrix, or the join logic moves, this fails and names the figure, rather
than the manuscript quietly going stale - which is the failure mode this
repository keeps re-learning.

Ten of the fourteen are integer counts, which the integrity harness's
reconcile check cannot bind (it searches for "3112" while the manuscript
correctly writes "3,112"). That is exactly why they are pinned here instead.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pandas")

from scripts.compute_section33_decoy_binding_join import compute_join  # noqa: E402

# Every value below is quoted from docs/paper.md Section 3.3. Keep this table
# and the manuscript in sync deliberately: an edit to one should fail here until
# the other is updated on purpose.
EXPECTED = {
    "decoy_rows_active": 3112,
    "decoy_rows_in_matrix": 218,
    "decoy_distinct_peptides_in_matrix": 168,
    "decoy_max_presentation_median_rows": 0.761,
    "decoy_max_presentation_median_distinct": 0.740,
    "decoy_max_presentation_min": 0.503,
    "decoy_max_presentation_max": 0.982,
    "positive_max_presentation_median_active": 0.712,
    "positive_rows_active_in_matrix": 6431,
    "positive_max_presentation_median_all_rows": 0.705,
    "positive_rows_all_in_matrix": 7037,
    "positive_rows_quarantined_in_matrix": 606,
    "decoy_rows_absent_from_matrix": 2894,
    "decoy_absent_from_matrix_pct": 93.0,
    "zero_vector_active_positives": 1624,
    "tested_negative_rows_active": 22467,
    "tested_negative_rows_absent_from_matrix": 0,
    "iedb_api_rows_active": 1963,
    "iedb_api_rows_absent_from_matrix": 0,
}


@pytest.fixture(scope="module")
def joined():
    return compute_join().set_index("metric")["value"].to_dict()


@pytest.mark.parametrize("metric,expected", sorted(EXPECTED.items()))
def test_section33_figure_matches_manuscript(joined, metric, expected):
    assert metric in joined, f"{metric} missing from the join output"
    assert joined[metric] == pytest.approx(expected), (
        f"Section 3.3 prints {expected} for {metric}, join now yields "
        f"{joined[metric]}. Either the corpus moved or docs/paper.md is stale - "
        "resolve deliberately, do not relax this assertion."
    )


def test_coverage_asymmetry_holds(joined):
    """The actual Section 3.3 claim: decoys are uncovered, real negatives are not.

    Asserted as a relationship rather than three separate constants so it keeps
    meaning if the corpus grows. 93% of decoys absent against 0 of either
    real-negative class is the asymmetry the section is built on.
    """
    assert joined["decoy_absent_from_matrix_pct"] > 90.0
    assert joined["tested_negative_rows_absent_from_matrix"] == 0
    assert joined["iedb_api_rows_absent_from_matrix"] == 0


def test_zero_vector_positives_are_a_single_virus(joined):
    """Section 3.3 rests on "every one of the 1,624 is an HIV-1 peptide".

    That single-virus fact is what makes the five-of-six-folds result one
    virus's rows rather than five independent replications - the manuscript
    says so explicitly, so it is load-bearing and pinned here.
    """
    assert joined["zero_vector_active_positives_distinct_viruses"] == 1
