"""The leakage audit's near-homolog arm: the relation, the clustering, the fold count.

Peptide-grouped CV closes the exact-duplicate channel by construction and
leaves near-homologs untouched, so the near-homolog arm is the only thing in
`scripts/audit_cv_leakage.py` that can see that second channel. A silently
narrowed relation would not fail anything else in the suite: it would simply
report a smaller, still-plausible exposure, which is the failure mode these
tests exist to make loud.

MUTATION VERIFICATION, measured rather than asserted. Four mutants were applied
to `_near_homolog_adjacency` and this module was run against each; the named
test is the one that pins that edge type, and the count is how many tests in
this file the mutant killed in total:

  drop hamming1 (skip every wildcard bucket)  -> 7 failures, incl.
      test_hamming1_edge_is_detected
  drop indel1 (never look up a deletion)      -> 1 failure:
      test_internal_single_indel_edge_is_detected
  drop containment (never scan a sub-window)  -> 2 failures, incl.
      test_containment_edge_is_detected
  star-link a wildcard bucket to its first
      member instead of all-pairs             -> 1 failure:
      test_hamming1_links_every_pair_in_a_wildcard_bucket

indel1 is the thinnest of the three at one killing test, which is why it has a
dedicated negative partner (test_two_residue_deletion_is_not_an_indel_edge)
rather than resting on a shared fixture. Widening the relation is covered
separately by the Hamming-2, two-residue-deletion and unrelated-peptide tests.
All fixtures are hand-built peptide strings; nothing here reads the live
corpus, the binding matrix or any cache.
"""

import numpy as np
import pandas as pd
import pytest

from scripts import audit_cv_leakage as acl

# 9mers used throughout. BASE and BASE_H1 differ at position 4 alone; BASE_H2
# differs from BASE at two positions; UNRELATED shares no window with any.
BASE = "SLYNTVATL"
BASE_H1 = "SLYNKVATL"
BASE_H2 = "SLYNKVWTL"
UNRELATED = "GGGGGGGGG"


def _adj(*peptides: str) -> dict[str, frozenset[str]]:
    return acl._near_homolog_adjacency(peptides)


# --------------------------------------------------------------------------
# Edge type 1: substring containment
# --------------------------------------------------------------------------


def test_containment_edge_is_detected():
    """An INTERNAL 9mer window of an 11mer. FAILS if containment is dropped.

    The length difference is 2, so neither hamming1 (equal length only) nor
    indel1 (one residue only) can supply this edge; containment is the sole
    source and the assertion is therefore specific to it.
    """
    long_peptide = "KSLYNTVATLQ"
    assert long_peptide[1:10] == BASE
    adjacency = _adj(long_peptide, BASE)
    assert adjacency[BASE] == frozenset({long_peptide})
    assert adjacency[long_peptide] == frozenset({BASE})


def test_containment_uses_only_lengths_the_corpus_carries():
    """A 3mer window is not looked up when no 3mer is in the corpus.

    The sub-window scan is capped at the lengths actually present, which is
    what keeps it bounded. This test pins that the cap is a performance
    device and not a correctness one: with a 3mer present the edge appears.
    """
    without_short = _adj(BASE, UNRELATED)
    assert BASE not in without_short
    with_short = _adj(BASE, UNRELATED, "YNT")
    assert with_short["YNT"] == frozenset({BASE})


# --------------------------------------------------------------------------
# Edge type 2: Hamming distance 1, equal length
# --------------------------------------------------------------------------


def test_hamming1_edge_is_detected():
    """Two equal-length peptides differing at exactly one position.

    FAILS if the positional-wildcard bucketing is dropped. Containment cannot
    supply this edge (equal lengths, and neither is a window of the other) and
    neither can indel1 (equal lengths).
    """
    adjacency = _adj(BASE, BASE_H1)
    assert adjacency[BASE] == frozenset({BASE_H1})
    assert adjacency[BASE_H1] == frozenset({BASE})


def test_hamming2_is_not_an_edge():
    """The relation must not widen past one substitution."""
    adjacency = _adj(BASE, BASE_H2)
    assert adjacency == {}


def test_hamming1_links_every_pair_in_a_wildcard_bucket():
    """Three peptides differing at the same position are pairwise related.

    Star-linking a bucket to its first member would be cheaper and would give
    the same CLUSTERS, but a wrong ADJACENCY, and adjacency is what the fold
    count consumes. This test fails under that shortcut.
    """
    third = "SLYNRVATL"
    adjacency = _adj(BASE, BASE_H1, third)
    assert adjacency[BASE] == frozenset({BASE_H1, third})
    assert adjacency[BASE_H1] == frozenset({BASE, third})
    assert adjacency[third] == frozenset({BASE, BASE_H1})


# --------------------------------------------------------------------------
# Edge type 3: single indel
# --------------------------------------------------------------------------


def test_internal_single_indel_edge_is_detected():
    """An 8mer formed by deleting ONE INTERNAL residue of a 9mer.

    FAILS if the deletion lookup is dropped. The deletion is internal, so the
    8mer is a subsequence and not a contiguous sub-window: containment cannot
    supply this edge. The lengths differ, so hamming1 cannot either.
    """
    deleted = BASE[:4] + BASE[5:]  # SLYN|T|VATL -> SLYNVATL
    assert deleted == "SLYNVATL"
    assert deleted not in (BASE[:8], BASE[1:])  # not a contiguous window
    adjacency = _adj(BASE, deleted)
    assert adjacency[BASE] == frozenset({deleted})
    assert adjacency[deleted] == frozenset({BASE})


def test_two_residue_deletion_is_not_an_indel_edge():
    """indel1 means exactly one residue. A two-residue internal deletion is not an edge."""
    deleted_two = BASE[:3] + BASE[5:]  # SLY|NT|VATL -> SLYVATL
    assert deleted_two == "SLYVATL"
    adjacency = _adj(BASE, deleted_two)
    assert adjacency == {}


# --------------------------------------------------------------------------
# Shape of the relation itself
# --------------------------------------------------------------------------


def test_unrelated_peptides_are_absent_from_the_mapping():
    """A peptide with no near-homolog carries no key at all, so len() is the count.

    `_homology_ab` reports `peptides_with_near_homolog` as `len(adjacency)`.
    Emitting an empty frozenset for an isolated peptide would silently turn
    that count into the full distinct-peptide count.
    """
    adjacency = _adj(BASE, UNRELATED)
    assert adjacency == {}
    assert len(adjacency) == 0


def test_no_self_edge_even_when_a_peptide_repeats():
    """An exact repeat is the duplicate channel `_fold_overlap` measures, not a homolog."""
    adjacency = _adj(BASE, BASE, BASE)
    assert adjacency == {}


def test_adjacency_is_symmetric():
    peptides = [BASE, BASE_H1, "KSLYNTVATLQ", BASE[:4] + BASE[5:], UNRELATED]
    adjacency = _adj(*peptides)
    for peptide, neighbours in adjacency.items():
        for neighbour in neighbours:
            assert peptide in adjacency[neighbour]


# --------------------------------------------------------------------------
# Union-find clustering
# --------------------------------------------------------------------------


def test_clusters_are_transitive_where_edges_are_not():
    """BASE-BASE_H1 and BASE_H1-BASE_H2 are edges; BASE-BASE_H2 is not, yet all three cluster.

    This is the property that makes the cluster count the right instrument for
    "would a homology-grouped splitter be constructible" while adjacency stays
    the right instrument for "is there a near-homolog in the train fold".
    """
    distinct = sorted({BASE, BASE_H1, BASE_H2})
    adjacency = acl._near_homolog_adjacency(distinct)
    assert BASE_H2 not in adjacency[BASE]
    clusters = acl._homology_clusters(distinct, adjacency)
    assert clusters[BASE] == clusters[BASE_H1] == clusters[BASE_H2]


def test_isolated_peptides_are_their_own_cluster():
    distinct = sorted({BASE, BASE_H1, UNRELATED})
    clusters = acl._homology_clusters(distinct, acl._near_homolog_adjacency(distinct))
    assert clusters[UNRELATED] != clusters[BASE]
    assert len(set(clusters.values())) == 2


# --------------------------------------------------------------------------
# The fold count
# --------------------------------------------------------------------------


def _split(train: list[int], test: list[int]):
    return (np.array(train), np.array(test))


def test_fold_rows_counts_rows_not_peptides():
    """One held-out peptide on three rows, with a near-homolog in train, leaks three ROWS."""
    peptides = np.array([BASE, BASE, BASE, BASE_H1, UNRELATED])
    adjacency = acl._near_homolog_adjacency(peptides)
    rows = acl._near_homolog_fold_rows(
        "arm", peptides, [_split([3, 4], [0, 1, 2])], adjacency
    )
    fold = next(r for r in rows if r["metric"] == "fold0_near_homolog_row_pct")
    assert fold["n_leaked"] == 3
    assert fold["n_test"] == 3
    assert fold["value"] == pytest.approx(100.0)


def test_a_held_out_row_with_no_near_homolog_in_train_is_not_counted():
    """The homolog exists in the corpus but sits in the same held-out fold."""
    peptides = np.array([BASE, BASE_H1, UNRELATED])
    adjacency = acl._near_homolog_adjacency(peptides)
    rows = acl._near_homolog_fold_rows("arm", peptides, [_split([2], [0, 1])], adjacency)
    fold = next(r for r in rows if r["metric"] == "fold0_near_homolog_row_pct")
    assert fold["n_leaked"] == 0


def test_overall_row_pct_pools_folds_rather_than_averaging_them():
    """Uneven folds: 1 of 1 and 1 of 3 pool to 50%, not to the 66.7% a fold mean gives."""
    peptides = np.array([BASE, BASE_H1, UNRELATED, UNRELATED])
    adjacency = acl._near_homolog_adjacency(peptides)
    rows = acl._near_homolog_fold_rows(
        "arm", peptides, [_split([1, 2, 3], [0]), _split([0], [1, 2, 3])], adjacency
    )
    folds = [r["value"] for r in rows if r["metric"].startswith("fold")]
    assert folds == pytest.approx([100.0, 100.0 / 3])
    overall = next(r for r in rows if r["metric"] == "overall_near_homolog_row_pct")
    assert overall["n_test"] == 4
    assert overall["n_leaked"] == 2
    assert overall["value"] == pytest.approx(50.0)
    assert overall["value"] != pytest.approx(float(np.mean(folds)))


def test_exact_peptide_control_is_zero_for_a_grouped_split():
    peptides = np.array([BASE, BASE, BASE_H1, UNRELATED])
    adjacency = acl._near_homolog_adjacency(peptides)
    rows = acl._near_homolog_fold_rows("arm", peptides, [_split([2, 3], [0, 1])], adjacency)
    control = next(r for r in rows if r["metric"] == "overall_exact_peptide_overlap_pct")
    assert control["value"] == pytest.approx(0.0)


def test_exact_peptide_control_fires_when_the_split_is_not_grouped():
    """The control's whole job: a split that puts one peptide on both sides is caught.

    Without this the near-homolog percentage beside it could be read as a
    grouped-CV figure when the split was never grouped.
    """
    peptides = np.array([BASE, BASE, BASE_H1, UNRELATED])
    adjacency = acl._near_homolog_adjacency(peptides)
    rows = acl._near_homolog_fold_rows("arm", peptides, [_split([0, 2, 3], [1])], adjacency)
    control = next(r for r in rows if r["metric"] == "overall_exact_peptide_overlap_pct")
    assert control["value"] == pytest.approx(100.0)


# --------------------------------------------------------------------------
# The arm end to end, on a hand-built corpus
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_corpus() -> pd.DataFrame:
    """30 peptides over 40 rows: 10 near-homolog pairs and 10 isolated peptides.

    Sized so StratifiedGroupKFold(n_splits=5) has enough groups per class, and
    labelled so the near-homolog partner of a held-out peptide is generally in
    another fold. Nothing here touches the live corpus.
    """
    residues = "ACDEFGHIKLMNPQRSTVWY"
    rows = []
    # 10 hamming1 pairs: a homopolymer 8mer stem plus L or M. Distinct stems
    # differ at eight positions, so no edge crosses a pair.
    for i in range(10):
        stem = residues[i] * 8
        for j, peptide in enumerate((stem + "L", stem + "M")):
            rows.append(
                {
                    "peptide": peptide,
                    "label": (i + j) % 2,
                    "negative_origin": "iedb_api",
                    "hla_allele": "HLA-A*02:01",
                }
            )
    # 10 isolated peptides, drawn from the residues the stems above do not use,
    # each on two rows so the row/peptide distinction is exercised end to end.
    for i in range(10):
        peptide = residues[10 + i] * 9
        for allele in ("HLA-A*02:01", "HLA-B*07:02"):
            rows.append(
                {
                    "peptide": peptide,
                    "label": i % 2,
                    "negative_origin": "iedb_api",
                    "hla_allele": allele,
                }
            )
    return pd.DataFrame(rows)


def test_homology_ab_emits_both_grouped_arms_and_the_relation(tiny_corpus):
    result = pd.DataFrame(acl._homology_ab(tiny_corpus))
    assert set(result["config"]) == {
        "homology_relation",
        "peptide_grouped_splitter",
        "production_grouped_splitter",
    }
    for config in ("peptide_grouped_splitter", "production_grouped_splitter"):
        arm = result[result["config"] == config]
        assert {f"fold{i}_near_homolog_row_pct" for i in range(acl.N_SPLITS)} <= set(arm["metric"])
        control = arm[arm["metric"] == "overall_exact_peptide_overlap_pct"]["value"].iloc[0]
        assert control == pytest.approx(0.0), "both arms must be peptide-grouped"
        overall = arm[arm["metric"] == "overall_near_homolog_row_pct"]["value"].iloc[0]
        assert 0.0 <= overall <= 100.0


def test_homology_relation_counts_are_internally_consistent(tiny_corpus):
    result = pd.DataFrame(acl._homology_ab(tiny_corpus))
    relation = result[result["config"] == "homology_relation"].set_index("metric")["value"]
    assert relation["distinct_peptides"] == 30
    assert relation["peptides_with_near_homolog"] == 20
    assert relation["peptides_with_near_homolog_pct"] == pytest.approx(100.0 * 20 / 30)
    assert relation["near_homolog_edges"] == 10  # one per pair, none across pairs
    assert relation["n_clusters"] == 20  # 10 pairs + 10 singletons
    assert relation["singleton_clusters"] == 10
    assert relation["largest_cluster_peptides"] == 2


def test_run_calls_the_homology_arm(monkeypatch):
    """A regression guard: the arm has to be wired into run(), not merely defined.

    Every other arm is stubbed out, so this test neither trains a model nor
    reads the corpus.
    """
    called: list[str] = []
    monkeypatch.setattr(acl, "_load_active", lambda path: pd.DataFrame({"peptide": ["A"], "label": [1]}))
    for name in (
        "_dataset_shape",
        "_fold_overlap",
        "_splitter_ab",
        "_per_virus_and_pooled_honest_ab",
        "_tier_a_ab",
        "_feature_mode_ab",
        "_vaccinia_ablation",
    ):
        monkeypatch.setattr(acl, name, lambda *a, **k: [])
    monkeypatch.setattr(
        acl,
        "_homology_ab",
        lambda active: called.append("homology") or [{"config": "homology_relation", "metric": "distinct_peptides", "value": 1.0}],
    )

    result = acl.run(acl.DATASET_PATH)

    assert called == ["homology"]
    assert (result["config"] == "homology_relation").any()
