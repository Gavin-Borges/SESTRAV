"""Invariants for the two hardcoded HLA pocket pseudo-sequence tables.

The repository keeps the same ten 34-residue NetMHCpan pseudo-sequences in TWO
hand-maintained places:

  1. ``src/verify/mhc_pseudo_sequences.json``   -> SESTRAV-VERIFY's structural GNN
  2. ``scripts/extract_allele_aware_data.py``   -> ``HLA_PSEUDOSEQ``, mode-166 features

Nothing at runtime made them agree, and they silently drifted apart: claims
register D30 records that the script's table had collapsed to two distinct
strings across ten alleles, while eight of the JSON's ten slots held a sequence
that was not that allele's. Of those eight, exactly one was a mislabelled slot
rather than a merely-wrong value - ``HLA-A*02:01`` carried ``HLA-A*01:01``'s
real sequence verbatim; the other seven matched no panel allele.

The two failed differently. The script's block was labelled "Source: NetMHCpan
4.1 MHC_pseudo.dat", naming a real, fetchable file whose contents refuted it -
a false claim, and therefore a checkable one. The JSON claimed nothing at all:
ten bare allele keys, no comment, no source field, so its wrong values were
unfalsifiable rather than falsified. A claims audit found both; no test could
have.

These tests encode the invariants that make that class of drift impossible to
reintroduce silently. They deliberately assert structure and cross-table
agreement rather than pinning the ten literal sequences: pinning the values
would only restate the tables in a third place, adding a fourth thing to keep
in sync. Provenance of the values themselves is a source-verification question
(see the D30 row and the header comment in the script), not a unit-test one.
"""

import json
from pathlib import Path

import pytest

from scripts.extract_allele_aware_data import HLA_PSEUDOSEQ, PSEUDO_LEN

# The 20 standard amino acids. Both consumers look each residue up through a
# four-property .get(aa, <default>) chain, so anything outside this set silently
# degrades to a default rather than raising: a stray character becomes a
# plausible-looking feature vector, not an error. The defaults are not even the
# same on both sides - StructuralPeptideMHCDataset.__init__ falls back to
# VDW_VOL.get(aa, 110.0) for pocket residues where pseudoseq_to_features uses
# 0.0 - so the same bad character would corrupt the two subsystems differently.
# Same failure class as D30; the length guards do not cover it.
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

JSON_PATH = Path(__file__).resolve().parents[1] / "src" / "verify" / "mhc_pseudo_sequences.json"


@pytest.fixture(scope="module")
def json_raw():
    """The JSON exactly as it ships, metadata keys included."""
    with open(JSON_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def json_table(json_raw):
    """Allele entries only. Underscore-prefixed keys are metadata (see _source)."""
    return {k: v for k, v in json_raw.items() if not k.startswith("_")}


@pytest.fixture(scope="module")
def tables(json_table):
    """Both tables, so each invariant is asserted against both in one place."""
    return {"mhc_pseudo_sequences.json": json_table, "HLA_PSEUDOSEQ": HLA_PSEUDOSEQ}


def test_json_records_its_own_provenance(json_raw):
    """The JSON shipped unsourced, which is why wrong values there were unfalsifiable.

    The script's copy at least carried a source attribution - a false one, which
    is how D30 was found at all. This file carried none: ten bare allele keys and
    nothing to check them against. A reader could not tell a real pseudo-sequence
    from a fabricated one without leaving the file.
    """
    source = json_raw.get("_source", "")
    assert source, f"{JSON_PATH.name} carries no _source field"
    assert "MHC_pseudo.dat" in source, "the _source field does not name the source file"


def test_json_has_no_duplicate_keys():
    """A repeated JSON key parses silently, last-one-wins, losing an allele."""
    raw = JSON_PATH.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    # object_pairs_hook sees every pair, including ones json.load would collapse.
    pairs = json.loads(raw, object_pairs_hook=lambda kv: kv)
    assert len(pairs) == len(parsed), f"duplicate key(s) in {JSON_PATH.name}"


def test_every_sequence_is_exactly_the_pocket_length(tables):
    for name, table in tables.items():
        for allele, seq in table.items():
            assert len(seq) == PSEUDO_LEN, (
                f"{name}[{allele!r}] is {len(seq)} chars, expected {PSEUDO_LEN}. "
                "A short entry is not a smaller correct encoding - it frame-shifts "
                "every position after the missing residue."
            )


def test_every_sequence_uses_only_standard_amino_acids(tables):
    for name, table in tables.items():
        for allele, seq in table.items():
            bad = sorted(set(seq) - STANDARD_AA)
            assert not bad, f"{name}[{allele!r}] contains non-standard residue(s): {bad}"


def test_no_two_alleles_share_a_sequence(tables):
    """The D30 degeneracy: ten alleles collapsing onto two distinct strings."""
    for name, table in tables.items():
        by_seq = {}
        for allele, seq in table.items():
            by_seq.setdefault(seq, []).append(allele)
        collisions = {s: a for s, a in by_seq.items() if len(a) > 1}
        assert not collisions, (
            f"{name} has alleles sharing an identical pseudo-sequence: "
            f"{sorted(collisions.values())}. Identical strings cannot encode "
            "different pockets, so the derived features carry no allele signal."
        )


def test_the_two_tables_hold_the_same_alleles(tables):
    json_keys = set(tables["mhc_pseudo_sequences.json"])
    script_keys = set(tables["HLA_PSEUDOSEQ"])
    assert json_keys == script_keys, (
        f"only in JSON: {sorted(json_keys - script_keys)}; "
        f"only in HLA_PSEUDOSEQ: {sorted(script_keys - json_keys)}"
    )


def test_the_two_tables_agree_on_every_sequence(tables):
    """The invariant whose absence let the two copies drift apart."""
    json_table = tables["mhc_pseudo_sequences.json"]
    disagreements = {
        allele: (json_table[allele], HLA_PSEUDOSEQ[allele])
        for allele in sorted(set(json_table) & set(HLA_PSEUDOSEQ))
        if json_table[allele] != HLA_PSEUDOSEQ[allele]
    }
    assert not disagreements, (
        "mhc_pseudo_sequences.json and HLA_PSEUDOSEQ disagree for "
        f"{sorted(disagreements)}. They describe the same ten pockets and feed "
        "two different subsystems; a divergence means one of them is wrong."
    )


def test_a02_01_does_not_hold_a01_01s_sequence(tables):
    """Named regression for D30's most consequential single defect.

    ``YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY`` is HLA-A*01:01's true sequence. It sat
    in the A*02:01 slot, so the most-studied class I allele in the corpus was
    handed another allele's pocket chemistry under a correct-looking label - the
    kind of error the generic invariants above would not catch, because a
    misfiled-but-real sequence is the right length, the right alphabet, and
    distinct from its neighbours.
    """
    a01_01 = "YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY"
    for name, table in tables.items():
        assert table["HLA-A*01:01"] == a01_01, (
            f"{name}: HLA-A*01:01 no longer holds its own known sequence; this "
            "test's premise has moved and the assertion below is meaningless."
        )
        assert table["HLA-A*02:01"] != a01_01, (
            f"{name}: the HLA-A*02:01 slot holds HLA-A*01:01's sequence (D30)."
        )
