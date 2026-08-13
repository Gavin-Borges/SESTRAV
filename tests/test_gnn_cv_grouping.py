"""Peptide-grouping tests for the GNN cross-validation path (src/train_gnn.py).

Defect D15 established that an UNGROUPED splitter leaks: every mode-31 feature
is a pure function of the peptide string, and the v5 corpus is deduplicated on
(peptide, hla_allele) rather than on peptide alone, so two rows sharing a
peptide are feature-identical. Phase 0 re-baselined src/train_classifier.py
under PeptideGroupedKFold; the GNN track kept sklearn's StratifiedKFold at both
CV call sites until this repair.

These mirror the shape of the Phase 0 negative control in
tests/test_ml_utils.py::test_multi_stratified_kfold_can_leak_peptides_across_folds:
a positive test that the shipped splitter is peptide-disjoint, and a negative
control proving the splitter it replaced leaks on the SAME fixture. Without the
negative control, the disjointness test could pass merely because the fixture
happened to have no repeated peptides.

The splitter is exercised directly rather than through a training run: both
entry points now obtain their folds from build_cv_splits, so it is the whole
unit under test, and no ESM-2 cache or GPU budget is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="src.train_gnn imports torch")

from src.train_gnn import (  # noqa: E402
    GNN_CV_SPLITTER,
    SPLITTER_COLUMN,
    build_cv_splits,
    build_oof_records,
)


# ---------------------------------------------------------------------------
# Fixture - peptides repeated across rows, the shape the v5 corpus actually has
# ---------------------------------------------------------------------------


def _train_pool(n_peptides: int = 40, rows_per_peptide: int = 3) -> pd.DataFrame:
    """A miniature training pool with duplicate peptides across HLA alleles.

    Matches the column set src/train_gnn.py reads off the real corpus:
    peptide, label, hla_allele, negative_origin.
    """
    rows = []
    alleles = ["HLA-A*02:01", "HLA-B*07:02", "HLA-B*35:03"]
    for i in range(n_peptides):
        for j in range(rows_per_peptide):
            rows.append(
                {
                    "peptide": f"PEPTIDE{i:03d}",
                    "label": i % 2,
                    "hla_allele": alleles[j % len(alleles)],
                    "negative_origin": "tested_negative" if i % 2 == 0 else None,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# build_cv_splits - the repair
# ---------------------------------------------------------------------------


def test_gnn_cv_splits_have_zero_peptide_overlap_in_every_fold():
    """THE disjointness test: no peptide may appear on both sides of a fold."""
    pool = _train_pool()
    peptides = pool["peptide"].to_numpy()
    y = pool["label"].to_numpy()

    splits = build_cv_splits(pool, y, n_splits=5, seed=42)

    assert len(splits) == 5
    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        overlap = set(peptides[train_idx]) & set(peptides[val_idx])
        assert not overlap, f"fold {fold} leaks {len(overlap)} peptide(s) into training"


def test_ungrouped_splitter_leaks_peptides_on_the_same_fixture():
    """NEGATIVE CONTROL - the splitter train_gnn.py used before this repair.

    Literally the pre-repair call: StratifiedKFold(n_splits=5, shuffle=True,
    random_state=seed).split(X_feats, y). If this ever stops leaking, the
    fixture has lost its duplicate peptides and the test above proves nothing.
    """
    from sklearn.model_selection import StratifiedKFold

    pool = _train_pool()
    peptides = pool["peptide"].to_numpy()
    y = pool["label"].to_numpy()
    X_feats = pd.DataFrame(np.zeros((len(pool), 3)))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    leaked_any = False
    for train_idx, val_idx in skf.split(X_feats, y):
        if set(peptides[train_idx]) & set(peptides[val_idx]):
            leaked_any = True
            break

    assert leaked_any, "the ungrouped splitter did not leak; fixture no longer has duplicates"


def test_gnn_cv_splits_cover_every_row_exactly_once():
    pool = _train_pool()
    y = pool["label"].to_numpy()
    splits = build_cv_splits(pool, y, n_splits=5, seed=42)

    held_out: list[int] = []
    for _, val_idx in splits:
        held_out.extend(val_idx.tolist())
    assert sorted(held_out) == list(range(len(pool)))


def test_gnn_cv_splits_are_deterministic_for_a_given_seed():
    pool = _train_pool()
    y = pool["label"].to_numpy()
    a = build_cv_splits(pool, y, n_splits=5, seed=7)
    b = build_cv_splits(pool, y, n_splits=5, seed=7)
    assert [(tr.tolist(), te.tolist()) for tr, te in a] == [
        (tr.tolist(), te.tolist()) for tr, te in b
    ]


def test_gnn_cv_splits_reject_a_pool_without_peptides():
    """No peptide column means no group, which is the leak this closes."""
    pool = _train_pool().drop(columns=["peptide"])
    y = pool["label"].to_numpy()
    with pytest.raises(ValueError, match="requires a 'peptide' column"):
        build_cv_splits(pool, y, n_splits=5, seed=42)


def test_gnn_cv_splits_work_without_the_optional_metadata_columns():
    """negative_origin / hla_allele are optional; only peptide is mandatory."""
    pool = _train_pool().drop(columns=["negative_origin", "hla_allele"])
    y = pool["label"].to_numpy()
    peptides = pool["peptide"].to_numpy()

    for train_idx, val_idx in build_cv_splits(pool, y, n_splits=5, seed=42):
        assert not (set(peptides[train_idx]) & set(peptides[val_idx]))


# ---------------------------------------------------------------------------
# build_oof_records - fold identity and splitter provenance in the artifact
# ---------------------------------------------------------------------------


def test_oof_records_carry_fold_and_splitter_provenance():
    pool = _train_pool(n_peptides=4, rows_per_peptide=2)
    val_idx = np.array([0, 1, 2])
    records = build_oof_records(
        pool,
        val_idx,
        val_labels=np.array([1.0, 0.0, 1.0]),
        val_preds=np.array([0.9, 0.1, 0.8]),
        fold=3,
    )

    assert len(records) == 3
    assert all(r["fold"] == 3 for r in records)
    assert all(r[SPLITTER_COLUMN] == GNN_CV_SPLITTER for r in records)
    assert [r["peptide"] for r in records] == pool["peptide"].iloc[val_idx].tolist()


def test_oof_records_carry_the_allele_so_rows_can_be_joined_one_to_one():
    """(peptide, hla_allele) is the v5 dedup key; peptide alone is not unique."""
    pool = _train_pool(n_peptides=4, rows_per_peptide=3)
    val_idx = np.array([0, 1, 2])
    records = build_oof_records(
        pool, val_idx, np.array([1.0, 1.0, 1.0]), np.array([0.5, 0.5, 0.5]), fold=1
    )
    assert [r["hla_allele"] for r in records] == pool["hla_allele"].iloc[val_idx].tolist()


def test_oof_records_omit_the_allele_when_the_corpus_has_none():
    pool = _train_pool(n_peptides=4, rows_per_peptide=2).drop(columns=["hla_allele"])
    records = build_oof_records(
        pool, np.array([0]), np.array([1.0]), np.array([0.7]), fold=1
    )
    assert "hla_allele" not in records[0]
    assert records[0][SPLITTER_COLUMN] == GNN_CV_SPLITTER


def test_oof_frame_from_a_full_cv_pass_satisfies_the_gate1_precondition():
    """End-to-end on the artifact schema: what train_gnn.py writes must clear Gate 1's
    splitter precondition. This is the join between the two files under repair."""
    from src.verify.promote_gnn import grouped_splitter_violation

    pool = _train_pool()
    y = pool["label"].to_numpy()
    rng = np.random.default_rng(0)

    rows: list[dict] = []
    for fold, (_train_idx, val_idx) in enumerate(build_cv_splits(pool, y, 5, 42), 1):
        rows.extend(
            build_oof_records(
                pool,
                val_idx,
                y[val_idx].astype(float),
                rng.uniform(0, 1, len(val_idx)),
                fold,
            )
        )
    oof_df = pd.DataFrame(rows)

    assert set(["peptide", "label", "gnn_oof_score", "fold", SPLITTER_COLUMN]).issubset(
        oof_df.columns
    )
    assert sorted(oof_df["fold"].unique().tolist()) == [1, 2, 3, 4, 5]
    assert grouped_splitter_violation(oof_df) is None
