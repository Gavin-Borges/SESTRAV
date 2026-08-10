"""Tests for src/ml_utils.py.

Covers:
  make_stratification_key    shape, variation on inputs, None handling
  MultiStratifiedKFold       fold count, full coverage, no overlap,
                             fallback to label-only, label balance per fold
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml_utils import MultiStratifiedKFold, PeptideGroupedKFold, make_stratification_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_series(n: int = 100) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (labels, negative_origin, hla_alleles, peptides) for n samples."""
    rng = np.random.default_rng(0)
    labels = pd.Series([1] * (n // 2) + [0] * (n - n // 2))
    origins = pd.Series(
        ["tested_negative"] * (n // 4) + ["self_proteome_decoy"] * (n // 4) + [None] * (n - n // 2)
    )
    alleles = pd.Series(["HLA-A*02:01"] * (n // 2) + ["HLA-B*07:02"] * (n - n // 2))
    peptides = pd.Series(["GILGFVFTL"] * (n // 2) + ["GILGFVFTLA"] * (n - n // 2))
    return labels, origins, alleles, peptides


def _grouped_fixture(
    n_peptides: int = 40, rows_per_peptide: int = 3
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Peptides repeated across rows - the shape the v5 corpus actually has.

    Every feature_mode=31 feature is a pure function of the peptide string,
    so this shape (16,360 unique peptides over 35,597 v5 rows) is exactly why
    an ungrouped splitter leaks (docs/claims_register.md D15).
    """
    peptides, labels, origins, alleles = [], [], [], []
    for i in range(n_peptides):
        for _ in range(rows_per_peptide):
            peptides.append(f"PEPTIDE{i:03d}")
            labels.append(i % 2)
            origins.append("tested_negative" if i % 2 == 0 else None)
            alleles.append("HLA-A*02:01" if i % 3 else "HLA-B*07:02")
    return (
        pd.Series(labels),
        pd.Series(origins),
        pd.Series(alleles),
        pd.Series(peptides),
    )


# ---------------------------------------------------------------------------
# make_stratification_key
# ---------------------------------------------------------------------------


def test_make_stratification_key_length() -> None:
    labels, origins, alleles, peptides = _base_series(80)
    key = make_stratification_key(labels, origins, alleles, peptides)
    assert len(key) == 80


def test_make_stratification_key_varies_on_label() -> None:
    labels = pd.Series([1, 0])
    key = make_stratification_key(labels)
    assert key.iloc[0] != key.iloc[1]


def test_make_stratification_key_varies_on_origin() -> None:
    labels = pd.Series([0, 0])
    origins = pd.Series(["tested_negative", "self_proteome_decoy"])
    key = make_stratification_key(labels, negative_origin=origins)
    assert key.iloc[0] != key.iloc[1]


def test_make_stratification_key_varies_on_length() -> None:
    labels = pd.Series([1, 1])
    peptides = pd.Series(["GILGFVFTL", "GILGFVFTLA"])  # 9mer vs 10mer
    key = make_stratification_key(labels, peptides=peptides)
    assert key.iloc[0] != key.iloc[1]


def test_make_stratification_key_none_inputs_no_error() -> None:
    labels = pd.Series([1, 0, 1, 0])
    key = make_stratification_key(labels, None, None, None)
    assert len(key) == 4


def test_make_stratification_key_missing_allele_handled() -> None:
    labels = pd.Series([1, 0])
    alleles = pd.Series([None, "HLA-A*02:01"])
    key = make_stratification_key(labels, hla_alleles=alleles)
    assert len(key) == 2
    assert key.notna().all()


def test_make_stratification_key_dtype_is_string() -> None:
    labels, origins, alleles, peptides = _base_series(20)
    key = make_stratification_key(labels, origins, alleles, peptides)
    assert pd.api.types.is_string_dtype(key)


def test_make_stratification_key_iedb_api_bins_as_real() -> None:
    # "iedb_api" is a genuine assay-confirmed negative origin (same pairing as
    # scripts/build_dataset_v5.py's _real_neg_origins), so it must land in the
    # same origin bin as "tested_negative", not fall through to "unk".
    labels = pd.Series([0, 0])
    origins = pd.Series(["tested_negative", "iedb_api"])
    key = make_stratification_key(labels, negative_origin=origins)
    assert key.iloc[0] == key.iloc[1]


# ---------------------------------------------------------------------------
# MultiStratifiedKFold - basic properties
# ---------------------------------------------------------------------------


def test_multi_stratified_kfold_n_splits() -> None:
    mskf = MultiStratifiedKFold(n_splits=5)
    assert mskf.get_n_splits() == 5


def test_multi_stratified_kfold_produces_correct_n_folds() -> None:
    labels, origins, alleles, peptides = _base_series(100)
    mskf = MultiStratifiedKFold(n_splits=5, random_state=42)
    folds = list(mskf.split(np.zeros((100, 1)), labels, origins, alleles, peptides))
    assert len(folds) == 5


def test_multi_stratified_kfold_covers_all_samples() -> None:
    labels, origins, alleles, peptides = _base_series(100)
    mskf = MultiStratifiedKFold(n_splits=5, random_state=42)
    all_test_indices: list[int] = []
    for _, test_idx in mskf.split(np.zeros((100, 1)), labels, origins, alleles, peptides):
        all_test_indices.extend(test_idx.tolist())
    assert sorted(all_test_indices) == list(range(100))


def test_multi_stratified_kfold_no_overlap_in_fold() -> None:
    labels, origins, alleles, peptides = _base_series(100)
    mskf = MultiStratifiedKFold(n_splits=5, random_state=42)
    for train_idx, test_idx in mskf.split(np.zeros((100, 1)), labels, origins, alleles, peptides):
        assert len(set(train_idx) & set(test_idx)) == 0


def test_multi_stratified_kfold_both_classes_in_each_fold() -> None:
    labels, origins, alleles, peptides = _base_series(100)
    y_arr = labels.to_numpy()
    mskf = MultiStratifiedKFold(n_splits=5, random_state=42)
    for _, test_idx in mskf.split(np.zeros((100, 1)), labels, origins, alleles, peptides):
        classes_in_test = np.unique(y_arr[test_idx])
        assert 0 in classes_in_test
        assert 1 in classes_in_test


def test_multi_stratified_kfold_train_larger_than_test() -> None:
    labels, origins, alleles, peptides = _base_series(100)
    mskf = MultiStratifiedKFold(n_splits=5)
    for train_idx, test_idx in mskf.split(np.zeros((100, 1)), labels, origins, alleles, peptides):
        assert len(train_idx) > len(test_idx)


# ---------------------------------------------------------------------------
# MultiStratifiedKFold - fallback behavior
# ---------------------------------------------------------------------------


def test_multi_stratified_kfold_coarsens_on_sparse_strata() -> None:
    # High-cardinality supertype (unrecognized alleles) makes the full composite
    # too sparse; the ladder should drop to a coarser rung rather than silently
    # collapsing straight to label-only, and record which rung it used.
    n = 10
    labels = pd.Series([1, 0] * 5)
    origins = pd.Series([f"origin_{i}" for i in range(n)])  # all unique -> bins to "unk"
    alleles = pd.Series([f"HLA-A*0{i}:01" for i in range(n)])  # sparse supertype bins
    peptides = pd.Series(["GILGFVFTL"] * n)  # constant -> bins to "9mer"
    mskf = MultiStratifiedKFold(n_splits=2, min_stratum_size=5, random_state=0)
    folds = list(mskf.split(np.zeros((n, 1)), labels, origins, alleles, peptides))
    assert len(folds) == 2
    assert mskf.stratification_components_ is not None
    assert mskf.stratification_components_ != ("label", "origin", "supertype", "length")
    assert "supertype" not in mskf.stratification_components_


def test_multi_stratified_kfold_records_full_resolution_when_not_sparse() -> None:
    labels, origins, alleles, peptides = _base_series(100)
    mskf = MultiStratifiedKFold(n_splits=5, random_state=42)
    list(mskf.split(np.zeros((100, 1)), labels, origins, alleles, peptides))
    assert mskf.stratification_components_ == ("label", "origin", "supertype", "length")


def test_multi_stratified_kfold_raises_when_label_only_still_sparse() -> None:
    # Even the coarsest rung (label-only) has a minority class of 1 row here,
    # below min_stratum_size=5 - there is nowhere left to coarsen to.
    labels = pd.Series([1, 1, 0])
    mskf = MultiStratifiedKFold(n_splits=2, min_stratum_size=5, random_state=0)
    with pytest.raises(ValueError, match="Cannot stratify"):
        list(mskf.split(np.zeros((3, 1)), labels))


def test_multi_stratified_kfold_none_optional_args() -> None:
    n = 60
    labels = pd.Series([1] * 30 + [0] * 30)
    mskf = MultiStratifiedKFold(n_splits=3, random_state=0)
    folds = list(mskf.split(np.zeros((n, 1)), labels))
    assert len(folds) == 3


def test_multi_stratified_kfold_numpy_y_input() -> None:
    n = 60
    y_arr = np.array([1] * 30 + [0] * 30)
    mskf = MultiStratifiedKFold(n_splits=3, random_state=0)
    folds = list(mskf.split(np.zeros((n, 1)), y_arr))
    assert len(folds) == 3


def test_multi_stratified_kfold_shuffle_false_does_not_raise() -> None:
    # sklearn's StratifiedKFold raises if random_state is set while shuffle=False;
    # split() must not forward random_state in that case.
    labels, origins, alleles, peptides = _base_series(100)
    mskf = MultiStratifiedKFold(n_splits=5, shuffle=False, random_state=42)
    folds = list(mskf.split(np.zeros((100, 1)), labels, origins, alleles, peptides))
    assert len(folds) == 5


def test_multi_stratified_kfold_deterministic() -> None:
    labels, origins, alleles, peptides = _base_series(80)
    mskf_a = MultiStratifiedKFold(n_splits=4, random_state=7)
    mskf_b = MultiStratifiedKFold(n_splits=4, random_state=7)
    folds_a = [
        (tr.tolist(), te.tolist())
        for tr, te in mskf_a.split(np.zeros((80, 1)), labels, origins, alleles, peptides)
    ]
    folds_b = [
        (tr.tolist(), te.tolist())
        for tr, te in mskf_b.split(np.zeros((80, 1)), labels, origins, alleles, peptides)
    ]
    assert folds_a == folds_b


# ---------------------------------------------------------------------------
# MultiStratifiedKFold - negative control: it does NOT guarantee peptide
# disjointness. Without this, a fold-disjointness test on PeptideGroupedKFold
# alone could pass merely because a fixture happens to have no repeats.
# ---------------------------------------------------------------------------


def test_multi_stratified_kfold_can_leak_peptides_across_folds() -> None:
    labels, origins, alleles, peptides = _grouped_fixture()
    pep = peptides.to_numpy()
    mskf = MultiStratifiedKFold(n_splits=5, random_state=42)
    leaked_any = False
    for train_idx, test_idx in mskf.split(np.zeros((len(pep), 1)), labels, origins, alleles, peptides):
        if set(pep[train_idx]) & set(pep[test_idx]):
            leaked_any = True
            break
    assert leaked_any


# ---------------------------------------------------------------------------
# PeptideGroupedKFold
# ---------------------------------------------------------------------------


def test_peptide_grouped_kfold_n_splits() -> None:
    pgkf = PeptideGroupedKFold(n_splits=5)
    assert pgkf.get_n_splits() == 5


def test_peptide_grouped_kfold_requires_peptides() -> None:
    labels, origins, alleles, _peptides = _grouped_fixture()
    pgkf = PeptideGroupedKFold(n_splits=5, random_state=42)
    with pytest.raises(ValueError, match="requires peptides"):
        list(pgkf.split(np.zeros((len(labels), 1)), labels, origins, alleles))


def test_peptide_grouped_kfold_rejects_mismatched_length() -> None:
    labels, origins, alleles, peptides = _grouped_fixture()
    pgkf = PeptideGroupedKFold(n_splits=5, random_state=42)
    with pytest.raises(ValueError, match="a grouped split requires"):
        list(
            pgkf.split(
                np.zeros((len(labels), 1)), labels, origins, alleles, peptides.iloc[:-1]
            )
        )


def test_peptide_grouped_kfold_produces_correct_n_folds() -> None:
    labels, origins, alleles, peptides = _grouped_fixture()
    pgkf = PeptideGroupedKFold(n_splits=5, random_state=42)
    folds = list(pgkf.split(np.zeros((len(labels), 1)), labels, origins, alleles, peptides))
    assert len(folds) == 5


def test_peptide_grouped_kfold_covers_all_samples() -> None:
    labels, origins, alleles, peptides = _grouped_fixture()
    n = len(labels)
    pgkf = PeptideGroupedKFold(n_splits=5, random_state=42)
    all_test_indices: list[int] = []
    for _, test_idx in pgkf.split(np.zeros((n, 1)), labels, origins, alleles, peptides):
        all_test_indices.extend(test_idx.tolist())
    assert sorted(all_test_indices) == list(range(n))


def test_peptide_grouped_kfold_no_index_overlap_in_fold() -> None:
    labels, origins, alleles, peptides = _grouped_fixture()
    pgkf = PeptideGroupedKFold(n_splits=5, random_state=42)
    for train_idx, test_idx in pgkf.split(
        np.zeros((len(labels), 1)), labels, origins, alleles, peptides
    ):
        assert len(set(train_idx) & set(test_idx)) == 0


def test_peptide_grouped_kfold_folds_are_peptide_disjoint() -> None:
    # THE fold-disjointness test - closes the gap named in
    # docs/proposals/2026_feature_upgrade_roadmap.md Phase 0 step 3.
    labels, origins, alleles, peptides = _grouped_fixture()
    pep = peptides.to_numpy()
    pgkf = PeptideGroupedKFold(n_splits=5, random_state=42)
    for train_idx, test_idx in pgkf.split(
        np.zeros((len(pep), 1)), labels, origins, alleles, peptides
    ):
        assert not (set(pep[train_idx]) & set(pep[test_idx]))


def test_peptide_grouped_kfold_train_larger_than_test() -> None:
    labels, origins, alleles, peptides = _grouped_fixture()
    pgkf = PeptideGroupedKFold(n_splits=5, random_state=42)
    for train_idx, test_idx in pgkf.split(
        np.zeros((len(labels), 1)), labels, origins, alleles, peptides
    ):
        assert len(train_idx) > len(test_idx)


def test_peptide_grouped_kfold_records_full_resolution_when_not_sparse() -> None:
    labels, origins, alleles, peptides = _grouped_fixture()
    pgkf = PeptideGroupedKFold(n_splits=5, random_state=42)
    list(pgkf.split(np.zeros((len(labels), 1)), labels, origins, alleles, peptides))
    assert pgkf.stratification_components_ == ("label", "origin", "supertype", "length")


def test_peptide_grouped_kfold_shuffle_false_does_not_raise() -> None:
    labels, origins, alleles, peptides = _grouped_fixture()
    pgkf = PeptideGroupedKFold(n_splits=5, shuffle=False, random_state=42)
    folds = list(pgkf.split(np.zeros((len(labels), 1)), labels, origins, alleles, peptides))
    assert len(folds) == 5


def test_peptide_grouped_kfold_deterministic() -> None:
    labels, origins, alleles, peptides = _grouped_fixture()
    n = len(labels)
    pgkf_a = PeptideGroupedKFold(n_splits=5, random_state=7)
    pgkf_b = PeptideGroupedKFold(n_splits=5, random_state=7)
    folds_a = [
        (tr.tolist(), te.tolist())
        for tr, te in pgkf_a.split(np.zeros((n, 1)), labels, origins, alleles, peptides)
    ]
    folds_b = [
        (tr.tolist(), te.tolist())
        for tr, te in pgkf_b.split(np.zeros((n, 1)), labels, origins, alleles, peptides)
    ]
    assert folds_a == folds_b


def test_peptide_grouped_kfold_none_optional_args() -> None:
    # negative_origin/hla_alleles are optional; only peptides is mandatory.
    n_peptides = 30
    labels = pd.Series([i % 2 for i in range(n_peptides) for _ in range(3)])
    peptides = pd.Series([f"PEPTIDE{i:03d}" for i in range(n_peptides) for _ in range(3)])
    pgkf = PeptideGroupedKFold(n_splits=3, random_state=0)
    folds = list(pgkf.split(np.zeros((len(labels), 1)), labels, peptides=peptides))
    assert len(folds) == 3
