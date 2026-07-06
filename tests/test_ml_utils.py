"""Tests for src/ml_utils.py.

Covers:
  make_stratification_key    shape, variation on inputs, None handling
  MultiStratifiedKFold       fold count, full coverage, no overlap,
                             fallback to label-only, label balance per fold
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml_utils import MultiStratifiedKFold, make_stratification_key


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


def test_multi_stratified_kfold_fallback_on_sparse_strata() -> None:
    # Use n=10 with high-cardinality composite strata to force fallback.
    n = 10
    labels = pd.Series([1, 0] * 5)
    origins = pd.Series([f"origin_{i}" for i in range(n)])  # all unique -> sparse
    alleles = pd.Series([f"HLA-A*0{i}:01" for i in range(n)])
    peptides = pd.Series(["GILGFVFTL"] * n)
    mskf = MultiStratifiedKFold(n_splits=2, min_stratum_size=5, random_state=0)
    folds = list(mskf.split(np.zeros((n, 1)), labels, origins, alleles, peptides))
    assert len(folds) == 2


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
