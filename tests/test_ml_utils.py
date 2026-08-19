"""Tests for src/ml_utils.py.

Covers:
  make_stratification_key    shape, variation on inputs, None handling
  MultiStratifiedKFold       fold count, full coverage, no overlap,
                             fallback to label-only, label balance per fold
  pin_serial_scoring         bit-identity of parallel-fit + pinned-score against a
                             fully serial run (RF and the shipped XGB path);
                             XGBoost nthread invariance
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.ml_utils import (
    MultiStratifiedKFold,
    PeptideGroupedKFold,
    make_stratification_key,
    pin_serial_scoring,
)


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


# ---------------------------------------------------------------------------
# pin_serial_scoring
# ---------------------------------------------------------------------------
#
# n_jobs=-1 fitting is a production behaviour change (N11): trees are grown in
# parallel, which is invariant given random_state, but RandomForest's threaded
# predict_proba accumulates per-tree votes into a shared buffer in whatever
# order the worker threads finish, and float addition is not associative.
# pin_serial_scoring exists to force scoring back to single-threaded so the
# shipped artifact's predictions are reproducible. These tests bind the two
# empirical claims the adoption depended on, so a future change that breaks
# either is caught rather than silently shipped.


def _classification_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A fixed synthetic dataset sized to exercise real threaded fit/score,
    not a toy that finishes before any thread contention could occur."""
    X, y = make_classification(
        n_samples=1500,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_clusters_per_class=3,
        class_sep=0.5,
        random_state=42,
    )
    X_train, X_test = X[:1100], X[1100:]
    y_train = y[:1100]
    return X_train, y_train, X_test


def test_pin_serial_scoring_sets_n_jobs_to_one_and_returns_same_object() -> None:
    clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=0)
    returned = pin_serial_scoring(clf)
    assert returned is clf
    assert clf.n_jobs == 1


def test_pin_serial_scoring_leaves_models_without_n_jobs_unchanged() -> None:
    class NoNJobs:
        pass

    obj = NoNJobs()
    assert pin_serial_scoring(obj) is obj
    assert not hasattr(obj, "n_jobs")


def test_pin_serial_scoring_rf_bit_identical_to_fully_serial() -> None:
    """The N11 adoption claim: fit with n_jobs=-1, pin, predict -> identical to
    a fully serial (n_jobs=1 throughout) run. Exact equality, not np.allclose -
    the whole point is that no bit differs."""
    X_train, y_train, X_test = _classification_fixture()

    serial = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=1
    )
    serial.fit(X_train, y_train)
    serial_scores = serial.predict_proba(X_test)[:, 1]

    parallel_fit = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    )
    parallel_fit.fit(X_train, y_train)
    pin_serial_scoring(parallel_fit)
    pinned_scores = parallel_fit.predict_proba(X_test)[:, 1]

    assert np.array_equal(serial_scores, pinned_scores)


def test_pin_serial_scoring_rf_pinned_scoring_reproducible_across_runs() -> None:
    """Pinned scoring must be reproducible against itself, run to run - the
    property parallel scoring is not guaranteed to have."""
    X_train, y_train, X_test = _classification_fixture()
    runs = []
    for _ in range(3):
        clf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        )
        clf.fit(X_train, y_train)
        pin_serial_scoring(clf)
        runs.append(clf.predict_proba(X_test)[:, 1])
    assert np.array_equal(runs[0], runs[1])
    assert np.array_equal(runs[1], runs[2])


def test_pin_serial_scoring_pins_xgboost_nthread_via_set_params() -> None:
    """N11-b: the original patch only touched n_jobs, leaving XGBoost's
    nthread flip unpinned. Confirms the fix actually reaches nthread - plain
    setattr does not (nthread is a passthrough kwarg, invisible to hasattr
    until routed through get_params/set_params)."""
    clf = XGBClassifier(n_estimators=10, nthread=-1, random_state=0)
    assert clf.get_params()["nthread"] == -1
    pin_serial_scoring(clf)
    assert clf.get_params()["nthread"] == 1
    assert clf.get_params()["n_jobs"] == 1


def test_xgboost_nthread_does_not_change_predictions() -> None:
    """Binds the docstring's second claim: XGBoost's hist builder is
    thread-count invariant, so pin_serial_scoring's no-op on XGBClassifier
    (nthread, not n_jobs, controls its threading) is safe. If a future
    XGBoost upgrade breaks this invariance, this test is what catches it -
    not a docstring nobody re-runs."""
    X_train, y_train, X_test = _classification_fixture()

    def _fit_score(nthread: int) -> np.ndarray:
        clf = XGBClassifier(
            n_estimators=200,
            random_state=42,
            eval_metric="aucpr",
            objective="binary:logistic",
            nthread=nthread,
        )
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test)[:, 1]

    serial_scores = _fit_score(1)
    parallel_scores = _fit_score(-1)
    assert np.array_equal(serial_scores, parallel_scores)


def test_pin_serial_scoring_xgb_shipped_path_bit_identical_to_fully_serial() -> None:
    """The XGB half of the N11 claim, in the order train_models actually ships:
    fit with nthread=-1, pin, then predict. The invariance test above compares
    fit(-1)+predict(-1) against fit(1)+predict(1) and never routes through
    pin_serial_scoring - it stays green with the helper deleted outright - so
    what it binds is a property of XGBoost, not of this repo's combination."""
    X_train, y_train, X_test = _classification_fixture()

    def _shipped_clf(nthread: int) -> XGBClassifier:
        # train_models' xgb_kwargs, minus the data-derived scale_pos_weight
        # (it has no bearing on threading).
        return XGBClassifier(
            n_estimators=200,
            random_state=42,
            eval_metric="aucpr",
            objective="binary:logistic",
            nthread=nthread,
        )

    serial = _shipped_clf(1)
    serial.fit(X_train, y_train)
    serial_scores = serial.predict_proba(X_test)[:, 1]

    shipped = _shipped_clf(-1)
    shipped.fit(X_train, y_train)
    pin_serial_scoring(shipped)  # pinned AFTER fit, as train_models does

    # The pin has to reach an ALREADY-FITTED estimator; the set_params test
    # above only ever pins an unfitted one.
    assert shipped.get_params()["nthread"] == 1
    assert shipped.get_params()["n_jobs"] == 1
    # ...and has to reach the Booster, which is what joblib.dump pickles: an
    # unpinned booster carries nthread=-1 to every downstream consumer.
    # save_config reports generic_param values as strings.
    booster_cfg = json.loads(shipped.get_booster().save_config())
    assert booster_cfg["learner"]["generic_param"]["nthread"] == "1"

    assert np.array_equal(serial_scores, shipped.predict_proba(X_test)[:, 1])


def test_peptide_grouped_kfold_none_optional_args() -> None:
    # negative_origin/hla_alleles are optional; only peptides is mandatory.
    n_peptides = 30
    labels = pd.Series([i % 2 for i in range(n_peptides) for _ in range(3)])
    peptides = pd.Series([f"PEPTIDE{i:03d}" for i in range(n_peptides) for _ in range(3)])
    pgkf = PeptideGroupedKFold(n_splits=3, random_state=0)
    folds = list(pgkf.split(np.zeros((len(labels), 1)), labels, peptides=peptides))
    assert len(folds) == 3
