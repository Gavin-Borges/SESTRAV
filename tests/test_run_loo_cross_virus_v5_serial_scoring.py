"""The leave-one-virus-out benchmark scores on one thread.

scripts/run_loo_cross_virus_v5.py fits its RandomForest with n_jobs=-1 and used to
score with the same setting, so predict_proba added per-tree probabilities in
thread-completion order and one seed's metrics differed between runs. _fit_rf now
returns the forest through pin_serial_scoring, the helper train_models already uses
(tests/test_ml_utils.py binds the helper's own claims). These tests bind the
wiring: the fit stays parallel, the returned forest scores serially, and its scores
are bit-identical to a fully serial fit.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from scripts import run_loo_cross_virus_v5 as loo_v5


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=1500,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_clusters_per_class=3,
        class_sep=0.5,
        random_state=42,
    )
    return X[:1100], y[:1100], X[1100:]


def test_fit_rf_fits_in_parallel_and_returns_a_forest_that_scores_on_one_thread() -> None:
    assert loo_v5.RF_PARAMS["n_jobs"] == -1, "fitting is meant to use every core"
    X_train, y_train, _ = _fixture()
    clf = loo_v5._fit_rf(X_train, y_train)
    assert isinstance(clf, RandomForestClassifier)
    assert clf.get_params()["n_jobs"] == 1


def test_fit_rf_scores_bit_identical_to_a_fully_serial_fit() -> None:
    X_train, y_train, X_test = _fixture()
    serial = RandomForestClassifier(**{**loo_v5.RF_PARAMS, "n_jobs": 1})
    serial.fit(X_train, y_train)
    pinned = loo_v5._fit_rf(X_train, y_train)
    assert np.array_equal(serial.predict_proba(X_test)[:, 1], pinned.predict_proba(X_test)[:, 1])
