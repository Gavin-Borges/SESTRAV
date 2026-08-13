"""Tests for the monotonicity control in scripts/fit_per_virus_calibrator.py.

The script's originally pre-registered control B5 asserted that isotonic
calibration leaves within-virus AUC-ROC unchanged "by construction". That is
false: isotonic is monotone NON-decreasing, so it never inverts a pair but it
does collapse distinct scores into ties, and AUC scores a tied pair at 0.5.
B5 was measured false and retired; B5' (zero strictly discordant pairs) replaced
it. These tests pin both halves, so the retired belief cannot quietly return.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from scripts.fit_per_virus_calibrator import count_rank_inversions


def test_monotone_transform_has_no_inversions():
    raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert count_rank_inversions(raw, raw * 2.0) == 0


def test_ties_are_not_inversions():
    """A step function is the normal isotonic output and must not be flagged."""
    raw = np.array([0.1, 0.2, 0.3, 0.4])
    assert count_rank_inversions(raw, np.array([0.0, 0.0, 1.0, 1.0])) == 0


def test_a_real_inversion_is_detected():
    """The control must be capable of failing, not merely of returning 0."""
    raw = np.array([0.1, 0.2, 0.3])
    assert count_rank_inversions(raw, np.array([0.0, 1.0, 0.5])) == 1


def test_unsorted_input_is_handled():
    raw = np.array([0.5, 0.1, 0.3])
    assert count_rank_inversions(raw, np.array([1.0, 0.0, 0.5])) == 0
    assert count_rank_inversions(raw, np.array([0.0, 1.0, 0.5])) == 2


def test_isotonic_never_inverts_but_does_move_auc():
    """The measured fact that retired B5, pinned as a regression test.

    If a future change makes this assertion fail, the claim "isotonic leaves the
    rank ordering unchanged so AUC is fixed" would need re-examining rather than
    the test being deleted. As written it encodes the truth: zero inversions AND
    a non-zero AUC shift, caused purely by tie collapse.
    """
    rng = np.random.default_rng(0)
    labels = np.concatenate([np.ones(60, dtype=int), np.zeros(60, dtype=int)])
    raw = np.concatenate(
        [rng.uniform(0.4, 1.0, 60), rng.uniform(0.0, 0.6, 60)]
    )

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, labels)
    calibrated = iso.predict(raw)

    assert count_rank_inversions(raw, calibrated) == 0
    assert len(np.unique(calibrated)) < len(np.unique(raw)), (
        "isotonic must collapse distinct scores into ties; without ties there is "
        "nothing for this test to demonstrate"
    )
    assert roc_auc_score(labels, calibrated) != pytest.approx(
        roc_auc_score(labels, raw), abs=1e-9
    ), "AUC was expected to move under tie collapse; the retired B5 premise assumed it could not"
