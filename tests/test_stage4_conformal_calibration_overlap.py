"""Stage 4 flags every row whose peptide the conformal calibrator was calibrated on.

src/conformal.py's predict_interval is valid only for rows whose group is absent from
the calibration data and leaves that check to the caller; the groups are peptides.
Stage 4 passed every scored row with no check, and its default panels are the training
viruses, so a share of rows always overlaps: 4,176 of the 94,572 distinct peptides the
eight configured panels generate are in the shipped calibrator's calibration set. The
interval stays; the new in_calibration_set column says which rows it does not cover.

These drive the real _apply_conformal with a real fitted CrossVennAbers, replacing only
the file resolution and the verified load.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import functions.stage4_immunogenicity_scoring as s4
from src.conformal import fit_venn_abers


def _calibrator(peptides):
    rng = np.random.default_rng(0)
    n = len(peptides)
    scores = rng.uniform(0, 1, n)
    labels = (scores + rng.normal(0, 0.2, n) > 0.5).astype(int)
    return fit_venn_abers(scores, labels, np.asarray(peptides), n_folds=5, random_state=0)


def _apply(monkeypatch, frame, model):
    monkeypatch.setattr(s4, "_resolve_conformal_path", lambda *a, **k: "calibrator.joblib")
    monkeypatch.setattr(s4, "load_verified_joblib", lambda *a, **k: model)
    return s4._apply_conformal(frame, model_dir=None, conformal_path=None, freeze_mode=False)


def test_rows_whose_peptide_was_calibrated_on_are_flagged(monkeypatch):
    calibrated = [f"CAL{i:04d}" for i in range(60)]
    model = _calibrator(calibrated)
    frame = pd.DataFrame(
        {
            "peptide": ["CAL0003", "NEW0001", "CAL0059", "NEW0002"],
            "immunogenicity_score": [0.2, 0.4, 0.6, 0.8],
        }
    )

    assert _apply(monkeypatch, frame, model) is True

    assert frame["in_calibration_set"].tolist() == [True, False, True, False]
    # The intervals themselves are unchanged in kind: every row still gets one.
    assert frame[["lower_bound", "upper_bound", "interval_width"]].notna().all().all()


def test_the_flag_reads_the_loaded_calibrators_own_groups(monkeypatch):
    """Two calibrators over different peptides flag different rows for the same frame."""
    frame_a = pd.DataFrame({"peptide": ["AAA", "BBB"], "immunogenicity_score": [0.3, 0.7]})
    frame_b = frame_a.copy()
    one = _calibrator(["AAA"] + [f"X{i}" for i in range(40)])
    two = _calibrator(["BBB"] + [f"Y{i}" for i in range(40)])

    _apply(monkeypatch, frame_a, one)
    _apply(monkeypatch, frame_b, two)

    assert frame_a["in_calibration_set"].tolist() == [True, False]
    assert frame_b["in_calibration_set"].tolist() == [False, True]


def test_no_overlap_flags_nothing(monkeypatch):
    model = _calibrator([f"CAL{i:04d}" for i in range(30)])
    frame = pd.DataFrame({"peptide": ["NEW1", "NEW2"], "immunogenicity_score": [0.1, 0.9]})

    _apply(monkeypatch, frame, model)

    assert frame["in_calibration_set"].tolist() == [False, False]
