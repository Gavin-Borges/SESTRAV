"""Tests for scripts/fit_final_per_virus_calibrators.py's core fitting logic.

Complements tests/test_artifact_guard_contract.py (which covers the
overwrite-guard contract this module shares with every other artifact-writing
script) - these focus on fit_final_calibrators itself: which viruses get a
calibrator, that low-n viruses are skipped rather than silently mis-fit, and
that the written filename matches what functions.stage4_immunogenicity_scoring
will later look up (a promoted calibrator that the resolver can never find is
a worse defect than not promoting one at all).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from functions.stage4_immunogenicity_scoring import _sanitize_name as stage4_sanitize
from scripts.fit_final_per_virus_calibrators import (
    MIN_ROWS_TO_FIT,
    _sanitize_name,
    fit_final_calibrators,
)


def _oof_frame(virus_counts: dict[str, int], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for virus, n in virus_counts.items():
        labels = rng.integers(0, 2, size=n)
        scores = rng.uniform(0, 1, size=n)
        for label, score in zip(labels, scores):
            rows.append({"virus": virus, "label": label, "score": score})
    return pd.DataFrame(rows)


def test_sanitize_name_matches_stage4s_sanitizer():
    """The writer and the reader must agree on every target-virus name, or a
    promoted calibrator silently becomes unreachable."""
    from scripts.fit_calibrator import TARGET_VIRUSES

    for virus in TARGET_VIRUSES:
        assert _sanitize_name(virus) == stage4_sanitize(virus)


def test_fit_final_calibrators_fits_only_viruses_with_enough_rows():
    df = _oof_frame({"SARS-CoV-2": MIN_ROWS_TO_FIT + 10, "HPV": MIN_ROWS_TO_FIT - 1})
    result = fit_final_calibrators(df)
    assert set(result) == {"SARS-CoV-2"}
    assert result["SARS-CoV-2"]["n"] == MIN_ROWS_TO_FIT + 10


def test_fit_final_calibrators_skips_viruses_below_the_threshold_without_raising():
    df = _oof_frame({"HPV": 5})
    result = fit_final_calibrators(df)
    assert result == {}


def test_fit_final_calibrators_ignores_non_target_viruses():
    """A virus not in TARGET_VIRUSES (e.g. a decoy-only or off-panel row) must
    never get a calibrator, however many rows it has - only the nine target
    viruses are eligible."""
    df = _oof_frame({"Orthopoxvirus": MIN_ROWS_TO_FIT + 50})
    result = fit_final_calibrators(df)
    assert result == {}


def test_fit_final_calibrators_output_is_a_working_calibrator():
    df = _oof_frame({"CMV": MIN_ROWS_TO_FIT + 20})
    result = fit_final_calibrators(df)
    calibrator = result["CMV"]["calibrator"]
    predicted = calibrator.predict(np.array([0.0, 0.5, 1.0]))
    assert predicted.shape == (3,)
    assert np.all((predicted >= 0.0) & (predicted <= 1.0))


def test_fit_final_calibrators_reports_ece_and_brier_for_each_fitted_virus():
    df = _oof_frame({"DENV": MIN_ROWS_TO_FIT + 5})
    result = fit_final_calibrators(df)
    info = result["DENV"]
    for key in ("ece_raw", "ece_cal", "brier_raw", "brier_cal"):
        assert key in info
        assert np.isfinite(info[key])
