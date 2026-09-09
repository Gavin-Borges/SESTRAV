"""Tests for Cross Venn-Abers conformal prediction integration in Stage 4.

Validates that:
1. Stage 4 scoring produces lower_bound, upper_bound, and interval_width columns.
2. Output values are in [0, 1] with lower_bound <= upper_bound for every row.
3. Interval width equals upper_bound - lower_bound.
4. Monotonicity of lower_bound tracks rank order.
5. Disabling conformal via flag omits interval columns.
6. Explicit nonexistent conformal path under freeze_mode raises FileNotFoundError.
7. Coverage validation artifact models/v5/conformal_coverage_mode31.json conforms
   to schema and checksum manifest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from functions.stage4_immunogenicity_scoring import score_immunogenicity
from src.artifact_integrity import verify_artifact_checksum
from src.cli import main as cli_main
from src.features import FEATURE_COLUMNS_31


def _sample_feature_frame(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Build a deterministic DataFrame matching FEATURE_COLUMNS_31 layout."""
    rng = np.random.default_rng(seed)
    data: dict[str, Any] = {c: rng.uniform(0.0, 1.0, n) for c in FEATURE_COLUMNS_31}
    data["peptide"] = [f"PEPTIDE{i:03d}" for i in range(n)]
    return pd.DataFrame(data)


def test_conformal_coverage_artifact_and_checksum() -> None:
    """Verify the mode-31 conformal coverage artifact exists and matches checksum."""
    coverage_path = Path("models/v5/conformal_coverage_mode31.json")
    calibrator_path = Path("models/v5/conformal_calibrator.joblib")
    assert coverage_path.is_file(), "Coverage artifact not found"
    assert calibrator_path.is_file(), "Calibrator artifact not found"

    # Must pass checksum verification against manifest
    assert verify_artifact_checksum(coverage_path, required=True)
    assert verify_artifact_checksum(calibrator_path, required=True)

    with open(coverage_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["method"] == "RandomForest"
    assert data["feature_mode"] == 31
    assert data["n_samples"] == 35555
    assert data["n_groups"] == 16344
    assert data["n_folds"] == 5
    assert data["metrics"]["ordering_violations"] == 0
    assert data["metrics"]["calibration_in_large_gap"] < 0.001
    assert 0.0 < data["metrics"]["mean_interval_width"] < 0.05
    assert "validity_scope" in data


def test_stage4_applies_conformal_when_calibrator_present(tmp_path: Path) -> None:
    """End-to-end test: score_immunogenicity produces conformal columns."""
    df = _sample_feature_frame(n=25)
    model_path = "models/v5/rf_31feature_integrated.joblib"
    if not os.path.isfile(model_path):
        pytest.skip(f"Trained model {model_path} not found")

    out_dir = tmp_path / "results"
    ranked_df, _ = score_immunogenicity(
        features_df=df,
        proteome_id="TEST_PROTEOME",
        model_path=model_path,
        conformal=True,
        output_dir=str(out_dir),
    )

    # Columns present in DataFrame
    assert "lower_bound" in ranked_df.columns
    assert "upper_bound" in ranked_df.columns
    assert "interval_width" in ranked_df.columns

    # Check written CSV carries columns
    csv_path = out_dir / "TEST_PROTEOME_ranked.csv"
    assert csv_path.is_file()
    saved_df = pd.read_csv(csv_path)
    assert "lower_bound" in saved_df.columns
    assert "upper_bound" in saved_df.columns
    assert "interval_width" in saved_df.columns

    # Numerical invariants
    lower = ranked_df["lower_bound"].to_numpy(dtype=float)
    upper = ranked_df["upper_bound"].to_numpy(dtype=float)
    width = ranked_df["interval_width"].to_numpy(dtype=float)

    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert np.all(lower >= 0.0) and np.all(upper <= 1.0)
    assert np.all(lower <= upper), "lower_bound must be <= upper_bound"
    assert np.allclose(width, upper - lower, atol=1e-12)

    # Monotonicity check: ranked_df is sorted descending by score,
    # so lower_bound must be monotonically non-increasing.
    scores = ranked_df["immunogenicity_score"].to_numpy(dtype=float)
    assert (np.diff(scores) <= 1e-9).all(), "Scores must be descending"
    assert (np.diff(lower) <= 1e-9).all(), "Lower bound must track score rank order"


def test_stage4_conformal_disabled_omits_columns(tmp_path: Path) -> None:
    """When conformal=False, interval columns are omitted."""
    df = _sample_feature_frame(n=10)
    model_path = "models/v5/rf_31feature_integrated.joblib"
    if not os.path.isfile(model_path):
        pytest.skip(f"Trained model {model_path} not found")

    out_dir = tmp_path / "results"
    ranked_df, _ = score_immunogenicity(
        features_df=df,
        proteome_id="NO_CONFORMAL",
        model_path=model_path,
        conformal=False,
        output_dir=str(out_dir),
    )

    assert "lower_bound" not in ranked_df.columns
    assert "upper_bound" not in ranked_df.columns
    assert "interval_width" not in ranked_df.columns


def test_stage4_conformal_missing_explicit_path_freeze_mode(tmp_path: Path) -> None:
    """Missing explicit conformal_path under freeze_mode must raise FileNotFoundError."""
    df = _sample_feature_frame(n=5)
    model_path = "models/v5/rf_31feature_integrated.joblib"
    if not os.path.isfile(model_path):
        pytest.skip(f"Trained model {model_path} not found")

    with pytest.raises(FileNotFoundError, match="Conformal calibrator artifact not found"):
        score_immunogenicity(
            features_df=df,
            proteome_id="MISSING_PATH",
            model_path=model_path,
            conformal=True,
            conformal_path="nonexistent_calibrator.joblib",
            freeze_mode=True,
            output_dir=str(tmp_path),
        )


def test_cli_predict_parser_conformal_flags(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify predict CLI subcommand parser registers conformal flags."""
    # Calling cli_main(["predict", "--help"]) should exit with code 0 and describe flags
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["predict", "--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "--conformal" in captured.out
    assert "--conformal-calibrator" in captured.out
