"""Tests for src/consensus_ensemble.run_consensus() — the CSV-level orchestrator."""
import pandas as pd
import pytest

from src.consensus_ensemble import run_consensus


def _write_merged(path, rows=None):
    if rows is None:
        rows = [
            {"peptide": "GILGFVFTL", "label": 1, "sestrav_score": 0.9, "prime_score": 0.85},
            {"peptide": "NLVPMVATV", "label": 0, "sestrav_score": 0.2, "prime_score": 0.15},
            {"peptide": "KLGGALQAK", "label": 1, "sestrav_score": 0.8, "prime_score": 0.75},
        ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_run_consensus_happy_path(tmp_path):
    merged = tmp_path / "merged.csv"
    out = tmp_path / "out.csv"
    _write_merged(merged)
    df = run_consensus(str(merged), str(out), ["sestrav_score", "prime_score"])
    assert out.exists()
    assert "TCR_Recognition_Propensity_Score" in df.columns
    assert len(df) == 3
    # Higher-scoring peptide should have higher consensus score
    scores = df.set_index("peptide")["TCR_Recognition_Propensity_Score"]
    assert scores["GILGFVFTL"] > scores["NLVPMVATV"]


def test_run_consensus_with_weights(tmp_path):
    merged = tmp_path / "merged.csv"
    out = tmp_path / "out.csv"
    _write_merged(merged)
    df = run_consensus(
        str(merged), str(out), ["sestrav_score", "prime_score"],
        weights={"sestrav_score": 2.0, "prime_score": 1.0},
    )
    assert "TCR_Recognition_Propensity_Score" in df.columns


def test_run_consensus_nan_imputation(tmp_path):
    merged = tmp_path / "merged.csv"
    out = tmp_path / "out.csv"
    rows = [
        {"peptide": "GILGFVFTL", "label": 1, "sestrav_score": 0.9, "prime_score": None},
        {"peptide": "NLVPMVATV", "label": 0, "sestrav_score": 0.2, "prime_score": 0.1},
    ]
    _write_merged(merged, rows)
    df = run_consensus(str(merged), str(out), ["sestrav_score", "prime_score"])
    assert df["TCR_Recognition_Propensity_Score"].notna().all()


def test_run_consensus_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_consensus(str(tmp_path / "nope.csv"), str(tmp_path / "out.csv"), ["sestrav_score"])


def test_run_consensus_missing_all_columns_raises(tmp_path):
    merged = tmp_path / "merged.csv"
    pd.DataFrame([{"peptide": "GILGFVFTL", "label": 1}]).to_csv(merged, index=False)
    with pytest.raises(ValueError, match="None of the target score columns"):
        run_consensus(str(merged), str(tmp_path / "out.csv"), ["sestrav_score"])


def test_run_consensus_creates_output_dir(tmp_path):
    merged = tmp_path / "merged.csv"
    out = tmp_path / "nested" / "dir" / "out.csv"
    _write_merged(merged)
    run_consensus(str(merged), str(out), ["sestrav_score", "prime_score"])
    assert out.exists()
