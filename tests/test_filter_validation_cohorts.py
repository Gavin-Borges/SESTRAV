"""Tests for scripts/filter_validation_cohorts.py training-reference loading.

The previous exists-skip loop treated a missing corpus as an empty peptide
set, so contamination filtering removed nothing. The required path is now
data/immunogenicity_dataset_v5.csv and a missing file must raise.
"""

import os

import pandas as pd
import pytest

from scripts.filter_validation_cohorts import (
    TRAINING_DATA_PATHS,
    load_training_peptides,
)


def _posix(path):
    return path.replace("\\", "/")


def test_training_data_paths_point_at_tracked_v5():
    assert TRAINING_DATA_PATHS, "TRAINING_DATA_PATHS must not be empty"
    posix = [_posix(p) for p in TRAINING_DATA_PATHS]
    assert any(p.endswith("data/immunogenicity_dataset_v5.csv") for p in posix)
    basenames = {os.path.basename(p) for p in TRAINING_DATA_PATHS}
    assert "immunogenicity_dataset.csv" not in basenames
    assert "immunogenicity_dataset_v4.csv" not in basenames


def test_load_training_peptides_missing_reference_is_fatal(tmp_path, monkeypatch):
    missing = tmp_path / "no_such_corpus.csv"
    monkeypatch.setattr(
        "scripts.filter_validation_cohorts.TRAINING_DATA_PATHS",
        [str(missing)],
    )
    with pytest.raises(FileNotFoundError, match="Required training reference corpus missing"):
        load_training_peptides()


def test_load_training_peptides_reads_peptide_column(tmp_path, monkeypatch):
    csv_path = tmp_path / "ref.csv"
    pd.DataFrame({"peptide": ["YLQPRTFLL", "ylqprtfll", "GILGFVFTL"]}).to_csv(
        csv_path, index=False
    )
    monkeypatch.setattr(
        "scripts.filter_validation_cohorts.TRAINING_DATA_PATHS",
        [str(csv_path)],
    )
    peptides = load_training_peptides()
    assert set(peptides) == {"YLQPRTFLL", "GILGFVFTL"}


def test_load_training_peptides_rejects_missing_peptide_column(tmp_path, monkeypatch):
    csv_path = tmp_path / "ref.csv"
    pd.DataFrame({"sequence": ["YLQPRTFLL"]}).to_csv(csv_path, index=False)
    monkeypatch.setattr(
        "scripts.filter_validation_cohorts.TRAINING_DATA_PATHS",
        [str(csv_path)],
    )
    with pytest.raises(ValueError, match="has no peptide column"):
        load_training_peptides()
