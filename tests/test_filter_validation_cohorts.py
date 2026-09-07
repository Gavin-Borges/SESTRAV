"""Tests for scripts/filter_validation_cohorts.py contamination filtering.

Two separate defects are covered here, and it is worth keeping them distinct
because an earlier description of this work conflated them.

1. The previous exists-skip loop treated a missing corpus as an empty peptide
   set. That was NOT silent: main() turned the empty list into sys.exit(1), so
   the script refused to run. Raising in load_training_peptides is a usability
   fix, naming the real missing path at the point of the read. The required
   path is now data/immunogenicity_dataset_v5.csv.

2. filter_bidirectional_overlap DID have a silent failure, and it is the
   dangerous one: an empty training set made it return eval_df UNFILTERED,
   producing a cohort that was never contamination-checked but is
   indistinguishable downstream from one that was. That now raises.
"""

import os

import pandas as pd
import pytest

from scripts.filter_validation_cohorts import (
    TRAINING_DATA_PATHS,
    filter_bidirectional_overlap,
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


def test_filter_refuses_an_empty_training_set_instead_of_passing_through():
    # The dangerous branch. Before this guard, an empty training set returned
    # eval_df unchanged, so the caller got an UNFILTERED cohort that no
    # downstream artifact could distinguish from a filtered one.
    eval_df = pd.DataFrame(
        {
            "peptide": ["GLFYTRTGL", "AAYSDQWAL"],
            "label": [1, 0],
            "virus": ["X", "X"],
            "allele": ["HLA-A*02:01", "HLA-A*02:01"],
        }
    )
    with pytest.raises(ValueError, match="empty training peptide set"):
        filter_bidirectional_overlap(set(), eval_df, "X")


def test_filter_still_returns_early_on_an_empty_cohort():
    # The benign branch, kept as an early return: there is genuinely nothing to
    # filter. Guards that the fix above did not collapse both cases into a raise.
    empty = pd.DataFrame({"peptide": [], "label": [], "virus": [], "allele": []})
    out = filter_bidirectional_overlap({"GLFYTRTGL"}, empty, "X")
    assert out.empty


def test_filter_actually_removes_a_contaminated_peptide():
    # Anti-vacuity: proves the two guards above did not disable real filtering.
    eval_df = pd.DataFrame(
        {
            "peptide": ["GLFYTRTGL", "AAYSDQWAL"],
            "label": [1, 0],
            "virus": ["X", "X"],
            "allele": ["HLA-A*02:01", "HLA-A*02:01"],
        }
    )
    out = filter_bidirectional_overlap({"GLFYTRTGL"}, eval_df, "X")
    assert list(out["peptide"]) == ["AAYSDQWAL"]
