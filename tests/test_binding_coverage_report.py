"""A binding matrix that does not cover the corpus must say so.

prepare_features_30 and prepare_features_10 zero-fill any peptide absent from
the binding matrix. That is the right contract - it preserves row alignment,
which the callers depend on, since they index the feature frame positionally.
The defect is that it was SILENT: nothing in a run's output recorded how much of
the corpus the matrix actually reached, so an artifact could name one matrix
while its features were built from another.

That silence has a measured cost. On the v5 corpus,
models/peptide_binding_matrix_v4.csv reaches 8,725 of 35,555 rows, and those rows
are 73.2% positive against 6.1% for the rest, because v4 predates the corpus's
negative-class expansion. The all-zero block therefore acts as a label proxy
worth roughly +0.17 pooled AUC-PR, which is larger than the entire published
model-versus-comparator delta it would be read against.

These tests pin the report, and pin that the report changed no returned value.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import BINDING_ALLELE_COLUMNS
from src.train_classifier import prepare_features_10, prepare_features_30


def _matrix(tmp_path: Path, peptides, name="bm.csv") -> str:
    cols: dict[str, list] = {"peptide": list(peptides)}
    for allele in BINDING_ALLELE_COLUMNS:
        cols[allele] = [0.5] * len(peptides)
    path = tmp_path / name
    pd.DataFrame(cols).to_csv(path, index=False)
    return str(path)


def _df(peptides):
    return pd.DataFrame({"peptide": list(peptides), "label": [1] * len(peptides)})


PEPTIDES = ["AAAAAAAAA", "CCCCCCCCC", "DDDDDDDDD", "EEEEEEEEE"]


def test_full_coverage_reports_without_warning(tmp_path, capsys):
    df = _df(PEPTIDES)
    prepare_features_30(df, _matrix(tmp_path, PEPTIDES))
    out = capsys.readouterr().out
    assert "Binding coverage: 4/4 rows (100.0%)" in out
    assert "WARNING" not in out


def test_partial_coverage_warns_and_counts_the_zero_filled_rows(tmp_path, capsys):
    df = _df(PEPTIDES)
    prepare_features_30(df, _matrix(tmp_path, PEPTIDES[:1]))
    out = capsys.readouterr().out
    assert "WARNING: binding coverage" in out
    assert "1/4 rows (25.0%)" in out
    assert "3 zero-filled" in out


def test_the_report_names_the_matrix_it_used(tmp_path, capsys):
    """The run log must record WHICH matrix, not just that there was one.

    A config records the path it was handed; only this line records the path
    that was actually opened and how far it reached.
    """
    df = _df(PEPTIDES)
    path = _matrix(tmp_path, PEPTIDES[:2], name="peptide_binding_matrix_vX.csv")
    prepare_features_30(df, path)
    assert "peptide_binding_matrix_vX.csv" in capsys.readouterr().out


def test_mode10_also_reports(tmp_path, capsys):
    df = _df(PEPTIDES)
    prepare_features_10(df, _matrix(tmp_path, PEPTIDES[:3]))
    out = capsys.readouterr().out
    assert "WARNING: binding coverage" in out
    assert "3/4 rows (75.0%)" in out


@pytest.mark.parametrize("builder", [prepare_features_30, prepare_features_10])
def test_reporting_changes_no_returned_value(tmp_path, builder):
    """The contract is unchanged: zero-fill, no drop, no NaN, alignment kept."""
    df = _df(PEPTIDES)
    X = builder(df, _matrix(tmp_path, PEPTIDES[:-1]))
    assert len(X) == len(df)
    assert not X.isnull().any().any()
    bind = X[BINDING_ALLELE_COLUMNS]
    np.testing.assert_array_equal(bind.iloc[-1].to_numpy(), np.zeros(10))
    assert (bind.iloc[0].to_numpy() == 0.5).all()


def test_both_builders_are_wired_to_the_reporter():
    """A reporter that exists but is not called is the failure this catches."""
    import src.train_classifier as tc

    source = Path(tc.__file__).read_text(encoding="utf-8")
    call = "\n    _report_binding_coverage(peptides, binding_lookup, binding_matrix_path)\n"
    # Indented, so the `def` line does not count as a call site.
    assert source.count(call) == 2
    assert source.count("def _report_binding_coverage(") == 1


def test_empty_corpus_does_not_divide_by_zero(tmp_path, capsys):
    df = _df([])
    prepare_features_10(df, _matrix(tmp_path, PEPTIDES))
    assert "WARNING" not in capsys.readouterr().out
