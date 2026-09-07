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
from src.train_classifier import (
    HLA_PSEUDO_COLS,
    prepare_features_10,
    prepare_features_30,
    prepare_features_50,
    prepare_features_166,
)


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
    call = "_report_binding_coverage(peptides, binding_lookup, binding_matrix_path)"
    # FOUR, not two. prepare_features_30 and _10 were instrumented first; modes
    # 50 and 166 carry a byte-identical zero-fill and were missed, so a reporter
    # that existed and was half-wired read as a complete fix. Asserted exactly
    # rather than with >=, so a NEW join site added later fails here instead of
    # inheriting the same silence. Counting call sites at any indentation, since
    # mode 166's sits inside a conditional; the `def` line is excluded by the
    # separate count below rather than by leading whitespace.
    assert source.count(call) - source.count("def " + call.split("(")[0] + "(") == 4
    assert source.count("def _report_binding_coverage(") == 1
    # The delegating builders must stay delegating, or they need their own call.
    for delegator in ("prepare_features_30_esm", "prepare_features_30_graph"):
        body = source.split("def " + delegator + "(", 1)[1].split("\ndef ", 1)[0]
        assert "prepare_features_30(df, binding_matrix_path)" in body


def test_empty_corpus_does_not_divide_by_zero(tmp_path, capsys):
    df = _df([])
    prepare_features_10(df, _matrix(tmp_path, PEPTIDES))
    assert "WARNING" not in capsys.readouterr().out


def _allele_df(peptides):
    """A 166-mode frame: peptides plus the 136 HLA pseudo-sequence columns."""
    frame = _df(peptides)
    for col in HLA_PSEUDO_COLS:
        frame[col] = 0.25
    return frame


def test_mode50_reports_coverage(tmp_path, capsys):
    prepare_features_50(_df(PEPTIDES), _matrix(tmp_path, PEPTIDES[:1]))
    out = capsys.readouterr().out
    assert "WARNING: binding coverage" in out
    assert "1/4 rows (25.0%)" in out
    assert "3 zero-filled" in out


def test_mode166_reports_coverage(tmp_path, capsys):
    prepare_features_166(_allele_df(PEPTIDES), _matrix(tmp_path, PEPTIDES[:2]))
    out = capsys.readouterr().out
    assert "WARNING: binding coverage" in out
    assert "2/4 rows (50.0%)" in out


def test_mode166_full_coverage_does_not_warn(tmp_path, capsys):
    prepare_features_166(_allele_df(PEPTIDES), _matrix(tmp_path, PEPTIDES))
    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "4/4 rows (100.0%)" in out


def test_mode166_warns_loudly_on_a_matrix_without_the_allele_columns(tmp_path, capsys):
    """The worst case of all, and it used to be completely silent.

    Modes 10, 30 and 50 raise when the matrix carries fewer than ten allele
    columns. Mode 166 instead builds np.zeros((len(df), 10)) for the WHOLE corpus
    and returns it labelled with the real column names, so an entirely wrong
    matrix yields a feature block indistinguishable from a corpus that genuinely
    scores zero against every allele.

    That contract is pinned by
    tests/test_train_classifier.py::test_prepare_features_166_no_allele_cols_uses_zeros
    and is deliberately NOT changed here. Whether mode 166 should raise like its
    siblings is an owner policy call. What is fixed is the silence.
    """
    path = tmp_path / "no_alleles.csv"
    pd.DataFrame({"peptide": PEPTIDES, "unrelated": [1.0] * len(PEPTIDES)}).to_csv(
        path, index=False
    )

    X = prepare_features_166(_allele_df(PEPTIDES), str(path))

    out = capsys.readouterr().out
    assert "WARNING: binding coverage" in out
    assert "0/4 rows (0.0%)" in out
    assert "NONE of the 10 expected allele columns" in out
    assert "no_alleles.csv" in out
    # Behaviour unchanged: still an all-zero block, still 166 columns.
    assert X.shape == (4, 166)
    assert (X[BINDING_ALLELE_COLUMNS] == 0.0).all().all()


@pytest.mark.parametrize(
    "builder,frame",
    [(prepare_features_50, _df), (prepare_features_166, _allele_df)],
)
def test_reporting_changes_no_returned_value_for_50_and_166(tmp_path, builder, frame):
    X = builder(frame(PEPTIDES), _matrix(tmp_path, PEPTIDES[:1]))
    assert len(X) == len(PEPTIDES)
    assert not X.isnull().any().any()
    bind = X[BINDING_ALLELE_COLUMNS]
    np.testing.assert_array_equal(bind.iloc[-1].to_numpy(), np.zeros(10))
    assert (bind.iloc[0].to_numpy() == 0.5).all()
