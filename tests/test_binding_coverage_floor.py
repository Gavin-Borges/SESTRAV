"""A coverage shortfall can be made to STOP a run, not just describe one.

`tests/test_binding_coverage_report.py` pins that the reporter speaks. This file
pins that it can also refuse. The two are separate contracts and the second was
missing: a training run handed a matrix built from a different corpus printed a
WARNING and then trained on it anyway, which is how a stale matrix reaches a
shipped artifact without anything failing.

WHAT THE FLOOR IS SET AGAINST, measured 2026-09-18 from the two tracked files by
`git show`, over the corpus's 35,597 ACTIVE rows:

    models/peptide_binding_matrix_v5.csv    31,079 / 35,597 rows    87.31%
    models/peptide_binding_matrix_v4.csv     8,767 / 35,597 rows    24.63%

Name the population with those figures. They are ACTIVE rows; a separate analysis
of the OOF SCORING pool reports 35,555 rows, and the module docstring in
`src/train_classifier.py` quotes v4 as 8,725/35,555 from that third population.
The numbers are consistent, not interchangeable.

Any floor strictly between 0.2463 and 0.8731 separates the shipped matrix from the
stale one. Nothing here hardcodes a production floor: the default is OFF, exactly
preserving the previous report-only behaviour, because the comment on mode 166's
all-zero path records that whether a shortfall should raise "is an owner policy
call". This supplies the mechanism and leaves the policy switch unset.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import BINDING_ALLELE_COLUMNS
from src.train_classifier import (
    BINDING_COVERAGE_FLOOR_ENV,
    BindingCoverageBelowFloor,
    _report_binding_coverage,
    _resolve_binding_coverage_floor,
    prepare_features_30,
)

PEPTIDES = ["AAAAAAAAA", "CCCCCCCCC", "DDDDDDDDD", "EEEEEEEEE"]


def _lookup(peptides):
    cols = {allele: [0.5] * len(peptides) for allele in BINDING_ALLELE_COLUMNS}
    return pd.DataFrame(cols, index=pd.Index(list(peptides), name="peptide"))


def _matrix_csv(tmp_path, peptides, name="bm.csv"):
    cols: dict[str, list] = {"peptide": list(peptides)}
    for allele in BINDING_ALLELE_COLUMNS:
        cols[allele] = [0.5] * len(peptides)
    path = tmp_path / name
    pd.DataFrame(cols).to_csv(path, index=False)
    return str(path)


@pytest.fixture(autouse=True)
def _clear_floor_env(monkeypatch):
    """Every test states its own floor; an inherited one would make results a lottery."""
    monkeypatch.delenv(BINDING_COVERAGE_FLOOR_ENV, raising=False)


# --- the default must not change behaviour ---------------------------------


def test_default_does_not_raise_even_at_zero_coverage(capsys):
    """The whole point of the default is that it is the old behaviour, exactly."""
    missing = _report_binding_coverage(PEPTIDES, _lookup(["ZZZZZZZZZ"]), "bm.csv")
    assert missing == 4
    assert "WARNING: binding coverage: 0/4 rows (0.0%)" in capsys.readouterr().out


def test_empty_corpus_returns_before_any_floor_check(monkeypatch):
    """The divide-by-zero guard must win over the floor, not the other way round."""
    monkeypatch.setenv(BINDING_COVERAGE_FLOOR_ENV, "0.99")
    assert _report_binding_coverage([], _lookup(PEPTIDES), "bm.csv") == 0


# --- enforcement ------------------------------------------------------------


def test_raises_when_coverage_is_below_an_explicit_floor():
    with pytest.raises(BindingCoverageBelowFloor) as excinfo:
        _report_binding_coverage(PEPTIDES, _lookup(PEPTIDES[:1]), "bm.csv", min_coverage=0.80)
    message = str(excinfo.value)
    assert "1/4" in message and "25.00%" in message and "80.00%" in message
    assert "bm.csv" in message


def test_does_not_raise_when_coverage_meets_the_floor():
    assert _report_binding_coverage(PEPTIDES, _lookup(PEPTIDES), "bm.csv", min_coverage=0.80) == 0


def test_floor_is_inclusive_at_exactly_the_boundary():
    """25% against a 0.25 floor must PASS; a strict '<' is the intended comparison."""
    assert (
        _report_binding_coverage(PEPTIDES, _lookup(PEPTIDES[:1]), "bm.csv", min_coverage=0.25) == 3
    )


def test_environment_variable_enforces_the_floor(monkeypatch):
    monkeypatch.setenv(BINDING_COVERAGE_FLOOR_ENV, "0.80")
    with pytest.raises(BindingCoverageBelowFloor):
        _report_binding_coverage(PEPTIDES, _lookup(PEPTIDES[:1]), "bm.csv")


def test_explicit_argument_overrides_the_environment(monkeypatch):
    monkeypatch.setenv(BINDING_COVERAGE_FLOOR_ENV, "0.99")
    assert _report_binding_coverage(PEPTIDES, _lookup(PEPTIDES), "bm.csv", min_coverage=0.10) == 0


@pytest.mark.parametrize("raw", ["", "   "])
def test_blank_environment_value_means_report_only(monkeypatch, raw):
    monkeypatch.setenv(BINDING_COVERAGE_FLOOR_ENV, raw)
    assert _report_binding_coverage(PEPTIDES, _lookup([]), "bm.csv") == 4


# --- a misconfigured floor must fail loudly, never silently disable itself ---


def test_a_non_numeric_floor_raises_and_names_the_variable(monkeypatch):
    monkeypatch.setenv(BINDING_COVERAGE_FLOOR_ENV, "eighty percent")
    with pytest.raises(ValueError, match=BINDING_COVERAGE_FLOOR_ENV):
        _resolve_binding_coverage_floor()


def test_a_percentage_rather_than_a_fraction_is_rejected(monkeypatch):
    """80 is the obvious typo for 0.80 and would otherwise fail every run."""
    monkeypatch.setenv(BINDING_COVERAGE_FLOOR_ENV, "80")
    with pytest.raises(ValueError, match="FRACTION"):
        _resolve_binding_coverage_floor()


def test_a_negative_floor_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        _resolve_binding_coverage_floor(-0.1)


def test_unset_resolves_to_none():
    assert _resolve_binding_coverage_floor() is None


# --- end to end through a real builder --------------------------------------


def test_a_builder_refuses_a_matrix_that_misses_the_corpus(tmp_path, monkeypatch):
    """The floor has to bite on the production join path, not only in isolation."""
    monkeypatch.setenv(BINDING_COVERAGE_FLOOR_ENV, "0.80")
    df = pd.DataFrame({"peptide": PEPTIDES, "label": [1] * len(PEPTIDES)})
    with pytest.raises(BindingCoverageBelowFloor):
        prepare_features_30(df, _matrix_csv(tmp_path, PEPTIDES[:1]))


def test_a_builder_accepts_a_matrix_that_covers_the_corpus(tmp_path, monkeypatch):
    monkeypatch.setenv(BINDING_COVERAGE_FLOOR_ENV, "0.80")
    df = pd.DataFrame({"peptide": PEPTIDES, "label": [1] * len(PEPTIDES)})
    features = prepare_features_30(df, _matrix_csv(tmp_path, PEPTIDES))
    assert len(features) == len(PEPTIDES)
