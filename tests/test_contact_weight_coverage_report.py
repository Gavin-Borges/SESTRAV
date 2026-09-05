"""prepare_features_51 must say how far the contact-weight panel reached.

ALLELE_CONTACT_WEIGHTS carries ten alleles. Any row whose allele is not one of
them receives POPULATION_AVG_CONTACT_WEIGHTS. That fallback is deliberate and is
NOT changed by these tests or by the code they cover. The defect was that its
extent was unrecorded: the substituted weights sit in the same numeric scale as
measured ones, so a fallback row is indistinguishable from a row whose allele
genuinely is average.

On a corpus with hundreds of distinct alleles the fallback is the majority path,
not an edge case, which is why a silent one is worth a report.

These tests pin the report and pin that the returned frame did not change.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    ALLELE_CONTACT_WEIGHTS,
    BINDING_ALLELE_COLUMNS,
    CONTACT_WEIGHT_COLUMNS,
    POPULATION_AVG_CONTACT_WEIGHTS,
)
from src.train_classifier import _report_contact_weight_coverage, prepare_features_51

# Two alleles that ARE in the panel, and one that is not.
IN_PANEL = sorted(ALLELE_CONTACT_WEIGHTS)[:2]
OFF_PANEL = "HLA-C*07:01_not_in_panel"
PEPTIDES = ["AAAAAAAAA", "CCCCCCCCC", "DDDDDDDDD"]


def _matrix(tmp_path, peptides, name="bm.csv"):
    cols = {"peptide": list(peptides)}
    for allele in BINDING_ALLELE_COLUMNS:
        cols[allele] = [0.5] * len(peptides)
    path = tmp_path / name
    pd.DataFrame(cols).to_csv(path, index=False)
    return str(path)


def _df(peptides, alleles):
    frame = pd.DataFrame({"peptide": list(peptides), "label": [1] * len(peptides)})
    if alleles is not None:
        frame["hla_allele"] = alleles
    return frame


def test_partial_coverage_warns_and_counts_the_fallback_rows(tmp_path, capsys):
    df = _df(PEPTIDES, [IN_PANEL[0], OFF_PANEL, OFF_PANEL])

    prepare_features_51(df, _matrix(tmp_path, PEPTIDES))

    out = capsys.readouterr().out
    assert "WARNING: contact-weight coverage" in out
    assert "1/3 rows (33.3%)" in out
    assert "2 fell back to POPULATION_AVG_CONTACT_WEIGHTS" in out


def test_full_coverage_reports_without_warning(tmp_path, capsys):
    df = _df(PEPTIDES[:2], IN_PANEL)

    prepare_features_51(df, _matrix(tmp_path, PEPTIDES[:2]))

    out = capsys.readouterr().out
    assert "Contact-weight coverage: 2/2 rows (100.0%)" in out
    assert "WARNING" not in out
    assert "0 fell back" in out


def test_missing_allele_column_is_named_in_the_report(tmp_path, capsys):
    # Every row falls back, and the report must say WHY: the column is absent,
    # which is a different failure from an allele the panel does not carry.
    df = _df(PEPTIDES, None)

    prepare_features_51(df, _matrix(tmp_path, PEPTIDES))

    out = capsys.readouterr().out
    assert "WARNING: contact-weight coverage" in out
    assert "0/3 rows (0.0%)" in out
    assert "column 'hla_allele' is ABSENT from the frame" in out


def test_report_names_the_panel_size(tmp_path, capsys):
    """A reader must be able to tell 1/3 against a 10-allele panel from 1/3
    against a 200-allele one; the second would mean something quite different."""
    df = _df(PEPTIDES, [IN_PANEL[0], OFF_PANEL, OFF_PANEL])

    prepare_features_51(df, _matrix(tmp_path, PEPTIDES))

    assert f"({len(ALLELE_CONTACT_WEIGHTS)} alleles)" in capsys.readouterr().out


@pytest.mark.parametrize("alleles", [[IN_PANEL[0], OFF_PANEL, OFF_PANEL], None])
def test_reporting_changes_no_returned_value(tmp_path, alleles):
    """The report is observation only. The fallback contract is unchanged: an
    off-panel or absent allele still receives POPULATION_AVG_CONTACT_WEIGHTS."""
    df = _df(PEPTIDES, alleles)

    X = prepare_features_51(df, _matrix(tmp_path, PEPTIDES))

    assert len(X) == len(PEPTIDES)
    assert not X.isnull().any().any()
    avg = np.asarray(POPULATION_AVG_CONTACT_WEIGHTS, dtype=float)
    # The last two rows are off-panel or column-less in both parametrisations.
    for row in (1, 2):
        np.testing.assert_allclose(
            X.loc[row, list(CONTACT_WEIGHT_COLUMNS)].to_numpy(dtype=float), avg
        )
    if alleles is not None:
        on_panel = np.asarray(ALLELE_CONTACT_WEIGHTS[IN_PANEL[0]], dtype=float)
        np.testing.assert_allclose(
            X.loc[0, list(CONTACT_WEIGHT_COLUMNS)].to_numpy(dtype=float), on_panel
        )


def test_empty_input_reports_nothing_and_does_not_divide_by_zero(capsys):
    """Exercised on the reporter directly, not through prepare_features_51.

    prepare_features_51 already raises KeyError on an empty frame, inside
    prepare_features_50's physico block, and it does so identically on
    origin/main. That is pre-existing behaviour and is deliberately not touched
    here, so the guard is tested where it actually lives.
    """
    assert _report_contact_weight_coverage([], "hla_allele", True) == 0
    assert capsys.readouterr().out == ""
