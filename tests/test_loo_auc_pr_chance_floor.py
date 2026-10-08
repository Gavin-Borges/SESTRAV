"""The AUC-PR chance floor the LOO cross-virus table is read against.

Average precision has no fixed chance level: it equals the test set's positive
rate. The LOO table's two largest AUC-PR figures sit on test sets that are 98.5
and 97.7 per cent positive, so both are BELOW chance while looking like the
table's best results. The generator now publishes the floor and the lift beside
each figure.

The last test reads the tracked artifact, so it fails if a regeneration changes
the published story. That is deliberate: the disclosure is a statement about
those nine rows, and it should not be able to go stale silently.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_loo_cross_virus_v5 import auc_pr_chance_floor, auc_pr_lift

TRACKED_LOO_CSV = Path(__file__).resolve().parents[1] / "results" / "loo_cross_virus_v5_clean.csv"


def test_floor_is_the_positive_rate_of_the_test_set() -> None:
    """Four decimals, matching auc_pr, not the three pos_rate_test carries."""
    assert auc_pr_chance_floor(806, 818) == 0.9853
    assert auc_pr_chance_floor(186, 323) == 0.5759
    assert auc_pr_chance_floor(1, 3) == 0.3333


def test_floor_spans_the_whole_range() -> None:
    assert auc_pr_chance_floor(0, 10) == 0.0
    assert auc_pr_chance_floor(10, 10) == 1.0


def test_floor_refuses_inputs_that_are_not_a_positive_rate() -> None:
    """A floor outside 0 to 1 would silently invert every lift beside it."""
    with pytest.raises(ValueError, match="non-empty"):
        auc_pr_chance_floor(0, 0)
    with pytest.raises(ValueError, match="not within"):
        auc_pr_chance_floor(11, 10)
    with pytest.raises(ValueError, match="not within"):
        auc_pr_chance_floor(-1, 10)


def test_lift_keeps_the_sign_that_carries_the_meaning() -> None:
    """Below chance must read negative; the magnitudes here are small."""
    assert auc_pr_lift(0.9804, 0.9853) == -0.0049
    assert auc_pr_lift(0.8284, 0.7312) == 0.0972
    assert auc_pr_lift(0.5, 0.5) == 0.0


def test_three_decimal_rounding_would_have_cost_a_tenth_of_a_lift() -> None:
    """Why the floor is published at four decimals and not reused from pos_rate_test.

    DENV's floor is 0.98533. Rounded to three decimals, as pos_rate_test is, it
    becomes 0.985, moving the lift from -0.0049 to -0.0046 - a shift of 3e-04
    against a lift of about 5e-03.
    """
    exact = auc_pr_chance_floor(806, 818)
    three_dp = round(806 / 818, 3)
    assert exact != three_dp
    assert abs(auc_pr_lift(0.9804, exact) - auc_pr_lift(0.9804, three_dp)) >= 2e-4


def test_the_tracked_loo_table_still_tells_the_disclosed_story() -> None:
    """Five of the nine LOO viruses score below their own chance floor.

    Among them are the table's two largest AUC-PR figures, DENV and HIV-1. The
    floor is recomputed here from n_test_pos and n_test_total, which the tracked
    artifact already carries, so this test holds today and does not wait on a
    regeneration.
    """
    frame = pd.read_csv(TRACKED_LOO_CSV)
    assert len(frame) == 9

    floors = frame["n_test_pos"] / frame["n_test_total"]
    below = frame.loc[frame["auc_pr"] < floors, "test_virus"].tolist()
    assert len(below) == 5, f"expected five viruses below their floor, got {below}"

    two_largest = frame.nlargest(2, "auc_pr")["test_virus"].tolist()
    assert set(two_largest) <= set(below), (
        f"the two largest AUC-PR figures {two_largest} are expected to be below chance"
    )


def test_both_columns_are_wired_into_the_emitted_row_and_the_summary() -> None:
    """The functions are only useful if run_loo actually writes their output.

    Running the generator retrains nine leave-one-virus-out models, so the
    wiring is checked statically: both names must appear as keys of the row
    dictionary that becomes the CSV, and in the summary table's column list.
    """
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_loo_cross_virus_v5.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    dict_keys: set[str] = set()
    list_strings: list[set[str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            dict_keys |= {
                k.value
                for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
        if isinstance(node, ast.List):
            list_strings.append(
                {
                    e.value
                    for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                }
            )

    for name in ("auc_pr_chance_floor", "auc_pr_lift"):
        assert name in dict_keys, f"{name} is not a key of any emitted row"
        assert any({name, "auc_pr", "test_virus"} <= group for group in list_strings), (
            f"{name} is missing from the summary table's columns"
        )


def test_pos_rate_test_is_the_same_quantity_at_lower_precision() -> None:
    """The floor is not a new measurement, which is why no auc_pr value moves.

    pos_rate_test is already n_test_pos / n_test_total rounded to three
    decimals. Each tracked value must therefore agree with the four-decimal
    floor to within three-decimal rounding.
    """
    frame = pd.read_csv(TRACKED_LOO_CSV)
    for row in frame.itertuples():
        floor = auc_pr_chance_floor(row.n_test_pos, row.n_test_total)
        assert abs(row.pos_rate_test - floor) <= 5e-4, row.test_virus
