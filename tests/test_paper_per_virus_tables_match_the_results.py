"""The paper's per-virus tables are the tracked results, cell for cell.

The manuscript readiness audit found the whole per-virus real-negatives-only column
unbound: `auc_roc_real_neg_only` is named by no claim in the integrity manifest, and
the five decomposition claims all pin the `Mean` row only. So of Tables 2b and 3b,
nothing enforced any per-virus cell - only the three regime means. The honest
real-neg-only scale is the one the paper leads per-virus quality assessment with, so
it is the column least able to afford that.

These tests parse the two tables out of `docs/paper.md` and bind them to
`results/per_virus_eval_v5_mode31.csv`, `results/loo_cross_virus_v5_clean.csv` and
`results/loo_binding_confound_decomposition.csv`. They also bind the decomposition
file itself to the full-precision sources it summarises, which is where the honest
column actually comes from.

One subtlety the tests pin deliberately, because a reader will hit it. The caption
defines decoy inflation as regime 1 minus regime 2, but those differences were taken
at full precision and then rounded, so recomputing them from the table's own 3-decimal
columns disagrees by 0.001 on four of the nine rows (DENV, EBV, HBV, HIV-1). The
published values are right; the recomputation is what is lossy.

The existing `tests/test_compute_loo_binding_confound_results_guard.py` covers the
generating script's output behaviour on patched sources. It asserts nothing about the
tracked values or about the paper, so these tests are complementary rather than a
second copy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER = REPO_ROOT / "docs" / "paper.md"
PER_VIRUS = REPO_ROOT / "results" / "per_virus_eval_v5_mode31.csv"
LOO = REPO_ROOT / "results" / "loo_cross_virus_v5_clean.csv"
DECOMP = REPO_ROOT / "results" / "loo_binding_confound_decomposition.csv"

PANEL = ["CMV", "DENV", "EBV", "HBV", "HCV", "HIV-1", "HPV", "IAV", "SARS-CoV-2"]
# Rows where the caption's definition, recomputed from the PUBLISHED 3-decimal columns,
# lands 0.001 away from the published derived value. Measured, not assumed.
ROUNDING_DISAGREES = {"DENV", "EBV", "HBV", "HIV-1"}


def _markdown_table(text: str, header_starts_with: str) -> dict[str, list[str]]:
    """Rows of the markdown table whose header line starts with the given prefix."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(header_starts_with):
            break
    else:
        raise AssertionError(f"no table header starting {header_starts_with!r} in {PAPER.name}")
    rows: dict[str, list[str]] = {}
    for line in lines[i + 2 :]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # A footnote marker rides on the value in this paper; it is not part of the number.
        rows[cells[0]] = [c.rstrip("*").strip() for c in cells[1:]]
    return rows


@pytest.fixture(scope="module")
def paper_text() -> str:
    return PAPER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def decomposition() -> pd.DataFrame:
    return pd.read_csv(DECOMP)


@pytest.fixture(scope="module")
def per_virus() -> pd.DataFrame:
    return pd.read_csv(PER_VIRUS).set_index("virus")


@pytest.fixture(scope="module")
def loo() -> pd.DataFrame:
    return pd.read_csv(LOO).set_index("test_virus")


def test_table_3b_is_the_decomposition_file_cell_for_cell(paper_text, decomposition) -> None:
    table = _markdown_table(paper_text, "| Virus      | R1 within, all neg")
    rows = decomposition.set_index("virus")
    assert set(table) == set(rows.index), (
        "Table 3b's row set has drifted from the decomposition file"
    )
    columns = [
        "within_all_neg",
        "within_real_neg",
        "loo_cross_virus",
        "decoy_inflation",
        "transfer_gap",
    ]
    for virus, cells in table.items():
        for position, column in enumerate(columns):
            assert cells[position] == f"{rows.loc[virus, column]:.3f}", f"Table 3b {virus} {column}"
        decoy_frac = rows.loc[virus, "decoy_frac_neg"]
        expected = "" if pd.isna(decoy_frac) else f"{decoy_frac:.3f}"
        assert cells[5] == expected, f"Table 3b {virus} decoy fraction"


def test_table_2b_is_the_per_virus_file_and_the_decomposition(
    paper_text, per_virus, decomposition
) -> None:
    table = _markdown_table(paper_text, "| Virus      | Within-CV AUC-ROC | Real-neg-only AUC-ROC")
    rows = decomposition.set_index("virus")
    assert set(table) == set(PANEL), "Table 2b covers the nine panel pathogens"
    for virus, cells in table.items():
        assert cells[0] == f"{per_virus.loc[virus, 'auc_roc']:.3f}", f"Table 2b {virus} within-CV"
        assert cells[1] == f"{per_virus.loc[virus, 'auc_roc_real_neg_only']:.3f}", (
            f"Table 2b {virus} honest"
        )
        # The caption defines Gap as real-neg-only MINUS within-CV, the negated inflation.
        gap = -rows.loc[virus, "decoy_inflation"]
        assert float(cells[2]) == pytest.approx(gap, abs=5e-4), f"Table 2b {virus} gap"
        assert cells[3] == str(int(per_virus.loc[virus, "n_neg_real"])), (
            f"Table 2b {virus} n_real_neg"
        )


def test_the_decomposition_regimes_are_the_full_precision_sources_rounded(
    decomposition, per_virus, loo
) -> None:
    for row in decomposition[decomposition["virus"] != "Mean"].itertuples(index=False):
        assert round(per_virus.loc[row.virus, "auc_roc"], 3) == row.within_all_neg
        assert round(per_virus.loc[row.virus, "auc_roc_real_neg_only"], 3) == row.within_real_neg
        assert round(loo.loc[row.virus, "auc_roc"], 3) == row.loo_cross_virus


def test_the_derived_columns_are_full_precision_differences(decomposition, per_virus, loo) -> None:
    for row in decomposition[decomposition["virus"] != "Mean"].itertuples(index=False):
        r1 = per_virus.loc[row.virus, "auc_roc"]
        r2 = per_virus.loc[row.virus, "auc_roc_real_neg_only"]
        r3 = loo.loc[row.virus, "auc_roc"]
        assert round(r1 - r2, 3) == round(row.decoy_inflation, 3), f"{row.virus} decoy inflation"
        assert round(r2 - r3, 3) == round(row.transfer_gap, 3), f"{row.virus} transfer gap"


def test_recomputing_from_the_published_columns_is_lossy_on_exactly_four_rows(
    decomposition,
) -> None:
    """Guards the docstring's warning, so it cannot quietly go stale."""
    disagreeing = set()
    for row in decomposition[decomposition["virus"] != "Mean"].itertuples(index=False):
        from_published = round(row.within_all_neg - row.within_real_neg, 3)
        if from_published != round(row.decoy_inflation, 3):
            disagreeing.add(row.virus)
        elif round(row.within_real_neg - row.loo_cross_virus, 3) != round(row.transfer_gap, 3):
            disagreeing.add(row.virus)
    assert disagreeing == ROUNDING_DISAGREES


def test_the_mean_row_is_the_mean_of_the_nine(decomposition) -> None:
    per = decomposition[decomposition["virus"] != "Mean"]
    mean = decomposition[decomposition["virus"] == "Mean"].iloc[0]
    assert len(per) == 9
    for column in (
        "within_all_neg",
        "within_real_neg",
        "loo_cross_virus",
        "decoy_inflation",
        "transfer_gap",
    ):
        assert round(per[column].mean(), 3) == round(mean[column], 3), column
    assert pd.isna(mean["decoy_frac_neg"]), "the Mean row leaves the decoy fraction blank"
