"""The Abstract's corpus-composition numbers are derivable from tracked inputs.

The manuscript readiness audit found three of them single-carrier and unbound:
13,358, 22,239 and 60.2% occur in `docs/paper.md` and in no other tracked file,
and no claim in the integrity manifest resolves to any of them. Their neighbours
35,597 and 21,432 appear elsewhere but are equally underived. So a corpus rebuild
could move the partition and leave the Abstract stating the old split, and nothing
in the suite or in CI would notice.

Each test below derives a figure from the tracked corpus and then requires the
paper's own text to carry it, so the corpus and the manuscript bind each other in
both directions: a rebuild that moves a count fails here, and an edit that changes
a number in the paper fails here too.

Two definitions are taken from the paper rather than invented. "Active" is
`is_quarantined == False`, which the Methods call the active pool. The
"nine-pathogen panel" is the set of leave-one-out folds, because the paper defines
those folds as one per eligible pathogen; it is read from
`results/loo_cross_virus_v5_clean.csv` rather than hardcoded here, so adding or
dropping a fold is visible. Loading three columns of the corpus costs about
0.03 s, so these tests read live state rather than restating it as constants.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src.iedb_data_loader import GOLD_STANDARD_EPITOPES

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "data" / "immunogenicity_dataset_v5.csv"
LOO = REPO_ROOT / "results" / "loo_cross_virus_v5_clean.csv"
PAPER = REPO_ROOT / "docs" / "paper.md"

TOTAL_ROWS = 51_185
ACTIVE_ROWS = 35_597
IN_PANEL = 13_358
OUT_OF_PANEL = 22_239
VACCINIA = 21_432
VACCINIA_LABEL = "Orthopoxvirus vaccinia"
CMV_FOLD_TRAIN_VIRAL = 34_243
GOLD_STANDARD_ROWS = 42


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return pd.read_csv(CORPUS, usecols=["peptide", "virus", "is_quarantined"], low_memory=False)


@pytest.fixture(scope="module")
def active(corpus: pd.DataFrame) -> pd.DataFrame:
    return corpus[~corpus["is_quarantined"].astype(bool)]


@pytest.fixture(scope="module")
def panel() -> set[str]:
    return set(pd.read_csv(LOO)["test_virus"])


@pytest.fixture(scope="module")
def paper_text() -> str:
    return PAPER.read_text(encoding="utf-8")


def test_the_panel_is_the_set_of_leave_one_out_folds(panel: set[str], active: pd.DataFrame) -> None:
    assert len(panel) == 9, (
        f"the paper describes nine folds, one per eligible pathogen: {sorted(panel)}"
    )
    assert panel <= set(active["virus"]), "every fold's pathogen must appear among the active rows"
    assert VACCINIA_LABEL not in panel, (
        "vaccinia is the negative background, never a panel pathogen"
    )


def test_the_active_pool_is_the_non_quarantined_rows(
    corpus: pd.DataFrame, active: pd.DataFrame
) -> None:
    assert len(corpus) == TOTAL_ROWS
    assert len(active) == ACTIVE_ROWS


def test_the_panel_partition_matches_the_abstract(active: pd.DataFrame, panel: set[str]) -> None:
    in_panel = active["virus"].isin(panel)
    assert int(in_panel.sum()) == IN_PANEL
    assert int((~in_panel).sum()) == OUT_OF_PANEL
    # The partition is exhaustive, which is why the two counts may be read as a split.
    assert IN_PANEL + OUT_OF_PANEL == ACTIVE_ROWS
    assert round(100.0 * in_panel.mean(), 1) == 37.5
    assert round(100.0 * (~in_panel).mean(), 1) == 62.5


def test_the_vaccinia_bloc_matches_the_abstract(active: pd.DataFrame) -> None:
    vaccinia = active["virus"] == VACCINIA_LABEL
    assert int(vaccinia.sum()) == VACCINIA
    assert round(100.0 * vaccinia.mean(), 1) == 60.2
    # Vaccinia is the largest single component of the out-of-panel bloc, not of the panel.
    assert VACCINIA < OUT_OF_PANEL


def test_the_gold_standard_holdout_is_the_rows_its_peptides_match(active: pd.DataFrame) -> None:
    held = active["peptide"].isin(set(GOLD_STANDARD_EPITOPES))
    assert int(held.sum()) == GOLD_STANDARD_ROWS
    assert len(GOLD_STANDARD_EPITOPES) == 16, (
        "42 rows, not 42 peptides: one peptide can carry several alleles"
    )


def test_the_holdout_falls_entirely_in_two_panel_viruses(active: pd.DataFrame) -> None:
    """Load-bearing for the fold arithmetic below, and a disclosure in its own right.

    All 42 held-out rows are EBV or HPV, so for those two folds the holdout and the
    held-out pathogen overlap and the fold total cannot subtract both in full. It also
    means the holdout removes rows from two of the nine panel viruses and none of the
    other seven. This test states the distribution; it makes no claim about its effect.
    """
    held = active[active["peptide"].isin(set(GOLD_STANDARD_EPITOPES))]
    assert dict(held["virus"].value_counts()) == {"EBV": 37, "HPV": 5}


def test_the_cmv_fold_training_total_is_the_active_pool_less_cmv_and_the_holdout(
    active: pd.DataFrame,
) -> None:
    cmv_rows = int((active["virus"] == "CMV").sum())
    derived = ACTIVE_ROWS - cmv_rows - GOLD_STANDARD_ROWS
    assert derived == CMV_FOLD_TRAIN_VIRAL
    folds = pd.read_csv(LOO).set_index("test_virus")
    assert int(folds.loc["CMV", "n_train_viral"]) == CMV_FOLD_TRAIN_VIRAL
    # The paper rounds this share to a whole percent.
    assert round(100.0 * OUT_OF_PANEL / CMV_FOLD_TRAIN_VIRAL) == 65


def test_every_fold_total_is_the_active_pool_less_its_own_pathogen_and_the_holdout(
    active: pd.DataFrame,
) -> None:
    """All nine fold totals, exactly, from the corpus alone.

    The paper gives the CMV fold as its worked example, where the arithmetic is simply
    the active pool less CMV and less the holdout. That form is NOT general: for EBV and
    HPV the holdout rows ARE rows of the held-out pathogen, so subtracting both
    double-counts them, by 37 and 5 respectively. Subtracting the held-out pathogen's
    NON-holdout rows reproduces all nine totals.
    """
    counts = active["virus"].value_counts()
    held = active[active["peptide"].isin(set(GOLD_STANDARD_EPITOPES))]["virus"].value_counts()
    folds = pd.read_csv(LOO)
    assert len(folds) == 9
    for row in folds.itertuples(index=False):
        own = int(counts[row.test_virus]) - int(held.get(row.test_virus, 0))
        expected = ACTIVE_ROWS - GOLD_STANDARD_ROWS - own
        assert int(row.n_train_viral) == expected, f"{row.test_virus} fold total"


@pytest.mark.parametrize(
    "literal",
    ["51,185", "35,597", "13,358", "22,239", "21,432", "34,243", "60.2%", "37.5%", "62.5%"],
)
def test_the_paper_states_the_figure(paper_text: str, literal: str) -> None:
    assert literal in paper_text, f"docs/paper.md no longer states {literal}"


def test_the_paper_states_the_partition_as_a_split_of_the_active_pool(paper_text: str) -> None:
    # Binds the SENTENCE, not just the tokens: a reader is told these two sum to the active pool.
    sentence = re.search(
        r"Of these 35,597 active rows, 13,358 \(37\.5%\)[^.]*?the remaining 22,239 \(62\.5%\)",
        paper_text,
        re.DOTALL,
    )
    assert sentence is not None, "the Methods sentence partitioning the active pool has changed"
