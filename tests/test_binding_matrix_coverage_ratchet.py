"""The tracked binding matrix must not drift further behind the tracked corpus.

WHAT THIS IS. A ratchet, not a bug report. It compares two TRACKED artifacts -
`data/immunogenicity_dataset_v5.csv` and `models/peptide_binding_matrix_v5.csv` -
and never rebuilds either, so it runs in a clone and in CI with no model, no
network and no MHCflurry. The gap it measures is KNOWN and was ruled on
deliberately (do not regenerate); the job here is to keep it VISIBLE and to stop
it widening silently, which is exactly what happened to produce it.

WHY IT EXISTS. The matrix is a derived artifact whose generator (the corpus) moved
after it was built, and nothing detected that. Every number below is therefore
pinned in the direction that allows IMPROVEMENT and blocks REGRESSION: a rebuild
that covers more peptides keeps these green, while a corpus that grows away from
the matrix turns them red.

THE MEASURED STATE, reproduced from tracked files by this module's own helpers
and independently matching the figures recorded when the gap was first sized:

    active rows (is_quarantined == False)      35,597   of 51,185 total
    active unique peptides                     16,360
    matrix unique peptides                     18,535
    covered active rows                        31,079   (87.31%)
    uncovered active rows                       4,518   (12.69%)
    covered active unique peptides             13,075   (79.92%)

STATE THE POPULATION WITH EVERY NUMBER. "35,597 active" and "51,185 total" are
both true and describe different things. A separate analysis of the OOF SCORING
pool reports 35,555 rows / 16,344 unique peptides; that is a third population and
its figures are NOT interchangeable with these.

WHAT THE ZERO-FILL COSTS, stated precisely because the loose version is wrong.
An uncovered peptide is not dropped - it trains on an all-zero binding vector. On
the DECOY-FREE per-virus numbers that actually gate releases, the attributable
contribution of that channel was measured as exactly +0.0000 for 8 of 9 target
viruses and -0.0853 (deflationary) for HIV-1, because 8 of the 9 have zero
uncovered rows once decoys are excluded. So this is NOT a live inflation of the
honest headline. It IS large on the decoy-CARRIED pooled per-virus numbers, where
it reached +0.2721 AUC-ROC for DENV. Do not describe it as inflating the headline,
and do not describe it as harmless.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS = PROJECT_ROOT / "data" / "immunogenicity_dataset_v5.csv"
MATRIX = PROJECT_ROOT / "models" / "peptide_binding_matrix_v5.csv"

# Baselines measured 2026-09-17 at origin/main d064dbc, from the two tracked files
# named above. Assertions are on integer COUNTS rather than fractions so the
# comparison is exact; fractions appear only in failure messages.
BASELINE_ACTIVE_ROWS = 35_597
BASELINE_ACTIVE_UNIQUE = 16_360
BASELINE_UNCOVERED_ROWS = 4_518
BASELINE_UNCOVERED_UNIQUE = 3_285

# The uncovered set is not a random sample. Every uncovered row comes from exactly
# one of these two provenances, and each is single-class: allele_matched_nonbinder
# is 100% negative, iedb_positive is 100% positive. A THIRD provenance appearing
# here would be a change in kind, not degree, and is what this pins.
EXPECTED_UNCOVERED_ORIGINS = {"allele_matched_nonbinder", "iedb_positive"}


@pytest.fixture(scope="module")
def frames() -> tuple[pd.DataFrame, set[str]]:
    """Load only the columns needed; the full corpus read is wasteful in CI."""
    if not CORPUS.exists():  # pragma: no cover - both files are tracked
        pytest.skip(f"tracked corpus absent: {CORPUS}")
    if not MATRIX.exists():  # pragma: no cover
        pytest.skip(f"tracked matrix absent: {MATRIX}")

    corpus = pd.read_csv(
        CORPUS,
        low_memory=False,
        usecols=["peptide", "is_quarantined", "label", "negative_origin"],
    )
    matrix_peptides = set(pd.read_csv(MATRIX, usecols=["peptide"])["peptide"].dropna())
    return corpus, matrix_peptides


def _active(corpus: pd.DataFrame) -> pd.DataFrame:
    """Rows the pipeline actually trains on, i.e. after the quarantine filter."""
    if "is_quarantined" in corpus.columns:
        return corpus[~corpus["is_quarantined"].astype(bool)]
    return corpus


def test_active_population_is_what_the_baselines_describe(
    frames: tuple[pd.DataFrame, set[str]],
) -> None:
    """Guard the DENOMINATOR before trusting any ratio built on it.

    Without this, a corpus that shrank could raise the coverage fraction while
    covering strictly fewer peptides, and every other test here would go green
    for the wrong reason.
    """
    corpus, _ = frames
    active = _active(corpus)

    assert len(active) == BASELINE_ACTIVE_ROWS, (
        f"active row count moved: {len(active)} vs baseline {BASELINE_ACTIVE_ROWS}. "
        "That is not necessarily wrong, but every coverage baseline in this file "
        "was measured against the old population and must be re-derived."
    )
    assert active["peptide"].nunique() == BASELINE_ACTIVE_UNIQUE


def test_uncovered_active_rows_do_not_increase(
    frames: tuple[pd.DataFrame, set[str]],
) -> None:
    """The ratchet. Fewer uncovered rows is always fine; more is a regression."""
    corpus, matrix_peptides = frames
    active = _active(corpus)

    uncovered = ~active["peptide"].isin(matrix_peptides)
    n_uncovered = int(uncovered.sum())
    n_total = len(active)

    assert n_uncovered <= BASELINE_UNCOVERED_ROWS, (
        f"binding-matrix coverage REGRESSED: {n_uncovered} uncovered active rows "
        f"({n_uncovered / n_total:.2%} of {n_total}) against a baseline of "
        f"{BASELINE_UNCOVERED_ROWS} ({BASELINE_UNCOVERED_ROWS / n_total:.2%}). "
        "The corpus has moved further ahead of the matrix. Either rebuild the "
        "matrix (scripts/build_binding_matrix_v5.py) or lower this baseline "
        "deliberately, with the reason recorded."
    )


def test_uncovered_unique_peptides_do_not_increase(
    frames: tuple[pd.DataFrame, set[str]],
) -> None:
    """Row coverage and peptide coverage can move independently.

    A single new uncovered peptide repeated across many rows inflates the row
    count without adding breadth, and a rare peptide does the reverse. Pin both.
    """
    corpus, matrix_peptides = frames
    active_peptides = set(_active(corpus)["peptide"].dropna())

    n_uncovered = len(active_peptides - matrix_peptides)
    assert n_uncovered <= BASELINE_UNCOVERED_UNIQUE, (
        f"unique-peptide coverage REGRESSED: {n_uncovered} uncovered unique "
        f"peptides of {len(active_peptides)} active, against a baseline of "
        f"{BASELINE_UNCOVERED_UNIQUE}."
    )


def test_uncovered_rows_come_from_exactly_the_known_provenances(
    frames: tuple[pd.DataFrame, set[str]],
) -> None:
    """A new provenance in the uncovered set is a change in KIND, not degree.

    The count tests above would stay green if 500 uncovered decoy rows were
    replaced by 500 uncovered rows from a new source. That would be a different
    defect wearing the same number.
    """
    corpus, matrix_peptides = frames
    active = _active(corpus)

    uncovered = active[~active["peptide"].isin(matrix_peptides)]
    origins = set(uncovered["negative_origin"].dropna().unique())

    unexpected = origins - EXPECTED_UNCOVERED_ORIGINS
    assert not unexpected, (
        f"uncovered rows now include unrecognised negative_origin value(s): "
        f"{sorted(unexpected)}. Known set is {sorted(EXPECTED_UNCOVERED_ORIGINS)}. "
        "Investigate before widening this set."
    )


def test_missingness_remains_label_correlated_and_is_not_silently_resolved(
    frames: tuple[pd.DataFrame, set[str]],
) -> None:
    """Keep the leakage channel VISIBLE, which is this contract's real purpose.

    'Binding vector is all zeros' is correlated with the label in training, and
    that pattern cannot occur at inference. This test does not enforce a
    threshold - what coverage is acceptable is an owner policy question, the same
    position src/train_classifier.py takes for its own coverage reporter. It
    asserts only that the asymmetry is still REAL and still MEASURABLE, so that
    a future reader cannot conclude from a green suite that it went away.
    """
    corpus, matrix_peptides = frames
    active = _active(corpus)

    covered_mask = active["peptide"].isin(matrix_peptides)
    uncovered_rate = active.loc[~covered_mask, "label"].mean()
    covered_rate = active.loc[covered_mask, "label"].mean()

    assert uncovered_rate > covered_rate, (
        "the coverage/label asymmetry has changed sign or vanished "
        f"(uncovered {uncovered_rate:.4f} vs covered {covered_rate:.4f}). "
        "If a rebuild genuinely resolved it, retire this test and say so; do not "
        "just relax it."
    )
    # Each provenance in the uncovered set is single-class, which is WHY the
    # asymmetry exists. Pin that mechanism rather than the magnitude.
    uncovered = active[~covered_mask]
    for origin, group in uncovered.groupby("negative_origin"):
        distinct = set(group["label"].unique())
        assert len(distinct) == 1, (
            f"negative_origin '{origin}' is no longer single-class in the "
            f"uncovered set: labels {sorted(distinct)}"
        )


def test_matrix_covers_peptides_beyond_the_active_corpus(
    frames: tuple[pd.DataFrame, set[str]],
) -> None:
    """The matrix is not a subset of the corpus, and that is expected.

    It retains peptides from earlier corpora. Recording this stops a future
    reader from 'fixing' the gap by pruning the matrix to the corpus, which would
    shrink the served domain while leaving coverage of active rows unchanged.
    """
    corpus, matrix_peptides = frames
    active_peptides = set(_active(corpus)["peptide"].dropna())

    beyond = matrix_peptides - active_peptides
    assert beyond, "matrix no longer carries any peptide outside the active corpus"
    assert len(matrix_peptides) > len(active_peptides)
