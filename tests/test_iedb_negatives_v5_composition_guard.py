"""Composition guard for the tracked data/iedb_negatives_v5.csv artifact.

WHY THIS EXISTS. This file is not what the current generator produces, and that
divergence has to stay visible and stay bounded.

The history, established by re-derivation rather than inference:

- Commit dcbb1b1 added a 34,358-row file, written by the generator as it stood
  at that time. That run is reproducible today: checking out the pre-fix
  generator (`git show a15d05e^:scripts/ingest_iedb_negatives.py`) and running
  it on the same raw export reproduces the artifact bit-for-bit.
- Commit 58bbc15 then edited the FINISHED file in place, dropping the 1,888 rows
  whose assay_type is "biological activity" - a post-hoc filter, not a re-run.
  Verified as a strict order-preserving subset: 0 rows added, 0 modified.
- scripts/ingest_iedb_negatives.py later gained that same exclusion as Filter 6,
  but applies it BEFORE the intra-export dedup rather than after. A current
  re-run therefore yields 32,506 rows: a strict SUPERSET of this file, adding 36
  rows that this file lacks.

So a re-run legitimately disagrees with the committed artifact, and that is
expected rather than a fault. What must NOT happen again is another silent
in-place edit of a finished data file. These tests pin the composition so that
any such edit fails loudly instead of being discovered by audit a year later.

Delete this file when the artifact is regenerated to 32,506 rows and the code
path and the data agree again.

Note on reading data/: the repository's own tooling reads these paths directly
and so does this test. Some developer shells deny-list `data/` for interactive
reads; that restriction does not apply to the test process.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NEGATIVES_CSV = REPO_ROOT / "data" / "iedb_negatives_v5.csv"

# Measured at origin/main 5a23a4e with the csv module, header excluded, and
# cross-checked against `wc -l` minus one after confirming a single trailing LF.
EXPECTED_ROWS = 32470

# 58bbc15 removed exactly these, and every one carried label 0.
EXCLUDED_ASSAY_TYPE = "biological activity"
ROWS_REMOVED_BY_58BBC15 = 1888

# Every constant below is an INDEPENDENTLY MEASURED literal, never derived from
# another one. Deriving them would make the consistency test at the bottom a
# tautology that passes for any value of EXPECTED_ROWS.
ROWS_BEFORE_58BBC15 = 34358  # csv.reader over `git show 58bbc15^:data/iedb_negatives_v5.csv`
CURRENT_GENERATOR_ROWS = 32506  # measured re-run of scripts/ingest_iedb_negatives.py
KNOWN_REGENERATION_SURPLUS = 36  # whole-row multiset diff, all 23 fields


def _rows():
    if not NEGATIVES_CSV.exists():
        pytest.skip(f"{NEGATIVES_CSV.name} not present in this checkout")
    with open(NEGATIVES_CSV, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


def test_row_count_is_pinned():
    """A silent in-place edit of this file changes the row count. Catch it."""
    _, rows = _rows()
    assert len(rows) == EXPECTED_ROWS, (
        f"data/iedb_negatives_v5.csv has {len(rows)} rows, expected {EXPECTED_ROWS}. "
        "If this file was regenerated deliberately, update this guard and the "
        "provenance sidecar together - the sidecar is the attestation."
    )


def test_assay_type_column_is_present():
    """The pin below is meaningless if the column silently disappears."""
    fieldnames, _ = _rows()
    assert fieldnames is not None
    assert "assay_type" in fieldnames, (
        f"assay_type column is gone; header is {fieldnames}"
    )


def test_biological_activity_rows_are_absent():
    """
    The whole point of 58bbc15.

    A negative "biological activity" outcome (for example a failed binding
    assay) does not imply the peptide is non-immunogenic in a T-cell context,
    so such rows must not serve as immunogenicity negatives. This was the
    documented root cause of the HPV AUC-ROC inversion.
    """
    _, rows = _rows()
    offenders = [r for r in rows if r["assay_type"] == EXCLUDED_ASSAY_TYPE]

    assert not offenders, (
        f"{len(offenders)} rows with assay_type == {EXCLUDED_ASSAY_TYPE!r} are back "
        "in the negatives set. These are the rows 58bbc15 removed."
    )


def test_every_row_is_a_negative():
    """This artifact is the negatives pool; a positive in it would be a leak."""
    fieldnames, rows = _rows()
    assert fieldnames is not None
    if "label" not in fieldnames:
        pytest.skip("no label column in this schema")

    non_negative = {r["label"] for r in rows if r["label"] != "0"}
    assert not non_negative, f"non-zero labels present in the negatives pool: {non_negative}"


def test_known_generator_divergence_is_stated_not_silent():
    """
    Documents the accepted gap between this artifact and a current re-run.

    Each constant is an independently measured literal, so these really are
    cross-checks and not restatements: change EXPECTED_ROWS alone and both
    assertions fail. Running the real generator needs a 1.34 GB raw export that
    is not in a checkout, so the re-run figure cannot be measured here; it is
    pinned instead, and the measurement itself lives in the C2 provenance audit.
    """
    assert ROWS_BEFORE_58BBC15 - ROWS_REMOVED_BY_58BBC15 == EXPECTED_ROWS, (
        f"{ROWS_BEFORE_58BBC15} - {ROWS_REMOVED_BY_58BBC15} != {EXPECTED_ROWS}; "
        "the recorded history no longer reconciles with the pinned row count."
    )
    assert CURRENT_GENERATOR_ROWS - EXPECTED_ROWS == KNOWN_REGENERATION_SURPLUS, (
        f"{CURRENT_GENERATOR_ROWS} - {EXPECTED_ROWS} != {KNOWN_REGENERATION_SURPLUS}; "
        "the accepted divergence from a generator re-run has changed size."
    )
    assert KNOWN_REGENERATION_SURPLUS > 0, (
        "A current re-run is expected to be a strict SUPERSET of the committed "
        "file. If that is no longer true, the divergence has changed shape and "
        "the provenance record needs re-deriving, not updating."
    )
