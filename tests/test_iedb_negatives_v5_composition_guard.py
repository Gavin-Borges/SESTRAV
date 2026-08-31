"""Composition guard for the tracked data/iedb_negatives_v5.csv artifact.

WHY THIS EXISTS. The tracked file is the current generator's adopted output.
Its composition must remain pinned so that a silent in-place edit fails loudly.

The history, established by re-derivation rather than inference:

- Commit dcbb1b1 added a 34,358-row file, written by the generator as it stood
  at that time. That run is reproducible today: checking out the pre-fix
  generator (`git show a15d05e^:scripts/ingest_iedb_negatives.py`) and running
  it on the same raw export reproduces the artifact bit-for-bit.
- Commit 58bbc15 then edited the FINISHED file in place, dropping the 1,888 rows
  whose assay_type is "biological activity" - a post-hoc filter, not a re-run.
- On 2026-08-31 the artifact was regenerated with that exclusion before
  intra-export dedup. Its 32,506 rows were verified across all 23 fields as a
  strict superset of the prior 32,470 rows: 0 absent and 36 added.

What must NOT happen again is another silent in-place edit of a finished data
file. These tests make any such edit fail loudly instead of surfacing in audit.

The regeneration was deliberately staged and has NOT been propagated downstream,
so the last test here guards the second way this could go quiet: the tracked
data/iedb_negatives_v5_merged.csv, and the shipped corpus built from it, still
descend from the superseded 32,470-row version. That lag is recorded in the
downstream_status block of the provenance sidecar and pinned below, so neither
propagating it nor widening it can happen without a test saying so.

Note on reading data/: the repository's own tooling reads these paths directly
and so does this test. Some developer shells deny-list `data/` for interactive
reads; that restriction does not apply to the test process.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NEGATIVES_CSV = REPO_ROOT / "data" / "iedb_negatives_v5.csv"
MERGED_CSV = REPO_ROOT / "data" / "iedb_negatives_v5_merged.csv"

# Every constant in this module is an INDEPENDENTLY MEASURED literal, never
# derived from another one. Deriving any of them would turn the identity test at
# the bottom into a tautology that passes for any value of EXPECTED_ROWS, which
# is precisely the decay this file exists to prevent.
#
# EXPECTED_ROWS is cited by CONTENT DIGEST, not by commit. An in-branch SHA does
# not survive a rebase and "the staged blob" stops resolving the moment the index
# is committed, so neither is a durable citation (.claude/rules/git-instruments.md
# R7 and R8). The artifact these tests run against is the one whose LF blob hashes
# to the digest below, which is also the output_checksum_sha256 recorded in
# data/iedb_negatives_v5_provenance.json:
#   git show <any ref carrying it>:data/iedb_negatives_v5.csv | sha256sum
#   -> f4b7a130b1d8d87341dc9a946f255766eacf17b896aca80dddcc85e6cfd92a41
EXPECTED_ROWS = 32506  # csv.reader over that blob, header excluded

EXCLUDED_ASSAY_TYPE = "biological activity"

# The recorded history. Each is measured from the blob named beside it, and every
# one of those commits is a permanent ancestor of main, so the citations survive a
# rebase of this branch.
ROWS_BEFORE_58BBC15 = 34358  # csv.reader over `git show 58bbc15^:data/iedb_negatives_v5.csv`
ROWS_REMOVED_BY_58BBC15 = 1888  # of those rows, the ones whose assay_type is EXCLUDED_ASSAY_TYPE
ROWS_AFTER_58BBC15 = 32470  # csv.reader over `git show 58bbc15:data/iedb_negatives_v5.csv`
REGENERATION_SURPLUS = 36  # whole-row multiset diff against that blob, all 23 fields

# The downstream lag, measured as a whole-row multiset difference over the 23
# fields the two files share. Both operands are tracked, so unlike the generator
# re-run this figure is re-derivable inside a clone with no raw export present.
# It currently equals REGENERATION_SURPLUS, because the merged file was built
# before the regeneration and so lacks exactly the rows the regeneration added.
# Those two constants are NOT the same quantity and will diverge the moment the
# merge is rebuilt, so neither is defined in terms of the other.
MERGED_ROWS = 36689
MERGED_LAG_ROWS = 36


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


def test_recorded_history_reconciles_with_the_pinned_row_count():
    """
    Keeps EXPECTED_ROWS from being a lone bare pin that no other value contradicts.

    Each constant here was measured from a different Git blob, so these are real
    cross-checks rather than restatements of one number: change EXPECTED_ROWS on
    its own and the second assertion fails, which is the property a single pinned
    literal cannot have. All three blobs are reachable from main, so this test
    needs no raw export and no network.
    """
    assert ROWS_BEFORE_58BBC15 - ROWS_REMOVED_BY_58BBC15 == ROWS_AFTER_58BBC15, (
        f"{ROWS_BEFORE_58BBC15} - {ROWS_REMOVED_BY_58BBC15} != {ROWS_AFTER_58BBC15}; "
        "the recorded history of the in-place edit at 58bbc15 no longer reconciles."
    )
    assert ROWS_AFTER_58BBC15 + REGENERATION_SURPLUS == EXPECTED_ROWS, (
        f"{ROWS_AFTER_58BBC15} + {REGENERATION_SURPLUS} != {EXPECTED_ROWS}; "
        "the 2026-08-31 regeneration no longer reconciles with the pinned row "
        "count. Re-derive it against the artifact rather than adjusting a literal "
        "to make this pass."
    )


def test_downstream_merge_lag_is_pinned_not_silent():
    """
    The 2026-08-31 regeneration was staged and stops here on purpose.

    data/iedb_negatives_v5_merged.csv is the direct consumer of this file and the
    ancestor of the shipped corpus, and it still holds the pre-regeneration rows.
    Pinning the gap means propagation cannot land unannounced and a further edit
    to either side cannot widen it unnoticed. When the merge is rebuilt, this lag
    goes to zero and this test fails by design: at that point update the guard and
    the sidecar's downstream_status block together, because the sidecar is the
    attestation.
    """
    if not MERGED_CSV.exists():
        pytest.skip(f"{MERGED_CSV.name} not present in this checkout")

    fieldnames, rows = _rows()
    assert fieldnames is not None

    with open(MERGED_CSV, encoding="utf-8", newline="") as fh:
        merged_reader = csv.DictReader(fh)
        merged_fields = merged_reader.fieldnames
        assert merged_fields == fieldnames, (
            "the merged negatives no longer share this file's schema, so the row-level "
            f"comparison below is not meaningful: {merged_fields} != {fieldnames}"
        )
        merged = Counter(tuple(r[k] for k in fieldnames) for r in merged_reader)

    assert sum(merged.values()) == MERGED_ROWS, (
        f"{MERGED_CSV.name} has {sum(merged.values())} rows, expected {MERGED_ROWS}."
    )

    ours = Counter(tuple(r[k] for k in fieldnames) for r in rows)
    lag = sum((ours - merged).values())

    assert lag == MERGED_LAG_ROWS, (
        f"{lag} rows of {NEGATIVES_CSV.name} are missing from {MERGED_CSV.name}, "
        f"expected exactly {MERGED_LAG_ROWS}. Zero means the merge was rebuilt and the "
        "staged regeneration has propagated; anything else means one side changed "
        "without the other. Either way, update this guard and the downstream_status "
        "block of data/iedb_negatives_v5_provenance.json together."
    )
