"""Scoring-pool identity recorded by src/verify/promote_gnn.py.

Gate 1's splitter precondition proves HOW the folds were built. It proves
nothing about WHICH ROWS were in them, and both GNN training entry points stamp
identical fold/splitter columns through one shared build_oof_records helper,
with the v1 path writing the default OOF_PATH. A frame computed over the wrong
corpus therefore clears that precondition indistinguishably. These tests pin the
digest that closes the gap.

EVERY DIGEST BELOW IS A LITERAL CONSTANT, NEVER A HASH COMPARED AGAINST ITSELF.
A test of the form `sha256(x) == sha256(x)` is tautological and survives
switching the implementation to md5; this file exists partly because that exact
test already shipped here once. The literals below were measured against the
implementation under test, and each one names the mutation it detects.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import src.verify.promote_gnn as pgnn
from src.verify.promote_gnn import (
    GateResult,
    POOL_DIGEST_VERSION,
    POOL_DRIFT_MATCH,
    POOL_DRIFT_UNPINNED,
    POOL_IDENTITY_COLUMNS,
    _pool_row_bytes,
    check_promotion_gates,
    format_pool_identity,
    oof_pool_identity,
    pool_identity_drift,
)

# ---------------------------------------------------------------------------
# Pinned digests
#
# Measured against the shipped implementation over the three-row fixture below.
# They are the whole point of this file: each is sensitive to a specific
# mutation, named in the test that asserts it.
# ---------------------------------------------------------------------------

PAIR_DIGEST = "0844d329dda07b7f140834dd70fbcaa64ab3db9edb7df5d17b5878d00b4e2d7d"
PEPTIDE_ONLY_DIGEST = "735e11d78e5fdb9dd9e3d163e61092259290f831c35b82db894062cc0b3b94fb"
DUPLICATED_ROW_DIGEST = "205849e6aae642e67a44a94fc7486fa46ee1c9078ca4af17045fec8d6541823e"
NULL_ALLELE_DIGEST = "c1faffbcfed631d94ffce3088985d7b1796a5e313bf40ac8ee1b5e9ebedd3aac"


def _fixture() -> pd.DataFrame:
    """Three rows in the schema src/train_gnn.py's build_oof_records writes.

    Deliberately NOT in sorted order: 'SIINFEKL' sorts after both of the others,
    so a digest computed without the sort differs from the pinned value even for
    this canonical frame.
    """
    return pd.DataFrame(
        [
            ("SIINFEKL", "HLA-A*02:01", 1),
            ("GILGFVFTL", "HLA-A*02:01", 1),
            ("NLVPMVATV", "HLA-B*07:02", 0),
        ],
        columns=["peptide", "hla_allele", "label"],
    )


# ---------------------------------------------------------------------------
# The pinned digest
# ---------------------------------------------------------------------------


def test_pool_digest_equals_a_pinned_literal():
    """FAILS IF: the hash is switched to md5/blake2b, the version preamble is
    dropped or renamed, the field/record separators change, the null-field
    normalisation changes, or the sort is removed."""
    identity = oof_pool_identity(_fixture())

    assert identity.digest == PAIR_DIGEST
    assert identity.n_rows == 3
    assert identity.n_positives == 2
    assert identity.columns == ("peptide", "hla_allele")
    # A sha256 hexdigest is 64 hex characters; md5 is 32. Pinning the length
    # separately makes an algorithm swap fail loudly rather than only by value.
    assert len(identity.digest) == 64


def test_pool_digest_is_independent_of_row_order():
    """FAILS IF: _sha256_pool stops sorting its records.

    Both frames are asserted against the SAME pinned literal rather than
    against each other, so a no-op implementation returning a constant cannot
    satisfy this while satisfying the tests above.
    """
    df = _fixture()
    shuffled = df.iloc[[2, 0, 1]].reset_index(drop=True)

    assert oof_pool_identity(df).digest == PAIR_DIGEST
    assert oof_pool_identity(shuffled).digest == PAIR_DIGEST


def test_pool_digest_survives_a_csv_round_trip():
    """FAILS IF: the encoding starts depending on a dtype that read_csv changes.

    The artifact is a CSV on disk, so the digest of a frame in memory and the
    digest of that frame written and read back must be the same number.
    """
    df = _fixture()
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    reloaded = pd.read_csv(io.StringIO(buffer.getvalue()))

    assert oof_pool_identity(reloaded).digest == PAIR_DIGEST


# ---------------------------------------------------------------------------
# What the digest must distinguish
# ---------------------------------------------------------------------------


def test_a_peptide_only_frame_gets_a_different_digest():
    """FAILS IF: the covered column names stop being bound into the preamble.

    'hla_allele' is optional in the artifact - build_oof_records writes it only
    when the corpus supplies it, and the tracked models/gnn_oof_predictions.csv
    does not carry it. A peptide-only digest and a peptide-plus-allele digest
    over the same peptides must never be equal, or an allele-less frame could
    be reported as matching an allele-bearing expectation.
    """
    identity = oof_pool_identity(_fixture()[["peptide", "label"]])

    assert identity.columns == ("peptide",)
    assert identity.digest == PEPTIDE_ONLY_DIGEST
    assert identity.digest != PAIR_DIGEST


def test_pool_digest_preserves_row_multiplicity():
    """FAILS IF: the records are de-duplicated instead of sorted.

    The pre-registration counts ROWS, so a frame carrying a row twice is a
    different pool even though its set of distinct rows is unchanged.
    """
    df = _fixture()
    duplicated = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    identity = oof_pool_identity(duplicated)

    assert identity.n_rows == 4
    assert identity.digest == DUPLICATED_ROW_DIGEST
    assert identity.digest != PAIR_DIGEST


def test_a_null_allele_is_normalised_and_does_not_render_as_nan():
    """FAILS IF: a null field is stringified with str(), which yields 'nan'.

    A CSV round trip cannot distinguish an empty allele field from a missing
    one, so the two must hash identically - but neither may collide with a real
    allele, and neither may depend on how pandas renders a float NaN.
    """
    df = _fixture()
    with_nan = df.copy()
    with_nan.loc[1, "hla_allele"] = np.nan
    with_empty = df.copy()
    with_empty.loc[1, "hla_allele"] = ""

    assert oof_pool_identity(with_nan).digest == NULL_ALLELE_DIGEST
    assert oof_pool_identity(with_empty).digest == NULL_ALLELE_DIGEST
    assert oof_pool_identity(with_nan).digest != PAIR_DIGEST


def test_pool_digest_covers_the_whole_frame_not_a_64_kib_prefix():
    """FAILS IF: the digest is computed from a bounded read of its input.

    src/train_gnn.py's dataset cache tag is a bare open(...).read(65536), so it
    fingerprints only the first 64 KiB. This test makes that mutation
    detectable here: the two frames below produce byte streams whose first
    64 KiB are IDENTICAL and which differ only near the end, so a 64 KiB-bounded
    digest would report them as the same pool.
    """
    n = 5000
    big = pd.DataFrame(
        {
            "peptide": [f"PEP{i:05d}AAAAA" for i in range(n)],
            "hla_allele": ["HLA-A*02:01"] * n,
            "label": [i % 2 for i in range(n)],
        }
    )
    mutated = big.copy()
    mutated.loc[n - 1, "peptide"] = "ZZZ99999AAAAA"

    columns = ("peptide", "hla_allele")
    big_stream = b"".join(sorted(_pool_row_bytes(big, columns)))
    mutated_stream = b"".join(sorted(_pool_row_bytes(mutated, columns)))

    # The premise of the test, asserted rather than assumed.
    assert len(big_stream) > 65536
    assert big_stream[:65536] == mutated_stream[:65536]
    assert big_stream != mutated_stream

    assert oof_pool_identity(big).digest != oof_pool_identity(mutated).digest


# ---------------------------------------------------------------------------
# What the digest deliberately does NOT cover
# ---------------------------------------------------------------------------


def test_labels_are_outside_the_digest_but_inside_the_census():
    """FAILS IF: 'label' is added to POOL_IDENTITY_COLUMNS, or n_positives is
    dropped.

    The digest answers "which rows", so a label flip must not move it - the two
    tracked OOF frames in this repo already spell the same logical label '0.0'
    and '0'. The cost is that a label flip is invisible to the digest, which is
    exactly why n_positives is recorded next to it and must move.
    """
    df = _fixture()
    flipped = df.copy()
    flipped.loc[0, "label"] = 0

    assert "label" not in POOL_IDENTITY_COLUMNS
    assert oof_pool_identity(flipped).digest == PAIR_DIGEST
    assert oof_pool_identity(df).n_positives == 2
    assert oof_pool_identity(flipped).n_positives == 1


def test_scores_do_not_move_the_digest():
    """FAILS IF: 'gnn_oof_score' is folded into the identity columns.

    A score-bearing digest would be a digest over float repr formatting and
    would change on a harmless writer change.
    """
    df = _fixture()
    df["gnn_oof_score"] = [0.9, 0.8, 0.1]
    rescored = df.copy()
    rescored["gnn_oof_score"] = [0.1, 0.2, 0.3]

    assert oof_pool_identity(df).digest == PAIR_DIGEST
    assert oof_pool_identity(rescored).digest == PAIR_DIGEST


def test_a_frame_with_no_identity_column_reports_unavailable():
    """FAILS IF: an identity-less frame is given a digest anyway.

    Hashing an empty record set would return one constant for every such frame,
    which reads as "these pools match". The absence has to be stated.
    """
    identity = oof_pool_identity(_fixture()[["label"]])

    assert identity.columns == ()
    assert identity.digest == ""
    assert "UNAVAILABLE" in format_pool_identity(identity)


# ---------------------------------------------------------------------------
# Soft comparison - recorded, never gated
# ---------------------------------------------------------------------------


def test_nothing_is_pinned_on_the_shipped_module():
    """FAILS IF: an expected pool is pinned without updating this file.

    Pinning is an owner decision that has to be taken together with ratifying
    docs/gnn_gate_retry_preregistration.md; it must not arrive silently.
    """
    assert pgnn.EXPECTED_POOL_ROWS is None
    assert pgnn.EXPECTED_POOL_DIGEST is None
    assert pool_identity_drift(oof_pool_identity(_fixture())) == POOL_DRIFT_UNPINNED


def test_drift_is_reported_when_the_pinned_pool_does_not_match():
    """FAILS IF: the soft comparison stops comparing, or reports a mismatch as
    a match."""
    identity = oof_pool_identity(_fixture())

    with patch.object(pgnn, "EXPECTED_POOL_ROWS", 4242), patch.object(
        pgnn, "EXPECTED_POOL_DIGEST", "0" * 64
    ):
        status = pool_identity_drift(identity)

    assert status.startswith("DRIFT")
    assert "n_rows 3 != expected 4242" in status
    assert PAIR_DIGEST in status


def test_a_matching_pin_reports_match_not_unpinned():
    """FAILS IF: 'nothing pinned' and 'matches the pin' collapse into one
    value. They are opposite facts."""
    identity = oof_pool_identity(_fixture())

    with patch.object(pgnn, "EXPECTED_POOL_ROWS", 3), patch.object(
        pgnn, "EXPECTED_POOL_DIGEST", PAIR_DIGEST
    ):
        status = pool_identity_drift(identity)

    assert status == POOL_DRIFT_MATCH
    assert status != POOL_DRIFT_UNPINNED


# ---------------------------------------------------------------------------
# Runner wiring
# ---------------------------------------------------------------------------


def _mock_checkpoint() -> MagicMock:
    p = MagicMock(spec=Path)
    p.exists.return_value = True
    return p


def _passing(name: str) -> GateResult:
    return GateResult(name=name, passed=True, value=0.9, threshold=">= 0.85")


def _run_scorecard(df: pd.DataFrame) -> bool:
    with (
        patch.object(pgnn, "GNN_CHECKPOINT", _mock_checkpoint()),
        patch("src.verify.promote_gnn._load_oof", return_value=df),
        patch("src.verify.promote_gnn.gate1_generalization", return_value=_passing("Gate 1")),
        patch("src.verify.promote_gnn.gate2_stability", return_value=_passing("Gate 2")),
        patch("src.verify.promote_gnn.gate4_calibration", return_value=_passing("Gate 4")),
        patch("src.verify.promote_gnn.gate5_escape_sensitivity", return_value=_passing("Gate 5")),
        patch("src.verify.promote_gnn.gate3_latency", return_value=_passing("Gate 3")),
    ):
        return check_promotion_gates()


def test_the_scorecard_prints_the_measured_pool_digest(caplog):
    """FAILS IF: check_promotion_gates stops recording the pool identity, or
    records a placeholder instead of the digest of the frame it scored.

    The assertion is on the LITERAL digest of the frame handed to the runner,
    not on the presence of a keyword, so a line reading 'pool_sha256=TODO'
    fails.
    """
    caplog.set_level(logging.INFO, logger="gnn-promote")

    assert _run_scorecard(_fixture()) is True

    text = caplog.text
    assert PAIR_DIGEST in text
    assert "n_rows=3" in text
    assert "n_positives=2" in text
    assert "identity_columns=peptide+hla_allele" in text


def test_a_drifting_pool_warns_but_does_not_change_the_verdict(caplog):
    """FAILS IF: the pool comparison is promoted to a gate.

    It is instrumentation. Every real gate here passes, so the verdict must
    stay True while the drift is still reported at WARNING.
    """
    caplog.set_level(logging.INFO, logger="gnn-promote")

    with patch.object(pgnn, "EXPECTED_POOL_ROWS", 4242), patch.object(
        pgnn, "EXPECTED_POOL_DIGEST", "0" * 64
    ):
        verdict = _run_scorecard(_fixture())

    assert verdict is True
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert "DRIFT" in caplog.text
    assert "ALL GATES PASSED" in caplog.text


def test_the_digest_version_is_stated_in_the_module():
    """FAILS IF: the encoding changes without bumping the version tag.

    The tag is hashed into the preamble, so bumping it changes every digest -
    which is the point: a recorded digest is only meaningful next to the
    encoding that produced it.
    """
    assert POOL_DIGEST_VERSION == "sestrav-oof-pool-v1"
    assert oof_pool_identity(_fixture()).digest == PAIR_DIGEST
