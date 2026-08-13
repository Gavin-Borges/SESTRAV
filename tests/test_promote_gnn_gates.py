"""Unit tests for src/verify/promote_gnn.py promotion gate logic.

Gates 1, 2, 4, 5 are pure DataFrame operations with no model-file dependency.
Gate 3 (latency) requires real model checkpoints and is integration-only.
"""

import numpy as np
import pandas as pd

from src.verify.promote_gnn import (
    _sha256_file,
    gate1_generalization,
    gate2_stability,
    gate4_calibration,
    gate5_escape_sensitivity,
    grouped_splitter_violation,
    GATE1_AUC_PR_MIN,
    GATE2_STD_MAX,
    GATE4_ECE_MAX,
    GATE5_SENSITIVITY_MIN,
    GROUPED_SPLITTERS,
    SPLITTER_COLUMN,
)

# The marker src/train_gnn.py stamps on every OOF row it writes. Pinned here
# rather than imported from src.train_gnn so these tests stay torch-free: the
# two modules must agree on the literal, and a divergence should break the
# artifact contract loudly.
GROUPED_MARKER = "PeptideGroupedKFold"


def _oof_df(
    n_pos=50,
    n_neg=50,
    pos_score_mean=0.8,
    neg_score_mean=0.2,
    seed=0,
    splitter: str | None = GROUPED_MARKER,
    n_folds: int | None = None,
):
    """Synthetic OOF frame in the schema src/train_gnn.py now writes.

    splitter=None reproduces the pre-repair artifact shape (no provenance
    column at all), which is exactly what the tracked v4 CSV carries.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([1] * n_pos + [0] * n_neg)
    scores = np.concatenate(
        [
            rng.normal(pos_score_mean, 0.05, n_pos).clip(0, 1),
            rng.normal(neg_score_mean, 0.05, n_neg).clip(0, 1),
        ]
    )
    df = pd.DataFrame({"label": labels, "gnn_oof_score": scores})
    if splitter is not None:
        df[SPLITTER_COLUMN] = splitter
    if n_folds is not None:
        df["fold"] = np.tile(np.arange(1, n_folds + 1), len(df) // n_folds + 1)[: len(df)]
    return df


# ---------------------------------------------------------------------------
# Gate 1 - splitter precondition
#
# This is the defect the gate existed without: the tracked v4 OOF artifact
# (columns peptide,label,gnn_oof_score) was produced by an UNGROUPED
# StratifiedKFold, scores AUC-PR 0.7160, and therefore cleared the 0.65
# threshold that is anchored on the peptide-grouped RF baseline of 0.6058.
# Gate 1 must now refuse to score a frame that cannot demonstrate its splitter.
# ---------------------------------------------------------------------------


def test_gate1_fails_on_an_unmarked_frame_the_shape_of_the_tracked_v4_artifact():
    """A frame with no splitter column FAILS - it is not skipped, and not passed.

    Scores are deliberately excellent (AUC-PR near 1.0, far above
    GATE1_AUC_PR_MIN), so the only thing that can fail this gate is the missing
    provenance. That is the point: a high number from an unknown splitter is
    not evidence of generalization.
    """
    df = _oof_df(pos_score_mean=0.95, neg_score_mean=0.05, splitter=None)
    assert SPLITTER_COLUMN not in df.columns
    assert list(df.columns) == ["label", "gnn_oof_score"]

    r = gate1_generalization(df)

    assert not r.passed
    assert "NOT PEPTIDE-GROUPED" in str(r.value)
    assert SPLITTER_COLUMN in str(r.value)


def test_gate1_fails_on_a_frame_marked_with_an_ungrouped_splitter():
    df = _oof_df(pos_score_mean=0.95, neg_score_mean=0.05, splitter="StratifiedKFold")
    r = gate1_generalization(df)
    assert not r.passed
    assert "StratifiedKFold" in str(r.value)


def test_gate1_fails_on_a_frame_with_mixed_splitter_provenance():
    """One grouped half plus one ungrouped half is not a grouped frame."""
    df = _oof_df(pos_score_mean=0.95, neg_score_mean=0.05)
    df.loc[df.index[:10], SPLITTER_COLUMN] = "StratifiedKFold"
    r = gate1_generalization(df)
    assert not r.passed


def test_gate1_fails_when_the_splitter_column_is_entirely_null():
    df = _oof_df(pos_score_mean=0.95, neg_score_mean=0.05)
    df[SPLITTER_COLUMN] = np.nan
    r = gate1_generalization(df)
    assert not r.passed
    assert "null" in str(r.value)


def test_gate1_failure_value_is_never_a_score():
    """The leaked AUC-PR must not be reported alongside a threshold it did not meet."""
    df = _oof_df(pos_score_mean=0.95, neg_score_mean=0.05, splitter=None)
    r = gate1_generalization(df)
    assert not isinstance(r.value, float)


def test_grouped_splitter_violation_accepts_every_declared_grouped_splitter():
    for name in GROUPED_SPLITTERS:
        df = _oof_df(splitter=name)
        assert grouped_splitter_violation(df) is None


def test_the_marker_train_gnn_writes_is_accepted():
    assert GROUPED_MARKER in GROUPED_SPLITTERS


# ---------------------------------------------------------------------------
# Gate 1 - Generalization (AUC-PR), on properly marked frames
# ---------------------------------------------------------------------------


def test_gate1_passes_on_good_predictions():
    df = _oof_df(pos_score_mean=0.9, neg_score_mean=0.1)
    r = gate1_generalization(df)
    assert r.passed
    assert isinstance(r.value, float)
    assert r.value >= GATE1_AUC_PR_MIN


def test_gate1_fails_on_random_predictions():
    rng = np.random.default_rng(1)
    n = 100
    df = pd.DataFrame(
        {
            "label": rng.integers(0, 2, n),
            "gnn_oof_score": rng.uniform(0, 1, n),
            SPLITTER_COLUMN: GROUPED_MARKER,
        }
    )
    r = gate1_generalization(df)
    # AUC-PR near chance (about 0.5) should fail the GATE1_AUC_PR_MIN threshold
    assert not r.passed
    assert isinstance(r.value, float)


# ---------------------------------------------------------------------------
# Gate 2 - Stability (cross-fold AUC-PR std)
# ---------------------------------------------------------------------------


def test_gate2_passes_with_stable_folds():
    rng = np.random.default_rng(2)
    n = 200
    labels = np.array([1, 0] * (n // 2))
    scores = np.where(labels == 1, rng.normal(0.85, 0.02, n), rng.normal(0.15, 0.02, n))
    scores = scores.clip(0, 1)
    folds = np.tile(np.arange(1, 6), n // 5)[:n]
    df = pd.DataFrame({"label": labels, "gnn_oof_score": scores, "fold": folds})
    r = gate2_stability(df)
    assert r.passed
    assert isinstance(r.value, float)
    assert r.value <= GATE2_STD_MAX
    assert "per-fold over 5 folds" in r.name


def test_gate2_fails_without_a_fold_column():
    """The phantom --save-fold-ids flag is gone; no fold identity means no verdict.

    The old fallback computed a jackknife leave-one-ROW-out std, which is the
    standard error of a single pooled AUC-PR rather than the spread across
    folds, and it always ran because the flag the docstring named never
    existed. Cross-fold stability is not computable from an unfolded frame, so
    the gate fails rather than substituting a different statistic.
    """
    df = _oof_df(n_pos=30, n_neg=30, pos_score_mean=0.9, neg_score_mean=0.1)
    assert "fold" not in df.columns

    r = gate2_stability(df)

    assert not r.passed
    assert "fold" in str(r.value)
    assert "jackknife" not in r.name.lower()
    assert "jackknife" not in str(r.value).lower()


def test_gate2_docstring_disclaims_the_phantom_flag():
    """If --save-fold-ids is named at all, it must be named as nonexistent."""
    doc = gate2_stability.__doc__ or ""
    if "--save-fold-ids" in doc:
        assert "never" in doc, "the docstring still describes --save-fold-ids as a real flag"


def test_gate2_fails_with_high_variance_folds():
    rng = np.random.default_rng(99)
    rows = []
    for fold in range(1, 6):
        # Alternate sharply separated / barely separated folds to inject std.
        pos_mean = 0.9 if fold % 2 == 1 else 0.12
        for _ in range(20):
            rows.append(
                {"label": 1, "gnn_oof_score": float(rng.normal(pos_mean, 0.01)), "fold": fold}
            )
        for _ in range(20):
            rows.append({"label": 0, "gnn_oof_score": float(rng.normal(0.1, 0.01)), "fold": fold})
    df = pd.DataFrame(rows)

    r = gate2_stability(df)

    assert isinstance(r.value, float)
    assert not r.passed
    assert r.value > GATE2_STD_MAX


def test_gate2_fails_when_fewer_than_two_folds_are_scoreable():
    """Single-class folds are reported, not silently dropped: dropping shrinks the std."""
    df = pd.DataFrame(
        {
            "label": [1, 1, 1, 1, 0, 1],
            "gnn_oof_score": [0.9, 0.8, 0.7, 0.6, 0.1, 0.95],
            "fold": [1, 1, 2, 2, 3, 3],
        }
    )
    r = gate2_stability(df)
    assert not r.passed
    assert "scoreable fold" in str(r.value)


# ---------------------------------------------------------------------------
# Gate 4 - Calibration (ECE)
# ---------------------------------------------------------------------------


def test_gate4_passes_on_well_calibrated_scores():
    # Construct perfectly calibrated data: within each of the 15 equal-width
    # bins the empirical positive rate matches the bin's centre probability,
    # so ECE = 0 by construction.
    bins = np.linspace(0.0, 1.0, 16)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        centre = (lo + hi) / 2.0
        n_bin = 40
        n_pos = round(centre * n_bin)
        rows.extend([{"label": 1, "gnn_oof_score": centre}] * n_pos)
        rows.extend([{"label": 0, "gnn_oof_score": centre}] * (n_bin - n_pos))
    df = pd.DataFrame(rows)
    r = gate4_calibration(df)
    assert r.passed, f"ECE={r.value} expected < {GATE4_ECE_MAX}"
    assert r.value < GATE4_ECE_MAX


def test_gate4_fails_on_overconfident_scores():
    n = 100
    # Model always predicts 0.99 but half are actually negative
    df = pd.DataFrame(
        {
            "label": np.array([1] * 50 + [0] * 50),
            "gnn_oof_score": np.full(n, 0.99),
        }
    )
    r = gate4_calibration(df)
    assert not r.passed
    assert r.value >= GATE4_ECE_MAX


# ---------------------------------------------------------------------------
# Gate 5 - Escape Sensitivity
# ---------------------------------------------------------------------------


def test_gate5_passes_when_positives_above_median_negative():
    df = _oof_df(pos_score_mean=0.9, neg_score_mean=0.2)
    r = gate5_escape_sensitivity(df)
    assert r.passed
    assert r.value >= GATE5_SENSITIVITY_MIN


def test_gate5_fails_when_positives_below_median_negative():
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "label": np.array([1] * 50 + [0] * 50),
            # positives have low score, negatives have high score (reversed)
            "gnn_oof_score": np.concatenate(
                [
                    rng.normal(0.1, 0.05, 50).clip(0, 1),
                    rng.normal(0.9, 0.05, 50).clip(0, 1),
                ]
            ),
        }
    )
    r = gate5_escape_sensitivity(df)
    assert not r.passed


def test_gate5_returns_failure_on_single_class():
    df = pd.DataFrame(
        {
            "label": np.ones(50),
            "gnn_oof_score": np.linspace(0.5, 0.9, 50),
        }
    )
    r = gate5_escape_sensitivity(df)
    assert not r.passed
    assert "Insufficient" in str(r.value)


# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------


def test_sha256_file_is_deterministic(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"SESTRAV test content")
    assert _sha256_file(p) == _sha256_file(p)


def test_sha256_file_changes_with_content(tmp_path):
    p1 = tmp_path / "a.bin"
    p2 = tmp_path / "b.bin"
    p1.write_bytes(b"content A")
    p2.write_bytes(b"content B")
    assert _sha256_file(p1) != _sha256_file(p2)
