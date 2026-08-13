"""Tests for the check_promotion_gates / promote_model runner path.

The gate functions themselves are tested in test_promote_gnn_gates.py.
These tests focus on:
  - check_promotion_gates() short-circuit / aggregation logic
  - promote_model() config-mutation and checksum behaviour
All heavy I/O (torch.load, joblib, real model files) is mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import src.verify.promote_gnn as pgnn
from src.verify.promote_gnn import (
    GateResult,
    check_promotion_gates,
    promote_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_path(exists: bool = True, content: bytes = b"") -> MagicMock:
    """Return a Path-like MagicMock with a controllable .exists() and .open()."""
    p = MagicMock(spec=Path)
    p.exists.return_value = exists
    return p


def _passing_gate(name: str) -> GateResult:
    return GateResult(name=name, passed=True, value=0.9, threshold=">= 0.85")


def _failing_gate(name: str) -> GateResult:
    return GateResult(name=name, passed=False, value=0.3, threshold=">= 0.85")


def _good_oof() -> pd.DataFrame:
    """An OOF frame in the schema src/train_gnn.py writes: fold + splitter included."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "label": np.array([1] * 60 + [0] * 60),
            "gnn_oof_score": np.concatenate(
                [
                    rng.normal(0.9, 0.03, 60).clip(0, 1),
                    rng.normal(0.1, 0.03, 60).clip(0, 1),
                ]
            ),
            "fold": np.tile(np.arange(1, 6), 24),
            pgnn.SPLITTER_COLUMN: "PeptideGroupedKFold",
        }
    )


# ---------------------------------------------------------------------------
# check_promotion_gates - checkpoint-not-found path
# ---------------------------------------------------------------------------


def test_check_gates_returns_false_when_checkpoint_missing():
    with patch.object(pgnn, "GNN_CHECKPOINT", _mock_path(exists=False)):
        result = check_promotion_gates()
    assert result is False


# ---------------------------------------------------------------------------
# check_promotion_gates - OOF file not found
# ---------------------------------------------------------------------------


def test_check_gates_returns_false_when_oof_missing():
    with (
        patch.object(pgnn, "GNN_CHECKPOINT", _mock_path(exists=True)),
        patch("src.verify.promote_gnn._load_oof", side_effect=FileNotFoundError("OOF not found")),
    ):
        result = check_promotion_gates()
    assert result is False


# ---------------------------------------------------------------------------
# check_promotion_gates - OOF file has wrong schema
# ---------------------------------------------------------------------------


def test_check_gates_returns_false_when_oof_schema_bad():
    with (
        patch.object(pgnn, "GNN_CHECKPOINT", _mock_path(exists=True)),
        patch(
            "src.verify.promote_gnn._load_oof", side_effect=ValueError("OOF file missing columns")
        ),
    ):
        result = check_promotion_gates()
    assert result is False


# ---------------------------------------------------------------------------
# check_promotion_gates - a single gate fails
# ---------------------------------------------------------------------------


def test_check_gates_returns_false_when_one_gate_fails():
    with (
        patch.object(pgnn, "GNN_CHECKPOINT", _mock_path(exists=True)),
        patch("src.verify.promote_gnn._load_oof", return_value=_good_oof()),
        patch("src.verify.promote_gnn.gate1_generalization", return_value=_failing_gate("Gate 1")),
        patch("src.verify.promote_gnn.gate2_stability", return_value=_passing_gate("Gate 2")),
        patch("src.verify.promote_gnn.gate4_calibration", return_value=_passing_gate("Gate 4")),
        patch(
            "src.verify.promote_gnn.gate5_escape_sensitivity", return_value=_passing_gate("Gate 5")
        ),
        patch("src.verify.promote_gnn.gate3_latency", return_value=_passing_gate("Gate 3")),
    ):
        result = check_promotion_gates()
    assert result is False


# ---------------------------------------------------------------------------
# check_promotion_gates - all five gates pass
# ---------------------------------------------------------------------------


def test_check_gates_returns_true_when_all_pass():
    with (
        patch.object(pgnn, "GNN_CHECKPOINT", _mock_path(exists=True)),
        patch("src.verify.promote_gnn._load_oof", return_value=_good_oof()),
        patch("src.verify.promote_gnn.gate1_generalization", return_value=_passing_gate("Gate 1")),
        patch("src.verify.promote_gnn.gate2_stability", return_value=_passing_gate("Gate 2")),
        patch("src.verify.promote_gnn.gate4_calibration", return_value=_passing_gate("Gate 4")),
        patch(
            "src.verify.promote_gnn.gate5_escape_sensitivity", return_value=_passing_gate("Gate 5")
        ),
        patch("src.verify.promote_gnn.gate3_latency", return_value=_passing_gate("Gate 3")),
    ):
        result = check_promotion_gates()
    assert result is True


# ---------------------------------------------------------------------------
# check_promotion_gates - gate raises unexpectedly → treated as failure
# ---------------------------------------------------------------------------


def test_check_gates_treats_gate_exception_as_failure():
    def _raising_gate1(_df):
        raise RuntimeError("unexpected error")

    with (
        patch.object(pgnn, "GNN_CHECKPOINT", _mock_path(exists=True)),
        patch("src.verify.promote_gnn._load_oof", return_value=_good_oof()),
        patch("src.verify.promote_gnn.gate1_generalization", _raising_gate1),
        patch("src.verify.promote_gnn.gate2_stability", return_value=_passing_gate("Gate 2")),
        patch("src.verify.promote_gnn.gate4_calibration", return_value=_passing_gate("Gate 4")),
        patch(
            "src.verify.promote_gnn.gate5_escape_sensitivity", return_value=_passing_gate("Gate 5")
        ),
        patch("src.verify.promote_gnn.gate3_latency", return_value=_passing_gate("Gate 3")),
    ):
        result = check_promotion_gates()
    assert result is False


# ---------------------------------------------------------------------------
# promote_model - gates fail → config.yaml not written
# ---------------------------------------------------------------------------


def test_promote_model_does_not_modify_config_when_gates_fail(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_path: old_model.pt\n")

    with (
        patch("src.verify.promote_gnn.check_promotion_gates", return_value=False),
        patch.object(pgnn, "CONFIG_PATH", config_path),
    ):
        promote_model()

    assert config_path.read_text() == "model_path: old_model.pt\n"


# ---------------------------------------------------------------------------
# promote_model - gates pass → config.yaml updated
# ---------------------------------------------------------------------------


def test_promote_model_updates_config_when_gates_pass(tmp_path):
    import yaml

    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_path: old_model.pt\n")

    checkpoint = tmp_path / "structural_gnn_v2.pth"
    checkpoint.write_bytes(b"fake-weights")

    checksum_path = tmp_path / "checksums.json"
    mock_update = MagicMock()

    with (
        patch("src.verify.promote_gnn.check_promotion_gates", return_value=True),
        patch.object(pgnn, "CONFIG_PATH", config_path),
        patch.object(pgnn, "GNN_CHECKPOINT", checkpoint),
        patch.object(pgnn, "CHECKSUM_FILE", checksum_path),
        patch("src.artifact_integrity.update_checksum_manifest", mock_update),
    ):
        promote_model()

    updated = yaml.safe_load(config_path.read_text())
    assert updated["model_path"] == str(checkpoint)


# ---------------------------------------------------------------------------
# promote_model - --dry-run evaluates the gates without mutating anything
#
# Every case here forces gates-pass, because the interesting question is what
# happens on the ONE path that writes. A dry run that only avoids writes when
# the gates already blocked promotion would prove nothing.
# ---------------------------------------------------------------------------


def test_dry_run_leaves_config_yaml_unmodified(tmp_path):
    config_path = tmp_path / "config.yaml"
    original = "model_path: models/rf_31feature_integrated.joblib\nseed: 42\n"
    config_path.write_text(original)

    checkpoint = tmp_path / "structural_gnn_v2.pth"
    checkpoint.write_bytes(b"fake-weights")
    checksum_path = tmp_path / "checksums.json"
    mock_update = MagicMock()

    with (
        patch("src.verify.promote_gnn.check_promotion_gates", return_value=True),
        patch.object(pgnn, "CONFIG_PATH", config_path),
        patch.object(pgnn, "GNN_CHECKPOINT", checkpoint),
        patch.object(pgnn, "CHECKSUM_FILE", checksum_path),
        patch("src.artifact_integrity.update_checksum_manifest", mock_update),
    ):
        promote_model(dry_run=True)

    assert config_path.read_text() == original
    mock_update.assert_not_called()
    assert not checksum_path.exists()


def test_the_same_fixture_does_mutate_config_without_dry_run(tmp_path):
    """Companion to the test above: proves the dry run is what prevented the write."""
    import yaml

    config_path = tmp_path / "config.yaml"
    original = "model_path: models/rf_31feature_integrated.joblib\nseed: 42\n"
    config_path.write_text(original)

    checkpoint = tmp_path / "structural_gnn_v2.pth"
    checkpoint.write_bytes(b"fake-weights")
    checksum_path = tmp_path / "checksums.json"
    mock_update = MagicMock()

    with (
        patch("src.verify.promote_gnn.check_promotion_gates", return_value=True),
        patch.object(pgnn, "CONFIG_PATH", config_path),
        patch.object(pgnn, "GNN_CHECKPOINT", checkpoint),
        patch.object(pgnn, "CHECKSUM_FILE", checksum_path),
        patch("src.artifact_integrity.update_checksum_manifest", mock_update),
    ):
        promote_model(dry_run=False)

    assert config_path.read_text() != original
    assert yaml.safe_load(config_path.read_text())["model_path"] == str(checkpoint)
    mock_update.assert_called_once()


def test_promote_model_defaults_to_writing_so_dry_run_must_be_explicit(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_path: old.pt\n")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"weights")

    with (
        patch("src.verify.promote_gnn.check_promotion_gates", return_value=True),
        patch.object(pgnn, "CONFIG_PATH", config_path),
        patch.object(pgnn, "GNN_CHECKPOINT", checkpoint),
        patch.object(pgnn, "CHECKSUM_FILE", tmp_path / "checksums.json"),
        patch("src.artifact_integrity.update_checksum_manifest", MagicMock()),
    ):
        promote_model()

    assert config_path.read_text() != "model_path: old.pt\n"


def test_cli_exposes_dry_run_and_defaults_it_off():
    parser = pgnn._build_arg_parser()
    assert parser.parse_args([]).dry_run is False
    assert parser.parse_args(["--dry-run"]).dry_run is True


# ---------------------------------------------------------------------------
# --oof: scoring a scratch OOF frame without dirtying the tracked artifact
#
# Without this override, OOF_PATH is a module constant with no CLI access, so a
# scratch GNN run could only be scored by first overwriting
# models/gnn_oof_predictions.csv - which is exactly the tracked artifact a
# scratch run must not touch. The flag must therefore be advertised, parsed AND
# forwarded; the first two without the third is a silent no-op.
# ---------------------------------------------------------------------------


def test_cli_exposes_oof_and_defaults_it_to_none():
    parser = pgnn._build_arg_parser()
    assert parser.parse_args([]).oof is None
    assert parser.parse_args(["--oof", "models/scratch/run/oof.csv"]).oof == Path(
        "models/scratch/run/oof.csv"
    )


def test_main_forwards_oof_into_promote_model():
    """Advertising and parsing the flag is not the same as wiring it."""
    source = (Path(pgnn.__file__)).read_text(encoding="utf-8")
    call_start = source.index("    promote_model(")
    call_block = source[call_start : source.index(")", call_start)]
    assert "oof_path=_args.oof" in call_block, (
        "__main__ parses --oof but never forwards it to promote_model"
    )


def test_promote_model_threads_oof_path_into_check_promotion_gates():
    seen: dict[str, object] = {}

    def _capture(oof_path=None):
        seen["oof_path"] = oof_path
        return False  # gates fail, so nothing is written

    with patch("src.verify.promote_gnn.check_promotion_gates", _capture):
        promote_model(oof_path=Path("models/scratch/run/oof.csv"))

    assert seen["oof_path"] == Path("models/scratch/run/oof.csv")


def test_check_promotion_gates_threads_oof_path_into_load_oof():
    seen: dict[str, object] = {}

    def _capture(oof_path=None):
        seen["oof_path"] = oof_path
        return _good_oof()

    with (
        patch.object(pgnn, "GNN_CHECKPOINT", _mock_path(exists=True)),
        patch("src.verify.promote_gnn._load_oof", _capture),
        patch("src.verify.promote_gnn.gate1_generalization", return_value=_passing_gate("Gate 1")),
        patch("src.verify.promote_gnn.gate2_stability", return_value=_passing_gate("Gate 2")),
        patch("src.verify.promote_gnn.gate4_calibration", return_value=_passing_gate("Gate 4")),
        patch(
            "src.verify.promote_gnn.gate5_escape_sensitivity", return_value=_passing_gate("Gate 5")
        ),
        patch("src.verify.promote_gnn.gate3_latency", return_value=_passing_gate("Gate 3")),
    ):
        check_promotion_gates(Path("models/scratch/run/oof.csv"))

    assert seen["oof_path"] == Path("models/scratch/run/oof.csv")


def test_load_oof_reads_the_override_and_leaves_the_default_untouched(tmp_path):
    """The override must win even when the tracked default also exists.

    An implementation that reads OOF_PATH unconditionally would still PASS a
    test where the default is missing, because the override would only be
    exercised on the error path. Both files exist here, with different content.
    """
    from src.verify.promote_gnn import _load_oof

    tracked = tmp_path / "tracked_oof.csv"
    tracked.write_text("label,gnn_oof_score\n1,0.11\n0,0.12\n")
    scratch = tmp_path / "scratch_oof.csv"
    scratch.write_text("label,gnn_oof_score\n1,0.91\n0,0.92\n1,0.93\n")

    with patch.object(pgnn, "OOF_PATH", tracked):
        overridden = _load_oof(scratch)
        default = _load_oof()

    assert len(overridden) == 3
    assert overridden["gnn_oof_score"].iloc[0] == 0.91
    assert len(default) == 2, "passing no override must still read OOF_PATH"
    assert tracked.read_text() == "label,gnn_oof_score\n1,0.11\n0,0.12\n"


# ---------------------------------------------------------------------------
# promote_model - missing config.yaml is handled gracefully (warning only)
# ---------------------------------------------------------------------------


def test_promote_model_no_config_does_not_crash(tmp_path):
    checkpoint = tmp_path / "structural_gnn_v2.pth"
    checkpoint.write_bytes(b"fake-weights")
    checksum_path = tmp_path / "checksums.json"
    absent_config = tmp_path / "nonexistent_config.yaml"

    mock_update = MagicMock()

    with (
        patch("src.verify.promote_gnn.check_promotion_gates", return_value=True),
        patch.object(pgnn, "CONFIG_PATH", absent_config),
        patch.object(pgnn, "GNN_CHECKPOINT", checkpoint),
        patch.object(pgnn, "CHECKSUM_FILE", checksum_path),
        patch("src.artifact_integrity.update_checksum_manifest", mock_update),
    ):
        promote_model()  # must not raise


# ---------------------------------------------------------------------------
# _load_oof - direct unit tests
# ---------------------------------------------------------------------------


def test_load_oof_raises_when_file_missing(tmp_path):
    from src.verify.promote_gnn import _load_oof
    import pytest

    with (
        patch.object(pgnn, "OOF_PATH", tmp_path / "nonexistent.csv"),
    ):
        with pytest.raises(FileNotFoundError):
            _load_oof()


def test_load_oof_raises_when_columns_missing(tmp_path):
    from src.verify.promote_gnn import _load_oof
    import pytest

    bad_csv = tmp_path / "oof.csv"
    bad_csv.write_text("score,other\n0.9,x\n")

    with patch.object(pgnn, "OOF_PATH", bad_csv):
        with pytest.raises(ValueError, match="missing columns"):
            _load_oof()


def test_load_oof_returns_dataframe_on_valid_file(tmp_path):
    from src.verify.promote_gnn import _load_oof

    oof_csv = tmp_path / "oof.csv"
    oof_csv.write_text("label,gnn_oof_score\n1,0.9\n0,0.1\n")

    with patch.object(pgnn, "OOF_PATH", oof_csv):
        df = _load_oof()

    assert list(df.columns) >= ["label", "gnn_oof_score"]
    assert len(df) == 2


# ---------------------------------------------------------------------------
# _time_model_ms - direct unit test
# ---------------------------------------------------------------------------


def test_time_model_ms_returns_positive_float():
    from src.verify.promote_gnn import _time_model_ms
    import torch

    x = torch.zeros(2, 3)
    ms = _time_model_ms(lambda a, b: a + b, x, x, warmup=1, reps=3)
    assert isinstance(ms, float)
    assert ms >= 0.0


# ---------------------------------------------------------------------------
# check_promotion_gates - gate3_latency exception handler (lines 350-352)
# ---------------------------------------------------------------------------


def test_check_gates_treats_gate3_exception_as_failure():
    def _raise_gate3():
        raise RuntimeError("GPU OOM in gate3")

    with (
        patch.object(pgnn, "GNN_CHECKPOINT", _mock_path(exists=True)),
        patch("src.verify.promote_gnn._load_oof", return_value=_good_oof()),
        patch("src.verify.promote_gnn.gate1_generalization", return_value=_passing_gate("Gate 1")),
        patch("src.verify.promote_gnn.gate2_stability", return_value=_passing_gate("Gate 2")),
        patch("src.verify.promote_gnn.gate4_calibration", return_value=_passing_gate("Gate 4")),
        patch(
            "src.verify.promote_gnn.gate5_escape_sensitivity", return_value=_passing_gate("Gate 5")
        ),
        patch("src.verify.promote_gnn.gate3_latency", _raise_gate3),
    ):
        result = check_promotion_gates()
    assert result is False


# ---------------------------------------------------------------------------
# promote_model - update_checksum_manifest raises → re-raised (lines 407-409)
# ---------------------------------------------------------------------------


def test_promote_model_reraises_on_checksum_failure(tmp_path):
    import pytest

    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_path: old.pt\n")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"weights")
    checksum_path = tmp_path / "checksums.json"

    with (
        patch("src.verify.promote_gnn.check_promotion_gates", return_value=True),
        patch.object(pgnn, "CONFIG_PATH", config_path),
        patch.object(pgnn, "GNN_CHECKPOINT", checkpoint),
        patch.object(pgnn, "CHECKSUM_FILE", checksum_path),
        patch(
            "src.artifact_integrity.update_checksum_manifest", side_effect=RuntimeError("disk full")
        ),
        pytest.raises(RuntimeError, match="disk full"),
    ):
        promote_model()


# ---------------------------------------------------------------------------
# gate3_latency - GNN v2.1: PyG batch must be passed to GraphPredictorV2.forward()
# ---------------------------------------------------------------------------


def test_gate3_latency_passes_pyg_batch_to_forward():
    """gate3_latency must supply a PyG Data batch to GraphPredictorV2.forward().

    Uses GraphPredictorV2 (GINEConv + ESM-2) which accepts a batched Data object
    rather than the (node_x, feat_x, adj) signature of the v1 GraphPredictor.
    """
    from src.gnn.models import GraphPredictorV2
    from src.features import TRAIN_FEATURE_COLUMNS
    from src.verify.promote_gnn import gate3_latency

    n_features = len(TRAIN_FEATURE_COLUMNS)
    real_state = GraphPredictorV2(num_continuous_features=n_features).state_dict()

    class _FakeRF:
        n_features_in_ = n_features

        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    with (
        patch.object(pgnn, "GNN_CHECKPOINT", _mock_path(exists=True)),
        patch.object(pgnn, "RF_MODEL_PATH", _mock_path(exists=True)),
        patch.object(pgnn, "GNN_CONFIG", _mock_path(exists=False)),
        patch("src.artifact_integrity.load_verified_joblib", return_value=_FakeRF()),
        patch("torch.load", return_value=real_state),
    ):
        result = gate3_latency()

    assert isinstance(result, GateResult)
    # A successful run produces "GNN=...ms, RF=...ms, ratio=...×" as the value
    assert "ratio=" in str(result.value)


# ---------------------------------------------------------------------------
# gate3_latency - RF model not found path
# ---------------------------------------------------------------------------


def test_gate3_latency_fails_when_rf_missing():
    from src.verify.promote_gnn import gate3_latency

    with (
        patch.object(pgnn, "GNN_CHECKPOINT", _mock_path(exists=True)),
        patch.object(pgnn, "RF_MODEL_PATH", _mock_path(exists=False)),
    ):
        result = gate3_latency()

    assert result.passed is False
    assert "RF model not found" in str(result.value)


# ---------------------------------------------------------------------------
# gate3_latency - GNN checkpoint not found path
# ---------------------------------------------------------------------------


def test_gate3_latency_fails_when_gnn_missing():
    import numpy as np
    from src.features import TRAIN_FEATURE_COLUMNS
    from src.verify.promote_gnn import gate3_latency

    n_features = len(TRAIN_FEATURE_COLUMNS)

    class _FakeRF:
        n_features_in_ = n_features

        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    with (
        patch.object(pgnn, "GNN_CHECKPOINT", _mock_path(exists=False)),
        patch.object(pgnn, "RF_MODEL_PATH", _mock_path(exists=True)),
        patch("src.artifact_integrity.load_verified_joblib", return_value=_FakeRF()),
    ):
        result = gate3_latency()

    assert result.passed is False
    assert "GNN checkpoint not found" in str(result.value)
