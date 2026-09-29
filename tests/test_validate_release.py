"""Cover the release pre-flight gate's feature-count check.

Why this file exists: `src/ci/validate_release.py` compared `model.n_features_in_`
against a hardcoded `(21, 30, 50)`. That tuple predates feature mode 31, which is
what `config.yaml` declares and what `models/rf_31feature_integrated.joblib`
actually carries, so the gate rejected the model the project ships. It is not dead
code either: `run_pipeline.sh` runs it under `set -euo pipefail`, so the whole
pipeline aborted before `pipeline.py` started. Measured 2026-09-17, the gate exited
1 with "Model expects unsupported feature count: 31" after loading MHCflurry, the
config and the model successfully; the stale tuple was the only failure.

The check now reads its expected value from configuration. These cases pin that,
and the file previously had no test coverage of any kind.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.ci.validate_release import check_feature_count

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"


class _Model:
    """Minimal stand-in exposing only what the check reads."""

    def __init__(self, n_features_in_: int) -> None:
        self.n_features_in_ = n_features_in_


class _FeaturelessModel:
    """An artifact exposing no feature count, as a torch .pt checkpoint does."""


def test_agreeing_feature_count_passes_and_is_returned() -> None:
    assert check_feature_count(_Model(31), 31) == 31


def test_disagreeing_feature_count_raises_and_names_both_numbers() -> None:
    with pytest.raises(ValueError) as excinfo:
        check_feature_count(_Model(30), 31)
    message = str(excinfo.value)
    assert "30" in message and "31" in message


def test_artifact_without_a_feature_count_is_skipped_not_rejected() -> None:
    """Mirrors ModelRegistry.validate_signature, which skips rather than fails.

    Pinned because the previous implementation read `model.n_features_in_`
    directly, so a torch checkpoint would have raised AttributeError and been
    reported as a dependency failure rather than as a skipped comparison.
    """
    assert check_feature_count(_FeaturelessModel(), 31) is None


def test_the_configured_feature_mode_is_accepted_by_the_gate() -> None:
    """The regression pin: the shipped configuration must pass its own gate.

    This is the exact case the hardcoded (21, 30, 50) rejected. Reads the real
    config rather than a literal so that changing `feature_mode` cannot silently
    reintroduce the mismatch.
    """
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    feature_mode = config["feature_mode"]
    assert check_feature_count(_Model(feature_mode), feature_mode) == feature_mode


def test_the_stale_allowlist_would_have_rejected_the_shipped_configuration() -> None:
    """Documents the defect so a future edit cannot quietly restore it."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["feature_mode"] not in (21, 30, 50), (
        "config.yaml's feature_mode is back inside the retired hardcoded allowlist; "
        "the regression pin above no longer distinguishes the fix from the defect."
    )
