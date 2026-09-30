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

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.ci.validate_release import check_feature_count
from src.features import FEATURE_COLUMNS_31, FEATURE_COLUMNS_51

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
    shipped_feature_count = len(FEATURE_COLUMNS_31)
    assert check_feature_count(_Model(shipped_feature_count), feature_mode) == shipped_feature_count


def test_mode_51_maps_to_its_55_canonical_columns() -> None:
    assert len(FEATURE_COLUMNS_51) == 55
    assert check_feature_count(_Model(55), 51) == 55
    with pytest.raises(ValueError, match="requires 55"):
        check_feature_count(_Model(51), 51)


def test_the_stale_allowlist_would_have_rejected_the_shipped_configuration() -> None:
    """Documents the defect so a future edit cannot quietly restore it."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["feature_mode"] not in (21, 30, 50), (
        "config.yaml's feature_mode is back inside the retired hardcoded allowlist; "
        "the regression pin above no longer distinguishes the fix from the defect."
    )


def test_running_the_file_by_path_imports_this_tree(tmp_path: Path) -> None:
    """run_pipeline.sh runs `python src/ci/validate_release.py`, by path.

    Run that way, Python puts src/ci on sys.path, not the repository root. The
    module-level `from src.features import ...` then fails where the package is not
    installed, or, where an installed copy exists (an editable install of another
    checkout, or a wheel), silently resolves to that copy and validates the wrong
    tree. runpy executes the module's top level without calling main().
    """
    script = REPO_ROOT / "src" / "ci" / "validate_release.py"
    probe = "\n".join(
        [
            "import runpy, sys",
            "sys.path[0] = sys.argv[1]",
            "runpy.run_path(sys.argv[2], run_name='validate_release_by_path')",
            "print(sys.modules['src.features'].__file__)",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", probe, str(script.parent), str(script)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    resolved = Path(result.stdout.strip().splitlines()[-1]).resolve()
    assert resolved.is_relative_to(REPO_ROOT.resolve()), (
        f"src.features resolved to {resolved}, outside the tree under test {REPO_ROOT}"
    )
