"""Model and config resolution must not depend on the current working directory.

`ModelRegistry.resolve_model` built its base as `Path("models").resolve()`, which resolves
against os.getcwd(). Every model lookup therefore depended on where the process started:
`uvicorn api.main:app` from any directory but the repo root raised FileNotFoundError during
startup, masked in the API image only by `WORKDIR /app` in Dockerfile.api.

`api/main.py` had already computed the correct project-root-anchored paths and then did not
use two of them: `_MODEL_PATH` was defined and never read, and `_CONFIG_PATH` was defined
and never read while `SestravConfig.load()` was called with its cwd-relative default.

Order matters and is asserted below: the config load runs BEFORE the model load, so fixing
the registry alone would not have made startup cwd-independent. Both are covered.

The pre-existing tests/test_model_registry.py could not catch this. Its resolution test
asserts only `resolved.name` and `"models" in resolved.parts`, both of which hold under
either behaviour, which is why the defect survived.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from src.core.config import SestravConfig
from src.core.model_registry import ModelRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent

# NOTE: MODELS_DIR is deliberately NOT imported at module scope. Importing it here would
# make this whole file fail to COLLECT against the unfixed code, which proves only that a
# constant is missing. Every test below asserts observable behaviour instead, so each one
# fails with a real assertion when the cwd-relative base is restored.


def _registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(SestravConfig.model_construct(output_dir=tmp_path))


def test_models_dir_is_anchored_to_the_repo_not_the_cwd():
    from src.core.model_registry import MODELS_DIR

    assert MODELS_DIR == REPO_ROOT / "models"


def test_resolve_model_is_identical_from_an_unrelated_cwd(tmp_path, monkeypatch):
    """The regression itself: same answer regardless of where the process started."""
    registry = _registry(tmp_path)
    from_root = registry.resolve_model("rf_31feature_integrated.joblib")

    monkeypatch.chdir(tmp_path)
    from_elsewhere = registry.resolve_model("rf_31feature_integrated.joblib")

    assert from_root == from_elsewhere
    assert from_elsewhere == REPO_ROOT / "models" / "rf_31feature_integrated.joblib"


def test_resolve_model_from_a_foreign_cwd_does_not_point_into_that_cwd(tmp_path, monkeypatch):
    """Guards the specific failure shape: a path under the wrong root that then does not
    exist, surfacing as FileNotFoundError at API startup rather than as a path bug."""
    monkeypatch.chdir(tmp_path)
    resolved = _registry(tmp_path).resolve_model("rf_31feature_integrated.joblib")
    assert tmp_path not in resolved.parents


def test_confinement_still_rejects_escapes_from_a_foreign_cwd(tmp_path, monkeypatch):
    """The confinement check must stay fail-closed, and must not itself depend on cwd."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        _registry(tmp_path).resolve_model("../secrets.env")


def test_api_loads_config_from_the_project_root_not_the_cwd():
    """_CONFIG_PATH must actually be wired into the load call, not merely defined."""
    from api import main as api_main

    src = inspect.getsource(api_main.ModelManager.load)
    assert "SestravConfig.load(_CONFIG_PATH)" in src
    assert api_main._CONFIG_PATH == REPO_ROOT / "config.yaml"


def test_config_load_precedes_model_load_so_both_fixes_are_required():
    """If the config load stopped running first this test should fail loudly, because the
    rationale recorded above (and in the commit) would no longer hold."""
    from api import main as api_main

    src = inspect.getsource(api_main.ModelManager.load)
    assert src.index("SestravConfig.load") < src.index("self.registry.load")


def test_api_defines_no_unused_model_path_constant():
    """_MODEL_PATH was defined and never read, duplicating a model name that config.yaml
    is responsible for choosing. It reads as a fallback and was not one."""
    from api import main as api_main

    assert not hasattr(api_main, "_MODEL_PATH")
