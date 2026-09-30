import sys
from pathlib import Path
import logging
from typing import Any

# run_pipeline.sh runs this file by path, which puts src/ci on sys.path rather than
# the repository root. Add this file's own tree at the front of sys.path when it is
# not already listed, so the first-party import below resolves to this tree rather
# than failing, or reaching an installed copy through that copy's import finder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (  # noqa: E402
    FEATURE_COLUMNS_10,
    FEATURE_COLUMNS_30,
    FEATURE_COLUMNS_31,
    FEATURE_COLUMNS_33,
    FEATURE_COLUMNS_35,
    FEATURE_COLUMNS_50,
    FEATURE_COLUMNS_51,
    FEATURE_COLUMNS_ALLELE,
    TRAIN_FEATURE_COLUMNS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


FEATURE_COLUMNS_BY_MODE = {
    10: FEATURE_COLUMNS_10,
    21: TRAIN_FEATURE_COLUMNS,
    30: FEATURE_COLUMNS_30,
    31: FEATURE_COLUMNS_31,
    33: FEATURE_COLUMNS_33,
    35: FEATURE_COLUMNS_35,
    50: FEATURE_COLUMNS_50,
    51: FEATURE_COLUMNS_51,
    166: FEATURE_COLUMNS_ALLELE,
}


def check_feature_count(model: Any, feature_mode: int) -> int | None:
    """Raise unless the model's feature count agrees with the configured mode.

    Returns the model's feature count, or None when the artifact does not expose
    one. Feature-count agreement is a CONFIG question and not a fixed list: this
    gate previously compared against a hardcoded (21, 30, 50), which predates
    feature mode 31 and therefore rejected the model `config.yaml` ships. Because
    `run_pipeline.sh` runs this gate under `set -euo pipefail`, that stale tuple
    aborted the whole pipeline before `pipeline.py` ever started.

    A mode's NAME is not always its feature COUNT (mode 51 carries 55 columns),
    so the expected count is derived from the canonical feature list for that
    mode. The None case mirrors `ModelRegistry.validate_signature`, which skips
    the comparison for an artifact that exposes no `n_features_in_`.
    """
    columns = FEATURE_COLUMNS_BY_MODE.get(feature_mode)
    if columns is None:
        raise ValueError(f"Unsupported feature_mode={feature_mode}.")
    expected_features = len(columns)
    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and n_features != expected_features:
        raise ValueError(
            f"Model exposes {n_features} features but config.yaml declares "
            f"feature_mode={feature_mode}, which requires {expected_features}."
        )
    return n_features


def main():
    try:
        from mhcflurry import Class1PresentationPredictor

        # The source tree was put on sys.path at import time, above.
        project_root = PROJECT_ROOT

        from src.core.config import SestravConfig
        from src.core.model_registry import ModelRegistry

        # Test MHCflurry
        predictor = Class1PresentationPredictor.load()

        # Test Config loading
        config = SestravConfig.load(project_root / "config.yaml")
        registry = ModelRegistry(config)

        # Test model loading
        model = registry.load(config.model_path.name)
        n_features = check_feature_count(model, config.feature_mode)

        print("Pre-flight checks PASSED")
        print(f"  Model name: {config.model_path.name}")
        print(f"  Model features: {n_features if n_features is not None else 'not exposed'}")
        print(f"  Antigens: {config.antigens}")
        print(f"  Alleles: {len(config.alleles)} alleles")
        print("  MHCflurry loaded OK")

    # Suppression justified: this is a pre-flight dependency gate whose entire job is
    # to turn ANY import, model-load or validation failure into one clean non-zero
    # exit. Do not open this comment with the noqa token itself - ruff reads a
    # leading one as a BLANKET directive and then reports it as unused.
    except Exception as e:  # noqa: BLE001 - pre-flight gate; must normalize every failure to exit 1
        logging.error(f"FATAL: Dependency verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
