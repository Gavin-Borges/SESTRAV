# SESTRAV 2.0 Architectural Alignment

**To:** Abdelrahman, Iris, Charles, Emine  
**From:** SESTRAV Maintainers  
**Subject:** SESTRAV 2.0-rc1 Architecture & Data Flow Overhaul

Hi team,

We have officially cut `release/2.0-rc1`. As part of the technical debt remediation and OpenSSF compliance push, we have fundamentally overhauled how SESTRAV handles configuration, file persistence, and model loading. 

**It is critical that all future contributions strictly adhere to these new abstractions.** Bypassing them will re-introduce hardcoded paths, break our CI/CD security gates, and violate our `freeze_mode` constraints.

## 1. Stop Using `dict` Configs -> Use `SestravConfig`
We have eliminated loose configuration dictionaries and arbitrary `yaml.load()` calls.
*   **Old Way:** `config = yaml.safe_load(open('config.yaml'))` -> `batch_size = config['batch_size']`
*   **New Way:** Import `SestravConfig` from `src.core.config`.
    ```python
    from src.core.config import SestravConfig
    config = SestravConfig.from_yaml("config.yaml")
    batch_size = config.training.batch_size
    ```
*   **Why:** Pydantic strictly validates all parameters. If an invalid hyperparameter or missing data path is supplied, it immediately throws a clear schema error instead of failing silently downstream.

## 2. Stop Manual Matrix I/O -> Use `FeatureStore`
Do not manually use `pd.read_csv()` or `df.to_csv()` scattered across scripts.
*   **Old Way:** `df = pd.read_csv("../../09_Data/dataset.csv")`
*   **New Way:** Import `FeatureStore` from `src.core.feature_store`.
    ```python
    from src.core.feature_store import FeatureStore
    store = FeatureStore(config)
    df = store.load_dataset("immunogenicity_dataset_v3.csv")
    ```
*   **Why:** `FeatureStore` guarantees deterministic loading, applies the correct dtypes, and respects `freeze_mode` constraints (i.e., it will block any writes if the pipeline is frozen).

## 3. Stop Hardcoding `.joblib` Paths -> Use `ModelRegistry`
Hardcoded model paths (e.g., `rf_30feature_integrated.joblib`) are gone.
*   **Old Way:** `model = joblib.load(os.path.join(model_dir, "rf_model.joblib"))`
*   **New Way:** Import `ModelRegistry` from `src.core.model_registry`.
    ```python
    from src.core.model_registry import ModelRegistry
    registry = ModelRegistry(config)
    model = registry.load("rf", version="v2")
    ```
*   **Why:** The `ModelRegistry` ensures safe serialization/deserialization and checks that the required feature schema matches the loaded model artifacts.

## What's Next?
We are moving towards expanding our Graph Neural Networks (GNNs) with complex structural biology integrations (e.g., AlphaFold PDB predictions) under Phase 3. 

Please review the codebase in `release/2.0-rc1` and the new unit tests in `tests/` before submitting your next PR. Let's keep the pipeline clean, fast, and secure.

Best,
Gavin
