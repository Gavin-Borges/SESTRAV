# SESTRAV Project Context
SESTRAV (Structural Epitope Scoring via TCR Recognition and Vaccinology) is a machine learning pipeline designed to predict T-cell epitope immunogenicity, specifically targeting therapeutic vaccine discovery for oncoviruses (such as HPV and EBV). It processes immunogenicity datasets (e.g. IEDB/VDJdb) via a Snakemake pipeline and utilizes multiple models including Random Forest, XGBoost, PyTorch ANN, and PyTorch Geometric GNN models to score candidate peptides and identify optimal vaccine targets.

## Architecture & Entry Points
- **Config & Workflows**: 
  - [config.yaml](config.yaml): Global settings (feature mode, active alleles, model choices, freeze rules).
  - [Snakefile](Snakefile) / [pipeline.smk](pipeline.smk): Full multi-stage pipeline configuration (peptide generation -> binding prediction -> feature extraction -> immunogenicity scoring -> reporting).
- **Core Pipeline Scripts (`scripts/`)**:
  - [stage1.py](scripts/stage1.py): Generates candidate peptides from proteome FASTA files.
  - [stage2.py](scripts/stage2.py): Predicts MHC-peptide binding using `mhcflurry`.
  - [stage3.py](scripts/stage3.py): Extracts immunogenicity feature vectors.
  - [stage4.py](scripts/stage4.py): Scores and ranks epitopes, outputting distribution plots.
- **Source Modules (`src/`)**:
  - [features.py](src/features.py): Vectorized immunogenicity feature extraction.
  - [model.py](src/model.py): Model registry and classification heads.
  - [train_ann.py](src/train_ann.py) / [train_gnn.py](src/train_gnn.py): Training scripts.
  - [verify/](src/verify): Multi-virus extractors, target configuration, and validation tests.
- **Data Directories**:
  - [data/](data): Curated reference datasets, proteomes, and dataset governance check targets.

## Execution Commands
Always run commands using the `sestrav` Conda environment on Windows (`conda run -n sestrav ...`).

### 1. Running Tests (Fast/Smoke Tests)
- Run a specific test file:
  `conda run -n sestrav pytest tests/test_features.py`
- Run only fast smoke tests:
  `conda run -n sestrav pytest -m smoke`
- Run all test suite except heavy/slow ones:
  `conda run -n sestrav pytest -m "not (heavy or slow)"`

### 2. Snakemake Pipeline Commands
- Run a dry-run of the pipeline to validate rules:
  `conda run -n sestrav snakemake -n`
- Run specific stages (e.g., scoring) with custom configuration:
  `conda run -n sestrav snakemake results/HPV16_18_panel8_ranked.csv --cores 1`
- Run the full pipeline:
  `conda run -n sestrav snakemake --cores 4`

### 3. Training & Validation Scripts
- Train ANN model:
  `conda run -n sestrav python -m src.train_ann --data data/immunogenicity_dataset_v3.csv --model-dir models/ann --feature-mode 30`
- Generate final validation report:
  `conda run -n sestrav python -m src.final_validation_report --results-dir results --model-dir models --data data/immunogenicity_dataset_v3.csv`

### 4. Checking Repository Health & State
- Run the repository checker script to verify imports, git cleanliness, and dataset freeze validation status:
  `conda run -n sestrav python scripts/check_repo_status.py`

## Code & Repo Conventions
- **Naming Style**: Strict Python `snake_case` for variables, functions, and files. PascalCase for classes.
- **Exception Handling**: Avoid bare `except:` blocks; always catch specific exceptions (e.g., `ValueError`, `KeyError`, `FileNotFoundError`) and include descriptive logging or error propagation.
- **Vectorization**: Prefer vectorized PyTorch operations or pandas/numpy vectorized calculations over raw Python loops for performance.
- **Git Workflow**: Always prefix branches with `feature/` or `fix/` (e.g. `feature/gnn-integration`). Avoid committing heavy artifacts, datasets, or build caches.
- **Code Edits**: Deliver highly minimal, target-specific diffs rather than full-file replacements.
