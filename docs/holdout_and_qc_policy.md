# SESTRAV 2.0 Holdout and Quality Control Policy

**Date:** 2026-05-21  
**Scope:** Dataset Versioning, Quality Control (QC) Gates, and Validation Holdouts

## 1. Dataset Version Governance
All new datasets processed by the pipeline must be formally versioned in `config.yaml` under `dataset_governance`.
- **Traceability:** Prior versions must be preserved in the configuration to allow for exact reproduction of previous benchmark claims.
- **Deltas:** Any incremental retraining must publish delta metrics against the previously frozen version.

## 2. Strict Quality Control (QC) Gates
Before any model (RF, XGB, ANN, or GNN) is allowed to train or evaluate on a new dataset export, the following QC checks are programmatically enforced via `src/data_curation_qc.py`:
- **Validity:** The dataset must contain all required schema columns (`Epitope - Name`, `Assay - Qualitative Measure`).
- **Yield Threshold:** The post-deduplication yield must be at least `min_peptide_yield` (default: 500 peptides).
- **Conflict Threshold:** The ratio of conflicting peptide assays (where perfect 50/50 conflict results in a dropped row) must not exceed `max_conflict_ratio` (default: 15% of unique peptides). If it does, the run is blocked.

## 3. Holdout Separation Rules
- **No Test Set Contamination:** The gold-standard test set (Tier A / Tier B external peptides) is permanently quarantined. 
- **Train/Validation Split:** Training pipelines must utilize a strict stratified 5-fold cross-validation or an 80/20 train/val split. The random seed must be fixed (seed=42) for reproducibility.
- **Cross-Virus Isolation:** When running cross-virus transfer experiments (e.g., EBV $\rightarrow$ HPV), the target virus must not exist in the training manifold in any capacity.
- **Optional Experiments (ANN/GNN):** All experimental runs in these tracks must explicitly report sample counts and dataset versioning, enforcing the exact same holdout rules as the canonical track without exception.

*Failure to adhere to these gates will trigger hard exceptions during the pipeline execution.*
