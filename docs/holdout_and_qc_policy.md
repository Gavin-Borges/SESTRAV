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
- **Gold-standard epitope quarantine (SCOPE CORRECTED 2026-08-08):** the 16 named canonical
  epitopes in `GOLD_STANDARD_EPITOPES` (`src/iedb_data_loader.py:24`) are excluded from the
  training pool, enforced at `src/train_classifier.py:555`. **This is a 16-peptide exclusion, not
  a quarantine of the Tier A / Tier B external benchmark sets.** The prior wording ("the
  gold-standard test set (Tier A / Tier B external peptides) is permanently quarantined")
  overstated a 16-peptide list as a 704-peptide (Tier A) and ~4,000-peptide (Tier B) quarantine.
  414 of the 704 Tier A peptides are present in the v5 training corpus. Any claim that benchmark
  evaluation reflects "genuinely unseen data" is unsupported by this rule and must not be made.
- **Train/Validation Split:** Training pipelines must utilize a stratified 5-fold cross-validation
  or an 80/20 train/val split. The random seed must be fixed (seed=42) for reproducibility.
  **AMENDED 2026-08-08 (D15):** stratification alone is NOT sufficient. `MultiStratifiedKFold`
  (`src/ml_utils.py`) accepts a `peptides=` argument but uses it only to bin length for
  stratification, never as a fold group, so rows sharing a peptide across different HLA alleles
  land on opposite sides of the boundary - measured at 71.0% of held-out rows. Because every
  `feature_mode=31` feature is a pure function of the peptide string, those rows are
  feature-identical and the split leaks. **New certified runs must group by peptide** (see Phase 0
  of `docs/proposals/2026_feature_upgrade_roadmap.md`). Until a grouped splitter is the production
  default, any metric produced by this path must be reported with an explicit splitter disclosure
  per `docs/claims_register.md` D15. This clause previously mandated the ungrouped splitter and
  therefore mandated the defect; that is corrected here.
- **Cross-Virus Isolation:** When running cross-virus transfer experiments (e.g., EBV $\rightarrow$ HPV), the target virus must not exist in the training manifold in any capacity.
- **Optional Experiments (ANN/GNN):** All experimental runs in these tracks must explicitly report sample counts and dataset versioning, enforcing the exact same holdout rules as the canonical track without exception.

*Failure to adhere to these gates will trigger hard exceptions during the pipeline execution.*
