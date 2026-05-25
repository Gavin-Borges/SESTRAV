# SESTRAV Reproducibility and Finalization Status

Date: 2026-04-25

Canonical pointer: `docs/current_evidence_freeze.md`

Status note (2026-05-14): This file captures the 2026-04-25 freeze cycle. For the latest pre-publish rerun and go/no-go decision, see `docs/final_publish_gate_report_20260514.md`.

## Scope of this pass

- Full comparative finalization: v1 vs v2 dataset evaluation
- 3x full pipeline cycles per dataset version (6 total)
- Dataset selection decision and final evidence freeze
- Platt calibrator refit on selected (v2) dataset
- Final release bundle generation

## Environment Used for Execution

- Conda env: `sestrav` (created from `environment.yml`)
- Python: 3.11.15
- scikit-learn: 1.6.1
- Core commands executed from the repository root only

## Preflight Tests

- Historical scope note: the pass count below reflects the dated 2026-04-25
  environment snapshot for this document, not the latest repository-wide test
  baseline.
- **PASS** (`20 passed, 8 warnings`)
- Command: `conda run -n sestrav python -m pytest tests/ -q`
- sklearn version warnings (model trained on 1.8.0, env has 1.6.1) are non-blocking; models retrained fresh each cycle.

## v1 Dataset Stability (3 Cycles)

Dataset: `immunogenicity_dataset.csv` (928 peptides, 5.58:1 class ratio)

| Metric | Cycle 1 | Cycle 2 | Cycle 3 | Variance |
|--------|---------|---------|---------|----------|
| RF AUC-ROC | 0.8200 | 0.8200 | 0.8200 | 0.0000 |
| RF AUC-PR | 0.9526 | 0.9526 | 0.9526 | 0.0000 |
| AUC-PR above trivial | +0.1045 | +0.1045 | +0.1045 | 0.0000 |
| ISSR@10 | 0.9667 | 0.9667 | 0.9667 | 0.0000 |
| H2 R10 | 0.9775 | 0.9775 | 0.9775 | 0.0000 |
| GS found | 15/15 | 15/15 | 15/15 | 0 |
| GS top 25% | 11/15 | 11/15 | 11/15 | 0 |

**Result: Perfectly reproducible (zero variance, fixed random_state=42).**

## v2 Dataset Stability (3 Cycles)

Dataset: `immunogenicity_dataset.csv` (720 peptides, 2.36:1 class ratio)

| Metric | Cycle 1 | Cycle 2 | Cycle 3 | Variance |
|--------|---------|---------|---------|----------|
| RF AUC-ROC | 0.7536 | 0.7536 | 0.7536 | 0.0000 |
| RF AUC-PR | 0.8497 | 0.8497 | 0.8497 | 0.0000 |
| AUC-PR above trivial | +0.1469 | +0.1469 | +0.1469 | 0.0000 |
| ISSR@10 | 0.8571 | 0.8571 | 0.8571 | 0.0000 |
| H2 R10 | 0.9836 | 0.9836 | 0.9836 | 0.0000 |
| GS found | 15/15 | 15/15 | 15/15 | 0 |
| GS top 25% | 9/15 | 9/15 | 9/15 | 0 |

**Result: Perfectly reproducible (zero variance, fixed random_state=42).**

## Dataset Selection Decision

**Selected: v2 dataset** for final sharing and release.

See `results/multi_run_stability_report.md` for full rationale. Key factors:
- Above-trivial AUC-PR is 41% better for v2 (+0.147 vs +0.105)
- Better class balance (2.36:1 vs 5.58:1) produces more honest metrics
- Gold-standard negative discrimination (9/10 pushed down) is v2-only
- TCR features contribute 40% of SHAP explanation power

## Platt Calibrator Refit (2026-04-25)

- Refit on v2 OOF predictions (704 samples, 69.6% positive)
- Raw Brier: 0.1696, Calibrated Brier: 0.1686
- Saved to `models/platt_calibrator.joblib`
- Verified working in subsequent pipeline run (calibration applied to both proteomes)

## Evidence Freeze Bundle (2026-04-25)

- **PASS**
- Command: `conda run -n sestrav python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v2`
- Generated:
  - `release_artifacts/sestrav-v2-20260425T040154Z.manifest.json`
  - `release_artifacts/sestrav-v2-20260425T040154Z.zip`

## Persistent Warnings / Risks

- **Biological claim boundary:** repeated runs improve reproducibility confidence, not biological validation certainty.
- **sklearn version note:** Environment has sklearn 1.6.1; models trained in same env for consistency.
- **XGBoost SHAP:** TreeExplainer incompatibility with current `shap` library; RF SHAP (canonical model) works correctly.

## Recommended Release-Gate Command Chain

```bash
conda run -n sestrav python -m pytest tests/ -q
conda run -n sestrav python -m src.train_classifier --data immunogenicity_dataset.csv --feature-mode 30 --binding-matrix models/peptide_binding_matrix_v3.csv
conda run -n sestrav snakemake --snakefile pipeline.smk --cores 4 --forceall
conda run -n sestrav python -m src.final_validation_report --results-dir results --model-dir models --data immunogenicity_dataset.csv --binding-matrix models/peptide_binding_matrix_v3.csv --model-path models/rf_30feature_integrated.joblib --dataset-mode expansion_alpha --dataset-version 2.0.0-alpha
conda run -n sestrav python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v2
```
