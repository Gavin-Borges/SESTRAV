# SESTRAV v1.0.0 Release Notes

## Release Summary

SESTRAV v1.0.0 finalizes the semester core pipeline with reproducible validation and canonical track selection for public release using the **v2 dataset** (IEDB-20260424-EBV_HPV16_UPDATED).

- Canonical release track: **30-feature integrated model/config**
- Secondary track: **21-feature legacy comparator** (progress/future reporting)
- Training dataset: **v2** (720 peptides, 2.36:1 class ratio)

## What Is Included

- Four-stage pipeline (peptide generation, multi-allele binding prediction, feature extraction, immunogenicity scoring)
- Final validation bundle generation:
  - `results/gold_standard_validation.csv`
  - `results/baseline_comparison.csv`
  - `results/h2_tier_a_summary.csv`
  - `results/final_validation_report.md`
- Multi-run stability evidence: `results/multi_run_stability_report.md` (3x v1 + 3x v2 cycles, zero variance)
- Canonical selection scorecard and evidence-freeze documentation:
  - `docs/canonical_selection_scorecard.md`
  - `docs/colloquium_evidence_freeze_v2.md`
- v1/v2 dataset comparison: `results/v1_v2_quality_comparison.md`
- Platt calibrator refit on v2 class distribution

## Key Results (v2)

- RF AUC-ROC: 0.7536 | AUC-PR: 0.8497 | Above-trivial AUC-PR: +0.147 (+41% vs v1)
- Gold-standard positive recovery: 15/15 found, 9/15 in top 25%
- Gold-standard negative discrimination: 9/10 pushed down (TCR features add value)
- SHAP feature split: 60% binding / 40% TCR-contact features
- H2 Tier A decision: NOT SUPPORTED (R10=0.9836, below >=2 threshold)

## Reproducibility Commands

```bash
conda env create -f environment.yml
conda activate sestrav
pip install snakemake
mhcflurry-downloads fetch models_class1_presentation
python -m src.train_classifier --data immunogenicity_dataset.csv --feature-mode 30 --binding-matrix models/peptide_binding_matrix.csv
python -m src.train_classifier --data immunogenicity_dataset.csv --feature-mode 21
python -m pytest tests/ -v
snakemake --snakefile pipeline.smk --cores 4
python -m src.final_validation_report --results-dir results --model-dir models --data immunogenicity_dataset.csv --binding-matrix models/peptide_binding_matrix.csv --model-path models/rf_30feature_integrated.joblib --dataset-mode modeB_updated --dataset-version IEDB-20260424-EBV_HPV16_UPDATED-v2
python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v1
```

## Known Environment Notes

- Base Python 3.13 is not compatible with this `mhcflurry` stack.
- Use project conda env with Python 3.11.
- `setuptools==80.9.0` is required for `pkg_resources` compatibility with current `mhcflurry` release.
- Model serialization warnings may appear if sklearn versions differ from model training environment.
- XGBoost SHAP TreeExplainer has a compatibility issue with current `shap` library version; RF SHAP (the canonical model) works correctly.

## Canonical Decision Statement

The 30-feature integrated track is selected as canonical because it best balances:

- predictive performance evidence,
- biological defensibility,
- reproducibility readiness, and
- alignment with proposal scope.

The v2 dataset is selected over v1 because:

- Above-trivial AUC-PR is 41% better (+0.147 vs +0.105)
- Class balance is more honest (2.36:1 vs 5.58:1)
- 52% more negative training examples (214 vs 141)
- Gold-standard negative discrimination (9/10 pushed down) is a novel capability
- TCR features contribute 40% of model explanation power (SHAP)

The 21-feature track remains documented as a legacy comparator.

## Limitation and Claim Boundary (Required)

SESTRAV v1 should be communicated as a reproducible computational prioritization prototype.
It should not be described as biologically or clinically validated in this release.

Use:

- `docs/limitations_statement_v1.md`
- `docs/colloquium_evidence_freeze_v2.md`
- `results/final_validation_report.md`

for standardized non-overclaim language and current supported statements.
