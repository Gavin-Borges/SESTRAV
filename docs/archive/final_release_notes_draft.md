# SESTRAV v2.0.0 Release Notes

## Release Summary

SESTRAV v2.0.0 finalizes the semester core pipeline and integrates advanced computational biology models and validation tracks for public release using the **v2.0.0-alpha dataset** (expansion_alpha).

- **Canonical release track**: **30-feature integrated model/config** (20 physicochemical + 10 multi-allele MHC binding features)
- **Secondary/Optional track**: Neural Network (FlexibleMLP) and Graph Neural Network (GCN/GAT) benchmark modules
- **Legacy comparator track**: **21-feature sequence-only configuration** (for historical comparison)
- **Training dataset**: **v2.0.0-alpha** (720 peptides, 2.36:1 class ratio)

## What Is Included

- **Four-stage pipeline**: Peptide generation, multi-allele binding prediction, feature extraction, and immunogenicity scoring (RF and XGBoost)
- **FlexibleMLP Extension**: PyTorch ANN classification with 14-configuration hyperparameter architecture search
- **GNN Benchmark Suite**: GCN, GAT, and Bipartite Peptide-Allele graphs for structure-based benchmarking
- **Ablation Studies**: Multi-group feature ablation analyses to quantify contact-residue contribution
- **Final validation bundle generation**:
  - `results/gold_standard_validation.csv`
  - `results/baseline_comparison.csv`
  - `results/h2_tier_a_summary.csv`
  - `results/final_validation_report.md`
- **Security & Dependency Hardening**:
  - Refactored scripts clean of `bandit` security findings (such as shell injections, path handling, and try-catch safety)
  - Upgraded dependencies inside `environments/requirements.lock` resolving 9 CVEs/vulnerabilities
- **Multi-run stability evidence**: `results/multi_run_stability_report.md` demonstrating perfect deterministic reproducibility
- **Platt calibrator refit** on the v2 class distribution to output calibrated probabilities

## Key Results (v2)

- RF AUC-ROC: `0.7536` | AUC-PR: `0.8497` | Above-trivial AUC-PR: `+0.147` (+41% vs v1)
- Gold-standard positive recovery: `15/15` found, `7/15` in top 25%
- Gold-standard negative discrimination: `9/10` pushed down (TCR features add value)
- SHAP feature split: 60% binding / 40% TCR-contact features
- H2 Tier A decision: **NOT SUPPORTED** ($R_{10} = 0.9494$, $R_{25} = 1.0208$, below standard threshold)

## Reproducibility Commands

```bash
conda env create -f environment.yml
conda activate sestrav
pip install snakemake
mhcflurry-downloads fetch models_class1_presentation
python -m src.train_classifier --data immunogenicity_dataset.csv --feature-mode 30 --binding-matrix models/peptide_binding_matrix_v3.csv
python -m src.train_classifier --data immunogenicity_dataset.csv --feature-mode 21
python -m pytest tests/ -v
snakemake --snakefile pipeline.smk --cores 4
python -m src.final_validation_report --results-dir results --model-dir models --data immunogenicity_dataset.csv --binding-matrix models/peptide_binding_matrix_v3.csv --model-path models/rf_30feature_integrated.joblib --dataset-mode expansion_alpha --dataset-version 2.0.0-alpha
python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v2
```

## Known Environment Notes

- Base Python 3.13 is not compatible with this `mhcflurry` stack. Use the project conda env with Python 3.11.
- `setuptools==80.9.0` is required for `pkg_resources` compatibility with the `mhcflurry` release.
- Model serialization warnings may appear if scikit-learn versions differ from the model training environment (models should be trained fresh each cycle).
- XGBoost SHAP TreeExplainer has compatibility issues with the `shap` library version; RF SHAP (the canonical model) works correctly.

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

SESTRAV v2 should be communicated as a reproducible computational prioritization prototype. It should not be described as biologically or clinically validated in this release.

Use:
- `docs/limitations_statement_v1.md`
- `docs/colloquium_evidence_freeze_v2_20260524.md`
- `results/final_validation_report.md`

for standardized non-overclaim language and current supported statements.
