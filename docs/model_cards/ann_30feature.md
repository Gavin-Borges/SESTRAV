# SESTRAV Model Card: Artificial Neural Network (30-Feature)

## Model Details
- **Model Type:** Feed-Forward Multilayer Perceptron (PyTorch)
- **Architecture:** 64-32 ReLU hidden layers with Dropout (0.3).
- **Version:** SESTRAV v2.0 (ANN benchmark module)
- **Primary Use:** Scoring the relative immunogenicity of peptide candidates presented by MHC Class I molecules.
- **Input Features (30):** 20 TCR-facing physicochemical features + 10 multi-allele MHCflurry presentation scores.
- **Output:** A continuous probability score [0.0 - 1.0].

## Intended Use
- **Primary Domain:** HPV16 and EBV derived epitopes (8-11 amino acids).
- **Out-of-Scope:** This model is **NOT** a clinical decision-making tool. It acts as an advanced benchmarking track alongside the canonical Random Forest. 

## Training Data
- **Source:** IEDB (curated exports, `v2.0`).
- **Holdout Policy:** Gold Standard Tier A and Tier B validation sets were excluded.
- **Optimization:** Evaluated over 14 distinct architectures via grid search; trained using BCEWithLogitsLoss with Adam optimization and early stopping.

## Evaluation and Performance
- **Leakage disclosure (added 2026-08-10, `docs/claims_register.md` D15).** The metrics below were computed under a splitter that stratifies but does **not** group by peptide. Because the feature vector is a pure function of the peptide string, peptides recorded under more than one HLA allele can land on both sides of a fold boundary as feature-identical rows. On the v5 corpus this effect was measured at +37.0% AUC-PR for the mode-31 RF (0.8347 ungrouped vs 0.6092 peptide-grouped); the v5 RF/XGB ledgers were re-baselined under a peptide-grouped splitter on 2026-08-10. **This ANN card's numbers are v3-era and have NOT been re-measured under the grouped splitter**, so they should be read as optimistic, not conservative, and are not comparable to the current peptide-grouped v5 figures reported in `README.md` and `docs/paper.md`.
- **Primary Metrics:** Evaluated via stratified 5-fold cross-validation (ungrouped by peptide - see disclosure above) on the v3 (2.0.0-alpha) dataset. Average metrics on unseen folds:
  - AUC-ROC: ~0.67 (exact: 0.670)
  - AUC-PR: ~0.83 (exact: 0.825)
  - ISSR@10: ~0.88 (exact: 0.880)
  - ISSR@25: ~0.93 (exact: 0.930)
- **Explainability:** SHAP DeepExplainer analysis indicates allele-specific binding features drive primary node activations, supported by physicochemical structural constraints.

## Limitations
- **Data Efficiency:** As an ANN, performance heavily relies on dataset size and uniformity. The model may exhibit higher variance than the Random Forest on sparse viral subgroups.
- **Uncertainty Quantification:** Without Monte-Carlo Dropout explicitly enabled at inference (`mc_dropout=true` in `config.yaml`), the raw probability score should not be interpreted as absolute confidence.
