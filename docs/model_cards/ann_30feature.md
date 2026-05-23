# SESTRAV Model Card: Artificial Neural Network (30-Feature)

## Model Details
- **Model Type:** Feed-Forward Multilayer Perceptron (PyTorch)
- **Architecture:** 64-32 ReLU hidden layers with Dropout (0.3).
- **Version:** SESTRAV v2.0 (CMB 523 Project 2 aligned)
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
- **Primary Metrics:** Evaluated via stratified 5-fold cross-validation. Average metrics on unseen folds:
  - AUC-ROC: ~0.78
  - AUC-PR: ~0.94
  - ISSR@10: ~0.97
- **Explainability:** SHAP DeepExplainer analysis indicates allele-specific binding features drive primary node activations, supported by physicochemical structural constraints.

## Limitations
- **Data Efficiency:** As an ANN, performance heavily relies on dataset size and uniformity. The model may exhibit higher variance than the Random Forest on sparse viral subgroups.
- **Uncertainty Quantification:** Without Monte-Carlo Dropout explicitly enabled at inference (`mc_dropout=true` in `config.yaml`), the raw probability score should not be interpreted as absolute confidence.
