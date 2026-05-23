# SESTRAV Model Card: RandomForest (30-Feature Integrated)

## Model Details
- **Model Type:** Random Forest Classifier (Scikit-Learn)
- **Version:** SESTRAV v2.0
- **Primary Use:** Scoring the relative immunogenicity of peptide candidates presented by MHC Class I molecules for therapeutic vaccine triage.
- **Input Features (30):** 20 TCR-facing physicochemical features (hydrophobicity, aromaticity, vdw_volume, charge at positions p4-p8) + 10 multi-allele MHCflurry presentation context scores.
- **Output:** A continuous probability score [0.0 - 1.0] representing likelihood of triggering a T-cell response.

## Intended Use
- **Primary Domain:** HPV16 and EBV derived epitopes (8-11 amino acids).
- **Out-of-Scope:** This model is **NOT** a clinical decision-making tool. It is a research-grade prioritization aid to reduce wet-lab trial and error. It should not be used in isolation to approve candidates for human trials.

## Training Data
- **Source:** IEDB (curated exports, mapped to dataset version `v2.0` in `config.yaml`).
- **Holdout Policy:** Any peptides within the defined Gold Standard Tier A and Tier B validation sets were strictly removed from the training manifold.
- **Biases Addressed:** MHC processing heavily favors 9-mers, and the dataset is historically skewed toward EBV. Training was conducted using inverse-frequency sample weights (by length and taxonomy) to neutralize these representation skews.

## Evaluation and Performance
- **Primary Metrics:** Evaluated via stratified 5-fold cross-validation. Average metrics on unseen folds:
  - AUC-ROC: ~0.80
  - AUC-PR: ~0.94
  - ISSR@10: ~0.97
- **Subgroup Fairness:** Performance has been audited across EBV and HPV subgroups. Refer to `results/scoring_error_audit.md` for current limitations.

## Limitations
- **Presentation Blindness:** The model assumes the provided multi-allele presentation context captures all biologically relevant MHC presentation bottlenecks. It does not account for intracellular proteasomal cleavage pathways directly.
- **Extrapolation:** Scoring non-viral (e.g., neoantigen, bacterial) peptides is strictly exploratory, and confidence scores may be unreliable.
