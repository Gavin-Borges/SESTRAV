# SESTRAV Model Card: XGBoost (30-Feature Integrated)

## Model Details
- **Model Type:** Gradient Boosted Trees Classifier (XGBoost)
- **Version:** SESTRAV v2.0
- **Primary Use:** Scoring the relative immunogenicity of peptide candidates presented by MHC Class I molecules for therapeutic vaccine triage.
- **Input Features (30):** 20 TCR-facing physicochemical features (hydrophobicity, aromaticity, vdw_volume, charge at positions p4-p8) + 10 multi-allele MHCflurry presentation context scores.
- **Output:** A continuous probability score [0.0 - 1.0] representing likelihood of triggering a T-cell response.

## Intended Use
- **Primary Domain:** HPV16 and EBV derived epitopes (8-11 amino acids).
- **Out-of-Scope:** This model is **NOT** a clinical decision-making tool. It is a research-grade prioritization aid to reduce wet-lab trial and error. It should not be used in isolation to approve candidates for human trials.

## Training Data
- **Source:** IEDB (curated exports, mapped to dataset version `2.0.0-alpha` in `config.yaml`).
- **Holdout Policy:** Any peptides within the defined Gold Standard Tier A and Tier B validation sets were strictly removed from the training manifold.
- **Biases Addressed:** Training was conducted using inverse-frequency sample weights (by length and taxonomy) to neutralize taxonomic and length representation skews.

## Evaluation and Performance
- **Leakage disclosure (added 2026-08-10, `docs/claims_register.md` D15).** The metrics below were computed under a splitter that stratifies but does **not** group by peptide. Because the feature vector is a pure function of the peptide string, peptides recorded under more than one HLA allele can land on both sides of a fold boundary as feature-identical rows. On the v5 corpus this effect was measured at +37.0% AUC-PR for the mode-31 RF (0.8347 ungrouped vs 0.6092 peptide-grouped); the v5 RF/XGB ledgers were re-baselined under a peptide-grouped splitter on 2026-08-10 (v5 mode-31 XGB: AUC-ROC 0.8093, AUC-PR 0.5597). **This card's numbers are v3-era and have NOT been re-measured under the grouped splitter**, so they should be read as optimistic, not conservative, and are not comparable to the current peptide-grouped v5 figures reported in `README.md` and `docs/paper.md`.
- **Primary Metrics:** Evaluated via stratified 5-fold cross-validation (ungrouped by peptide - see disclosure above) on the v3 (2.0.0-alpha) dataset. Average metrics on unseen folds:
  - AUC-ROC: ~0.67 (exact: 0.6650)
  - AUC-PR: ~0.81 (exact: 0.8050)
  - ISSR@10: ~0.87 (exact: 0.8650)
  - ISSR@25: ~0.92 (exact: 0.9150)
- **Score Resolution:** Unlike the Random Forest, which produces tied scores at 1.0 for a small fraction of peptides, the XGBoost model provides finer continuous score resolution.

## Limitations
- **Presentation Blindness:** The model assumes the provided multi-allele presentation context captures all biologically relevant MHC presentation bottlenecks. It does not account for intracellular proteasomal cleavage pathways directly.
- **Extrapolation:** Scoring non-viral (e.g., neoantigen, bacterial) peptides is strictly exploratory, and confidence scores may be unreliable.
