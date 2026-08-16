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
- **Holdout Policy (SCOPE CORRECTED 2026-08-14):** the 16 named canonical epitopes in `GOLD_STANDARD_EPITOPES` (`src/iedb_data_loader.py`) are excluded from the training pool by the `gs_mask` exclusion in `src/ann_benchmark.py`. This is a 16-peptide exclusion and nothing else. It is **not** the exclusion of "Gold Standard Tier A and Tier B validation sets" this card previously claimed: a substantial share of the Tier A field is present in the training corpus, so Tier A is not a held-out benchmark for this model. This is the same claim `docs/claims_register.md` D16 retracted on both RF model cards on 2026-08-08; it survived here until the 2026-08-14 accuracy sweep.
- **Optimization:** Evaluated over 14 distinct architectures via grid search; trained using BCEWithLogitsLoss with Adam optimization and early stopping.

## Evaluation and Performance
- **Leakage disclosure (added 2026-08-10, `docs/claims_register.md` D15).** The metrics below were computed under a splitter that stratifies but does **not** group by peptide. Because the feature vector is a pure function of the peptide string, peptides recorded under more than one HLA allele can land on both sides of a fold boundary as feature-identical rows. On the v5 corpus this effect was measured at +37.0% AUC-PR for the mode-31 RF (0.8347 ungrouped vs 0.6092 peptide-grouped); the v5 RF/XGB ledgers were re-baselined under a peptide-grouped splitter on 2026-08-10. **This ANN card's numbers are v3-era and have NOT been re-measured under the grouped splitter**, so they should be read as optimistic, not conservative, and are not comparable to the current peptide-grouped v5 figures reported in `README.md` and `docs/paper.md`.
- **Primary Metrics:** Evaluated via stratified 5-fold cross-validation (ungrouped by peptide - see disclosure above) on the v3 (2.0.0-alpha) dataset. Average metrics on unseen folds:
  - AUC-ROC: ~0.67 (representative: 0.670)
  - AUC-PR: ~0.83 (representative: 0.825)
  - ISSR@10: ~0.88 (representative: 0.880)
  - ISSR@25: ~0.93 (representative: 0.930)
- **Provenance caveat on the four figures above (added 2026-08-15, S2/S3).** They were relabelled from "exact:" to "representative:" because that is what their own source says: they descend from the 3-decimal 30-feature table in `docs/model_evaluation_summary.md`, whose note reads "AUC-PR values shown are representative of the 30-feature track. Exact values depend on the training run seed and dataset split." Calling a self-declared representative value "exact" asserted a precision the source never claimed. **They are also contradicted by the one tracked ANN artifact**, `models/ann_cv_summary.csv`, which reads AUC-ROC 0.6083, AUC-PR 0.7820, ISSR@10 0.8571, ISSR@25 0.8057 - the last a 0.124 gap. That artifact was committed in the same commit as these figures and has never been modified since, so the disagreement is original, not drift. **The figures above are NOT re-cited to it here**, because the two describe possibly different evaluation scopes (the ANN out-of-fold artifact carries 704 rows; the table above is headed 720 peptides) and substituting across an unresolved scope difference would trade an unbound number for a mis-scoped one. Treat all four as v3-era, seed-dependent, and unverified pending that resolution.
- **Explainability:** SHAP DeepExplainer analysis indicates allele-specific binding features drive primary node activations, supported by physicochemical structural constraints.

## Limitations
- **Data Efficiency:** As an ANN, performance heavily relies on dataset size and uniformity. The model may exhibit higher variance than the Random Forest on sparse viral subgroups.
- **Uncertainty Quantification:** Without Monte-Carlo Dropout explicitly enabled at inference (`mc_dropout=true` in `config.yaml`), the raw probability score should not be interpreted as absolute confidence.
