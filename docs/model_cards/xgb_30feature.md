---
status: historical-v3
---

# SESTRAV Model Card: XGBoost (30-Feature Integrated)

> **HISTORICAL (v3 corpus era).** Every metric on this card was measured on the 2026-06 v3 dataset
> (`data/immunogenicity_dataset_v3.csv`, mapped from `dataset_version: 2.0.0-alpha`) under a splitter
> that does not group by peptide (`docs/claims_register.md` D15). None of it has been re-measured
> under the current v5 corpus or the peptide-grouped splitter, and it is not comparable to the
> current production figures in `README.md` / `docs/paper.md`. **No tracked cross-validation
> artifact for this configuration exists at all** - see the Evaluation section below. Retained for
> reproducibility of a prior result, not as a current claim.

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
- **Holdout Policy (SCOPE CORRECTED 2026-08-14):** the 16 named canonical epitopes in `GOLD_STANDARD_EPITOPES` (`src/iedb_data_loader.py`) are excluded from the training pool by the `gs_mask` exclusion in `train_models` (`src/train_classifier.py`). This is a 16-peptide exclusion and nothing else. It is **not** the "strict removal of Tier A and Tier B Gold Standard validation peptides" this card previously claimed: a substantial share of the Tier A field is present in the training corpus, so Tier A is not a held-out benchmark for this model. This is the same claim `docs/claims_register.md` D16 retracted on both RF model cards on 2026-08-08; it survived here until the 2026-08-14 accuracy sweep.
- **Biases Addressed:** Training was conducted using inverse-frequency sample weights (by length and taxonomy) to neutralize taxonomic and length representation skews.

## Evaluation and Performance
- **Leakage disclosure (added 2026-08-10, `docs/claims_register.md` D15).** The metrics below were computed under a splitter that stratifies but does **not** group by peptide. Because the feature vector is a pure function of the peptide string, peptides recorded under more than one HLA allele can land on both sides of a fold boundary as feature-identical rows. On the v5 corpus this effect was measured at +37.0% AUC-PR for the mode-31 RF (0.8347 ungrouped vs 0.6092 peptide-grouped); the v5 RF/XGB ledgers were re-baselined under a peptide-grouped splitter on 2026-08-10 (v5 mode-31 XGB: AUC-ROC 0.8093, AUC-PR 0.5597). **This card's numbers are v3-era and have NOT been re-measured under the grouped splitter**, so they should be read as optimistic, not conservative, and are not comparable to the current peptide-grouped v5 figures reported in `README.md` and `docs/paper.md`.
- **Primary Metrics:** Evaluated via stratified 5-fold cross-validation (ungrouped by peptide - see disclosure above) on the v3 (2.0.0-alpha) dataset. Average metrics on unseen folds:
  - AUC-ROC: ~0.67 (representative: 0.665)
  - AUC-PR: ~0.81 (representative: 0.805)
  - ISSR@10: ~0.87 (representative: 0.865)
  - ISSR@25: ~0.92 (representative: 0.915)
- **Provenance caveat on the four figures above (added 2026-08-15, S2).** Two corrections in one pass. They were relabelled from "exact:" to "representative:" because that is what their own source says: they descend from the 3-decimal 30-feature table in `docs/model_evaluation_summary.md`, whose note reads "AUC-PR values shown are representative of the 30-feature track. Exact values depend on the training run seed and dataset split." **They were also de-padded from 4 decimals back to the 3 the source actually carries** - they had been written as 0.6650 / 0.8050 / 0.8650 / 0.9150, which is the source's 0.665 / 0.805 / 0.865 / 0.915 with a trailing zero appended, giving a measured-to-4dp appearance the underlying run never supported. **No tracked 30-feature XGB cross-validation artifact exists**: a repository-wide sweep for these four literals returns only this card itself. Treat all four as v3-era, seed-dependent, and unbound.
- **Score Resolution:** Unlike the Random Forest, which produces tied scores at 1.0 for a small fraction of peptides, the XGBoost model provides finer continuous score resolution.

## Limitations
- **Presentation Blindness:** The model assumes the provided multi-allele presentation context captures all biologically relevant MHC presentation bottlenecks. It does not account for intracellular proteasomal cleavage pathways directly.
- **Extrapolation:** Scoring non-viral (e.g., neoantigen, bacterial) peptides is strictly exploratory, and confidence scores may be unreliable.
