---
status: historical-v3
---

# SESTRAV Model Card: Artificial Neural Network (30-Feature)

> **HISTORICAL (v3 corpus era).** Every metric on this card was measured on the 2026-06 v3 dataset
> (`data/immunogenicity_dataset_v3.csv`, n=1,004) or its 2026-05 Tier A predecessor, under a splitter
> that does not group by peptide (`docs/claims_register.md` D15). None of it has been re-measured
> under the current v5 corpus or the peptide-grouped splitter, and it is not comparable to the
> current production figures in `README.md` / `docs/paper.md`. Retained for reproducibility of a
> prior result, not as a current claim. See `models/ann_cv_summary.csv` for the one tracked artifact
> this card's own text partially disagrees with, and the Evaluation section below for why.

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
- **Primary Metrics (RETRACTED AS UNBOUND, 2026-08-17; propagated to this card 2026-08-27).** Evaluated via stratified 5-fold cross-validation (ungrouped by peptide - see disclosure above) on the v3 (2.0.0-alpha) dataset. Average metrics on unseen folds:
  - AUC-ROC: ~~representative 0.670~~ - retracted, unbound
  - AUC-PR: ~~representative 0.825~~ - retracted, unbound
  - ISSR@10: ~~representative 0.880~~ - retracted, unbound
  - ISSR@25: ~~representative 0.930~~ - retracted, unbound
- **Why they are retracted (added 2026-08-27).** All four are the **256-128-64 dropout-0.2** ANN
  column of the cross-validation table in `docs/model_evaluation_summary.md`, and that entire
  column was retracted as UNBOUND on 2026-08-17: every cell traced to
  `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/bootstrap_metric_cis.csv`, a
  path absent from this repository and from any local workspace, so none of them can be reproduced,
  checked, or bound to provenance. They are **withdrawn, not superseded** - no replacement ANN
  figure is quoted here, deliberately, and `models/ann_cv_summary.csv` is not the replacement (see
  the architecture paragraph below). Until today this card carried the figures with hedges but
  without the retraction: the hedges were never the defect, the missing retraction was.
- **Scope of that retraction: the 256-128-64 dropout-0.2 run, and nothing else.** It does NOT reach
  the separate unweighted n=1,004 feature-ablation row `combined_30` (AUC-PR 0.825, AUC-ROC 0.670)
  in `docs/model_evaluation_summary.md`, which is cited in `README.md` and in
  `docs/model_cards/rf_30feature.md`. That is a different measurement on a different field which
  merely collides numerically with two of the cells above, and conflating the two fields is
  precisely the error `docs/claims_register.md` D16 records.
- **Provenance caveat on the four figures above (added 2026-08-15, S2/S3).** They were relabelled from "exact:" to "representative:" because that is what their own source says: they descend from the 3-decimal 30-feature table in `docs/model_evaluation_summary.md`, whose note reads "AUC-PR values shown are representative of the 30-feature track. Exact values depend on the training run seed and dataset split." Calling a self-declared representative value "exact" asserted a precision the source never claimed. **They are also contradicted by the one tracked ANN artifact**, `models/ann_cv_summary.csv`, which reads AUC-ROC 0.6083, AUC-PR 0.7820, ISSR@10 0.8571, ISSR@25 0.8057 - the last a 0.124 gap. That artifact was committed in the same commit as these figures and has never been modified since, so the disagreement is original, not drift. **The scope objection recorded here on 2026-08-15 was WRONG and is withdrawn (2026-08-17).** It held that the two "describe possibly different evaluation scopes (the ANN out-of-fold artifact carries 704 rows; the table above is headed 720 peptides)". Measured directly: **704 is not a different scope, it is 720 minus the 16 `GOLD_STANDARD_EPITOPES` held out by `gs_mask` in `src.ann_benchmark`.** The peptide set in `models/ann_oof_predictions.csv` equals, by exact set equality, the rows of `results/external_validation_input.csv` carrying `is_gold_standard_holdout == False` - 704 vs 704, zero on either side only, zero label or virus mismatches, and zero gold-standard epitopes present in the ANN pool. Same corpus, same era, same labels.

**The figures above are still NOT re-cited to that artifact, but for a different and real reason: the ARCHITECTURE differs.** `models/ann_cv_summary.csv` and `models/ann_oof_predictions.csv` were produced by the **legacy 64-32 ReLU dropout 0.3** network - the only architecture that existed in `src.ann_benchmark` when they were committed - which is what this card's own `Architecture` field describes. The four figures above descend instead from the 256-128-64 dropout 0.2 run reported in `docs/model_evaluation_summary.md`. Substituting across that difference would trade an unbound number for a mis-attributed one, which is the D16 failure class. All four are now marked RETRACTED AS UNBOUND above (2026-08-17): treat them as withdrawn, not as v3-era figures a reader may still cite behind a caveat.
- **Explainability:** SHAP DeepExplainer analysis indicates allele-specific binding features drive primary node activations, supported by physicochemical structural constraints.

## Limitations
- **Data Efficiency:** As an ANN, performance heavily relies on dataset size and uniformity. The model may exhibit higher variance than the Random Forest on sparse viral subgroups.
- **Uncertainty Quantification:** Without Monte-Carlo Dropout explicitly enabled at inference (`mc_dropout=true` in `config.yaml`), the raw probability score should not be interpreted as absolute confidence.
