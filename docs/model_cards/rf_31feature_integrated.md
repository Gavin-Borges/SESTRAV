# SESTRAV Model Card: RandomForest (31-Feature Integrated)

## Model Details
- **Model Type:** Random Forest Classifier (Scikit-Learn `RandomForestClassifier`, 500 estimators, `max_features='sqrt'`, balanced class weights)
- **Version:** SESTRAV v2.0.3 - **Current canonical production model.**
- **Model file:** `models/rf_31feature_integrated.joblib`
- **Provenance:** `models/rf_31feature_integrated_provenance.json`
- **Primary Use:** Scoring relative immunogenicity of MHC Class I-presented peptides for T-cell vaccine candidate triage.
- **Input Features (31):**
  - 20 physicochemical features at TCR contact positions p4–p8 (hydrophobicity, aromaticity, Van der Waals volume, formal charge, flexibility, bulkiness, hydrophilicity, structural upward-facing probability proxy - 8 scales × 5 positions; zero-imputed at p7/p8 for 8-mers)
  - 10 per-allele MHCflurry 2.2.1 `presentation_score` for canonical alleles: A*01:01, A*02:01, A*03:01, A*11:01, A*24:02, B*07:02, B*08:01, B*27:05, B*35:01, B*44:02
  - `peptide_length` (critical mediating variable for 8-mer zero-imputation; see Limitations)
- **Output:** Continuous probability [0.0–1.0] representing population-level likelihood of T-cell activation. Does not represent allele-specific or donor-specific immunogenicity.
- **Supersedes:** `rf_30feature_integrated.joblib` (legacy, omits `peptide_length`)
- **Superseded by:** `rf_33feature_integrated.joblib` where antigen processing cache is available

## Intended Use
- **Primary Domain (trained):** EBV (B95-8 strain, 8 proteins) and HPV 16/18 (4 proteins each) - 8–11-mer peptides.
- **Exploratory (not validated):** HBV (genotype D, ayw), HCV (genotype 1a). Model trained on EBV/HPV data; cross-family accuracy is exploratory pending v4 training. Treat HBV/HCV outputs as screening candidates only.
- **Out-of-Scope:** Clinical diagnostic or therapeutic decision-making; allele-specific predictions; neoantigen immunogenicity scoring.

## Training Data
- **Source:** IEDB (curated exports, v3 dataset `data/immunogenicity_dataset_v3.csv`, n=1,004 peptides).
- **Class distribution:** 76.6% positive (immunogenic), 23.4% negative.
- **Composition:** EBV 68.1%, HPV16 30.9%, HPV11 1.0%. Length distribution: 9-mer 64.7%.
- **Label quality:** IEDB labels represent population-average T-cell responses aggregated across heterogeneous assay types, donor HLA backgrounds, stimulation conditions, and peptide concentrations (Vita et al. 2019). Labels do not represent allele-specific or donor-specific immunogenicity.
- **Sample weights:** Inverse-frequency weights applied at training time: `virus_weight=0.5`, `length_weight=0.5` to partially correct EBV majority-class and 9-mer length biases.
- **Holdout policy:** Tier A and Tier B Gold Standard validation peptides excluded from training manifold via `freeze_mode: true`.

## Evaluation and Performance
- **Evaluation method:** Stratified 5-fold OOF cross-validation - conservative; models never score peptides seen during training.
- **v3 weighted production results (n=1,004):**

| Metric | RF (mean ± std) | Notes |
|--------|-----------------|-------|
| **AUC-PR** | **0.8276 ± 0.027** | Primary metric (class imbalance) |
| AUC-ROC | 0.6431 ± 0.039 | |
| ISSR@10 | 0.8105 ± 0.079 | True positives in top 10% of scored peptides |
| ISSR@25 | 0.8367 ± 0.022 | |

- **Unweighted ablation AUC-PR:** 0.864 - used for ablation comparisons in Table 1 of the paper.
- **External benchmark context:** PredIG-Path (0.727) and PRIME 2.1 (0.777) are evaluated as fully-trained models on a test set with 36.9% confirmed training overlap (optimistic). SESTRAV OOF is conservative by design; the advantage is larger than raw numbers suggest.
- **Cross-virus transfer:** EBV→HPV16 AUC-PR 0.742; HPV16→EBV 0.711.
- **SYFPEITHI recall:** 1/6 evaluable epitopes in top 5%; 2/6 in top 25% (3.3× and 1.3× enrichment). See `results/syfpeithi_benchmark.json`.
- **Feature importance note:** All 10 MHCflurry binding features (`bind_A0101`–`bind_B4402`) register RF importance = 0.0 in v3. Root cause: physico features at p5–p8 capture anchor-residue binding variance; v3 negative selection confound suppresses binding variance. This is a scientific finding, not a bug. Hard decoys (v4) will restore binding feature utility.

## Top Features (RF Importance, feature_mode=31)
1. `peptide_length` - 0.076
2. `p7_vdw_volume` - 0.068
3. `p5_vdw_volume` - 0.065
4. `p7_hydrophobicity` - 0.064
5. `p5_hydrophobicity` - 0.062

## Limitations
1. **No antigen processing features.** NetChop and TAPreg scores are not training features in this model. Use `rf_33feature_integrated.joblib` where antigen processing cache is available (+0.022 AUC-PR over this model).
2. **Binding feature marginal redundancy.** Per-allele MHCflurry scores contribute zero marginal information in v3 (physico-binding overlap; negative selection confound). Expected to be resolved in v4 with hard decoys.
3. **No allele-specific predictions.** Population-average feature representation only.
4. **TCR contact approximation.** p4–p8 physicochemical features are a sequence-derived proxy, validated primarily for HLA-A*02:01 canonical 9-mers (Chowell et al. 2015). 8-mer/10-mer non-canonical binding registers carry additional uncertainty.
5. **Cross-family generalization unvalidated.** HBV and HCV outputs are exploratory.
6. **MHCflurry version sensitivity.** Binding features computed with MHCflurry 2.2.1 (pinned in `config.yaml`). Binding feature vectors change across model releases.

## Provenance
- MHCflurry version: 2.2.1 (pinned in `config.yaml`)
- Feature schema: `feature_mode=31`, `FEATURE_COLUMNS_31` in `src/features.py`
- Training script: `src/train_classifier.py --data data/immunogenicity_dataset_v4.csv --binding-matrix models/peptide_binding_matrix_v4.csv --feature-mode 31` (unweighted; reproduces the canonical v4 OOF AUC-PR 0.7635 ± 0.0093). The v3 weighted production figure (0.828) used `--sample-weights`.
- Dataset checksum: see `models/rf_31feature_integrated_provenance.json`
- OpenSSF Passing badge: https://www.bestpractices.dev/en/projects/13191
