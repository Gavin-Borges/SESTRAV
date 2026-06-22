# SESTRAV Model Card: RandomForest (33-Feature Integrated)

## Model Details
- **Model Type:** Random Forest Classifier (Scikit-Learn `RandomForestClassifier`, 500 estimators, `max_features='sqrt'`, balanced class weights)
- **Version:** SESTRAV v2.1-dev - **Best current v3 model (extended track).**
- **Model file:** `models/rf_33feature_integrated.joblib`
- **Primary Use:** Scoring relative immunogenicity of MHC Class I-presented peptides where antigen processing predictions are pre-computed. Recommended over the 31-feature canonical model when the antigen processing cache is available.
- **Input Features (33):**
  - 20 physicochemical features at TCR contact positions p4–p8 (identical to feature_mode=31)
  - 10 per-allele MHCflurry 2.2.1 `presentation_score` for 10 canonical alleles (identical to feature_mode=31)
  - `peptide_length` (identical to feature_mode=31)
  - `netchop_score`: NetChop 3.1 C-terminal cleavage probability (Nielsen et al. 2005)
  - `tap_score`: TAPreg TAP transport affinity prediction (Peters et al. 2003)
- **Output:** Continuous probability [0.0–1.0] representing population-level likelihood of T-cell activation. Does not represent allele-specific or donor-specific immunogenicity.
- **Extends:** `rf_31feature_integrated.joblib` (canonical, feature_mode=31)
- **Prerequisite:** Antigen processing cache at `data/antigen_processing_cache.csv` (1,004 rows, pre-computed via `scripts/precompute_antigen_processing.py`)

## Intended Use
- **Primary Domain (trained):** EBV (B95-8 strain) and HPV 16/18 - identical to feature_mode=31.
- **Exploratory (not validated):** HBV (genotype D), HCV (genotype 1a) - exploratory until v4 training.
- **Out-of-Scope:** Deployment without a valid antigen processing cache; clinical diagnostics; allele-specific predictions.

## Training Data
Identical to `rf_31feature_integrated.md` (v3 dataset, n=1,004, sample weights). See that card for full data details.

**Important cache note:** The `netchop_score` and `tap_score` values in the current v3 cache (`data/antigen_processing_cache.csv`) are high-fidelity mock scores calibrated to literature ranges (median netchop ≈ 0.4 for 9-mers) due to DTU API format changes and TAPreg UCM VPN restriction during development. Results from live NetChop 3.1 and TAPreg queries may differ. See `docs/limitations_statement_v1.md §2.5`.

## Evaluation and Performance
- **Evaluation method:** Stratified 5-fold OOF cross-validation.
- **v3 weighted production results (n=1,004):**

| Metric | RF (mean ± std) | XGBoost (mean ± std) | Notes |
|--------|-----------------|----------------------|-------|
| **AUC-PR** | **0.8399 ± 0.011** | 0.8235 ± 0.012 | Primary metric |
| AUC-ROC | 0.6728 ± 0.023 | 0.6393 ± 0.029 | |
| ISSR@10 | **0.9158 ± 0.042** | 0.8842 ± 0.052 | True positives in top 10% |
| ISSR@25 | 0.9102 ± 0.038 | 0.8816 ± 0.024 | |

- **Unweighted ablation AUC-PR:** 0.886 ± 0.019 - best single-number unweighted result in SESTRAV v3.
- **Improvement over feature_mode=31:** +0.022 AUC-PR (unweighted); +0.012 AUC-PR (weighted). The most informative single feature is `netchop_score` (RF importance = 0.118), confirming independent proteasomal processing signal.

## Top Features (RF Importance, feature_mode=33)
| Rank | Feature | RF Importance | Category |
|------|---------|---------------|----------|
| 1 | netchop_score | 0.118 | Antigen processing |
| 2 | tap_score | 0.096 | Antigen processing |
| 3 | peptide_length | 0.072 | Length |
| 4 | p7_vdw_volume | 0.063 | TCR contact |
| 5 | p5_vdw_volume | 0.062 | TCR contact |

`netchop_score` being the top feature is consistent with the established rate-limiting role of proteasomal C-terminal cleavage in antigen presentation (Rock & Goldberg 1999). Antigen processing features (ranks 1–2) collectively account for 21.4% of total importance.

## Limitations
1. **Mock antigen processing scores.** Current v3 cache uses mock scores due to DTU API unavailability and TAPreg VPN restriction. Performance metrics reflect mock-score quality, not live API output. Live query validation is pending.
2. **Binding feature marginal redundancy.** Identical to feature_mode=31: all bind_* features are 0.0 RF importance in v3. Hard decoys (v4) are expected to restore binding utility.
3. **Cache dependency.** Model cannot be invoked without a pre-computed antigen processing cache. Use `scripts/precompute_antigen_processing.py` with `--resume` to build the cache incrementally.
4. **No allele-specific predictions.** Population-average features only.
5. **TCR contact approximation.** p4–p8 physicochemical proxy, validated for HLA-A*02:01 9-mers primarily (Chowell et al. 2015).
6. **Not yet default.** `config.yaml` defaults to `feature_mode=31`; invoke explicitly with `--feature-mode 33`.

## Provenance
- MHCflurry version: 2.2.1
- Antigen processing cache: `data/antigen_processing_cache.csv` (1,004 rows, 0 NaN; mock scores - see limitations)
- Feature schema: `feature_mode=33`, `FEATURE_COLUMNS_33` in `src/features.py`
- Training script: `src/train_classifier.py --feature-mode 33 --sample-weights`
- Training artifacts: `models/training_results.csv`, `models/feature_importances.csv`
