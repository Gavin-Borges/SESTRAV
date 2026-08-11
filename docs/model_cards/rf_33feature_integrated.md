# SESTRAV Model Card: RandomForest (33-Feature Integrated)

## Model Details
- **Model Type:** Random Forest Classifier (Scikit-Learn `RandomForestClassifier`, 500 estimators, `max_features='sqrt'`, balanced class weights). **Flagged 2026-08-09, unresolved (H3):** this card's own Provenance section below cites `src/train_classifier.py --feature-mode 33 --sample-weights` as the reproducing command, but that tracked script hardcodes `n_estimators=200` (`src/train_classifier.py`, the `rf_kwargs` dict in `train_models`), not 500. Not yet resolved whether "500" reflects an unpreserved earlier script version or was never verified; see the same flag on `rf_31feature_integrated.md` for detail. Do not treat either the 500 or 200 count as confirmed for this card's own results until checked.
- **Version:** SESTRAV v2.1-dev - extended track. **"Best current v3 model" RETRACTED 2026-08-10 (D18):** its margin was measured on mock antigen-processing features, and under the certified v5 peptide-grouped splitter this configuration exceeds `mode_31` by +0.0027 AUC-PR.
- **Model file:** `models/rf_33feature_integrated.joblib`
- **Primary Use:** Scoring relative immunogenicity of MHC Class I-presented peptides where antigen processing predictions are pre-computed. **The recommendation that previously stood here - "Recommended over the 31-feature canonical model when the antigen processing cache is available" - is RETRACTED (D18).** It rested on a +0.022 AUC-PR margin measured on mock features. **`mode_31` remains the canonical production track**; do not prefer this configuration over it on the strength of the antigen-processing features until they are real.
- **Input Features (33):**
  - 20 physicochemical features at TCR contact positions p4-p8 (identical to feature_mode=31)
  - 10 per-allele MHCflurry 2.2.1 `presentation_score` for 10 canonical alleles (identical to feature_mode=31)
  - `peptide_length` (identical to feature_mode=31)
  - `netchop_score`: **MOCK** proteasomal C-terminal cleavage score. Named for NetChop 3.1 (Nielsen et al. 2005) but **not produced by it** - see the cache note and Limitations 1 below, and `docs/claims_register.md` D18.
  - `tap_score`: **MOCK** TAP transport affinity score. Named for TAPreg (Peters et al. 2003) but **not produced by it** - same disclosure.
- **Output:** Continuous probability [0.0-1.0] representing population-level likelihood of T-cell activation. Does not represent allele-specific or donor-specific immunogenicity.
- **Extends:** `rf_31feature_integrated.joblib` (canonical, feature_mode=31)
- **Prerequisite:** Antigen processing cache at `data/antigen_processing_cache.csv` (1,004 rows, pre-computed via `scripts/precompute_antigen_processing.py`)

## Intended Use
- **Primary Domain (trained):** EBV (B95-8 strain) and HPV 16/18 - identical to feature_mode=31.
- **Exploratory (not validated):** HBV (genotype D), HCV (genotype 1a) - exploratory until v4 training.
- **Out-of-Scope:** Deployment without a valid antigen processing cache; clinical diagnostics; allele-specific predictions.

## Training Data
Identical to `rf_31feature_integrated.md` (v3 dataset, n=1,004, sample weights). See that card for full data details.

**Important cache note:** The `netchop_score` and `tap_score` values in the current v3 cache (`data/antigen_processing_cache.csv`) are **mock** scores, produced locally because of DTU API format changes and the TAPreg UCM VPN restriction during development. **They are NOT NetChop 3.1 / TAPreg output and they are NOT reproducible** (`docs/claims_register.md` **D18**): the generator is a hand-coded rule (a base value plus fixed increments for hydrophobic and basic residues and a proline penalty) with a small jitter from `hash()`, which CPython salts per process, so re-running yields different values and the seed behind the shipped cache was never recorded. An earlier version of this note called them "high-fidelity ... calibrated to literature ranges" (median netchop ~0.4 for 9-mers); that overstates a hand-coded rule and is corrected here. Results from live NetChop 3.1 and TAPreg queries would differ. See `docs/limitations_statement_v1.md` Section 2.5.

## Evaluation and Performance
- **Evaluation method (v3 table below):** Stratified 5-fold OOF cross-validation on the v3 corpus (n=1,004), described further below. **This is a separate corpus/era from the v5 remediation described next**, and Phase 0's peptide-grouped splitter was not retroactively applied to v3.
- **v5 mode-33 leakage disclosure and remediation (2026-08-09/2026-08-10, `docs/claims_register.md` D15):** on the v5 corpus, every `feature_mode=33` feature is a pure function of the peptide string plus its precomputed antigen-processing score, so a splitter that does not group by peptide can place feature-identical rows on both sides of a fold boundary. This was first disclosed 2026-08-09 for mode 31 only (production (ungrouped) 0.8347 vs peptide-grouped 0.6092 AUC-PR, +0.2255/+37.0%, `results/cv_leakage_audit.csv`), with mode-33's own delta explicitly flagged as unmeasured. **It has since been measured and remediated:** `src/train_classifier.py --feature-mode 33` now defaults to a peptide-grouped splitter, with in-fold (training-rows-only) median imputation of the antigen-processing scores (`--no-fold-impute` reproduces the pre-remediation whole-cache-median behavior). The v5 mode-33 ablation cell in `models/v5/training_results_ablation.csv` moved from AUC-ROC 0.9434/AUC-PR 0.8313 (ungrouped, retracted) to **AUC-ROC 0.8167/AUC-PR 0.6085** (peptide-grouped, certified). In-fold imputation alone (isolated from the splitter change) moves mode-33 AUC-PR by roughly -0.001, since only about 12.6% of active v5 rows require imputation.
- **v3 weighted production results (n=1,004):**

| Metric | RF (mean ± std) | XGBoost (mean ± std) | Notes |
|--------|-----------------|----------------------|-------|
| **AUC-PR** | **0.8399 ± 0.011** | 0.8235 ± 0.012 | Primary metric |
| AUC-ROC | 0.6728 ± 0.023 | 0.6393 ± 0.029 | |
| ISSR@10 | **0.9158 ± 0.042** | 0.8842 ± 0.052 | Fraction of the top-10% ranked peptides that are true positives (precision within the top decile) |
| ISSR@25 | 0.9102 ± 0.038 | 0.8816 ± 0.024 | |

- **Unweighted ablation AUC-PR:** 0.886 ± 0.019 - highest single-number unweighted result in SESTRAV v3, **but measured on mock antigen-processing features (D18) and therefore not a real-predictor result.**
- **Improvement over feature_mode=31:** +0.022 AUC-PR (unweighted); +0.012 AUC-PR (weighted).
  > **RETRACTED CLAUSE (2026-08-10, `docs/claims_register.md` D18).** This bullet previously ended
  > "The most informative single feature is `netchop_score` (RF importance = 0.118), **confirming
  > independent proteasomal processing signal**." That inference is withdrawn. `netchop_score` is a
  > locally generated MOCK value (see Limitations 1) whose generator **already assumes** hydrophobic
  > and basic C-terminal cleavage preference, so its importance rank cannot be evidence *for* that
  > mechanism - the inference is circular. The importance figure itself is
  > unchanged and is retained below as a description of the fitted model, not as biology. The
  > +0.022 gain is likewise a mock-feature gain. **Corroboration:** under the certified v5
  > peptide-grouped splitter, mode 33 exceeds mode 31 by **+0.0027 AUC-PR** (0.6085 vs 0.6058,
  > `models/v5/training_results_ablation.csv`) - approximately nothing. Same error class as D13.

## Top Features (RF Importance, feature_mode=33)
| Rank | Feature | RF Importance | Category |
|------|---------|---------------|----------|
| 1 | netchop_score | 0.118 | Antigen processing |
| 2 | tap_score | 0.096 | Antigen processing |
| 3 | peptide_length | 0.072 | Length |
| 4 | p7_vdw_volume | 0.063 | TCR contact |
| 5 | p5_vdw_volume | 0.062 | TCR contact |

> **RETRACTED PARAGRAPH (2026-08-10, `docs/claims_register.md` D18).** This paragraph read:
> "`netchop_score` being the top feature is consistent with the established rate-limiting role of
> proteasomal C-terminal cleavage in antigen presentation (Rock & Goldberg 1999)." **That is a
> biological interpretation of a mock feature and is withdrawn.** Both `netchop_score` and
> `tap_score` are locally generated placeholder values (Limitations 1) built from a hand-coded rule
> that already encodes the very cleavage and transport preferences the sentence claimed to confirm,
> so the inference is circular and no conclusion about proteasomal cleavage or TAP transport
> follows from their importance ranks. What the table above
> does show, and all it shows, is that the fitted RF distributed 21.4% of its total importance
> across two synthetic columns - which is itself worth knowing, because it means the reported
> feature ranking of this configuration is not interpretable as biology. Cite Rock & Goldberg 1999
> for the mechanism if needed, but not this model as evidence for it.

## Limitations
1. **Mock antigen processing scores.** Current v3 cache uses mock scores due to DTU API unavailability and TAPreg VPN restriction. Performance metrics reflect mock-score quality, not live API output. Live query validation is pending.
2. **Binding feature marginal redundancy.** Identical to feature_mode=31: all bind_* features are 0.0 RF importance in v3. Hard decoys (v4) are expected to restore binding utility.
3. **Cache dependency.** Model cannot be invoked without a pre-computed antigen processing cache. Use `scripts/precompute_antigen_processing.py` with `--resume` to build the cache incrementally.
4. **No allele-specific predictions.** Population-average features only.
5. **TCR contact approximation.** p4-p8 physicochemical proxy, validated for HLA-A*02:01 9-mers primarily (Chowell et al. 2015).
6. **Not yet default.** `config.yaml` defaults to `feature_mode=31`; invoke explicitly with `--feature-mode 33`.

## Provenance
- MHCflurry version: 2.2.1
- Antigen processing cache: `data/antigen_processing_cache.csv` (1,004 rows, 0 NaN; mock scores - see limitations)
- Feature schema: `feature_mode=33`, `FEATURE_COLUMNS_33` in `src/features.py`
- Training script: `src/train_classifier.py --feature-mode 33 --sample-weights`
- Training artifacts: `models/training_results.csv`, `models/feature_importances.csv`
