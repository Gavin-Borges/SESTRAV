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
- **Prerequisite:** Antigen processing cache at `data/antigen_processing_cache.csv` (**29,299 rows** as tracked today; pre-computed via `scripts/precompute_antigen_processing.py`). *Corrected 2026-08-11: this read "1,004 rows", which was accurate only at the v3 generation (`0e1ff88`, 2026-06-18) and matched the 1,004-peptide v3 corpus. The cache was expanded the next day and has been 29,299 rows since `dcbb1b1` (2026-06-26). The v3 figures on this card remain v3-scoped.*

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
<!-- status: historical-v3 -->
- **v3 weighted production results (n=1,004):**

| Metric | RF (mean ± std) | XGBoost (mean ± std) | Notes |
|--------|-----------------|----------------------|-------|
| **AUC-PR** | **0.8399 ± 0.011** | 0.8235 ± 0.012 | Primary metric |
| AUC-ROC | 0.6728 ± 0.023 | 0.6393 ± 0.029 | |
| ISSR@10 | **0.9158 ± 0.042** | 0.8842 ± 0.052 | Fraction of the top-10% ranked peptides that are true positives (precision within the top decile) |
| ISSR@25 | 0.9102 ± 0.038 | 0.8816 ± 0.024 | |

> **PROVENANCE SPLIT (added 2026-08-15, S1). The eight cells above do NOT come from one source, and
> the two halves need opposite treatment.** Re-derived cell by cell against `models/training_results.csv`
> as it stood at `6a51995` (the mode-33 v3 run, committed 2026-06-18 18:14 - this table was written
> at 20:41 the same day, so that artifact was already in the tree when the numbers were typed).
>
> **The four AUC cells REPRODUCE and are sound.** RF AUC-PR, XGB AUC-PR, RF AUC-ROC and XGB AUC-ROC
> all match that artifact at the precision printed. Their only defect is a **dead citation**: the
> Provenance section below points at the bare tracked path, but the *current* `models/training_results.csv`
> is a v5 mode-31 run and binds none of these - the file was overwritten in place by successive
> retrains (the `--model-dir` default that allowed this is since removed). **Cite the artifact at
> `6a51995`, not the live path.** One cosmetic inconsistency, noted so it is not mistaken for drift:
> XGB AUC-PR's std is truncated to `0.012` (true value 0.012746) while RF AUC-ROC's is rounded up to
> `0.023` (true 0.022914).
>
> **The four ISSR cells are UNSOURCED and must not be relied on.** That same artifact holds RF
> ISSR@10 0.8211 and ISSR@25 0.8694, and XGB 0.8000 and 0.8286 - none are the printed figures, and
> the gaps are far too large to be rounding. The printed pair also asserts ISSR@10 > ISSR@25 for
> **both** models, while the artifact has the **opposite** ordering for both, as does the sibling
> `rf_31feature_integrated.md` v3 table. A sweep of every blob in every commit reachable from `--all`
> for these four values returns **zero hits: no artifact at any point in this repository's history
> carries them**, so their origin cannot be identified. The artifact's values are quoted here only so
> a reader can check this claim - **they are expressly NOT offered as corrected replacements**, because
> there is no evidence this table was ever attempting to report that run's ISSR, and substituting them
> would invent a provenance rather than establish one. Treat all four as withdrawn pending
> re-derivation by whoever owns this table.
>
> **The same eight cells are mirrored in `docs/model_evaluation_summary.md` and carry the identical
> split** - neither file can be corrected alone.

- **Unweighted ablation AUC-PR:** 0.886 ± 0.019 - highest single-number unweighted result in SESTRAV v3, **but measured on mock antigen-processing features (D18) and therefore not a real-predictor result.** **Provenance unverified (2026-08-15, S1):** no ablation-results artifact existed at `6a51995`, so this figure is likely in the same unsourced class as the ISSR cells above; it was not swept for and is flagged rather than asserted either way.
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
2. **Binding feature marginal redundancy - RETRACTED 2026-08-11 (D17).** This read: "Identical to feature_mode=31: all bind_* features are 0.0 RF importance in v3. Hard decoys (v4) are expected to restore binding utility." The observation of 0.0 importance is real; the explanation was not. Those zeros come from the all-zeros v3 binding-matrix placeholder (`f360b90`, fixed at `37d1d67`), so they are a data defect rather than evidence of redundancy, and hard decoys were never the remedy. This card incorporated the 31-feature card's explanation by reference; that explanation is now withdrawn there too.
3. **Cache dependency.** Model cannot be invoked without a pre-computed antigen processing cache. Use `scripts/precompute_antigen_processing.py` with `--resume` to build the cache incrementally.
4. **No allele-specific predictions.** Population-average features only.
5. **TCR contact approximation.** p4-p8 physicochemical proxy, validated for HLA-A*02:01 9-mers primarily (Chowell et al. 2015).
6. **Not yet default.** `config.yaml` defaults to `feature_mode=31`; invoke explicitly with `--feature-mode 33`.

## Provenance
- MHCflurry version: 2.2.1
- Antigen processing cache: `data/antigen_processing_cache.csv` (**29,299 rows** as tracked today, 0 NaN; mock scores - see limitations). *Corrected 2026-08-11: previously "1,004 rows" - true at the v3 generation (`0e1ff88`), stale since `dcbb1b1`.*
- Feature schema: `feature_mode=33`, `FEATURE_COLUMNS_33` in `src/features.py`
- Training script: `src/train_classifier.py --feature-mode 33 --sample-weights`
- Training artifacts: `models/training_results.csv`, `models/feature_importances.csv`
