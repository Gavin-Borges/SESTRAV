# SESTRAV Model Card: RandomForest (31-Feature Integrated)

## Model Details
- **Model Type:** Random Forest Classifier (Scikit-Learn `RandomForestClassifier`, 500 estimators, `max_features='sqrt'`, balanced class weights). **Flagged 2026-08-09, unresolved (H3, `_local/state/conflict_register_2026-08-08.md`):** the currently tracked `src/train_classifier.py:666` hardcodes `n_estimators=200` for every configuration it trains, including the v5 canonical run described under Provenance below; there is no CLI flag to raise it to 500, and history at that line shows 200 as far back as it is tracked. Whether "500" here ever matched a real historical run (e.g. an earlier, unpreserved version of the training script) or was always an unverified copy is not yet determined - do not treat it as confirmed for either the v3 or v5 generation until archaeologically checked. Do not resolve by silently changing it to 200: see D16's caution that the Tier A generation's own hyperparameters (200, unweighted) were independently fingerprinted from stored scores, a standard this line has not yet met in either direction.
- **Version:** SESTRAV v2.1.0 - **Current canonical production model.**
- **Model file:** `models/rf_31feature_integrated.joblib` - **not distributed with this repository.** `models/*.joblib` is gitignored, so a fresh clone contains no model binary (confirm the live rule with `git check-ignore -v models/rf_31feature_integrated.joblib`); a reader obtains it by training locally with the command under [Provenance](#provenance) (see README, "Reproducibility and Data Provenance").
- **Provenance:** `models/model_artifact_checksums.json` - records the SHA-256 digest and byte size of `rf_31feature_integrated.joblib`, so a locally trained binary can be compared against the reference build. See [Provenance](#provenance) below for the full record.
- **Primary Use:** Scoring relative immunogenicity of MHC Class I-presented peptides for T-cell vaccine candidate triage.
- **Input Features (31):**
  - 20 physicochemical features at TCR contact positions p4-p8 (hydrophobicity, aromaticity, Van der Waals volume, formal charge, flexibility, bulkiness, hydrophilicity, structural upward-facing probability proxy - 8 scales × 5 positions; zero-imputed at p7/p8 for 8-mers)
  - 10 per-allele MHCflurry 2.2.1 `presentation_score` for canonical alleles: A*01:01, A*02:01, A*03:01, A*11:01, A*24:02, B*07:02, B*08:01, B*27:05, B*35:01, B*44:02
  - `peptide_length` (critical mediating variable for 8-mer zero-imputation; see Limitations)
- **Output:** Continuous probability [0.0-1.0] representing population-level likelihood of T-cell activation. Does not represent allele-specific or donor-specific immunogenicity.
- **Supersedes:** `rf_30feature_integrated.joblib` (legacy, omits `peptide_length`)
- **Superseded by:** `rf_33feature_integrated.joblib` where antigen processing cache is available

## Intended Use
- **Primary Domain (trained):** EBV (B95-8 strain, 8 proteins) and HPV 16/18 (4 proteins each) - 8-11-mer peptides.
- **Exploratory (not validated):** HBV (genotype D, ayw), HCV (genotype 1a). Model trained on EBV/HPV data; cross-family accuracy is exploratory pending v4 training. Treat HBV/HCV outputs as screening candidates only.
- **Out-of-Scope:** Clinical diagnostic or therapeutic decision-making; allele-specific predictions; neoantigen immunogenicity scoring.

## Training Data
- **Source:** IEDB (curated exports, v3 dataset `data/immunogenicity_dataset_v3.csv`, n=1,004 peptides).
- **Class distribution:** 76.6% positive (immunogenic), 23.4% negative.
- **Composition:** EBV 68.1%, HPV16 30.9%, HPV11 1.0%. Length distribution: 9-mer 64.7%.
- **Label quality:** IEDB labels represent population-average T-cell responses aggregated across heterogeneous assay types, donor HLA backgrounds, stimulation conditions, and peptide concentrations (Vita et al. 2019). Labels do not represent allele-specific or donor-specific immunogenicity.
- **Sample weights:** Inverse-frequency weights applied at training time: `virus_weight=0.5`, `length_weight=0.5` to partially correct EBV majority-class and 9-mer length biases.
- **Holdout policy (SCOPE CORRECTED 2026-08-08):** the 16 named canonical epitopes in `GOLD_STANDARD_EPITOPES` (`src/iedb_data_loader.py:24`) are excluded from the training pool (`src/train_classifier.py:555`). This is a 16-peptide exclusion. It is **not** a quarantine of the Tier A / Tier B external benchmark sets, as this card previously stated - 414 of the 704 Tier A peptides are present in the v5 training corpus. See `docs/claims_register.md` D16.

## Evaluation and Performance
- **Evaluation method:** Stratified 5-fold out-of-fold cross-validation. **The previous wording - "conservative; models never score peptides seen during training" - is WITHDRAWN (2026-08-08).** Folds are stratified but not grouped by peptide, so a peptide recorded under more than one HLA allele can appear on both sides of a fold boundary; 71.0% of held-out rows share their exact peptide with the training fold. Because every mode-31 feature is a pure function of the peptide string, those rows are feature-identical. Reported CV metrics are therefore optimistic, not conservative (`docs/claims_register.md` D15).
- **v3 weighted production results (n=1,004):**

| Metric | RF (mean ± std) | Notes |
|--------|-----------------|-------|
| **AUC-PR** | **0.8276 ± 0.027** | Primary metric (class imbalance) |
| AUC-ROC | 0.6431 ± 0.039 | |
| ISSR@10 | 0.8105 ± 0.079 | Fraction of the top-10% ranked peptides that are true positives (precision within the top decile) |
| ISSR@25 | 0.8367 ± 0.022 | |

- **Unweighted ablation AUC-PR:** 0.864 - used for ablation comparisons in Table 1 of the paper.
- **External benchmark context:** On the certified Tier A field (n=704, a different evaluation set from the n=1,004 v3 results above), the closest external tool is BigMHC (0.822, a near-tie; fully trained on undisclosed data, edges SESTRAV on top-decile precision, ISSR@10 0.917 vs 0.843). SESTRAV does not lead the ISSR@10 metric: binding-only (0.861) and MixMHCpred 2.2 (0.847) also exceed it, placing SESTRAV 4th of 5 on top-decile precision even though it leads on the primary AUC-PR metric. SESTRAV's out-of-fold arm is NOT conservative: folds are ungrouped by peptide, so it is leakage-inflated (D15). The previous "conservative by design" claim is withdrawn. PRIME and PredIG are compared on capabilities only; their metric head-to-head is not reproducible from a certified results file and is not reported.
- **Cross-virus transfer:** EBV→HPV16 AUC-PR 0.742; HPV16→EBV 0.711.
- **SYFPEITHI recall:** 1/6 evaluable epitopes in top 5%; 2/6 in top 25% (3.3× and 1.3× enrichment). See `results/syfpeithi_benchmark.json`.
- **Feature importance note:** All 10 MHCflurry binding features (`bind_A0101`-`bind_B4402`) register RF importance = 0.0 in v3. Root cause: physico features at p5-p8 capture anchor-residue binding variance; v3 negative selection confound suppresses binding variance. This is a scientific finding, not a bug. Hard decoys (v4) will restore binding feature utility.

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
4. **TCR contact approximation.** p4-p8 physicochemical features are a sequence-derived proxy, validated primarily for HLA-A*02:01 canonical 9-mers (Chowell et al. 2015). 8-mer/10-mer non-canonical binding registers carry additional uncertainty.
5. **Cross-family generalization unvalidated.** HBV and HCV outputs are exploratory.
6. **MHCflurry version sensitivity.** Binding features computed with MHCflurry 2.2.1 (pinned in `config.yaml`). Binding feature vectors change across model releases.

## Provenance
- MHCflurry version: 2.2.1 (pinned in `config.yaml`)
- Feature schema: `feature_mode=31`, `FEATURE_COLUMNS_31` in `src/features.py`
- Training script (current canonical, v5 inputs): `src/train_classifier.py --data data/immunogenicity_dataset_v5.csv --model-dir models/local --binding-matrix models/peptide_binding_matrix_v5.csv --feature-mode 31` (unweighted). Both inputs are tracked, so this command runs from a clean clone; `--model-dir` is required and `models/local/` is a gitignored scratch destination, so the run cannot replace the published artifacts at the root of `models/`. v5 is the canonical corpus for this model per `config.yaml` (`dataset_mode: v5_iedb_negatives_merge`, `dataset_version: 5.0.0`, `binding_matrix_path: models/peptide_binding_matrix_v5.csv`); the model was rebuilt on v5 in `e6aafe2` (2026-07-03). Metrics for this command as committed: `models/v5/training_results_mode31.csv` records RF 5-fold OOF AUC-ROC 0.9429 ± 0.0036 and AUC-PR 0.8312 ± 0.0084 (last written by `d3972f7`, 2026-07-05). **Neither is a headline metric.** Both are measured over the full v5 training matrix, whose negative class is dominated by central-tolerance self-peptides and cross-pathogen decoy panels rather than same-pathogen negatives (see `virus_composition_table` in `data/immunogenicity_dataset_v5_provenance.json`: 21,432 all-negative `Orthopoxvirus vaccinia` rows and 8,811 all-negative `Self` rows). The honest (decoy-corrected) same-pathogen figures are the per-virus within-CV table - mean AUC-ROC 0.751 over nine viruses (`results/per_virus_eval_v5_mode31.csv`) - and pooled same-pathogen AUC-ROC 0.712 on real IEDB negatives; the earlier pooled figures (AUC-PR 0.7678, AUC-ROC 0.9368) are retracted as decoy-inflated (`docs/claims_register.md` D12, now marked superseded-in-part by D15). **Neither 0.751 nor 0.712 is leakage-corrected**: both are separately peptide-leakage-exposed and reproduce lower - 0.6587 and 0.5989 - under a peptide-grouped splitter (`docs/claims_register.md` D15).
- Historical v4 baseline (not reproducible from a clean clone): the OOF AUC-PR 0.7635 ± 0.0093 was produced by the earlier v4 command `src/train_classifier.py --data data/immunogenicity_dataset_v4.csv --binding-matrix models/peptide_binding_matrix_v4.csv --feature-mode 31` (unweighted) against the frozen v4 corpus, which is gitignored (`git check-ignore -v data/immunogenicity_dataset_v4.csv`). This figure is a v4 measurement and is **not** the metric the v5 command above produces; it has not been restated against v5. **Correction (2026-08-09):** this line previously called 0.828 "the v3 weighted production figure" and said it used `--sample-weights`. That was wrong: 0.828 is a separate 30-feature, unweighted, 200-tree measurement on the n=704 Tier A field (`docs/claims_register.md` D16), not a weighted 31-feature result. This model's own weighted v3 production figure is **0.8276 ± 0.027** (`--sample-weights`, n=1,004, see the table above) - 0.0002 away from 0.828 by coincidence, which is the root cause of the mislabel this correction retracts.
- Artifact checksum: `models/model_artifact_checksums.json` (SHA-256 + byte size for `rf_31feature_integrated.joblib`; helpers in `src/artifact_integrity.py`)
- Dataset checksum (v5, canonical): the v5 corpus ships with its build-time provenance sidecar and schema - `data/immunogenicity_dataset_v5_provenance.json` and `data/immunogenicity_dataset_v5_schema.json` - alongside the binding matrix sidecar `models/peptide_binding_matrix_v5.provenance.json` and the committed v5 mode-31 metrics (`models/v5/training_results_mode31.csv`, `models/v5/rf_oof_predictions_mode31.csv`). All are tracked, so a reader can verify the training corpus from a clean clone.
- Dataset checksum (v4, historical baseline): none ships. `data/immunogenicity_dataset_v4.csv` and `data/immunogenicity_dataset_v4_provenance.json` are both gitignored (`git check-ignore -v data/immunogenicity_dataset_v4.csv data/immunogenicity_dataset_v4_provenance.json`), so the v4 training corpus and its provenance sidecar are not distributed with this repository. What a reader can verify for v4 from a clone instead: the v4 schema (`data/immunogenicity_dataset_v4_schema.json`) and the binding matrix with its sidecar (`models/peptide_binding_matrix_v4.csv`, `models/peptide_binding_matrix_v4.provenance.json`). The v4 mode-31 metric files are **not** among them. `models/training_results_mode31.csv` and `models/rf_oof_predictions_mode31.csv` were overwritten in place by the v5 retrain at `e6aafe2` (2026-07-03) and rewritten by successive v5 rebuilds through `d3972f7` (2026-07-05), because `src/train_classifier.py` defaulted `--model-dir` to `models` at the time; at those tracked paths they now hold v5 content, not v4. That default is gone: `--model-dir` is required and a run aborts before training rather than replacing artifacts that already exist, unless `--allow-overwrite` is passed. The v4 values survive only in git history, at commit `211bf34` (2026-06-22) - `git show 211bf34:models/training_results_mode31.csv`.
- OpenSSF Passing badge: https://www.bestpractices.dev/en/projects/13191
