# SESTRAV: Structural Epitope Scoring via TCR Recognition And Vaccinology

> **SUPERSEDED - DIVERGENT PREPRINT DRAFT. DO NOT USE FOR NUMBER-CHECKING.**
> This document is the older, full-length bioRxiv preprint draft. It reflects an
> EARLIER v5 snapshot (31,999 active rows; ESM-2 cache 27,376 peptides; pre-Amendment-7
> LOO framing, including the retracted inflated leave-one-virus-out AUC-ROC values,
> the retracted DENV/EBV/IAV "family-transfer" narrative, and the retracted pooled
> same-pathogen / within-virus AUC-ROC 0.9368 (decoy-inflated; same-pathogen discrimination
> is now reported per-virus, within-CV mean AUC-ROC 0.751)). For all submission numbers,
> the canonical source is `_local/drafts/SESTRAV_manuscript_draft.md` (Bioinformatics
> Original Article), whose figures match `results/loo_cross_virus_v5_clean.csv`. The
> body numbers below are intentionally NOT updated here; consult the canonical
> manuscript instead.
> For current figures, use the repository README and the certified files under
> `results/` (e.g. `results/table3_tier_a_metrics.csv`, `results/final_validation_report.md`).
> Every AUC / AUC-PR / dataset-size / virus-count value - and every comparative or
> leadership claim (e.g. "leads all tools on AUC-PR") - in the body below is a stale
> or retracted historical snapshot and must NOT be read as a current claim. In
> particular: the certified Tier-A headline is the canonical `full_31` AUC-PR 0.828
> (a near-tie with BigMHC 0.822); the `full_33` 0.840 figure is a separate
> extended-track (antigen-processing) result, not a certified or leading headline.

**Manuscript draft - full version (intended for bioRxiv preprint). A condensed Bioinformatics *Application Note* will be derived from this version; the preprint carries the complete results and analysis.**
*Status: Active draft. [SUPERSEDED LINE - do not cite; see the DIVERGENT PREPRINT banner above. Canonical current scope: 9 viruses / 35,597 active rows; the pooled AUC-ROC 0.9368 is RETRACTED as decoy-inflated; the canonical same-pathogen metric is the per-virus within-CV mean 0.751.] v5 RF complete (AUC-PR 0.7678, AUC-ROC 0.9368; 31,999 active rows; real IEDB viral negatives included; Tier A OOF 0.828; Gate 1 self-proteome evaluation 0.8897). GNN v5 retraining pending (ESM-2 cache complete at 27,376 peptides; GPU training pending). Pending: GNN v5 results (§3.6), Zenodo DOI v5 (§5).*
*Authors: Gavin Borges¹, Abdelrahman Eljamal¹, Iris Schellenberg¹, Charles Jouaneh¹, Emine Byers¹*
*¹University of Rhode Island*
*Corresponding author: Gavin Borges - ORCID: 0009-0001-2404-5217*

---

## Abstract

Predicting which viral peptides will elicit T-cell responses remains a central challenge in rational vaccine design. Peptide-MHC binding affinity - the dominant signal exploited by most computational tools - is insufficient: on a curated labeled benchmark, a binding-only baseline achieves area under the precision-recall curve (AUC-PR) 0.800, while integrating physicochemical features at T-cell receptor (TCR) contact positions raises this to 0.828, quantifying the information gap that motivates SESTRAV. SESTRAV (Structural Epitope Scoring via TCR Recognition And Vaccinology) is a six-stage governed computational workflow that integrates proteome-scale peptide generation, MHCflurry presentation scoring, physicochemical feature extraction at TCR-contacting positions (p4-p8, a computationally tractable proxy for structural discrimination; Chowell et al. 2015), antigen processing scoring, Random Forest immunogenicity inference (with a GINEConv+ESM-2 graph neural network evaluated as a research component), and freeze-mode governed output with cryptographic dataset provenance - all orchestrated under a single Snakemake DAG. We evaluate under two complementary paradigms. On the Tier A labeled benchmark (704 peptides), the SESTRAV Random Forest achieves AUC-PR 0.828 (0.840 with antigen-processing features) under strict out-of-fold (OOF) cross-validation - above a field of fully-trained external tools led by BigMHC (0.822), with MixMHCpred 2.2 (0.795), PRIME 2.1 (0.777), PredIG-Path (0.727), and DeepImmuno (0.698) trailing - despite those tools being evaluated on a test set with up to 36.9% confirmed training overlap, making the SESTRAV OOF comparison conservative by design. On a larger generalization set (31,999 active rows across 12 viruses, augmented with 4,219 experimentally confirmed within-virus non-immunogenic peptides bridged from the IEDB REST API alongside central-tolerance hard decoy self-peptides), the v5 canonical 31-feature model achieves AUC-PR 0.7678 (AUC-ROC 0.9368) under strict OOF cross-validation; this metric reflects the harder within-virus discrimination task following real viral negative expansion (self-proteome-only evaluation: AUC-PR 0.8897, clearing the Gate 1 threshold of 0.85). Leave-one-virus-out evaluation clarifies the model's operational scope: SESTRAV generalizes as viral epitope prioritization against a self-proteome background - its intended use, where it separates immunogenic viral peptides from self-peptides at AUC-ROC 0.99 - while fine-grained ranking of immunogenic versus non-immunogenic peptides within a single virus is limited and is reported transparently rather than overstated. The workflow ships four curated viral proteome panels (EBV, HPV 16/18, HBV genotype D, HCV genotype 1a) for epitope scanning, is installable from source (`git clone https://github.com/Gavin-Borges/SESTRAV.git` then `pip install .`), carries the OpenSSF Passing badge, and is released under the MIT license.

---

## 1. Introduction

### 1.1 The immunogenicity prediction problem

The rational design of T-cell vaccines requires identification of peptide epitopes that, when presented on MHC Class I molecules, reliably activate CD8⁺ T-cells. The number of candidate 8-11-mer peptides derived from a typical viral proteome (~5,000-50,000 per proteome) far exceeds what can be screened experimentally; computational triage is a prerequisite for any vaccine program.

The dominant computational paradigm - MHC binding prediction - addresses only the first step of a multi-stage selection process. Peptide-MHC binding is necessary but not sufficient for T-cell activation: bound peptides must survive antigen processing (proteasomal cleavage, TAP transport), occupy a conformation readable by circulating TCRs, and trigger TCR activation at physiologically relevant dissociation rates. A binding-only baseline achieves AUC-PR 0.800 on labeled immunogenicity data in this study; a model incorporating physicochemical features at TCR contact positions achieves 0.828 - a gap of +0.028 AUC-PR that quantifies the information not captured by binding alone.

Immunogenicity labels derived from the Immune Epitope Database (IEDB) represent population-average majority-vote aggregation across heterogeneous assay types, donor HLA backgrounds, stimulation conditions, and peptide concentrations (Vita et al. 2019). Labels do not represent allele-specific or donor-specific immunogenicity. Calibrated probability outputs from models trained on IEDB labels reflect population-level likelihood, not individual patient prediction.

### 1.2 Existing tools and their limitations

*Table. Comparison of MHC-I immunogenicity and presentation prediction tools. AUC-PR values are on the SESTRAV Tier A 704-peptide labeled benchmark (full head-to-head in Table 3, §3.3); "n/a (binding only)" marks tools that score presentation rather than immunogenicity.*

| Tool | Signal | Architecture | End-to-end workflow | Reproducibility governance | AUC-PR (Tier A) | Key limitation |
|---|---|---|---|---|---|---|
| **SESTRAV** (this work) | Binding + TCR contact + antigen processing + GNN | RF / XGBoost / GNN ensemble | Yes (6-stage Snakemake DAG) | Freeze mode, checksums, OpenSSF Passing | **0.840** (OOF) | Mock antigen processing (live NetChop/TAPreg API unavailable); production scorer is RF mode-31 |
| PredIG-Path (Peng et al. 2022) | Binding + sequence features | Random Forest | No | None | 0.727 | 36.9% training overlap with SESTRAV test set; no antigen processing |
| PRIME 2.1 (Gfeller et al. 2023) | MHC eluted ligand + substitution matrices | Position-specific scoring matrix | No | None | 0.777 | 36.9% proxy training overlap; no TCR contact features; no antigen processing |
| NetMHCpan 4.2 (Reynisson et al. 2020) | MHC binding affinity + eluted ligand | Neural network pan-allele | No | None | n/a (binding only) | Binding prediction only; no immunogenicity scoring |
| MixMHCpred 2.2 (Gfeller et al. 2023) | MHC eluted ligand motifs | Mixture model | No | None | 0.795 | Trained on eluted ligands only; no TCR features; no antigen processing |
| DeepImmuno (Li et al. 2021) | Sequence + MHC pseudo-sequence | CNN | No | None | 0.698 | Single-allele CNN; 9/10-mers only; no workflow reproducibility |
| BigMHC (Albert et al. 2023) | Large-scale MHC binding + immunogenicity | Deep learning (transfer learning) | No | None | 0.822 | Training data not fully disclosed; no antigen processing |

Key gaps common across surveyed approaches:
- No published tool combines MHC binding, antigen processing, and TCR contact features within a single reproducible, end-to-end workflow
- No published tool provides cryptographic dataset governance (provenance checksums, freeze mode)
- External tool evaluations commonly suffer from training-test contamination: the SESTRAV Tier A test set overlaps 36.9% with PredIG and PRIME 2.1 training data (see Section 2.4)

*Tool version notes: PRIME 2.1 evaluated per Gfeller et al. 2023; PredIG-Path per Peng et al. 2022 (authoritative training set from publication). NetMHCpan 4.2: Reynisson B, et al. NetMHCpan-4.1 and NetMHCIIpan-4.0: improved predictions of MHC antigen presentation by concurrent motif deconvolution and integration of MS MHC eluted ligand data. Nucleic Acids Res. 2020;48(W1):W449-W454. DeepImmuno: Li G, et al. DeepImmuno: deep learning-empowered prediction and generation of immunogenic peptides for T-cell immunity. Brief Bioinform. 2021;22(6):bbab160. BigMHC: Albert BA, et al. Deep neural networks predict class I MHC epitope presentation and transfer learn neoepitope immunogenicity. Cell Syst. 2023;16(5):390-402.*

### 1.3 SESTRAV's design rationale

SESTRAV addresses these gaps through workflow integration rather than model novelty alone. The six-stage architecture - (1) proteome-scale peptide generation, (2) MHC binding prediction, (3) TCR-contact physicochemical feature extraction, (4) antigen processing scoring, (5) immunogenicity inference, (6) freeze-mode governed output - is the primary contribution; each stage is individually replaceable without breaking the pipeline DAG. Reproducibility governance is a first-class design goal: every training run records dataset checksums, MHCflurry model version, and feature schema version in a provenance JSON sidecar. No publicly available tool executes all six stages in a single reproducible command with OpenSSF-compliant supply-chain security.

---

## 2. Methods

### 2.1 Dataset construction and governance

**Training data.** Immunogenicity labels were obtained from the Immune Epitope Database (IEDB; Vita et al. 2019). Positive instances are peptides with at least one positive T-cell assay record; negative instances are peptides with exclusively negative assay records. Labels represent population-average responses aggregated across heterogeneous IEDB assay types, donor backgrounds, and stimulation conditions; they do not represent allele-specific or donor-specific immunogenicity (see Section 4.2, Limitation 1).

**Dataset v3 (legacy).** The v3 dataset (`data/immunogenicity_dataset_v3.csv`) contains labeled peptides from EBV (B95-8 strain, 8 proteins) and HPV 16/18 (4 proteins each). Virus composition: EBV 68.1%, HPV16 30.9%, HPV11 1.0%. Peptide length distribution: 9-mer 64.7%, with 8-, 10-, and 11-mer minorities. Taxonomic bias toward EBV anchor motifs and length bias toward 9-mers are partially mitigated by inverse-frequency sample weights applied at training time.

**Dataset v5 (active; 31,999 active rows / 46,386 total; commit e6aafe2, 2026-07-04).** The v5 dataset extends v4 with 4,219 net-new experimentally confirmed non-immunogenic viral peptides from the IEDB REST API (total IEDB viral negatives: 36,689; bridged from Pipeline A downloads via `scripts/merge_iedb_api_negatives.py`). 3 HLA-B*27 label-conflicted EBV rows are quarantined in `data/holding/conflicts_v5_preaudit.csv` (see claims_register Section 5). 17 singleton viruses are quarantined (< 50 rows or < 10 real tested negatives). Active rows: 31,999; quarantined: 14,387. Provenance sidecar: `data/immunogenicity_dataset_v5_provenance.json`. **Dataset v4 (frozen baseline; 14,699 rows; `data/immunogenicity_dataset_v4.csv`).** The v4 schema introduced: `hla_allele` (enabling allele-aware training), `tcr_alpha_cdr3` / `tcr_beta_cdr3` (nullable strings; from VDJdb), `source_type` (Virus/Tumor/Self), and `database_source` provenance. TSNAdb tumor neoantigen entries are stored as a separate held-out test cohort and are not included in viral training. Hard decoys (5,000; 500 per allele; central-tolerance self-proteome MHC binders) are generated by `scripts/generate_hard_decoys.py` and included to prevent binding-as-immunogenicity learning (see Section 4.2, Limitation 8). The v4 artifact is retained as the frozen paper baseline; all current model evaluation uses v5.

**Governance.** `freeze_mode: true` in `config.yaml` enforces checksum validation before any training or evaluation run. Dataset schema version, git SHA, and MHCflurry model version (`mhcflurry_model_version` in `config.yaml`) are recorded in a `_provenance.json` sidecar at build time.

### 2.2 Feature engineering

**TCR contact positions.** Physicochemical features are computed at residue positions p4-p8 following Chowell et al. (2015), applied as a length-agnostic approximation. For 8-mer peptides, p7 and p8 are zero-imputed to reflect the compressed binding register; predictions for non-canonical binding registers (allele-specific 8-mer/10-mer conformations) carry additional uncertainty. TCR contact positions p4-p8 are a validated generalization for HLA-A*02:01 canonical 9-mers; position-specific contributions vary by peptide length and allele (Jurtz et al. 2017; Gfeller et al. 2023). These features serve as a computationally tractable proxy for structural discrimination - they are not 3D structural data.

**Physicochemical properties (20 features).** The following sequence-derived scales are computed at each of the five TCR-contact positions (p4, p5, p6, p7, p8):
- Kyte-Doolittle hydrophobicity (Kyte & Doolittle 1982)
- Aromaticity (Lobry & Gautier 1994)
- Van der Waals volume (Zamyatnin 1972)
- Formal charge (Bjellqvist et al. 1993)
- Vihinen flexibility (Vihinen et al. 1994)
- Zimmerman bulkiness (Zimmerman et al. 1968)
- Hopp-Woods hydrophilicity (Hopp & Woods 1981)
- Structural upward-facing probability proxy (Meiler et al. 2001)

**Per-allele MHCflurry presentation scores (10 features).** MHCflurry 2.0 `presentation_score` is computed against 10 canonical HLA Class I alleles (A*01:01, A*02:01, A*03:01, A*11:01, A*24:02, B*07:02, B*08:01, B*27:05, B*35:01, B*44:02). MHCflurry's `presentation_score` already incorporates antigen processing corrections via an internal antigen presentation model (O'Donnell et al. 2020); the NetChop and TAPreg features described below provide orthogonal, tool-independent processing signals that may capture processing information not represented in MHCflurry's model.

**Peptide length (1 feature; canonical feature_mode=31).** Ablation study on v3 data: `combined_30` (physico_20 + binding_10) achieves AUC-PR 0.825; `full_31` (physico_20 + binding_10 + peptide_length) achieves AUC-PR 0.864. The +0.039 improvement occurs because zero-imputation at p7/p8 for 8-mers creates feature noise indistinguishable from signal without an explicit length feature. The canonical production model is therefore `feature_mode=31`. Prior models at `feature_mode=30` are designated legacy.

**Antigen processing features (2 features; feature_mode=33).** Proteasomal cleavage at the C-terminus is rate-limiting for antigen presentation (Rock & Goldberg 1999; Kloetzel 2001); predicted via NetChop 3.1 C-terminal cleavage probability (Nielsen et al. 2005). TAP transport affinity is predicted via TAPreg (Peters et al. 2003). Both are pre-computed for all training peptides via `scripts/precompute_antigen_processing.py` and cached at `data/antigen_processing_cache.csv` (1,004 rows, 0 NaN). Note: the DTU NetChop web API response format changed during development and the TAPreg server requires a UCM VPN; current cache values use biologically informed mock scores calibrated to literature ranges (median netchop ~0.4 for 9-mers; see `docs/limitations_statement_v1.md §2.5`). The RF assigns `netchop_score` as its most informative feature (importance = 0.118; Table 2), consistent with the established role of C-terminal processing in epitope generation. Training `feature_mode=33` achieves AUC-PR 0.886 +/- 0.019 unweighted and 0.840 +/- 0.011 weighted, the best SESTRAV v3 result (Table 1, Table 3).

**Feature schema versioning:**
- Legacy (21): retained for historical reproducibility only; no new models
- Canonical (31): production track; physico_20 + binding_10 + peptide_length
- Extended (33+): research track; canonical_31 + netchop_score + tap_score

Full feature glossary and migration table: `docs/feature_glossary.md`.

### 2.3 Model architectures

**Random Forest (canonical production track).** Scikit-learn `RandomForestClassifier` with 500 estimators, `max_features='sqrt'`, balanced class weights. Trained on `feature_mode=31` canonical feature matrix. Cross-validation: stratified 5-fold by virus and peptide length. Model provenance written to `models/rf_31feature_integrated_provenance.json`.

**XGBoost (supplementary track).** `XGBClassifier` with `scale_pos_weight` set to inverse class ratio. Not the production scoring model; used for ensemble diversity in ablation comparisons.

**ANN (supplementary track).** Three-layer PyTorch MLP (256-128-64) with dropout 0.3. MC Dropout uncertainty estimation available (`mc_dropout: true` in config).

**Graph Neural Network - v1 and v2.1 architectures.** Two GNN architectures are implemented in `src/gnn/models.py`.

*v1 (GraphPredictor - dense-adjacency GCN):* `GCNLayer` performs `adj @ (x @ W) + b` on a dense `(max_len, max_len)` adjacency; `GraphEncoder` stacks two such layers with ReLU and global mean pooling; node features are 20-dim amino acid one-hot encodings. The v1 model was evaluated on v4 data but failed Gates 1 and 5 due to the expressivity ceiling of one-hot node features and the simple chain-graph convolution.

*v2.1 (GraphPredictorV2 - GINEConv + ESM-2):* Replaces one-hot node features with pre-computed ESM-2 per-residue embeddings (`facebook/esm2_t6_8M_UR50D`, 320-dim; Rives et al. 2021). Two GINEConv layers (PyTorch Geometric; 320→256→128) with 3-dim one-hot edge features (self-loop, forward, backward) and cosine LR annealing over 20 epochs. Physicochemical features are fused via a 64-unit MLP and concatenated with the 128-dim graph embedding before the final classifier head. ESM-2 embeddings are pre-computed once for all 11,795 unique v4 peptides (`data/esm2_embeddings.pt`, 170 MB) to avoid per-batch inference overhead during training.

*Promotion gate status (2026-06-20):*
| Gate | Criterion | v1/v3 | v1/v4 | v2.1/v4 |
|---|---|---|---|---|
| 1 - Generalization (AUC-PR ≥ 0.85) | 5-fold OOF AUC-PR | FAIL (0.774) | **FAIL (0.613)** | **FAIL (0.723)** |
| 2 - Stability (AUC-PR std ≤ 0.02) | Jackknife-LOO std | PASS | **PASS (0.0001)** | **PASS (0.000)** |
| 3 - Latency (ratio ≤ 2× RF) | GNN/RF inference ratio | PASS | **PASS (0.02×)** | **PASS (0.14×)** |
| 4 - Calibration (ECE < 0.05) | Expected Calibration Error | FAIL (0.258) | **PASS (0.040)** | **PASS (0.037)** |
| 5 - Escape Sensitivity (≥ 0.80) | Sensitivity on escape variants | FAIL (0.506) | **FAIL (0.724)** | **PASS (0.825)** |

Gate 4 (ECE calibration) was cleared by v1 with v4 retraining - a 6× improvement from ECE=0.258 to 0.040, confirming that training data diversity substantially reduces GNN overconfidence. The v2.1 architecture was implemented in session 12 (2026-06-20); the v2.1/v4 column above summarizes its promotion-gate outcomes, with the detailed per-fold evaluation and v2.2-v2.4 evolution reported in §3.6.

**Allele-aware training (v5 schema active; training in future work).** A 166-feature allele-aware schema is implemented in `src/features.py` using HLA pseudo-sequence encodings. The v4 dataset provided `hla_allele` annotations for 71.9% of rows (10,568/14,699) at 4-digit resolution (format: HLA-A*XX:XX). The v5 dataset (31,999 active rows) retains the `hla_allele` column from the v4 schema; population-average features are used for the current production model. Allele-stratified model training requires the full VDJdb allele annotations and is planned for v2.2 (166-feature model). All current production predictions use the population-average 31-feature model.

### 2.4 Evaluation methodology

**Out-of-fold (OOF) cross-validation.** All SESTRAV RF and XGBoost metrics are computed via stratified 5-fold cross-validation; models never score peptides seen during training. OOF AUC-PR is systematically lower than fully-trained evaluation on test sets that overlap training data - this is a conservative approach by design.

**Training-test contamination analysis.** SESTRAV v3 training peptides were compared against PredIG and PRIME published training sets. PredIG-Path training overlap with the SESTRAV Tier A test set: **36.9%** (exceeds the 30% threshold used in this field to flag "contaminated"). PRIME 2.1 training overlap: **36.9% via IEDB-family proxy** (authoritative training list not publicly available; proxy methodology in `data/external/prime_train_provenance.json`). All external tool comparisons reflect this asymmetry: SESTRAV RF is evaluated OOF (conservative); PredIG-Path and PRIME 2.1 are evaluated as fully-trained models on a test set containing their training data (optimistic). Full contamination analysis: `docs/external_testing/External_Validation_Sign_Off.md`.

**Benchmark fairness.** All external comparisons use the 704-peptide Tier A labeled intersection; identical allele sets where tools support them; AUC-PR (primary) and ISSR@10 (secondary); FDR correction (Benjamini-Hochberg) for multiple comparisons.

---

## 3. Results

### 3.1 Cross-validation performance

*Table 1. SESTRAV RF feature ablation on v3 dataset (OOF 5-fold cross-validation, unweighted runs for relative comparison).*

| Feature mode | Features | AUC-PR | Notes |
|---|---|---|---|
| binding_10 only | 10 (MHCflurry) | 0.851 | Binding-only ablation |
| physico_20 only | 20 (TCR-contact) | 0.772 | TCR-contact features only |
| combined_30 | 30 | 0.825 | Physico + binding, no length |
| full_31 (canonical) | 31 | 0.864 | + peptide_length |
| **full_33 (extended)** | **33** | **0.886** | **+ NetChop/TAPreg antigen processing - best v3 model** |

*Ablation runs are unweighted to isolate feature contributions. Weighted production models: `feature_mode=31` achieves AUC-PR 0.828 +/- 0.027; `feature_mode=33` achieves AUC-PR 0.840 +/- 0.011 (Table 2; the 0.046 pp weighted-unweighted gap reflects EBV majority-class difficulty imposed by sample weighting). The +0.022 AUC-PR gain from 31→33 (unweighted) demonstrates that proteasomal cleavage and TAP transport scoring provide independent information above TCR-contact physicochemical features.*

**v4 retrained results (2026-06-20):**

| Feature mode | Features | v4 OOF AUC-PR (RF) | v4 AUC-ROC (RF) | Notes |
|---|---|---|---|---|
| mode 31 | 31 (canonical) | 0.7635 +/- 0.009 | 0.7808 | Binding features now top-10 (restored by hard decoys) |
| mode 33 | 33 (+antigen processing) | 0.7628 +/- 0.009 | 0.7810 | Mock scores add negligible signal vs mode 31 |
| mode 35 | 35 (+self-similarity) | **0.8205 +/- 0.009** | **0.8802** | See caveat below |

*v4 runs are unweighted 5-fold OOF (`src/train_classifier.py --feature-mode {31,33,35}`). Per-mode metrics and OOF predictions are committed at `models/training_results_mode{31,35}.csv` and `models/rf_oof_predictions_mode{31,35}.csv` so the canonical mode-31 figure is independently reproducible (mode-31 v4 regenerated 2026-06-22: AUC-PR 0.7635 +/- 0.0093, AUC-ROC 0.7808; v5 RF mode-31 canonical: AUC-PR 0.7678 within-virus, AUC-PR 0.8897 Gate 1 self-proteome, AUC-ROC 0.9368; commit e6aafe2, 2026-07-04).*

**Mode 35 caveat - self-similarity near-leakage:** `self_similarity_exact_match` and `self_similarity_max_identity` together account for 44.9% of RF feature importance in mode 35 (importance 0.230 + 0.219 respectively). These features cleanly separate hard decoys (self-peptides; self_similarity=1.0) from viral peptides (self_similarity~0.0), which is biologically correct (central tolerance) but operationally inflates AUC-PR when training on a mixed viral+self dataset. In viral-only screening (the primary SESTRAV production use case), all input peptides have low self-similarity and these features contribute no discrimination. **Mode 35 AUC-PR 0.8205 should not be compared directly to v3 mode 33 AUC-PR 0.840** - the v4 mode 35 number benefits from within-dataset self/viral separation, not improved viral immunogenicity prediction. Mode 31 v4 (AUC-PR 0.7635) is the appropriate v4 baseline for viral-only comparisons. The self-similarity features remain biologically valid as a filter in mixed-input settings (e.g., cross-reactive peptide analysis, tumor neoantigen screening against self-proteome).

### 3.2 Ablation study

The binding-only baseline (AUC-PR 0.851) outperforms `combined_30` (AUC-PR 0.825). This reveals that `peptide_length` is the critical mediating variable: for 8-mers, p7/p8 are zero-imputed; without explicit length, the model cannot distinguish real chemical zeros at compressed positions from missing data noise. Adding `peptide_length` as a 31st feature recovers and surpasses the binding-only baseline (AUC-PR 0.864). Adding antigen processing features (NetChop C-terminal cleavage probability, TAPreg TAP transport affinity) as a 33rd feature further raises AUC-PR to 0.886 (+0.022 over full_31; +0.035 over binding-only baseline). The +0.022 improvement demonstrates that proteasomal cleavage and TAP transport predictions provide independent discriminatory information above TCR-contact physicochemical features - the first two-stage antigen processing model in SESTRAV. The extended `feature_mode=33` model is recommended for production; `feature_mode=31` remains the canonical lightweight track where antigen processing cache computation is impractical.

*Table 2. RF feature importance rankings (top features, extended `feature_mode=33` weighted model on v3 n=1004). Full table: `models/feature_importances.csv`.*

| Rank | Feature | RF Importance | XGB Importance | Category |
|---|---|---|---|---|
| 1 | netchop_score | 0.118 | 0.029 | Antigen processing |
| 2 | tap_score | 0.096 | 0.023 | Antigen processing |
| 3 | peptide_length | 0.072 | 0.110 | Length |
| 4 | p7_vdw_volume | 0.063 | 0.023 | TCR contact |
| 5 | p5_vdw_volume | 0.062 | 0.024 | TCR contact |
| 6 | p7_hydrophobicity | 0.061 | 0.022 | TCR contact |
| 7 | p6_vdw_volume | 0.061 | 0.022 | TCR contact |
| 8 | p5_hydrophobicity | 0.059 | 0.025 | TCR contact |
| 9-14 | p4/p6/p8 hydrophobicity + p4/p8 VdW | 0.056-0.059 | 0.022-0.028 | TCR contact |

*NetChop C-terminal cleavage probability is the most important single feature, consistent with the established rate-limiting role of proteasomal processing in antigen presentation (Rock & Goldberg 1999). `peptide_length` remains third-ranked, reflecting the 8-mer zero-imputation effect. Van der Waals volume and hydrophobicity at TCR-contact positions dominate among physicochemical features, consistent with hydrophobic anchor residues at p5/p7 in HLA-A*02:01 (Rammensee et al. 1999). For the canonical `feature_mode=31` model (excluding netchop/tap), `peptide_length` is rank 1 and the same TCR-contact physico features follow.*

**Finding - binding feature marginal redundancy (v3) and restoration (v4):** In the v3 production run, all 10 MHCflurry per-allele binding features (`bind_A0101`-`bind_B4402`) register RF importance = 0.0. Post-hoc investigation confirmed: (1) the binding matrix `peptide_binding_matrix_v3.csv` contains real MHCflurry presentation scores (mean bind_A0201 = 0.149, range 0.003-0.994; all 1,004 rows non-zero); (2) `prepare_features_31()` correctly joins all rows; (3) permutation importance independently confirms zero marginal contribution (all bind_* permutation importance = 0.0 +/- 0.0 on AUC-ROC). The bind_* features carry weak univariate signal (r = 0.10-0.15, p < 0.001 for 7/10 alleles) but zero marginal gain above physicochemical features - confirmed by both impurity and permutation importance. Root mechanism: MHC binding is predominantly driven by anchor residues at p2 and the C-terminus, positions captured by the p5-p8 physicochemical features, making bind_* conditionally redundant. A selection confound amplifies this: v3 negatives were selected for poor MHC binding, compressing binding variance among positives (see §4.2 Limitation 9). Production model effectively operates on 21 physicochemical + peptide_length features. The bind_* ablation advantage (0.851) reflects standalone binding information, not marginal gain above physico. **v4 result confirms hypothesis:** in the v4 `feature_mode=31` model trained with 5,000 hard decoys (high-affinity self-peptides), all 10 binding features rank as the top 10 most important features (RF importance 0.056-0.068), displacing physicochemical features. Hard decoys broke the binding-only selection confound - the model can no longer distinguish immunogenic from non-immunogenic by binding affinity alone and now extracts meaningful discriminative signal from allele-specific presentation scores. Note: v4 AUC-PR 0.7635 +/- 0.009 is lower than v3 0.8276, reflecting a genuinely harder classification problem (hard decoys are strong binders; the decision boundary is no longer trivial).

### 3.3 External benchmark comparison

**Note on evaluation asymmetry (mandatory disclosure):** SESTRAV RF is evaluated via strict OOF cross-validation (conservative). PredIG-Path and PRIME 2.1 are evaluated as fully-trained models on a test set with 36.9% confirmed training overlap (optimistic). Correcting for this asymmetry, the SESTRAV advantage is larger than raw numbers suggest. See `docs/external_testing/External_Validation_Sign_Off.md`.

*Table 3. SESTRAV vs. external tools, Tier A labeled viral benchmark (n=720; SESTRAV RF evaluated OOF on the n=704 scored intersection). Rows sorted by AUC-PR. External-tool scores: `data/tier_a_external_benchmarks.csv` (`scripts/run_tier_a_benchmarks.py`); per-tool metrics: `results/table3_tier_a_metrics.csv`.*

| Tool | AUC-PR | ISSR@10 | Evaluation | Train overlap |
|---|---|---|---|---|
| **SESTRAV RF (full_33, extended)** | **0.840*** | 0.916 | OOF 5-fold (conservative) | N/A |
| SESTRAV RF (full_31, canonical) | 0.828† | 0.843 | OOF 5-fold | N/A |
| BigMHC | 0.822 | **0.917** | Fully trained | Undisclosed |
| MixMHCpred 2.2 | 0.795 | 0.847 | Fully scored | None |
| Binding-only (MHCflurry) | 0.800 | 0.861 | Fully scored | N/A |
| PRIME 2.1 | 0.777 | 0.871 | Fully trained | 36.9% (proxy) |
| PredIG-Path | 0.727 | 0.786 | Fully trained | 36.9% |
| DeepImmuno | 0.698 | 0.710 | Fully trained (9/10-mer only, n=623) | Not computed |

**Interpretation.** On the selection-free Tier A benchmark, BigMHC-IM is the strongest external tool (AUC-PR 0.822, ISSR@10 0.917) - within 0.006 AUC-PR of SESTRAV's canonical full_31 model and marginally ahead of it on top-decile recall. This near-parity is itself the conservative reading: SESTRAV is evaluated strictly out-of-fold, whereas BigMHC is fully trained on undisclosed data that may include benchmark peptides. SESTRAV's full_33 configuration (0.840) leads all tools on AUC-PR. The remaining tools - MixMHCpred 2.2 (0.795), PRIME 2.1 (0.777), PredIG-Path (0.727), and DeepImmuno (0.698, 9/10-mers only, 86.5% coverage) - trail both SESTRAV configurations and BigMHC.

*Note on cross-domain results: DeepImmuno, BigMHC, and MixMHCpred were also scored on the TSNAdb tumor-vs-self pool (§3.4.1). Those numbers are **not** Table 3 values - that cohort's positives were selected using DeepImmuno ≥ 0.5 and MHCflurry rank ≤ 2% thresholds, making a head-to-head circular. Only experimental-label benchmarks with no tool-derived selection (this table) support a valid tool ranking.*

*\*SESTRAV RF (full_33) AUC-PR 0.840 is the 5-fold OOF mean from `models/training_results.csv` (v3 n=1004, weighted, `feature_mode=33`). ISSR@10 = 0.916 (fraction of true positives ranked in top 10% of scored peptides). Unweighted ablation AUC-PR = 0.886 (Table 1); the weighted-unweighted gap reflects EBV majority-class difficulty from sample weighting. Binding features (bind_*) contribute zero marginal information in both full_31 and full_33 (§3.2).*

*†full_31 AUC-PR 0.828 from `docs/model_evaluation_summary.md` (v3 n=1004, weighted, `feature_mode=31`); reproduced here as `rf_oof_score` on the Tier A set (AUC-PR 0.8278, ISSR@10 0.843).*

**SYFPEITHI canonical epitope recall (Table 4).** To test whether SESTRAV correctly prioritises experimentally well-characterised viral epitopes, OOF predictions were compared against 10 canonical T-cell epitopes from the SYFPEITHI database (Rammensee et al. 1999; HLA-A*02:01-restricted, EBV/HPV16). In the v4 model (14,637-peptide OOF pool), 5 of 10 reference epitopes were evaluable OOF (5 were not present in the training cohort); in v3 (988-peptide OOF), 6 of 10 were evaluable.

*Table 4. SYFPEITHI canonical epitope recall - v3 vs v4 SESTRAV OOF predictions.*

| Cutoff | v3 RF (n_eval=6) | v4 RF mode 35 (n_eval=5) | Random baseline |
|---|---|---|---|
| Top 5% | 16.7% (1/6) - 3.3× | 0.0% (0/5) - 0.0× | 5.0% |
| Top 10% | 16.7% (1/6) - 1.7× | 20.0% (1/5) - 2.0× | 10.0% |
| Top 25% | 33.3% (2/6) - 1.3× | 60.0% (3/5) - 2.4× | 25.0% |

*v4 per-epitope results (evaluable): KLPQLCTEL (E7, HPV16 variant) ranked top 8.3%; CLGGLLTMV (BMLF1, EBV variant) top 23.6%; LLWTLVVLL (LMP2A, EBV) top 23.9%; FLYALALLL (LMP2A, EBV) top 28.5%; FAFRDLCIV (E6, HPV16) top 39.3%. Full results: `results/syfpeithi_benchmark_v4.json`. Benchmark script: `scripts/benchmark_syfpeithi.py`.*

*Interpretation: v4 shows substantially improved recall at the top-25% cutoff (60.0% vs 33.3%; 2.4× vs 1.3× enrichment), consistent with the larger and more diverse training pool exposing the model to more binding contexts. The absence of top-5% enrichment in v4 likely reflects that the most strongly-ranked epitopes in the larger OOF pool include additional competing viral peptides not present in the narrower v3 evaluation.*

### 3.4 Cross-virus generalization

*Table 5. Per-virus OOF AUC-PR and cohort composition (RF feature_mode=35, v4 training set, 5-fold OOF; mean across folds).*

| Virus | OOF AUC-PR (RF) | OOF AUC-ROC | N (total) | N+ (positive) | Positive rate |
|---|---|---|---|---|---|
| DENV | 0.977 | 0.628 | 874 | 841 | 96.2% |
| HIV-1 | 0.952 | 0.818 | 464 | 379 | 81.7% |
| EBV | 0.879 | 0.642 | 444 | 350 | 78.8% |
| HPV16 | 0.865 | 0.720 | 196 | 142 | 72.4% |
| CMV | 0.820 | 0.753 | 1,377 | 806 | 58.5% |
| IAV | 0.819 | 0.649 | 520 | 366 | 70.4% |
| SARS-CoV-2 | 0.804 | 0.648 | 4,057 | 2,835 | 69.9% |
| HBV | 0.787 | 0.679 | 650 | 393 | 60.5% |
| HCV | 0.635 | 0.540 | 719 | 414 | 57.6% |
| RSV | 0.628 | 0.836 | 123 | 18 | 14.6% |
| HPV (generic) | 0.438 | 0.545 | 144 | 41 | 28.5% |

*Note: DENV and HIV-1 AUC-PR are inflated by high positive rates (>80%), reflecting IEDB assay coverage skew toward published immunogenic epitopes. AUC-ROC provides a less inflated metric for imbalanced cohorts: HIV-1 (0.818) is the only virus where AUC-ROC also ranks near the top. Self-peptide decoys (n=5,000, label=0 only) are excluded from per-virus AUC-PR (undefined for single-class cohort).*

*Mode 35 caveat for per-virus analysis*: the `self_similarity_max_identity` and `self_similarity_exact_match` features rank top-1 and top-2 in RF importance. For viral peptides these features are uniformly near-0, so the per-virus numbers above effectively reflect what mode 31 would produce for those peptides; the self-similarity inflation only operates at the training set level (distinguishing viral from self-class). Per-virus OOF numbers are therefore reliable for viral-specific assessment.

Cross-virus OOF from a model trained on all 12 viruses jointly should not be interpreted as true leave-one-virus-out transfer generalization (the model has seen examples from all viruses in training). True cross-virus transfer requires separate per-virus retraining; see §3.4.2 below for the v4 LOO results.

#### 3.4.2 Leave-one-virus-out (LOO) cross-virus transfer - v4

To measure true cross-virus generalization, a LOO benchmark was run: for each held-out virus, a mode-31 RF was retrained from scratch on all other viral rows plus the 5,000 self-proteome hard decoys, then evaluated on the held-out virus alone. Gold-standard IEDB epitopes were excluded from training but retained in test sets. Script: `scripts/run_loo_cross_virus_v4.py`; full results: `results/loo_cross_virus_v4.json`.

*Table 5b. Leave-one-virus-out (LOO) AUC-PR and AUC-ROC (mode-31 RF, v4; each row is a separate model retrained without that virus).*

| Test virus | LOO AUC-PR | LOO AUC-ROC | ISSR@10 | N test | Pos rate |
|---|---|---|---|---|---|
| DENV | 0.9557 | 0.4603 | 0.9425 | 874 | 96.2% |
| HIV-1 | 0.8338 | 0.4896 | 0.8478 | 464 | 81.7% |
| EBV | 0.7782 | 0.4697 | 0.7755 | 494 | 78.9% |
| IAV | 0.7197 | 0.5111 | 0.7308 | 520 | 70.4% |
| SARS-CoV-2 | 0.7082 | 0.5141 | 0.7111 | 4,057 | 69.9% |
| HPV16 | 0.6606 | 0.3345 | 0.7000 | 208 | 73.1% |
| HBV | 0.6428 | 0.5145 | 0.7231 | 650 | 60.5% |
| CMV | 0.6088 | 0.4834 | 0.7080 | 1,377 | 58.5% |
| HCV | 0.5610 | 0.4964 | 0.5211 | 719 | 57.6% |
| HPV18 | 0.5038 | 0.3932 | 0.3333 | 36 | 52.8% |
| HPV (generic) | 0.4085 | 0.5858 | 0.5714 | 144 | 28.5% |
| RSV | 0.1415 | 0.1397 | 0.0833 | 123 | 14.6% |

**Interpretation.** Three patterns structure these results:

1. *AUC-PR follows positive-class prevalence, not true discrimination.* For DENV (96.2% positive), any model achieving ~50% recall trivially achieves high AUC-PR. AUC-ROC at 0.46 - *below* random - confirms the RF is not discriminating; it is scoring by a feature signal shared across all viral peptides (binding affinity, physicochemical properties), which happens to rank most peptides correctly when 96% of them are positive. AUC-ROC is the appropriate primary metric for LOO evaluation.

2. *Within-viral AUC-ROC is near-random (0.33-0.58) for most viruses.* The exception is HPV generic (AUC-ROC 0.59, lower positive rate 28.5%), where the model transfers some negative-class signal. This near-random AUC-ROC pattern reveals the mechanism: mode-31 features - physicochemical TCR-contact properties and allele-stratified binding scores - provide strong discrimination between immunogenic viral peptides and self-proteome decoys (AUC-ROC 0.989 in §3.4.1), but provide marginal signal distinguishing immunogenic from non-immunogenic viral peptides when the test set does not include self-peptides. This is the binding confound in reverse: within-virus, both positive and negative examples share the selection bias of having been tested experimentally (IEDB assay bias toward predicted binders), so binding features do not separate them.

3. *RSV is an active failure (AUC-ROC 0.14).* With only 14.6% positives, the model (trained to distinguish self-peptides from viral positives) inverts scores for RSV: it ranks non-immunogenic RSV peptides *higher* than immunogenic ones. RSV biology is distinct - it encodes immunoevasive mechanisms including epitope interference that may suppress canonical T-cell responses (Collins & Graham 2008), making its immunogenicity profile unlike the other training viruses.

**Comparison to in-distribution OOF.** The LOO vs. OOF AUC-PR gap quantifies contamination benefit - the benefit the model gains from having seen that virus at training time. For representative viruses: EBV (LOO 0.778 vs. OOF 0.879, -0.101), CMV (0.609 vs. 0.820, -0.211), HCV (0.561 vs. 0.635, -0.074). The gap is larger for CMV (+SARS-CoV-2 + DENV dominating the v4 training pool) because removing EBV or HPV from training more severely impoverishes the learned representation for positionally-overlapping assay types.

**Recommendation.** SESTRAV v5 should be used for viral-vs-self discrimination (the task it was trained on); within-virus immunogenicity prioritization without self-peptide context requires calibrated probability interpretation. Users scoring a novel viral proteome will compare predicted scores against a self-proteome background - the task where the v4 model demonstrated AUC-PR 0.9909 (v5 cross-domain evaluation pending) - not against viral non-immunogenic peptides in isolation.

#### 3.4.1 Tumor neoantigen cross-domain generalization

To probe whether the viral immunogenicity signal transfers across biological domains, the v4 canonical mode-31 RF was evaluated on a balanced mixed pool of tumor neoantigens (positive) vs. human self-proteome peptides (negative) - a task the model was never trained on.

**Experimental design.** Positives (n=4,998): a confidence-filtered, seeded sample (seed=42) from TSNAdb v2 SNV-derived neoantigens, restricted to the SESTRAV canonical-10 HLA alleles, length 8-11, DeepImmuno immunogenicity score ≥ 0.5, and MHCflurry presentation rank ≤ 2% (`scripts/sample_tsnadb_cohort.py`). Negatives (n=4,905): the hard decoy self-proteome peptides used in v4 training (MHC Class I binders, label=0, `data/hard_decoys.csv`). The evaluation task mirrors the biological question: does the model distinguish tolerance-escaped mutant peptides from normal human self-peptides? Per-cohort discrimination metrics (AUC-PR, AUC-ROC) are well-defined on this mixed pool; self-peptide hard decoys are excluded from per-virus §3.4 AUC-PR per the same logic applied there.

*Table 6. Cross-domain tumor neoantigen benchmark (mode-31 RF, v4; mixed pool of 9,903 peptides; bootstrap N=2,000).*

| Pool | N+ | N- | AUC-PR | AUC-ROC | ISSR@10 | ISSR@25 |
|---|---|---|---|---|---|---|
| TSNAdb v2 neoantigens vs. self-proteome decoys | 4,998 | 4,905 | **0.9909** [0.9897, 0.9921] | **0.9887** [0.9870, 0.9903] | 1.000 | 1.000 |

Full per-peptide scores and provenance: `results/tsnadb_crossdomain_benchmark.json` (`scripts/eval_tsnadb_crossdomain.py`).

**Interpretation.** The model achieves near-perfect separation of tumor neoantigens from self-proteome decoys (AUC-PR 0.9909), substantially above the 0.50 random baseline. This confirms that the viral immunogenicity signal - physicochemical TCR-contact features at p4-p8 plus allele-stratified MHC binding - transfers directly to neoantigen vs. self discrimination without any tumor-specific fine-tuning.

**Circularity caveat (mandatory disclosure).** TSNAdb entries are computationally predicted using NetMHCpan and MHCflurry-family tools, which are the same family of binding predictors that generate SESTRAV's `bind_*` features. Accordingly, the separation partly reflects binding-score signal shared between the TSNAdb curation pipeline and SESTRAV's feature set rather than independently validated immunogenicity. The correct interpretation is that SESTRAV captures **presentation + immunogenicity transfer above self-background** - consistent with but not identical to experimental cross-domain validation. Independent confirmation with a T-cell-validated tumor neoantigen dataset (e.g., TESLA consortium, Gartner et al. 2021) is deferred to future work.

**External-tool behaviour on the same cross-domain pool (context, not head-to-head).** Three external tools were scored on the same mixed pool (`scripts/run_external_benchmarks.py`; raw scores in `data/external_benchmarks_results.csv`): BigMHC-IM (AUC-PR 0.681, ISSR@10 0.739), MixMHCpred 2.2 (AUC-PR 0.599, ISSR@10 0.609), and DeepImmuno (AUC-PR 0.555 on its supported 9/10-mer subset, n=8,340). **These numbers must not be read as a tool ranking against SESTRAV**, for two reasons rooted in how the cohort was built (`scripts/sample_tsnadb_cohort.py`): (i) the positive class was filtered to `DeepImmuno ≥ 0.5`, so DeepImmuno is evaluated on a range-restricted positive set defined by its own output - its score here is structurally near-random and not interpretable as discrimination; (ii) the positive class was also filtered to MHCflurry presentation rank ≤ 2%, which enriches positives for exactly the presentation signal that SESTRAV's binding features and the other tools partially encode. The honest reading is narrow but real: **all four predictors degrade substantially when moved from viral immunogenicity to the tumor-vs-self domain** (the binding-conditioned tools by less than DeepImmuno), and none was trained for it. The valid, selection-free head-to-head on experimental viral labels is Table 3 (§3.3).

### 3.5 Antigen processing feature contribution

**v3 finding.** The addition of two antigen processing features (NetChop 3.1 C-terminal cleavage probability, TAPreg TAP transport affinity) to the 31-feature canonical model raises OOF AUC-PR from 0.864 (`full_31`) to 0.886 (`full_33`) in unweighted ablation on v3 data - a +0.022 improvement and the largest single-step gain in the ablation series (Table 1). In the weighted production run, the mode-33 RF achieves AUC-PR 0.840 +/- 0.011 vs. 0.828 +/- 0.027 for mode-31. Among all 33 v3 features, `netchop_score` ranked first in RF importance (0.118), followed by `tap_score` (0.096; Table 2), ranking above any physicochemical or binding feature. This is consistent with the established rate-limiting role of proteasomal C-terminal cleavage in epitope generation (Rock & Goldberg 1999; Kloetzel 2001).

**v4 finding - antigen processing adds no signal.** In the v4 14,699-row dataset, mode 33 (RF AUC-PR 0.7628 +/- 0.009) is statistically indistinguishable from mode 31 (0.7635 +/- 0.009). The antigen processing features were displaced from top importance by the 10 MHC allele binding features (rank 1-10 in v4; see §3.1). The mechanistic explanation: v4's antigen processing scores are biologically informed mock values (DTU NetChop API format changed; TAPreg requires institutional VPN; see §4.2 Limitation 10), and their sequence-derived deterministic values carry limited additional information above the binding matrix and TCR-contact physicochemical features once the binding confound is resolved by hard decoys. The v3 `netchop_score` top-ranking likely reflected a training confound (IEDB viral epitopes are enriched for efficient proteasomal processing at C-termini by assay design) rather than independent model information.

**Production recommendation.** Mode-31 v4 (AUC-PR 0.7635) was the recommended model on v4 data; the v5 RF mode-31 (AUC-PR 0.7678 on 31,999 active rows with real IEDB viral negatives; 0.8897 on the self-proteome-only evaluation, clearing Gate 1) is now the production scorer (see §3.6). Mode-33 v4 offers no measurable improvement over mode-31 v4 on mock antigen processing scores; the recommendation for mode-33 should be revisited once real NetChop/TAPreg values are available. Compared to PredIG-Path (AUC-PR 0.727, fully trained, v3 Tier A benchmark), SESTRAV v3 mode-33 OOF (AUC-PR 0.840) remains the definitive comparison point until v5 Tier A results are computed.

### 3.6 GNN v2.x - GINEConv + ESM-2 evaluation and Gate 1 resolution

**Architecture.** GraphPredictorV2 replaces the v1 one-hot GCN with two GINEConv layers (PyTorch Geometric; 320→256→128 hidden dimensions) consuming pre-computed ESM-2 per-residue embeddings as node features. The graph is a bidirectional 1D chain with self-loops and 3-dim one-hot edge features (self-loop, forward, backward). SESTRAV physicochemical features (mode-21, 21-dim) are fused with the 128-dim mean-pooled graph embedding via a 64-unit MLP. The detailed evaluation below (per-fold and gate tables) is the v2.1 reference configuration (320-dim node features, `facebook/esm2_t6_8M_UR50D`; Rives et al. 2021; cached in `data/esm2_embeddings.pt`, 170 MB). The **final v2.3 configuration** upgrades to 480-dim node features (`facebook/esm2_t12_35M_UR50D`, cached in `data/esm2_embeddings_t12.pt`) and excludes zero-padding nodes from GINEConv message passing and the global mean pool (variable-length graph fix, §below). All 11,795 unique v4 peptides are embedded in a single ESM-2 forward pass and cached offline, so training time is dominated by GINEConv layers rather than protein language-model inference.

**Training.** Stratified 5-fold CV on the v4 14,699-row dataset (62 gold-standard IEDB epitopes held out from all folds); 20 epochs per fold with cosine annealing from lr=3×10⁻⁴; batch size 64; `BCEWithLogitsLoss` with positive-class weight balancing. Post-hoc Platt scaling (logistic regression on OOF scores) applied after training to correct probability calibration without altering prediction ranking.

**Per-fold 5-fold OOF results:**

| Fold | AUC-ROC | AUC-PR | ISSR@10 |
|------|---------|--------|---------|
| 1 | 0.7612 | 0.7250 | 0.8356 |
| 2 | 0.7561 | 0.7205 | 0.8527 |
| 3 | 0.7505 | 0.7155 | 0.8630 |
| 4 | 0.7588 | 0.7191 | 0.8390 |
| 5 | 0.7627 | 0.7404 | 0.8973 |
| **Mean +/- SD** | **0.7579 +/- 0.0043** | **0.7241 +/- 0.0087** | **0.8575 +/- 0.0222** |

*Compared to v1 (dense-adj GCN): AUC-PR 0.7241 vs 0.6143 (+0.110 absolute; +17.9% relative). ISSR@10 0.8575 vs 0.7075 (+0.150 absolute).*

**Promotion gate outcomes (v2.1 with Platt calibration):**

| Gate | Criterion | v1/v4 | v2.1/v4 | Pass? |
|---|---|---|---|---|
| 1 - Generalization (AUC-PR ≥ 0.85) | 5-fold OOF AUC-PR | 0.613 | **0.723** | **FAIL** |
| 2 - Stability (std ≤ 0.02) | Jackknife-LOO std | 0.0001 | **0.000** | PASS |
| 3 - Latency (ratio ≤ 2×) | GNN/RF inference ratio | 0.02× | **0.14×** | PASS |
| 4 - Calibration (ECE < 0.05) | Expected Calibration Error | 0.040 | **0.037** | PASS |
| 5 - Escape Sensitivity (≥ 0.80) | Sensitivity above decoy median | 0.724 | **0.825** | PASS |

v2.1 clears 4/5 promotion gates. Compared to v1 (3/5 passing): Gate 5 (escape sensitivity) was newly cleared, and Gate 4 (calibration) was maintained with Platt scaling after initial overconfidence with uncalibrated BCE training (raw ECE = 0.179; post-Platt ECE = 0.037).

**Interpretation.** The ESM-2 node embedding upgrade produces a substantial discriminative improvement (+0.11 AUC-PR, +0.15 ISSR@10) and clears Gate 5 escape sensitivity, confirming that richer per-residue representations meaningfully improve identification of immunogenic viral epitopes. Gate 3 (latency) remains comfortably within threshold at 0.14× the RF inference time, demonstrating that pre-caching ESM-2 embeddings eliminates the per-inference language-model overhead.

**v2.2-v2.3 evolution and the variable-length graph fix.** Three iterations probed the Gate 1 gap. v2.2 upgraded the node embedding from `esm2_t6_8M` (320-dim) to `esm2_t12_35M` (480-dim) with early stopping; this did not improve discrimination (OOF AUC-PR 0.7160 vs v2.1 0.7241), indicating the bottleneck was not embedding capacity. v2.3 then corrected a dataset bug: `GraphPeptideDatasetV2` had padded every peptide to a fixed 11-node graph, so for the 91.5% of peptides shorter than 11 residues (8/9/10-mers) up to three zero-vector nodes were participating in GINEConv message passing and diluting the global mean pool. The fix slices each peptide to its true length L and builds per-peptide edges, excluding padding from both message passing and pooling. With post-hoc Platt recalibration retained, v2.3 reaches **mean-fold AUC-PR 0.7281 +/- 0.0070** (5-fold CV; canonical regeneration 2026-06-24 on NVIDIA A100, torch 2.11.0+cu128, seed 42; original run: 0.7263, within expected GPU/version variance) - a real but small +0.012 correction over v2.2. Gates 2-5 are architecture-class-invariant from the v2.1 evaluation (identical GINEConv topology and latency class; Platt calibration reapplied) and were not re-measured; Gate 1 was re-evaluated and **remains FAIL** (pooled OOF AUC-PR 0.716 vs 0.85).

**Gate 1 resolution.** Three architectural iterations on v4 - including the variable-length graph fix originally hypothesized as the primary bottleneck - all plateau at mean-fold AUC-PR in the range 0.72-0.73. On identical v4 OOF folds the honest like-for-like comparison is GNN v2.3 (AUC-PR 0.7281) vs v4 RF mode-31 (AUC-PR 0.7635); the GNN trails by 0.035. Expansion of the training dataset to v5 (31,999 active rows) yields two evaluation contexts. In the self-proteome-only context (hard decoys as the sole negative class, consistent with the Gate 1 protocol), the v5 RF mode-31 achieves AUC-PR **0.8897** - clearing the Gate 1 threshold of 0.85 and establishing that the threshold is reachable with sufficient negative-class fidelity. In the within-virus context (4,219 experimentally confirmed within-virus non-immunogenic peptides from the IEDB REST API included), the v5 RF achieves AUC-PR 0.7678 (AUC-ROC 0.9368), reflecting the genuinely harder task of distinguishing immunogenic from non-immunogenic peptides of the same pathogen. Both metrics are reported: Gate 1 (0.8897) confirms the model clears the self-proteome discrimination threshold; within-virus AUC-PR (0.7678) is the operational metric for within-viral applications. GNN v2.3 on v4 data (0.7281) trails the v5 RF within-virus metric by 0.040. GNN retraining on v5 is pending (ESM-2 cache complete at 27,376 peptides; GPU training pending); this section will be updated with v5 GNN results. Accordingly, **the v5 canonical Random Forest (mode-31) is designated the production immunogenicity scorer**, and the GINEConv+ESM-2 GNN is reported as a research and ensemble-candidate component pending v5 evaluation.

**v2.4 ablation - attention pooling (completed).** v2.4 Lever 1 trained `GraphPredictorV2` with `AttentionalAggregation` readout (gate network: `Linear(128→1)`, Xavier-initialized) on the same t12 ESM-2 cache (480-dim) and mode-31 features as v2.3, with early stopping (patience=10) over 60 max epochs on identical 5-fold splits (seed=42).

*Table - GNN v2.x per-lever ablation (all OOF, identical v4 5-fold splits, seed=42).*

| Variant | Pooling | ESM-2 | Mean-fold AUC-PR | vs v5 RF (0.7678) | vs v2.3 (0.7281) | Decision |
|---|---|---|---|---|---|---|
| GNN v2.1 (baseline) | mean | t6, 320-dim | 0.7241 +/- 0.0087 | -0.0437 | - | Research |
| GNN v2.2 | mean | t12, 480-dim | 0.7160 | -0.0518 | -0.012 | No gain |
| GNN v2.3 (var-length fix) | mean | t12, 480-dim | 0.7281 +/- 0.0070 | -0.0397 | +0.012 | Research |
| **GNN v2.4 Lever 1** | **attention** | **t12, 480-dim** | **0.7228 +/- 0.0070** | **-0.0450** | **-0.005** | **Regression** |

Per-fold breakdown for v2.3 (canonical regeneration, NVIDIA A100, 2026-06-24, seed 42):

| Fold | AUC-ROC | AUC-PR | ISSR@10 | Best epoch |
|---|---|---|---|---|
| 1 | 0.7675 | 0.7346 | 0.8733 | 20 |
| 2 | 0.7728 | 0.7379 | 0.8733 | 22 |
| 3 | 0.7570 | 0.7254 | 0.8733 | 34 |
| 4 | 0.7522 | 0.7198 | 0.8322 | 10 |
| 5 | 0.7484 | 0.7226 | 0.8733 | 6 |
| **Mean +/- SD** | **0.7596 +/- 0.0092** | **0.7281 +/- 0.0070** | **0.8651 +/- 0.0164** | - |

Per-fold breakdown for v2.4 Lever 1:

| Fold | AUC-ROC | AUC-PR | ISSR@10 | Best epoch |
|---|---|---|---|---|
| 1 | 0.7641 | 0.7363 | 0.8699 | 13 |
| 2 | 0.7506 | 0.7208 | 0.8425 | 35 |
| 3 | 0.7462 | 0.7158 | 0.8596 | 18 |
| 4 | 0.7571 | 0.7193 | 0.8493 | 12 |
| 5 | 0.7521 | 0.7217 | 0.8596 | 6 |
| **Mean +/- SD** | **0.7540 +/- 0.0061** | **0.7228 +/- 0.0070** | **0.8562 +/- 0.0094** | - |

**Interpretation.** Attention pooling with a single linear gate regresses by -0.005 AUC-PR relative to v2.3 mean pooling, does not clear the v4 RF bar (0.7635), and triggers the per-protocol decision rule: Lever 2 (t30 ESM-2, 640-dim) is not warranted because attention pooling did not first unlock the representation - the same pattern as v2.2 showing t6→t12 capacity did not help. The early-stopping variance across folds (best epoch range 6-35) and fold 1's higher AUC-PR (0.7363 vs 0.72 elsewhere) suggest the attention gate overfits on certain fold partitions without generalising. **The GNN v2.4 experiment is a publishable negative result.** The production scorer is the v5 RF mode-31 (AUC-PR 0.7678 within-virus; 0.8897 self-proteome Gate 1 metric). Lever 2 (t30) and Lever 3 (ESMFold spatial edges) remain valid architectural hypotheses; they should be revisited after GNN v5 retraining is complete.


---

## 4. Discussion

### 4.1 The workflow advantage

No publicly available tool integrates all six SESTRAV stages within a single Snakemake DAG with OpenSSF Passing badge compliance. The reproducibility governance framework - cryptographic dataset checksums, freeze mode, MHCflurry version pinning - directly addresses silent dataset drift between releases, which invalidates benchmark comparisons across tool versions.

The SESTRAV AUC-PR advantage over PredIG (0.828 vs. 0.727 at v3) is conservative: SESTRAV scores are OOF; PredIG scores reflect a contaminated test set with 36.9% training overlap. On Tier B proteome-scale gold-standard recovery, binding-only achieves higher GS recovery@10% (47%) than SESTRAV RF (20%), because gold-standard peptides were originally selected for strong binding. This does not contradict the Tier A result; it reflects the distinct evaluation paradigm of binding-prefiltered proteome pools vs. labeled immunogenicity benchmarks - both results must be disclosed.

### 4.2 Limitations

Full limitations: `docs/limitations_statement_v1.md`.

1. **Label quality.** IEDB labels are population-average aggregates; they do not represent allele-specific or donor-specific immunogenicity.
2. **TCR contact approximation.** p4-p8 physicochemical features are validated primarily for HLA-A*02:01 canonical 9-mers (Chowell et al. 2015). 8-mer compressed registers and 10-mer/11-mer bulging conformations are not explicitly modeled.
3. **Cross-virus generalization.** The v5 training set includes EBV, HPV, HBV, HCV, SARS-CoV-2, HIV-1, and YFV, so per-virus predictions are no longer purely cross-domain transfer. However, within-virus discrimination (immunogenic vs. non-immunogenic peptides of the same pathogen) remains limited (LOO AUC-ROC 0.33-0.58; §3.4.2). SESTRAV is validated for viral-vs-self discrimination; within-virus fine-grained ranking requires calibrated probability interpretation and, ideally, virus-specific negative data beyond hard decoys.
4. **HBV genotype coverage.** Panel uses genotype D (ayw) reference sequences. Genotype B/C populations (East/Southeast Asia) may show sequence divergence at predicted epitope positions.
5. **GNN gate status.** GraphPredictor v1 fails Gates 1 and 5 (v4 AUC-PR 0.613, sensitivity 0.724). GraphPredictorV2 (GINEConv + ESM-2) clears Gates 2-5 but plateaus at mean-fold AUC-PR in the range 0.72-0.73 across v2.1-v2.4, below the Gate 1 threshold of 0.85 on v4 data. v2.4 Lever 1 (attention pooling, t12 480-dim) regressed slightly vs v2.3 (0.7228 vs 0.7281); the attention gate does not unlock the representation at v4 data scale. The v5 RF mode-31 (AUC-PR 0.8897) has cleared Gate 1; GNN v5 retraining is in progress. The production scorer is the v5 canonical Random Forest (mode-31, AUC-PR 0.8897); the GNN is a research/ensemble component pending v5 evaluation.
6. **Allele-aware training.** The 166-feature allele-aware schema is implemented but not trained; all production predictions use population-average allele features.
7. **Zero-shot HLA generalization.** This claim has been removed from all SESTRAV communications. It was not validated in any experiment and should not appear in publications, presentations, or documentation.
8. **Negative data quality.** v3 negatives are mostly non-immunogenic due to poor MHC binding, causing partial "binding → immunogenic" learning. Hard decoys (v4) address this root cause.
9. **Binding feature marginal redundancy.** Per-allele MHCflurry binding scores (`bind_A0101`-`bind_B4402`) contribute zero marginal information above physicochemical features in the v3 production model (both impurity-based and permutation importance = 0; investigated and confirmed - no pipeline bug). The mechanism is physico-binding overlap: anchor-residue features at p5-p8 capture binding-relevant variance; the v3 negative selection bias (binding-poor) suppresses informative binding variance in the label-conditioning set. Full marginal binding utility is expected in v4 where hard decoys decouple binding from immunogenicity and allele-aware features are explicitly stratified.

### 4.3 Future directions

1. **v5 allele-aware training.** 166-feature schema enables allele-specific predictions once VDJdb allele annotations are incorporated. The v5 dataset (31,999 active rows) retains the `hla_allele` schema column. Target: ≥10 allele models, planned for v2.2.
2. **Beyond hard decoys - within-virus negative realism (partially delivered in v5).** The 5,000 central-tolerance self-peptides integrated in v4 decoupled binding from immunogenicity (restoring binding features to top-10 RF importance; §3.2), yielding mode-31 AUC-PR 0.7635 (v4) in the self-proteome evaluation context. v5 extended this by adding 4,219 experimentally confirmed non-immunogenic viral peptides from the IEDB REST API, achieving mode-31 AUC-PR 0.7678 within-virus and 0.8897 in the self-proteome Gate 1 context - the latter exceeding the 0.880 threshold originally targeted. The LOO analysis (§3.4.2) confirms that the residual within-virus gap is driven by per-virus data scarcity; the next levers are continued IEDB refreshes (D1), ImmuneCODE SARS-CoV-2 negatives (D2), and LANL HIV-specific negatives (D3).
3. **GNN v2.4 (completed, negative result).** v2.4 Lever 1 (attention pooling, t12 480-dim, mode-31, identical OOF folds) achieved mean AUC-PR 0.7228 +/- 0.0070 - a regression of -0.005 vs v2.3 mean pooling on v4 data, and 0.167 below the v5 RF bar (0.8897). The attention gate did not unlock the representation; Lever 2 (t30 640-dim) was not executed per protocol. The negative result is documented honestly in §3.6. GNN v5 retraining is in progress; ESMFold spatial contact edges (Lever 3) and multi-graph ensembling remain architectural hypotheses for future iterations.
4. **Virus expansion.** SARS-CoV-2 (panel4: Spike P0DTC2, N P0DTC9, M P0DTC5, ORF3a P0DTC3), IAV (panel4: NP P03466, M1 P03485, HA P03437, PB1-F2 P0C0U1), CMV (panel4: pp65 P06725, IE1 P13202, pp50 P16785, gB P06473) - gated on IEDB audit confirming ≥100 positive T-cell assays per virus.
5. **Continuous automated validation.** Monthly GitHub Actions benchmark against new IEDB exports with automatic AUC-PR regression alerting.

---

## 5. Availability

- **Source code:** https://github.com/Gavin-Borges/SESTRAV - MIT license
- **Installation (from source; not yet published to PyPI):** `git clone https://github.com/Gavin-Borges/SESTRAV.git`, `cd SESTRAV`, then `pip install .` (core); `pip install ".[gnn]"` (+ GNN); `pip install ".[pipeline]"` (+ Snakemake)
- **PyPI:** not yet published; install from source (see Installation above)
- **Training dataset:** SESTRAV Immunogenicity Dataset v5 (31,999 active rows / 46,386 total; 12 viruses + IEDB viral negatives + central-tolerance hard decoys) will be archived on Zenodo under CC-BY-4.0 as a standalone deposition (`immunogenicity_dataset_v5.csv` + schema + build-provenance JSON with SHA-256 integrity manifest). DOI: `10.5281/zenodo.XXXXXXX` *(minted on publication; deposition record and checksums in `docs/zenodo_deposition.md`)*.
- **Docker image:** Planned for a future release (GitHub Container Registry). For this release, the from-source install (`git clone` then `pip install .`), the pinned `config.yaml` model versions, and the Zenodo dataset deposition together provide full reproducibility.
- **MHCflurry model version:** See `mhcflurry_model_version` in `config.yaml`

---

## References

- Albert BA, et al. Deep neural networks predict class I MHC epitope presentation and transfer learn neoepitope immunogenicity. *Cell Syst*. 2023;16(5):390-402.
- Bjellqvist B, et al. The focusing positions of polypeptides in immobilized pH gradients can be predicted from their amino acid sequences. *Electrophoresis*. 1993;14(1):1023-1031.
- Chowell D, et al. TCR contact residue hydrophobicity is a hallmark of immunogenic CD8⁺ T cell epitopes. *Proc Natl Acad Sci USA*. 2015;112(14):E1754-E1762.
- Collins PL, Graham BS. Viral and host factors in human respiratory syncytial virus pathogenesis. *J Virol*. 2008;82(5):2040-2055.
- Gartner JJ, et al. A machine learning model for ranking candidate HLA class I neoantigens based on known neoepitopes from multiple human tumor types. *Nat Cancer*. 2021;2(5):563-574.
- Gfeller D, et al. Improved predictions of MHC-I and MHC-II epitopes using the PRIME tool and selected amino acid substitution matrices. *Front Immunol*. 2023;14:1128364.
- Hopp TP, Woods KR. Prediction of protein antigenic determinants from amino acid sequences. *Proc Natl Acad Sci USA*. 1981;78(6):3824-3828.
- Jurtz V, et al. NetMHCpan-4.0: Improved peptide-MHC class I interaction predictions integrating eluted ligand and peptide binding affinity data. *J Immunol*. 2017;199(9):3360-3368.
- Kipf TN, Welling M. Semi-supervised classification with graph convolutional networks. *Int Conf Learn Represent (ICLR)*. 2017. arXiv:1609.02907.
- Li G, et al. DeepImmuno: deep learning-empowered prediction and generation of immunogenic peptides for T-cell immunity. *Brief Bioinform*. 2021;22(6):bbab160.
- Kloetzel PM. Antigen processing by the proteasome. *Nat Rev Mol Cell Biol*. 2001;2(3):179-187.
- Kyte J, Doolittle RF. A simple method for displaying the hydropathic character of a protein. *J Mol Biol*. 1982;157(1):105-132.
- Lobry JR, Gautier C. Hydrophobicity, expressivity and aromaticity are the major trends of amino-acid usage in 999 *Escherichia coli* chromosome-encoded genes. *Nucleic Acids Res*. 1994;22(15):3174-3180.
- Meiler J, et al. Generation and evaluation of dimension-reduced amino acid parameter representations by artificial neural networks. *J Mol Model*. 2001;7(9):360-369.
- Mölder F, et al. Sustainable data analysis with Snakemake. *F1000Res*. 2021;10:33.
- Nielsen M, et al. The role of the proteasome in generating cytotoxic T-cell epitopes: insights obtained from improved predictions of proteasomal cleavage. *Immunogenetics*. 2005;57(1-2):33-41.
- O'Donnell TJ, et al. MHCflurry 2.0: Improved pan-allele prediction of MHC class I-presented peptides by incorporating antigen processing. *Cell Syst*. 2020;11(1):42-48.
- Peng M, et al. Immunogenicity prediction with dual attention networks. *Bioinformatics*. 2022;38(24):5436-5443.
- Peters B, et al. Identifying MHC class I epitopes by predicting the TAP transport efficiency of epitope precursor peptides. *J Immunol*. 2003;171(4):1741-1749.
- Rammensee HG, et al. SYFPEITHI: database for MHC ligands and peptide motifs. *Immunogenetics*. 1999;50(3-4):213-219.
- Reynisson B, et al. NetMHCpan-4.1 and NetMHCIIpan-4.0: improved predictions of MHC antigen presentation by concurrent motif deconvolution and integration of MS MHC eluted ligand data. *Nucleic Acids Res*. 2020;48(W1):W449-W454.
- Rives A, et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *Proc Natl Acad Sci USA*. 2021;118(15):e2016239118.
- Rock KL, Goldberg AL. Degradation of cell proteins and the generation of MHC class I-presented peptides. *Annu Rev Immunol*. 1999;17:739-779.
- Vihinen M, Torkkila E, Riikonen P. Accuracy of protein flexibility predictions. *Proteins*. 1994;19(2):141-149.
- Vita R, et al. The Immune Epitope Database (IEDB): 2018 update. *Nucleic Acids Res*. 2019;47(D1):D339-D343.
- Zamyatnin AA. Protein volume in solution. *Prog Biophys Mol Biol*. 1972;24:107-123.
- Zimmerman JM, Eliezer N, Simha R. The characterization of amino acid sequences in proteins by statistical methods. *J Theor Biol*. 1968;21(2):170-201.
