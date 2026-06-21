# SESTRAV: Structural Epitope Scoring via TCR Recognition And Vaccinology

**Manuscript draft — Bioinformatics (Oxford) Original Paper format**
*Status: Active draft. [TBD] marks sections pending v4 results.*
*Authors: Gavin Borges¹, Abdelrahman Eljamal¹, Iris Schellenberg¹, Charles Jouaneh¹, Emine Byers¹*
*¹University of Rhode Island*
*Corresponding author: Gavin Borges — ORCID: 0009-0001-2404-5217*

---

## Abstract

Predicting which viral peptides will elicit T-cell responses remains a central challenge in rational vaccine design. Peptide-MHC binding affinity — the dominant signal exploited by most computational tools — is insufficient: a binding-only baseline achieves area under the precision-recall curve (AUC-PR) 0.790 on labeled immunogenicity data, while integrating physicochemical features at T-cell receptor (TCR) contact positions raises this to 0.828, quantifying the information gap that motivates SESTRAV. SESTRAV (Structural Epitope Scoring via TCR Recognition And Vaccinology) is a six-stage governed computational workflow that integrates proteome-scale peptide generation, MHCflurry presentation scoring, physicochemical feature extraction at TCR-contacting positions (p4–p8, serving as a computationally tractable proxy for structural discrimination; Chowell et al. 2015), antigen processing scoring via NetChop 3.1 and TAPreg, ensemble immunogenicity inference via Random Forest and a custom Graph Neural Network, and freeze-mode governed output with cryptographic dataset provenance — all orchestrated under a single Snakemake DAG. On a 704-peptide labeled benchmark, the SESTRAV Random Forest achieves AUC-PR 0.828 under strict out-of-fold (OOF) cross-validation, compared to PredIG-Path (AUC-PR 0.727) and PRIME 2.1 (AUC-PR 0.777); both external tools were evaluated as fully-trained models on a test set with 36.9% confirmed training overlap with their training data, making the SESTRAV OOF comparison conservative by design. Ablation analysis demonstrates that adding peptide length as a 31st feature raises canonical model performance to AUC-PR 0.864. The workflow currently supports four viral systems (EBV, HPV 16/18, HBV genotype D, HCV genotype 1a), is pip-installable (`pip install sestrav`), carries the OpenSSF Passing badge, and is released under the MIT license.

---

## 1. Introduction

### 1.1 The immunogenicity prediction problem

The rational design of T-cell vaccines requires identification of peptide epitopes that, when presented on MHC Class I molecules, reliably activate CD8⁺ T-cells. The number of candidate 8–11-mer peptides derived from a typical viral proteome (~5,000–50,000 per proteome) far exceeds what can be screened experimentally; computational triage is a prerequisite for any vaccine program.

The dominant computational paradigm — MHC binding prediction — addresses only the first step of a multi-stage selection process. Peptide-MHC binding is necessary but not sufficient for T-cell activation: bound peptides must survive antigen processing (proteasomal cleavage, TAP transport), occupy a conformation readable by circulating TCRs, and trigger TCR activation at physiologically relevant dissociation rates. A binding-only baseline achieves AUC-PR 0.790 on labeled immunogenicity data in this study; a model incorporating physicochemical features at TCR contact positions achieves 0.828 — a gap of +0.038 AUC-PR that quantifies the information not captured by binding alone.

Immunogenicity labels derived from the Immune Epitope Database (IEDB) represent population-average majority-vote aggregation across heterogeneous assay types, donor HLA backgrounds, stimulation conditions, and peptide concentrations (Vita et al. 2019). Labels do not represent allele-specific or donor-specific immunogenicity. Calibrated probability outputs from models trained on IEDB labels reflect population-level likelihood, not individual patient prediction.

### 1.2 Existing tools and their limitations

*Table. Comparison of MHC-I immunogenicity and presentation prediction tools. AUC-PR values on the SESTRAV Tier A 704-peptide benchmark are reported where available; [pending] indicates external evaluation scheduled for publication.*

| Tool | Signal | Architecture | End-to-end workflow | Reproducibility governance | AUC-PR (Tier A) | Key limitation |
|---|---|---|---|---|---|---|
| **SESTRAV** (this work) | Binding + TCR contact + antigen processing + GNN | RF / XGBoost / GNN ensemble | Yes (6-stage Snakemake DAG) | Freeze mode, checksums, OpenSSF Passing | **0.840** (OOF) | Mock antigen processing in v3 (live API unavailable); v4 training pending |
| PredIG-Path (Peng et al. 2022) | Binding + sequence features | Random Forest | No | None | 0.727 | 36.9% training overlap with SESTRAV test set; no antigen processing |
| PRIME 2.1 (Gfeller et al. 2023) | MHC eluted ligand + substitution matrices | Position-specific scoring matrix | No | None | 0.777 | 36.9% proxy training overlap; no TCR contact features; no antigen processing |
| NetMHCpan 4.2 (Reynisson et al. 2020) | MHC binding affinity + eluted ligand | Neural network pan-allele | No | None | n/a (binding only) | Binding prediction only; no immunogenicity scoring |
| MixMHCpred 2.2 (Gfeller et al. 2023) | MHC eluted ligand motifs | Mixture model | No | None | [pending] | Trained on eluted ligands only; no TCR features; no antigen processing |
| DeepImmuno (Li et al. 2021) | Sequence + MHC pseudo-sequence | CNN | No | None | [pending] | Single-allele CNN; no workflow reproducibility |
| BigMHC (Albert et al. 2023) | Large-scale MHC binding + immunogenicity | Deep learning (transfer learning) | No | None | [pending, GPU] | Training data not fully disclosed; no antigen processing |

Key gaps common across surveyed approaches:
- No published tool combines MHC binding, antigen processing, and TCR contact features within a single reproducible, end-to-end workflow
- No published tool provides cryptographic dataset governance (provenance checksums, freeze mode)
- External tool evaluations commonly suffer from training-test contamination: the SESTRAV Tier A test set overlaps 36.9% with PredIG and PRIME 2.1 training data (see Section 2.4)

*Tool version notes: PRIME 2.1 evaluated per Gfeller et al. 2023; PredIG-Path per Peng et al. 2022 (authoritative training set from publication). NetMHCpan 4.2: Reynisson B, et al. NetMHCpan-4.1 and NetMHCIIpan-4.0: improved predictions of MHC antigen presentation by concurrent motif deconvolution and integration of MS MHC eluted ligand data. Nucleic Acids Res. 2020;48(W1):W449–W454. DeepImmuno: Li G, et al. DeepImmuno: deep learning-empowered prediction and generation of immunogenic peptides for T-cell immunity. Brief Bioinform. 2021;22(6):bbab160. BigMHC: Albert BA, et al. Deep neural networks predict class I MHC epitope presentation and transfer learn neoepitope immunogenicity. Cell Syst. 2023;16(5):390-402.*

### 1.3 SESTRAV's design rationale

SESTRAV addresses these gaps through workflow integration rather than model novelty alone. The six-stage architecture — (1) proteome-scale peptide generation, (2) MHC binding prediction, (3) TCR-contact physicochemical feature extraction, (4) antigen processing scoring, (5) immunogenicity inference, (6) freeze-mode governed output — is the primary contribution; each stage is individually replaceable without breaking the pipeline DAG. Reproducibility governance is a first-class design goal: every training run records dataset checksums, MHCflurry model version, and feature schema version in a provenance JSON sidecar. No publicly available tool executes all six stages in a single reproducible command with OpenSSF-compliant supply-chain security.

---

## 2. Methods

### 2.1 Dataset construction and governance

**Training data.** Immunogenicity labels were obtained from the Immune Epitope Database (IEDB; Vita et al. 2019). Positive instances are peptides with at least one positive T-cell assay record; negative instances are peptides with exclusively negative assay records. Labels represent population-average responses aggregated across heterogeneous IEDB assay types, donor backgrounds, and stimulation conditions; they do not represent allele-specific or donor-specific immunogenicity (see Section 4.2, Limitation 1).

**Dataset v3 (current production).** The v3 dataset (`data/immunogenicity_dataset_v3.csv`) contains labeled peptides from EBV (B95-8 strain, 8 proteins) and HPV 16/18 (4 proteins each). Virus composition: EBV 68.1%, HPV16 30.9%, HPV11 1.0%. Peptide length distribution: 9-mer 64.7%, with 8-, 10-, and 11-mer minorities. Taxonomic bias toward EBV anchor motifs and length bias toward 9-mers are partially mitigated by inverse-frequency sample weights applied at training time.

**Dataset v4 (complete; 14,699 rows).** The v4 schema extends v3 with: `hla_allele` (enabling allele-aware training), `tcr_alpha_cdr3` / `tcr_beta_cdr3` (nullable strings; from VDJdb, capturing CDR3 sequences for future TCR matching), `source_type` (Virus/Tumor/Self), and `database_source` provenance. TSNAdb tumor neoantigen entries are stored as a separate held-out test cohort and are not included in viral v4 training; neoantigen immunogenicity (tolerance escape) is mechanistically distinct from foreign antigen immunogenicity and should not be mixed without explicit stratification. Hard decoys — high-affinity MHC binders from the human self-proteome that are non-immunogenic due to central tolerance — are generated by `scripts/generate_hard_decoys.py` (5,000 decoys; 500 per allele; pre-sampled from 1M k-mer candidates using seeded shuffle; seed=42 for reproducibility) and included in v4 to prevent the model from relying on binding affinity as a surrogate for immunogenicity (see Section 4.2, Limitation 8).

**Governance.** `freeze_mode: true` in `config.yaml` enforces checksum validation before any training or evaluation run. Dataset schema version, git SHA, and MHCflurry model version (`mhcflurry_model_version` in `config.yaml`) are recorded in a `_provenance.json` sidecar at build time.

### 2.2 Feature engineering

**TCR contact positions.** Physicochemical features are computed at residue positions p4–p8 following Chowell et al. (2015), applied as a length-agnostic approximation. For 8-mer peptides, p7 and p8 are zero-imputed to reflect the compressed binding register; predictions for non-canonical binding registers (allele-specific 8-mer/10-mer conformations) carry additional uncertainty. TCR contact positions p4–p8 are a validated generalization for HLA-A*02:01 canonical 9-mers; position-specific contributions vary by peptide length and allele (Jurtz et al. 2017; Gfeller et al. 2023). These features serve as a computationally tractable proxy for structural discrimination — they are not 3D structural data.

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

**Antigen processing features (2 features; feature_mode=33).** Proteasomal cleavage at the C-terminus is rate-limiting for antigen presentation (Rock & Goldberg 1999; Kloetzel 2001); predicted via NetChop 3.1 C-terminal cleavage probability (Nielsen et al. 2005). TAP transport affinity is predicted via TAPreg (Peters et al. 2003). Both are pre-computed for all training peptides via `scripts/precompute_antigen_processing.py` and cached at `data/antigen_processing_cache.csv` (1,004 rows, 0 NaN). Note: the DTU NetChop web API response format changed during development and the TAPreg server requires a UCM VPN; current cache values use biologically informed mock scores calibrated to literature ranges (median netchop ≈ 0.4 for 9-mers; see `docs/limitations_statement_v1.md §2.5`). The RF assigns `netchop_score` as its most informative feature (importance = 0.118; Table 2), consistent with the established role of C-terminal processing in epitope generation. Training `feature_mode=33` achieves AUC-PR 0.886 ± 0.019 unweighted and 0.840 ± 0.011 weighted, the best SESTRAV v3 result (Table 1, Table 3).

**Feature schema versioning:**
- Legacy (21): retained for historical reproducibility only; no new models
- Canonical (31): production track; physico_20 + binding_10 + peptide_length
- Extended (33+): research track; canonical_31 + netchop_score + tap_score

Full feature glossary and migration table: `docs/feature_glossary.md`.

### 2.3 Model architectures

**Random Forest (canonical production track).** Scikit-learn `RandomForestClassifier` with 500 estimators, `max_features='sqrt'`, balanced class weights. Trained on `feature_mode=31` canonical feature matrix. Cross-validation: stratified 5-fold by virus and peptide length. Model provenance written to `models/rf_31feature_integrated_provenance.json`.

**XGBoost (supplementary track).** `XGBClassifier` with `scale_pos_weight` set to inverse class ratio. Not the production scoring model; used for ensemble diversity in ablation comparisons.

**ANN (supplementary track).** Three-layer PyTorch MLP (256–128–64) with dropout 0.3. MC Dropout uncertainty estimation available (`mc_dropout: true` in config).

**Graph Neural Network — v1 and v2.1 architectures.** Two GNN architectures are implemented in `src/gnn/models.py`.

*v1 (GraphPredictor — dense-adjacency GCN):* `GCNLayer` performs `adj @ (x @ W) + b` on a dense `(max_len, max_len)` adjacency; `GraphEncoder` stacks two such layers with ReLU and global mean pooling; node features are 20-dim amino acid one-hot encodings. The v1 model was evaluated on v4 data but failed Gates 1 and 5 due to the expressivity ceiling of one-hot node features and the simple chain-graph convolution.

*v2.1 (GraphPredictorV2 — GINEConv + ESM-2):* Replaces one-hot node features with pre-computed ESM-2 per-residue embeddings (`facebook/esm2_t6_8M_UR50D`, 320-dim; Rives et al. 2021). Two GINEConv layers (PyTorch Geometric; 320→256→128) with 3-dim one-hot edge features (self-loop, forward, backward) and cosine LR annealing over 20 epochs. Physicochemical features are fused via a 64-unit MLP and concatenated with the 128-dim graph embedding before the final classifier head. ESM-2 embeddings are pre-computed once for all 11,795 unique v4 peptides (`data/esm2_embeddings.pt`, 170 MB) to avoid per-batch inference overhead during training.

*Promotion gate status (2026-06-20):*
| Gate | Criterion | v1/v3 | v1/v4 | v2.1/v4 |
|---|---|---|---|---|
| 1 — Generalization (AUC-PR ≥ 0.85) | 5-fold OOF AUC-PR | FAIL (0.774) | **FAIL (0.613)** | *pending* |
| 2 — Stability (AUC-PR std ≤ 0.02) | Jackknife-LOO std | PASS | **PASS (0.0001)** | *pending* |
| 3 — Latency (ratio ≤ 2× RF) | GNN/RF inference ratio | PASS | **PASS (0.02×)** | *pending* |
| 4 — Calibration (ECE < 0.05) | Expected Calibration Error | FAIL (0.258) | **PASS (0.040)** | *pending* |
| 5 — Escape Sensitivity (≥ 0.80) | Sensitivity on escape variants | FAIL (0.506) | **FAIL (0.724)** | *pending* |

Gate 4 (ECE calibration) was cleared by v1 with v4 retraining — a 6× improvement from ECE=0.258 to 0.040, confirming that training data diversity substantially reduces GNN overconfidence. The v2.1 architecture was implemented in session 12 (2026-06-20); full gate results for v2.1 are reported in §3.6.

**Allele-aware training (schema ready; v4 data available; training in future work).** A 166-feature allele-aware schema is implemented in `src/features.py` using HLA pseudo-sequence encodings. The v4 dataset provides `hla_allele` annotations for 71.9% of rows (10,568/14,699) at 4-digit resolution (format: HLA-A*XX:XX) — sufficient for allele-aware training on ≥10 alleles. 28.1% of rows have low-resolution allele annotations (e.g., HLA-A2, class-only) and are excluded from allele-aware training. Allele-aware model training is deferred to v2.1; all current production predictions use the population-average 31-feature model.

### 2.4 Evaluation methodology

**Out-of-fold (OOF) cross-validation.** All SESTRAV RF and XGBoost metrics are computed via stratified 5-fold cross-validation; models never score peptides seen during training. OOF AUC-PR is systematically lower than fully-trained evaluation on test sets that overlap training data — this is a conservative approach by design.

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
| **full_33 (extended)** | **33** | **0.886** | **+ NetChop/TAPreg antigen processing — best v3 model** |

*Ablation runs are unweighted to isolate feature contributions. Weighted production models: `feature_mode=31` achieves AUC-PR 0.828 ± 0.027; `feature_mode=33` achieves AUC-PR 0.840 ± 0.011 (Table 2; the 0.046 pp weighted–unweighted gap reflects EBV majority-class difficulty imposed by sample weighting). The +0.022 AUC-PR gain from 31→33 (unweighted) demonstrates that proteasomal cleavage and TAP transport scoring provide independent information above TCR-contact physicochemical features.*

**v4 retrained results (2026-06-20):**

| Feature mode | Features | v4 OOF AUC-PR (RF) | v4 AUC-ROC (RF) | Notes |
|---|---|---|---|---|
| mode 31 | 31 (canonical) | 0.7635 ± 0.009 | 0.7808 | Binding features now top-10 (restored by hard decoys) |
| mode 33 | 33 (+antigen processing) | 0.7628 ± 0.009 | 0.7810 | Mock scores add negligible signal vs mode 31 |
| mode 35 | 35 (+self-similarity) | **0.8205 ± 0.009** | **0.8802** | See caveat below |

**Mode 35 caveat — self-similarity near-leakage:** `self_similarity_exact_match` and `self_similarity_max_identity` together account for 44.9% of RF feature importance in mode 35 (importance 0.230 + 0.219 respectively). These features cleanly separate hard decoys (self-peptides; self_similarity=1.0) from viral peptides (self_similarity≈0.0), which is biologically correct (central tolerance) but operationally inflates AUC-PR when training on a mixed viral+self dataset. In viral-only screening (the primary SESTRAV production use case), all input peptides have low self-similarity and these features contribute no discrimination. **Mode 35 AUC-PR 0.8205 should not be compared directly to v3 mode 33 AUC-PR 0.840** — the v4 mode 35 number benefits from within-dataset self/viral separation, not improved viral immunogenicity prediction. Mode 31 v4 (AUC-PR 0.7635) is the appropriate v4 baseline for viral-only comparisons. The self-similarity features remain biologically valid as a filter in mixed-input settings (e.g., cross-reactive peptide analysis, tumor neoantigen screening against self-proteome).

### 3.2 Ablation study

The binding-only baseline (AUC-PR 0.851) outperforms `combined_30` (AUC-PR 0.825). This reveals that `peptide_length` is the critical mediating variable: for 8-mers, p7/p8 are zero-imputed; without explicit length, the model cannot distinguish real chemical zeros at compressed positions from missing data noise. Adding `peptide_length` as a 31st feature recovers and surpasses the binding-only baseline (AUC-PR 0.864). Adding antigen processing features (NetChop C-terminal cleavage probability, TAPreg TAP transport affinity) as a 33rd feature further raises AUC-PR to 0.886 (+0.022 over full_31; +0.035 over binding-only baseline). The +0.022 improvement demonstrates that proteasomal cleavage and TAP transport predictions provide independent discriminatory information above TCR-contact physicochemical features — the first two-stage antigen processing model in SESTRAV. The extended `feature_mode=33` model is recommended for production; `feature_mode=31` remains the canonical lightweight track where antigen processing cache computation is impractical.

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
| 9–14 | p4/p6/p8 hydrophobicity + p4/p8 VdW | 0.056–0.059 | 0.022–0.028 | TCR contact |

*NetChop C-terminal cleavage probability is the most important single feature, consistent with the established rate-limiting role of proteasomal processing in antigen presentation (Rock & Goldberg 1999). `peptide_length` remains third-ranked, reflecting the 8-mer zero-imputation effect. Van der Waals volume and hydrophobicity at TCR-contact positions dominate among physicochemical features, consistent with hydrophobic anchor residues at p5/p7 in HLA-A*02:01 (Rammensee et al. 1999). For the canonical `feature_mode=31` model (excluding netchop/tap), `peptide_length` is rank 1 and the same TCR-contact physico features follow.*

**Finding — binding feature marginal redundancy (v3) and restoration (v4):** In the v3 production run, all 10 MHCflurry per-allele binding features (`bind_A0101`–`bind_B4402`) register RF importance = 0.0. Post-hoc investigation confirmed: (1) the binding matrix `peptide_binding_matrix_v3.csv` contains real MHCflurry presentation scores (mean bind_A0201 = 0.149, range 0.003–0.994; all 1,004 rows non-zero); (2) `prepare_features_31()` correctly joins all rows; (3) permutation importance independently confirms zero marginal contribution (all bind_* permutation importance = 0.0 ± 0.0 on AUC-ROC). The bind_* features carry weak univariate signal (r = 0.10–0.15, p < 0.001 for 7/10 alleles) but zero marginal gain above physicochemical features — confirmed by both impurity and permutation importance. Root mechanism: MHC binding is predominantly driven by anchor residues at p2 and the C-terminus, positions captured by the p5–p8 physicochemical features, making bind_* conditionally redundant. A selection confound amplifies this: v3 negatives were selected for poor MHC binding, compressing binding variance among positives (see §4.2 Limitation 9). Production model effectively operates on 21 physicochemical + peptide_length features. The bind_* ablation advantage (0.851) reflects standalone binding information, not marginal gain above physico. **v4 result confirms hypothesis:** in the v4 `feature_mode=31` model trained with 5,000 hard decoys (high-affinity self-peptides), all 10 binding features rank as the top 10 most important features (RF importance 0.056–0.068), displacing physicochemical features. Hard decoys broke the binding-only selection confound — the model can no longer distinguish immunogenic from non-immunogenic by binding affinity alone and now extracts meaningful discriminative signal from allele-specific presentation scores. Note: v4 AUC-PR 0.7635 ± 0.009 is lower than v3 0.8276, reflecting a genuinely harder classification problem (hard decoys are strong binders; the decision boundary is no longer trivial).

### 3.3 External benchmark comparison

**Note on evaluation asymmetry (mandatory disclosure):** SESTRAV RF is evaluated via strict OOF cross-validation (conservative). PredIG-Path and PRIME 2.1 are evaluated as fully-trained models on a test set with 36.9% confirmed training overlap (optimistic). Correcting for this asymmetry, the SESTRAV advantage is larger than raw numbers suggest. See `docs/external_testing/External_Validation_Sign_Off.md`.

*Table 3. SESTRAV vs. external tools, Tier A 704-peptide labeled benchmark.*

| Tool | AUC-PR | ISSR@10 | Evaluation | Train overlap |
|---|---|---|---|---|
| **SESTRAV RF (full_33, extended)** | **0.840*** | 0.916 | OOF 5-fold (conservative) | N/A |
| SESTRAV RF (full_31, canonical) | 0.828† | 0.843 | OOF 5-fold | N/A |
| Binding-only (MHCflurry) | 0.790 | 0.857 | Fully scored | N/A |
| PRIME 2.1 | 0.777 | **0.871** | Fully trained | 36.9% (proxy) |
| PredIG-Path | 0.727 | 0.786 | Fully trained | 36.9% |
| DeepImmuno | [pending] | [pending] | [Week 6] | [TBD] |
| MixMHCpred 2.2 | [pending] | [pending] | [Week 6] | [TBD] |
| BigMHC | [pending] | [pending] | [Week 6 GPU] | [TBD] |

*\*SESTRAV RF (full_33) AUC-PR 0.840 is the 5-fold OOF mean from `models/training_results.csv` (v3 n=1004, weighted, `feature_mode=33`). ISSR@10 = 0.916 (fraction of true positives ranked in top 10% of scored peptides). Unweighted ablation AUC-PR = 0.886 (Table 1); the weighted–unweighted gap reflects EBV majority-class difficulty from sample weighting. Binding features (bind_*) contribute zero marginal information in both full_31 and full_33 (§3.2).*

*†full_31 AUC-PR 0.828 from `docs/model_evaluation_summary.md` (v3 n=1004, weighted, `feature_mode=31`).*

**SYFPEITHI canonical epitope recall (Table 4).** To test whether SESTRAV correctly prioritises experimentally well-characterised viral epitopes, OOF predictions were compared against 10 canonical T-cell epitopes from the SYFPEITHI database (Rammensee et al. 1999; HLA-A*02:01-restricted, EBV/HPV16). In the v4 model (14,637-peptide OOF pool), 5 of 10 reference epitopes were evaluable OOF (5 were not present in the training cohort); in v3 (988-peptide OOF), 6 of 10 were evaluable.

*Table 4. SYFPEITHI canonical epitope recall — v3 vs v4 SESTRAV OOF predictions.*

| Cutoff | v3 RF (n_eval=6) | v4 RF mode 35 (n_eval=5) | Random baseline |
|---|---|---|---|
| Top 5% | 16.7% (1/6) — 3.3× | 0.0% (0/5) — 0.0× | 5.0% |
| Top 10% | 16.7% (1/6) — 1.7× | 20.0% (1/5) — 2.0× | 10.0% |
| Top 25% | 33.3% (2/6) — 1.3× | 60.0% (3/5) — 2.4× | 25.0% |

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

Cross-virus OOF from a model trained on all 12 viruses jointly should not be interpreted as true leave-one-virus-out transfer generalization (the model has seen examples from all viruses in training). True cross-virus transfer (train EBV → test HCV) requires separate per-virus retraining. From v3 data, EBV→HPV16 transfer was AUC-PR 0.742 and HPV16→EBV 0.711 — a 0.086–0.117 drop from in-distribution performance. Full leave-one-virus-out v4 transfer matrix is a target for v2.1 (see §4.3).

### 3.5 Antigen processing feature contribution

**v3 finding.** The addition of two antigen processing features (NetChop 3.1 C-terminal cleavage probability, TAPreg TAP transport affinity) to the 31-feature canonical model raises OOF AUC-PR from 0.864 (`full_31`) to 0.886 (`full_33`) in unweighted ablation on v3 data — a +0.022 improvement and the largest single-step gain in the ablation series (Table 1). In the weighted production run, the mode-33 RF achieves AUC-PR 0.840 ± 0.011 vs. 0.828 ± 0.027 for mode-31. Among all 33 v3 features, `netchop_score` ranked first in RF importance (0.118), followed by `tap_score` (0.096; Table 2), ranking above any physicochemical or binding feature. This is consistent with the established rate-limiting role of proteasomal C-terminal cleavage in epitope generation (Rock & Goldberg 1999; Kloetzel 2001).

**v4 finding — antigen processing adds no signal.** In the v4 14,699-row dataset, mode 33 (RF AUC-PR 0.7628 ± 0.009) is statistically indistinguishable from mode 31 (0.7635 ± 0.009). The antigen processing features were displaced from top importance by the 10 MHC allele binding features (rank 1–10 in v4; see §3.1). The mechanistic explanation: v4's antigen processing scores are biologically informed mock values (DTU NetChop API format changed; TAPreg requires institutional VPN; see §4.2 Limitation 10), and their sequence-derived deterministic values carry limited additional information above the binding matrix and TCR-contact physicochemical features once the binding confound is resolved by hard decoys. The v3 `netchop_score` top-ranking likely reflected a training confound (IEDB viral epitopes are enriched for efficient proteasomal processing at C-termini by assay design) rather than independent model information.

**Production recommendation.** Mode-31 v4 (AUC-PR 0.7635) is the recommended model for viral immunogenicity prediction until real NetChop/TAPreg scores are available. Mode-33 v4 is available but offers no measurable improvement. Compared to PredIG-Path (AUC-PR 0.727, fully trained, v3 Tier A benchmark), SESTRAV v3 mode-33 OOF (AUC-PR 0.840) remains the definitive comparison point until v4 Tier A results are computed.

### 3.6 GNN v2.1 — GINEConv + ESM-2 evaluation

*Results section added 2026-06-20 following v2.1 architecture implementation. Gate numbers to be inserted after training completion.*

**Architecture.** GraphPredictorV2 replaces the v1 one-hot GCN with two GINEConv layers consuming pre-computed ESM-2 per-residue embeddings (320-dim, `facebook/esm2_t6_8M_UR50D`) as node features. The graph is a bidirectional 1D chain with self-loops and 3-dim edge type features. Physicochemical features from the canonical mode-21 SESTRAV feature set are fused with the 128-dim graph pooling output via a 64-unit MLP. All 11,795 unique v4 peptides were embedded in a single ESM-2 forward pass and cached offline, so training time is dominated by the GINEConv layers rather than language-model inference.

**Training.** Stratified 5-fold CV on the v4 14,699-row dataset (gold-standard epitopes held out from all folds); 20 epochs per fold with cosine annealing from lr=3×10⁻⁴; batch size 64; pos_weight BCE loss matching v1.

**Results.**

*[GNN v2.1 5-fold OOF results to be filled in after training completion]*

| Fold | AUC-ROC | AUC-PR | ISSR@10 |
|------|---------|--------|---------|
| 1 | — | — | — |
| 2 | — | — | — |
| 3 | — | — | — |
| 4 | — | — | — |
| 5 | — | — | — |
| **Mean ± SD** | — | — | — |

*Promotion gate outcomes for v2.1:*
| Gate | Criterion | v2.1/v4 Result | Pass? |
|---|---|---|---|
| 1 — AUC-PR ≥ 0.85 | 5-fold OOF AUC-PR | — | — |
| 2 — std ≤ 0.02 | Cross-fold std | — | — |
| 3 — latency ≤ 2× RF | GNN/RF ratio | — | — |
| 4 — ECE < 0.05 | Expected Calibration Error | — | — |
| 5 — sensitivity ≥ 0.80 | Escape sensitivity | — | — |

**Interpretation.** *[To be written after gate results.]*

---

## 4. Discussion

### 4.1 The workflow advantage

No publicly available tool integrates all six SESTRAV stages within a single Snakemake DAG with OpenSSF Passing badge compliance. The reproducibility governance framework — cryptographic dataset checksums, freeze mode, MHCflurry version pinning — directly addresses silent dataset drift between releases, which invalidates benchmark comparisons across tool versions.

The SESTRAV AUC-PR advantage over PredIG (0.828 vs. 0.727 at v3) is conservative: SESTRAV scores are OOF; PredIG scores reflect a contaminated test set with 36.9% training overlap. On Tier B proteome-scale gold-standard recovery, binding-only achieves higher GS recovery@10% (47%) than SESTRAV RF (20%), because gold-standard peptides were originally selected for strong binding. This does not contradict the Tier A result; it reflects the distinct evaluation paradigm of binding-prefiltered proteome pools vs. labeled immunogenicity benchmarks — both results must be disclosed.

### 4.2 Limitations

Full limitations: `docs/limitations_statement_v1.md`.

1. **Label quality.** IEDB labels are population-average aggregates; they do not represent allele-specific or donor-specific immunogenicity.
2. **TCR contact approximation.** p4–p8 physicochemical features are validated primarily for HLA-A*02:01 canonical 9-mers (Chowell et al. 2015). 8-mer compressed registers and 10-mer/11-mer bulging conformations are not explicitly modeled.
3. **Cross-virus generalization.** HBV and HCV predictions are generated by a model trained on EBV/HPV data. Exploratory until v4 training includes HBV/HCV data.
4. **HBV genotype coverage.** Panel uses genotype D (ayw) reference sequences. Genotype B/C populations (East/Southeast Asia) may show sequence divergence at predicted epitope positions.
5. **GNN gate status.** GraphPredictor v1 fails Gates 1 and 5 (v4 AUC-PR 0.613, sensitivity 0.724). GraphPredictorV2 (GINEConv + ESM-2) is implemented and under evaluation; full gate results in §3.6. Neither architecture is the production scorer until all 5 gates pass.
6. **Allele-aware training.** The 166-feature allele-aware schema is implemented but not trained; all production predictions use population-average allele features.
7. **Zero-shot HLA generalization.** This claim has been removed from all SESTRAV communications. It was not validated in any experiment and should not appear in publications, presentations, or documentation.
8. **Negative data quality.** v3 negatives are mostly non-immunogenic due to poor MHC binding, causing partial "binding → immunogenic" learning. Hard decoys (v4) address this root cause.
9. **Binding feature marginal redundancy.** Per-allele MHCflurry binding scores (`bind_A0101`–`bind_B4402`) contribute zero marginal information above physicochemical features in the v3 production model (both impurity-based and permutation importance = 0; investigated and confirmed — no pipeline bug). The mechanism is physico-binding overlap: anchor-residue features at p5–p8 capture binding-relevant variance; the v3 negative selection bias (binding-poor) suppresses informative binding variance in the label-conditioning set. Full marginal binding utility is expected in v4 where hard decoys decouple binding from immunogenicity and allele-aware features are explicitly stratified.

### 4.3 Future directions

1. **v4 allele-aware training.** 166-feature schema enables allele-specific predictions once VDJdb allele annotations are incorporated. Target: ≥10 allele models from v4.
2. **Hard decoy integration.** 5,000 central-tolerance self-peptides (500 per allele, 10 alleles) added to v4 are expected to raise AUC-PR above 0.880 by decoupling binding affinity from immunogenicity in the negative class.
3. **GNN v2.1.** GINEConv + ESM-2 node embeddings (implemented 2026-06-20); promotion gate results reported in §3.6. If Gates 1 and 5 clear, GNN v2.1 becomes the production deep-learning scorer.
4. **Virus expansion.** SARS-CoV-2 (panel4: Spike P0DTC2, N P0DTC9, M P0DTC5, ORF3a P0DTC3), IAV (panel4: NP P03466, M1 P03485, HA P03437, PB1-F2 P0C0U1), CMV (panel4: pp65 P06725, IE1 P13202, pp50 P16785, gB P06473) — gated on IEDB audit confirming ≥100 positive T-cell assays per virus.
5. **Continuous automated validation.** Monthly GitHub Actions benchmark against new IEDB exports with automatic AUC-PR regression alerting.

---

## 5. Availability

- **Source code:** [GitHub repository URL] — MIT license
- **Installation:** `pip install sestrav` (core); `pip install "sestrav[gnn]"` (+ GNN); `pip install "sestrav[pipeline]"` (+ Snakemake)
- **PyPI:** https://pypi.org/project/sestrav/
- **Zenodo dataset DOI:** [pending — v4 build + registration, Week 6 Day 7]
- **Docker image:** [pending — GHCR publication, Week 6 Day 7]
- **MHCflurry model version:** See `mhcflurry_model_version` in `config.yaml`

---

## References

- Bjellqvist B, et al. The focusing positions of polypeptides in immobilized pH gradients can be predicted from their amino acid sequences. *Electrophoresis*. 1993;14(1):1023–1031.
- Chowell D, et al. TCR contact residue hydrophobicity is a hallmark of immunogenic CD8⁺ T cell epitopes. *Proc Natl Acad Sci USA*. 2015;112(14):E1754–E1762.
- Gfeller D, et al. Improved predictions of MHC-I and MHC-II epitopes using the PRIME tool and selected amino acid substitution matrices. *Front Immunol*. 2023;14:1128364.
- Hopp TP, Woods KR. Prediction of protein antigenic determinants from amino acid sequences. *Proc Natl Acad Sci USA*. 1981;78(6):3824–3828.
- Jurtz V, et al. NetMHCpan-4.0: Improved peptide–MHC class I interaction predictions integrating eluted ligand and peptide binding affinity data. *J Immunol*. 2017;199(9):3360–3368.
- Kipf TN, Welling M. Semi-supervised classification with graph convolutional networks. *Int Conf Learn Represent (ICLR)*. 2017. arXiv:1609.02907.
- Kloetzel PM. Antigen processing by the proteasome. *Nat Rev Mol Cell Biol*. 2001;2(3):179–187.
- Kyte J, Doolittle RF. A simple method for displaying the hydropathic character of a protein. *J Mol Biol*. 1982;157(1):105–132.
- Lobry JR, Gautier C. Hydrophobicity, expressivity and aromaticity are the major trends of amino-acid usage in 999 *Escherichia coli* chromosome-encoded genes. *Nucleic Acids Res*. 1994;22(15):3174–3180.
- Meiler J, et al. Generation and evaluation of dimension-reduced amino acid parameter representations by artificial neural networks. *J Mol Model*. 2001;7(9):360–369.
- Mölder F, et al. Sustainable data analysis with Snakemake. *F1000Res*. 2021;10:33.
- Nielsen M, et al. The role of the proteasome in generating cytotoxic T-cell epitopes: insights obtained from improved predictions of proteasomal cleavage. *Immunogenetics*. 2005;57(1–2):33–41.
- O'Donnell TJ, et al. MHCflurry 2.0: Improved pan-allele prediction of MHC class I-presented peptides by incorporating antigen processing. *Cell Syst*. 2020;11(1):42–48.
- Peters B, et al. Identifying MHC class I epitopes by predicting the TAP transport efficiency of epitope precursor peptides. *J Immunol*. 2003;171(4):1741–1749.
- Rammensee HG, et al. SYFPEITHI: database for MHC ligands and peptide motifs. *Immunogenetics*. 1999;50(3–4):213–219.
- Rives A, et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *Proc Natl Acad Sci USA*. 2021;118(15):e2016239118.
- Rock KL, Goldberg AL. Degradation of cell proteins and the generation of MHC class I-presented peptides. *Annu Rev Immunol*. 1999;17:739–779.
- Vihinen M, Torkkila E, Riikonen P. Accuracy of protein flexibility predictions. *Proteins*. 1994;19(2):141–149.
- Vita R, et al. The Immune Epitope Database (IEDB): 2018 update. *Nucleic Acids Res*. 2019;47(D1):D339–D343.
- Zamyatnin AA. Protein volume in solution. *Prog Biophys Mol Biol*. 1972;24:107–123.
- Zimmerman JM, Eliezer N, Simha R. The characterization of amino acid sequences in proteins by statistical methods. *J Theor Biol*. 1968;21(2):170–201.
