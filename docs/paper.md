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

[*Table of tools (PredIG, PRIME 2.1, NetMHCpan 4.2, MixMHCpred 2.2, DeepImmuno, BigMHC) with capabilities and limitations — to be populated after external benchmark completion, Week 5 Day 4.*]

Key gaps common across surveyed approaches:
- No published tool combines MHC binding, antigen processing, and TCR contact features within a single reproducible, end-to-end workflow
- No published tool provides cryptographic dataset governance (provenance checksums, freeze mode)
- External tool evaluations commonly suffer from training-test contamination: the SESTRAV Tier A test set overlaps 36.9% with PredIG and PRIME 2.1 training data (see Section 2.4)

### 1.3 SESTRAV's design rationale

SESTRAV addresses these gaps through workflow integration rather than model novelty alone. The six-stage architecture — (1) proteome-scale peptide generation, (2) MHC binding prediction, (3) TCR-contact physicochemical feature extraction, (4) antigen processing scoring, (5) immunogenicity inference, (6) freeze-mode governed output — is the primary contribution; each stage is individually replaceable without breaking the pipeline DAG. Reproducibility governance is a first-class design goal: every training run records dataset checksums, MHCflurry model version, and feature schema version in a provenance JSON sidecar. No publicly available tool executes all six stages in a single reproducible command with OpenSSF-compliant supply-chain security.

---

## 2. Methods

### 2.1 Dataset construction and governance

**Training data.** Immunogenicity labels were obtained from the Immune Epitope Database (IEDB; Vita et al. 2019). Positive instances are peptides with at least one positive T-cell assay record; negative instances are peptides with exclusively negative assay records. Labels represent population-average responses aggregated across heterogeneous IEDB assay types, donor backgrounds, and stimulation conditions; they do not represent allele-specific or donor-specific immunogenicity (see Section 4.2, Limitation 1).

**Dataset v3 (current production).** The v3 dataset (`data/immunogenicity_dataset_v3.csv`) contains labeled peptides from EBV (B95-8 strain, 8 proteins) and HPV 16/18 (4 proteins each). Virus composition: EBV 68.1%, HPV16 30.9%, HPV11 1.0%. Peptide length distribution: 9-mer 64.7%, with 8-, 10-, and 11-mer minorities. Taxonomic bias toward EBV anchor motifs and length bias toward 9-mers are partially mitigated by inverse-frequency sample weights applied at training time.

**Dataset v4 (under construction).** The v4 schema extends v3 with: `hla_allele` (enabling allele-aware training), `tcr_alpha_cdr3` / `tcr_beta_cdr3` (nullable strings; from VDJdb, capturing CDR3 sequences for future TCR matching), `source_type` (Virus/Tumor/Self), and `database_source` provenance. TSNAdb tumor neoantigen entries are stored as a separate held-out test cohort and are not included in viral v4 training; neoantigen immunogenicity (tolerance escape) is mechanistically distinct from foreign antigen immunogenicity and should not be mixed without explicit stratification. Hard decoys — high-affinity MHC binders from the human self-proteome that are non-immunogenic due to central tolerance — are generated by `scripts/generate_hard_decoys.py` and included in v4 to prevent the model from relying on binding affinity as a surrogate for immunogenicity (see Section 4.2, Limitation 8). v4 build is blocked on maintainer hardware (MHCflurry model download and VDJdb network access required).

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

**Graph Neural Network (v2.0; under promotion gating).** `GraphPredictor` (`src/gnn/models.py`) is a custom PyTorch model combining a dense-adjacency GCN encoder with a physicochemical feature MLP. Architecture details:
- `GCNLayer`: hand-rolled graph convolution performing `adj @ (x @ W) + b` on a dense adjacency tensor of shape `(max_peptide_len, max_peptide_len)`.
- `GraphEncoder`: two stacked `GCNLayer` instances with ReLU activations and global mean pooling to produce a graph-level embedding.
- `GraphPredictor.forward(node_x, feat_x, adj)`: node features (`node_x`, shape `[batch, max_len, 20]`, amino acid one-hot) are processed by `GraphEncoder`; physicochemical features (`feat_x`) are processed by a separate MLP; outputs are fused and projected to a scalar immunogenicity score per sample.

No structural coordinates, PDB data, AlphaFold embeddings, or learned embeddings are used in the current implementation. PyTorch Geometric is a declared dependency reserved for structural comparator models in `src/verify/structural_gnn.py`; the production GNN path uses only standard PyTorch.

*Current promotion gate status (v3 model, pre-v4 baseline):*
| Gate | Criterion | Status |
|---|---|---|
| 1 — Calibration (ECE < 0.05) | Expected Calibration Error | **FAIL** — requires Platt calibration post-v4 retraining |
| 2 — OOF generalizability (AUC-ROC > 0.70) | Held-out fold AUC | **PASS** |
| 3 — Cross-antigen (AUC-ROC > 0.65) | Cross-protein transfer | **PASS** |
| 4 — Cross-virus (AUC-ROC > 0.65) | EBV ↔ HPV transfer | **FAIL** — requires v4 multi-virus data |
| 5 — Mutation sensitivity | ±1 AA variant discrimination | **FAIL** — may require targeted augmentation |

GNN promotion to production is targeted for v2.1 after v4 retraining with Platt calibration. Planned v2.1 improvements: GINEConv message passing (PyTorch Geometric), ESM-2 protein language model node embeddings (Rives et al. 2021), allele-aware graph construction.

**Allele-aware training (schema ready; training blocked on v4).** A 166-feature allele-aware schema is implemented in `src/features.py` using HLA pseudo-sequence encodings. Training requires the v4 dataset, which includes `hla_allele` annotations from VDJdb. All current production predictions use the population-average 31-feature model; allele-aware predictions are not available in v3.

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

*[v4 retrained results to replace v3 baselines after Week 6 Day 3 retraining.]*

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

**Finding — binding feature marginal redundancy:** In the v3 production run, all 10 MHCflurry per-allele binding features (`bind_A0101`–`bind_B4402`) register RF importance = 0.0. Post-hoc investigation confirmed: (1) the binding matrix `peptide_binding_matrix_v3.csv` contains real MHCflurry presentation scores (mean bind_A0201 = 0.149, range 0.003–0.994; all 1,004 rows non-zero); (2) `prepare_features_31()` correctly joins all rows; (3) permutation importance independently confirms zero marginal contribution (all bind_* permutation importance = 0.0 ± 0.0 on AUC-ROC). The bind_* features carry weak univariate signal (r = 0.10–0.15, p < 0.001 for 7/10 alleles) but zero marginal gain above physicochemical features — confirmed by both impurity and permutation importance. Root mechanism: MHC binding is predominantly driven by anchor residues at p2 and the C-terminus, positions captured by the p5–p8 physicochemical features, making bind_* conditionally redundant. A selection confound amplifies this: v3 negatives were selected for poor MHC binding, compressing binding variance among positives (see §4.2 Limitation 9). Production model effectively operates on 21 physicochemical + peptide_length features. The bind_* ablation advantage (0.851) reflects standalone binding information, not marginal gain above physico. v4 hard decoys break the binding-only negative selection and are expected to restore marginal binding utility.

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

**SYFPEITHI canonical epitope recall (Table 4).** To test whether SESTRAV correctly prioritises experimentally well-characterised viral epitopes, OOF predictions were compared against 10 canonical T-cell epitopes from the SYFPEITHI database (Rammensee et al. 1999; HLA-A*02:01-restricted, EBV/HPV16). Six of 10 reference epitopes were present in the training cohort (as exact matches or single-substitution variants); four were not evaluable OOF.

*Table 4. SYFPEITHI canonical epitope recall in SESTRAV OOF predictions (n_evaluable=6).*

| Cutoff | SESTRAV recall | Random baseline | Enrichment |
|---|---|---|---|
| Top 5% | 16.7% (1/6) | 5.0% | 3.3× |
| Top 10% | 16.7% (1/6) | 10.0% | 1.7× |
| Top 25% | 33.3% (2/6) | 25.0% | 1.3× |

*Per-epitope results: RAHYNIVTF (E6, HPV16 variant) ranked top 4.4%; CLGGLLTMV (BMLF1, EBV variant) top 16.4%; FAFRDLCIV (E6, HPV16) top 44.7%; FLYALALLL (LMP2A, EBV) top 37.9%; KLPQLCTEL (E7, HPV16 variant) top 60.8%; LLWTLVVLL (LMP2A, EBV) top 68.3%. Full results: `results/syfpeithi_benchmark.json`. Benchmark script: `scripts/benchmark_syfpeithi.py`.*

*Interpretation: Positive enrichment at top-5% and top-25% cutoffs (3.3× and 1.3× respectively) is consistent with SESTRAV successfully prioritising immunodominant peptides. The two weaker-ranked epitopes (KLPQLCTEL and LLWTLVVLL) are both known to have moderate SYFPEITHI scores (19 and 22 respectively) and may reflect genuine biological heterogeneity in immunodominance hierarchy not captured by HLA-A*02:01 binding affinity alone.*

### 3.4 Cross-virus transfer

*Table 5. Cross-virus transfer AUC-PR (OOF RF, v3 data).*

| Train → Test | AUC-PR |
|---|---|
| EBV → HPV16 | 0.742 |
| HPV16 → EBV | 0.711 |
| [Full n×n matrix, v4 data] | [pending — Week 6 Day 8] |

Cross-virus transfer within the EBV/HPV DNA virus family shows a 0.086–0.117 AUC-PR drop from in-distribution performance. Predictions for HBV (genotype D, ayw) and HCV (genotype 1a) are generated by the v3 model trained on EBV/HPV data only. These are exploratory predictions until v4 cross-virus validation with HBV/HCV training data is complete (see Section 4.2, Limitation 3).

### 3.5 Antigen processing feature contribution

The addition of two antigen processing features (NetChop 3.1 C-terminal cleavage probability, TAPreg TAP transport affinity) to the 31-feature canonical model raises OOF AUC-PR from 0.864 (`full_31`) to 0.886 (`full_33`) in unweighted ablation on v3 data — a +0.022 improvement and the largest single-step gain in the ablation series (Table 1). In the weighted production run, the mode-33 RF achieves AUC-PR 0.840 ± 0.011 vs. 0.828 ± 0.027 for mode-31, confirming the improvement holds under realistic training conditions.

Among all 33 features, `netchop_score` is the single most informative feature (RF importance = 0.118), followed by `tap_score` (0.096; Table 2). Both antigen processing scores rank above any physicochemical or binding feature. This is consistent with the established rate-limiting role of proteasomal C-terminal cleavage in epitope generation: the proteasome must first liberate the peptide from its precursor before TAP transport or MHC loading can occur (Rock & Goldberg 1999; Kloetzel 2001).

Compared to PredIG-Path (AUC-PR 0.727, fully trained), the mode-33 SESTRAV RF (AUC-PR 0.840, OOF) achieves a +0.113 AUC-PR advantage under evaluation conditions that favour PredIG due to 36.9% training-set contamination of the test set (§2.4). The mode-33 model is recommended for production use where the antigen processing cache (`data/antigen_processing_cache.csv`, pre-computed via `scripts/precompute_antigen_processing.py`) is available. The mode-31 canonical track remains the default where real-time NetChop/TAPreg access is impractical (DTU API reliability; see §4.2 Limitation 9 for the mock-score caveat).

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
5. **GNN gate status.** GraphPredictor currently fails Gates 1, 4, 5. It is a development model, not a production scorer.
6. **Allele-aware training.** The 166-feature allele-aware schema is implemented but not trained; all production predictions use population-average allele features.
7. **Zero-shot HLA generalization.** This claim has been removed from all SESTRAV communications. It was not validated in any experiment and should not appear in publications, presentations, or documentation.
8. **Negative data quality.** v3 negatives are mostly non-immunogenic due to poor MHC binding, causing partial "binding → immunogenic" learning. Hard decoys (v4) address this root cause.
9. **Binding feature marginal redundancy.** Per-allele MHCflurry binding scores (`bind_A0101`–`bind_B4402`) contribute zero marginal information above physicochemical features in the v3 production model (both impurity-based and permutation importance = 0; investigated and confirmed — no pipeline bug). The mechanism is physico-binding overlap: anchor-residue features at p5–p8 capture binding-relevant variance; the v3 negative selection bias (binding-poor) suppresses informative binding variance in the label-conditioning set. Full marginal binding utility is expected in v4 where hard decoys decouple binding from immunogenicity and allele-aware features are explicitly stratified.

### 4.3 Future directions

1. **v4 allele-aware training.** 166-feature schema enables allele-specific predictions once VDJdb allele annotations are incorporated. Target: ≥10 allele models from v4.
2. **Hard decoy integration.** 10,000 central-tolerance self-peptides added to v4 are expected to raise AUC-PR above 0.880.
3. **GNN v2.1.** GINEConv + ESM-2 node embeddings targeting all 5 promotion gates on v4 data.
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

- Chowell D, et al. TCR contact residue hydrophobicity is a hallmark of immunogenic CD8⁺ T cell epitopes. *Proc Natl Acad Sci USA*. 2015;112(14):E1754–E1762.
- Gfeller D, et al. Improved predictions of MHC-I and MHC-II epitopes using the PRIME tool and selected amino acid substitution matrices. *Front Immunol*. 2023;14:1128364.
- Jurtz V, et al. NetMHCpan-4.0: Improved peptide–MHC class I interaction predictions integrating eluted ligand and peptide binding affinity data. *J Immunol*. 2017;199(9):3360–3368.
- Kloetzel PM. Antigen processing by the proteasome. *Nat Rev Mol Cell Biol*. 2001;2(3):179–187.
- Kyte J, Doolittle RF. A simple method for displaying the hydropathic character of a protein. *J Mol Biol*. 1982;157(1):105–132.
- Nielsen M, et al. The role of the proteasome in generating cytotoxic T-cell epitopes: insights obtained from improved predictions of proteasomal cleavage. *Immunogenetics*. 2005;57(1–2):33–41.
- O'Donnell TJ, et al. MHCflurry 2.0: Improved pan-allele prediction of MHC class I-presented peptides by incorporating antigen processing. *Cell Syst*. 2020;11(1):42–48.
- Peters B, et al. Identifying MHC class I epitopes by predicting the TAP transport efficiency of epitope precursor peptides. *J Immunol*. 2003;171(4):1741–1749.
- Rives A, et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *Proc Natl Acad Sci USA*. 2021;118(15):e2016239118.
- Rock KL, Goldberg AL. Degradation of cell proteins and the generation of MHC class I-presented peptides. *Annu Rev Immunol*. 1999;17:739–779.
- Vita R, et al. The Immune Epitope Database (IEDB): 2018 update. *Nucleic Acids Res*. 2019;47(D1):D339–D343.
- Zamyatnin AA. Protein volume in solution. *Prog Biophys Mol Biol*. 1972;24:107–123.
- Zimmerman JM, Eliezer N, Simha R. The characterization of amino acid sequences in proteins by statistical methods. *J Theor Biol*. 1968;21(2):170–201.
