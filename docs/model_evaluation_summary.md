# SESTRAV Model Evaluation Summary

## v5 Canonical Track: 31-Feature RF (Trained 2026-07-04)

> **Status:** Current production model. Dataset: v5, 35,597 active rows (51,185 total),
> **5-fold peptide-grouped OOF (re-baselined 2026-08-10, closing `docs/claims_register.md` D15).**
> Per-virus within-CV (below) regenerated 2026-08-10 under the grouped splitter. Two
> retractions are now folded in: the fragile pooled "same-pathogen AUC-ROC 0.9368" headline
> from the e6aafe2 build (2026-07-04) was RETIRED 2026-07-11 as decoy-inflated, and the
> ungrouped figures that replaced it (per-virus mean 0.751, pooled 0.8312, pooled-honest
> 0.712) were retracted 2026-08-10 as peptide-leakage-inflated. The canonical
> same-pathogen discrimination metric is the peptide-grouped per-virus within-CV table
> (mean AUC-ROC **0.658**; `results/per_virus_eval_v5_mode31.csv`).
> Model: `models/rf_31feature_integrated.joblib` (retrained)
> OOF predictions: `models/rf_oof_predictions.csv`, `models/rf_oof_predictions_mode31.csv`

| Evaluation context | AUC-PR | AUC-ROC | Notes |
|---|---|---|---|
| Within-virus (same-pathogen discrimination) | see per-virus table | **0.658** (mean) [^retracted] | Canonical metric: per-virus within-CV mean over 9 viruses, peptide-grouped (`results/per_virus_eval_v5_mode31.csv`). Prior ungrouped 0.751 retracted. |
| Pooled cross-validation (whole v5 corpus) | **0.6055** | **0.8137** | Peptide-grouped, re-baselined 2026-08-10 (`models/v5/training_results_mode31.csv`). **Two superseded predecessors:** this row once read "Self-proteome Gate 1, AUC-PR 0.8897, Gate 1 threshold protocol" - wrong on two counts (0.8897 is the pooled `auc_pr` of an earlier 2026-06-26 v5 build, and no self-proteome-vs-viral evaluation artifact exists here; "Gate 1" is a GNN promotion threshold, `src/verify/promote_gnn.py`, unrelated to this RF metric). It then read 0.8312, which was itself peptide-leakage-inflated (D15) and is now retracted in favour of the grouped 0.6055. The same ledger also carries `rf_cv_mean_no_vaccinia` (AUC-PR 0.7328 / AUC-ROC 0.6702), an OOF re-slice excluding the vaccinia bloc from validation - not a refit on a vaccinia-free corpus. |

[^retracted]: Two successive retractions apply to this figure, and they are different defects.
**(1) Decoy inflation (2026-07-11):** the previously reported pooled same-pathogen AUC-ROC
0.9368 only reproduces when easy negatives - the synthetic allele-matched non-binders and the
out-of-panel vaccinia bloc (which is assay-confirmed, not synthetic - D19) -
are mixed in as if they were same-pathogen negatives; RETRACTED. **(2) Peptide leakage
(2026-08-10, D15):** the decoy-corrected figures that replaced it were computed under a
splitter that stratified but did not group by peptide, and are themselves retracted - pooled
honest same-pathogen ROC 0.712 -> **0.602**, per-virus within-CV mean 0.751 -> **0.658**.
The current figures are decoy-corrected AND peptide-grouped. The pooled same-pathogen AUC-PR
(now 0.8711) remains a base-rate artifact (8003 positive vs 1851 negative, about 81% positive)
and is NOT reported as a headline. The canonical, reproducible same-pathogen metric is the
per-virus within-CV table below (mean AUC-ROC 0.658), from `scripts/evaluate_per_virus.py`.
See `docs/claims_register.md` D15 (remediated) and D12 (superseded-in-part by D15).

**Per-virus within-CV results (Amendment 6 thresholds; regenerated 2026-08-10 under the peptide-grouped splitter, on the 35,597-row v5 dataset):**

| Virus | AUC-ROC | Threshold | Status |
|---|---|---|---|
| HPV | 0.482 | >= 0.58 | FAIL |
| EBV | 0.711 | >= 0.57 | PASS (post B*27 conflict quarantine) |

> **Note:** HPV within-CV fell from 0.598 (e6aafe2 snapshot) to 0.561 on the ungrouped
> 35,597-row build, and to **0.482** under the peptide-grouped splitter (2026-08-10) - further
> below the 0.58 Amendment-6 threshold, and consistent with the manuscript's characterization
> of HPV as an active generalization failure. EBV rose from 0.667 to 0.790 ungrouped, and sits
> at **0.711** peptide-grouped, still clearing its 0.57 threshold. Prior ungrouped values
> (HPV 0.561, EBV 0.790) are retracted per `docs/claims_register.md` D15.

> **Note:** v3/v4 sections below are historical. The v5 31-feature RF is the canonical
> production scorer. All public-facing comparisons should use v5 figures.

---

## v2 Extended Track: 33-Feature Integrated (Trained 2026-06-18)

> **Status:** ~~Best current v3 model.~~ **"Best current v3 model" RETRACTED 2026-08-10 (`docs/claims_register.md` D18)** - the margin was measured on mock antigen-processing features; `mode_31` remains the canonical production track. Dataset: v3 n=1004 peptides, 76.6% positive, 5-fold OOF.
> Sample weights applied (virus_weight=0.5, length_weight=0.5).
> Models: `models/rf_33feature_integrated.joblib`, `models/xgb_33feature_integrated.joblib`
> Features: 20 physico (p4-p8) + 10 binding (MHCflurry) + peptide_length + netchop_score + tap_score

| Metric | RF (mean ± std) | XGBoost (mean ± std) | Notes |
|--------|-----------------|----------------------|-------|
| **AUC-PR** | **0.8399 ± 0.011** | 0.8235 ± 0.012 | Primary metric |
| **AUC-ROC** | 0.6728 ± 0.023 | 0.6393 ± 0.029 | |
| **ISSR@10** | **0.9158 ± 0.042** | 0.8842 ± 0.052 | Fraction of the top-10% ranked peptides that are true positives (precision within the top decile) |
| **ISSR@25** | 0.9102 ± 0.038 | 0.8816 ± 0.024 | Fraction of the top-25% ranked peptides that are true positives (precision within the top quartile) |

> **PROVENANCE SPLIT (added 2026-08-15, S1). This table is a mirror of the one in
> `docs/model_cards/rf_33feature_integrated.md`, and its eight cells do NOT share a source.** That
> card carries the full derivation; the finding in brief, re-derived against `models/training_results.csv`
> at `6a51995` (the mode-33 v3 run, in the tree before either copy of this table was written):
> **the four AUC cells reproduce exactly** and are sound, needing only a citation re-anchor to that
> commit rather than to the bare tracked path (the live file is now a v5 mode-31 run and binds none
> of them). **The four ISSR cells reproduce from nothing.** The artifact holds RF 0.8211 / 0.8694 and
> XGB 0.8000 / 0.8286; the printed figures also assert ISSR@10 > ISSR@25 for both models where the
> artifact and the sibling v3 table both have the opposite ordering. An all-history blob sweep for
> the printed values returns zero hits, so their origin is unidentifiable. The artifact values are
> given so a reader can verify, **not as replacements** - substituting them would invent a
> provenance. Treat all four ISSR cells as withdrawn pending re-derivation.

> **Unweighted ablation AUC-PR (feature_mode=33): 0.8863 ± 0.019** - highest single-number
> unweighted result for SESTRAV on v3, **but measured on mock antigen-processing features (D18) and
> therefore not a real-predictor result.** Improvement over feature_mode=31 unweighted (0.864):
> +0.022 AUC-PR - likewise a mock-feature gain.
> Top feature: netchop_score (RF importance=0.118). **The clause "confirming independent proteasomal processing signal" is RETRACTED (2026-08-10, `docs/claims_register.md` D18):** `netchop_score` is a locally generated mock value, not NetChop 3.1 output, and its generator already assumes the cleavage preference the clause claimed to confirm.

---

## v2 Canonical Track: 31-Feature Integrated (Trained 2026-06-18)

> **Status:** Models trained and saved. Dataset: v3, 1004 peptides, 76.6% positive, 5-fold stratified OOF.
> Sample weights applied (virus_weight=0.5, length_weight=0.5) to correct EBV/HPV16 and 9-mer skew.
> Models: `models/rf_31feature_integrated.joblib`, `models/xgb_31feature_integrated.joblib`

| Metric | RF (mean ± std) | XGBoost (mean ± std) | Notes |
|--------|-----------------|----------------------|-------|
| **AUC-PR** | **0.8276 ± 0.027** | 0.8205 ± 0.010 | Primary metric (class imbalance) |
| **AUC-ROC** | 0.6431 ± 0.039 | 0.6062 ± 0.037 | |
| **ISSR@10** | 0.8105 ± 0.079 | 0.8105 ± 0.042 | Fraction of the top-10% ranked peptides that are true positives (precision within the top decile) |
| **ISSR@25** | 0.8367 ± 0.022 | 0.8408 ± 0.015 | Fraction of the top-25% ranked peptides that are true positives (precision within the top quartile) |

> **Note on ablation estimate:** An early unweighted ablation projected `full_31` AUC-PR 0.864.
> The actual result with sample weights is 0.8276.
> **Correction (2026-08-09):** this note previously called 0.8276 "consistent with the frozen
> v2.0.0 30-feature result (0.828)" and treated the two as a reconciliation. They are not the
> same measurement: 0.8276 is a weighted 31-feature cross-validation mean over the full n=1,004
> v3 corpus, while 0.828 is an unweighted 30-feature, 200-tree measurement over the n=704 Tier A
> field benchmark (`docs/claims_register.md` D16). The two values are 0.0002 apart by
> coincidence; treating that coincidence as agreement is the root cause of 0.828 being mislabeled
> as a `full_31`/`mode_31` result elsewhere in this repository's history. The 31-feature model's
> own weighted result on this table (0.8276) should be compared against the 30-feature weighted
> result (0.810 ± 0.025) directly below, not against 0.828.
---

## v2 Canonical Track: 30-Feature Integrated (720 peptides, 2.36:1 class ratio)

The canonical evaluation track uses 20 physicochemical features (p4-p8 x 4 properties) plus 10 per-allele MHC binding features. All results are 5-fold stratified cross-validation with gold-standard epitopes held out.

Release-scope boundary: The optional ANN/GNN benchmark track is supplementary and not part of the canonical publish gate. The canonical publish gate is based on the RF-configured Stage 1-4 workflow and frozen validation artifacts.
ANN/GNN values are sourced from the ANN/GNN benchmark study and mirrored in SESTRAV-Dev docs.

### Cross-Validation Results (5-fold stratified)

> **THE ANN COLUMN IS RETRACTED AS UNBOUND, 2026-08-17. Do not cite it.** Every ANN figure in
> the table below was sourced to `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/bootstrap_metric_cis.csv`,
> a path **absent from this repository and from any local workspace** - an external course
> deliverable that was never vendored - so none of them can be reproduced, checked, or ever bound
> to provenance. They are **withdrawn, not superseded**: no replacement ANN figure is quoted here,
> deliberately. This mirrors `docs/nn_gnn_optional_module_guide.md`, which retracted the same
> numbers, and `README.md`, which states that no accuracy figure is quoted for this architecture.
>
> **`models/ann_cv_summary.csv` is NOT the replacement.** It reads AUC-PR 0.7820 +/- 0.0239, but
> it measures the **legacy 64-32 dropout-0.3** network over the 704 peptides left after the 16
> gold-standard epitopes are held out - a different architecture on a different pool. Citing it
> under the 256-128-64 label would trade an unbound number for a mis-attributed one.

| Metric | RF (mean ± std) | XGBoost (mean ± std) | ANN 256-128-64 (mean ± std) |
|--------|-----------------|---------------------|---------------------------|
| **AUC-PR** | 0.810 ± 0.025 | 0.805 ± 0.028 | ~~0.825 ± 0.025~~ retracted, unbound |
| **AUC-ROC** | 0.670 ± 0.042 | 0.665 ± 0.045 | ~~0.670 ± 0.040~~ retracted, unbound |
| **ISSR@10** | 0.870 ± 0.050 | 0.865 ± 0.055 | ~~0.880 ± 0.045~~ retracted, unbound |
| **ISSR@25** | 0.920 ± 0.035 | 0.915 ± 0.038 | ~~0.930 ± 0.030~~ retracted, unbound |

**No best-performer claim is made for the 30-feature track.** This line previously read *"Best
benchmark performer (30-feature track): ANN (256-128-64 ReLU dropout 0.2) - highest AUC-PR and
ISSR@10 in this comparison table."* That superlative rested entirely on the retracted ANN column
above, so it is withdrawn with it rather than reassigned to another model: the RF and XGBoost
values in this table are themselves described below as representative rather than certified, and
ranking two representative columns against each other would assert a result none of them supports.

> **Note:** AUC-PR values shown are representative of the 30-feature track. Exact values depend on the training run seed and dataset split. Run `src/train_classifier.py` and `src/ann_benchmark.py` locally to reproduce; both require `--model-dir`, so point them at a scratch directory such as `models/local` rather than at the published artifacts in `models/`.

> **Provenance note:** The canonical optional ANN/GNN evidence source is documented in
> `docs/nn_gnn_optional_module_guide.md`, which defines the optional ANN/GNN benchmark
> track and its boundary against the canonical release gate.

### ANN Architecture Search (14 Configurations)

The architecture search evaluates depth, width, activation function, and dropout rate. The best configuration (256-128-64 ReLU dropout 0.2) was identified from the systematic architecture search.

Saved to `ann_architecture_search.csv` inside the `--model-dir` the run was
pointed at (generated locally via the `--search` flag). `--model-dir` is
required and has no default, so a search run writes wherever it is told rather
than into `models/`. Project 2 run metadata and benchmark lineage are
documented in `docs/nn_gnn_optional_module_guide.md`.

### GNN Benchmark Results (Exploratory)

| Model | AUC-PR (mean ± std) | AUC-ROC (mean ± std) |
|-------|---------------------|---------------------|
| GCN (2-layer) | 0.778 ± 0.030 | 0.614 ± 0.050 |
| GAT (2-layer, 4-head) | **0.796 ± 0.028** | **0.637 ± 0.045** |
| Bipartite Peptide-Allele | 0.789 ± 0.032 | 0.612 ± 0.055 |

GNNs underperform tabular models on this dataset but capture structural inter-residue patterns that fixed-position features cannot represent. Included for representation space characterization.

### Exact Project 2 Optional Benchmark Values

**There is no such source, and this pointer contradicted this document's own retraction until
2026-08-31.** It read "For exact unrounded synced values, see:" and then listed three files
under `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/` - including
`bootstrap_metric_cis.csv`, the very path the ANN retraction above calls "absent from this
repository and from any local workspace". The same document therefore retracted a source and
directed readers to it, roughly fifty lines apart.

The whole `CMB 523 ...` directory is unvendored: `git ls-files` returns nothing matching
`CMB`, and it is not on disk. No exact unrounded values are recoverable for the ANN or GNN
figures. **`models/gnn_sequence_benchmark.csv` and `models/gnn_bipartite_benchmark.csv` are
tracked and share the two filenames, but hold DIFFERENT values and are NOT the source** - see
the measured comparison in `docs/nn_gnn_optional_module_guide.md`, which retracts those
figures rather than substituting them.

For the GNN track's reproducible status see `docs/nn_gnn_optional_module_guide.md`.

### Ablation Study Results (Feature Group Contributions)

| Feature Set | Features | AUC-PR (mean) | AUC-ROC (mean) |
|-------------|----------|---------------|----------------|
| `physico_20` | 20 | 0.772 | 0.577 |
| `binding_10` | 10 | 0.851 | 0.727 |
| `sestrav_21` | 21 | 0.784 | 0.622 |
| `combined_30` | 30 | 0.825 | 0.670 |
| `full_31` | 31 | 0.864 | 0.743 |
| **`full_33`** | **33** | **0.886** | **0.751** |

Antigen processing features (netchop_score, tap_score) add +0.022 AUC-PR above `full_31`, confirming
independent proteasomal and TAP transport signal. The `full_33` model is the best v3 result and the
recommended production track where antigen processing cache is available.

> **RETRACTED (2026-08-10, `docs/claims_register.md` D18).** All three claims in the paragraph above
> are withdrawn. `netchop_score` and `tap_score` are locally generated MOCK values, not NetChop 3.1
> or TAPreg output, so the +0.022 gain "confirms" no proteasomal or TAP transport signal - the
> generator already encodes those preferences, making the inference circular. The "best v3 result"
> and "recommended production track" designations are withdrawn with it: under the certified v5
> peptide-grouped splitter, mode 33 exceeds mode 31 by **+0.0027 AUC-PR** (0.6085 vs 0.6058), and
> **`mode_31` remains the canonical production track.**

> **Do not confuse `full_33`'s AUC-ROC 0.751 in the table above with the retracted v5 per-virus
> within-CV mean of 0.751.** They are numerically identical by coincidence and are entirely
> different quantities: this one is a v3 unweighted feature-ablation AUC-ROC over n=1,004, and
> is NOT affected by the D15 peptide-grouping remediation (which re-baselined the v5 corpus
> only). The v5 per-virus mean is now 0.658. This table is v3-era throughout and was not
> re-measured under the peptide-grouped splitter.

---

## Legacy Benchmark Line: 21-Feature Sequence-Only

This section documents the legacy 21-feature benchmark retained for reproducibility and historical comparison. These results use the v1 dataset (912 training peptides).

### Cross-Validation Results (5-fold stratified, 912 training peptides)

| Metric | RF (mean ± std) | XGBoost (mean ± std) | ANN/MLP (mean ± std) |
|--------|-----------------|---------------------|---------------------|
| **AUC-ROC** | **0.726 ± 0.042** | 0.685 ± 0.053 | 0.676 ± 0.055 |
| **AUC-PR** | **0.919 ± 0.021** | 0.912 ± 0.019 | 0.911 ± 0.029 |
| **ISSR@10** | 0.911 ± 0.057 | 0.911 ± 0.027 | **0.933 ± 0.082** |
| **ISSR@25** | **0.947 ± 0.036** | 0.938 ± 0.022 | 0.929 ± 0.038 |

**Best model (legacy benchmark line): RandomForest** - highest AUC-ROC, AUC-PR, and ISSR@25.
This document should be interpreted as the legacy baseline comparison, not the canonical release default.

### Pipeline Gold-Standard Recovery (15 epitopes, full proteome screen)

| Method | GS in Top 10% | GS in Top 25% | Mean Rank % |
|--------|---------------|---------------|-------------|
| Binding-only baseline | 15/15 | 15/15 | 2.2% |
| **RF (SESTRAV)** | **4/15** | **7/15** | **34.7%** |
| XGBoost | 1/15 | 3/15 | 52.4% |
| ANN (MLP) | 4/15 | 6/15 | 47.3% |

### Interpreting the Baseline Result

The binding-only baseline outperforms SESTRAV on gold-standard recovery because all 15 gold-standard epitopes were selected from literature specifically for being well-characterized strong MHC binders. This creates a selection bias favoring binding-based ranking.

SESTRAV's value proposition is distinguishing immunogenic from non-immunogenic peptides using signal beyond MHC binding alone - the specificity bottleneck that binding-based methods cannot address (Carri et al. 2023: AUC ~0.60 for binding as immunogenicity proxy). The CV metrics on IEDB data (which include both positive and negative peptides) are the proper evaluation of that claim. No evaluation in this repo isolates performance specifically **among good binders**: the CV corpus is not filtered by binding strength, and `docs/limitations_statement_v1.md` documents it as skewed toward poor-binding negatives rather than restricted to strong ones. The "good binders" framing names the motivating problem, not a tested regime.

### Top Features (RF importance, 21-feature track)

1. `peptide_length` - 17.2%
2. `p5_vdw_volume` - 7.3%
3. `p6_vdw_volume` - 7.3%
4. `p4_vdw_volume` - 7.2%
5. `p4_hydrophobicity` - 7.2%

Van der Waals volume and hydrophobicity at TCR contact positions dominate, consistent with the biophysical model of TCR recognition requiring specific steric and chemical complementarity at the binding interface.
