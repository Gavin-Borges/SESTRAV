# SESTRAV Model Evaluation Summary

## v2 Canonical Track: 30-Feature Integrated (720 peptides, 2.36:1 class ratio)

The canonical evaluation track uses 20 physicochemical features (p4–p8 × 4 properties) plus 10 per-allele MHC binding features. All results are 5-fold stratified cross-validation with gold-standard epitopes held out.

Release-scope boundary: The optional ANN/GNN benchmark track is supplementary and not part of the canonical publish gate. The canonical publish gate is based on the RF-configured Stage 1-4 workflow and frozen validation artifacts.
ANN/GNN values are sourced from Project 2 evidence and mirrored in SESTRAV-Dev docs.

### Cross-Validation Results (5-fold stratified)

| Metric | RF (mean ± std) | XGBoost (mean ± std) | ANN 256-128-64 (mean ± std) |
|--------|-----------------|---------------------|---------------------------|
| **AUC-PR** | 0.810 ± 0.025 | 0.805 ± 0.028 | **0.825 ± 0.025** |
| **AUC-ROC** | 0.670 ± 0.042 | 0.665 ± 0.045 | 0.670 ± 0.040 |
| **ISSR@10** | 0.870 ± 0.050 | 0.865 ± 0.055 | **0.880 ± 0.045** |
| **ISSR@25** | 0.920 ± 0.035 | 0.915 ± 0.038 | **0.930 ± 0.030** |

**Best benchmark performer (30-feature track): ANN (256-128-64 ReLU dropout 0.2)** — highest AUC-PR and ISSR@10 in this comparison table.

> **Note:** AUC-PR values shown are representative of the 30-feature track. Exact values depend on the training run seed and dataset split. Run `src/train_classifier.py` and `src/ann_benchmark.py` locally to reproduce.

> **Provenance note:** The canonical optional ANN/GNN evidence source for the
> latest Project 2 sync is documented in `docs/nn_gnn_project2_sync_matrix.md`.

### ANN Architecture Search (14 Configurations, Project 2)

The architecture search evaluates depth, width, activation function, and dropout rate. The best configuration (256-128-64 ReLU dropout 0.2) was identified from the CMB 523 Project 2 systematic search.

Saved to: `models/ann_architecture_search.csv` (generated locally via `--search`
flag). Project 2 run metadata and benchmark lineage are documented in
`docs/nn_gnn_optional_module_guide.md`.

### GNN Benchmark Results (Exploratory)

| Model | AUC-PR (mean ± std) | AUC-ROC (mean ± std) |
|-------|---------------------|---------------------|
| GCN (2-layer) | 0.778 ± 0.030 | 0.614 ± 0.050 |
| GAT (2-layer, 4-head) | **0.796 ± 0.028** | **0.637 ± 0.045** |
| Bipartite Peptide-Allele | 0.789 ± 0.032 | 0.612 ± 0.055 |

GNNs underperform tabular models on this dataset but capture structural inter-residue patterns that fixed-position features cannot represent. Included for representation space characterization.

### Exact Project 2 Optional Benchmark Values

For exact unrounded synced values, see:
- `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/bootstrap_metric_cis.csv`
- `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/gnn_sequence_benchmark.csv`
- `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/gnn_bipartite_benchmark.csv`
- `docs/nn_gnn_optional_module_guide.md`

### Ablation Study Results (Feature Group Contributions)

| Feature Set | Features | AUC-PR (mean) | AUC-ROC (mean) |
|-------------|----------|---------------|----------------|
| `physico_20` | 20 | 0.772 | 0.577 |
| `binding_10` | 10 | 0.851 | 0.727 |
| `sestrav_21` | 21 | 0.784 | 0.622 |
| `combined_30` | 30 | 0.825 | 0.670 |
| `full_31` | 31 | **0.864** | **0.743** |

Binding features are the strongest individual group; physicochemical features provide complementary discrimination when combined. The full 31-feature set (including peptide_length) yields the best overall performance.

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

**Best model (legacy benchmark line): RandomForest** — highest AUC-ROC, AUC-PR, and ISSR@25.
This document should be interpreted as the legacy baseline comparison, not the canonical release default.

### Pipeline Gold-Standard Recovery (15 epitopes, full proteome screen)

| Method | GS in Top 10% | GS in Top 25% | Mean Rank % |
|--------|---------------|---------------|-------------|
| Binding-only baseline | 15/15 | 15/15 | 2.2% |
| **RF (SESTRAV)** | **6/15** | **8/15** | **27.1%** |
| XGBoost | 2/15 | 6/15 | 35.6% |
| ANN (MLP) | 0/15 | 3/15 | 36.0% |

### Interpreting the Baseline Result

The binding-only baseline outperforms SESTRAV on gold-standard recovery because all 15 gold-standard epitopes were selected from literature specifically for being well-characterized strong MHC binders. This creates a selection bias favoring binding-based ranking.

SESTRAV's value proposition is distinguishing immunogenic from non-immunogenic peptides **among good binders** — the specificity bottleneck that binding-based methods cannot address (Carri et al. 2023: AUC ~0.60 for binding as immunogenicity proxy). The CV metrics on IEDB data (which include both positive and negative peptides) are the proper evaluation.

### Top Features (RF importance, 21-feature track)

1. `peptide_length` — 17.2%
2. `p5_vdw_volume` — 7.3%
3. `p6_vdw_volume` — 7.3%
4. `p4_vdw_volume` — 7.2%
5. `p4_hydrophobicity` — 7.2%

Van der Waals volume and hydrophobicity at TCR contact positions dominate, consistent with the biophysical model of TCR recognition requiring specific steric and chemical complementarity at the binding interface.
