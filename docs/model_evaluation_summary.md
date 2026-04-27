# SESTRAV Model Evaluation Summary

This document summarizes the legacy 21-feature benchmark line kept for reproducibility and historical comparison. The canonical public release track is configured separately in `config.yaml` and documented in `README.md`.

## Cross-Validation Results (5-fold stratified, 912 training peptides)

| Metric | RF (mean +/- std) | XGBoost (mean +/- std) | ANN/MLP (mean +/- std) |
|--------|-------------------|----------------------|----------------------|
| **AUC-ROC** | **0.726 +/- 0.042** | 0.685 +/- 0.053 | 0.676 +/- 0.055 |
| **AUC-PR** | **0.919 +/- 0.021** | 0.912 +/- 0.019 | 0.911 +/- 0.029 |
| **ISSR@10** | 0.911 +/- 0.057 | 0.911 +/- 0.027 | **0.933 +/- 0.082** |
| **ISSR@25** | **0.947 +/- 0.036** | 0.938 +/- 0.022 | 0.929 +/- 0.038 |

**Best model (legacy benchmark line): RandomForest** — highest AUC-ROC, AUC-PR, and ISSR@25.
This document should be interpreted as the legacy baseline comparison, not the canonical release default.

## Pipeline Gold-Standard Recovery (15 epitopes, full proteome screen)

| Method | GS in Top 10% | GS in Top 25% | Mean Rank % |
|--------|---------------|---------------|-------------|
| Binding-only baseline | 15/15 | 15/15 | 2.2% |
| **RF (SESTRAV)** | **6/15** | **8/15** | **27.1%** |
| XGBoost | 2/15 | 6/15 | 35.6% |
| ANN (MLP) | 0/15 | 3/15 | 36.0% |

### Interpreting the Baseline Result

The binding-only baseline outperforms SESTRAV on gold-standard recovery because all 15 gold-standard epitopes were selected from literature specifically for being well-characterized strong MHC binders. This creates a selection bias favoring binding-based ranking.

SESTRAV's value proposition is distinguishing immunogenic from non-immunogenic peptides **among good binders** — the specificity bottleneck that binding-based methods cannot address (Carri et al. 2023: AUC ~0.60 for binding as immunogenicity proxy). The CV metrics on IEDB data (which include both positive and negative peptides) are the proper evaluation.

## Top Features (RF importance)

1. `peptide_length` — 17.2%
2. `p5_vdw_volume` — 7.3%
3. `p6_vdw_volume` — 7.3%
4. `p4_vdw_volume` — 7.2%
5. `p4_hydrophobicity` — 7.2%

Van der Waals volume and hydrophobicity at TCR contact positions dominate, consistent with the biophysical model of TCR recognition requiring specific steric and chemical complementarity at the binding interface.
