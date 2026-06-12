# SESTRAV Naming and Feature Migration Specification

This specification details the transition from the legacy naming conventions and feature schemas to the canonical standard introduced in SESTRAV v2.0. This document provides researchers with the necessary details to map their historical data or upgrade their pipelines to the 30-feature canonical standard.

---

## 1. Feature Space Evolution

SESTRAV has evolved its feature extraction framework ([src/features.py](../src/features.py)) to support three primary feature schemas:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Feature Schema Modes                            │
├──────────────────────────┬───────────────────────┬──────────────────────────┤
│    Legacy 21-Feature     │ Canonical 30-Feature  │   Experimental 50-Feat   │
│ (20 Physico + 1 Length)  │ (20 Physico + 10 MHC) │  (40 Physico + 10 MHC)   │
└──────────────────────────┴───────────────────────┴──────────────────────────┘
```

### 1.1 Legacy 21-Feature Mode
*   **Composition:** 20 physicochemical features + 1 peptide length feature (21 features total for training).
*   **Physicochemical representation:** Computed at TCR-contact positions (p4–p8) using 4 properties: Kyte-Doolittle hydrophobicity, aromaticity, Van der Waals volume, and formal charge.
*   **The Binding Score:** The single `binding_score` feature (MHCflurry presentation score) is excluded during training because IEDB training datasets lack allele annotations (~0% coverage), but it is supported as an inference-only feature (yielding a 22-feature matrix at prediction time).

### 1.2 Canonical 30-Feature Mode (v2.0 Default)
*   **Composition:** 20 physicochemical features + 10 per-allele MHC presentation scores (30 features total).
*   **Allele Representation:** Replaces the single, inference-only `binding_score` with 10 distinct per-allele MHCflurry presentation scores (`bind_A0101` through `bind_B4402`).
*   **Targeting:** Default configuration for all production-grade RF, XGBoost, and deep-learning models. (An optional 31st feature, `peptide_length`, can be appended).

### 1.3 Experimental 50-Feature Mode
*   **Composition:** 40 physicochemical features + 10 per-allele MHC presentation scores (50 features total).
*   **Physicochemical representation:** Expands the TCR contact positions (p4–p8) to utilize all 7 properties: hydrophobicity, aromaticity, volume, charge, flexibility (Vihinen), bulkiness (Zimmerman), and hydrophilicity (Hopp-Woods), alongside a structural upward-facing probability proxy.

---

## 2. Naming Map (Legacy to Canonical)

To ensure backward compatibility, the pipeline maintains a mapping of legacy strings to their canonical replacements:

### 2.1 Proteome IDs
Proteome IDs represent the target sliding-window sequences run through Stage 1.

| Legacy Alias | Canonical Replacement | Reason for Change |
| :--- | :--- | :--- |
| `HPV_8_FASTAs` | `HPV16_18_panel8` | Prevents false inference of "HPV strain 8" by specifying strains 16/18. |
| `EBV_8_FASTAs` | `EBV_B95_8_panel8` | Clarifies B95-8 strain context and removes the generic plural suffix. |
| `EBV_panel8_B958` | `EBV_B95_8_panel8` | Standardizes B95-8 strain formatting. |

### 2.2 Model Names
Model file suffixes are canonicalized to explicitly denote their feature count and legacy status.

| Legacy Model Filename | Canonical Replacement | Feature Count | Track |
| :--- | :--- | :---: | :--- |
| `rf_immunogenicity.joblib` | `rf_21feature_legacy.joblib` | 21 | Legacy |
| `xgb_immunogenicity.joblib` | `xgb_21feature_legacy.joblib` | 21 | Legacy |
| `ann_immunogenicity.pt` | `ann_21feature_legacy.pt` | 21 | Legacy |
| `rf_30f_immunogenicity.joblib` | `rf_30feature_integrated.joblib` | 30 | Canonical |
| `xgb_30f_immunogenicity.joblib` | `xgb_30feature_integrated.joblib` | 30 | Canonical |
| `ann_30f_immunogenicity.pt` | `ann_30feature_integrated.pt` | 30 | Canonical |

---

## 3. Compatibility Enforcement Layer (`src/naming.py`)

Runtime aliasing is handled transparently by [src/naming.py](../src/naming.py) to prevent breaking existing analysis scripts:

*   **Transparent Load Resolution:** `resolve_model_path(path)` dynamically checks if a requested canonical model file is missing, automatically looking up and loading the corresponding legacy file if available.
*   **Proteome ID Expansion:** `canonicalize_proteome_id(id)` maps legacy proteome IDs to their canonical replacements.
*   **Write-Path Policy:** All newly generated results default to canonical names (e.g., `results/EBV_B95_8_panel8_ranked.csv`) to enforce standard naming conventions.

---

## 4. Upgrade Guidelines: Transitioning from 21 to 30 Features

### 4.1 Upgrading Your Data Matrices
To upgrade legacy 21-feature datasets, researchers must compute the 10 per-allele MHCflurry presentation scores for their peptide sequences and replace the single `binding_score` column. 

#### Python Example: Programmatic Feature Upgrading
```python
import pandas as pd
from src.features import compute_features_for_dataset, FEATURE_COLUMNS_30

# 1. Load your legacy dataset containing raw peptide sequences
df = pd.read_csv("data/legacy_peptides.csv") # must contain a 'peptide' column

# 2. Extract presentation scores across the 10 MHCflurry alleles
# (Ensure your conda environment has MHCflurry configured and downloaded)
# Under the hood, this computes the canonical 30-feature schema
df_canonical_features = compute_features_for_dataset(
    df, 
    peptide_col='peptide', 
    binding_col='presentation_score' # fallback if single score exists
)

# 3. Filter to the canonical column list
df_out = df_canonical_features[['peptide'] + FEATURE_COLUMNS_30]
df_out.to_csv("data/canonical_30feature_matrix.csv", index=False)
```

### 4.2 Upgrading Model Inference
When loading pre-trained classifiers, the model dynamically detects the input dimensionality via `n_features_in_` and determines whether to evaluate using the 21-feature legacy track or the 30-feature canonical track. To force canonical evaluation, ensure that the input DataFrame contains the exact columns defined in `FEATURE_COLUMNS_30` (imported from `src.features`).
