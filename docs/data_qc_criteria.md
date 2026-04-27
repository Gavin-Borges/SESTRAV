# SESTRAV Data Quality Control Criteria

## Peptide Filtering (Stage 1 + IEDB Loader)

| Criterion | Rule | Rationale |
|-----------|------|-----------|
| Length | 8-11 amino acids only | MHC class I binding groove accommodates 8-11mers |
| Amino acids | Standard 20 only (ACDEFGHIKLMNPQRSTVWY) | Non-standard residues (U, X, B, Z, J) lack physicochemical lookup values |
| MHC class | Class I only | Class II alleles (HLA-DR/DP/DQ) are excluded from the binding panel |

## IEDB Data Cleaning

| Step | Rule | Module |
|------|------|--------|
| Format detection | Auto-detect Epitope Table vs T-cell Assay format | `iedb_data_loader._detect_format()` |
| Label extraction | Epitope Table: from filename ("positive"/"negative"); T-cell Assay: from Qualitative Measure column | `iedb_data_loader._label_from_filename()`, `.map_label()` |
| Virus tagging | From filename pattern (EBV, HPV16, HPV11), handles "HVP16" typo | `iedb_data_loader._virus_from_filename()` |
| Duplicate resolution | Per-peptide majority vote: mean(label) >= 0.5 = positive | `iedb_data_loader.load_and_clean_iedb()` |
| Gold-standard hold-out | 16 records (15 unique peptides + FLRGRAYGL variant) excluded from training | `train_classifier.train_models()` |

## Feature Extraction QC

| Check | Rule |
|-------|------|
| Feature count | 22 per peptide in legacy mode (21 for training), 30 in canonical mode (20 physico + 10 allele binding) |
| Position overlap | p7/p8 zero-imputed when overlapping p6 or C-terminal anchor (8-mers) |
| Missing values | None allowed; all amino acid lookups have defaults |
| Binding score | Set to 0.0 during training; real MHCflurry values during inference |

## Model QC

| Check | Rule |
|-------|------|
| `n_features_in_` | Must equal 21 (legacy TRAIN_FEATURE_COLUMNS) or 30 (canonical FEATURE_COLUMNS_30) |
| `predict_proba` output | Shape (n, 2), rows sum to 1.0, all values in [0, 1] |
| Cross-validation | Stratified 5-fold, `random_state=42`, identical splits across models |
| Class imbalance | Handled via balanced weights (RF), scale_pos_weight (XGB), per-sample weighting (ANN) |
