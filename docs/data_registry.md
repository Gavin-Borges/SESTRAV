# SESTRAV Dataset Registry

This registry tracks the versions of the `immunogenicity_dataset.csv` used for model training and evaluation within SESTRAV.

## Version 3 (Current)
- **File:** `data/immunogenicity_dataset_v3.csv`
- **Date:** 2026-05-22
- **Description:** Expanded dataset including the updated IEDB data for EBV and HPV16 (T-cell positive and negative), as well as HPV11 data.
- **Total Peptides:** 1004 (773 positive, 231 negative)
- **Processing:**
  - Length constraint: 8-11 amino acids.
  - Duplicate resolution: Majority vote (≥0.5 -> 1).
  - Gold Standard Holdout: 16 core epitopes removed from training data.
- **Input Sources:**
  - `09_Data/UPDATED_EBV_epitope table_IEDB_T-cell positive.xlsx`
  - `09_Data/UPDATED_EBV_epitope table_IEDB_T-cell negative.xlsx`
  - `09_Data/UPDATED_HPV16_epitope table_IEDB_T-cell positive.xlsx`
  - `09_Data/UPDATED_HVP16_epitope table_IEDB_T-cell negative.xlsx`
  - `09_Data/HPV11_epitope table_IEDB_T-cell positive.xlsx`
  - `09_Data/HPV11_epitope table_IEDB_T-cell negative.xlsx`
  - Plus original data sources (fallback).

## Version 2 (Legacy)
- **File:** `data/immunogenicity_dataset_v2.csv` (previously `immunogenicity_dataset.csv`)
- **Date:** 2026-04-13
- **Description:** Initial 928-peptide dataset derived from the original 6 IEDB xlsx files.
- **Positives:** ~653
- **Negatives:** ~275
- **Input Sources:**
  - `09_Data/EBV_epitope table_IEDB_T-cell positive.xlsx`
  - `09_Data/EBV_epitope table_IEDB_T-cell negative.xlsx`
  - `09_Data/HPV16_epitope table_IEDB_T-cell positive.xlsx`
  - `09_Data/HVP16_epitope table_IEDB_T-cell negative.xlsx`

## Version 1 (Archived)
- **File:** `data/immunogenicity_dataset_v1_archived.csv`
- **Description:** Early format-incompatible data before the Epitope Table format rewrite.
