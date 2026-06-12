# IEDB T-Cell Assay Export Comparison Report (June 6, 2026)

This report documents the ingestion, filtering, and comparative statistics for the newly exported IEDB T-cell assay datasets for Epstein-Barr Virus (EBV) and Human Papillomavirus type 16 (HPV16).

---

## 1. Ingestion Summary

The raw IEDB exports were downloaded and saved to [data/raw/iedb_tcell_assay/](data/raw/iedb_tcell_assay/):
- **EBV Export:** `ebv_20260606.csv` (3733 valid T-cell assays)
- **HPV16 Export:** `hpv16_20260606.csv` (587 T-cell assays)

The files were parsed using [src/iedb_data_loader.py](src/iedb_data_loader.py) with the updated flat-header combined column mapping logic.

---

## 2. Comparative Statistics

Below is the side-by-side comparison of the newly ingested June 2026 T-cell Assay dataset versus the current SESTRAV v2/v3 baseline:

| Metric | June 6, 2026 Ingestion | v2.0.0 Baseline | Comparison & Impact |
|--------|------------------------|-----------------|---------------------|
| **Total Valid Assays (Rows)** | **4,320** | N/A (Epitope Table) | The new export represents raw assay-level records, rather than pre-collapsed epitope entries. |
| **Unique Peptides** | **605** | **720** | Resolves to 605 unique 8-11mer peptides. This covers 84% of the v2 baseline sequence diversity. |
| **HLA Allele Coverage** | **12.73%** | **0.00%** | **Significant biological improvement.** 12.73% of the unique peptides now have non-null HLA restriction annotation, laying the foundation for Stage 4 allele-aware training. |
| **Duplicate Conflict Rate** | **26.28%** | 2.37% (v2) | 26.28% of unique peptides had conflicting assay labels in the raw data, resolved using a majority-vote filter (mean label >= 0.5 is positive). |
| **Positive Class Prevalence** | **68.60%** | **70.30%** | Prevalence is highly stable (68.6% vs. 70.3%), maintaining the target class ratio requirements (approx 2.18:1 pos:neg). |
| **Protein (Antigen) Coverage** | **100.0%** (605/605) | 100.0% | Normalization successfully mapped all 605 peptides to standard gene symbols (e.g. LMP2A, E7). |

---

## 3. Allele-Aware Extraction Subset

Using `scripts/extract_allele_aware_data.py`, the dataset was filtered and mapped to the target 10-allele panel (e.g., HLA-A*02:01, HLA-B*08:01):
- **Raw Kept Records:** 85 rows
- **Resolved (Peptide, Allele) Pairs:** 63 pairs
- **Unique Peptides:** 59
- **Unique Alleles:** 6 represented (HLA-A*02:01, HLA-A*03:01, HLA-A*24:02, HLA-B*07:02, HLA-B*08:01, HLA-B*27:05)
- **Gold-Standard Holdouts Removed:** 11 rows (correctly excluded from training)
- **Active Training Pairs:** 52 pairs
- **Class Balance:** 41.27% positive

---

## 4. Key Findings

1. **HLA Restrictions Unlocked:** For the first time, SESTRAV has active HLA restriction context (12.73% coverage overall, rising to 100% in the dedicated allele-aware subset).
2. **High Conflict Rate:** The 26.28% conflict rate reflects significant biological variability across different assay methodologies in the IEDB. The majority vote system handles this robustly.
3. **Training Readiness:** The output training dataset is saved at `data/allele_aware/IEDB-20260606-EBV_HPV16_ALLELE_AWARE-v1.csv` and is ready for Stage 4 training.
