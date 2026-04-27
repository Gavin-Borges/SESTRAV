# IEDB Training Data Summary

## Source

All training data was curated from the **Immune Epitope Database (IEDB, https://www.iedb.org/)** for T-cell epitope assays against EBV and HPV16 antigens.
This v1 summary reflects the frozen baseline release dataset.

## Dataset Versioning (v1 policy)

- Frozen baseline version ID: `IEDB-20260424-EBV_HPV16_BASELINE-v1`
- Expanded reruns must use a new version ID and be labeled exploratory.
- Any expanded-dataset report must include side-by-side comparison versus baseline on:
  - `results/h2_tier_a_summary.csv`
  - `results/baseline_comparison.csv`

## Dataset Overview

| Statistic | Value |
|-----------|-------|
| Total records | 928 |
| Unique peptides | 928 |
| Positive (immunogenic) | 787 (84.8%) |
| Negative (non-immunogenic) | 141 (15.2%) |
| Class imbalance ratio | 5.6:1 (pos:neg) |

## Breakdown by Virus

| Virus | Total | Positive | Negative | Pos Rate |
|-------|-------|----------|----------|----------|
| EBV | 675 | 589 | 86 | 87.3% |
| HPV16 | 253 | 198 | 55 | 78.3% |

## Peptide Length Distribution

| Length | Count | Percentage |
|--------|-------|-----------|
| 8-mer | 44 | 4.7% |
| 9-mer | 595 | 64.1% |
| 10-mer | 221 | 23.8% |
| 11-mer | 68 | 7.3% |

9-mers dominate, consistent with the canonical HLA-A*02:01 preference for 9-mer peptides.

## Gold-Standard Hold-Out

16 records (covering 15 unique gold-standard epitopes) are held out from training and used exclusively for pipeline-level validation. These are well-characterized immunogenic epitopes from the published literature:

- EBV: 10 epitopes (BMLF1, LMP1, LMP2, EBNA1, EBNA3A/B/C, BZLF1, GP350)
- HPV16: 5 epitopes (E6, E7)

## Class Imbalance Handling

The heavy positive skew (84.8%) is addressed during training via:
- **RF**: `class_weight='balanced'` (inverse-frequency weighting)
- **XGBoost**: `scale_pos_weight = n_neg / n_pos` (negative upweighting)
- **ANN**: per-sample loss weighting using class-inversed frequencies

## Data Quality Notes

- All peptides contain only standard amino acids (20 canonical AAs)
- No duplicate peptides exist in the dataset
- Peptide lengths are restricted to 8–11 residues (MHC-I compatible)
- IEDB assay types include: IFN-γ ELISPOT, 51Cr release, multimer staining, intracellular cytokine staining

---

## Mode B: Updated IEDB Data (v2)

### Version ID: `IEDB-20260424-EBV_HPV16_UPDATED-v2`

This section documents the **Mode B exploratory** dataset produced from
updated IEDB Epitope Table exports downloaded on 2026-04-24.

### Source Files

| File | Rows | Unique 8-11mers |
|------|------|-----------------|
| `UPDATED_EBV_epitope table_IEDB_T-cell positive.xlsx` | 672 | 324 |
| `UPDATED_EBV_epitope table_IEDB_T-cell negative.xlsx` | 1140 | 287 |
| `UPDATED_HPV16_epitope table_IEDB_T-cell positive.xlsx` | 544 | 182 |
| `UPDATED_HVP16_epitope table_IEDB_T-cell negative.xlsx` | 443 | 128 |

### Inclusion / Exclusion Criteria

- Format: IEDB Epitope Table exports (labels derived from filename)
- Length filter: 8–11 residues (MHC class I compatible)
- Amino acid filter: standard 20 canonical amino acids only
- Duplicate resolution: majority-vote across pos/neg file appearances
- 201 cross-label conflicts resolved by majority vote

### Dataset Overview (v2)

| Statistic | Value |
|-----------|-------|
| Total unique peptides | 720 |
| Positive (immunogenic) | 506 (70.3%) |
| Negative (non-immunogenic) | 214 (29.7%) |
| Class imbalance ratio | 2.36:1 (pos:neg) |

### Breakdown by Virus (v2)

| Virus | Total | Positive | Negative | Pos Rate |
|-------|-------|----------|----------|----------|
| EBV | 470 | 324 | 146 | 68.9% |
| HPV16 | 250 | 182 | 68 | 72.8% |

### Peptide Length Distribution (v2)

| Length | Count | Percentage |
|--------|-------|-----------|
| 8-mer | 37 | 5.1% |
| 9-mer | 411 | 57.1% |
| 10-mer | 212 | 29.4% |
| 11-mer | 60 | 8.3% |

### Gold-Standard Hold-Out Compatibility

All 16 gold-standard epitopes (15 unique sequences) are present in v2 and
correctly labeled as positive. The hold-out split is fully compatible with v1.

### Class Balance Comparison (v1 vs v2)

| Metric | v1 Baseline | v2 Updated |
|--------|-------------|------------|
| Total peptides | 928 | 720 |
| Positive % | 84.8% | 70.3% |
| Negative % | 15.2% | 29.7% |
| Pos:Neg ratio | 5.6:1 | 2.36:1 |

The v2 dataset has substantially improved class balance, reducing the extreme
positive skew that was a known limitation of the v1 baseline.
