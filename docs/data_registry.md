# SESTRAV Dataset Registry

This registry tracks all dataset versions used for model training and evaluation within SESTRAV.
Each entry follows the standard version ID format: `IEDB-YYYYMMDD-<PATHOGENS>-v<N>`.

> **QC Gate Policy:** Until `scripts/data_qc_gate.py` exists, QC gate status is recorded as
> "manual review" per the master plan. Automated QC gating is targeted for Stage 2.4.

---

## Version 3 (Current — Active Training Set)

**Version ID:** `IEDB-20260522-EBV_HPV16_HPV11_UPDATED-v3`
**File:** `data/immunogenicity_dataset_v3.csv`
**Date Ingested:** 2026-05-22
**Status:** Active — used for all v2.0.0 model training and external benchmarking

### Composition

| Pathogen | Count | Fraction |
|----------|-------|----------|
| EBV (Epstein-Barr Virus) | ~684 | 68.1% |
| HPV16 (Human Papillomavirus type 16) | ~310 | 30.9% |
| HPV11 (Human Papillomavirus type 11) | ~10 | 1.0% |

| Class | Count | Fraction |
|-------|-------|----------|
| Positive (immunogenic) | 773 | 77.0% |
| Negative (non-immunogenic) | 231 | 23.0% |
| **Total** | **1004** | 100% |

**Class ratio:** 3.35:1 (positive:negative)
> ⚠️ This exceeds the target governance range of 1.5:1–4:1. Satisfies upper bound but warrants monitoring.

### Peptide Length Distribution

| Length | Count | Fraction |
|--------|-------|---------|
| 8-mer | ~65 | ~6.5% |
| 9-mer | ~650 | ~64.7% |
| 10-mer | ~180 | ~17.9% |
| 11-mer | ~109 | ~10.9% |

Length constraint applied: 8–11 amino acids.

### Allele Coverage

**Allele coverage: ~0% (non-null HLA allele annotation)**
> ⚠️ **Mandatory disclosure:** Training data contains no HLA allele context. This is the single largest
> biological gap in the current model. Allele-aware training (Stage 4) is the critical path forward.
> Allele coverage will be measured after IEDB T-cell Assay exports are downloaded and ingested.

### Conflict Resolution

- Duplicate resolution policy: Majority vote (≥0.5 → positive label)
- Conflict rate: not yet quantified for v3 (manual review status)

### QC Gate Status

| Gate | Status | Notes |
|------|--------|-------|
| Length filter (8–11 AA) | ✅ Manual PASS | Enforced in `iedb_data_loader.py` |
| Duplicate majority vote | ✅ Manual PASS | Applies to replicate assay records |
| Class ratio check | ⚠️ Monitor | 3.35:1 — near upper governance bound |
| Allele coverage | ❌ 0% | Blocker for Stage 4 |
| Gold-standard holdout removed | ✅ Manual PASS | 16 core epitopes excluded from training |
| Contamination gate (external eval) | ✅ Measured | 36.9% overlap with PRIME/PredIG training sets |
| Automated QC gate script | 🔲 Pending | `data_qc_gate.py` not yet written |

### Input Sources

| File | Pathogen | Class |
|------|----------|-------|
| `09_Data/UPDATED_EBV_epitope table_IEDB_T-cell positive.xlsx` | EBV | Positive |
| `09_Data/UPDATED_EBV_epitope table_IEDB_T-cell negative.xlsx` | EBV | Negative |
| `09_Data/UPDATED_HPV16_epitope table_IEDB_T-cell positive.xlsx` | HPV16 | Positive |
| `09_Data/UPDATED_HVP16_epitope table_IEDB_T-cell negative.xlsx` | HPV16 | Negative |
| `09_Data/HPV11_epitope table_IEDB_T-cell positive.xlsx` | HPV11 | Positive |
| `09_Data/HPV11_epitope table_IEDB_T-cell negative.xlsx` | HPV11 | Negative |
| Original v2 sources (fallback overlap) | EBV + HPV16 | Both |

### Key Differences vs. v2

- Added HPV11 data (small cohort; weight-balanced during training)
- Updated IEDB exports for EBV and HPV16 (post-2026-04 assay records included)
- Gold-standard holdout expanded from 15 → 16 epitopes

---

## Version 2 (Legacy — v2.0.0 Baseline Reference)

**Version ID:** `IEDB-20260413-EBV_HPV16-v2`
**File:** `data/immunogenicity_dataset_v2.csv`
*(previously the root-level `immunogenicity_dataset.csv` before v3 superseded it)*
**Date Ingested:** 2026-04-13
**Status:** Legacy — retained for historical comparability; not used in active training

### Composition

| Class | Count | Fraction |
|-------|-------|----------|
| Positive (immunogenic) | ~653 | ~70.3% |
| Negative (non-immunogenic) | ~275 | ~29.7% |
| **Total** | **~928** | 100% |

**Class ratio:** ~2.37:1 (positive:negative) — within target governance range (1.5:1–4:1) ✅

| Pathogen | Fraction |
|----------|----------|
| EBV | ~69% |
| HPV16 | ~31% |

**Allele coverage: ~0%** — same limitation as v3.

### External Benchmark Reference (v2.0.0)

The frozen external Tier A benchmark was run against **N=720 intersecting peptides** from this dataset
that appear in the PRIME 2.1 and PredIG-Path evaluation inputs.

| Tool | AUC-PR (N=720) | AUC-PR (clean holdout, N=451) |
|------|----------------|-------------------------------|
| **RF (SESTRAV 2.0)** | **0.828** | **0.822** |
| PRIME 2.1 | 0.777 | 0.720 |
| PredIG-Path | 0.727 | — |

Contamination finding: external tools show **36.9%** training set intersection with eval set.
Clean holdout (contamination-excluded): N=451.

### QC Gate Status

| Gate | Status | Notes |
|------|--------|-------|
| Length filter (8–11 AA) | ✅ Manual PASS | |
| Duplicate majority vote | ✅ Manual PASS | |
| Class ratio check | ✅ Manual PASS | 2.37:1 — within range |
| Allele coverage | ❌ 0% | |
| Gold-standard holdout removed | ✅ Manual PASS | 15 core epitopes |

### Input Sources

| File | Pathogen | Class |
|------|----------|-------|
| `09_Data/EBV_epitope table_IEDB_T-cell positive.xlsx` | EBV | Positive |
| `09_Data/EBV_epitope table_IEDB_T-cell negative.xlsx` | EBV | Negative |
| `09_Data/HPV16_epitope table_IEDB_T-cell positive.xlsx` | HPV16 | Positive |
| `09_Data/HVP16_epitope table_IEDB_T-cell negative.xlsx` | HPV16 | Negative |

---

## Version 1 (Archived — Pre-Schema)

**Version ID:** `IEDB-pre20260413-EBV_HPV16-v1`
**File:** `data/immunogenicity_dataset_v1_archived.csv`
**Date:** pre-2026-04-13
**Status:** Archived — format-incompatible; do not use

### Notes

Early format before the Epitope Table schema rewrite. Column names and label conventions
differ from v2+. Retained for audit trail only.

---

## Candidate Dataset (Pending Ingestion)

### IEDB T-Cell Assay Exports — June 2026

**Version ID (tentative):** `IEDB-20260606-EBV_HPV16_TCELL_EXPORT-candidate`
**Status:** 🔲 Not yet downloaded — **Priority 1A action**
**Target files:**
- `data/raw/iedb_tcell_assay/ebv_20260606.csv`
- `data/raw/iedb_tcell_assay/hpv16_20260606.csv`

**Download instructions:**
1. iedb.org → Advanced Search → T Cell Assays
2. Filter: Organism = `Human herpesvirus 4 (Epstein-Barr virus)`, MHC class = `I`
3. Export full table → `ebv_20260606.csv`
4. Repeat for HPV16 (Organism = `Human papillomavirus type 16`, MHC class = `I`)
5. Run `python -m src.iedb_data_loader tcell_assay <path>` on both files

**Success signal:** Allele coverage > 60% non-null in the new exports.

**Why this matters:** Current training data has 0% allele coverage. These exports use the
T-Cell Assay format which includes MHC restriction data (HLA allele) that is absent in the
Epitope Table format used by v1–v3. Ingesting this data is the prerequisite for allele-aware
training (Stage 4).

---

## Registry Governance

| Field | Required for v3+ Entry |
|-------|----------------------|
| Version ID | ✅ Standard format `IEDB-YYYYMMDD-PATHOGENS-v<N>` |
| Row count | ✅ |
| Unique peptide count | ✅ |
| Allele coverage fraction | ✅ (disclose 0% explicitly) |
| Pathogen breakdown | ✅ |
| Class ratio | ✅ |
| Conflict resolution stats | ✅ |
| QC gate pass/fail | ✅ (mark as "manual review" until `data_qc_gate.py` exists) |
| Input source file list | ✅ |

*Last updated: 2026-06-06 | Based on master plan v5.0 Stage 2.3 requirements*

---

## MLOps Data Provenance Audit Trail

**Date Logged:** 2026-06-06
**Purpose:** Audit trail for pipeline validation run and AUC-PR results.

- **Dataset File:** `immunogenicity_dataset.csv`
- **Total Row Count:** 1,004
- **Positive to Negative Class Ratio:** 3.35:1 (773 positive, 231 negative)
- **Git Commit Hash:** `b07a5d6845d3b3669a35138f00e2fbef1f18fad5`

