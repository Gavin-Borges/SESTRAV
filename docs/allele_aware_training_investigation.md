# Allele-Aware Training Label Investigation

**Date:** 2026-04-24
**Status:** Investigation complete, migration path defined

## Problem Statement

SESTRAV's training labels are derived from IEDB **Epitope Table** export filenames
("T-cell positive" / "T-cell negative"). This format provides no per-peptide
HLA allele restriction context. The model therefore treats immunogenicity as
allele-independent, even though a peptide's immunogenicity is fundamentally
allele-specific: a peptide that is immunogenic when presented by HLA-A*02:01
may not be immunogenic when presented by HLA-B*07:02.

## Current State

### What the loader already supports

`src/iedb_data_loader.py` handles two IEDB export formats:

1. **Epitope Table** (current v2 data source):
   - Labels from filename only ("T-cell positive" / "T-cell negative")
   - No `Qualitative Measure` column, no `Allele` column
   - 32 columns with peptide in col 2, antigen in col 6, organism in col 8
   - **All 720 training peptides use this format**

2. **T-cell Assay** (supported but not currently used):
   - Labels from per-row `Qualitative Measure` column (Positive/Negative)
   - Optional `Allele` column with HLA restriction
   - The loader already extracts, standardizes, and propagates allele info
   - MHC class II alleles are filtered out

### Data audit findings

From `results/data_bias_audit.md`:
- Missing allele: **720/720 (100%)** in current training data
- This is entirely due to using Epitope Table format as source

## IEDB T-cell Assay Export Format

IEDB provides an alternative export: **T-cell Assay** results. These include:

| Column | Content | Relevance |
|--------|---------|-----------|
| Description | Peptide sequence | Equivalent to Epitope Table col 2 |
| Qualitative Measure | "Positive", "Positive-High", "Negative" | Per-assay immunogenicity label |
| Allele | "HLA-A*02:01" etc. | HLA restriction element |
| Assay | Assay type (ELISPOT, multimer, etc.) | Context for label confidence |
| Response Frequency | Quantitative response | Could enable graded labels |

### Advantages of T-cell Assay format

1. **Per-assay labels**: Each row is a distinct assay result, not a collapsed epitope. A peptide tested in 3 assays yields 3 rows with potentially different outcomes.
2. **Allele context**: Each assay row specifies which HLA allele was used for restriction.
3. **Assay type**: Enables filtering by assay confidence (e.g. ELISPOT vs computational prediction).
4. **Quantitative data**: Response frequency could support continuous label regression.

### Disadvantages / risks

1. **Data volume change**: T-cell Assay exports may have different row counts than Epitope Table exports for the same query, because they report per-assay rather than per-epitope.
2. **Label granularity**: Multiple assays per peptide-allele pair may conflict; deduplication becomes per-peptide-allele rather than per-peptide.
3. **Gold-standard compatibility**: The 15 gold-standard epitopes must be verified present in the new export.
4. **Reproducibility**: Switching source format changes the training data and all downstream metrics; requires a full Mode B revalidation.

## Recommended Migration Path

### Phase 1: Dual-source data collection (low risk)

Download IEDB T-cell Assay exports for EBV and HPV16 alongside the existing
Epitope Table exports. Do NOT replace the existing training data yet.

**IEDB query parameters:**
- Organism: Human herpesvirus 4 (EBV), Human papillomavirus type 16
- Assay: T-cell assay
- MHC restriction: MHC class I only
- Export format: Full T-cell Assay table (not Epitope Table)

### Phase 2: Allele-aware dataset construction

1. Load T-cell Assay exports using the existing `tcell_assay` path in `iedb_data_loader.py`
2. Filter to the 10-allele panel used by SESTRAV (HLA-A*02:01 ... HLA-B*44:02)
3. Create allele-specific labels: `(peptide, allele) -> label` rather than `peptide -> label`
4. Resolve conflicts per peptide-allele pair via majority vote
5. Assign version ID: `IEDB-YYYYMMDD-EBV_HPV16_ALLELE_AWARE-v1`

### Phase 3: Allele-aware model training

Two approaches to evaluate:

**Approach A — Allele-conditioned features:**
Extend the 30-feature vector to include a one-hot or embedding for the presenting
allele. Train on `(peptide_features, allele) -> label` tuples. This quadruples
the training data (each peptide appears per tested allele) but requires model
architecture changes.

**Approach B — Allele-filtered training:**
For each allele in the panel, train a specialized model using only peptides
with confirmed immunogenicity labels for that allele. This produces 10
allele-specific classifiers but reduces per-model training data.

### Phase 4: Validation

1. Gold-standard positive recovery (all 15 epitopes)
2. Gold-standard negative discrimination (10 strong-binder negatives)
3. Side-by-side comparison with current allele-blind model using Mode A/B policy
4. Per-allele subgroup evaluation using `src/subgroup_eval.py`

## Code Infrastructure Already in Place

The loader's `tcell_assay` path (lines 337-400 of `iedb_data_loader.py`) already:
- Auto-detects the format via `_detect_format()`
- Extracts peptide, label, and allele columns
- Standardizes allele names via `standardize_allele()`
- Filters out MHC class II
- Propagates allele through deduplication

The training pipeline (`train_classifier.py`) already:
- Reads `strain` subgroup column when present (now also `protein`)
- Could be extended to read `allele` as an additional subgroup
- `subgroup_eval.py` can evaluate per-allele performance with no changes

## Immediate Action Items

1. **No code changes needed** for the investigation itself. The loader infrastructure
   is ready for T-cell Assay format.
2. **Data acquisition required**: Download T-cell Assay exports from IEDB for EBV
   and HPV16 and place in a versioned data directory.
3. **Comparison run**: Execute `load_and_clean_iedb()` on T-cell Assay exports and
   compare peptide/label coverage against current v2 Epitope Table dataset.
4. **Mode B policy**: Any allele-aware training run must follow
   `docs/iedb_mode_policy.md` requirements.
