# SESTRAV 2.0 Data Admissibility Checklist

## Purpose
This document defines the strict criteria required for admitting any non-IEDB dataset (or expanded IEDB exports) into the SESTRAV training or validation manifold. Adherence to these criteria ensures that data growth does not compromise reproducibility, metadata provenance, or scoring validity.

## 1. Metadata Completeness
Every incoming dataset must be accompanied by the following metadata:
- [ ] **Data Source Provenance:** (e.g., Publication DOI, Consortium Link, Lab Name)
- [ ] **License/Usage Constraints:** (Must be compatible with open-source/research usage)
- [ ] **Assay Type:** (e.g., ELISPOT, Tetramer, Multimer, Mass Spectrometry)
- [ ] **Timestamp / Version:** (Date of extraction or data freeze)

## 2. Formatting and Schema Compliance
The dataset must seamlessly map into the SESTRAV format:
- [ ] **Peptide Sequences:** Must be strictly amino acid letters (no non-standard tokens).
- [ ] **Peptide Length:** 8 to 11 amino acids only.
- [ ] **Target Alleles:** Must be resolvable to a standard 4-digit HLA allele format (e.g., `HLA-A*02:01`).
- [ ] **Label Consistency:** Must provide a clear positive/negative immunogenicity label or an ordinal assay score that can be deterministically thresholded.

## 3. Duplicate and Conflict Resolution
Before merging, the following rules apply to duplicate peptides:
- [ ] **Intra-dataset Conflicts:** Peptides with conflicting labels within the same dataset must be resolved (e.g., drop row or majority vote). 
- [ ] **Cross-dataset Conflicts:** If the new dataset conflicts with canonical IEDB data, the canonical data takes precedence unless an explicit override rule is documented in the version changelog.
- [ ] **Conflict Threshold:** If the incoming dataset introduces >15% conflict with existing data, it must be flagged for manual review and blocked from automated merging.

## 4. Post-Merge Validation
- [ ] **Yield Check:** The final dataset post-merge must not inexplicably drop in positive sample count.
- [ ] **Virus Composition:** Record the resulting distribution of pathogen targets (e.g., % EBV, % HPV, % Other).
- [ ] **Dataset Versioning:** The updated merged dataset must be given a new unique version identifier in `config.yaml` and documented in `docs/data_registry.md`.
