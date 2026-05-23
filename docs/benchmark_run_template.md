# SESTRAV External Benchmark Execution Template

## Purpose
This document provides a repeatable protocol and checklist for running external benchmarks (e.g., PredIG, PRIME, other tools) against SESTRAV. It ensures that any new comparisons are reproducible, maintain strict holdout boundaries, and record proper provenance.

## 1. Run Setup & Metadata

Before beginning an external benchmark run, copy this template to `results/external_tool_outputs/<run_id>/benchmark_manifest.md` and fill it out.

- **Run ID:** `extval_YYYYMMDD_HHMM_[Initials]_[Tier]`
- **Date:** 
- **Operator:** 
- **Objective:** (e.g., Re-run PredIG comparison on v2 dataset, Evaluate new tool X)

## 2. Dataset Definition

Define the exact datasets used for the benchmark to ensure reproducibility.

- **SESTRAV Dataset Version:** 
- **SESTRAV Model Version/Features:** (e.g., 30-feature RF)
- **External Tool(s) Evaluated:** 
  - Tool A Version / Commit Hash: 
  - Tool B Version / Commit Hash: 
- **Proteome/Virus Scope:** (e.g., HPV16, EBV, both)
- **HLA Allele Scope:** (List all evaluated alleles or reference an external list)
- **Peptide Constraints:** (e.g., 8-11 mers only)

## 3. Pre-Run Quality Checklist

Verify these constraints are met before launching the evaluation script:

- [ ] **Holdout Preservation:** The benchmark dataset has strictly excluded any gold-standard training data used for the SESTRAV model.
- [ ] **Harmonized Alleles:** The external tools and SESTRAV are evaluating the exact same list of HLA alleles for each peptide.
- [ ] **Sanity Checks Passed:** Peptides contain only valid amino acids and meet the size constraints of all evaluated tools.

## 4. Execution Protocol

1. **Format Conversion:** Run the script to export SESTRAV candidates into the input format required by the external tool. (e.g., `python scripts/export_for_external.py --tool prime`)
2. **Tool Execution:** Execute the external tool using its documented standard procedure. Record the exact command used here:
   ```bash
   # Add execution command here
   ```
3. **Merge Results:** Use the `run_external_benchmark.sh` script or equivalent Python utility to merge SESTRAV scores with the external tool outputs based on Peptide + Allele combinations.
4. **Metric Generation:** Generate ROC, PR, and ISSR scores.
   - Script run: `python src/external_validation_finalize.py --run-dir <run_id> --merged <merged_file>`

## 5. Post-Run Audit and Freezing

After the run has completed and metrics are generated:

- [ ] Verify that missing data (peptides skipped by external tools) were handled according to the defined collapse rules (e.g., assigning a neutral score or dropping them consistently across all tools).
- [ ] Freeze the directory: Do not modify any files inside `results/external_tool_outputs/<run_id>/` after the final metrics are calculated. If an error is found, create a new Run ID.
- [ ] Update `docs/benchmark_performance_report.md` (or equivalent) with the results and reference this Run ID.

## 6. Limitations & Notes
*Document any issues encountered, tools that timed out, or edge cases handling ambiguous alleles.*

- 
