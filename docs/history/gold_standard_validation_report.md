# SESTRAV Gold-Standard Validation Report

## Overview

15 well-characterized epitopes (10 EBV + 5 HPV) are tracked through every pipeline stage to verify reproducibility and ranking behavior. These epitopes are held out from model training, with an additional FLR strain-variant holdout (`FLRGRAYGL`) retained in training exclusion logic.

This document is a narrative companion to the committed machine-readable outputs:

- `results/gold_standard_validation.csv`
- `results/baseline_comparison.csv`

## Current Committed Snapshot (v2 dataset, regenerated April 25, 2026)

### Stage 1: Peptide Generation — 15/15 (100%)

All 15 gold-standard peptides are present in Stage 1 outputs.

### Stage 2: MHC Binding — 15/15 found, 15/15 strong binders

All 15 peptides are present in Stage 2 binding outputs and pass the strong-binder criterion used in validation.

### Stage 4: Immunogenicity Ranking — 15/15 found, 9/15 in top 25%

The committed `results/gold_standard_validation.csv` records:

- 15/15 peptides found in Stage 4 output
- 9/15 peptides in top 25% (EBV: 5/10, HPV: 4/5)
- Top-ranked gold-standard peptide: `CLGGLLTMV` at rank 2 (EBV)
- Best HPV: `RAHYNIVTF` at rank 84

These values were regenerated from a clean forced workflow run using the v2 dataset (720 peptides, 2.36:1 class ratio):

- `snakemake --snakefile pipeline.smk --cores 4 --forceall`
- `snakemake --snakefile pipeline.smk full_validation_report --cores 4 --forceall`

They supersede all older local runs and should be treated as the auditable repository snapshot unless a newer validated bundle is committed.

## Baseline Context

`results/baseline_comparison.csv` in this regenerated snapshot reports:

| Method | Virus | GS Found | Top 10% | Top 25% | Mean Rank % |
|--------|-------|----------|---------|---------|-------------|
| RF (SESTRAV) | EBV | 10 | 4 | 5 | 26.6% |
| RF (SESTRAV) | HPV | 5 | 3 | 4 | 14.6% |
| RF (SESTRAV) | Combined | 15 | 7 | 9 | 20.6% |
| Binding-only | Combined | 15 | 15 | 15 | 2.2% |

The binding-only baseline dominates on gold-standard recovery because all 15 epitopes are strong binders by construction. SESTRAV's value proposition is distinguishing immunogenic from non-immunogenic peptides among good binders — the specificity bottleneck — and achieving gold-standard negative discrimination (9/10 negatives pushed down).

## Strain Notes

- **FLRGRAYGI**: B95-8 EBNA3A carries the Ile variant (FLRGRAYGI). The Leu variant (FLRGRAYGL) is from EBV type-2 strains. Both are held out from training.
- **HPVGEADYFEY**: B95-8 EBNA1 (P03211) contains the Glu variant. GD1-strain EBNA1 has HPVGDADYFEY (Asp at p5).
- **GLCTLVAML**: From BMLF1 (SM protein), present in the EBV panel proteome FASTA (`EBV_B95_8_panel8` canonical run ID).
