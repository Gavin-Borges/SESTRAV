# SESTRAV Data Directory

This directory contains all input datasets, reference sequences, and archives for the SESTRAV pipeline.

---

## Primary Training Dataset

| File | Version | Records | Description |
|------|---------|---------|-------------|
| `immunogenicity_dataset_v3.csv` | v2.0.0-alpha | 720 peptides (506 pos / 214 neg) | **Canonical training dataset.** Used by all training, evaluation, and validation modules. Source: curated IEDB records (IEDB-20260521-EXPANDED-v2.1). SHA256 checksum in `config.yaml` under `dataset_governance`. |

> **This is the file all pipeline and training commands should reference.** The root-level `immunogenicity_dataset.csv` has been removed; the canonical path is `data/immunogenicity_dataset_v3.csv`.

### Dataset Governance

Version governance metadata (version, provenance timestamp, source databases, checksum, QC thresholds) is embedded in `config.yaml` under the `dataset_governance` key. In `freeze_mode: true`, the pipeline enforces a SHA256 checksum match against this file before proceeding.

### Dataset Version History

| Version | File | Records | Notes |
|---------|------|---------|-------|
| v1 (archived) | `archive/immunogenicity_dataset_v1_archived.csv` | ~407 peptides | Original dataset; 5.58:1 class ratio |
| v2 (archived) | `archive/immunogenicity_dataset_v2.csv` | ~720 peptides | Intermediate expansion; superseded |
| v2.0.0-alpha (canonical) | `immunogenicity_dataset_v3.csv` | 720 peptides | Current canonical; 2.36:1 class ratio |

---

## Archive

`archive/` contains previous dataset versions retained for provenance and reproducibility:

| File | Description |
|------|-------------|
| `immunogenicity_dataset_v1_archived.csv` | v1 dataset (5.58:1 class ratio; BPS 542/CMB 522 original) |
| `immunogenicity_dataset_v2.csv` | v2 intermediate dataset (superseded by v3) |

---

## Allele-Aware Subset

`allele_aware/` contains a pan-allele training subset with HLA pseudo-sequence features:

| File | Description |
|------|-------------|
| `allele_aware_training_data.csv` | Pan-allele training data with per-allele binding features |
| `allele_mapping.json` | HLA allele to pseudo-sequence pocket residue mapping |
| `provenance.json` | Generation provenance for the allele-aware subset |

---

## Consensus Sequences

`consensus_sequences/` contains reference protein sequences used for binding matrix generation:

| File | Description |
|------|-------------|
| `EBV_*.fasta` | EBV protein consensus sequences (EBNA1, LMP1, etc.) |
| `HPV16_*.fasta` | HPV-16 protein consensus sequences (E2, E5, E6, E7) |
| `HPV18_*.fasta` | HPV-18 protein consensus sequences (E2, E5, E6, E7) |

These are used by `src/generate_binding_matrix.py` to pre-compute the peptide binding matrix (`models/peptide_binding_matrix_v3.csv`).

---

## External Reference Data

`external/` contains training datasets from external tools used for benchmarking context:

| File | Description |
|------|-------------|
| `prime_train_peptides.csv` | PRIME training peptides (used for overlap detection) |
| `predig_reference_data.csv` | PredIG reference scoring data |

These are used to detect training data overlap when computing fairness-adjusted external validation metrics.

---

## Proteome FASTA Files

`proteomes/` contains the curated 8-antigen panel FASTA files that are the primary input for the Stage 1–4 pipeline:

| File | Proteome ID | Antigens |
|------|-------------|---------|
| `HPV16_18_panel8.fasta` | `HPV16_18_panel8` | HPV-16 and HPV-18: E2, E5, E6, E7 (4 per strain = 8 total) |
| `EBV_B95_8_panel8.fasta` | `EBV_B95_8_panel8` | EBV B95-8: EBNA1, EBNA3A, EBNA3B, LMP1, LMP2A, gp350, BZLF1, BRLF1 |

> See [`docs/antigen_accessions.md`](../docs/antigen_accessions.md) for full UniProt accession IDs and gene names.
