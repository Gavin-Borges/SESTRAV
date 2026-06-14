# SESTRAV Scripts Directory

This directory contains utility scripts for external benchmark orchestration, environment setup, data extraction, and Snakemake stage wrappers. It is **not** a general-purpose scripts bin — each file has a specific role described below.

---

## 1. Snakemake Stage Wrappers

These are called directly by `pipeline.smk` rules and are not meant to be run standalone:

| Script | Snakemake Rule | Purpose |
|--------|---------------|---------|
| `stage1.py` | `generate_peptides` | Sliding-window peptide generation from FASTA |
| `stage2.py` | `predict_binding` | MHCflurry binding prediction |
| `stage3.py` | `extract_features` | TCR physicochemical feature extraction |
| `stage4.py` | `score_immunogenicity` | RF/XGB immunogenicity scoring + plotting |

---

## 2. External Benchmark Orchestration (Main Entry Points)

These are the scripts you run to execute the Tier A / Tier B external validation workflow:

| Script | Platform | Purpose |
|--------|----------|---------|
| `run_external_tier_a.ps1` | Windows PowerShell | Orchestrates full Tier A: PredIG (Docker) + PRIME (WSL2) + comparison |
| `run_external_tier_b.ps1` | Windows PowerShell | Orchestrates full Tier B: proteome-wide PredIG + PRIME |
| `run_prime_tier_a_wsl.sh` | WSL2 / Linux | Runs PRIME tool for Tier A peptide sets |
| `run_prime_tier_b_wsl.sh` | WSL2 / Linux | Runs PRIME tool for Tier B peptide sets (chunked) |
| `run_prime_tier_b_hpv_resume.sh` | WSL2 / Linux | Resume helper for interrupted Tier B HPV chunks |
| `run_predig_batched.py` | Python | Runs PredIG via Docker in 5000-row batches |
| `init_external_run.ps1` | PowerShell | Initializes timestamped run directory under `results/external_tool_outputs/` |

---

## 3. WSL2 / PRIME Environment Setup

Run these **once** before the first external validation run on a new machine:

| Script | Purpose |
|--------|---------|
| `install_prime_wsl.py` | Python-based PRIME installer for WSL2 |
| `install_prime_wsl.sh` | Shell-based PRIME installer for WSL2 |
| `prime_diagnose_wsl.sh` | Diagnoses PRIME installation issues in WSL2 |
| `prime_env_check.sh` | Checks PRIME environment readiness |
| `prime_smoke_tier_b.sh` | Smoke-tests PRIME with a small Tier B subset |

---

## 4. Container / Tool Smoke Tests

Quick one-off health checks:

| Script | Purpose |
|--------|---------|
| `docker_prime_smoke.sh` | Verifies PredIG Docker image pulls and responds |
| `mixmhcpred_smoke.sh` | Smoke-tests MixMHCpred installation |
| `make_predig_smoke.py` | Generates a 50-row smoke input for PredIG |

---

## 5. Data Extraction and Reporting Utilities

Post-pipeline analysis helpers:

| Script | Purpose |
|--------|---------|
| `extract_allele_aware_data.py` | Extracts allele-aware training subset from the dataset |
| `extract_wetlab.py` | Extracts wet-lab-confirmed subset from the dataset |
| `compute_ann_baseline_summary.py` | Summarizes ANN benchmark metrics from model CSVs |
| `generate_baseline_report.py` | Generates HTML/markdown baseline comparison report |
| `regenerate_shareout_pngs.py` | Regenerates all PNGs for `results/shareout_20260426/` |
| `scoring_error_audit.py` | Audits Stage 4 scoring discrepancies |

---

## Notes

- All scripts assume they are run from the **repository root** (not from within `scripts/`).
- Scripts that require WSL2 (`run_prime_*.sh`, `install_prime_wsl.*`) need Windows Subsystem for Linux 2 installed and a working Ubuntu distribution.
- PredIG scripts require Docker Desktop running with the `bsceapm/predig` image pulled.
- For the external validation workflow, use the `run_external_*`, `run_prime_*`, and `run_predig_*` scripts in this directory.
