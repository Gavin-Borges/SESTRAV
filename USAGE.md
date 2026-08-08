# SESTRAV Usage Guide

Quick-start reference for the `sestrav` command-line interface.

## Installation

```bash
# Core install from source (not yet published to PyPI)
git clone https://github.com/Gavin-Borges/SESTRAV.git
cd SESTRAV
pip install .

# With GNN structural scorer
pip install ".[gnn]"

# With Snakemake pipeline runner
pip install ".[pipeline]"

# With Streamlit demo app (optional expansion panels)
pip install ".[demo]"

# Developer install (includes ruff, mypy, pytest)
pip install -e ".[dev]"
```

After installing, download the MHCflurry presentation models (required for binding prediction):
```bash
mhcflurry-downloads fetch models_class1_presentation
```

---

## Five-Command Quickstart

### 1. Check your environment

```bash
sestrav info
```

Expected output:
```
============================================================
SESTRAV Environment Info
============================================================
  sestrav version : 2.1.0
  mhcflurry       : 2.2.1
  torch           : 2.2.0
  CUDA            : not available
  feature_mode    : 31
  model_path      : models/rf_31feature_integrated.joblib
  mhcflurry_pin   : 2.2.1
  allele panel    : 10 alleles (HLA-A*02:01, HLA-A*01:01, HLA-A*03:01...)
  viruses/panels  : HPV16_18_panel8, EBV_B95_8_panel8, HBV_ayw_panel4, HCV_1a_panel4
============================================================
```

### 2. Score a viral proteome

```bash
sestrav predict \
  --fasta data/proteomes/EBV_B95_8_panel8.fasta \
  --model models/rf_31feature_integrated.joblib \
  --output results/ebv_run/
```

This generates `results/ebv_run/EBV_B95_8_panel8_ranked.csv` with columns:
`peptide, protein_id, immunogenicity_score, rank, allele, presentation_score`

To restrict to specific alleles or lengths:
```bash
sestrav predict \
  --fasta data/proteomes/EBV_B95_8_panel8.fasta \
  --model models/rf_31feature_integrated.joblib \
  --alleles HLA-A*02:01 HLA-B*07:02 \
  --lengths 9 10 \
  --output results/ebv_a02_run/
```

### 3. Validate on labeled data (cross-validation)

```bash
sestrav validate \
  --dataset data/immunogenicity_dataset_v5.csv \
  --model-dir models/local \
  --feature-mode 31 \
  --binding-matrix models/peptide_binding_matrix_v5.csv \
  --sample-weights \
  --report results/validation_report_v5.md
```

`--model-dir` is required and has no default: `sestrav validate` retrains, so it writes
the same artifact set as `python -m src.train_classifier` and needs the same deliberate
destination. `models/local/` is gitignored. A run aborts before training if it would
replace artifacts already present in the target directory; add `--allow-overwrite` when
replacing them is the intent.

Expected per-virus within-CV mean AUC-ROC ~ 0.751 on the v5 dataset (35,597 active rows /
51,185 total; the canonical same-pathogen discrimination metric, `results/per_virus_eval_v5_mode31.csv`).
The pooled AUC-PR is a base-rate artifact and is not reported as a headline; see
`docs/model_evaluation_summary.md`.

### 4. Benchmark against gold standard

```bash
sestrav benchmark \
  --predictions results/ebv_run/EBV_B95_8_panel8_ranked.csv \
  --output results/benchmark_report.md
```

Compares your ranked output to the SESTRAV gold-standard epitope list and reports
AUC-PR, AUC-ROC, ISSR@10, and ISSR@25.

### 5. Run the full governed pipeline (Snakemake)

For release-grade, reproducible runs that enforce checksums and freeze_mode:
```bash
snakemake --snakefile pipeline.smk --cores 4
```

The CLI (`sestrav predict`) is for interactive/exploratory use. For publication-quality
results, use the Snakemake pipeline with `freeze_mode: true` in `config.yaml`.

---

## All Subcommands

### `sestrav info`

No arguments required. Prints environment summary and warns on version mismatches.

### `sestrav predict`

| Argument | Required | Description |
|----------|----------|-------------|
| `--fasta` | Yes | Path to FASTA proteome file |
| `--model` | Yes | Path to trained `.joblib` model |
| `--output` | No | Output directory (default: `results/cli_predict/`) |
| `--alleles` | No | HLA alleles, space-separated (default: 10-allele panel) |
| `--lengths` | No | Peptide lengths (default: 8 9 10 11) |
| `--feature-mode` | No | Feature mode override (default: from `config.yaml`) |

### `sestrav validate`

| Argument | Required | Description |
|----------|----------|-------------|
| `--dataset` | Yes | Path to labeled immunogenicity CSV |
| `--binding-matrix` | Cond. | Required for `--feature-mode 30` or `31` |
| `--folds` | No | CV folds (default: 5) |
| `--feature-mode` | No | Feature mode override |
| `--sample-weights` | No | Apply EBV/HPV16 bias-correction weights |
| `--report` | No | Path to write markdown report |
| `--model-dir` | Yes | Directory to write trained models and metrics. No default: use a scratch directory such as `models/local`, and pass `models` only when replacing the published artifacts |
| `--allow-overwrite` | No | Replace training artifacts already present in `--model-dir` (without it the run aborts before training) |

### `sestrav benchmark`

| Argument | Required | Description |
|----------|----------|-------------|
| `--predictions` | Yes | Path to ranked predictions CSV |
| `--score-column` | No | Score column name (auto-detected if omitted) |
| `--output` | No | Path to write markdown benchmark report |

---

## Reproducing the Published Results

The exact published benchmark (AUC-PR 0.828, OOF RF, frozen 2026-05-20) uses:

```bash
# 1. Run full Snakemake pipeline with freeze_mode
snakemake --snakefile pipeline.smk --cores 4 --config freeze_mode=true

# 2. Reproduce the external benchmark comparison
python scripts/benchmark_runner.py --tier A --run-id reproduce_v3 --skip-freeze-check
```

See `docs/external_testing/External_Validation_Sign_Off.md` for the contamination
analysis and `docs/model_evaluation_summary.md` for all benchmark results.

> **Evaluation note:** SESTRAV RF is evaluated strictly out-of-fold (conservative estimate),
> while external tools are fully scored on the same peptides. On the certified Tier A
> head-to-head (`results/table3_tier_a_metrics.csv`), SESTRAV RF (AUC-PR 0.828) posts the highest point AUC-PR - a statistical near-tie with BigMHC
> (0.822), the MHCflurry binding-only baseline (0.800), MixMHCpred 2.2 (0.795), and
> DeepImmuno (0.698). Because SESTRAV is scored out-of-fold while the external tools score
> every peptide directly, the comparison is conservative by construction for SESTRAV.

---

## Configuration

Key settings in `config.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `feature_mode` | `31` | Feature tier: 21 (legacy), 30 (legacy), 31 (canonical), 33 (extended) |
| `model_path` | `models/rf_31feature_integrated.joblib` | Production model path |
| `mhcflurry_model_version` | `2.2.1` | Pinned MHCflurry version (do not change without rebuilding dataset) |
| `freeze_mode` | `true` | Enforce strict checksums and governance guardrails |
| `alleles` | 10-allele panel | HLA alleles for binding prediction |
| `peptide_lengths` | `[8, 9, 10, 11]` | Peptide lengths for generation |
| `antigen_processing_cache_path` | `null` | Cache CSV for feature_mode=33 (built by `scripts/precompute_antigen_processing.py`) |
