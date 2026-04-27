# SESTRAV — Structural Epitope Scoring via TCR Recognition And Vaccinology

A structurally informed immunogenicity prediction pipeline for therapeutic epitope discovery in oncogenic viruses (HPV and EBV).

## What SESTRAV Does

Most computational tools stop at predicting MHC binding — whether a peptide will be presented on the cell surface. But binding is necessary, not sufficient: many peptides bind MHC well yet never activate a T-cell response (AUC ~0.60 when binding is used as an immunogenicity proxy; Carri et al. 2023).

SESTRAV addresses this **specificity bottleneck** by extracting physicochemical features at the peptide positions that face the TCR (positions p4–p8) and combining them with multi-allele MHC binding features (30 total) to train a classifier on experimentally validated immunogenicity data from the IEDB.

## Release Tracks (Canonical + Progress)

SESTRAV is published as a dual-track project:

- **Canonical public track (main):** 30-feature configuration (physicochemical + multi-allele binding features) with `config.yaml` defaults.
- **Legacy comparator track:** 21-feature sequence-only configuration retained for historical comparability with earlier artifacts.

The canonical track is the maintained default path, but committed validation artifacts currently indicate mixed biological support and should be interpreted as exploratory computational evidence rather than finalized biological validation.

## Canonical Source-of-Truth Policy

SESTRAV v1 uses this repository as the single authoritative source and one default execution track.

- Authoritative source: this repository (`main` branch)
- Canonical execution track: 30-feature integrated (`feature_mode: 30`)
- Reference policy: `docs/canonical_source_of_truth.md`

## Current Validation Status

The committed release evidence (v2 dataset, 720 peptides, 2.36:1 class ratio) currently reports:

- `results/final_validation_report.md`: H2 Tier A decision is **NOT SUPPORTED** (`R10 = 0.9836`, below the `>= 2` threshold)
- `results/gold_standard_validation.csv`: 15/15 gold-standard positives found, 9/15 in top 25%
- `results/baseline_comparison.csv`: RF recovers 7/15 in top 10% and 9/15 in top 25%; binding-only baseline recovers 15/15 (expected for strong-binder gold set)
- Gold-standard negative discrimination: 9/10 negatives pushed down vs binding-only (TCR features add value)
- Feature contribution (SHAP): 60% binding / 40% TCR-contact features

Use this repository as a reproducible computational framework and validation workbench; do not frame the current committed outputs as biologically validated endpoints.

## Pipeline Overview

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Stage 1         │    │  Stage 2         │    │  Stage 3         │    │  Stage 4         │
│  Peptide         │───▶│  MHC Binding     │───▶│  TCR Feature     │───▶│  Immunogenicity  │
│  Generation      │    │  Prediction      │    │  Extraction      │    │  Scoring         │
│                  │    │                  │    │                  │    │                  │
│  FASTA → 8-11mer │    │  MHCflurry       │    │  20 physico +    │    │  RF/XGBoost      │
│  sliding window  │    │  10 HLA alleles  │    │  10 allele bind  │    │  classifier      │
│                  │    │                  │    │  = 30 features   │    │  → ranked output │
└──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
```

**Input:** Viral proteome FASTA files (16 antigens: 8 HPV + 8 EBV)

**Output:** Ranked epitope candidates with immunogenicity scores and SHAP interpretability

## Input Data and Naming Conventions

SESTRAV's default execution path runs on data already bundled in this repository:

- `data/proteomes/*.fasta` for Stage 1-4 inference
- `immunogenicity_dataset.csv` and `models/peptide_binding_matrix.csv` for training/evaluation modules
- `config.yaml` for antigen, allele, and model-path selection

This means a local run does not require user-uploaded input files unless you intentionally replace repository data/config.

### Proteome ID Naming

Each pipeline run is identified by a **proteome ID** that encodes the virus, strain, and panel size:

| Proteome ID | Virus | Strain(s) | Antigens | FASTA File |
|---|---|---|---|---|
| `HPV16_18_panel8` | Human Papillomavirus | HPV-16 and HPV-18 | 8 proteins (E2, E5, E6, E7 from each strain) | `data/proteomes/HPV16_18_panel8.fasta` |
| `EBV_B95_8_panel8` | Epstein-Barr Virus | B95-8 | 8 proteins (EBNA1, EBNA3A, EBNA3B, LMP1, LMP2A, gp350, BZLF1, BRLF1) | `data/proteomes/EBV_B95_8_panel8.fasta` |

Reading the ID: `EBV_B95_8_panel8` means "8-antigen panel from the EBV B95-8 strain." The `panel8` segment indicates the antigen count in this project (8 per virus). See [`docs/antigen_accessions.md`](docs/antigen_accessions.md) for full UniProt accessions and gene names.

### Output File Naming

The pipeline produces per-proteome output files using the pattern `results/{proteome_id}_{suffix}`:

| File Suffix | Pipeline Stage | Contents |
|---|---|---|
| `_peptides.csv` | Stage 1 — Peptide Generation | All 8-11mer peptides from sliding window over FASTA input |
| `_binding.csv` | Stage 2 — MHC Binding Prediction | MHCflurry presentation scores for 10 HLA alleles |
| `_features.csv` | Stage 3 — TCR Feature Extraction | 30 features per peptide (20 physicochemical + 10 per-allele binding) |
| `_ranked.csv` | Stage 4 — Immunogenicity Scoring | Final scored and ranked epitope candidates |
| `_top20_immunogenicity.png` | Stage 4 — Plotting | Bar chart of the top 20 predicted immunogenic peptides |
| `_score_distribution.png` | Stage 4 — Plotting | Histogram of immunogenicity score distribution across all peptides |

For example, `results/HPV16_18_panel8_ranked.csv` is the final ranked output for the HPV panel.

### Validation and Analysis Outputs

These committed files are cross-proteome validation summaries (not per-proteome):

| File | Description |
|---|---|
| `results/gold_standard_validation.csv` | Recovery of 15 known immunogenic epitopes |
| `results/gold_standard_negative_validation.csv` | Discrimination of 10 known non-immunogenic epitopes |
| `results/baseline_comparison.csv` | RF vs XGBoost vs binding-only comparison |
| `results/h2_tier_a_summary.csv` | H2 Tier A enrichment analysis |
| `results/final_validation_report.md` | Headline validation decision and artifact links |
| `results/calibration_metrics.csv` | Platt calibration reliability metrics |
| `results/shap_values_rf.csv` | SHAP feature importance values for the RF model |
| `results/multi_run_stability_report.md` | Stability across repeated training/evaluation cycles |

### Shareout Snapshots

Folders named `results/shareout_YYYYMMDD/` (e.g., `results/shareout_20260426/`) are dated presentation snapshots containing plots and summary files frozen for team shareouts. They duplicate a subset of the main results at a specific point in time.

### Model Artifact Naming

Model filenames encode the feature track and algorithm:

| Naming Pattern | Meaning |
|---|---|
| `rf_30feature_integrated.joblib` | Random Forest trained on the canonical 30-feature track (20 physicochemical + 10 binding) |
| `xgb_30feature_integrated.joblib` | XGBoost trained on the same 30-feature track |
| `ann_30feature_integrated.pt` | PyTorch ANN trained on the same 30-feature track |
| `rf_21feature_legacy.joblib` | Random Forest trained on the legacy 21-feature sequence-only track |

Model binaries are not committed to git (they are generated locally via `src/train_classifier.py`). Only model metadata CSVs and configuration JSONs are committed.

### Legacy Alias Compatibility

Earlier versions of this project used different proteome IDs and model names. For one release, the pipeline accepts legacy aliases transparently via `src/naming.py`:

- `HPV_8_FASTAs` and `EBV_8_FASTAs` resolve to their canonical equivalents
- `EBV_panel8_B958` resolves to `EBV_B95_8_panel8`
- Old model names like `rf_30f_immunogenicity.joblib` resolve to `rf_30feature_integrated.joblib`

These aliases will be removed in the next release. Canonical names should be used for all new work.

### Further Reference

- [`docs/output_naming_standard_v1.md`](docs/output_naming_standard_v1.md) — full naming policy for all output files
- [`docs/feature_glossary.md`](docs/feature_glossary.md) — definitions of all 30 feature columns
- [`docs/antigen_accessions.md`](docs/antigen_accessions.md) — protein names, UniProt IDs, and gene identifiers

## Directory Structure

```
.
├── config.yaml            # Central configuration (alleles, antigens, parameters)
├── pipeline.smk           # Snakemake workflow (4 rules)
├── pipeline.py            # Standalone Python entry point
├── functions/             # Core pipeline stage logic
│   ├── stage1_peptide_generation.py
│   ├── stage2_mhc_binding_prediction.py
│   ├── stage3_tcr_feature_extraction.py
│   └── stage4_immunogenicity_scoring.py
├── scripts/               # Snakemake wrapper scripts
│   ├── stage1.py … stage4.py
├── src/                   # Shared modules
│   ├── features.py             # 22/30-feature extraction (Shared Hub)
│   ├── evaluate_metrics.py     # AUC-ROC, AUC-PR, ISSR@10, ISSR@25
│   ├── train_classifier.py     # Offline model training
│   ├── iedb_data_loader.py     # IEDB data cleaning and loading
│   ├── gold_standard.py        # Gold-standard validation (15 epitopes)
│   ├── baseline_comparison.py  # RF vs XGB vs binding-only comparison
│   ├── ann_benchmark.py        # CMB 523 ANN benchmark (optional, requires PyTorch)
│   ├── shap_analysis.py        # SHAP explainability plots
│   └── training_plots.py       # Cross-validation visualization
├── data/proteomes/        # Curated 8:8 antigen FASTA files
├── models/                # Training results CSVs (trained .joblib models are gitignored)
├── results/               # Committed validation snapshots + locally generated outputs
├── tests/                 # Unit and integration tests
├── docs/                  # Technical documentation
├── requirements.txt       # Core Python dependencies
├── requirements-ann.txt   # Additional dependencies for ANN benchmark (includes PyTorch)
├── environment.yml        # Conda environment specification
├── Dockerfile             # Docker container definition
└── singularity.def        # Singularity container definition (HPC)
```

## Feature Schemas

At each TCR contact position, SESTRAV computes physicochemical properties:

| Property | Scale | Source |
|---|---|---|
| Hydrophobicity | Kyte-Doolittle (range -4.5 to +4.5) | Kyte & Doolittle, 1982 |
| Aromaticity | Binary (F, W, Y, H = 1) | Standard biochemistry |
| Van der Waals Volume | Å³ | Zamyatnin, 1972 |
| Charge at pH 7 | K/R = +1, D/E = -1, others = 0 | Standard biochemistry |

Track definitions:

- **Legacy 21-feature track:** sequence-only training features (`binding_score` excluded); maintained as a reproducible comparator.
- **Canonical 30-feature track:** physicochemical + multi-allele binding feature matrix, used as the default release track.

Stage 4 auto-detects model feature expectations and applies compatible columns.

## Quick Start

### 1a. Setup (Conda)
```bash
conda env create -f environment.yml
conda activate sestrav
pip install snakemake
mhcflurry-downloads fetch models_class1_presentation
```

The conda environment installs `requirements.txt` automatically via `environment.yml`.
Use the conda environment (Python 3.11). Python 3.13 environments can fail in `mhcflurry-downloads` due upstream incompatibilities.

### 1b. Setup (venv — alternative)
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
pip install snakemake
mhcflurry-downloads fetch models_class1_presentation
```

### 2. Train Models

Models must be trained before running the pipeline in production mode. The training dataset (`immunogenicity_dataset.csv`) is included in the repository.

```bash
# Canonical release track (default in config.yaml)
python -m src.train_classifier \
  --data immunogenicity_dataset.csv \
  --feature-mode 30 \
  --binding-matrix models/peptide_binding_matrix.csv

# Legacy comparator track (used in baseline comparison reports)
python -m src.train_classifier \
  --data immunogenicity_dataset.csv \
  --feature-mode 21
```

This produces model artifacts in `models/` for both tracks using 5-fold stratified cross-validation. Gold-standard epitopes are automatically held out from training.

Model binaries (`*.joblib`, `*.pt`) are intentionally excluded from git. A fresh clone must run training before canonical pipeline scoring/validation.

> **Note:** Without trained models, the pipeline falls back to a prototype mode that uses binding-derived pseudo-labels. This mode exists for end-to-end testing only and is **not scientifically valid**.

### 3. Run the Pipeline

```bash
# Via Snakemake (recommended)
snakemake --snakefile pipeline.smk --cores 4

# Standalone
python pipeline.py
```

The default Snakemake target runs Stages 1-4 and plotting outputs only. This keeps the fresh-clone canonical workflow reproducible without requiring legacy comparator binaries.

`snakemake` is installed separately from `requirements.txt` to avoid dependency conflicts with TensorFlow-backed `mhcflurry`.

### 4. Run Tests
```bash
python -m pytest tests/ -v

# Or run individually
python tests/test_features.py
python tests/test_metrics.py
python tests/test_pipeline_integration.py
```

### 5. Validate Release Readiness (recommended before GitHub commit)

Run this exact sequence before committing:

```bash
# 1) Confirm your staged/unstaged Git state
git status

# 2) Validate pipeline logic
python -m pytest tests/test_features.py tests/test_metrics.py tests/test_pipeline_integration.py -q

# 3) (Optional) Validate workflow rule wiring
snakemake --snakefile pipeline.smk --dry-run --cores 1
```

Sanity-check that these files exist and remain accurate in your commit:

- `README.md`: setup, run, and validation commands that match the current code.
- `requirements.txt`: all Python packages needed for pipeline + tests.
- `requirements-ann.txt`: only optional ANN benchmark extras.
- `environment.yml`: conda bootstrap that installs `requirements.txt`.
- `config.yaml`: canonical defaults (antigens, alleles, feature mode, model path).
- `pipeline.smk`: source-of-truth stage wiring and expected outputs.

### 6. Post-Pipeline Analysis (optional)

After the pipeline completes, run gold-standard validation, baseline comparison, and SHAP analysis:

```bash
python run_analysis.py
```

To generate the full validation bundle explicitly (gold-standard + baseline comparison + H2 Tier A):

```bash
# requires locally generated model binaries (see "Train Models" above)
snakemake --snakefile pipeline.smk full_validation_report --cores 4
```

### 7. ANN Benchmark (optional, requires PyTorch)

The CMB 523 ANN benchmark requires PyTorch. Install additional dependencies:

```bash
pip install -r requirements-ann.txt
python -m src.ann_benchmark --data immunogenicity_dataset.csv
```

### 8. Build a release artifact bundle (recommended)

```bash
python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v1
```

This command creates:

- a SHA256 checksum manifest (`*.manifest.json`) for canonical validation outputs
- a zipped bundle (`*.zip`) suitable for GitHub Release assets

## Public Data and Reproducibility

SESTRAV is released so a new user can reproduce the canonical Stage 1-4 workflow from a fresh clone.

Included in this repository:

- `immunogenicity_dataset.csv` (training dataset used by the released classifiers)
- `immunogenicity_dataset_v1_archived.csv` and `immunogenicity_dataset_v2.csv` (historical dataset versions retained for provenance; not used by any code path)
- `data/proteomes/*.fasta` (EBV/HPV proteome inputs)
- `models/peptide_binding_matrix.csv` and model metadata/report CSV/JSON files
- Pipeline code, tests, and docs needed to regenerate results

Generated locally by each user (not committed):

- Trained model binaries (`models/*.joblib`, `models/*.pkl`, `models/*.pt`)
- Most workflow outputs in `results/` (except committed validation snapshots)
- Runtime caches (`.snakemake/`, `.pytest_cache/`)

Reproducibility note:

- The default release path is the canonical 30-feature track.
- A fresh clone should run canonical model training before production scoring.
- If you need strict model compatibility, run with package versions in `requirements.txt`.
- For strict verification of a specific run, publish the output bundle generated by `src.release_bundle`.

Data provenance and citation:

- The training labels are derived from curated immunogenicity evidence prepared from IEDB-linked records in this project workflow.
- Publications or downstream reuse should cite both the SESTRAV repository and original upstream data sources used to build the released dataset.

Validation scope:

- This repository provides computational validation evidence (CV metrics, H2 Tier A analysis, and gold-standard recovery).
- Wet-lab biological validation is not included in this release and should be described as future work.
- Standardized external-communication language is maintained in `docs/limitations_statement_v1.md`.

Naming references for collaborators:

- `docs/naming_migration_spec.md`
- `docs/output_naming_standard_v1.md`
- `docs/naming_surface_audit_20260424.md`

## Container Quick Start

> **Important:** The Docker image does not include trained model binaries (`*.joblib`, `*.pt`) or Snakemake. Without models, the pipeline falls back to a prototype mode with binding-derived pseudo-labels that is **not scientifically valid**. You must either train models inside the container or bind-mount a host directory containing pre-trained models.

Build image:

```bash
docker build -t sestrav:latest .
```

Train models inside the container (required before production scoring):

```bash
docker run --rm -v "$(pwd)/models:/app/models" sestrav:latest \
  -m src.train_classifier --data immunogenicity_dataset.csv \
  --feature-mode 30 --binding-matrix models/peptide_binding_matrix.csv
```

Run pipeline with bind-mounted host output and model directories:

Linux/macOS:

```bash
mkdir -p results
docker run --rm \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/results:/app/results" \
  sestrav:latest
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force results | Out-Null
docker run --rm `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/results:/app/results" `
  sestrav:latest
```

Container smoke test (recommended before release):

```bash
docker run --rm sestrav:latest -m pytest tests/ -q
```

## Antigens

| Virus | Antigens |
|---|---|
| EBV (8) | EBNA1, EBNA3A, EBNA3B, LMP1, LMP2A, gp350, BZLF1, BRLF1 |
| HPV (8) | HPV16 E2, E5, E6, E7; HPV18 E2, E5, E6, E7 |

## Evaluation Metrics

| Metric | Description | Role |
|---|---|---|
| AUC-PR | Area Under Precision-Recall Curve | **Primary metric** (imbalanced data) |
| AUC-ROC | Area Under ROC Curve | Overall discrimination |
| ISSR@10 | True positive fraction in top 10% | Enrichment (PredIG convention) |
| ISSR@25 | True positive fraction in top 25% | Enrichment |

## License

MIT License — see [LICENSE](LICENSE) for details.

## Team

- **Gavin Borges** — Scientific Integrator & ML/Computational Assistance
- **Abdelrahman Eljamal** — ML Engineer & Computational Architect
- **Iris Schellenberg** — Translational Vaccine Strategy
- **Charles Jouaneh** — Vaccine Strategy & Bioinformatic Pipeline Development
- **Emine Byers** — Structural Immunology & Data Curation

University of Rhode Island — BPS 542 / CMB 522 / CSC 522 / STA 522: Bioinformatics I
