# SESTRAV: Structural Epitope Scoring via TCR Recognition And Vaccinology

![CI — Contamination Gate](https://img.shields.io/badge/CI-contamination_gate-blue?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Version](https://img.shields.io/badge/version-2.0.0--rc1-informational?style=flat-square)
[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/PENDING_ID/badge)](https://bestpractices.coreinfrastructure.org/projects/PENDING_ID)

A structurally informed immunogenicity prediction pipeline for therapeutic epitope discovery in oncogenic viruses (HPV and EBV). Version 2.0 centers a canonical reproducible 30-feature release path and includes ANN/GNN/Colab extensions as optional benchmark workflows.

SESTRAV is a structurally informed computational pipeline for therapeutic epitope discovery in oncogenic viruses, with a primary focus on Human Papillomavirus (HPV) and Epstein-Barr Virus (EBV). It integrates MHC binding predictions with TCR-facing physicochemical features to improve immunogenicity classification beyond traditional MHC-binding tools.

The canonical release uses a reproducible 30-feature model (20 physicochemical properties at TCR-contact positions + 10 multi-allele MHC binding scores). Optional extensions include Artificial Neural Network (ANN), Graph Neural Network (GNN), and Google Colab workflows for benchmarking.

## Background and Motivation

Most computational pipelines focus on MHC presentation, predicting whether a peptide is displayed on the cell surface. However, binding affinity alone is a weak proxy for immunogenicity (typical AUC ≈ 0.60 when used directly; Carri et al., 2023). SESTRAV addresses this limitation by extracting features from TCR-contact residues (primarily positions p4–p8) and training classifiers on experimentally validated immunogenicity data from the IEDB.

This approach combines structural insights with multi-allele binding predictions to better discriminate true immunogenic epitopes.

## Release Tracks and Policy

* **Canonical track (default):** 30-feature configuration (20 physicochemical + 10 multi-allele MHC binding). This is the maintained release path.
* **Legacy comparator track:** 21-feature sequence-only configuration retained for historical reproducibility.

**Source of Truth:** SESTRAV v2 designates this repository (main branch) as the single authoritative source. For release-grade reproducibility, enable `freeze_mode: true` in `config.yaml`. Freeze mode enforces strict guardrails: no Stage 4 prototype fallback, no mixed legacy/canonical output stems, and atomic artifact updates.

## Security & Compliance Posture

SESTRAV 2.0 maintains a rigorous security posture suitable for biomedical data pipelines.
*   **SAST & CI:** All commits are gated by Bandit, CodeQL, and Semgrep via GitHub Actions.
*   **Dependency Pinning:** Environment files use strict `--require-hashes` to mitigate supply-chain attacks.
*   **Data Integrity:** The pipeline uses `freeze_mode` constraints to guarantee data immutability during reproducibility benchmarking.

For vulnerability reporting, refer to `SECURITY.md`. For a detailed compliance matrix against OpenSSF standards, see `docs/security_compliance.md`.

## Validation Status

The committed release evidence (v3 dataset, 1004 peptides, 3.35:1 class ratio) provides the following computational validation:

| Metric / Check | Result |
| :--- | :--- |
| H2 Tier A decision (R10 ≥ 2.0) | **Not supported** (R10 = 0.9494) |
| Gold-standard positive recovery | 15/15 positives found; 7/15 in top 25% |
| Binding-only baseline comparison | Baseline recovers 15/15 (expected for strong-binder set) |
| Gold-standard negative discrimination | 9/10 negatives pushed down vs. binding-only |
| SHAP feature contribution | ~60% MHC binding, ~40% TCR-contact features |

**Important:** These results constitute computational validation only and do not establish biological efficacy. Wet-lab experimental confirmation is required for any therapeutic claims. See `results/final_validation_report.md` and `docs/limitations_statement_v1.md` for full details.

## External Benchmark Results (v2.0.0, Frozen)

SESTRAV RF was benchmarked head-to-head against two state-of-the-art tools — PRIME 2.1 (Nielsen & Andreatta, 2021) and PredIG-Path (Farriol-Duran et al., 2025) — on a curated T-cell immunogenicity dataset (EBV + HPV16, MHC class I, IEDB-sourced).

### Tier A Results (N=720 intersection set)

| Tool | AUC-PR | AUC-ROC | ISSR@10 | ISSR@25 |
|------|--------|---------|---------|----------|
| **RF (SESTRAV 2.0)** | **0.828** | **0.776** | — | — |
| PRIME 2.1 | 0.777 | 0.724 | — | — |
| PredIG-Path | 0.727 | — | — | — |

> **Primary metric:** AUC-PR is the primary metric because the dataset is class-imbalanced (positives ≈ 70%).
> AUC-PR baseline (random model) ≈ positive class prevalence.

### Benchmark Overlap and Clean-Holdout Comparison

Because SESTRAV and the comparator tools all draw on IEDB-derived data, a systematic overlap analysis was run to keep the comparison fair. Using exact + substring matching against an IEDB-proxy peptide reference, an estimated **36.9% of the evaluation set overlaps that proxy reference** — i.e. peptides that any IEDB-trained model could plausibly have seen during training. To remove this ambiguity, results are also reported on the overlap-excluded clean holdout:

| Tool | AUC-PR (clean holdout, N=451) | Δ vs. intersection set |
|------|-------------------------------|------------------------|
| **RF (SESTRAV 2.0)** | **0.822** | −0.006 |
| PRIME 2.1 | 0.720 | −0.057 |

The clean holdout is the appropriate rigorous comparator: it removes evaluation peptides that any IEDB-trained tool could have encountered during training, so the comparison does not hinge on the unknown composition of external tools' training sets. Overlap is estimated against an IEDB-proxy reference (the authoritative training sets for the external tools are not publicly released), so these figures should be read as approximate and not as a claim about any specific tool's training data.

### SHAP Feature Attribution

SHAP analysis (Random Forest, 720 samples) attributes the decision to:
- **60%** MHC binding features (per-allele presentation scores)
- **40%** TCR-contact physicochemical features (positions p4–p8)

This 60/40 split confirms that TCR features provide meaningful independent signal beyond binding alone, consistent with the 9/10 gold-standard negative discrimination result.

See [`results/external_benchmark_comparison.md`](results/external_benchmark_comparison.md) for full methodology and [`results/shap_values_rf.csv`](results/shap_values_rf.csv) for per-feature SHAP values.

## Pipeline Overview

SESTRAV proceeds through four stages:

1. **Peptide Generation:** Sliding-window extraction of 8-11mer peptides from viral proteome FASTA files.
2. **MHC Binding Prediction:** MHCflurry presentation scores across 10 common HLA alleles.
3. **TCR Feature Extraction:** 20 physicochemical properties at TCR-facing positions plus 10 binding scores (30 features total).
4. **Immunogenicity Scoring:** Ensemble classification (RF, XGBoost) with optional ANN/GNN benchmarks; produces ranked candidates with SHAP interpretability.

**Input:** Viral proteome FASTA files (default: 8 HPV + 8 EBV antigens).
**Output:** Ranked epitope candidates with immunogenicity scores, SHAP values, and visualizations.

## Input Data and Naming Conventions

SESTRAV runs on bundled repository data by default. User-uploaded files are unnecessary unless intentionally overriding defaults.

### Proteome Identifiers

| Proteome ID | Virus | Strain(s) | Antigens | FASTA File |
| :--- | :--- | :--- | :--- | :--- |
| `HPV16_18_panel8` | Human Papillomavirus | HPV-16, HPV-18 | 8 (E2, E5, E6, E7 from each strain) | `data/proteomes/HPV16_18_panel8.fasta` |
| `EBV_B95_8_panel8` | Epstein-Barr Virus | B95-8 | 8 (EBNA1, EBNA3A, EBNA3B, LMP1, LMP2A, gp350, BZLF1, BRLF1) | `data/proteomes/EBV_B95_8_panel8.fasta` |

*Full UniProt accessions are available in `docs/antigen_accessions.md`.*

### Output File Naming

Per-proteome outputs follow the pattern `results/{proteome_id}_{suffix}`:

| Suffix | Contents |
| :--- | :--- |
| `_peptides.csv` | All 8-11mer peptides (Stage 1) |
| `_binding.csv` | MHCflurry presentation scores (Stage 2) |
| `_features.csv` | 30 features per peptide (Stage 3) |
| `_ranked.csv` | Final scored and ranked epitope candidates (Stage 4) |
| `_top20_immunogenicity.png` | Bar chart of top 20 predicted immunogenic peptides |
| `_score_distribution.png` | Histogram of score distribution across all peptides |

Validation and analysis outputs (committed) are summarized in `results/`; see the repository for the complete list.

## Feature Schemas

At each TCR contact position, SESTRAV computes the following physicochemical properties. Unless otherwise referenced, properties are based on canonical amino acid physicochemical classifications.

| Property | Scale / Definition | Source |
| :--- | :--- | :--- |
| Hydrophobicity | Kyte-Doolittle (-4.5 to +4.5) | Kyte & Doolittle, 1982 |
| Aromaticity | Binary (F, W, Y, H = 1) | Canonical |
| Van der Waals volume | Å³ | Zamyatnin, 1972 |
| Charge at pH 7 | K/R = +1, D/E = -1, others = 0 | Canonical |
| Flexibility | Vihinen flexibility (0.904 - 1.102) | Vihinen et al., 1994 |
| Bulkiness | Zimmerman bulkiness (3.4 - 21.67) | Zimmerman et al., 1968 |
| Hydrophilicity | Hopp-Woods (-3.4 to 3.0) | Hopp & Woods, 1981 |
| TCR upward probability | Heuristic derived from structural alignments | Internal structural mapping |

### Track Definitions

| Track | Features | Use Case |
| :--- | :--- | :--- |
| Canonical (30-feature) | 20 physicochemical + 10 binding | Default release track |
| Legacy (21-feature) | Sequence-only (binding excluded) | Historical comparator |
| Expanded (50-feature) | 40 physicochemical + 10 binding | Extended evaluation |
| Allele-aware (166) | Canonical + 136 HLA pocket pseudo-sequences | Pan-allele modeling |

Stage 4 auto-detects the appropriate feature set for each trained model.

## Biological Data Limitations & Mitigation

The input training data for SESTRAV contains severe biological biases inherent to public datasets (like IEDB). A quantitative breakdown of these taxonomic and topological skews is detailed in the [data_bias_audit_v3.md](docs/data_bias_audit_v3.md) report.

* **Taxonomic skew:** EBV 68.13%, HPV16 30.88%, HPV11 1.00%.
* **Length skew:** 9-mer peptides 64.74%.

To prevent machine learning models from over-indexing on EBV-specific anchor motifs and 9-mer length preferences (which would lead to poor generalization on minority taxa like HPV11 or non-canonical peptide lengths), the `compute_sample_weights()` function in [features.py](src/features.py) is **CRITICAL**. It dynamically calculates sample weights to up-weight minority taxa and non-9-mer peptides during model training, balancing the learning signal and ensuring robust pan-viral performance.

## Quick Start

### 1. Environment Setup (Conda recommended)

```bash
conda env create -f environment.yml
conda activate sestrav
mhcflurry-downloads fetch models_class1_presentation
```

For a `venv`-based setup:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
pip install -r requirements.txt
pip install snakemake
mhcflurry-downloads fetch models_class1_presentation
```

### 2. Train Models

Models must be trained before production pipeline execution.

```bash
# Canonical 30-feature track
python -m src.train_classifier \
  --data data/immunogenicity_dataset_v3.csv \
  --feature-mode 30 \
  --binding-matrix models/peptide_binding_matrix_v3.csv

# Legacy comparator track
python -m src.train_classifier \
  --data data/immunogenicity_dataset_v3.csv \
  --feature-mode 21
```

*Note:* Without trained models, the pipeline falls back to a prototype mode using binding-derived pseudo-labels (for testing only; not scientifically valid).

### 3. Run the Pipeline

```bash
# Snakemake (recommended)
snakemake --snakefile pipeline.smk --cores 4

# Standalone entry point
python pipeline.py
```

### 4. Validate Release Readiness

```bash
git status
python -m pytest tests/test_features.py tests/test_metrics.py tests/test_pipeline_integration.py -q
snakemake --snakefile pipeline.smk --dry-run --cores 1   # optional
```

For freeze-grade validation:

```bash
snakemake --snakefile pipeline.smk full_validation_report --cores 4 --forceall
```

### 5. Post-Pipeline Analysis (Optional)

Generate full validation report:

```bash
snakemake --snakefile pipeline.smk full_validation_report --cores 4
```

Prepare inputs for external tool comparison (PredIG, PRIME):

```bash
python -m src.prepare_external_validation_inputs
python -m src.external_benchmark_comparison --predig ... --prime ...
```

See `scripts/README.md` for the external-validation utilities and workflow.

### 6. ANN / GNN Benchmarks (Optional)

* **ANN:** `pip install -r requirements-ann.txt`, then `python -m src.ann_benchmark --help`.
Default architecture: 256-128-64 ReLU, dropout 0.2 (AUC-PR = 0.8252 ± 0.0248).
* **GNN:** `pip install -r requirements-gnn.txt`, then `python -m src.gnn_benchmark --help`.
Implements GCN, GAT, and Bipartite GNN on peptide backbone graphs.

### 7. Google Colab

A Colab-ready script is available in `notebooks/SESTRAV_Colab_Pipeline.py`; see `notebooks/README.md` for details.

## Container Quick Start

The Docker image does **not** include trained models. Build and then train:

```bash
docker build -t sestrav:latest .
docker run --rm -v "$(pwd)/models:/app/models" sestrav:latest \
  -m src.train_classifier --data data/immunogenicity_dataset_v3.csv \
  --feature-mode 30 --binding-matrix models/peptide_binding_matrix_v3.csv
```

Run the pipeline with bind-mounted directories:

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
docker run --rm -v "$(pwd)/data:/app/data:ro" sestrav:latest -m pytest tests/ -q --basetemp=tmp_pytest
```


### API & Demo Quick Start (Docker Compose)

A two-service Docker Compose stack serves the FastAPI microservice and Streamlit demo
from pre-trained model artifacts. Model binaries must be present in `models/` before
launching.

```bash
# Build and launch both services
docker compose up --build

# FastAPI docs:  http://localhost:8000/docs
# Streamlit demo: http://localhost:8501
```

Single-peptide API request:

```bash
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{"sequence":"GILGFVFTL","allele":"HLA-A*02:01"}'
```

Both services bind to `127.0.0.1` only (loopback) to prevent unintended public
exposure on shared research machines.  Model artifacts are mounted read-only.



| Virus | Antigens |
|---|---|
| EBV (8) | EBNA1, EBNA3A, EBNA3B, LMP1, LMP2A, gp350, BZLF1, BRLF1 |
| HPV (8) | HPV16 E2, E5, E6, E7; HPV18 E2, E5, E6, E7 |

## Evaluation Metrics

All metrics are computed by `src/evaluate_metrics.py`.

| Metric | Description |
| --- | --- |
| AUC-PR | Area Under Precision-Recall Curve (primary metric, robust to class imbalance) |
| AUC-ROC | Area Under ROC Curve |
| ISSR@10/25 | True positive fraction in top 10% / 25% (enrichment) |
| Precision@10/25 | Precision among top 10% / 25% predictions |
| Recall@10/25 | Recall captured in top 10% / 25% |
| NDCG@10/25 | Normalized Discounted Cumulative Gain at top 10% / 25% |

## Reproducibility and Data Provenance

**Included in this repository:**

* Training dataset (`data/immunogenicity_dataset_v3.csv`)
* Viral proteomes (`data/proteomes/`)
* Binding matrix (`models/peptide_binding_matrix_v3.csv`) and model metadata
* All pipeline code, tests, and documentation

**Generated locally (excluded from git):**

* Trained model binaries (`*.joblib`, `*.pt`)
* Most workflow outputs in `results/` (except committed validation snapshots)
* Runtime caches

A fresh clone must run model training before production scoring. Release bundles with SHA256 manifests can be created via `python -m src.release_bundle`.

Training labels are derived from curated IEDB-linked immunogenicity evidence. Publications should cite both this repository and the original upstream data sources.

## Documentation

| Document | Description |
| --- | --- |
| `docs/feature_glossary.md` | Feature definitions and track schemas |
| `docs/antigen_accessions.md` | Full UniProt accessions and gene names |
| `docs/output_naming_standard_v1.md` | Output file naming policy |
| `docs/naming_migration_spec.md` | Legacy alias compatibility details |
| `docs/validation_summary.md` | Detailed validation results and interpretation |
| `docs/limitations_statement_v1.md` | Standardized external communication language |

## Cite This Work

If you use SESTRAV in your research, please cite this repository:

```bibtex
@software{borges2026sestrav,
  author    = {Borges, Gavin and Eljamal, Abdelrahman and Schellenberg, Iris and
               Jouaneh, Charles and Byers, Emine},
  title     = {{SESTRAV}: Structural Epitope Scoring via {TCR} Recognition And Vaccinology},
  year      = {2026},
  url       = {https://github.com/gavin-borges/SESTRAV},
  version   = {2.0.0-rc1}
}
```

See [`CITATION.cff`](CITATION.cff) for the full machine-readable citation.

## License

MIT License. See `LICENSE` for details.

## Maintainers and Contributors

**Lead Developer & Maintainer (SESTRAV 2.0)**

* Gavin Borges

**Original SESTRAV 1.0 Foundation Team (University of Rhode Island)**

* Abdelrahman Eljamal: ML Engineer & Computational Architect
* Iris Schellenberg: Translational Vaccine Strategy, Data Finding, and Curation
* Charles Jouaneh: Vaccine Strategy & Bioinformatic Pipeline Development
* Emine Byers: Structural Immunology & Data Curation

*Academic affiliations: BPS 542 / CMB 522 / CSC 522 / STA 522: Bioinformatics I | CMB 523: Bioinformatics II*
