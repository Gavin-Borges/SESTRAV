# SESTRAV Colab Notebooks

This directory contains Google Colab-compatible notebooks for running the SESTRAV pipeline and model training/evaluation experiments without a local installation.

The optional ANN/GNN benchmark track is supplementary and not part of the canonical publish gate.
ANN/GNN values are sourced from Project 2 evidence and mirrored in SESTRAV-Dev docs.
Colab notebooks are convenience workflows for this optional track.

## Available Notebooks

### `SESTRAV_Colab_Pipeline.py`

A complete Colab-ready script that runs the SESTRAV model training and evaluation pipeline directly in Google Colab. Covers:

1. **Setup & Installation** — Installs all required dependencies
2. **Clone & Path Setup** — Clones the SESTRAV repository and configures imports
3. **Data Loading** — Loads the IEDB immunogenicity dataset
4. **Feature Extraction** — Computes both 21-feature (legacy) and 30-feature (canonical) matrices
5. **RF/XGBoost Training** — Trains production tree classifiers with 5-fold CV
6. **ANN Training** — Trains the FlexibleMLP (256-128-64 ReLU d0.2)
7. **Architecture Search** — (Optional) evaluates all 14 Project 2 configurations
8. **GNN Benchmarks** — (Optional) runs GCN, GAT, and Bipartite GNN (requires `torch-geometric`)
9. **Ablation Study** — Evaluates feature group contributions
10. **Model Comparison** — Side-by-side summary of all classifiers
11. **Google Drive Export** — (Optional) saves results to Google Drive

### Usage in Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `SESTRAV_Colab_Pipeline.py` or paste its cells into a new notebook
3. Run cells sequentially — the script handles all installation and setup
4. Results will be saved to the Colab runtime; optionally mount Google Drive for persistence

### Prerequisites

- A Google account with Colab access
- ~15 minutes of GPU time (T4 or better recommended for ANN training)
- ~2GB of disk space for dependencies and MHCflurry models

### Notes

- The script clones the SESTRAV repository directly from GitHub
- All `src/` imports work because the repo root is added to `sys.path`
- GNN benchmarks require `torch-geometric` which may take a few minutes to install
- MHCflurry model download is only needed if running Stage 2 (binding prediction) from scratch
- ANN/GNN sections are optional benchmark extensions; canonical release artifacts are produced from the main repository workflow (`pipeline.smk` + `full_validation_report`)
- Project 2 Colab evidence uses its own package stack (see Project 2 `run_metadata.json`); local SESTRAV-Dev runs can differ slightly by environment while preserving pipeline semantics

See:
- `docs/nn_gnn_project2_sync_matrix.md`
- `docs/nn_gnn_optional_module_guide.md`
