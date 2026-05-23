# SESTRAV Optional ANN/GNN Module Guide

Date: 2026-05-14

This guide documents the optional ANN/GNN benchmark track as an advanced,
reproducible extension of SESTRAV. It is intentionally separate from the
canonical release gate.

## Positioning and Boundaries

- Canonical release gate: Stage 1-4 workflow + `full_validation_report` outputs.
- The optional ANN/GNN benchmark track is supplementary and not part of the canonical publish gate.
- ANN/GNN values are sourced from Project 2 evidence and mirrored in SESTRAV-Dev docs.
- Optional track scope: ANN architecture search, GNN benchmarks, ablation, Colab
  experimentation.
- Promotion to core track is governed by criteria in
  `docs/external_validation_data_expansion_roadmap.md`.

## Provenance

- Primary source report:
  `CMB 523 Injection for SESTRAV Progress/523 Project 2/Project2_Report.md`
- Run metadata source:
  `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/run_metadata.json`
  - `run_id`: `run_20260502_012922_23a21e76`
  - `seed`: `42`
  - Project 2 best config: `256-128-64 ReLU d0.2`

## Exact Benchmark Values (Project 2)

### ANN (30-feature best)

From Project 2 report and bootstrap export:
- AUC-PR (CV mean): `0.8252`
- AUC-ROC (CV mean): `0.6699`
- Bootstrap CI source:
  `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/bootstrap_metric_cis.csv`
  - AUC-PR CI: `[0.7838, 0.8546]`
  - AUC-ROC CI: `[0.6164, 0.7034]`

### GNN

From:
- `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/gnn_sequence_benchmark.csv`
- `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/gnn_bipartite_benchmark.csv`

Exact means:
- GCN (2-layer): AUC-PR `0.7781`, AUC-ROC `0.6138`
- GAT (2-layer, 4-head): AUC-PR `0.7956`, AUC-ROC `0.6366`
- Bipartite peptide-allele: AUC-PR `0.7886`, AUC-ROC `0.6124`

## Runtime and Dependency Matrix

| Track | Entry Point | Dependencies | Output Pattern |
|---|---|---|---|
| Core canonical | `pipeline.smk`, `src/final_validation_report.py` | `requirements.txt` | `results/*` |
| ANN optional | `src/ann_benchmark.py` | `requirements-ann.txt` | `models/ann_*.pt`, `models/ann_architecture_search.csv` |
| GNN optional | `src/gnn_benchmark.py` | `requirements-gnn.txt` | `models/gnn_sequence_benchmark.csv`, `models/gnn_bipartite_benchmark.csv` |
| Colab optional | `notebooks/SESTRAV_Colab_Pipeline.py` | Colab + optional installs in notebook | Colab runtime outputs / exported artifacts |

## Threshold and Calibration Lineage

Threshold/calibration values must be interpreted by model family:

- ANN Colab experiment thresholds:
  `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/thresholds.json`
  - `youden_threshold`: `0.7282`
  - `f1_threshold`: `0.1000`
- ANN Colab calibration:
  `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/calibration_summary.json`
  - selected method: `isotonic`
- RF production threshold file (SESTRAV runtime):
  `models/optimal_thresholds.json`

These are different workflows and should not be mixed in release claims.

## Required Claim Language

- Allowed:
  - “ANN/GNN are optional benchmark modules backported from Project 2.”
  - “Canonical publish gate remains RF/XGBoost Stage 1-4 outputs.”
  - “Project 2 exact values are documented with source file lineage.”
- Not allowed:
  - “ANN/GNN are part of canonical release validation gates.”
  - “Thresholds are universally interchangeable across RF and ANN.”
