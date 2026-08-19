# SESTRAV Optional ANN/GNN Module Guide

Date: 2026-05-14

This guide documents the optional ANN/GNN benchmark track as an advanced,
reproducible extension of SESTRAV. It is intentionally separate from the
canonical release gate.

## Positioning and Boundaries

- Canonical release gate: Stage 1-4 workflow + `full_validation_report` outputs.
- The optional ANN/GNN benchmark track is supplementary and not part of the canonical publish gate.
- ANN/GNN values are sourced from the benchmark study and mirrored in SESTRAV-Dev docs.
- Optional track scope: ANN architecture search, GNN benchmarks, ablation, Colab
  experimentation.
- Promotion to core track is governed by the project's ANN/GNN promotion criteria.

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

**RETRACTED AS UNBOUND, 2026-08-17.** Every figure previously listed here was sourced to
`CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/bootstrap_metric_cis.csv`.
That path is **absent from this repository and from any local workspace** - it was an external
course deliverable that was never vendored - so none of these numbers can be reproduced, checked,
or ever bound to provenance. They are withdrawn rather than restated:

- ~~AUC-PR (CV mean): `0.8252`~~ - retracted, unbound
- ~~AUC-ROC (CV mean): `0.6699`~~ - retracted, unbound
- ~~AUC-PR bootstrap CI: `[0.7838, 0.8546]`~~ - retracted, unbound
- ~~AUC-ROC bootstrap CI: `[0.6164, 0.7034]`~~ - retracted, unbound

**What this repository can actually show for the ANN.** `models/ann_cv_summary.csv` (tracked)
reports AUC-PR **0.7820 +/- 0.0239** and AUC-ROC **0.6083 +/- 0.0578**, means and population
standard deviations over 5 folds, re-derived from `models/ann_oof_predictions.csv` and reproducing
digit-for-digit. **Read its scope before quoting it:** it measures the legacy **64-32 ReLU dropout
0.3** network, not the 256-128-64 dropout 0.2 the heading above describes, and it covers the 704
peptides left after the 16 `GOLD_STANDARD_EPITOPES` are held out of the 720-peptide corpus in
`results/external_validation_input.csv` (verified by exact set equality). It is a different
architecture on a different pool, so it is **not** a substitute for the retracted figures.

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
| ANN optional | `src/ann_benchmark.py` | `requirements.txt` (`torch` is pinned there) | `<--model-dir>/ann_*.pt`, `<--model-dir>/ann_architecture_search.csv` (`--model-dir` is required and has no default) |
| GNN optional | `src/gnn_benchmark.py` | `requirements.txt` plus the `gnn` extra (`pip install ".[gnn]"`) | `<--output-dir>/gnn_sequence_benchmark.csv`, `<--output-dir>/gnn_bipartite_benchmark.csv` (`--output-dir` is required and has no default) |
| GNN training | `src/train_gnn.py` | `requirements.txt` plus the `gnn` extra | `<--model-dir>/structural_gnn_v2*.pth`, `<--model-dir>/gnn_config*.json`, plus `gnn_oof_predictions*.csv` in the **parent** of `--model-dir` (`--model-dir` is required and has no default) |
| GNN promotion check | `src/verify/promote_gnn.py` | `requirements.txt` plus the `gnn` extra | Scorecard on stdout; `--dry-run` leaves `config.yaml` and `models/model_artifact_checksums.json` unwritten |
| Colab optional | `notebooks/SESTRAV_Colab_Pipeline.py` | Colab + optional installs in notebook | Colab runtime outputs / exported artifacts |

## GNN Training Track: Splitter and OOF Schema

The `src/train_gnn.py` row above is the training track, not the Project 2 benchmark
study, and its cross-validation contract changed on 2026-08-12:

- Folds are **peptide-grouped and composite-stratified**
  (`src.ml_utils.PeptideGroupedKFold` via `build_cv_splits`) at both training entry
  points. The track previously used an ungrouped `StratifiedKFold`, which lets rows
  sharing a peptide fall on both sides of a fold boundary (`docs/claims_register.md`
  D15).
- Each `gnn_oof_predictions*.csv` row is written by `build_oof_records` with the
  schema `peptide,hla_allele,label,gnn_oof_score,fold,splitter`. The former schema
  was `peptide,label,gnn_oof_score`; a three-column frame is therefore a pre-repair
  artifact and is rejected by promotion Gate 1.
- The Project 2 GCN/GAT/bipartite figures quoted above come from the separate
  benchmark study, not from this training track, and carry no splitter provenance of
  this kind. Do not compare them with peptide-grouped numbers.

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
  - "ANN/GNN are optional benchmark modules from the ANN/GNN benchmark study."
  - "Canonical publish gate remains RF/XGBoost Stage 1-4 outputs."
  - "Project 2 exact values are documented with source file lineage."
- Not allowed:
  - "ANN/GNN are part of canonical release validation gates."
  - "Thresholds are universally interchangeable across RF and ANN."
