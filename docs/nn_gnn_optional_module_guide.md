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

**Every source in this section is UNVENDORED and unopenable by any reader.** The whole
`CMB 523 Injection for SESTRAV Progress/` directory is absent from this repository and from
any local workspace (`git ls-files` matches nothing on `CMB`; the path is not on disk). The
identifiers below are recorded as the historical provenance of an external course
deliverable, **not** as artifacts anyone can retrieve or verify.

- Primary source report (UNVENDORED):
  `CMB 523 Injection for SESTRAV Progress/523 Project 2/Project2_Report.md`
- Run metadata source (UNVENDORED):
  `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/run_metadata.json`
  - `run_id`: `run_20260502_012922_23a21e76`
  - `seed`: `42`
  - Project 2 best config: `256-128-64 ReLU d0.2`

## Project 2 Benchmark Values (sources, and which of them exist)

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

**RETRACTED AS UNBOUND, 2026-08-30, on the same grounds as the ANN column above, and with
an added trap.** These figures were sourced to
`CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/gnn_sequence_benchmark.csv`
and `.../gnn_bipartite_benchmark.csv`. That directory is **absent from this repository and
from any local workspace** (`git ls-files` returns nothing matching `CMB`, and the path is
not on disk), so none of these numbers can be reproduced or checked.

Retained struck-through as the record of what was reported:
- ~~GCN (2-layer): AUC-PR `0.7781`, AUC-ROC `0.6138`~~
- ~~GAT (2-layer, 4-head): AUC-PR `0.7956`, AUC-ROC `0.6366`~~
- ~~Bipartite peptide-allele: AUC-PR `0.7886`, AUC-ROC `0.6124`~~

**DO NOT "fix" this by repointing at the identically-named tracked files.** This repository
does track `models/gnn_sequence_benchmark.csv` and `models/gnn_bipartite_benchmark.csv`, and
the obvious repair is to cite those. **Measured 2026-08-30, they hold DIFFERENT values:**

| Model | Retracted figure (AUC-PR / AUC-ROC) | Tracked file (`auc_pr_mean` / `auc_roc_mean`) |
|---|---|---|
| GCN (2-layer) | 0.7781 / 0.6138 | **0.8465 / 0.6535** |
| GAT (2-layer, 4-head) | 0.7956 / 0.6366 | **0.8352 / 0.6088** |
| Bipartite peptide-allele | 0.7886 / 0.6124 | **0.8120 / 0.5611** |

Not one pair matches, and none is a rounding of the other. Substituting them would trade an
unbound number for a **mis-attributed** one under a label the measurement does not belong
to - the D16 failure class, and the identical trap already documented above for the ANN
column, where `models/ann_cv_summary.csv` measures a different architecture on a different
peptide pool. The tracked values are given here so a reader can verify the mismatch, **not
as replacements**. Quote the tracked files under their own provenance or not at all.

**What this repository does track**, under the same two basenames:
`models/gnn_sequence_benchmark.csv` and `models/gnn_bipartite_benchmark.csv`, read by
`scripts/generate_baseline_report.py`. They do not match the three lines above, and they rank GCN
first on both metrics. Neither has a provenance sidecar and neither records a splitter, corpus or
seed, so they are named here as the artifacts a reader can actually open - not as corrected values
for the list above, and not as a basis for ranking any GNN against any RF or ANN figure.

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

Threshold/calibration values must be interpreted by model family. **The two ANN Colab
sources below are UNVENDORED** - part of the same absent `CMB 523 ...` directory - so their
values are recorded as history and cannot be verified by a reader. Only the RF production
threshold file at the end of this section is a tracked artifact.

- ANN Colab experiment thresholds (UNVENDORED):
  `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/thresholds.json`
  - `youden_threshold`: `0.7282`
  - `f1_threshold`: `0.1000`
- ANN Colab calibration (UNVENDORED):
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
