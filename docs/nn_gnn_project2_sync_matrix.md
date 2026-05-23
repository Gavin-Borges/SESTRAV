# SESTRAV ANN/GNN Source-of-Truth Sync Matrix

Date: 2026-05-14

This matrix maps authoritative CMB 523 Project 2 artifacts to their mirrored
code paths and documentation in `SESTRAV-Dev`.

## Scope

- Canonical release gate remains RF/XGBoost and Stage 1-4 focused.
- The optional ANN/GNN benchmark track is supplementary and not part of the canonical publish gate.
- ANN/GNN values are sourced from Project 2 evidence and mirrored in SESTRAV-Dev docs.

## Project 2 -> SESTRAV-Dev Mapping

| Project 2 Artifact | Purpose | SESTRAV-Dev Target | Sync Status |
|---|---|---|---|
| `CMB 523 Injection for SESTRAV Progress/523 Project 2/Project2_Report.md` | Narrative and headline ANN/GNN claims | `README.md`, `docs/model_evaluation_summary.md` | Synced with optional-track framing |
| `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/run_metadata.json` | Repro run id, seed, package context, best config | `docs/nn_gnn_optional_module_guide.md` | Synced |
| `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/gnn_sequence_benchmark.csv` | Exact GCN/GAT benchmark outputs | `docs/model_evaluation_summary.md`, `docs/nn_gnn_optional_module_guide.md` | Synced (rounded in summary, exact values documented in guide) |
| `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/gnn_bipartite_benchmark.csv` | Exact bipartite GNN benchmark outputs | `docs/model_evaluation_summary.md`, `docs/nn_gnn_optional_module_guide.md` | Synced |
| `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/bootstrap_metric_cis.csv` | ANN bootstrap confidence intervals | `docs/nn_gnn_optional_module_guide.md` | Synced |
| `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/calibration_summary.json` | Calibration method and Brier values | `docs/nn_gnn_optional_module_guide.md` | Synced |
| `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/thresholds.json` | ANN threshold values from Colab | `docs/nn_gnn_optional_module_guide.md` | Synced (explicitly separated from RF thresholds) |
| `CMB 523 Injection for SESTRAV Progress/523 Project 2/SESTRAV_exports/feature_config.json` | 30-feature ordering and scaler metadata | `src/ann_benchmark.py`, `src/baseline_comparison.py`, `docs/nn_gnn_optional_module_guide.md` | Synced (column semantics aligned) |
| `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/cross_virus_stress_main.csv` | External-style transfer stress evidence | `docs/external_validation_data_expansion_roadmap.md` | Synced as next-step validation anchor |

## Runtime Alignment Summary

| Concern | Previous Risk | Current State |
|---|---|---|
| ANN baseline import path | `ImmunogenicityMLP` import mismatch in `src/baseline_comparison.py` | Resolved to `FlexibleMLP` from `src/model.py` |
| ANN checkpoint feature mode | Legacy-only ANN path in baseline comparison | Resolved to prefer `ann_30feature_integrated.pt`, fallback to legacy |
| ANN score semantics | Logits used as score in baseline path | Standardized to sigmoid probabilities |
| ANN architecture restoration | Hardcoded model class width | Checkpoint architecture metadata is now respected |

## Policy Notes

- Canonical release policy remains in `docs/canonical_source_of_truth.md`.
- Optional NN/GNN evidence policy is documented in
  `docs/nn_gnn_optional_module_guide.md`.
- Promotion to core track requires external validation, stability, and data
  growth criteria in `docs/external_validation_data_expansion_roadmap.md`.
