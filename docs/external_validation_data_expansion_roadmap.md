# External Validation and Data Expansion Roadmap

Date: 2026-05-14

This roadmap defines how SESTRAV should advance the optional ANN/GNN benchmark
track toward fundamental status through external validation, data growth, and
reproducibility gates.

## Current Baseline

- Canonical operational gate: `results/final_validation_report.md` and
  companion artifacts.
- Optional ANN/GNN benchmark track evidence: Project 2 benchmark outputs and backported modules
  in `src/ann_benchmark.py` and `src/gnn_benchmark.py`.
- Current known transfer stress source:
  `CMB 523 Injection for SESTRAV Progress/523 Project 2/Colab_outputs/cross_virus_stress_main.csv`.

## Workstream A: External Validation

**Status (2026-05-20):** Computational Tier A + Tier B **complete**. Sign-off:
`11_External-Testing/External_Validation_Sign_Off.md`. Wet-lab holdout not started.

### A1. Cross-virus transfer replication inside SESTRAV-Dev — **Done**

- Recreate Project 2 transfer experiment format inside Dev docs and scripts.
- Required output schema should match:
  `train,test,auc_roc,auc_pr,issr_10,issr_25,...`
- Minimum artifact:
  `results/external_validation_cross_virus.csv` (delivered in Tier A finalize).

### A2. Tool-to-tool external benchmark comparison — **Done (Tier A)**

- Compare optional ANN/GNN ranking behavior against available non-SESTRAV tools
  and published baselines in reproducible slices.
- Record tool version, input format, filtering assumptions, and evaluation scope.
- Minimum artifact:
  `results/external_benchmark_comparison.md` (PredIG + PRIME + RF OOF, v2.0 MCDA).

### A2b. Tier B proteome face validity — **Done**

- Run `extval_20260520_1750_gb_tierB`: 4000 peptides, 4-tool GS recovery.
- Memo: `11_External-Testing/External_Validation_TierB_Memo.md`.

### A3. Holdout policy hardening

- Maintain strict train/validation/test separation rules and preserve gold
  standard holdout exclusion in all optional experiments.
- Every external experiment must publish dataset version and sample counts.

## Workstream B: Data Expansion

### B1. Dataset version governance

- Every new dataset refresh gets a unique version id in config/docs.
- Preserve previous versions for reproducibility and claims traceability.
- Record positive/negative counts, virus composition, deduplication rules.

### B2. Data quality gates before model reruns

- Enforce QC checks on peptide validity, duplicates, label consistency, and
  assay provenance before ANN/GNN reruns.
- If QC fails, do not update benchmark claims.

### B3. Incremental retraining protocol

- Rerun ANN/GNN benchmark suite on each approved dataset bump.
- Publish deltas against prior frozen optional benchmark table.
- Maintain separate “experimental” and “candidate-for-promotion” statuses.

## Promotion Criteria: Optional -> Fundamental

ANN/GNN can be considered for core SESTRAV promotion only when all are met:

1. **Data maturity**
   - Sufficient expanded training data with stable class ratio and documented QC.
2. **Generalization**
   - Cross-virus and/or external benchmark performance is stable and above
     pre-registered acceptance thresholds.
3. **Reproducibility**
   - Clean-clone reproducibility path (including optional dependencies) is
     documented and verified.
4. **Operational reliability**
   - Optional smoke checks pass in CI or documented equivalent runtime checks.
5. **Documentation readiness**
   - Exact provenance, metrics, calibration/threshold lineage, and caveats are
     frozen in release docs.

## Implementation Order (Recommended)

1. Mirror cross-virus stress outputs into Dev `results/` + docs.
2. Add external benchmark documentation template and first benchmark pass.
3. Add dataset-version governance and QC checklists to docs.
4. Define numeric promotion thresholds and decision rubric.
5. Run a promotion-readiness review against the five criteria above.
