# SESTRAV-Dev Final Publish Gate Report

Date: 2026-05-14

Canonical pointer: `docs/current_evidence_freeze.md`

This report records the execution of the final publish gate plan (docs audit + runtime rerun + artifact reconciliation) for `SESTRAV-Dev`.

## Final Recommendation

**Pass with caveats** for GitHub publish preparation.

Go/No-Go status by gate:
- Documentation consistency gate: **PASS**
- Optional extension boundary gate (ANN/GNN/Colab): **PASS**
- Runtime reproducibility gate: **PASS WITH CAVEAT**
- Artifact/checksum reconciliation gate: **PASS WITH UPDATE REQUIRED**

## 1) Canonical Truth Matrix

| Gate Item | Source of Truth | Current State |
|---|---|---|
| Canonical feature mode | `config.yaml` (`feature_mode: 30`) | PASS |
| Canonical model path | `config.yaml` (`models/rf_30feature_integrated.joblib`) | PASS |
| Canonical validation artifacts required | `docs/submission_checklist.md`, `docs/github_submission_guide.md` | PASS |
| Release docs command correctness | `README.md`, `docs/github_submission_guide.md` | PASS (fixed docker test command) |
| Biological claim boundary | `README.md` + freeze docs | PASS (computational validation language retained) |

## 2) Optional Track Boundary (ANN/GNN/Colab)

Verified and normalized as optional/exploratory in:
- `README.md`
- `notebooks/README.md`
- `docs/model_evaluation_summary.md`
- `docs/submission_checklist.md`
- `docs/github_submission_guide.md`

Decision: The optional ANN/GNN benchmark track is supplementary and not part of the canonical publish gate.
ANN/GNN values are sourced from Project 2 evidence and mirrored in SESTRAV-Dev docs.

## 3) Runtime Reproducibility Evidence

Executed commands and results:

1. `python -m pytest tests/ -q`  
   - Result: **17 passed, 5 skipped**
2. `python -m src.train_classifier --data immunogenicity_dataset.csv --feature-mode 30 --binding-matrix models/peptide_binding_matrix.csv`  
   - Result: **PASS**; canonical RF/XGB models regenerated
3. `snakemake --snakefile pipeline.smk --cores 4 --forceall`  
   - Result: **PASS**; Stage 1-4 outputs regenerated for both proteomes
4. `snakemake --snakefile pipeline.smk full_validation_report --cores 4 --forceall`  
   - Result: **PASS**; final validation bundle regenerated
5. `python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v2`  
   - Result: **PASS**; new manifest/zip produced (`20260514T181938Z`)

Container checks:
- `docker build -t sestrav:latest .` -> **BLOCKED** (Docker daemon unavailable on this machine)
- `apptainer --version` -> **NOT AVAILABLE** (`apptainer` command not installed)

## 4) Artifact Reconciliation

Required canonical artifact files exist:
- `results/final_validation_report.md`
- `results/h2_tier_a_summary.csv`
- `results/h2_tier_a_summary.md`
- `results/h2_tier_a_fold_metrics.csv`
- `results/gold_standard_validation.csv`
- `results/baseline_comparison.csv`

Key output values after rerun:
- H2 R10: `0.9836`
- H2 R25: `1.0138`
- Decision: **NOT SUPPORTED**
- Gold-standard found: `15/15`
- Gold-standard top 25%: `9/15`

Checksum reconciliation outcome:
- New hashes differ from `docs/colloquium_evidence_freeze_v2.md` (expected after forced rerun and regenerated artifacts).
- New freeze document created: `docs/colloquium_evidence_freeze_v2_20260514.md`
- New release manifest: `release_artifacts/sestrav-v2-20260514T181938Z.manifest.json`

## 5) Required Follow-Ups Before Final Public Release

1. Run container smoke tests on a machine with Docker daemon available (or on target HPC with Apptainer).
2. Use refreshed freeze file (`docs/colloquium_evidence_freeze_v2_20260514.md`) for external claims tied to rerun artifacts.
3. During GitHub Release creation, attach:
   - `release_artifacts/sestrav-v2-20260514T181938Z.zip`
   - `release_artifacts/sestrav-v2-20260514T181938Z.manifest.json`

If step 1 is completed successfully, the gate can be promoted from **Pass with caveats** to **Pass**.
