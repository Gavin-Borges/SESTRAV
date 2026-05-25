# SESTRAV-Dev Final Publish Gate Report

Date: 2026-05-24

Canonical pointer: `docs/current_evidence_freeze.md`

This report records the execution of the final publish gate plan (docs audit + runtime rerun + artifact reconciliation) for `SESTRAV-Dev`.

## Final Recommendation

**Pass** for GitHub publish preparation. All compliance audits, dependency updates, and containerization verification checks have been completed successfully.

Go/No-Go status by gate:
- Documentation consistency gate: **PASS**
- Optional extension boundary gate (ANN/GNN/Colab): **PASS**
- Runtime reproducibility gate: **PASS**
- Artifact/checksum reconciliation gate: **PASS**

## 1) Canonical Truth Matrix

| Gate Item | Source of Truth | Current State |
|---|---|---|
| Canonical feature mode | `config.yaml` (`feature_mode: 30`) | PASS |
| Canonical model path | `config.yaml` (`models/rf_30feature_integrated.joblib`) | PASS |
| Canonical validation artifacts required | `docs/submission_checklist.md`, `docs/github_submission_guide.md` | PASS |
| Release docs command correctness | `README.md`, `docs/github_submission_guide.md`, `docs/submission_checklist.md` | PASS (fixed docker entrypoint syntax) |
| Biological claim boundary | `README.md` + freeze docs | PASS (computational validation language retained) |

## 2) Optional Track Boundary (ANN/GNN/Colab)

Verified and normalized as optional/exploratory in:
- `README.md`
- `notebooks/README.md`
- `docs/model_evaluation_summary.md`
- `docs/submission_checklist.md`
- `docs/github_submission_guide.md`

Decision: The optional ANN/GNN benchmark track is supplementary and not part of the canonical publish gate. ANN/GNN values are sourced from Project 2 evidence and mirrored in SESTRAV-Dev docs.

## 3) Runtime Reproducibility Evidence

Executed commands and results:

1. `python -m pytest tests/ -q`  
   - Result: **27 passed, 6 skipped** (on host)
2. `python -m src.train_classifier --data immunogenicity_dataset.csv --feature-mode 30 --binding-matrix models/peptide_binding_matrix_v3.csv`  
   - Result: **PASS**; canonical RF/XGB models regenerated/verified
3. `snakemake --snakefile pipeline.smk --cores 4`  
   - Result: **PASS**; Stage 1-4 outputs regenerated for both proteomes
4. `snakemake --snakefile pipeline.smk full_validation_report --cores 4`  
   - Result: **PASS**; final validation bundle regenerated
5. `python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v2`  
   - Result: **PASS**; new manifest/zip produced (`20260524T050810Z`)

Container checks:
- `docker build -t sestrav:rc-test .` -> **PASS**
- `docker run --rm -v "${PWD}/data:/app/data:ro" sestrav:rc-test -m pytest tests/ -q --basetemp=tmp_pytest` -> **PASS** (27 passed, 6 skipped; verified container isolation via `--network none`)

## 4) Artifact Reconciliation

Required canonical artifact files exist:
- `results/final_validation_report.md`
- `results/h2_tier_a_summary.csv`
- `results/h2_tier_a_summary.md`
- `results/h2_tier_a_fold_metrics.csv`
- `results/gold_standard_validation.csv`
- `results/baseline_comparison.csv`

Key output values after rerun:
- H2 R10: `0.9494`
- H2 R25: `1.0208`
- Decision: **NOT SUPPORTED**
- Gold-standard found: `15/15`
- Gold-standard top 25%: `7/15`

Checksum reconciliation outcome:
- New freeze document created: `docs/colloquium_evidence_freeze_v2_20260524.md`
- New release manifest: `release_artifacts/sestrav-v2-20260524T050810Z.manifest.json`

## 5) Pre-Release Audits (Vulnerabilities & Security)

Compliance checks executed locally:
1. `bandit -r src/ functions/` -> **PASS** (0 vulnerabilities/findings)
2. `pip-audit -r environments/requirements.lock` -> **PASS** (0 known vulnerabilities found)

## 6) Release Action Items

1. Push all verified code modifications, security rules, and documentation updates to the GitHub repository.
2. During GitHub Release creation under tag `v2.0.0`, attach:
   - `release_artifacts/sestrav-v2-20260524T050810Z.zip`
   - `release_artifacts/sestrav-v2-20260524T050810Z.manifest.json`
