# SESTRAV Final Submission and Showcase Checklist

This checklist is for the **current finalized SESTRAV scope** (core semester deliverables only).
It is designed for: (1) internal reliability signoff, (2) GitHub release readiness, and
(3) non-local reproducible execution via container/HPC.

Canonical-selection support docs:
- `docs/canonical_selection_scorecard.md`
- `docs/final_closeout_execution.md`

---

## 1) Reliability and Completeness Gate

### 1.1 Code and tests

- [x] Core tests pass (`python -m pytest tests/ -q`)  
  Last verified: **20 passed**.
- [x] Core scoring and integration modules present:
  - `src/features.py`
  - `src/evaluate_metrics.py`
  - `src/train_classifier.py`
  - `src/gold_standard.py`
  - `src/baseline_comparison.py`
  - `src/h2_tier_a_evaluation.py`
  - `src/final_validation_report.py`

### 1.2 Core outputs (artifact-level)

- [x] H2 Tier A outputs generated:
  - `results/h2_tier_a_fold_metrics.csv`
  - `results/h2_tier_a_summary.csv`
  - `results/h2_tier_a_summary.md`
- [x] Final validation bundle generated:
  - `results/gold_standard_validation.csv`
  - `results/baseline_comparison.csv`
  - `results/final_validation_report.md`
- [x] Gold-standard stage recovery already demonstrated in Unity results.

### 1.3 Known reproducibility note

- [x] Re-ran pipeline + H2/final bundle from one clean forced workflow pass.
- [x] Re-run H2/final bundle in a pinned environment matching model serialization
      (models retrained under sklearn 1.8.0 on 2026-04-24; mismatch resolved).
- [x] Run pipeline/validation in project environment (Python 3.13, sklearn 1.8.0);
      models retrained to match runtime environment.
- [x] H2 Tier A script now emits inferential evidence fields:
      bootstrap 95% CI for `R10` and paired fold sign-flip p-value.
- [x] Generated canonical release artifact bundle locally:
      `python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v1`
      (`release_artifacts/sestrav-v1-20260421T211338Z.manifest.json`,
       `release_artifacts/sestrav-v1-20260421T211338Z.zip`)
- [ ] Upload manifest and zip to GitHub Releases.

---

## 2) Final Runtime Commands (Reference)

## 2.1 Local / Conda

```bash
conda env create -f environment.yml
conda activate sestrav
pip install -r requirements.txt
mhcflurry-downloads fetch models_class1_presentation
python -m src.train_classifier --data immunogenicity_dataset.csv --feature-mode 30 --binding-matrix models/peptide_binding_matrix.csv
python -m src.train_classifier --data immunogenicity_dataset.csv --feature-mode 21
python -m pytest tests/ -v
python -m src.h2_tier_a_evaluation --data immunogenicity_dataset.csv --model-path models/rf_30feature_integrated.joblib --binding-matrix models/peptide_binding_matrix.csv --output-dir results
python -m src.final_validation_report --results-dir results --model-dir models --data immunogenicity_dataset.csv --binding-matrix models/peptide_binding_matrix.csv --model-path models/rf_30feature_integrated.joblib --dataset-mode modeB_updated --dataset-version IEDB-20260424-EBV_HPV16_UPDATED-v2
python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v1
```

## 2.2 Snakemake final chain

```bash
python -m src.train_classifier --data immunogenicity_dataset.csv --feature-mode 30 --binding-matrix models/peptide_binding_matrix.csv
snakemake --snakefile pipeline.smk --cores 4
snakemake --snakefile pipeline.smk full_validation_report --cores 4
```

`full_validation_report` is an explicit target and writes:
`results/final_validation_report.md` and companion CSV/MD artifacts.

## 2.3 Unity/HPC

```bash
sbatch run_pipeline.sh
sbatch run_analysis.sh
```

---

## 3) GitHub Release Readiness

### 3.1 Repository structure and docs

- [x] Runtime docs in `README.md`
- [x] Container definitions: `Dockerfile`, `singularity.def`
- [x] CI workflow: `.github/workflows/ci.yml`
- [x] License file present
- [x] Core docs in `docs/`

### 3.2 Pre-release verification (recommended)

Run before publishing final tag:

```bash
python -m pytest tests/ -v
python -m src.final_validation_report --results-dir results --model-dir models --data immunogenicity_dataset.csv --binding-matrix models/peptide_binding_matrix.csv --model-path models/rf_30feature_integrated.joblib --dataset-mode modeB_updated --dataset-version IEDB-20260424-EBV_HPV16_UPDATED-v2
```

Confirm files exist:

- `results/final_validation_report.md`
- `results/h2_tier_a_summary.csv`
- `results/gold_standard_validation.csv`
- `results/baseline_comparison.csv`
- `release_artifacts/*.manifest.json`
- `release_artifacts/*.zip`

### 3.3 Suggested release flow

```bash
git add -A
git add -f results/final_validation_report.md results/h2_tier_a_summary.csv results/h2_tier_a_summary.md results/h2_tier_a_fold_metrics.csv results/gold_standard_validation.csv results/baseline_comparison.csv
git commit -m "Finalize SESTRAV core pipeline validation and submission artifacts"
git tag -a v1.0.0 -m "SESTRAV semester-final core release"
git push origin HEAD --tags
```

Then create a GitHub Release from `v1.0.0` and include:

- brief project description
- exact run commands
- known environment note (sklearn version for model compatibility)
- release artifact bundle (`*.zip`) and checksum manifest (`*.manifest.json`)
- links to final outputs in `results/` (committed validation snapshots)

---

## 4) Containerized Usability (Not local-only)

Current status note: container definitions are present, but final closeout still requires one verified build/run on the target machine class (local Docker host and/or Unity Apptainer node).

## 4.1 Docker

Build:

```bash
docker build -t sestrav:1.0 .
```

Run pipeline with mounted outputs:

```bash
docker run --rm -v "$(pwd)/results:/app/results" sestrav:1.0 python pipeline.py
```

Run final validation bundle in container:

```bash
docker run --rm -v "$(pwd)/results:/app/results" sestrav:1.0 \
  python -m src.final_validation_report \
  --results-dir results \
  --model-dir models \
  --data immunogenicity_dataset.csv \
  --binding-matrix models/peptide_binding_matrix.csv \
  --model-path models/rf_30feature_integrated.joblib
```

## 4.2 Singularity/Apptainer (Unity-friendly)

```bash
apptainer build sestrav.sif singularity.def
apptainer run sestrav.sif
```

Optional validation command:

```bash
apptainer exec sestrav.sif python -m src.final_validation_report --results-dir results --model-dir models --data immunogenicity_dataset.csv --binding-matrix models/peptide_binding_matrix.csv --model-path models/rf_30feature_integrated.joblib --dataset-mode modeB_updated --dataset-version IEDB-20260424-EBV_HPV16_UPDATED-v2
```

---

## 5) Showcase Script (Demo Sequence)

For a live showcase of usability:

1. Show `README.md` quick-start and architecture.
2. Run `python -m pytest tests/ -v` to prove reliability.
3. Run `python -m src.final_validation_report ...` to generate final outputs.
4. Open:
   - `results/final_validation_report.md`
   - `results/h2_tier_a_summary.md`
   - `results/gold_standard_validation.csv`
5. Briefly show `Dockerfile` and `singularity.def` to demonstrate portability beyond local machine.

This demonstrates SESTRAV as reproducible, test-verified, and runnable in local, container, and HPC settings.
