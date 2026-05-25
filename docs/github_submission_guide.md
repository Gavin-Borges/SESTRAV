# SESTRAV GitHub Submission Guide (Final)

Date: 2026-04-21

## What must be in the repository

- Source code (`functions/`, `src/`, `scripts/`, `tests/`, `pipeline.smk`, `pipeline.py`)
- Configuration and environment files (`config.yaml`, `requirements*.txt`, `environment.yml`, containers)
- Input data needed to reproduce training and scoring:
  - `immunogenicity_dataset.csv`
  - `data/proteomes/*.fasta`
  - `models/peptide_binding_matrix_v3.csv`
- Frozen validation snapshot artifacts:
  - `results/final_validation_report.md`
  - `results/h2_tier_a_summary.csv`
  - `results/h2_tier_a_summary.md`
  - `results/h2_tier_a_fold_metrics.csv`
  - `results/gold_standard_validation.csv`
  - `results/baseline_comparison.csv`
- License and documentation (`LICENSE`, `README.md`, `docs/`)

Do **not** commit local-only artifacts such as:
- `models/*.joblib`, `models/*.pkl`, `models/*.pt`
- transient Snakemake/pytest caches
- ad-hoc run outputs not part of the frozen validation snapshot

## Final local verification before push

```bash
# Use project env and generate local model binaries first (gitignored by policy)
python -m src.train_classifier --data immunogenicity_dataset.csv --feature-mode 30 --binding-matrix models/peptide_binding_matrix_v3.csv
python -m pytest tests/ -q
snakemake --snakefile pipeline.smk --cores 4 --forceall
snakemake --snakefile pipeline.smk full_validation_report --cores 4 --forceall
python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v2
```

Note: `full_validation_report` depends on `config.yaml` `model_path` (default `models/rf_30feature_integrated.joblib`).
Model binaries remain excluded from git and must be regenerated locally on a fresh clone.

Container verification:

```bash
docker build -t sestrav:latest .
docker run --rm -v "$(pwd)/data:/app/data:ro" sestrav:latest -m pytest tests/ -q --basetemp=tmp_pytest
```

Compliance verification (Static Analysis & Dependency Audit):

Ensure that security checks and dependency audits are run and pass cleanly:
```bash
# 1) Run Bandit security scan
bandit -r src/ functions/

# 2) Run dependency vulnerability audit
pip-audit -r environments/requirements.lock
```

The optional ANN/GNN benchmark track is supplementary and not part of the canonical publish gate.
ANN/GNN values are sourced from Project 2 evidence and mirrored in SESTRAV-Dev docs.

Recommended optional-module smoke checks (non-blocking):

```bash
python -m src.ann_benchmark --help
python -m src.gnn_benchmark --help
python -m src.baseline_comparison --help
```

Optional-module references:
- `docs/nn_gnn_project2_sync_matrix.md`
- `docs/nn_gnn_optional_module_guide.md`
- `docs/external_validation_data_expansion_roadmap.md`

## GitHub push checklist

1. Confirm final file set:
   - `git status`
2. Stage project files:
   - `git add -A`
   - `git add -f results/final_validation_report.md results/h2_tier_a_summary.csv results/h2_tier_a_summary.md results/h2_tier_a_fold_metrics.csv results/gold_standard_validation.csv results/baseline_comparison.csv`
3. Review staged changes:
   - `git diff --cached`
4. Commit:
   - `git commit -m "Finalize SESTRAV reproducible release and validation freeze"`
5. Tag release:
   - `git tag -a v2.0.0 -m "SESTRAV v2 reproducible release with external validation"`
6. Push branch and tag:
   - `git push origin HEAD`
   - `git push origin v2.0.0`
7. Create GitHub Release and upload:
   - `release_artifacts/*.manifest.json`
   - `release_artifacts/*.zip`

## License confirmation

The repository license is MIT and valid for open-source distribution as written.

Recommended ownership process (team-level):
1. Confirm all contributors agree to release under MIT.
2. Keep copyright line stable and accurate.
3. Ensure `LICENSE` is present at repo root.
4. Keep README license section pointing to `LICENSE`.

If your institution has IP policy requirements, get advisor/course approval before public release.

