# SESTRAV Final Closeout Execution Guide

This guide converts the roadmap into execution steps split by what can be completed inside Cursor vs what must be completed personally outside the editor.

## Canonical Release Rule

Publish one model/config as the main public release only after a final scorecard check:

1. Best held-out performance evidence (AUC-ROC, AUC-PR, ISSR@10, ISSR@25).
2. Biological claim integrity and limitation transparency.
3. Reproducible, stable, and documented execution.
4. Strongest match to proposal scope.

Expected current direction: 30-feature as canonical main release, 21-feature as progress/future track.

Canonical repository/source policy reference:

- `docs/canonical_source_of_truth.md`

## What Can Be Finalized In Cursor

- Align repository narrative to canonical-plus-progress release framing.
- Update docs to remove internal contradictions (feature counts, default model messaging, allele support).
- Add and maintain a canonical selection scorecard.
- Standardize command references for validation and release checks.
- Prepare GitHub-facing files (`README.md`, `docs/*`, release checklist text, citation metadata).
- Run local tests/validation commands when environment allows and capture outputs into docs.

## What You Must Do Personally (Outside Cursor)

- Final presentation production (slide design, speaking notes, rehearsal timing).
- Honors/community logistics (submission portals, abstract forms, presenter registration).
- Team role assignment and live Q&A rehearsal for biological validity critiques.
- Final GitHub web actions (create release, publish notes, attach large artifacts if not committed).
- Any institutional approvals or data-sharing confirmations required for public dissemination.

## Current Environment Blockers (Observed)

- Base interpreter in this session is Python 3.13.
- `mhcflurry==2.1.1` dependency resolution fails on Python 3.13 because of TensorFlow version constraints.
- `tensorflow` runtime used by `mhcflurry` requires NumPy 1.x compatibility in this stack.
- `snakemake` must be installed from pip on this Windows setup.

Required fix path:

1. Use the project conda environment from `environment.yml` (Python 3.11 target).
2. Install Python dependencies with:
   - `pip install -r requirements.txt`
3. Run `mhcflurry-downloads fetch models_class1_presentation`.
4. Train canonical local model artifacts:
   - `python -m src.train_classifier --data immunogenicity_dataset.csv --feature-mode 30 --binding-matrix models/peptide_binding_matrix_v3.csv`
5. Run `snakemake --snakefile pipeline.smk --cores 4 --forceall` in that environment for full artifact regeneration.
6. Run `snakemake --snakefile pipeline.smk full_validation_report --cores 4 --forceall` to refresh validation artifacts from the same run.

## Weekly Balance Plan (Coming Weeks)

Apply this balance each week to optimize outcomes:

- 40% model performance and ranking quality improvements.
- 35% biological validation and scientific claim robustness.
- 25% reproducibility, GitHub polish, and showcase communication.

Run a weekly checkpoint:

- Updated canonical scorecard.
- Updated risk/issues list.
- Updated release-readiness status.

## Minimum Evidence Required Before Final Publish

- Regenerated validation bundle for canonical track.
- Clear legacy/progress comparison outputs for the secondary track.
- One-page biological limitations summary aligned with actual behavior.
- Reproducible command path documented from environment setup to final report.
- Release note draft with artifact links and environment version note.
