# SESTRAV Finalization Remediation Plan

Date: 2026-04-21

This plan converts strict review findings into concrete finalization actions.
Items marked "In progress" already started in this repository.

---

## 1) Reproducibility closure

- [x] Add release bundle generator for canonical outputs (`src/release_bundle.py`).
- [x] Document release bundle workflow in `README.md`.
- [x] Add release artifact publication requirements to `docs/submission_checklist.md`.
- [x] Regenerate canonical outputs in current local env from one forced clean run (`snakemake --snakefile pipeline.smk --cores 4 --forceall` + `full_validation_report`).
- [x] Build release bundle locally:
  - `release_artifacts/sestrav-v1-20260421T211338Z.manifest.json`
  - `release_artifacts/sestrav-v1-20260421T211338Z.zip`
- [x] Re-run the same chain in pinned env (`python=3.13`, `sklearn=1.8.0`) for release-grade freeze.
- [x] Publish both:
  - `release_artifacts/sestrav-v2-20260425T040154Z.manifest.json`
  - `release_artifacts/sestrav-v2-20260425T040154Z.zip`
- [ ] Attach release bundle to GitHub Release (`v1.0.0` or final tag).

## 2) Statistical evidence closure (H2)

- [x] Extend `src/h2_tier_a_evaluation.py` to export:
  - bootstrap 95% CI for `R10`
  - paired fold sign-flip p-value for ISSR@10 delta
- [x] Update protocol doc `docs/H2_ISSR_evaluation_protocol.md` to reflect new outputs.
- [x] Re-run H2 Tier A as part of `full_validation_report` regeneration pass.
- [x] Re-run H2 Tier A in pinned env (2026-04-25, 3x v2 cycles, R10=0.9836).
- [x] Update scorecard and release notes with final inferential outputs from pinned rerun.

## 3) Biological claim boundary closure

- [x] Clarify validation scope in `README.md` (computational vs wet-lab).
- [x] Clarify gold-standard holdout variant note in `docs/gold_standard_validation_report.md`.
- [x] Ensure all presentation language uses "computational validation" unless new experimental evidence is added.

## 4) Final verification pass

- [x] Run full test suite: `python -m pytest tests/ -q` (`20 passed`)
  - Historical scope note: this value is from the dated remediation cycle and
    should not be interpreted as the latest repository-wide baseline.
- [x] Re-run final validation bundle:
  - `snakemake --snakefile pipeline.smk full_validation_report --cores 4 --forceall`
- [x] Build release bundle:
  - `python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v1`
- [x] Checksums generated in `release_artifacts/sestrav-v2-20260425T040154Z.manifest.json`.
- [ ] Attach artifacts in GitHub Release notes.

---

## Remaining blockers

- [x] Release-grade rerun in pinned environment completed (2026-04-25, 3x v1 + 3x v2 cycles).
- [x] Platt calibrator refit on v2 class distribution completed (2026-04-25).
- [x] Release bundle generated: `release_artifacts/sestrav-v2-20260425T040154Z.manifest.json` + `.zip`
- [ ] Checksum publication and GitHub Release attachment are still pending.
