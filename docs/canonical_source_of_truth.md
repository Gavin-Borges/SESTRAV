# SESTRAV Canonical Source of Truth Policy (v1)

This document defines the single authoritative project path and execution track for SESTRAV v1.
Use this policy for every rerun, report, and presentation claim.

## Authoritative Repository

- Canonical source: this repository (`main` branch)

All active v1 work must originate from this repository only.
Earlier development workspace copies are superseded and should not be used as release evidence.

## Canonical Track Rule

- Canonical model/config track: **30-feature integrated track**
- Canonical defaults:
  - `config.yaml` `feature_mode: 30`
  - `config.yaml` `model_path: models/rf_30feature_integrated.joblib`

The 21-feature track is retained as a historical comparator only.

Naming migration references:

- `docs/naming_migration_spec.md`
- `docs/output_naming_standard_v1.md`

## Rerun and Evidence Rules

1. All reruns must execute from the repository root with committed `config.yaml`.
2. Regenerated outputs must be written under `results/` in the repository.
3. Final public claims must reference:
   - `results/final_validation_report.md`
   - `results/h2_tier_a_summary.csv`
   - `results/gold_standard_validation.csv`
   - `results/baseline_comparison.csv`
4. If artifacts are regenerated, freeze a new evidence snapshot and checksums before using them externally.

## Documentation Rule

For v1 external communication, keep these documents aligned:

- `README.md`
- `docs/reproducibility_finalization_status.md`
- `docs/colloquium_evidence_freeze_v2.md`
- `docs/final_release_notes_draft.md`

If wording conflicts with this policy, update the docs above before distribution.
