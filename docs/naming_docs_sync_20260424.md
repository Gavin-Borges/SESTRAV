# Naming Docs Synchronization Log (2026-04-24)

This document tracks user-facing doc updates required to prevent naming interpretation drift during the one-release compatibility window.

## Synchronization Sequence

1. **Policy and standards first**
   - `docs/naming_migration_spec.md`
   - `docs/output_naming_standard_v1.md`
   - `docs/canonical_source_of_truth.md`
2. **Onboarding and walkthrough docs**
   - `README.md`
   - `docs/master_walkthrough_v1.md`
3. **Release and operations docs**
   - `docs/final_release_notes_draft.md`
   - `docs/github_submission_guide.md`
   - `docs/submission_checklist.md`
4. **Scientific context docs**
   - `docs/antigen_accessions.md`
   - `docs/gold_standard_validation_report.md`
   - `docs/canonical_selection_scorecard.md`
5. **Shareout package docs**
   - `docs/weekly_shareout_nextweek.md`
   - `docs/colloquium_evidence_freeze.md`

## Completed in This Pass

- Updated canonical EBV proteome stem references to `EBV_B95_8_panel8` in key onboarding/validation docs.
- Updated canonical model naming references to:
  - `rf_30feature_integrated.joblib`
  - `rf_21feature_legacy.joblib`
- Added links from README/canonical policy to migration and naming standards.
- Updated command snippets to include dataset mode/version where applicable.

## Intentional Compatibility Mentions Kept

- Legacy aliases remain documented where needed for one-release migration transparency.
- Historical audit/migration docs intentionally retain old identifiers for provenance.

## Next Release Action

After one release window, remove legacy aliases from:

- config fallback entries
- compatibility resolver maps
- legacy references in operational scripts
