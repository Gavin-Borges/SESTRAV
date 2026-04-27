# SESTRAV Naming Migration Specification

## Objective

Improve professional readability and reduce interpretation risk for external users while preserving pipeline stability with a one-release compatibility window.

## Canonical Terminology

- `proteome_id`: run-routing identifier used in config and output stems.
- `virus`: biological category label (`EBV`, `HPV`) used for scientific interpretation only.
- `dataset_mode`: run interpretation mode (`modeA_baseline`, `modeB_updated`).
- `dataset_version`: version token for IEDB composition (for example `IEDB-20260424-EBV_HPV16_BASELINE-v1`).

## Definitive Rename Map (Balanced Pass)

| current_name | proposed_name | risk_level | reason_for_change | compatibility_behavior | removal_release |
|---|---|---|---|---|---|
| `HPV_8_FASTAs` | `HPV16_18_panel8` | P0 | prevent false inference of HPV strain 8 | accepted as legacy alias where applicable | next release after v1 |
| `EBV_8_FASTAs` | `EBV_panel8_B958` | P0 | remove plural FASTA ambiguity | accepted as legacy alias where applicable | next release after v1 |
| `EBV_panel8_B958` | `EBV_B95_8_panel8` | P0 | clarify B95-8 strain semantics in run ID | canonical output uses new stem; old stem aliases accepted for one release | next release after v1 |
| `rf_30f_immunogenicity.joblib` | `rf_30feature_integrated.joblib` | P1 | remove shorthand and make track explicit | path resolver accepts both names | next release after v1 |
| `xgb_30f_immunogenicity.joblib` | `xgb_30feature_integrated.joblib` | P1 | remove shorthand and make track explicit | path resolver accepts both names | next release after v1 |
| `ann_30f_immunogenicity.pt` | `ann_30feature_integrated.pt` | P1 | remove shorthand and make track explicit | path resolver accepts both names | next release after v1 |
| `rf_immunogenicity.joblib` | `rf_21feature_legacy.joblib` | P1 | expose legacy 21-feature semantics | path resolver accepts both names | next release after v1 |
| `xgb_immunogenicity.joblib` | `xgb_21feature_legacy.joblib` | P1 | expose legacy 21-feature semantics | path resolver accepts both names | next release after v1 |
| `ann_immunogenicity.pt` | `ann_21feature_legacy.pt` | P1 | expose legacy 21-feature semantics | path resolver accepts both names | next release after v1 |

## Code/Docs Touchpoints for Rename Enforcement

- Config and workflow: `config.yaml`, `pipeline.smk`, `pipeline.py`
- Runtime scripts/functions: `scripts/stage*.py`, `functions/stage*.py`
- Analysis modules: `src/baseline_comparison.py`, `src/shap_analysis.py`, `src/final_validation_report.py`, `src/h2_tier_a_evaluation.py`, `src/gold_standard.py`
- Shell runners: `run_analysis.sh`, `run_pipeline.sh`
- Tests: `tests/test_pipeline_integration.py`
- User docs: `README.md`, `docs/master_walkthrough_v1.md`, `docs/final_release_notes_draft.md`, `docs/antigen_accessions.md`

## Compatibility Rules (One Release Window)

1. Read-path compatibility:
   - accept both legacy and canonical names in model/proteome resolvers.
2. Write-path canonicalization:
   - generate canonical stems in new outputs by default.
3. Transparency:
   - emit warnings when legacy names are used.
4. Documentation:
   - include a visible “legacy aliases” section in user-facing docs.

## Validation Gates

1. `pytest tests/ -q` passes after naming updates.
2. `snakemake --snakefile pipeline.smk --dry-run --cores 1` resolves DAG with canonical IDs.
3. Full forced run completes:
   - `snakemake --snakefile pipeline.smk --cores 4 --forceall`
4. Full validation bundle completes:
   - `snakemake --snakefile pipeline.smk full_validation_report --cores 4 --forceall`
5. Release manifest regenerates successfully:
   - `python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v1`
