# SESTRAV Naming Surface Audit (2026-04-24)

This audit inventories naming conventions that can be misread by first-time users
and identifies where they appear in code, pipeline wiring, outputs, tests, and docs.

## Audit Scope

- Config and workflow: `config.yaml`, `pipeline.smk`, `pipeline.py`
- Runtime stages: `scripts/stage*.py`, `functions/stage*.py`
- Analysis/reporting: `run_analysis.py`, `run_analysis.sh`, `run_pipeline.sh`, `src/*.py`
- Tests/CI: `tests/*`, `.github/workflows/ci.yml`
- User docs: `README.md`, `docs/*.md`
- Artifacts: `data/proteomes/`, `models/`, `results/`, `release_artifacts/`

## Hotspot Summary

| Risk | Surface | Current pattern | Why it can be misread |
|---|---|---|---|
| P0 | Proteome ID | `EBV_panel8_B958` | `B958` can be read as a distinct strain token instead of B95-8 context |
| P0 | Model names | `rf_immunogenicity.joblib` | Does not encode that this is legacy 21-feature track |
| P1 | Model names | `rf_30f_immunogenicity.joblib` | `30f` shorthand is less clear to external users |
| P1 | Mixed defaults | `run_analysis.sh` checks `rf_immunogenicity.joblib` while canonical track is 30-feature | Can trigger track confusion during post-pipeline analysis |
| P1 | Result files | `results/h2_tier_a_summary.csv`, `results/baseline_comparison.csv` | Does not encode mode/version in filename, easy to mix baseline vs exploratory reruns |
| P2 | Variable semantics | mixed use of `virus` and `proteome_id` in scripts/docs | Blurs biological labels vs routing identifiers |

## Key Locations Checked

### Proteome IDs and routing names

- `config.yaml`
- `pipeline.smk`
- `pipeline.py`
- `src/gold_standard.py`
- `run_pipeline.sh`
- `run_analysis.sh`
- `README.md`
- `docs/antigen_accessions.md`

### Model artifact naming semantics

- `config.yaml`
- `pipeline.smk`
- `src/final_validation_report.py`
- `src/h2_tier_a_evaluation.py`
- `src/baseline_comparison.py`
- `src/shap_analysis.py`
- `tests/test_pipeline_integration.py`
- `docs/colloquium_evidence_freeze.md`
- `docs/master_walkthrough_v1.md`

### Output naming and interpretation risk

- `results/h2_tier_a_summary.csv`
- `results/baseline_comparison.csv`
- `results/final_validation_report.md`
- `docs/iedb_mode_policy.md`
- `docs/weekly_shareout_nextweek.md`

## Audit Conclusions

1. The old `HPV_8_FASTAs` ambiguity has been corrected, but strain semantics are still not maximally clear in `EBV_panel8_B958`.
2. Model naming is currently the largest remaining readability gap for non-maintainers.
3. Compatibility-friendly aliases should be added so canonical naming can improve without breaking one-release workflows.
4. Interpretation-critical summary outputs should gain optional mode/version-tagged aliases to reduce cross-run confusion.

## Required Follow-up Artifacts

- `docs/naming_migration_spec.md` updated with full current->canonical mapping and deprecation timeline.
- `src/naming.py` compatibility utilities for model/proteome aliases.
- Docs synchronization pass across README + onboarding + release notes.
