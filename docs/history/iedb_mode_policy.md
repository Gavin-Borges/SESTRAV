# SESTRAV IEDB Input Mode Policy (v1)

This policy governs how IEDB input pool changes are handled in SESTRAV v1.

## Purpose

Support both:

- stable, presentation-ready baseline reporting
- controlled exploratory improvements through expanded IEDB input data

without blending claims across incompatible datasets.

## Mode Definitions

### Mode A: Frozen Baseline

- Default mode for release and presentation claims.
- Dataset version for this finalization pass:
  - `IEDB-20260424-EBV_HPV16_BASELINE-v1`
- Must align with the frozen artifact set in `docs/colloquium_evidence_freeze.md`.

### Mode B: Expanded IEDB Exploratory

- Optional mode for increasing/changing input pool (additional records, additional exports, revised filters).
- Must use a unique dataset version ID and explicit exploratory labeling.
- Cannot replace Mode A claims unless a new official freeze is created.

## Required Metadata for Any Mode B Run

1. IEDB query/export dates and source files.
2. Inclusion/exclusion criteria (length, assay type, quality filters).
3. Class balance and duplicate-resolution summary.
4. Gold-standard holdout compatibility statement.
5. Result comparison against Mode A on H2 and baseline metrics.

## Reporting Rules

- Slides/docs must mark each figure/table as Mode A or Mode B.
- Conclusions must not mix values across modes without an explicit comparison panel.
- If Mode B becomes the new canonical dataset, regenerate:
  - `results/final_validation_report.md`
  - `results/h2_tier_a_summary.csv`
  - `results/gold_standard_validation.csv`
  - `results/baseline_comparison.csv`
  - `docs/colloquium_evidence_freeze.md`

## Mode B Integration Readiness (2026-04-24)

The infrastructure for Mode B runs is in place:

- `config.yaml` accepts `dataset_mode` and `dataset_version` fields.
- `src/final_validation_report.py` generates mode/version-tagged summary aliases.
- `docs/output_naming_standard_v1.md` defines the tagged filename convention.
- Updated data files can be dropped in as a replacement `immunogenicity_dataset.csv` with a new version ID.

To execute a Mode B run, follow the steps in "Required Metadata for Any Mode B Run" above.

## Future Transition Rule

Mode B can be promoted to canonical only when:

1. The rerun is reproducible in the pinned environment.
2. The full validation bundle is regenerated successfully.
3. A fresh release bundle/manifest is produced.
4. The decision is recorded in the master decisions document.
