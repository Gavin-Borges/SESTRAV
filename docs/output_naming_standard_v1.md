# SESTRAV Output Naming Standard (v1)

This standard defines professional, interpretation-safe naming for files emitted by the SESTRAV pipeline.

## 1) Proteome Output Stem Convention

Use canonical `proteome_id` stems:

- `HPV16_18_panel8`
- `EBV_B95_8_panel8`

Legacy aliases accepted for one release:

- `HPV_8_FASTAs` -> `HPV16_18_panel8`
- `EBV_8_FASTAs` -> `EBV_B95_8_panel8`
- `EBV_panel8_B958` -> `EBV_B95_8_panel8`

## 2) Stage Output Convention

Per-proteome stage outputs:

- `{proteome_id}_peptides.csv`
- `{proteome_id}_binding.csv`
- `{proteome_id}_features.csv`
- `{proteome_id}_ranked.csv`
- `{proteome_id}_top20_immunogenicity.png`
- `{proteome_id}_score_distribution.png`

## 3) Interpretation-Critical Summary Convention

Keep legacy summary filenames for compatibility, and additionally emit mode/version-tagged aliases:

- `gold_standard_validation__{dataset_mode}__{dataset_version}.csv`
- `baseline_comparison__{dataset_mode}__{dataset_version}.csv`
- `h2_tier_a_summary__{dataset_mode}__{dataset_version}.csv`

Default metadata source is `config.yaml`:

- `dataset_mode`
- `dataset_version`

## 4) Model Artifact Convention

Preferred canonical names:

- `rf_30feature_integrated.joblib`
- `xgb_30feature_integrated.joblib`
- `ann_30feature_integrated.pt`
- `rf_21feature_legacy.joblib`
- `xgb_21feature_legacy.joblib`
- `ann_21feature_legacy.pt`

Legacy aliases are supported for one release through resolver logic.

## 5) Naming Safety Rules

1. Routing IDs must not imply unrelated biological meaning.
2. Track semantics must be explicit in model filenames (`30feature_integrated`, `21feature_legacy`).
3. Summary outputs shared externally must include mode/version-tagged aliases.
4. User-facing docs should define how to interpret each naming dimension.
