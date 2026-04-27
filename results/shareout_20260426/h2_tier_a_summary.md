# H2 Tier A Evaluation Summary

## Inputs
- Dataset: `immunogenicity_dataset.csv`
- Integrated model template: `models/rf_30feature_integrated.joblib`
- Binding matrix: `models/peptide_binding_matrix.csv`
- CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- Gold-standard peptides held out before CV: `16`

## Fold-aggregated metrics (mean +/- std)
- Integrated model ISSR@10: `0.8571 +/- 0.1129`
- Binding-only ISSR@10: `0.8714 +/- 0.0598`
- Integrated model ISSR@25: `0.8514 +/- 0.0239`
- Binding-only ISSR@25: `0.8286 +/- 0.0202`

## Enrichment ratios
- R10 = ISSR@10(integrated) / ISSR@10(binding-only): `0.9836`
- R25 = ISSR@25(integrated) / ISSR@25(binding-only): `1.0276`
- Bootstrap 95% CI for R10 (OOF): `[0.8750, 1.1186]`
- Fold-level paired sign-flip p-value for ISSR@10 delta > 0: `1.0000`
- Binding ISSR@10 denominator quality: `stable`

## H2 Decision
- Rule used: `R10 >= 2.0`, `binding ISSR@10 >= 0.08`, and `lower 95% CI(R10) >= 2.0`
- Result: **NOT SUPPORTED**

## Output files
- Fold metrics CSV: `results\h2_tier_a_fold_metrics.csv`
- Summary CSV: `results\h2_tier_a_summary.csv`
