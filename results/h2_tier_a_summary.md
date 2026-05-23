# H2 Tier A Evaluation Summary

## Inputs
- Dataset: `immunogenicity_dataset.csv`
- Integrated model template: `models/rf_30feature_integrated.joblib`
- Binding matrix: `models/peptide_binding_matrix_v3.csv`
- CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- Gold-standard peptides held out before CV: `16`

## Fold-aggregated metrics (mean +/- std)
- Integrated model ISSR@10: `0.7895 +/- 0.0912`
- Binding-only ISSR@10: `0.8316 +/- 0.0440`
- Integrated model ISSR@25: `0.8285 +/- 0.0396`
- Binding-only ISSR@25: `0.8117 +/- 0.0390`

## Enrichment ratios
- R10 = ISSR@10(integrated) / ISSR@10(binding-only): `0.9494`
- R25 = ISSR@25(integrated) / ISSR@25(binding-only): `1.0208`
- Bootstrap 95% CI for R10 (OOF): `[0.9091, 1.2188]`
- Fold-level paired sign-flip p-value for ISSR@10 delta > 0: `1.0000` (FDR Corrected: `1.0000`)
- Binding ISSR@10 denominator quality: `stable`

## H2 Decision
- Rule used: `R10 >= 2.0`, `binding ISSR@10 >= 0.08`, and `lower 95% CI(R10) >= 2.0`
- Result: **NOT SUPPORTED**

## Output files
- Fold metrics CSV: `h2_tier_a_fold_metrics.csv`
- Summary CSV: `h2_tier_a_summary.csv`
