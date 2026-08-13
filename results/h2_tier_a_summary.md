# H2 Tier A Evaluation Summary

## Inputs
- Dataset: `data/immunogenicity_dataset_v3.csv`
- Integrated model template: `models/rf_30feature_integrated.joblib` (sha256: `9a7bd2051c85e360c89a59f2bb1b4688e64a316900422e0cf45ba5967bf711f3`)
- Binding matrix: `models/peptide_binding_matrix_v3.csv`
- CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- Splitter note: this is a label-only (ungrouped) splitter, but it is **peptide-disjoint
  on this corpus by construction** - the v3 dataset is 1,004 rows over 1,004 unique
  peptides, so no peptide can straddle a fold boundary and a peptide-grouped re-run
  would be a no-op. This figure is therefore NOT exposed to the D15 leakage defect
  (`docs/claims_register.md` D15, D17); the disclosure is stated here rather than left
  for a reader to infer from the splitter name.
- Gold-standard peptides held out before CV: `16`

## Fold-aggregated metrics (mean +/- std)
- Integrated model ISSR@10: `0.9474 +/- 0.0372`
- Binding-only ISSR@10: `0.8947 +/- 0.0645`
- Integrated model ISSR@25: `0.9121 +/- 0.0180`
- Binding-only ISSR@25: `0.8828 +/- 0.0119`

## Enrichment ratios
- R10 = ISSR@10(integrated) / ISSR@10(binding-only): `1.0588`
- R25 = ISSR@25(integrated) / ISSR@25(binding-only): `1.0331`
- Bootstrap 95% CI for R10 (OOF): `[0.9778, 1.1220]`
- Fold-level paired sign-flip p-value for ISSR@10 delta > 0: `0.1875` (FDR Corrected: `0.1875`)
- Binding ISSR@10 denominator quality: `stable`

## H2 Decision
- Rule used: `R10 >= 2.0`, `binding ISSR@10 >= 0.08`, and `lower 95% CI(R10) >= 2.0`
- Result: **NOT SUPPORTED**

## Output files
- Fold metrics CSV: `h2_tier_a_fold_metrics.csv`
- Summary CSV: `h2_tier_a_summary.csv`
