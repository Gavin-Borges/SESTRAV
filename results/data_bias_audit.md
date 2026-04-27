# SESTRAV Data Bias/Skew Audit

## Dataset summary
- Total records: `720`
- Positives: `506`
- Negatives: `214`
- Positive rate: `0.7028`
- Unique peptides: `720`

## Metadata quality
- Missing virus: `0`
- Missing strain: `720`
- Missing allele: `720`
- Peptide length range: `8` to `11` (mean `9.41`)

## Label conflict risk
- Raw source records: `921`
- Peptides with conflicting raw labels: `201`

## Known pipeline risk points to monitor
- Labels inferred from Epitope Table filenames for those source files.
- Duplicate peptide conflict handling still uses majority-vote collapse.
- 30-feature mode still maps missing binding rows to all-zero vectors.

## Output files
- Audit summary CSV: `results\data_bias_audit_summary.csv`
- Virus/label breakdown CSV: `results\data_bias_audit_summary_virus_label_counts.csv`
