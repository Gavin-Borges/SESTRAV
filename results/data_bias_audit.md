> **SUPERSEDED HISTORICAL SNAPSHOT (dated 2026-04-26, the v1.0.0 720-peptide corpus). DO NOT CITE.**
> Every count below describes a dataset that no longer exists. This file was written by
> `src/data_bias_audit.py` at `69e0e5c` and has not been regenerated since, while the corpus it
> audits grew from the **720** records reported here to **51,185** rows in
> `data/immunogenicity_dataset_v5.csv`. The positive rate, label-conflict counts and metadata-
> quality figures are therefore stale by roughly seventy-fold and must NOT be read as a current
> characterisation of SESTRAV's data.
>
> **The two output files named at the end of this document have never existed in this
> repository.** `results/data_bias_audit_summary.csv` and
> `results/data_bias_audit_summary_virus_label_counts.csv` return zero hits across every ref in
> history and are absent from disk, so the "Output files" section points at nothing a reader can
> open. They are left in place rather than deleted, because removing them would hide that this
> snapshot was only ever partially published.
>
> For current corpus composition see `results/d29_corpus_composition.csv`, which carries a
> provenance sidecar, and `docs/data_registry.md`.
>
> Annotated rather than regenerated on purpose: re-running the generator would replace a
> historical record with new numbers under a v1.0.0 filename, which is a different artifact, not
> a corrected one. This banner mirrors the one `results/multi_run_stability_report.md` has
> carried since `74bb63b` - the same 2026-07-16 "mark stale snapshots" pass that missed this file.

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
