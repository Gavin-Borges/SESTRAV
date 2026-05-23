# SESTRAV Colloquium Evidence Freeze — v2 Refresh

This document freezes the refreshed canonical artifact set after the 2026-05-14 pre-publish rerun.
Use this file for external claims that rely on regenerated outputs.
Canonical pointer: `docs/current_evidence_freeze.md`

## Freeze Metadata

- Freeze date: 2026-05-14
- Dataset version: `IEDB-20260424-EBV_HPV16_UPDATED-v2`
- Canonical track: 30-feature integrated (`models/rf_30feature_integrated.joblib`)
- Environment observed in this rerun:
  - Python: 3.13.11
  - scikit-learn: 1.8.0
  - mhcflurry: 2.2.1
- Validation chain rerun:
  - `python -m pytest tests/ -q`
  - `python -m src.train_classifier --data immunogenicity_dataset.csv --feature-mode 30 --binding-matrix models/peptide_binding_matrix.csv`
  - `snakemake --snakefile pipeline.smk --cores 4 --forceall`
  - `snakemake --snakefile pipeline.smk full_validation_report --cores 4 --forceall`
  - `python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v2`
- Release bundle artifacts:
  - `release_artifacts/sestrav-v2-20260514T181938Z.manifest.json`
  - `release_artifacts/sestrav-v2-20260514T181938Z.zip`

## Frozen Artifacts (Canonical)

| File | Size (bytes) | SHA256 |
|---|---:|---|
| `results/final_validation_report.md` | 490 | `2113587559ec6869f3251753c7f1ffb78cfc5ade283a90adf83dacb40dc91282` |
| `results/h2_tier_a_summary.csv` | 1158 | `d1790f7ea13ebe91ace2a8a182c44276e584cc37768dd9ce6f8b63d90a567c8c` |
| `results/h2_tier_a_summary.md` | 1143 | `2675b48f3828ef825b7c1a053cd70e4913c386735b761c0ee51cd31985c11010` |
| `results/h2_tier_a_fold_metrics.csv` | 6444 | `60d1f0930ed9c34e5632a1e27c636f3114c524ea959afd0d1e86e11b5db6c259` |
| `results/gold_standard_validation.csv` | 1072 | `5785665f5fe44031c26773419be5986113e232591813d90b391fba402e94ed26` |
| `results/baseline_comparison.csv` | 890 | `996c06a41e28ca1c89cada7d94d37bad555a4f56b85e5fb31a43399e36a3184a` |

## Dataset Integrity

| File | Size (bytes) | SHA256 |
|---|---:|---|
| `immunogenicity_dataset.csv` | 13056 | `11ce4d771e68ba977c2761d3ed3898016f772563e2728e3b188b8ee758c5bbff` |

## Headline Values

- H2 Tier A decision: **NOT SUPPORTED**
- R10: `0.9836`
- R25: `1.0138`
- Gold-standard positives found: `15/15`
- Gold-standard positives in top 25%: `9/15`

## Supersession Note

This refresh supersedes hash-level claims in `docs/colloquium_evidence_freeze_v2.md` when communicating the latest rerun outputs.
