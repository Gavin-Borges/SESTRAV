# SESTRAV Colloquium Evidence Freeze — v2 Refresh (2026-05-24)

This document freezes the refreshed canonical artifact set after the 2026-05-24 pre-publish rerun.
Use this file for external claims that rely on regenerated outputs.
Canonical pointer: `docs/current_evidence_freeze.md`

## Freeze Metadata

- Freeze date: 2026-05-24
- Dataset version: `2.0.0-alpha`
- Canonical track: 30-feature integrated (`models/rf_30feature_integrated.joblib`)
- Environment observed in this rerun:
  - Python: 3.11.15
  - scikit-learn: 1.8.0
  - mhcflurry: 2.2.1
- Validation chain rerun:
  - `python -m pytest tests/ -q`
  - `python -m src.train_classifier --data immunogenicity_dataset.csv --feature-mode 30 --binding-matrix models/peptide_binding_matrix_v3.csv`
  - `snakemake --snakefile pipeline.smk --cores 4 --forceall`
  - `snakemake --snakefile pipeline.smk full_validation_report --cores 4 --forceall`
  - `python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v2`
- Release bundle artifacts:
  - `release_artifacts/sestrav-v2-20260524T050810Z.manifest.json`
  - `release_artifacts/sestrav-v2-20260524T050810Z.zip`

## Frozen Artifacts (Canonical)

| File | Size (bytes) | SHA256 |
|---|---:|---|
| `results/final_validation_report.md` | 852 | `c9de708d0674e17c64fc896255d341375aff41daea58c42432329e7cbce5135c` |
| `results/h2_tier_a_summary.csv` | 1155 | `a90498d354f14ddace3600ccc47c1325307a35b36c304dfacfa87763ef340331` |
| `results/h2_tier_a_summary.md` | 1156 | `36324636aa0983eef3300d74be4d3d223f5d3bc4b6ca00adf21d506543d1389e` |
| `results/h2_tier_a_fold_metrics.csv` | 9334 | `ed6a597778a996a0b9085a92f75fa979b2c84a740fbe569b59552e66c58e8d34` |
| `results/gold_standard_validation.csv` | 1075 | `52793cf7785d79f9852740736ecc33b0424b5ca2724f54e02faf9d0197a7c3cf` |
| `results/baseline_comparison.csv` | 1123 | `bb7ab6352dae0cef6cb30ff0bc65ff8573b55e312b66671368b2af476b38dfba` |

## Dataset Integrity

| File | Size (bytes) | SHA256 |
|---|---:|---|
| `immunogenicity_dataset.csv` | 20075 | `b95199092c40afb156bcaca9fc97176b4ed2a187901eec1e31b5f62bb8e19e5b` |

## Headline Values

- H2 Tier A decision: **NOT SUPPORTED**
- R10: `0.9494`
- R25: `1.0208`
- Gold-standard positives found: `15/15`
- Gold-standard positives in top 25%: `7/15`

## Supersession Note

This refresh supersedes hash-level claims in `docs/colloquium_evidence_freeze_v2_20260514.md` when communicating the latest rerun outputs.
