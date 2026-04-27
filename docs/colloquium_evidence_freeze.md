# SESTRAV Colloquium Evidence Freeze

This document freezes the artifact set selected for honors colloquium reporting.
Use only this evidence set for slides, talking points, and release claims.

## Freeze Metadata

- Freeze date: 2026-04-24
- Canonical track: 30-feature integrated (`models/rf_30feature_integrated.joblib`)
- Validation command:
  - `conda run -n sestrav snakemake --snakefile pipeline.smk --cores 4 --forceall`
  - `conda run -n sestrav snakemake --snakefile pipeline.smk full_validation_report --cores 4 --forceall`
  - `conda run -n sestrav python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v1`
- Validation environment:
  - Conda env: `sestrav`
  - Python: 3.11
  - `mhcflurry`: 2.1.1
  - `tensorflow`: 2.15.1
  - `numpy`: 1.26.4
- Release bundle artifacts:
  - `release_artifacts/sestrav-v1-20260424T203715Z.manifest.json`
  - `release_artifacts/sestrav-v1-20260424T203715Z.zip`

## Frozen Artifacts (Canonical)

| File | Size (bytes) | SHA256 |
|---|---:|---|
| `results/final_validation_report.md` | 490 | `5990f3589c2d2763b3096e6d82be8455e43452a1497dc3ea86ef0b9c19ea15d7` |
| `results/h2_tier_a_summary.csv` | 1140 | `2bbaf1de7cf10680ad6982cacad5508300ad6c5f8b1b11f356cdbbce9ed505b0` |
| `results/h2_tier_a_summary.md` | 1141 | `0cfa0bb14bca1dff0cbd979d379afadc60be9533430b6765eeaa04b2019e3d6c` |
| `results/h2_tier_a_fold_metrics.csv` | 3195 | `4160e05a1493d10d2067e3bb67da42e3ab9510491825f7cc4e577b68d600b6d9` |
| `results/gold_standard_validation.csv` | 1071 | `76b3b31698ff4553f18df57e221d947c2c2197069f84eb6213bc53cfdce86f99` |
| `results/baseline_comparison.csv` | 890 | `532fec05fa5a0354bc68fa1c2f754ebaa7fd87f03115e1eaacae6e8230afc238` |

## Canonical Headline Values (From Frozen Artifacts)

- Gold-standard stage recovery:
  - Stage 1: 15/15
  - Stage 2: 15/15 (strong binders 15/15)
- H2 Tier A decision row:
  - ISSR@10 ratio (integrated over binding-only): 0.9775
  - ISSR@25 ratio (integrated over binding-only): 1.0092
  - Criterion `R10 >= 2 and stable denominator`: not supported

## Presentation Rule

- For colloquium claims, reference only the frozen files and hashes above.
- If any artifact is regenerated later, create a new freeze document with a new date and hashes.
