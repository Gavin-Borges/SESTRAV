# SESTRAV Colloquium Evidence Freeze — v2

This document freezes the artifact set for the v2 dataset (IEDB-20260424-EBV_HPV16_UPDATED).
Use only this evidence set for slides, talking points, and release claims.

## Freeze Metadata

- Freeze date: 2026-04-25
- Dataset version: v2 (IEDB-20260424-EBV_HPV16_UPDATED)
- Canonical track: 30-feature integrated (`models/rf_30feature_integrated.joblib`)
- Source data: `09_Data/updated_only/` (4 UPDATED IEDB export files)
- Supersedes: `docs/colloquium_evidence_freeze.md` (v1)
- Dataset properties: 720 peptides, 506 pos (70.3%), 214 neg (29.7%), ratio 2.36:1
- Stability: 3x full pipeline cycles with zero variance (see `results/multi_run_stability_report.md`)
- Environment: `sestrav` (Python 3.11.15, sklearn 1.6.1)
- Platt calibrator: Refit on v2 OOF distribution (2026-04-25)

## Frozen Artifacts (Canonical)

### Core Pipeline Outputs

| File | Size (bytes) | SHA256 |
|---|---:|---|
| `results/final_validation_report.md` | 490 | `297c88f692862268f0d3589a0349fc3c19ae85593b2fce8c2c55e0c56819b112` |
| `results/h2_tier_a_summary.csv` | 1158 | `27d77bc8527450ab49e9fee88ceaecfe6238c24815f3fe219721cc63d79ae4c1` |
| `results/h2_tier_a_summary.md` | 1146 | `4a1bb801c2c19fc9f23a4f41541d1d7e5179fcb1b013bbf810dc0231e096b162` |
| `results/h2_tier_a_fold_metrics.csv` | 3379 | `0ddbb10df170098fa996635ffc69c97ef00b41b0e13b7fed152914a213c69179` |
| `results/gold_standard_validation.csv` | 1072 | `8bd0f6680e8ffd6d2155ff7eff7323bd7307037e6a3d2729a6f8b56958333896` |
| `results/baseline_comparison.csv` | 890 | `195d7321a1fda5901d2f971b19cb8161170a7c1af108a7d58b15fef90c8c6f20` |

### v2 Diagnostic Artifacts

| File | Description |
|---|---|
| `results/gold_standard_negative_validation.csv` | 10 curated strong-binding negatives; 9/10 pushed down |
| `results/calibration_metrics.csv` | Brier score and skill (v1 vs v2) |
| `results/v1_v2_quality_comparison.md` | Comprehensive v1/v2 comparison |
| `results/multi_run_stability_report.md` | 6-cycle stability evidence (3 v1 + 3 v2) |
| `results/negative_class_analysis.csv` | Negative class behavior analysis |
| `results/data_bias_audit.md` | Dataset bias/skew audit |

### Model & Feature Artifacts

| File | Description |
|---|---|
| `models/feature_importances.csv` | RF/XGB feature importance rankings |
| `models/rf_oof_predictions.csv` | Out-of-fold predictions for calibration |
| `models/platt_calibrator.joblib` | Platt calibrator refit on v2 distribution (2026-04-25) |
| `results/shap_values_rf.csv` | SHAP values for RF (60/40 binding/TCR split) |

### Training Dataset

| File | Size (bytes) | SHA256 |
|---|---:|---|
| `immunogenicity_dataset.csv` | 13056 | `11ce4d771e68ba977c2761d3ed3898016f772563e2728e3b188b8ee758c5bbff` |

## Canonical Headline Values (From Frozen Artifacts)

- Gold-standard positive recovery:
  - Stage 1: 15/15
  - Stage 2: 15/15 (strong binders 15/15)
  - Stage 4 found: 15/15
  - Stage 4 in top 25%: 9/15
- Gold-standard negative discrimination:
  - Model pushes down vs binding-only: 9/10 (90%)
  - Dramatically rejected (>50 pp shift): 3/10
- Feature contribution (SHAP, 30-feature RF):
  - Binding features: 59.9% of total |SHAP|
  - TCR-contact features: 40.1% of total |SHAP|
  - Top TCR features: p8_vdw_volume (rank 4), p7_vdw_volume (rank 5)
- Calibration:
  - v2 Brier Score: 0.170 (trivial: 0.212, skill: 0.198)
  - v1 Brier Score: 0.096 (trivial: 0.131, skill: 0.265)
- Negative class (at threshold 0.3559):
  - Correctly classified: 38/214 (17.8%)
  - Note: reflects 99% recall operating point, not model failure

## Why v2 Over v1

1. More honest class balance (2.36:1 vs 5.58:1)
2. Above-trivial AUC-PR is 41% better for v2 (+0.147 vs +0.105)
3. Gold standard negative discrimination is a new capability unique to v2
4. 52% more negative training examples (214 vs 141)
5. All 15 gold-standard positives fully recovered

## Presentation Rule

- For colloquium claims, reference only the frozen files and hashes above.
- If any artifact is regenerated later, create a new freeze document with a new date and hashes.
- Always present negative misclassification context alongside headline metrics.
