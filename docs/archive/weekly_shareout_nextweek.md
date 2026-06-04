# SESTRAV Shareout Pack (Slide-Ready) — v2 Dataset

This document maps ready-to-use figures and concise talking points for presentation.
Updated 2026-04-26 to reference the v2 evidence freeze and 30-feature canonical track.

## 1) Shareout Artifact Folder

Use this curated folder for slides:

- `results/shareout_20260426`

Included assets:

- `HPV16_18_panel8_top20_immunogenicity.png`
- `EBV_B95_8_panel8_top20_immunogenicity.png`
- `HPV16_18_panel8_score_distribution.png`
- `EBV_B95_8_panel8_score_distribution.png`
- `cv_roc_curves.png` (30-feature integrated track)
- `cv_precision_recall_curves.png` (30-feature integrated track)
- `cv_score_distributions.png` (30-feature integrated track)
- `shap_summary_rf.png` (30-feature RF SHAP beeswarm)
- `shap_bar_rf.png` (30-feature RF mean |SHAP| bar)
- `shap_waterfall_top_gs.png` (top gold-standard epitope local explanation)
- `calibration_reliability_diagram.png`
- `final_validation_report.md`
- `h2_tier_a_summary.csv`
- `h2_tier_a_summary.md`
- `baseline_comparison.csv`
- `gold_standard_validation.csv`

## 2) Headline Numbers (v2 Frozen Evidence)

- Dataset: v2 (720 peptides, 506 pos / 214 neg, 2.36:1 ratio)
- RF AUC-ROC: 0.7536 +/- 0.0365 (vs binding-only 0.6365, vs literature ~0.60)
- RF AUC-PR: 0.8497 +/- 0.0287 (above-trivial: +0.147, +41% improvement)
- H1 (AUC-ROC >= binding + 0.05): **SUPPORTED** (exceeds threshold by 3x)
- H2 Tier A (R10 >= 2): **NOT SUPPORTED** (R10 = 0.9836)
- Gold-standard positive recovery: 15/15 found, 9/15 in top 25%
- Gold-standard negative discrimination: 9/10 pushed down vs binding-only
- SHAP feature split: 60% binding / 40% TCR-contact features
- Calibration: Brier 0.170 (skill 0.198)
- Reproducibility: zero variance across 6 pipeline cycles

## 3) Slide Narrative (Concise)

### Slide 1 — Problem and Goal

- Problem: binding-only ranking achieves AUC ~0.60 as an immunogenicity proxy (Carri et al. 2023); >95% false-positive rate for true immunogenicity.
- Goal: reproducible TCR-aware computational prioritization pipeline for HPV/EBV.

### Slide 2 — Pipeline Architecture

- Show 4-stage flow: FASTA -> Peptides -> Binding (MHCflurry, 10 HLA alleles) -> TCR Features (p4-p8) -> Ranked Output.
- 16 antigens (8 HPV + 8 EBV), Snakemake-orchestrated, fully containerized.

Use:

- `docs/master_walkthrough_v1.md` (architecture diagram)

### Slide 3 — Feature Engineering (TCR Contact Model)

- 4 physicochemical properties x 5 TCR positions = 20 features + 10 allele binding = 30 features.
- Positions p4-p8 follow Chowell et al. 2015 canonical geometry.

Use:

- `docs/feature_glossary.md`

### Slide 4 — Candidate Ranking Outputs

- Show top-20 and score distributions for HPV and EBV panels.
- Message: pipeline produces interpretable ranked outputs consistently.

Use:

- `results/shareout_20260426/HPV16_18_panel8_top20_immunogenicity.png`
- `results/shareout_20260426/EBV_B95_8_panel8_top20_immunogenicity.png`
- `results/shareout_20260426/HPV16_18_panel8_score_distribution.png`
- `results/shareout_20260426/EBV_B95_8_panel8_score_distribution.png`

### Slide 5 — Cross-Validation Performance

- Show ROC and PR curves for the canonical 30-feature track.
- RF AUC-ROC: 0.7536 vs binding-only 0.6365 (H1 SUPPORTED).
- RF AUC-PR: 0.8497 (+41% above trivial prevalence baseline).

Use:

- `results/shareout_20260426/cv_roc_curves.png`
- `results/shareout_20260426/cv_precision_recall_curves.png`

### Slide 6 — Gold-Standard Validation

- 15/15 known epitopes recovered through all pipeline stages.
- 9/15 in top 25% under integrated model.
- 9/10 gold-standard negatives pushed down vs binding-only (novel v2 capability).

Use:

- `results/shareout_20260426/gold_standard_validation.csv`
- `results/shareout_20260426/baseline_comparison.csv`

### Slide 7 — SHAP Feature Interpretation

- 60% binding / 40% TCR-contact features by total |SHAP|.
- Top TCR features: p8 VdW volume (rank 4), p7 VdW volume (rank 5).

Use:

- `results/shareout_20260426/shap_summary_rf.png`
- `results/shareout_20260426/shap_bar_rf.png`

### Slide 8 — Reproducibility and Infrastructure

- Zero variance across 6 pipeline cycles (3x v1 + 3x v2).
- Docker/Singularity/Conda, 20 passing tests, SHA256 release manifests.

### Slide 9 — Scientific Integrity

- H2 Tier A: NOT SUPPORTED (R10 = 0.98, threshold = 2.0) — presented honestly.
- Virus-specific gap: EBV AUC-ROC 0.803 vs HPV16 0.651.
- No wet-lab validation — computational prioritization prototype only.

Use:

- `results/shareout_20260426/final_validation_report.md`
- `docs/limitations_statement_v1.md`

### Slide 10 — Next Steps and Collaboration

- Controlled IEDB expansion and external validation path.
- Expert collaboration for wet-lab candidate prioritization.

Use:

- `docs/research_roadmap_v1_to_v2.md`
- `docs/collaboration_packet_v2.md`

## 4) Required Verbatim Disclaimer (Slides)

SESTRAV v1 is a reproducible computational prioritization prototype.
Its outputs are candidate rankings for downstream validation, not biological or clinical proof.

## 5) Presenter Checklist

1. Use only assets from `results/shareout_20260426` for quantitative claims.
2. Keep all claims aligned with `docs/colloquium_evidence_freeze_v2.md`.
3. Include the limitation statement on at least one slide.
4. Close with collaboration asks from `docs/collaboration_packet_v2.md`.
5. Always present negative misclassification context alongside headline metrics.
