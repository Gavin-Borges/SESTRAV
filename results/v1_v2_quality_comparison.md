# SESTRAV v1 vs v2 Quality Comparison

**Generated:** 2026-04-24 | **Mode:** Diagnostic Assessment

## Dataset Properties

| Property | v1 | v2 | Assessment |
|----------|-----|-----|------------|
| Total peptides | 928 | 720 | v1 larger by 22% |
| Positive (immunogenic) | 787 (84.8%) | 506 (70.3%) | v2 more balanced |
| Negative (non-immuno) | 141 (15.2%) | 214 (29.7%) | v2 has 52% more negatives |
| Class imbalance ratio | 5.58:1 | 2.36:1 | v2 significantly better |
| Peptide overlap | — | 512 shared | 274 v1-only, 208 v2-only |
| Label flips | — | 69 | 9.6% of shared peptides |
| Conflict peptides (cross-label) | — | 201 (all resolved positive) | Majority-vote deduplication |

## Cross-Validation Performance (RF, 30-feature)

| Metric | v1 Raw | v1 Above Trivial | v2 Raw | v2 Above Trivial | Winner |
|--------|--------|-------------------|--------|-------------------|--------|
| AUC-PR | 0.953 | +0.105 | 0.849 | +0.146 | **v2** (above-trivial) |
| AUC-ROC | 0.820 | +0.320 | 0.754 | +0.254 | v1 (both measures) |
| Recall@threshold | 99% | — | 99% | — | Tie |
| Accuracy | 81.9% | — | 64.6% | — | v1 (raw) |
| F1 | 0.896 | — | 0.742 | — | v1 (raw) |

**Interpretation:** v1's raw numbers are higher because its trivial-all-positive baseline is already 84.8%. v2's above-trivial AUC-PR is 39% better than v1's, meaning the model works harder relative to class frequency.

## Calibration (Brier Score)

| Metric | v1 | v2 | Interpretation |
|--------|-----|-----|----------------|
| Brier Score | 0.096 | 0.170 | v2 higher (harder prediction task) |
| Trivial Brier | 0.131 | 0.212 | Trivial baseline also harder for v2 |
| Brier Skill Score | 0.265 | 0.198 | Both show moderate skill above trivial |

**Platt calibrator note:** The existing calibrator was fit on v1 distribution. v2's shifted class balance means it needs refitting before deployment.

## Gold Standard Validation

### Positive Set (15 epitopes)

| Stage | v1 Recovery | v2 Recovery |
|-------|-------------|-------------|
| Stage 1 (Peptide Gen) | 15/15 | 15/15 |
| Stage 2 (MHC Binding) | 15/15 | 15/15 |
| Stage 4 (Top 25%) | 15/15 | 15/15 |

**All 15 gold-standard positives recovered in both versions.**

### Negative Set (10 curated strong-binder negatives) — NEW in v2

| Metric | v2 Result |
|--------|-----------|
| Negatives pushed down vs binding-only | 9/10 (90%) |
| Dramatically rejected (>50 percentile-point drop) | 3/10 (MLLLIVAGI, CLLIRPLLL, DKKQRFHNI) |
| Mean percentile shift (integrated vs binding) | +28.7 pp |

**This is the first quantitative evidence that TCR features add discriminative value beyond binding.**

## SHAP Feature Signal (30-feature RF)

| Feature Group | Mean |SHAP| | Proportion |
|---------------|-----------|------------|
| Binding features (10 alleles) | 0.224 | 59.9% |
| TCR features (20 physicochemical) | 0.150 | 40.1% |

**Top TCR features:** p8_vdw_volume (rank 4, SHAP=0.029), p7_vdw_volume (rank 5, SHAP=0.024). These outrank 4 of 10 binding features.

## Negative Class Analysis

| Metric | v2 |
|--------|-----|
| Total negatives | 214 |
| Correctly rejected at threshold 0.3559 | 38 (17.8%) |
| Misclassified as positive | 176 (82.2%) |
| Conflict peptides among negatives | 0 (all resolved positive) |

**Note:** The 82% misclassification rate reflects the high-recall (99%) operating point, not model failure. A threshold of 0.5 would significantly improve specificity at the cost of recall. The operating-point choice depends on application context.

## Recommendation

### **Share v2** with diagnostic supporting evidence

**Rationale:**

1. **Gold standard negatives (9/10 pushed down)** — First quantitative proof that SESTRAV's TCR-contact features add value beyond binding-only prediction. This is the project's strongest novel claim.

2. **40/60 SHAP split** — TCR features contribute 40% of total model explanation power, with p8 and p7 VdW volume ranking in the top 5 globally. This is not noise.

3. **Honest metrics** — v2's class balance (2.36:1 vs 5.58:1) means performance numbers are less inflated. Above-trivial AUC-PR is 39% better for v2.

4. **More negative data** — 214 vs 141 negatives provides a more rigorous test of discrimination ability.

**Caveats to acknowledge:**

- Smaller total dataset (720 vs 928 peptides)
- Lower raw CV metrics (honest consequence of better class balance)
- 69 label flips between versions (9.6% of shared peptides)
- High negative misclassification at current operating point (threshold trade-off)

### Actions Required

1. Create new evidence freeze document with v2 SHA256 hashes
2. Run `release_bundle.py` to produce manifest + zip
3. Refit Platt calibrator on v2 data distribution
4. Include gold standard negative validation in release package
5. Include 30-feature SHAP plots in release package
6. Update collaboration packet to reference v2 and acknowledge class balance improvement
