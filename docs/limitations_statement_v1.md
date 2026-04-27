# SESTRAV v1/v2 Limitation and Claim-Boundary Statement

Use this exact framing across README, reports, slides, and collaborator outreach.

## Standard Language

SESTRAV is a reproducible **computational prioritization prototype** for immunogenic epitope ranking.
It is not a clinically validated or biologically confirmed vaccine design system in this version.

## What Can Be Claimed

1. The end-to-end computational pipeline is reproducible under the documented environment.
2. Stage outputs and validation artifacts are generated consistently across repeated reruns.
3. The current frozen evidence quantifies comparative behavior of integrated and binding-only strategies.
4. TCR-contact features contribute measurably to model explanatory power (40% SHAP share) and improve gold-standard negative rejection (9/10 pushed down).

## What Cannot Be Claimed

1. Wet-lab biological efficacy or clinical efficacy.
2. Generalized performance beyond the dataset and scope used in this prototype.
3. Strong canonical superiority where current frozen metrics do not support it.
4. Statistically significant enrichment of integrated over binding-only in the H2 Tier A evaluation (R10 = 0.98, threshold = 2.0).

## Required Disclaimer for External Use

When presenting ranked peptides:

- Describe them as **computationally prioritized candidates for downstream validation**.
- State that biological/clinical validation requires expert collaboration and wet-lab follow-up.

## Virus-Specific Performance Gap

The pooled model shows a consistent performance disparity between EBV and HPV16 subgroups.
This must be disclosed whenever virus-specific rankings are presented.

### RandomForest 5-Fold CV Subgroup Metrics (v2 dataset, 30-feature mode)

| Metric | EBV (mean) | HPV16 (mean) | Gap |
|--------|-----------|-------------|-----|
| AUC-ROC | 0.803 | 0.651 | -0.152 |
| AUC-PR | 0.872 | 0.822 | -0.050 |
| ISSR@10 | 0.908 | 0.830 | -0.078 |
| ISSR@25 | 0.879 | 0.816 | -0.063 |

### XGBoost 5-Fold CV Subgroup Metrics (v2 dataset, 30-feature mode)

| Metric | EBV (mean) | HPV16 (mean) | Gap |
|--------|-----------|-------------|-----|
| AUC-ROC | 0.766 | 0.587 | -0.179 |
| AUC-PR | 0.836 | 0.782 | -0.054 |
| ISSR@10 | 0.839 | 0.820 | -0.019 |
| ISSR@25 | 0.850 | 0.764 | -0.086 |

### Contributing Factors

1. **Data imbalance**: EBV has 470 training peptides vs 250 for HPV16 (1.9:1 ratio).
2. **Different immunogenic mechanisms**: HPV16 oncoproteins (E6/E7) have distinct immunodominance patterns compared to EBV latency and lytic antigens.
3. **Model is virus-agnostic**: A single decision boundary is learned for both viruses; HPV16-specific TCR-contact patterns may be underrepresented.
4. **Fold-level instability**: HPV16 AUC-ROC ranges from 0.595 to 0.693 across folds (RF), indicating high variance from the smaller sample.

### Implication

HPV16 peptide rankings should be interpreted with lower confidence than EBV rankings. Users conducting downstream validation should apply additional domain-expert review to HPV16 candidates.

## Strain and Metadata Limitations

1. **Strain-blind training**: Training data lacks strain-level granularity. The `virus` column only records "EBV" or "HPV16", not specific strains (e.g. B95-8 vs type-2 EBV, or which HPV16 gene a peptide derives from). The pipeline FASTA files use B95-8 reference sequences for EBV, but training epitopes may originate from mixed strains.
2. **Allele-blind training labels**: Immunogenicity labels are derived from IEDB Epitope Table filenames ("T-cell positive" / "T-cell negative") with no per-peptide HLA allele context. The model treats immunogenicity as allele-independent, though it uses per-allele binding features at inference.
3. **Protein context not used in training**: Training peptides have no gene/protein annotation. Therapeutically relevant HPV16 proteins (E6, E7) are not distinguishable from structural proteins (L1, L2) in the training signal.
4. **Label conflicts**: 201 cross-label peptides resolved by majority vote. 69 peptide labels flipped between v1 and v2 datasets.

## Class Imbalance and Operating Point

- v2 dataset: 2.36:1 positive-to-negative ratio (improved from v1's 5.58:1).
- At the recall-optimized threshold (0.3559), 82% of negatives are misclassified. This reflects the 99% recall operating point, not intrinsic model failure.
- Threshold selection should be application-dependent: high-recall for screening, balanced F1 for validation prioritization.

## TCR Feature Model Constraints

- TCR contact positions (p4-p8) assume canonical MHC-I binding geometry per Chowell et al. 2015.
- 8-mer peptides have p7 and p8 zero-imputed (only p4-p6 are informative), reducing discriminative power for the shortest peptides.
- Non-canonical peptide-MHC binding orientations are not modeled.
- Missing binding matrix entries default to all-zero vectors, which may bias predictions for rare peptides.

## Promotion Rule for Stronger Claims

Stronger biological claims require:

1. Additional external datasets and robust cross-cohort evaluation.
2. Prospective biological validation design with domain experts.
3. Updated evidence freeze and decision record documenting the new support level.
