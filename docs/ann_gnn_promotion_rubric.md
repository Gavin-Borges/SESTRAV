# ANN/GNN Promotion Rubric

This document defines the strict, numeric thresholds and operational criteria that the optional Artificial Neural Network (ANN) and Graph Neural Network (GNN) modules must achieve to be promoted to "fundamental" release status in SESTRAV, replacing the canonical 30-feature RF/XGBoost baseline.

## 1. Baseline Enrichment Supremacy
The advanced architectures must demonstrate a significant, stable improvement over the simplistic binding-only baseline (maximum per-allele binding score).

- **Threshold**: $R10 \ge 2.0$ (ISSR@10 must be at least double the binding-only baseline).
- **Stability**: The lower bound of the 95% Bootstrap Confidence Interval for R10 must be $\ge 2.0$.
- **Denominator Requirement**: The binding-only ISSR@10 baseline must be $\ge 0.08$ to ensure a mathematically stable ratio.

## 2. External Tool Supremacy
The modules must demonstrably outperform leading external, publicly available tools (e.g., PRIME, PredIG) on holdout viral proteomes.

- **Threshold**: The AUC-PR of the ANN/GNN models must be statistically superior (Bootstrap 95% CI lower bound $> 0$) compared to PRIME and PredIG.
- **Negative Discrimination**: The advanced models must correctly down-rank at least 80% of confirmed "hard negative" decoy peptides compared to the binding-only baseline.

## 3. Cross-Virus Generalization
True biological utility requires generalizing beyond the training virus distributions.

- **Threshold**: The cross-virus transfer enrichment (e.g., training exclusively on EBV and testing on HPV16) must maintain an $R10 \ge 1.5$.
- **Validation**: Stress-testing outputs (e.g., `results/external_validation_cross_virus.csv`) must not exhibit an AUC-ROC drop of more than $15\%$ compared to intra-virus cross-validation.

## 4. Reproducibility & Operational Reliability
- **CI/CD Constraints**: The models must load natively via `.pt` or equivalent artifact without raising dimensionality errors (e.g., 50-feature matching).
- **Speed**: Stage 4 scoring for a full viral proteome (e.g., 5,000 peptides) must complete in under 5 minutes on standard CPU resources.
- **Provenance**: Training dataset versions and QC conflict thresholds (e.g., max 15% conflict ratio) must be immutably locked in `config.yaml`.
