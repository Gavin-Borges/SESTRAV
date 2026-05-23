# External Benchmark Comparison Report

Generated: 2026-05-22T15:38:59.228651+00:00

## Provenance

- **base_input:** results\external_validation_input.csv
- **oof_predictions:** models/rf_oof_predictions.csv
- **predig_version:** bsceapm/predig:latest
- **predig_run_date:** unknown
- **prime_version:** PRIME 2.1 (GfellerLab master + compiled PRIME.x)
- **prime_run_date:** unknown

## Head-to-Head Metric Table

| tool | n_peptides | auc_roc | auc_pr | issr_10 | issr_25 | precision_10 | recall_10 | ndcg_10 | precision_25 | recall_25 | ndcg_25 | R10 | R25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SESTRAV RF (30-feat) | 704 | 0.7255 | 0.8278 | 0.8429 | 0.8352 | 0.8429 | 0.1204 | 0.8764 | 0.8352 | 0.3000 | 0.8522 | 0.9788 | 0.9956 |
| Binding-only (max) | 720 | 0.6335 | 0.7999 | 0.8611 | 0.8389 | 0.8611 | 0.1225 | 0.8601 | 0.8389 | 0.2984 | 0.8428 | 1.0000 | 1.0000 |

## Rank Correlation Matrix (Spearman rho)

| tool_a | tool_b | rho | p_value |
| --- | --- | --- | --- |
| SESTRAV RF (30-feat) | Binding-only (max) | 0.4673 | 0.0000 |

## Gold-Standard Negative Discrimination

| tool | negatives_evaluated | pushed_down_vs_binding |
| --- | --- | --- |
| SESTRAV RF (30-feat) | 10 | 5 |
| Binding-only (max) | 10 | 0 |

## Per-Virus Breakdown

| tool | virus | n | auc_roc | auc_pr | issr_10 | issr_25 | precision_10 | recall_10 | ndcg_10 | precision_25 | recall_25 | ndcg_25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SESTRAV RF (30-feat) | EBV | 459 | 0.7586 | 0.8350 | 0.8889 | 0.8246 | 0.8889 | 0.1278 | 0.9043 | 0.8246 | 0.3003 | 0.8456 |
| Binding-only (max) | EBV | 470 | 0.6131 | 0.7815 | 0.8511 | 0.8120 | 0.8511 | 0.1235 | 0.8483 | 0.8120 | 0.2932 | 0.8194 |
| SESTRAV RF (30-feat) | HPV16 | 245 | 0.6546 | 0.8171 | 0.8750 | 0.8361 | 0.8750 | 0.1186 | 0.8893 | 0.8361 | 0.2881 | 0.8484 |
| Binding-only (max) | HPV16 | 250 | 0.7195 | 0.8709 | 0.9600 | 0.9194 | 0.9600 | 0.1319 | 0.9685 | 0.9194 | 0.3132 | 0.9322 |

## Statistical Significance (Bootstrap 95% CI)

| comparison | auc_pr_diff_mean | auc_pr_diff_ci_low | auc_pr_diff_ci_high | auc_pr_significant | issr10_diff_mean | issr10_diff_ci_low | issr10_diff_ci_high | issr10_significant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SESTRAV RF (30-feat) vs Binding-only (max) | 0.0371 | 0.0017 | 0.0727 | yes | -0.0042 | -0.1143 | 0.1000 | no |

## Figures

- `external_benchmark_roc_pr_curves.png`
- `external_benchmark_score_distributions.png`

## Dataset Summary

- Total labeled peptides: 720
- Complete rows (all tools scored): 704
- Class balance: 506 positive / 214 negative
