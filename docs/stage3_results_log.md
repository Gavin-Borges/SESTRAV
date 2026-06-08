# Stage 3 Computational Validation Results Log

This log records the out-of-distribution evaluation results for the SESTRAV model on independent viral cohorts.

| Cohort | Peptides | Pos / Neg | AUC-PR (95% CI) | AUC-ROC (95% CI) | ISSR@10 (95% CI) | Target Met? |
| --- | --- | --- | --- | --- | --- | --- |
| SARS-CoV-2 | 75 | 51 / 24 | 0.8045 `[0.6952, 0.9054]` | 0.6389 `[0.5075, 0.7663]` | 0.8571 `[0.5714, 1.0000]` | YES |
| Influenza A | 429 | 271 / 158 | 0.6599 `[0.5962, 0.7260]` | 0.5365 `[0.4829, 0.5944]` | 0.6429 `[0.5000, 0.8095]` | NO |

*Note: Bootstrap intervals estimated via N=2,000 resamples.*
