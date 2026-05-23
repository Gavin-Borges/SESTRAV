# SESTRAV Feature Glossary

## 22-Feature Schema

SESTRAV extracts 22 features per peptide for immunogenicity prediction. Training uses 21 features (binding_score excluded); the full 22-feature set is available at inference.

### TCR Contact Positions

Positions p4-p8 are the peptide residues that face the T-cell receptor in the pMHC-TCR complex, following the Chowell et al. (2015) / PredIG convention.

| Position | Index Rule | 8-mer | 9-mer | 10-mer | 11-mer |
|----------|-----------|-------|-------|--------|--------|
| p4 | Fixed: index 3 | 3 | 3 | 3 | 3 |
| p5 | Fixed: index 4 | 4 | 4 | 4 | 4 |
| p6 | Fixed: index 5 | 5 | 5 | 5 | 5 |
| p7 | C-terminal: length - 3 | zero* | 6 | 7 | 8 |
| p8 | C-terminal: length - 2 | zero* | 7 | 8 | 9 |

*8-mers: p7 and p8 overlap with p6 or the C-terminal anchor, so they are zero-imputed.

### Properties at Each Position

| Feature | Scale | Range | Source |
|---------|-------|-------|--------|
| `p{4-8}_hydrophobicity` | Kyte-Doolittle | [-4.5, +4.5] | Kyte & Doolittle, 1982 |
| `p{4-8}_aromaticity` | Binary | {0, 1} | F, W, Y, H = 1; others = 0 |
| `p{4-8}_vdw_volume` | Van der Waals volume | [48, 163] Angstrom^3 | Zamyatnin, 1972 |
| `p{4-8}_charge` | Formal charge at pH 7 | {-1, 0, +1} | K/R = +1, D/E = -1, others = 0 |
| `p{4-8}_flexibility` | Vihinen flexibility | [0.904, 1.102] | Vihinen et al., 1994 |
| `p{4-8}_bulkiness` | Zimmerman bulkiness | [3.4, 21.67] | Zimmerman et al., 1968 |
| `p{4-8}_hydrophilicity`| Hopp & Woods | [-3.4, 3.0] | Hopp & Woods, 1981 |
| `p{4-8}_upward_prob` | Upward-facing proxy | [0.0, 0.9] | Structural alignment proxy |

### Global Features

| Feature | Description | Range |
|---------|-------------|-------|
| `binding_score` | MHCflurry presentation_score | [0, 1] (0 during training) |
| `peptide_length` | Amino acid count | {8, 9, 10, 11} |

### Feature Column Lists (Legacy 22-Feature Schema)

- `FEATURE_COLUMNS` (22): All features including binding_score
- `TRAIN_FEATURE_COLUMNS` (21): Excludes binding_score (used for legacy 21-feature model training)

## 30-Feature Canonical Schema (Default)

The canonical release track replaces the single `binding_score` with 10 per-allele MHCflurry presentation scores, for a total of 20 physicochemical + 10 binding = 30 features.

### Per-Allele Binding Features

| Feature | Allele | Source |
|---------|--------|--------|
| `bind_A0101` | HLA-A*01:01 | MHCflurry presentation_score |
| `bind_A0201` | HLA-A*02:01 | MHCflurry presentation_score |
| `bind_A0301` | HLA-A*03:01 | MHCflurry presentation_score |
| `bind_A1101` | HLA-A*11:01 | MHCflurry presentation_score |
| `bind_A2402` | HLA-A*24:02 | MHCflurry presentation_score |
| `bind_B0702` | HLA-B*07:02 | MHCflurry presentation_score |
| `bind_B0801` | HLA-B*08:01 | MHCflurry presentation_score |
| `bind_B2705` | HLA-B*27:05 | MHCflurry presentation_score |
| `bind_B3501` | HLA-B*35:01 | MHCflurry presentation_score |
| `bind_B4402` | HLA-B*44:02 | MHCflurry presentation_score |

### Canonical Feature Column Lists

- `FEATURE_COLUMNS_30` (30): 20 physicochemical (p4-p8 x 4 properties) + 10 per-allele binding
- `FEATURE_COLUMNS_31` (31): FEATURE_COLUMNS_30 + peptide_length (optional)

## 50-Feature Expanded Schema

The expanded schema (`FEATURE_COLUMNS_50`) utilizes all 8 properties at each of the 5 TCR contact positions (40 physicochemical features) plus the 10 per-allele binding scores. It is available as a configurable track for experimental model training.

## 166-Feature Allele-Aware Schema

The allele-aware schema (`FEATURE_COLUMNS_ALLELE`) builds on the canonical 30 features by appending 136 HLA pocket pseudo-sequence features (34 canonical pocket residues × 4 properties: hydrophobicity, aromaticity, volume, charge). This enables pan-allele training methodologies.

All lists are defined in `src/features.py` and imported by all downstream modules.

## Ablation Study Feature Groups

The following feature groups are defined in `src/ablation_study.py` for systematic evaluation of feature contributions:

| Group | Count | Composition |
|-------|-------|-------------|
| `physico_20` | 20 | TCR-contact physicochemical features only (p4-p8 × 4 properties) |
| `binding_10` | 10 | Per-allele MHC binding features only |
| `sestrav_21` | 21 | physico_20 + peptide_length (legacy training track) |
| `combined_30` | 30 | physico_20 + binding_10 (canonical track) |
| `full_31` | 31 | combined_30 + peptide_length |

## Evaluation Metrics

All metrics are computed by `src/evaluate_metrics.py`. The `evaluate()` function returns a 10-key dictionary:

**Core metrics (used in all reports):**
- `auc_roc` — Area Under the ROC Curve
- `auc_pr` — Area Under the Precision-Recall Curve (primary metric)
- `issr_10` — Immune-Stimulating Success Rate at top 10%
- `issr_25` — ISSR at top 25%

**Extended metrics (added in v2.0):**
- `precision_10`, `precision_25` — Precision at top 10%/25%
- `recall_10`, `recall_25` — Recall captured at top 10%/25%
- `ndcg_10`, `ndcg_25` — Normalized DCG at top 10%/25%
