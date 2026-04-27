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

All lists are defined in `src/features.py` and imported by all downstream modules.
