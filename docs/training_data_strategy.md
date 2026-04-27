# SESTRAV Training Data Strategy

## Data Source

All training data comes from the IEDB (Immune Epitope Database) Epitope Table exports for EBV and HPV16 T-cell assay results.

## Dataset Mode Policy (v1)

SESTRAV v1 supports two sanctioned data modes for reproducible communication:

- **Mode A — Frozen baseline (default for release claims):**
  - Uses the currently frozen dataset and evidence set.
  - Used for all colloquium/release headline values.
- **Mode B — Expanded IEDB exploratory rerun:**
  - Allows increasing/changing IEDB input pool.
  - Must be reported as exploratory and compared side-by-side with Mode A.

Do not merge conclusions across these modes without an explicit comparison table.

## Dataset Versioning Rules

Every training dataset update must receive a version ID:

- Format: `IEDB-YYYYMMDD-<scope>-vN`
- Example: `IEDB-20260424-EBV_HPV16_EXPANDED-v1`

For each version, record:

1. IEDB export date and query/filter criteria.
2. Source files used and row counts before/after QC.
3. Positive/negative counts and class ratio.
4. Gold-standard holdout count and peptide list checksum.
5. Output model artifacts and validation result file set.

## Dataset Composition

### Canonical (v2) Dataset

| Metric | Value |
|--------|-------|
| Total unique peptides | 720 |
| Positive (immunogenic) | 506 (70.3%) |
| Negative (non-immunogenic) | 214 (29.7%) |
| Class ratio | 2.36:1 |
| Viruses | EBV, HPV16 |
| Peptide lengths | 8-11 amino acids (MHC class I) |

### Archived (v1) Dataset

| Metric | Value |
|--------|-------|
| Total unique peptides | 928 |
| Positive (immunogenic) | ~787 (84.8%) |
| Negative (non-immunogenic) | ~141 (15.2%) |
| Viruses | EBV, HPV16 |
| Peptide lengths | 8-11 amino acids (MHC class I) |

## Processing Pipeline

1. **Load IEDB exports** — Handles both Epitope Table format (labels from filename) and T-cell Assay format (labels from Qualitative Measure column). Module: `src/iedb_data_loader.py`

2. **Filter** — Retain only 8-11mer peptides with standard amino acids (ACDEFGHIKLMNPQRSTVWY). Reject MHC class II alleles.

3. **Duplicate resolution** — Peptides appearing in both positive and negative exports are resolved by majority vote (mean label >= 0.5 = positive).

4. **Gold-standard hold-out** — 16 records (covering 15 unique epitopes + the FLRGRAYGL type-2 variant) are removed from training and reserved for pipeline validation.

5. **Feature computation** — 21 sequence-only features at TCR contact positions p4-p8. Binding score is set to 0 during training (IEDB Epitope Table exports lack allele information).

## Class Imbalance Handling

- **RandomForest**: `class_weight='balanced'` (automatic inverse-frequency weighting)
- **XGBoost**: `scale_pos_weight = n_negative / n_positive`
- **ANN (MLP)**: Per-sample loss weighting with inverse class frequency

## Cross-Validation

Stratified 5-fold CV with `random_state=42` for all models, ensuring identical train/validation splits for fair comparison.

## Gold-Standard Epitopes (15)

10 EBV + 5 HPV epitopes with well-characterized experimental immunogenicity, covering 7 HLA alleles. These are held out from training and used exclusively for pipeline validation. See `src/gold_standard.py` for the full list.

## Why binding_score Is Excluded from Training

IEDB Epitope Table exports contain only peptide sequences and T-cell assay outcomes — no allele information. Without knowing which HLA allele presented each peptide, MHCflurry binding predictions would be meaningless. Therefore, training uses 21 sequence-only features. During inference, the pipeline supplies real MHCflurry presentation_score values, and the model auto-detects its feature set via `n_features_in_`.

## Required Controls for Expanded Pool Reruns

When running Mode B (expanded IEDB), all of the following are required:

1. Run `src.data_bias_audit` and include summary outputs in `results/`.
2. Publish an updated `docs/iedb_data_counts.md` section for the new dataset version.
3. Regenerate `results/h2_tier_a_summary.csv` and `results/baseline_comparison.csv`.
4. Publish a baseline-vs-expanded comparison note in release documentation before external sharing.
