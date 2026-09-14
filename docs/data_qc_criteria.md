# SESTRAV Data Quality Control Criteria

## Peptide Filtering (Stage 1 + IEDB Loader)

| Criterion | Rule | Rationale |
|-----------|------|-----------|
| Length | 8-11 amino acids only | MHC class I binding groove accommodates 8-11mers |
| Amino acids | Standard 20 only (ACDEFGHIKLMNPQRSTVWY) | Non-standard residues (U, X, B, Z, J) lack physicochemical lookup values |
| MHC class | Class I only | Class II alleles (HLA-DR/DP/DQ) are excluded from the binding panel |

## IEDB Data Cleaning

| Step | Rule | Module |
|------|------|--------|
| Format detection | Auto-detect Epitope Table vs T-cell Assay format | `iedb_data_loader._detect_format()` |
| Label extraction | Epitope Table: from filename ("positive"/"negative"); T-cell Assay: from Qualitative Measure column | `iedb_data_loader._label_from_filename()`, `.map_label()` |
| Virus tagging | From filename pattern (EBV, HPV16, HPV11), handles "HVP16" typo | `iedb_data_loader._virus_from_filename()` |
| Duplicate resolution | Per-peptide majority vote: mean(label) >= 0.5 = positive | `iedb_data_loader.load_and_clean_iedb()` |
| Gold-standard hold-out | 16 records (15 unique peptides + FLRGRAYGL variant) excluded from training | `train_classifier.train_models()` |

## Feature Extraction QC

| Check | Rule |
|-------|------|
| Feature count | 22 per peptide in legacy mode (21 for training), 30 in canonical mode (20 physico + 10 allele binding) |
| Position overlap | p7/p8 zero-imputed when overlapping p6 or C-terminal anchor (8-mers) |
| Missing values | None allowed; all amino acid lookups have defaults |
| Binding score | Set to 0.0 during training; real MHCflurry values during inference |

## Model QC

| Check | Rule |
|-------|------|
| `n_features_in_` | Must equal 21 (legacy TRAIN_FEATURE_COLUMNS) or 30 (canonical FEATURE_COLUMNS_30) |
| `predict_proba` output | Shape (n, 2), rows sum to 1.0, all values in [0, 1] |
| Cross-validation | Stratified 5-fold, `random_state=42`, identical splits across models |
| Class imbalance | Handled via balanced weights (RF), scale_pos_weight (XGB), per-sample weighting (ANN) |

## Dataset Governance (admissibility gate)

Enforced by `scripts/data_qc_gate.py` against the single file named by `--dataset`, using
`dataset_governance.qc_thresholds` in `config.yaml`.

| Criterion | Rule | Rationale |
|-----------|------|-----------|
| Conflict ratio | `<= max_conflict_ratio` (0.15) | Groups on (peptide, allele); a peptide measured on two alleles is not a conflict |
| Null allele fraction | `<= max_null_allele_fraction` | Fails closed when no allele column resolves, since an unmeasurable fraction is not a passing one |
| Class ratio | within `class_ratio_bounds` | Corpus-specific; derived below |
| Peptide yield | `>= min_peptide_yield` (500) | Guards against a build that filtered away its own corpus |

### Derivation of `class_ratio_bounds` for the v5 corpus

**The previous bound `[1.5, 4.0]` was never derived for v5 and does not describe it.** It was
authored in `2c0fe5a` (2026-06-13), the same commit that created this gate, and fitted to the
then-shipped v3 corpus: 1,004 rows, 773 positive / 231 negative, ratio 3.3463. v2 measured
2.3645. Both sit inside the window. The quarantine concept did not yet exist anywhere in the
repository, so there was no sub-population for the bound to govern. The bound was already
wrong one corpus generation later: v4 measures 0.8346 and fails it. The window in which
`[1.5, 4.0]` was a true description of a shipped corpus ends at v3.

**What determines v5's ratio.** The build concatenates four parts, dedups on
(peptide, hla_allele) keep-first, then quarantines:

| Part | Rows in | Label | Designed or incidental |
|---|---|---|---|
| v4 positives | 6,687 | positive | designed, frozen artifact |
| Published panels | 9,206 (6,094 pos / 3,112 neg) | mixed | designed, fixed artifact |
| v4 hard decoys | 5,000 | negative | designed, carried forward frozen |
| IEDB negatives | 36,689 | negative | **incidental** - whatever the export returned |

Dedup removes 6,397 rows (4,069 positive, 2,328 negative), giving the shipped
**51,185 rows, 8,712 positive / 42,473 negative, ratio 0.2051**. That attribution is exact,
not estimated, and reconciles against `dedup_dropped` in the provenance sidecar.

**The numerator is entirely designed; the denominator is 19% designed and 81% incidental.**
Of 42,473 negatives, 8,112 are synthetic (5,000 self-proteome decoys plus 3,112
allele-matched non-binders) and 34,361 come from IEDB. `scripts/generate_hard_decoys.py`
takes `--num-decoys` as an absolute integer and contains no reference to the positive count,
so the decoy count is not a ratio and cannot track the positive pool.

**The bound therefore encodes the one property that is designed: v5 is deliberately
negative-dominated at roughly one positive per five negatives.** `[0.18, 0.29]` is the widest
window around the as-built 0.2051 that still rejects the loss, absence or duplication of any
whole non-decoy stream. Each edge is rounded inward, to two decimals, from the nearest such
failure: the panels failing to merge gives 0.1634 below, and an IEDB export returning only
the out-of-panel block gives 0.2949 above.

**Margin: 1,066 rows.** Breaching the floor requires losing 1,066 positives (12.2% of all
positives) or adding 5,928 negatives; breaching the ceiling requires adding 3,606 positives
or losing 12,432 negatives.

**Rejected alternatives, recorded so they are not re-proposed.** A window tight enough to
catch every modelled failure is `[0.198, 0.211]`, a margin of 249 rows or 0.49% of the
corpus. That is a checksum expressed as a ratio, and an exact SHA-256 checksum is already
pinned in `config.yaml` under `dataset_governance.provenance`. Scoping the gate to the
in-panel subset would let `[1.5, 4.0]` pass at 1.5104, but on a **37-row margin**, over nine
per-virus ratios spanning 0.606 to 3.341 of which six sit below 1.5, and it would require
giving the gate a concept of virus panels and quarantine that it does not have.

**What this bound does NOT catch, stated explicitly rather than implied.** Class ratio cannot
detect a failure of the decoy streams at any scope: the 5,000 self-proteome decoys are
entirely quarantined, and every perturbation of either decoy stream up to 2x moves the
full-file ratio by at most 13%, landing inside the window. It also does not catch a
regression in the dedup key, nor selection of the wrong IEDB input file - `data/` carries
both `iedb_negatives_v5.csv` (32,506 rows) and `iedb_negatives_v5_merged.csv` (36,689), and
substituting one for the other moves the ratio only 0.2051 to 0.2180. Those failures need
per-stream row-count assertions over the fields already recorded in
`data/immunogenicity_dataset_v5_provenance.json` (`v4_positives`, `v4_hard_decoys`,
`published_panels`, `iedb_negatives`, `dedup_dropped`), which is a separate and currently
unbuilt check.

**Scope.** The bound governs the dataset file as a whole. `scripts/data_qc_gate.py` has no
concept of `is_quarantined`, `virus` or any sub-population, takes a single `--dataset` path,
and under `freeze_mode` that path is pinned by checksum. A pool-scoped bound would be a
change to the gate's contract, not a clarification of it.
