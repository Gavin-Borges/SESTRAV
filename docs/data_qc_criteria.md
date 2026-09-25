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
| Per-stream row counts | hard equality against `dataset_governance.provenance.source_counts` | Enforced by `tools/check_dataset_provenance.py`, not by the QC gate. Catches the composition failures a ratio cannot see |

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
allele-matched non-binders) and 34,361 come from IEDB. Those are corpus-composition counts
and not training-pool membership: the 5,000 self-proteome decoys are all quarantined, so
they are absent from the within-CV training pool and enter only the LOO cross-virus
training pool, while the 3,112 allele-matched non-binders are not quarantined and sit
inside the target panel. `scripts/generate_hard_decoys.py` takes `--num-decoys` as an
absolute integer and contains no reference to the positive count, so the decoy count is not
a ratio and cannot track the positive pool.

**The bound therefore encodes the one property that is designed: v5 is deliberately
negative-dominated at roughly one positive per five negatives.**

**How the two edges are fixed.** The nearest modelled whole-stream failure on each side of
the as-built 0.2051 is the panels failing to merge, at 0.1634 below, and an IEDB export
returning only the out-of-panel block, at 0.2949 above. The window must exclude both.

- **Ceiling 0.29.** 0.2949 rounded inward to two decimals. This edge is tight against its
  failure by construction.
- **Floor 0.18.** Inward rounding of 0.1634 would give 0.17, and `[0.17, 0.29]` would also
  exclude both failures. **No modelled failure lands anywhere between 0.17 and 0.18**, so the
  two floors are coverage-equivalent: each rejects the same eleven of the twenty modelled
  modes. 0.18 is chosen as the tighter of two equivalents, because it gives an earlier signal
  on drift nobody modelled while still leaving a four-figure margin. 0.17 would leave 1,491
  rows, 0.18 leaves 1,066.

**Do not read the floor as derived by rounding.** It is not, and an earlier revision of this
section claimed both that the window was "the widest" available and that each edge was
"rounded inward to two decimals". Those two statements cannot both hold of 0.18, and the
inconsistency was caught by an independent re-derivation rather than by any gate.

**Margin: 1,066 rows**, meaning 1,066 is the largest number of positives that can be lost with
the bound still satisfied. The gate compares inclusively
(`class_ratio_bounds[0] <= class_ratio <= class_ratio_bounds[1]`), so losing exactly 1,066
gives 0.18002 and still PASSES; the first breach is at **1,067**. On the other three
directions the figure quoted is the first breaching value, not the last passing one: adding
5,928 negatives, adding 3,606 positives, or losing 12,432 negatives each breach. State which
convention a number uses, because this sentence previously mixed them.

**One population sits outside the ceiling, recorded so a future re-scope does not discover it
by failing.** The active, non-quarantined pool is 35,597 rows at 8,055 / 27,542, a ratio of
**0.2925** - above the 0.29 ceiling by 0.0025. The Scope paragraph below explains why the
gate governs the whole file rather than that pool, so this is not a live failure; but any
proposal to re-point the gate at the training pool must move the ceiling in the same change.

**Rejected alternatives, recorded so they are not re-proposed.** A window tight enough to
catch every modelled failure is `[0.198, 0.211]`, a margin of 249 rows or 0.49% of the
corpus. That is a checksum expressed as a ratio, and an exact SHA-256 checksum is already
pinned in `config.yaml` under `dataset_governance.provenance`. Scoping the gate to the
in-panel subset would let `[1.5, 4.0]` pass at 1.5104, but on a **37-row margin**, over nine
per-virus ratios spanning 0.606 to 3.341 of which seven sit below 1.5, and it would require
giving the gate a concept of virus panels and quarantine that it does not have.

**What this bound does NOT catch, stated explicitly rather than implied.** Class ratio cannot
detect a failure of the decoy streams at any scope: the 5,000 self-proteome decoys are
entirely quarantined, and every perturbation of either decoy stream up to 2x moves the
full-file ratio by at most 13%, landing inside the window. It also does not catch a
regression in the dedup key, nor selection of the wrong IEDB input file - `data/` carries
both `iedb_negatives_v5.csv` (32,506 rows) and `iedb_negatives_v5_merged.csv` (36,689), and
substituting one for the other moves the ratio only 0.2051 to 0.2180.

**Those failures are caught by a separate instrument, `tools/check_dataset_provenance.py`.**
It asserts per-stream row counts by hard equality against the fields
`data/immunogenicity_dataset_v5_provenance.json` already records, pinned in
`config.yaml` under `dataset_governance.provenance.source_counts`. It also checks the
sidecar's internal arithmetic, that the sidecar describes the shipped file by row count and
digest, and that its digest agrees with the checksum `freeze_mode` enforces. Read the two
gates together: the class ratio governs SHAPE and this one governs COMPOSITION, and neither
sees what the other does.

**Scope.** The bound governs the dataset file as a whole. `scripts/data_qc_gate.py` has no
concept of `is_quarantined`, `virus` or any sub-population, takes a single `--dataset` path,
and under `freeze_mode` that path is pinned by checksum. A pool-scoped bound would be a
change to the gate's contract, not a clarification of it.
