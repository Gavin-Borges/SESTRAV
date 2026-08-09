# 2026 Feature Schema Upgrade Roadmap

**Status:** Proposal - Phase 0 not yet started. No feature-schema code has changed as part of
this document; it is the analysis and plan only.
**Scope:** `feature_mode` 31/33/35 canonical + extended pipelines, v5 dataset, RF/XGB production
path, GNN research track.
**Primary evidence:** `results/cv_leakage_audit.csv` (reproduce with
`python scripts/audit_cv_leakage.py`), `models/v5/training_results_ablation.csv`.

## Executive summary

This roadmap was requested to rank seven 2026-era feature-schema upgrades (multidimensional
physicochemical descriptors, pMHC stability, self-proteome tolerance, agretope/epitope ratio,
proteasomal flanking, ESM-2 PCA, and AlphaFold/ESM-3 GNN features) for SESTRAV's peptide
immunogenicity model.

The audit that produced this document found a defect that has to be stated before any ranking is
meaningful: **the production cross-validation splitter does not group by peptide, and 71.0% of
every held-out test row has its exact peptide sitting in the training fold** (`results/
cv_leakage_audit.csv`, `production_splitter` / `overall_peptide_overlap_pct`). Every `feature_
mode=31` feature - the 20 physicochemical descriptors, the 10 fixed-panel MHCflurry binding
scores, and peptide length - is a pure function of the peptide string; the HLA allele column
never enters the vector. Rows sharing a peptide are therefore feature-identical, and an
identical Random Forest run (matched to production's exact config: `n_estimators=200`,
`random_state=42`, `class_weight=balanced`, unweighted) under the production splitter versus a
peptide-grouped splitter moves AUC-PR from 0.8347 to 0.6092 - a +0.2255 (+37.0%) inflation
attributable to leakage alone, not to any modeling choice. The two figures this project has cited
as the leakage-honest antidote to the retracted pooled 0.9368/0.7678 (`docs/claims_register.md`
D12) turn out to be downstream consumers of the same leaky OOF predictions and are themselves
measurably, if less severely, inflated: the per-virus mean (certified 0.751) reproduces at 0.7512
under the production splitter versus 0.6587 peptide-grouped (+14.0%), and the pooled honest
same-pathogen figure (certified 0.712) reproduces at 0.7124 versus 0.5989 peptide-grouped
(+19.0%) - see Section 2. The Tier A external benchmark is exposed by the same chain but the
leakage effect there is small (AUC-PR -0.0176 on the measurable subset); its real problem is that
the certified 0.828 is a **2026-05, 30-feature, unweighted, 200-tree** measurement mislabeled as
the canonical v5 `mode_31` figure (`docs/claims_register.md` D16) - a sound result for what it
is, but not a description of the shipped model.

The seven proposed upgrades separate by honest, peptide-grouped AUC-PR delta into a **-0.0037 to
+0.0096** band (measured for three of them; estimated for the rest against the same baseline).
That band is roughly 23x smaller than the leakage inflation. Ranking features against a harness
that cannot resolve a delta twenty times larger than itself is not proposal-level rigor, and
this repository holds itself to a claims register (`docs/claims_register.md`) precisely to catch
that class of mistake before it reaches a document.

**Recommendation:** treat this as a two-phase program. Phase 0 repairs the evaluation harness
(grouped CV, a fold-disjointness test, a vaccinia-decoy accounting fix). Phases 1-3 then execute
the feature work, re-ranked against the honest baseline. Sections 2-4 give the evidence; Section
5 gives the phased plan.

---

## 1. Codebase and pipeline sanity audit

Environment: `conda activate sestrav`. Verified package
versions: pandas 3.0.3, numpy 2.4.6, scikit-learn 1.8.0, xgboost 3.2.0, mhcflurry 2.2.1, torch
2.13.0+cu130, torch-geometric 2.7.0, snakemake 9.23.1, pytest 9.1.1. **`transformers` is not
installed in this environment** - this blocks any local ESM-2 regeneration (Section 4, proposals
6 and 7) until `pip install ".[gnn]"` is run.

### 1.1 Pipeline structure (premise correction)

This document was scoped assuming a `workflow/` directory with Snakemake rules and a
`config/config.yaml`. Neither exists. The actual layout:

- `Snakefile` -> `include: "pipeline.smk"` (root-level, not `workflow/rules/*.smk`)
- `pipeline.smk` + `standardize_outputs.smk` define all 14 rules (`generate_peptides`,
  `predict_binding`, `extract_features`, `score_immunogenicity`, `generate_hard_decoys`,
  `qc_dataset`, `train_ann`, `train_gnn`, `full_validation_report`, `extract_verify_data`,
  `evaluate_verify_gnn`, `run_prime`, `run_predig`, `standardize_predictor_outputs`)
- `config.yaml` at the repo root (no `config/` directory)
- The GNN track is `src/gnn/`, `src/train_gnn.py`, `src/verify/` - there is no
  `src/gnn_module/`

### 1.2 Snakemake DAG - PASS

```
snakemake --snakefile pipeline.smk --configfile tests/fixtures/dag_smoke/config.smoke.yaml --dry-run --cores 1
```
resolves cleanly: 12 jobs, matching the CI gate in `.github/workflows/ci.yml`. No structural
defect found in the pipeline layer.

### 1.3 Current feature schema - as implemented, confirmed by direct read of `src/features.py`

```python
FEATURE_COLUMNS_30 = PHYSICO_COLUMNS + BINDING_ALLELE_COLUMNS        # 20 + 10
FEATURE_COLUMNS_31 = FEATURE_COLUMNS_30 + ["peptide_length"]          # 31 (canonical, config.yaml default)
FEATURE_COLUMNS_33 = FEATURE_COLUMNS_31 + ["netchop_score", "tap_score"]                        # 33
FEATURE_COLUMNS_35 = FEATURE_COLUMNS_33 + ["self_similarity_max_identity", "self_similarity_exact_match"]  # 35
FEATURE_COLUMNS_50 = EXPANDED_PHYSICO_COLUMNS + BINDING_ALLELE_COLUMNS  # 40 + 10 (8 scales x 5 positions)
```

Physicochemical scales are hand-transcribed literature dicts in `src/features.py`, not a library
dependency: Kyte-Doolittle 1982 hydrophobicity, Zamyatnin 1972 Van der Waals volume, an
aromaticity/charge indicator, Vihinen 1994 flexibility, Zimmerman 1968 bulkiness, Hopp & Woods
1981 hydrophilicity, and a hand-set (non-literature) "upward-facing probability" proxy keyed only
on `(length, position)`.

Position handling (`get_tcr_positions`, `src/features.py`): p4-p6 are N-terminal-anchored at
fixed 0-based indices 3/4/5; p7/p8 are C-terminal-relative at `len-3`/`len-2` and are zero-
imputed when they collide with p6 or reach the C-terminal anchor (8-mers). Anchor positions
P2/P9 are **deliberately not extracted** - the code comment states this is to avoid anchor-
binding contamination of the TCR-contact signal. This is directly relevant to proposal 4
(Section 4) and to Amendment/claims-register entries about the physico/binding split.

**Edge-case handling, confirmed:** unknown amino acids and out-of-range positions zero-impute;
peptides absent from the binding matrix zero-fill silently (no count, no warning -
`src/train_classifier.py`); missing antigen-processing cache entries impute the cache-wide
median (`src/features.py`, count printed); missing self-similarity entries default to 0.0
(count printed). HLA allele-string normalization is implemented independently at least seven
times across `src/`, `scripts/`, `functions/`, and `app/` with no shared function - not a defect
this roadmap needs to fix, but worth flagging as adjacent technical debt.

### 1.4 Data leakage and split isolation - FAIL (see Section 2 for the full finding)

`src/train_classifier.py` cross-validates with `MultiStratifiedKFold`
(`src/ml_utils.py`), which accepts a `peptides=` argument but uses it only to bin peptide
length for stratification, never as a fold group. No `GroupKFold` or `StratifiedGroupKFold`
exists anywhere in the tracked tree. `LeaveOneGroupOut` on `metadata["protein"]` exists but is
opt-in (`--lopo`) and off by default. This is audited quantitatively in Section 2.

A second, related finding: every split executed in this audit printed `Composite stratum
minimum count 2 < min_stratum_size=5; falling back to label-only stratification.` The HLA-
supertype / negative-origin / length stratification `MultiStratifiedKFold` exists to provide
never activates on the v5 dataset - it silently degrades to plain `StratifiedKFold` on every
fold of every run.

### 1.5 freeze_mode and OpenSSF compliance - PASS, and compatible with additive feature work

`config.yaml` sets `freeze_mode: true` and `dataset_governance.require_checksum_match_in_freeze_
mode: true` against a pinned dataset checksum; `src/data_curation_qc.py` raises `RuntimeError`
on mismatch. `functions/stage4_immunogenicity_scoring.py` selects the scoring column set by
`model.n_features_in_`, so a new additive `feature_mode` does not by itself break freeze mode or
invalidate legacy artifacts - only an in-place dataset edit would require a governed version
bump. Supply-chain posture (hash-pinned lockfiles, SBOM freshness gate, frozen scientific-core
dependencies in `dependabot.yml`, CodeQL/semgrep/bandit/Scorecard) is strong and is a binding
constraint on which proposals are admissible (Section 3).

---

## 2. The measurement problem: peptide-level cross-validation leakage

Reproducible via `python scripts/audit_cv_leakage.py`, output `results/cv_leakage_audit.csv`
(provenance sidecar: `results/cv_leakage_audit.csv.provenance.json`, dataset SHA-256
`1c596ab7f80f33fb01d7d302f37db2cb5e824166c0dbaec41720d45414426ea7`, seed 42, RF n_estimators=200 -
matched to `src/train_classifier.py`'s exact production config).

**Dataset shape.** v5 active set: 35,597 rows, 16,360 unique peptides. 26,086 rows (73.3%) share
a peptide with at least one other row. `Orthopoxvirus vaccinia` - not one of the nine target
viruses - contributes 21,432 active rows, all label=0, 77.8% of all active negatives.

**Per-fold overlap under the production splitter** (`MultiStratifiedKFold`, mode 31):

| Fold | Test rows | Test rows whose exact peptide is also in train | Overlap |
|---|---|---|---|
| 0 | 7,120 | 5,056 | 71.0% |
| 1 | 7,120 | 5,076 | 71.3% |
| 2 | 7,119 | 5,054 | 71.0% |
| 3 | 7,119 | 5,053 | 71.0% |
| 4 | 7,119 | 5,041 | 70.8% |
| **Overall** | **35,597** | **25,280** | **71.0%** |

**Direct A/B, identical RF (200 trees, seed 42, class_weight=balanced - matched to
`src/train_classifier.py`'s production config), identical data, only the splitter changed:**

| Splitter | AUC-PR | AUC-ROC |
|---|---|---|
| Production (`MultiStratifiedKFold`, ungrouped) | 0.8347 +/- 0.0065 | 0.9440 |
| Peptide-grouped (`StratifiedGroupKFold`) | 0.6092 +/- 0.0229 | 0.8130 |
| **Leakage-attributable inflation** | **+0.2255 (+37.0%)** | **+0.1310** |

The production-splitter reproduction lands within 0.0035 AUC-PR / 0.0011 AUC-ROC of the certified
ledger cell (`models/v5/training_results_mode31.csv`: AUC-PR 0.8312, AUC-ROC 0.9429) - the
residual gap is CV-fold-shuffle variance between independent runs (same `random_state`, same
estimator count), not a modeling-choice difference, and is small enough that the reproduction is
treated as validated.

`STATE.md` already documents a related finding for `feature_mode=166` (297/4,169 peptides
spanning CV folds) and refers to a "grouped-CV guard" - that guard does not exist in code, and
no test in `tests/` asserts fold disjointness by peptide. This roadmap's Phase 0 (Section 5)
closes that gap.

**Vaccinia decoy bloc.** Excluding `Orthopoxvirus vaccinia` from the peptide-grouped evaluation
moves AUC-PR from 0.6092 to 0.7693 (+0.1601) while AUC-ROC falls 0.8130 -> 0.7427. The pooled
headline metric, as currently computed, is substantially a measure of "can the model recognise
vaccinia peptides," not of virus-immunogenicity discrimination across the nine target pathogens.

**The "honest" correctives are also leakage-inflated, just less severely.** This project has
cited two figures as the leakage-free antidote to the retracted pooled 0.9368/0.7678
(`docs/claims_register.md` D12): the per-virus mean AUC-ROC (`results/per_virus_eval_v5_mode31.csv`,
0.751) and the pooled honest same-pathogen AUC-ROC (`results/pooled_honest_same_pathogen.csv`,
Def A, 0.712). Both are computed from `models/v5/rf_oof_predictions_mode31.csv`, which is written
by the same `MultiStratifiedKFold` path as the pooled headline - not an independent splitter.
Reproducing both under a matched peptide-grouped splitter (same RF config as above):

| Metric | Production splitter | Peptide-grouped | Inflation |
|---|---|---|---|
| Per-virus mean AUC-ROC (9 target viruses) | 0.7512 +/- 0.1168 (certified: 0.751) | 0.6587 +/- 0.0908 | +0.0925 (+14.0%) |
| Pooled honest same-pathogen AUC-ROC (Def A) | 0.7124 (certified: 0.712) | 0.5989 | +0.1135 (+19.0%) |

The production-splitter reproductions match the certified figures almost exactly, validating the
methodology. Both correctives carry real, non-trivial leakage inflation - smaller than the pooled
headline's 37%, but not zero. Only the LOO cross-virus table
(`results/loo_cross_virus_v5_clean.csv`) is confirmed genuinely independent of this finding: it
trains a fresh model per held-out virus with explicit virus-level partitioning
(`scripts/run_loo_cross_virus_v5.py`), never uses `MultiStratifiedKFold`, and excludes gold-standard
peptides from training.

**The Tier A external benchmark is exposed by the same chain, but the leakage effect there is
small.** The SESTRAV arm of Tier A is the `rf_oof_score` column, which traces to the same ungrouped
OOF output (`src/train_classifier.py:781-786` -> `src/prepare_external_validation_inputs.py:100` ->
`scripts/run_tier_a_benchmarks.py:269`), so Tier A is not an independent held-out field. Measured on
the 414 field peptides resolvable to an active (non-quarantined) v5 row, changing only the
splitter:

| Metric | Production splitter | Peptide-grouped | Delta |
|---|---|---|---|
| AUC-PR | 0.8932 | 0.8756 | -0.0176 (-2.0%) |
| AUC-ROC | 0.5924 | 0.5680 | -0.0244 |
| ISSR@10 | 1.0000 | 0.9024 | -0.0976 |

Two limits on that delta, both mandatory to carry: the n=414 subset is **not representative** of the
certified n=704 field (positive rate 0.838 vs 0.696), so its absolute AUC-PR is not comparable to
the certified 0.828 and only the within-subset delta is interpretable; and those 414 are precisely
the peptides that DO sit in the training corpus, so this is closer to an upper bound on the
field-wide effect than to an underestimate. The practical conclusion is that peptide leakage does
**not** explain the Tier A headline - but a separate defect does bear on it: the Tier A SESTRAV arm
is a **2026-05, 30-feature, unweighted, 200-tree** measurement, not the v5 mode-31 result
`README.md` labels it as. Two independently verified facts establish this without relying on any
reproduction: `results/external_validation_input.csv` has exactly one commit in history,
`f360b90` (2026-05-23), at which `src/train_classifier.py` declared `--feature-mode
choices=[21, 30, 50, 166]` and `prepare_features_31` appeared zero times - `feature_mode=31` did
not exist yet, entering 26 days later at `27cdc61` (2026-06-18); and all 704 stored
`rf_oof_score` values are exact multiples of 1/200 (704/704, against 362/704 for 1/500), which
fingerprints `n_estimators=200`. The training corpus was the 720-row root
`immunogenicity_dataset.csv` at `69e0e5c` (the field is exactly 720 minus the 16
`GOLD_STANDARD_EPITOPES`); that file is **not tracked at HEAD** - it was deleted at `ec9aba0` -
but `69e0e5c` is an ancestor of `main`, so it remains recoverable from a clean clone. Of the 704
field peptides, **468** exist somewhere in v5 and **236** exist in neither v4 nor v5 (none are
v4-only); only **414** resolve to an active, non-quarantined v5 row, because a further **54**
appear only in quarantined rows - so **290** cannot be scored by the v5 model. An independent
reproduction of that 30-feature, unweighted, 200-tree configuration on the 720-row corpus
reports a bit-exact match to all three certified cells; a second run of the
same nominal configuration landed close but not exact (AUC-PR 0.8247, MAD 0.058), so
bit-exactness is **reported, not independently confirmed** - the two decisive facts above do not
depend on it. So the figure is sound for what it is, and mislabeled. Recorded as
`docs/claims_register.md` D16. The consequence for this roadmap: **the v5 mode-31 model has never
been evaluated on the full Tier A field and cannot be**, so any Phase 0 re-baseline that touches
Tier A necessarily produces a different, smaller (n=414) field whose numbers are not comparable to
the published ones.

**Cross-fold imputation (secondary finding).** `feature_mode=33`/`35` impute missing antigen-
processing scores with medians computed over the full cache before cross-validation begins; this
run's log line (`Imputed 4502 missing antigen processing scores with cache medians`) confirms
4,502 rows receive a statistic derived from all folds, not just the training fold. Feature-
selection leakage is not present - the feature sets are fixed constants, not fit at CV time.

**Why this matters for the feature ranking below:** the honest, peptide-grouped feature-mode
deltas measured in Section 4 (-0.0037 to +0.0096) are roughly 23x smaller than the 0.2255
leakage inflation. A ranking built on the ungrouped harness is not measuring the features; it is
measuring noise in how much of each mode's signal happens to correlate with duplicate-peptide
placement.

---

## 3. Feature trade-off matrix

Verdicts are relative to the peptide-grouped mode-31 baseline (AUC-PR 0.6092). "Measured"
entries come from `results/cv_leakage_audit.csv`; "estimated" entries are not yet run and are
flagged as such - do not cite them as certified numbers.

| Feature | Honest AUC-PR delta | Compute / latency overhead | Dependency footprint & OpenSSF risk | Verdict |
|---|---|---|---|---|
| 5D Z-scale / 8D VHSE descriptors | **+0.0096 (measured, mode 50 proxy)** | None - lookup-table swap, no inference-time cost change | Zero new dependencies. No supply-chain surface. | **Adopt** |
| N/C-terminal flanking + ERAAP trimming (real, replacing the mode-33 mock) | Estimated, not yet measured (repair, not new feature) | Requires source-protein flank sequence at inference time; one extra PSSM evaluation per peptide, cacheable | Zero new dependencies - `src/antigen_processing.py` already implements the real PSSMs (Keller 2020, Doytchinova 2004); the change is wiring, not a new library | **Adopt as repair** |
| Agretope/epitope (P2/P9 vs P4-P8) disconnect ratio | Estimated; genuinely novel signal, not measured on this dataset | ~3x MHCflurry calls per peptide (alanine-scan mutants), cacheable in the existing binding-matrix pattern | Zero new dependencies | **Prototype** |
| pMHC stability (t1/2, NetMHCstabpan) | Estimated from literature; orthogonal to affinity | External predictor call per peptide-allele pair; cacheable | DTU academic-license binary, non-redistributable; repo's one existing DTU integration (NetChop) is currently a silent mock (Section 4) - repeat risk is high without a hard-fail contract | **Gate** on a real (non-mock) integration |
| ESM-2 embeddings + PCA for RF/XGB | Estimated; existing `30_esm` mode suggests limited RF headroom at this dataset size | Model forward pass per peptide (cacheable), PCA fit must be inside the fold | `transformers` not installed locally; `src/features.py` ESM loader currently falls back to a SHA256-seeded **random** vector on load failure - a silent-garbage pattern that must be fixed before this is trustworthy | **Research only** |
| Self-proteome tolerance (RSAT / foreignness) | **-0.0037 (measured, existing mode 35)** | None - already implemented, binary exact-match only | Zero new dependencies | **Reject on this dataset** (Section 4 explains the structural confound) |
| AlphaFold3/Boltz-1 pLDDT + ESM-3/ESM-C in GINEConv | Not estimated - GNN promotion Gate 1 (AUC-PR >= 0.85) is unreachable at the current honest baseline | Structure prediction per peptide-MHC complex; GPU-bound | AlphaFold3 weights are gated/non-commercial - direct OpenSSF/redistribution conflict; Boltz-1 is licence-viable; `src/verify/structural_gnn.py` currently fabricates idealised coordinates rather than using real structures | **Defer** |

---

## 4. Scientific evaluation, in ranked order

### 4.1 Adopt: 5D Z-scales / 8D VHSE multidimensional descriptors

Mode 50 (`EXPANDED_PHYSICO_COLUMNS`, 8 scales x 5 positions = 40 physico features) is the
existing proxy for "more physicochemical dimensions" and is the largest feature-side gain
measured in this audit: peptide-grouped AUC-PR 0.6188 vs 0.6092 for mode 31 (+0.0096,
`results/cv_leakage_audit.csv`, metric `mode50_grouped_auc_pr`). Z-scales are themselves a PCA
reduction of ~29 physicochemical properties (Sandberg et al. 1998), so z1-z3 partially re-span
the existing KD hydrophobicity / Van der Waals volume / charge scales already in mode 31 -
expect a modest, not transformative, gain consistent with the mode-50 measurement, not a
step-change. Implementation is a pure lookup-table addition with no new dependency and no
inference-time cost change, which makes it the highest expected-value-per-engineering-hour item
on this list.

### 4.2 Adopt as repair: N/C-terminal flanking and ERAAP trimming

This is not really a new-feature proposal once the codebase is read closely. `feature_mode=33`
already ships `netchop_score`/`tap_score`, but `scripts/precompute_antigen_processing.py`
computes them via `query_netchop(..., mock_fallback=True)` in `src/external_predictors.py`,
whose own script comment states the values "are NOT real NetChop 3.1 / TAPreg predictions" and
whose generator uses `hash(char + pep)` - unstable across processes unless `PYTHONHASHSEED` is
pinned. Meanwhile `src/antigen_processing.py` contains a real ERAP/TAP PSSM implementation
(transcribed from Keller 2020 and Doytchinova 2004) that `src/train_classifier.py` never
imports - dead code sitting next to a mock that is documented as real in `docs/model_cards/
rf_33feature_integrated.md`. The correct 2026 upgrade path is to retire the mock, wire the real
PSSM code into the mode-33 build, add genuine N/C-terminal flanking probability using source-
protein context already available via `docs/antigen_accessions.md`, and correct the model card.
This converts a scientific-integrity liability into whatever real signal the PSSM approach
actually carries - which should be measured honestly (peptide-grouped) rather than assumed.

### 4.3 Prototype: agretope/epitope (P2/P9 vs P4-P8) disconnect ratio

The repo's own design choice supports this as the most promising *novel* signal on the list:
`src/features.py` deliberately excludes P2/P9 anchor positions from the feature vector to avoid
anchor-binding contamination of the TCR-contact signal, which means the current model has zero
information about anchor-vs-bulge binding-energy disconection. A cacheable in-silico alanine
scan (mutate P2/P9, re-score with the existing MHCflurry pipeline, take the delta against P4-P8
physicochemical reactivity) needs no new dependency, extends a pattern (`mutate_anchors` already
exists as an evaluation probe in `src/verify/sestrav_evaluator.py`) rather than inventing one,
and is genuinely orthogonal to everything currently in the vector. Recommended gate: ship only if
it clears the measured mode-to-mode noise floor (+/-0.024 std, from the mode-31/35/50 comparison
in `results/cv_leakage_audit.csv`) under peptide-grouped CV.

### 4.4 Gate: pMHC stability (t1/2 via NetMHCstabpan)

Well-supported in the literature as an affinity-orthogonal correlate of immunogenicity. The
concern here is not the biology, it is the repo's own recent history with exactly this class of
integration: NetChop (Section 4.2) is a DTU academic-license web service that this codebase
already integrated once, and that integration currently ships silent mock values labeled as real
in a public model card. NetMHCstabpan is DTU-licensed and non-redistributable on the same terms.
Before this is adopted, the integration must have a hard-failure contract - no `mock_fallback`
default, an explicit `RuntimeError` (matching the `freeze_mode` pattern already used elsewhere in
this codebase) if the service is unavailable, and provenance recorded the way `models/peptide_
binding_matrix_v5.provenance.json` records MHCflurry provenance.

### 4.5 Research only: ESM-2 embeddings + PCA for tree models

`transformers` is not installed in the local `sestrav` environment, and a `30_esm` feature mode
already exists in `src/features.py` (350 features: 30 baseline + ESM CLS token), so this is not
a from-scratch proposal - it is an unmeasured extension of existing code. Two concerns: at
16,360 unique peptides, adding 25-50 dense, correlated PCA components to a Random Forest is
unlikely to move the needle much past the mode-50 result (+0.0096), since RF importance already
concentrates on positional physico features per the mode-31 model card. More urgently: `src/
features.py`'s ESM loader currently falls back to a SHA256-seeded **random** 320-dimensional
vector when the model fails to load - the same silent-garbage failure mode as the NetChop mock
in Section 4.2. That fallback must be converted to a hard failure before any ESM-2 feature work
is trustworthy enough to measure, let alone ship. PCA components must be fit inside the CV fold,
not on the pooled embedding matrix.

### 4.6 Reject on this dataset: self-proteome tolerance (RSAT / foreignness)

Already implemented as `feature_mode=35` and already measured in this audit: peptide-grouped
AUC-PR 0.6055 vs 0.6092 for mode 31 (-0.0037, `results/cv_leakage_audit.csv`, metric `mode35_
grouped_auc_pr`), consistent in direction with the certified ungrouped ablation (`models/v5/
training_results_ablation.csv`: 33 -> 35 is -0.0002). Beyond the null result, there is a structural reason
this feature cannot generalise on the current dataset: `scripts/generate_hard_decoys.py` builds
the majority of hard negatives *by sampling the human self-proteome*. A "is this peptide human"
feature therefore predicts "negative" largely by construction of the decoy set, not by learned
immunological tolerance. The self-similarity implementation is also binary (exact 9-mer match
only, not graded identity - the code's own docstring says partial-identity alignment "would be
too slow") which caps its ceiling even where it is not confounded. Revisit only on a dataset
whose negative class is assay-derived rather than decoy-generated from the human proteome.

### 4.7 Defer: AlphaFold3/Boltz-1 pLDDT + ESM-3/ESM-C GINEConv features

Two independent blockers, either one sufficient to defer. First, AlphaFold3 model weights carry
a non-commercial, gated license - a direct conflict with this repository's OpenSSF supply-chain
posture (hash-pinned, redistributable dependencies only); Boltz-1 is license-viable but pLDDT
confidence on an isolated 9-mer peptide (rather than the full pMHC-TCR complex) carries limited
structural information. Second, and more decisively: `src/verify/structural_gnn.py` currently
fabricates idealised backbone coordinates (`generate_canonical_groove_coords`) rather than using
real structures, and the GNN promotion Gate 1 threshold (AUC-PR >= 0.85, `src/verify/promote_
gnn.py`) is unreachable relative to even the leaky RF ceiling (0.831) and dramatically
unreachable relative to the honest peptide-grouped baseline (0.6092). Spending compute on richer
GNN node features before the promotion gates are re-baselined against an honest number is
premature.

---

## 5. Phased refactoring plan

Backwards compatibility throughout: every new `feature_mode` value is additive; `feature_mode`
stays a runtime config knob (`config.yaml`); `functions/stage4_immunogenicity_scoring.py`
already selects the scoring column set by `model.n_features_in_`, so existing artifacts (for
example `models/rf_31feature_integrated.joblib`) continue to load and score unchanged after any
of the additions below ship.

### Phase 0 - Repair the instrument (blocking; nothing in Phases 1-3 is measurable before this)

1. **Add a peptide-grouped splitter.** Extend `src/ml_utils.py` with a grouped mode built on
   `sklearn.model_selection.StratifiedGroupKFold`, reusing the existing
   `make_stratification_key` for the label/origin/supertype/length composite while peptide
   becomes the group. Thread it through `_cross_validate` in `src/train_classifier.py` behind a
   config flag, defaulting to grouped for any new certified run.
2. **Fix the silent stratification fallback.** The `min_stratum_size=5` fallback in
   `MultiStratifiedKFold.split` fires on every v5 split observed in this audit. Either coarsen
   the composite key until it reliably survives, or raise instead of silently degrading to
   label-only stratification.
3. **Add a fold-disjointness test.** `tests/` has no assertion today that CV folds are peptide-
   disjoint - add one, closing the gap `STATE.md` already flags as an unimplemented "grouped-CV
   guard."
4. **Report the vaccinia-excluded metric alongside the pooled metric**, or move the vaccinia
   decoy bloc to a declared decoy-only partition, so the headline number is not 60% one non-
   target pathogen.
5. **Move antigen-processing imputation inside the fold** (`src/features.py`).
6. **Re-baseline every certified number that depends on the ungrouped splitter.** This step
   touches `docs/claims_register.md`, `results/`, the model cards, and `STATE.md`, and is gated
   by the `*_results_guard.py` test family and `src/artifact_guard.py` by design - sequence it
   deliberately rather than as a drive-by edit. Expect the certified headline to move from
   ~0.83 toward ~0.61 (or ~0.77 excluding the vaccinia bloc); this is an outward-facing change
   to numbers already published in `docs/paper.md` and OpenSSF evidence, and is called out as a
   standing decision for the repository owner below, not something this document resolves.

### Phase 1 - Cheap, dependency-free feature work (Section 4.1, 4.2)

7. Add `Z_SCALES`/`VHSE` tables to `src/features.py` alongside the existing `KD_HYDRO` etc.,
   define a new additive `FEATURE_COLUMNS_36` (5D x 5 positions + 10 binding + length), dispatch
   it in `train_classifier.py`'s mode selection, and register it in `VALID_FEATURE_MODES`
   (`scripts/batch_experiment_runner.py`). A/B against modes 31 and 50 under grouped CV.
8. Wire `src/antigen_processing.py`'s real PSSMs into the `feature_mode=33` build, retire the
   `external_predictors` mock path, regenerate `data/antigen_processing_cache.csv` with a
   provenance sidecar (mirroring `models/peptide_binding_matrix_v5.provenance.json`), and correct
   `docs/model_cards/rf_33feature_integrated.md`.

### Phase 2 - Prototype the one genuinely novel signal (Section 4.3)

9. New helper in `src/features.py` for the agretope/epitope disconnect ratio; cache alanine-scan
   deltas alongside the binding matrix using the existing binding-matrix caching pattern. Ship
   only if the grouped-CV delta clears the measured mode-to-mode noise floor.

### Phase 3 - Gated or deferred (Section 4.4-4.7)

10. NetMHCstabpan (4.4) behind a real-integration, hard-fail-on-unavailable contract. ESM-2 PCA
    (4.5) as a research branch once `transformers` is installed and the random-vector fallback
    in `src/features.py` is converted to a hard failure. AlphaFold3/Boltz-1 (4.7) only after the
    GNN promotion gates are re-baselined against the honest RF number from Phase 0.

---

## 6. Verification

- `pytest tests/ -q`, plus the new fold-disjointness test from Phase 0.
- `snakemake --snakefile pipeline.smk --configfile tests/fixtures/dag_smoke/config.smoke.yaml
  --dry-run --cores 1` continues to resolve 12 jobs.
- `ruff check .` and `mypy src/` (CI lint gate).
- `python tools/check_library_coverage.py --check`, then `pytest tests/ --cov-config=
  .coveragerc.library` against the 95% library-scope floor.
- `python scripts/data_qc_gate.py` to confirm the governance/checksum path is unaffected by
  additive feature-mode changes.
- Re-run `python scripts/audit_cv_leakage.py` after Phase 0 lands and confirm the production
  training path now reports the grouped number by default.

---

## 7. Standing decision for the repository owner

Phase 0 step 6 restates the project's certified headline result downward by roughly 0.23 AUC-PR
(or 0.07 if reported net of the vaccinia bloc instead), and also restates the two figures this
project has been citing as the leakage-honest corrective - the per-virus mean (0.751 -> 0.659,
-0.09) and the pooled honest same-pathogen figure (0.712 -> 0.599, -0.11) - both downward by a
smaller but still material amount. The Tier A headline (0.828) moves far less on leakage grounds
(-0.0176 AUC-PR on the measurable subset) but carries a separate, arguably harder problem: it is
a 2026-05, 30-feature, unweighted, 200-tree measurement mislabeled as the canonical v5 `mode_31`
result (D16). **Corrected 2026-08-09, and worth recording as a process note:** this paragraph
previously asserted that 0.828 "cannot be regenerated from tracked inputs at all" - the first,
since-retracted version of D16. That version was corrected twice: `a843385` (mislabeled, not
irreproducible) and `dd5a356` (30-feature/unweighted/200-tree, not 31-feature/500-tree). This
document tracked neither correction cleanly: `a843385` did edit it but missed this section,
leaving it at the original claim, while `dd5a356` never touched this file at all, so the sections
`a843385` did update were themselves left a generation behind. All three sites were reconciled
against the certified register in one pass on 2026-08-09. The number is sound and its provenance
is established - what is wrong is the label, so the cheap remedy is relabeling, not regeneration.
The genuinely hard constraint is different and survives: **the v5 mode-31 model has never been
evaluated on the full 704-peptide Tier A field and cannot be**, since 290 of those peptides are
absent from v5. So a re-baseline that touches Tier A is a replacement on a different, smaller
(n=414) field whose numbers are not comparable to the published ones - not a refresh of an
existing figure. These numbers currently reach `docs/paper.md`, the
OpenSSF evidence trail, and prior public communication. This document performs the analysis and states
the recommendation; it does not itself alter any published claim. Whether and when to execute
the re-baseline, and how to sequence the outward-facing disclosure, is a decision for the
repository owner - track it as a `/handoff` action item rather than as an automatic consequence
of this roadmap being accepted.
