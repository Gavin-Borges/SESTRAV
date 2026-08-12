# 2026 Feature Schema Upgrade Roadmap

**Status:** **Phase 0 EXECUTED 2026-08-10** (`docs/claims_register.md` D15, remediated) - the
evaluation harness has been repaired and every certified v5 number re-baselined under a
peptide-grouped splitter. Phases 1-3 (the feature-schema work) remain a proposal; no
feature-schema code has changed. **The measurements throughout this document are the
pre-remediation ones that motivated Phase 0 and are preserved as the historical record** - where
this document says the certified figures are 0.8312 / 0.751 / 0.712, those are the numbers
Phase 0 retracted, now 0.6058 / 0.658 / 0.6015. Do not cite this document for current figures;
cite `docs/claims_register.md` D15 or the ledgers under `models/v5/` and `results/`.
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
measurably, if less severely, inflated: the per-virus mean (then-certified 0.751, since
RETRACTED by D15) reproduces at 0.7512
under the production splitter versus 0.6587 peptide-grouped (+14.0%), and the pooled honest
same-pathogen figure (then-certified 0.712, since RETRACTED by D15) reproduces at 0.7124
versus 0.5989 peptide-grouped
(+19.0%) - see Section 2. The Tier A external benchmark does not carry the same exact-duplicate
exposure: its 720-row corpus has zero duplicate peptides, so that mechanism is a structural no-op
there, and the AUC-PR -0.0176 delta previously cited here as evidence of exposure in fact measures
a different, hypothetical question - a fresh v5-trained model re-scoring the resolvable subset, not
the certified pathway (`docs/claims_register.md` D22). A separate, unquantified risk applies
instead: 32.1% of the 704-peptide scored pool has a substring-level near-duplicate elsewhere in the
pool (D22). Tier A's real problem is that
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
(grouped CV, a fold-disjointness test, a vaccinia-bloc accounting fix). Phases 1-3 then execute
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
ledger cell (`models/v5/training_results_mode31.csv`, **as it stood before Phase 0**: AUC-PR
0.8312, AUC-ROC 0.9429 - both RETRACTED as peptide-leakage-inflated per D15; that file now holds
0.6058 / 0.8137) - the
residual gap is CV-fold-shuffle variance between independent runs (same `random_state`, same
estimator count), not a modeling-choice difference, and is small enough that the reproduction is
treated as validated.

`STATE.md` already documents a related finding for `feature_mode=166` (297/4,169 peptides
spanning CV folds) and refers to a "grouped-CV guard" - that guard does not exist in code, and
no test in `tests/` asserts fold disjointness by peptide. This roadmap's Phase 0 (Section 5)
closes that gap.

> **Closed 2026-08-10 (Phase 0, `30f1b76`).** Both clauses above are now false and are retained
> only as the historical statement of the gap. The guard exists as
> `PeptideGroupedKFold` (`src/ml_utils.py`), and fold disjointness is asserted at two layers:
> `tests/test_ml_utils.py::test_peptide_grouped_kfold_folds_are_peptide_disjoint` on the splitter,
> and `tests/test_train_classifier.py` on the OOF frame that `_cross_validate` actually emits. A
> paired negative control
> (`tests/test_ml_utils.py::test_multi_stratified_kfold_can_leak_peptides_across_folds`) proves the
> splitter-layer fixture would expose a regression rather than passing vacuously. That control
> pairs with the splitter layer only; the pipeline-layer test uses its own fixture.

**Vaccinia bloc.** Excluding `Orthopoxvirus vaccinia` from the peptide-grouped evaluation
moves AUC-PR from 0.6092 to 0.7693 (+0.1601) while AUC-ROC falls 0.8130 -> 0.7427. The pooled
headline metric, as currently computed, is substantially a measure of "can the model recognise
vaccinia peptides," not of virus-immunogenicity discrimination across the nine target pathogens.
This heading read "Vaccinia decoy bloc" until 2026-08-10. Those 21,432 rows are genuine IEDB
assay-confirmed negatives (`database_source = IEDB`), not decoys; the honest term is
*out-of-panel*. See `docs/claims_register.md` D19.

**The "honest" correctives are also leakage-inflated, just less severely.** This project has
cited two figures as the leakage-free antidote to the retracted pooled 0.9368/0.7678
(`docs/claims_register.md` D12): the per-virus mean AUC-ROC (`results/per_virus_eval_v5_mode31.csv`,
0.751) and the pooled honest same-pathogen AUC-ROC (`results/pooled_honest_same_pathogen.csv`,
Def A, 0.712). **Both figures are RETRACTED by D15** - those two files now hold **0.658** and
**0.6015** respectively, so do not read 0.751/0.712 as their current contents. Both are computed from `models/v5/rf_oof_predictions_mode31.csv`, which is written
by the same `MultiStratifiedKFold` path as the pooled headline - not an independent splitter.
Reproducing both under a matched peptide-grouped splitter (same RF config as above):

| Metric | Production splitter | Peptide-grouped | Inflation |
|---|---|---|---|
| Per-virus mean AUC-ROC (9 target viruses) | 0.7512 +/- 0.1168 (then-certified 0.751, RETRACTED D15) | 0.6587 +/- 0.0908 | +0.0925 (+14.0%) |
| Pooled honest same-pathogen AUC-ROC (Def A) | 0.7124 (then-certified 0.712, RETRACTED D15) | 0.5989 | +0.1135 (+19.0%) |

The production-splitter reproductions match the certified figures almost exactly, validating the
methodology. Both correctives carry real, non-trivial leakage inflation - smaller than the pooled
headline's 37%, but not zero. Only the LOO cross-virus table
(`results/loo_cross_virus_v5_clean.csv`) is confirmed genuinely independent of this finding: it
trains a fresh model per held-out virus with explicit virus-level partitioning
(`scripts/run_loo_cross_virus_v5.py`), never uses `MultiStratifiedKFold`, and excludes gold-standard
peptides from training.

**CORRECTED 2026-08-11 (D22): the code-trace argument below is real code behavior today, but does
not describe how the certified 0.828 was actually generated.** `results/external_validation_input.csv`
and `results/table3_tier_a_metrics.csv` were each committed exactly once (`f360b90` 2026-05-23,
`f5153152` 2026-06-21) and never regenerated, so the `rf_oof_score` merged into Tier A was never
produced by re-running today's `train_classifier.py` pipeline; it traces instead to the 720-row,
zero-duplicate-peptide `immunogenicity_dataset.csv` at `69e0e5c` (D16), on which the exact-duplicate
leakage mechanism this chain would otherwise imply is a structural no-op. A different, unquantified
risk applies instead: substring homology across 32.1% of the 704-peptide pool
(`docs/claims_register.md` D22). The table below is unaffected by this correction - it was always a
measurement of a fresh v5-trained model re-scoring the 414 resolvable Tier A peptides under two
splitters (a hypothetical "if today's model scored these peptides" question), not a measurement of
the certified pathway's own exposure. Measured on
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
| N/C-terminal flanking + ERAAP trimming (**proxy**, replacing the mode-33 mock) | Estimated, not yet measured (replacement, not repair) | Requires source-protein flank sequence at inference time; one extra PSSM evaluation per peptide, cacheable | Zero new dependencies - but **CORRECTED 2026-08-11: this cell previously read "`src/antigen_processing.py` already implements the real PSSMs (Keller 2020, Doytchinova 2004); the change is wiring, not a new library". That is false**, and the retraction block in Section 4.2 below already said so without this row being updated. By its own docstring the module emits *proxy* scores, "not tool-call wrappers to NetChop or NetCTL", and says those matrix weights "should be replaced by" the published log-odds matrices - i.e. they are not them. It also emits `erap_score` (ERAP N-terminal trimming), a different biological quantity from `netchop_score` (proteasomal C-terminal cleavage). | **Adopt as replacement** (see Section 4.2 retraction) |
| Agretope/epitope (P2/P9 vs P4-P8) disconnect ratio | Estimated; genuinely novel signal, not measured on this dataset | ~3x MHCflurry calls per peptide (alanine-scan mutants), cacheable in the existing binding-matrix pattern | Zero new dependencies | **Prototype** |
| pMHC stability (t1/2, NetMHCstabpan) | Estimated from literature; orthogonal to affinity | External predictor call per peptide-allele pair; cacheable | DTU academic-license binary, non-redistributable; repo's one existing DTU integration (NetChop) is currently a silent mock (Section 4) - repeat risk is high without a hard-fail contract | **Gate** on a real (non-mock) integration |
| ESM-2 embeddings + PCA for RF/XGB | Estimated; existing `30_esm` mode suggests limited RF headroom at this dataset size | Model forward pass per peptide (cacheable), PCA fit must be inside the fold | `transformers` not installed locally; `src/features.py` ESM loader currently falls back to a SHA256-seeded **random** vector on load failure - a silent-garbage pattern that must be fixed before this is trustworthy | **Research only** |
| Self-proteome tolerance (RSAT / foreignness) | **-0.0037 (measured, existing mode 35)** | None - already implemented, binary exact-match only | Zero new dependencies | **Reject on this dataset** (Section 4 explains the structural confound) |
| AlphaFold3/Boltz-1 pLDDT + ESM-3/ESM-C in GINEConv | Not estimated - GNN promotion Gate 1 was re-anchored 2026-08-10 from AUC-PR >= 0.85 (unreachable against the honest baseline) to >= 0.65 under a peptide-grouped splitter; the gate is now reachable in principle but this work remains unscoped | Structure prediction per peptide-MHC complex; GPU-bound | AlphaFold3 weights are gated/non-commercial - direct OpenSSF/redistribution conflict; Boltz-1 is licence-viable; `src/verify/structural_gnn.py` currently fabricates idealised coordinates rather than using real structures | **Defer** |

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

### 4.2 Adopt as replacement: N/C-terminal flanking and ERAAP trimming

This is not really a new-feature proposal once the codebase is read closely. `feature_mode=33`
already ships `netchop_score`/`tap_score`, but `scripts/precompute_antigen_processing.py`
computes them via `query_netchop(..., mock_fallback=True)` in `src/external_predictors.py`,
whose own script comment states the values "are NOT real NetChop 3.1 / TAPreg predictions" and
whose generator uses `hash(char + pep)` - unstable across processes unless `PYTHONHASHSEED` is
pinned. Meanwhile `src/antigen_processing.py` contains an ERAP/TAP PSSM **proxy** implementation
(**corrected 2026-08-11: this read "a real ERAP/TAP PSSM implementation"**; the module's own
docstring calls its outputs *proxy* scores and says those weights "should be replaced by" the published log-odds
matrices, so "real" is false here exactly as it was at the Section 3 trade-off table and at Phase 1
step 8) that `src/train_classifier.py` never
imports - dead code sitting next to a mock that WAS documented as real in `docs/model_cards/
rf_33feature_integrated.md` (corrected 2026-08-11: that card now documents the values as MOCK
in eleven places, from D18's 2026-08-10 pass; this sentence's present tense was left standing
when the three neighbouring 'real PSSM' claims were corrected). The correct 2026 upgrade path is to retire the mock, wire the proxy
PSSM code into the mode-33 build, add genuine N/C-terminal flanking probability using source-
protein context already available via `docs/antigen_accessions.md`, and correct the model card.
This converts a scientific-integrity liability into whatever real signal the PSSM approach
actually carries - which should be measured honestly (peptide-grouped) rather than assumed.

> **Partly actioned 2026-08-10 (`docs/claims_register.md` D18).** The documentation half is done:
> the mock is now disclosed at every tracked **documentation** surface that presented it as real,
> and the model card has been corrected. ~~**Code surfaces still present the features as real**~~
> **UPDATE (2026-08-10, later the same day): the code-disclosure surfaces are now CLOSED too.**
> `src/features.py` (the `FEATURE_COLUMNS_33` comment, which no longer calls them "orthogonal",
> and `load_antigen_processing_cache`'s docstring) and `src/train_classifier.py` (the
> `--feature-mode` `--help` text, which no longer reads "31+NetChop+TAPreg", plus the
> `--antigen-processing-cache` help, the printed `mode_label`, and `prepare_features_33`'s
> docstring) all now state the scores are mock and non-reproducible. The step-8 wording above
> ("documented as real ... correct the model card") is therefore satisfied, and is retained as the
> historical statement of what the gap was. **Disclosure is not repair:** the code half of the
> *remedy* is still NOT done - the mock still feeds `data/antigen_processing_cache.csv` and remains
> Phase 1 step 8. Two corrections to the paragraph
> above, both established while writing D18. (1) Wiring in `src/antigen_processing.py` is **not a
> like-for-like repair**: by its own docstring it emits *proxy* scores, "not tool-call wrappers to
> NetChop or NetCTL", and it produces `erap_score` (ERAP N-terminal trimming), a different
> biological quantity from `netchop_score` (proteasomal C-terminal cleavage). (2) The mock is not
> merely unstable across processes - because `hash()` is per-process salted and the original
> `PYTHONHASHSEED` was never recorded, **the shipped cache cannot be reproduced**, which makes
> replacement, not repair, the only route.

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
grouped_auc_pr`), consistent in direction with the certified peptide-grouped ablation (`models/v5/
training_results_ablation.csv`, regenerated under `PeptideGroupedKFold` on 2026-08-10: AUC-PR
0.6085 at mode 33 -> 0.6069 at mode 35, i.e. **-0.0016**; this parenthetical read "ungrouped ...
-0.0002" until 2026-08-10 and was wrong on both counts). This section previously added a "structural
reason" on top of that null result: that `scripts/generate_hard_decoys.py` builds the majority of
hard negatives *by sampling the human self-proteome*, so a "is this peptide human" feature would
predict "negative" by construction of the decoy set rather than by learned immunological
tolerance. **That reasoning is WITHDRAWN (2026-08-10).** It cannot apply to the measurement it was
attached to: `scripts/audit_cv_leakage.py` scores on `_load_active()`, which drops every
quarantined row, and all 5,000 `self_proteome_decoy` rows are quarantined - so **zero** of them are
present in the frame that produced 0.6055. The confound would bite only on a corpus that admitted
them, which this one does not. The null result stands on its own evidence; the explanation offered
for it did not. (This is the same class of error as D19: asserting a property of a scored
background without checking what that background actually contains.)
The self-similarity implementation is also binary (exact 9-mer match
only, not graded identity - the code's own docstring says partial-identity alignment "would be
too slow") which caps its ceiling. That binary ceiling, not a decoy confound, is the live
limitation - revisit if the feature is reimplemented with graded identity, or on a corpus that
actually admits self-proteome negatives into the scored pool.

### 4.7 Defer: AlphaFold3/Boltz-1 pLDDT + ESM-3/ESM-C GINEConv features

Two independent blockers, either one sufficient to defer. First, AlphaFold3 model weights carry
a non-commercial, gated license - a direct conflict with this repository's OpenSSF supply-chain
posture (hash-pinned, redistributable dependencies only); Boltz-1 is license-viable but pLDDT
confidence on an isolated 9-mer peptide (rather than the full pMHC-TCR complex) carries limited
structural information. Second, and more decisively: `src/verify/structural_gnn.py` currently
fabricates idealised backbone coordinates (`generate_canonical_groove_coords`) rather than using
real structures, and the GNN promotion Gate 1 threshold was re-anchored 2026-08-10 from AUC-PR >= 0.85 to
>= 0.65 under a peptide-grouped splitter (`src/verify/promote_gnn.py`), precisely because 0.85
was unreachable relative to even the leaky RF ceiling (0.831) and dramatically unreachable
relative to the honest peptide-grouped baseline (0.6058 certified). Spending compute on richer
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

> **EXECUTED 2026-08-10 (`30f1b76`, merged via PR #233). All six steps below have landed**, step 6
> with one declared carve-out (Tier A, see its row), verified
> against the tree rather than the commit message. The imperative wording is preserved as the
> historical statement of the work; two of its factual assertions are now false (step 2's "fires on
> every v5 split", step 3's "`tests/` has no assertion today"). How the landed form differs from
> what was proposed:
>
> | Step | As landed |
> |---|---|
> | 1 | `PeptideGroupedKFold` (`src/ml_utils.py`), threaded through `_cross_validate`. **The `src/train_classifier.py` CLI defaults to grouped**; the *Python API* keeps `cv_group_by=None` so the four indirect callers are untouched. `sestrav validate` via `src/cli.py` therefore still yields an ungrouped number and has no knob of its own. (There is no `sestrav train` subcommand; the CLI exposes `predict`, `validate`, `benchmark`, `info`.) |
> | 2 | Both offered remedies, in modified form: an explicit coarsening ladder that records the rung as `.stratification_components_` **and** raises rather than degrading further. Root cause was `_bin_origin` not recognising `iedb_api`. Note the second remedy was not taken literally as proposed: label-only remains a *reachable* rung, and the raise fires only when label-only is itself too sparse. What changed is that degradation is no longer silent. |
> | 3 | Asserted at two layers (splitter and emitted OOF frame) plus a paired negative control proving the fixture would expose a regression. |
> | 4 | First option taken: a vaccinia-excluded re-slice reported alongside the pooled metric. The vaccinia bloc was **not** repartitioned. The re-slice is not the corpus-refit counterpart in `results/cv_leakage_audit.csv` and must not be conflated with it. |
> | 5 | Mechanism spans `src/features.py` (an `impute=` flag) and `src/train_classifier.py` (the in-fold median arithmetic), gated to modes 33/35. The final full-pool refit still uses whole-cache medians, correctly: the full pool is that model's own training set. |
> | 6 | Done across `results/`, `models/v5/`, the model cards, `docs/claims_register.md` D15 and `CHANGELOG.md`. **Tier A deliberately not re-run** (only 414 of 704 peptides resolve to an active v5 row); 0.828 stands as a labeled historical figure per D16. |

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
   bloc to a declared out-of-panel partition, so the headline number is not 60% one non-
   target pathogen. (This read "vaccinia decoy bloc ... decoy-only partition" until
   2026-08-10; those rows are assay-confirmed negatives, not decoys - D19.)
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
8. **CORRECTED 2026-08-11 - this step rested on a false premise as previously written.** It read "Wire
   `src/antigen_processing.py`'s **real PSSMs** into the `feature_mode=33` build". There are no real
   PSSMs in that module: it emits *proxy* scores by its own docstring, which says those weights
   "should be replaced by" the published log-odds matrices - i.e. they are not them - and it
   produces `erap_score` (N-terminal trimming), not
   `netchop_score` (C-terminal cleavage). Section 4.2's retraction block established this; the step
   text was never updated to match.
   **What the step actually is:** replace the `external_predictors` mock path with the deterministic
   literature-informed ERAP/TAP proxy, regenerate `data/antigen_processing_cache.csv` with a
   provenance sidecar recording script + input sha256 (mirror `scripts/audit_cv_leakage.py`'s
   `_write_provenance`, which captures both, rather than
   `models/peptide_binding_matrix_v5.provenance.json`, which records no input checksum), and correct
   `docs/model_cards/rf_33feature_integrated.md`. This buys reproducibility and honest naming; it
   does NOT buy real antigen-processing predictions.
   **Two decisions to settle before writing code:** (a) whether the columns keep the misnomer
   `netchop_score` or are renamed to `erap_score` (a rename touches `FEATURE_COLUMNS_33`,
   `ANTIGEN_PROCESSING_COLUMNS`, `src/train_classifier.py`, `src/external_predictors.py`,
   `scripts/precompute_antigen_processing.py`, the cache header itself, six tracked test files
   carrying the literal column name, and every doc surface. `config.yaml` names the cache path
   and the feature mode but not the columns, so it is likely untouched. Enumerate before
   starting: this list is indicative, not certified complete); (b) that this forces a retrain of v5 ablation modes 33 and 35 and a number resync
   across roughly nine tracked surfaces. The canonical mode-31 figure (0.6058) is NOT affected.

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

> **Phase 0 arm of this checklist is discharged (2026-08-10).** The fold-disjointness test exists
> (named in the closure note in Section 2) and `scripts/audit_cv_leakage.py` carries a
> `production_grouped_splitter` arm bound to `PeptideGroupedKFold`. The remaining bullets stay
> live: they are the standing gate for the Phases 1-3 feature work, which has not started.

---

## 7. Standing decision for the repository owner

> **RESOLVED AND EXECUTED 2026-08-10.** This section is retained as the analysis that framed the
> decision; it is no longer an open question. The re-baseline was carried out (`30f1b76`) and
> published (PR #233), so the closing sentence below - "it does not itself alter any published
> claim ... a decision for the repository owner" - is now historically inverted and must not be
> read as live guidance. Two further notes for accuracy: the figures projected here were
> estimates, and the measured outcomes differ slightly (per-virus mean landed at **0.658**, not
> 0.659; pooled honest same-pathogen at **0.6015**, not 0.599). The Tier A constraint stated in
> the second half of this section **survives unchanged** and remains the live reason 0.828 was not
> re-run: the v5 mode-31 model has never been evaluated on the full 704-peptide field and cannot
> be. What remains genuinely open is not the re-baseline but its downstream sequencing, which is
> tracked in `CHANGELOG.md` `[Unreleased]` and `docs/claims_register.md` rather than here.

Phase 0 step 6 restates the project's certified headline result downward by roughly 0.23 AUC-PR
(or 0.07 if reported net of the vaccinia bloc instead), and also restates the two figures this
project has been citing as the leakage-honest corrective - both RETRACTED by D15 - the
per-virus mean (0.751 -> 0.658 as finally measured,
-0.09) and the pooled honest same-pathogen figure (0.712 -> 0.6015 as finally measured, -0.11)
- both downward by a
smaller but still material amount. The Tier A headline (0.828) does not carry the same
exact-duplicate exposure: its 720-row corpus has zero duplicate peptides, and the -0.0176 AUC-PR
delta previously cited here measures a fresh v5-trained model on the resolvable subset, not the
certified pathway (`docs/claims_register.md` D22). It carries a separate, arguably harder problem
instead: it is
a 2026-05, 30-feature, unweighted, 200-tree measurement mislabeled as the canonical v5 `mode_31`
result (D16), plus a disclosed, unquantified substring-homology risk (32.1% of the pool, D22). **Corrected 2026-08-09, and worth recording as a process note:** this paragraph
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
