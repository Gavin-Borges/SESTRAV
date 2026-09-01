# Pre-registration: any retry of the GNN promotion gates

**STATUS: DRAFT. NOT IN FORCE until the maintainer ratifies it by merging this file.**
A pre-registration has force only because the owner committed to it before seeing the
result. Until that happens this document records a proposal, not a rule, and nothing in it
licenses a run.

**Scope.** This governs any future evaluation of a model against the five canonical
promotion gates in `src/verify/promote_gnn.py`. It does not re-open anything by itself. It
exists so that whatever happens next is reportable either way, which is the only property
that makes a retry worth running at all.

---

## 1. The standing result, frozen

Measured 2026-08-13, recorded 2026-08-16, and quoted here from
`src/verify/promote_gnn.py` rather than restated from memory:

| Gate | Measured | Bar | Outcome |
|---|---|---|---|
| 1 - Generalization (pooled AUC-PR, peptide-grouped) | 0.6458 | >= 0.65 | **FAIL by 0.0042** |
| 2 - Stability (cross-fold AUC-PR std) | 0.0234 | <= 0.02 | **FAIL** |
| 3 - Latency | - | <= 2x RF | PASS |
| 4 - Calibration (ECE) | - | < 0.05 | PASS |
| 5 - Escape sensitivity | - | >= 0.80 | PASS |

The gates are an AND-conjunction, so this is a **null result on the pre-registered bar**.

Underneath the null sits a real effect, and the two must always be reported together:
against RF mode-31 the GNN scored AUC-PR **0.6458 vs 0.6055**, delta **+0.0402**, 95% CI
**[0.0286, 0.0520]**, excludes zero, p < 0.0001 (paired bootstrap, seed 20260813, 10,000
resamples, 35,555 rows matched 1:1). The architecture is measurably better on
discrimination and still misses the bar. **It is not promoted.**

> Arithmetic note, recorded so nobody "corrects" it later: `0.6458 - 0.6055 = 0.0403` at the
> 4-decimal values printed above, while the delta is quoted as `+0.0402`. That is a rounding
> artifact of quoting 4dp inputs for a delta computed at full precision. It is not a
> discrepancy and must not be "fixed" by changing either figure.

## 2. Three things to settle BEFORE any retry, not after

Each of these can move the bar. Settling them after seeing a result is indistinguishable
from moving the bar to fit the result.

### 2.1 Which RF baseline anchors Gate 1

`GATE1_AUC_PR_MIN = 0.65` was re-anchored on 2026-08-10 from 0.85. That re-anchoring was
legitimate: 0.85 had been set against the ungrouped RF baseline (pooled AUC-PR 0.8312),
which is retracted as peptide-leakage-inflated. But it is a precedent, and **a second
re-anchoring that happens to let 0.6458 through would not be legitimate.**

There is also an unresolved aggregation question in the anchor itself. The comment above
the constant anchors 0.65 to "the certified peptide-grouped RF baseline of **0.6058**",
which is the **fold-mean**. The gate is applied to a **pooled** GNN number, and the
published delta is computed against the **pooled** RF value **0.6055**. Those are different
quantities (see the pooled-vs-fold-mean separation being landed in `docs/data_registry.md`
and `docs/model_evaluation_summary.md`). The gap is 0.0003 and changes no current verdict.

**Commitment sought:** name the authoritative comparator for Gate 1 in writing, and record
that `GATE1_AUC_PR_MIN` does not change as a consequence. If the anchor text is corrected
from 0.6058 to 0.6055, the threshold stays 0.65.

### 2.2 Whether the corpus is actually fresh

`promote_gnn.py` states what re-opens the track: a new bar pre-registered before the run,
**evaluated on data not used to produce the result above (a fresh corpus or a genuinely
held-out cohort)**.

**A v5 corpus rebuild following the negatives regeneration is expected NOT to satisfy
that**, and this is the trap most likely to be walked into, because the rebuild does change
`data/immunogenicity_dataset_v5.csv` and so looks like a fresh corpus.

The reasoning is re-derived here from the build code, deliberately, so that it does not
depend on an unmerged branch. The same argument is recorded in
`data/iedb_negatives_v5_provenance.json`'s `downstream_status` block, which arrives with the
v5 negatives regeneration and is not on `main` at the time of writing:

- The regeneration adds 36 rows, whose `virus` is `Unknown` (29) or `Self` (7).
- Neither string is a key in `build_dataset_v5.py`'s `VIRUS_FAMILY_MAP` (27 keys) nor in
  `_dataset_utils.py`'s `_VIRUS_NAME_MAP` (21 keys, and nothing maps to them), so
  `virus_family` stays null and `apply_quarantine`'s null-family branch quarantines both
  values. The mask is `df["virus"].isin(quarantined_viruses)`, which catches every row
  carrying those values regardless of `source_type`.
- The competing explanation, the depth thresholds, is excluded by measurement: in the
  shipped corpus both clear them by wide margins (Unknown 4577 rows / 4576 real negatives,
  Self 3811 / 3811, against `MIN_ROWS_PER_VIRUS` 50 and `MIN_REAL_NEGATIVES_PER_VIRUS` 10).
- Gate 1 and the paired bootstrap are both scored on the **active** pool: RF pooled
  `n_rows = 35,555` in `results/pooled_cv_metrics_mode31.csv`, and the bootstrap matched
  35,555 rows 1:1.

If the added rows all land quarantined, the active pool and therefore the CV folds and the
held-out sets are unchanged, and a re-run is a re-run **against the same held-out set** -
precisely what the module prohibits.

**This remains a prediction about a rebuild that has not been run.** The commitment sought
is procedural, not substantive: **before any retry justified by "the corpus changed",
measure the active pool of the rebuilt corpus and show it differs.** If `n_rows` is still
35,555 and the row identities are unchanged, the corpus is not fresh and the retry is not
authorised by this clause.

### 2.3 The attempt budget

Selection over repeated attempts is the failure mode being guarded against, not any
particular model. A budget declared afterwards is not a budget.

**Commitment sought:** the number of evaluations against a given held-out set is declared
before the first one, and every evaluation run is reported, including the ones that
were not going to be written up.

## 3. Gate 2 is not a free pass

The plan calls Gate 2 (stability, cross-fold AUC-PR std 0.0234 vs <= 0.02) the legitimate
work. It is more legitimate than Gate 1, because reducing variance is a real engineering
objective rather than a bar-crossing exercise. **It carries the same selection hazard
anyway:** tuning until the std crosses 0.02 on the same five folds is the same procedure as
tuning until AUC-PR crosses 0.65 on the same folds.

**Protocol sought for any Gate 2 attempt:**

1. Declare the specific change in writing before running it (for example: seed averaging
   over N seeds, a named regularisation change, a named architectural change). "Try some
   things and see" is not a declaration.
2. Change one thing per evaluation.
3. Report the full scorecard each time, not only Gate 2. A variance fix that moves Gate 1
   is the most interesting possible outcome and the easiest to leave unmentioned.
4. Gate 1's verdict does not become passable as a side effect. If a Gate 2 change moves
   AUC-PR across 0.65 on the same held-out set, that is **not** a Gate 1 pass. It is a
   result requiring a fresh evaluation set under section 2.2.

## 4. What is reported, regardless of outcome

The null result stands and stays published until something legitimately supersedes it. Any
retry conducted under this pre-registration is reported with:

- the declared change, quoted from before the run;
- the full five-gate scorecard;
- the effect size and interval against the RF baseline, whichever direction it points;
- the number of evaluations run against that set, including abandoned ones.

A retry that produces a pass and a retry that produces another null are written up the same
way. That symmetry is the entire point, and it is why this document has to exist before the
run rather than after.

## 5. What this document does not do

It does not authorise a run, promote anything, change any threshold, or re-open Gate 1. It
does not amend `src/verify/promote_gnn.py`, whose standing prohibition remains the operative
rule. If this file and that module ever disagree, **the module wins** and this file is the
one to fix.
