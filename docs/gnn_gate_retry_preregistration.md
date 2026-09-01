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
`src/verify/promote_gnn.py` rather than restated from memory. This is the archive run, one
of eight against the same frame; **read section 1.1 with this table, never on its own:**

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

### 1.1 The figures above are the best of eight runs, not a single run

Stated here because sections 2.3 and 4 both demand it, and a pre-registration written
against selection over repeated attempts cannot itself quote one figure out of a series
without saying so.

Eight independent evaluations have been run against this same scoring frame: the 2026-08-13
archive run quoted above, a B5 baseline and its repeat, and five fresh seeds. The archive
run is **the most favourable of the eight on both reported quantities**.

| Quantity | Archive run (quoted above) | Series, n = 8 |
|---|---|---|
| Gate 1 pooled AUC-PR | 0.6458, the **maximum** | 0.6298 to 0.6458, **PASS 0 / 8** |
| Gate 1 deficit to 0.65 | 0.0042, the **smallest** | mean 0.0122 |
| Gate 2 std (ddof=0) | 0.0234, the **worst** | 0.0157 to 0.0234, **PASS 4 / 8** |
| GNN minus RF delta | +0.0402, the **maximum** | +0.0243 to +0.0402 |

Note which way this cuts. Disclosing the series makes the **null stronger, not weaker**: no
run in the series reaches the bar, so Gate 1's FAIL is better supported at n = 8 than the
single archive run implies. The effect underneath it also survives at the worst observation
(seed 7: delta +0.0243, 95% CI [0.0122, 0.0365], still excluding zero). The run series is
recorded in `_local/state/session_plan_2026-08-29.md`.

**Gate 2's convention is pinned here, because the tally depends on it.** Gate 2 is
`np.std(fold_auc_prs)`, i.e. **ddof=0**, which is what `promote_gnn.py` ships. The same
eight runs pass 3 / 8 at ddof=1. Any retry reports the convention next to the number.

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
depend on any single sidecar. The same argument is recorded in
`data/iedb_negatives_v5_provenance.json`'s `downstream_status` block, which is now on `main`
and corroborates this independently:

- The regeneration adds 36 rows, whose `virus` is `Unknown` (29) or `Self` (7).
- Neither string is a key in `build_dataset_v5.py`'s `VIRUS_FAMILY_MAP` (26 keys) nor in
  `_dataset_utils.py`'s `_VIRUS_NAME_MAP` (21 keys, and nothing maps to them), so
  `virus_family` stays null and `apply_quarantine`'s null-family branch quarantines both
  values. The mask is `df["virus"].isin(quarantined_viruses)`, which catches every row
  carrying those values regardless of `source_type`.
- The competing explanation, the depth thresholds, is excluded by measurement: in the
  shipped corpus both clear them by wide margins (Unknown 4577 rows / 4576 real negatives,
  Self 3811 / 3811, against `MIN_ROWS_PER_VIRUS` 50 and `MIN_REAL_NEGATIVES_PER_VIRUS` 10).
- Gate 1 and the paired bootstrap are both scored on the **post-holdout scoring pool**:
  `mode31_pooled_n_rows = 35,555` in `results/pooled_cv_metrics_mode31.csv`, and the
  bootstrap matched 35,555 rows 1:1.

> **Two different pools, and this clause turns on which one is measured.** The **active**
> pool of the shipped corpus is **35,597** rows (51,185 total less 15,588 quarantined). The
> **scoring** pool is **35,555**, the active pool less the 42-row gold-standard holdout, and
> it is the frame Gate 1 and the bootstrap are actually computed on. The separation is
> already drawn in `README.md` and `ARCHITECTURE.md`. It is repeated here because conflating
> the two is the specific way this clause would **fail open**: a reader who measures the
> active pool of an unchanged corpus gets 35,597, reads that as "differs from 35,555", and
> authorises precisely the re-run this section exists to prevent.

If the added rows all land quarantined, the scoring pool and therefore the CV folds and the
held-out sets are unchanged, and a re-run is a re-run **against the same held-out set** -
precisely what the module prohibits.

**This remains a prediction about a rebuild that has not been run.** The commitment sought
is procedural, not substantive: **before any retry justified by "the corpus changed",
measure the rebuilt corpus's 35,555-row scoring pool - not its active pool - and show that
it differs.** If that frame is still 35,555 rows and the row identities are unchanged, the
corpus is not fresh and the retry is not authorised by this clause. Report both counts, so
that the comparison being made is visible rather than inferred.

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

**The hazard is sharper than the section-1 table alone suggests, because Gate 2 is not a
standing FAIL.** Per section 1.1, the 0.0234 quoted there is the **worst** of the eight
runs; across the series Gate 2 passes **4 of 8** at ddof=0 (3 of 8 at ddof=1), the 0.02 bar
sits inside the observed 0.0157-0.0234 band, and the nearest miss is 0.020287. So a single
post-change run that lands under 0.02 is **not** evidence that anything was fixed: it is
inside the run-to-run spread of changing nothing at all. Contrast Gate 1, which fails 8 of
8. Note also that seed averaging, offered as an example change below, has in effect already
been run against this frame at n = 8.

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
5. Report Gate 2 as a **distribution over pre-declared seeds**, with the ddof convention
   stated, not as a single number. Given the 4 / 8 baseline tally, a Gate 2 claim resting on
   one run cannot be distinguished from a reseed.

## 4. What is reported, regardless of outcome

The null result stands and stays published until something legitimately supersedes it. Any
retry conducted under this pre-registration is reported with:

- the declared change, quoted from before the run;
- the full five-gate scorecard;
- the effect size and interval against the RF baseline, whichever direction it points;
- the number of evaluations run against that set, including abandoned ones, **and the
  observed range across them rather than only the best** - which is the standard section 1.1
  applies retrospectively to the existing series.

A retry that produces a pass and a retry that produces another null are written up the same
way. That symmetry is the entire point, and it is why this document has to exist before the
run rather than after.

## 5. What this document does not do

It does not authorise a run, promote anything, change any threshold, or re-open Gate 1. It
does not amend `src/verify/promote_gnn.py`, whose standing prohibition remains the operative
rule. If this file and that module ever disagree, **the module wins** and this file is the
one to fix.
