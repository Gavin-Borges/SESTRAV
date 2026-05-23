# H2 — ISSR@10 Enrichment vs Binding-Only: Testable Evaluation Protocol

**Purpose:** The SESTRAV proposal states a **Pipeline Enrichment** hypothesis (H2): the integrated pipeline should achieve **≥2×** enrichment of true immunogenic epitopes in the **top 10%** of ranked candidates (ISSR@10) compared to **binding-only** ranking. Early status reviews noted this was **hard to test** when “ISSR” (from `src/evaluate_metrics.py`) was conflated with **gold-standard recovery** on unlabeled proteome outputs. This document separates **tiers of evidence** and gives a reproducible procedure for each.

**Related code**

- `src/evaluate_metrics.py` — `issr_at_k()`, `evaluate()` (definition of ISSR@10 / ISSR@25 on **binary-labeled** data)
- `src/baseline_comparison.py` — gold-standard **recovery** in top 10% / 25% of **pipeline** candidate pools (not the same numerically as training-set ISSR)
- CMB 523 Project 2 ablation — “binding-only” as a **10-D ML model** vs “combined-30”; complementary to **raw binding rank** below

---

## 1. What H2 actually needs

H2 requires two **comparable rankings** of the **same set of peptides** and a **ground-truth label** for each peptide (immunogenic vs not) so that ISSR@k is well-defined:

- **Integrated score:** e.g. `immunogenicity_score` from the trained RF / XGB / ANN (21- or 30-feature), out-of-fold or on a held-out split.
- **Binding-only score:** a **single scalar per peptide** derived from MHCflurry (or NetMHCpan) without TCR physicochemical features—so the model is not “cheating” with the same feature space as the full model.

The **ratio** of interest is:

\[
R_{10} = \frac{\mathrm{ISSR@10}_{\text{integrated}}}{\mathrm{ISSR@10}_{\text{binding-only}}}
\]

**H2 (proposal):** \(R_{10} \geq 2\) (and typically also report \(R_{25}\) for context).

If the denominator is very small, the ratio is unstable; see **Section 4**.

---

## 2. Tier A (primary for H2) — Labeled IEDB benchmark, same metric as training

**Dataset:** `immunogenicity_dataset.csv` (N=720, v2 dataset), same binary labels used in `train_classifier.py`.

**Steps (recommended: 5-fold stratified CV, match `StratifiedKFold` in training):**

1. **Binding-only scalar** \(b_i\) for each peptide \(i\):  
   - **Default:** `max` of MHCflurry `presentation_score` across the **same 10 alleles** as production `config.yaml`.  
   - **Alternatives (sensitivity analysis):** `mean`, or `mean` of top-3 alleles—document which definition is used; stick to one for the headline H2 result.

2. **Integrated score** \(s_i\): out-of-fold predicted probability from the classifier under test (RF preferred for stability on small tabular data).

3. On **each fold’s validation set** (or a single held-out 20% test set if CV is too heavy), compute:
   - `evaluate(y_true, y_scores_binding)` → `issr_10_binding`, `issr_25_binding`
   - `evaluate(y_true, y_scores_integrated)` → `issr_10_integrated`, `issr_25_integrated`

   Use `src.evaluate_metrics.evaluate` so PredIG-aligned definitions match all other SESTRAV reports.

4. **Aggregate across folds:** report **mean ± std** of ISSR@10 for both methods; compute **mean ratio** \(\bar{R}_{10}\). Optionally bootstrap **95% CI** on the ratio (resample peptides within folds).

5. **Decision:** H2 supported if the **lower bound** of the CI for \(\bar{R}_{10}\) is ≥ 2, or (simpler rule) mean ratio ≥ 2 and binding ISSR is not degenerate (e.g. binding ISSR@10 > 0.05).

**Why this fixes the old “inconclusive” note:** ISSR is only defined where **labels exist**. Proteome-wide ranked lists without dense labels cannot yield ISSR@10 in the `evaluate_metrics` sense.

**Relation to CMB 523 Project 2:** The ablation table compares **binding-only ML** (10 features) vs **combined** (30 features)—that tests **feature groups**, not necessarily **raw binding rank**. For strict alignment with the proposal’s “binding affinity baseline,” also report **Tier A with \(b_i\)=max presentation** alongside the ablation numbers.

---

## 3. Tier B — Pipeline gold-standard enrichment (secondary; not ISSR on labels)

**What `baseline_comparison.py` already does:** For each virus, rank **all candidate peptides** in `{prefix}_features.csv` by RF / XGB / ANN vs **presentation_score**, then measure how many **gold-standard** peptides fall in the top 10% / 25% of ranks.

**Use:** Strong **face validity** for vaccine design (“do known epitopes float to the top?”). **Not** a substitute for Tier A when stating H2 against the proposal’s ISSR definition, because:

- Gold-standard sets are **small** (n=15) and **biased** toward strong binders.
- The candidate pool is **unlabeled** except for those few epitopes—so ISSR@10 **per `evaluate_metrics`** is undefined over the full pool.

Report Tier B as **“gold-standard recovery vs binding-only rank”** and avoid calling it “ISSR@10” unless you explicitly redefine the metric for that subset only.

---

## 4. Practical pitfalls

| Issue | Mitigation |
|-------|------------|
| **Ratio blows up** when binding ISSR≈0 | Report absolute ISSR values; require minimum denominator (e.g. ISSR@10(binding) ≥ 0.08) before interpreting ratio |
| **Integrated model uses binding columns** (30-feature) | Fair: binding-only uses the **same** MHCflurry run, only the **score** differs; disclose that integrated = physico + binding |
| **Data leakage** | Fit scaler / model **only on training fold**; binding scores may be precomputed for all peptides but **do not** tune thresholds on validation for the headline CV metric |
| **Mismatch with external tools** | PredIG/PRIME benchmarks are **Tier C** (future); not required for internal H2 consistency |

---

## 5. Minimal acceptance checklist

- [ ] Tier A run completed with documented **binding aggregate** (default: max presentation across 10 alleles).
- [ ] Same **720 labels** (v2 `immunogenicity_dataset.csv`) and **same folds** as production training — or **928** only if explicitly re-running v1 legacy (document which).
- [ ] Table: ISSR@10 / ISSR@25 for **binding rank** vs **integrated model** (mean ± std across folds).
- [x] `src/h2_tier_a_evaluation.py` now exports bootstrap 95% CI for \(R_{10}\) and fold-level paired sign-flip p-value.
- [ ] Optional: Tier B gold-standard table from `baseline_comparison.py` cited as complementary evidence only.

---

## 6. References in-repo

- Proposal: `01_Proposal/SESTRAV Proposal.txt` — H2 wording  
- Status discussion: `00_Active/sestrav_status_review.md.resolved` — Part 2 (minor items)  
- Project 2 ablation (binding ML vs combined): `CMB 523 Injection for SESTRAV Progress/523 Project 2/Project2_Report.md` — Section 4.3
