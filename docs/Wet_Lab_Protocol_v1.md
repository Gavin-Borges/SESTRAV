# SESTRAV v2: Wet-Lab Validation Protocol (Pre-Registration)

**Version:** 1.1.0 (revised 2026-08-08)
**Target Phase:** Phase 7 (Post-Funding Clinical Validation)

> **STATUS: NOT FUNDED. NOT IRB-APPROVED. NOT SCHEDULED. NO PARTICIPANT HAS BEEN RECRUITED AND
> NO ASSAY HAS BEEN RUN.** This document is a forward-looking pre-registration: it states, in
> advance and on the public record, what would be measured and what result would count as
> success, so that the criteria cannot be adjusted after seeing data. Nothing here describes
> work that has been performed. Any statement elsewhere that SESTRAV "has transitioned to
> prospective wet-lab validation" is incorrect.

> **REVISION NOTE (v1.1.0).** v1.0.0 had two substantive defects, both corrected below.
> (1) It was written against the **structural GNN**, which is not the production scorer - the
> GNN is a deferred, GPU-gated research track that has never been promoted through
> `src/verify/promote_gnn.py`, and its Gate-1 threshold (AUC-PR >= 0.85) is structurally
> unreachable. The protocol as written could not have been run against the system that actually
> exists. It now targets the production RF mode-31 scorer.
> (2) Its success criterion pre-committed to a **2.0x** enrichment ratio, while SESTRAV's own
> certified computational analog of that same ratio is **0.9494 - a null** (`results/h2_tier_a_summary.md`).
> Pre-registering a bar that the project's own evidence predicts will be missed by roughly
> half is not rigor; it invites a reviewer to read the eventual null as a failed prediction
> rather than as the confirmation of a disclosed prior. The criterion is now grounded in that
> prior.
>
> **ADDENDUM (2026-08-09), flagged not fixed - a substrate mismatch in that same prior.** The
> R10=0.9494 computational prior binds to `results/h2_tier_a_summary.csv` (`h2_decision` row,
> `issr_10_ratio_integrated_over_binding` = 0.9493670886075949) and its rendered companion
> `results/h2_tier_a_summary.md:17`; it is also restated in `results/final_validation_report.md:10`.
> It does **not** appear in `results/table3_tier_a_metrics.csv`, whose ISSR@10 ratio is
> 0.8429/0.8611 = 0.9788 - an earlier draft of this addendum co-cited that file, and a first
> correction then over-narrowed the citation to the `.md` alone; both are fixed here (2026-08-09).
> Note also that `h2_tier_a_summary.csv` records `model_path = models/rf_30feature_integrated.joblib`,
> `feature_count = 30`, `n_total_labeled = 1004` - which independently corroborates the
> substrate mismatch described below, though it is a different artifact from the n=704 Tier A arm
> that `docs/claims_register.md` D16 governs. Per D16, the SESTRAV arm of
> that benchmark is a 2026-05, **30-feature, unweighted, 200-tree** measurement - not the
> production RF **mode-31** (31-feature) model this protocol scores the physical peptide panel
> with (Section 2, Section "Objective"). The disclosed prior and the model actually under test
> are therefore not the same generation. This does not by itself invalidate the null as a
> reasonable working prior - both models draw on overlapping physicochemical/binding features and
> a null (R10 near 1.0) is a weak, easily-transferable prediction - but it means the specific
> figure 0.9494 is not a like-for-like computational analog of what this protocol tests, and that
> gap should be disclosed alongside the criterion rather than left implicit. No corrected
> mode-31-substrate R10 has been computed; state this exposure if the study proceeds, rather than
> re-deriving one without a grouped-splitter re-run (D15).

## Objective
To prospectively test, in a rigorous *in vitro* setting using PBMCs (Peripheral Blood Mononuclear
Cells) from human donors, whether the **production SESTRAV scorer** - the RandomForest mode-31
model (`models/rf_31feature_integrated.joblib`, `FEATURE_COLUMNS_31` in `src/features.py`) -
enriches for genuinely immunogenic peptides relative to an MHC-binding-affinity-only baseline.

**The pre-specified expectation is a null.** The certified computational estimate of this exact
enrichment ratio is 0.9494 (no enrichment over binding-only). This study is therefore powered and
framed as a genuine test of that null, not as a confirmatory exercise expected to clear a high
bar. A wet-lab result consistent with 0.9494 would corroborate the computational finding; a result
significantly above 1.0 would be a positive surprise worth reporting as such.

## 1. Donor Selection Criteria
We will recruit $N=10$ donors matching the following parameters:
- **Demographics:** Healthy adults (18-55 years).
- **Infection History:** PCR-confirmed history of SARS-CoV-2 or EBV infection (depending on target antigen panel).
- **HLA Typing:** High-resolution HLA-I genotyping must confirm the presence of at least one of the 10 canonical alleles mapped in `mhc_pseudo_sequences.json` (e.g., `HLA-A*02:01`, `HLA-B*07:02`).

## 2. Peptide Synthesis

Selection is **rank-based, not threshold-based.** SESTRAV outputs a population-level relative
score intended for candidate triage; it is not calibrated to support an absolute
"probability > 0.85 means immunogenic" reading, and v1.0.0's fixed GNN probability cutoffs
implied a calibration guarantee the model does not carry. Ranking also matches how the tool is
actually used - shortlisting the top of a list - and keeps the comparison against the
binding-only baseline like-for-like.

- Score the candidate antigen panel with the production RF mode-31 model and, independently,
  with the MHC-binding-affinity-only baseline (MHCflurry 2.2.1 presentation score).
- Synthesize the **top $N=50$ by SESTRAV rank**.
- Synthesize $N=50$ **high-binding but SESTRAV-deprioritized** peptides: predicted binder
  (affinity $< 500$ nM) drawn from the **bottom quartile of SESTRAV rank**. These are the
  discriminating cases - the "binds but is not immunogenic" set the tool claims to separate.
- Both sets must be drawn from the same antigen panel and matched on HLA restriction and peptide
  length distribution, so that rank is the only systematic difference between arms.
- Peptides must not appear in any SESTRAV training corpus. Verify against the frozen v5 dataset
  before synthesis and record the check; note that peptide-level overlap - not merely
  peptide-plus-allele overlap - is the criterion (see `docs/claims_register.md` D15).
- Purity must be $>95\%$ via HPLC.

## 3. ELISpot Assay Protocol
1. **PBMC Isolation:** Isolate via density gradient centrifugation. Cryopreserve until use.
2. **Stimulation:** Thaw PBMCs and rest overnight. Plate at $2 \times 10^5$ cells/well in IFN-$\gamma$ coated ELISpot plates.
3. **Peptide Pulsing:** Pulse wells with 1 $\mu$g/mL of synthesized peptides. Include positive controls (CEF pool) and negative controls (DMSO vehicle).
4. **Incubation:** Incubate for 18-24 hours at 37°C, 5% CO$_2$.
5. **Development & Readout:** Develop plates per manufacturer protocols and count Spot Forming Units (SFU) using an automated ELISpot reader.

## 4. Success Criteria ($R_{10}$ Enrichment)

**Disclosed prior (pre-registered).** The certified computational analog of this ratio is
$R_{10} = 0.9494$ - a null, reported honestly in `results/h2_tier_a_summary.md` and in
`docs/claims_register.md`. The primary hypothesis that SESTRAV enriches over binding-only is
**not** supported computationally. This study tests whether that null holds *in vitro*.

- **Metric:** Top-10 Enrichment Ratio,
  $R_{10} = \frac{\text{SFU}_{\text{SESTRAV Top 10}}}{\text{SFU}_{\text{Binding-Only Top 10}}}$
- **Primary criterion (enrichment demonstrated):** $R_{10} > 1.0$ with the lower bound of a
  bias-corrected bootstrap 95% CI also above 1.0. Any enrichment that is real and
  distinguishable from no-effect is a positive result. **No fixed multiple is pre-committed** -
  v1.0.0's $\ge 2.0$ bar exceeded the project's own computational estimate by more than 2x and
  would have manufactured a failure out of an expected null.
- **Secondary, and the outcome the disclosed prior predicts:** a CI that spans 1.0 corroborates
  the computational null. This is a **publishable, non-negative result** - it would be direct
  *in vitro* evidence about where the binding-immunogenicity specificity gap does and does not
  close, and it is pre-registered as such precisely so it cannot later be reframed as a failure.
- **Interpretation guard:** $R_{10} < 1.0$ with a CI excluding 1.0 would indicate the SESTRAV
  ranking performs *worse* than binding-only on this panel. That outcome must be reported with
  the same prominence as a positive one.
- **Significance testing:** Mann-Whitney U (two-sided, $p < 0.05$) comparing SFU between the
  SESTRAV-top and SESTRAV-deprioritized arms defined in Section 2. Report the effect size and CI
  alongside the p-value, never the p-value alone.
- **Power:** the study must be sized to detect the effect it pre-registers. With an expected
  ratio near 1.0, $N=10$ donors x 50 peptides per arm is a **pilot** - state the achieved power
  in the results and do not report a null from an underpowered pilot as evidence of absence.
