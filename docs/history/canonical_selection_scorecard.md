# Canonical Model Selection Scorecard

Use this scorecard to lock the single public canonical release and retain the other track as progress/future reporting.

## Candidates

- Candidate A: 30-feature integrated track (`models/rf_30feature_integrated.joblib`)
- Candidate B: 21-feature legacy sequence-only track (`models/rf_21feature_legacy.joblib`)
- Decision date: 2026-04-21 (updated 2026-04-25 with v2 dataset metrics)
- Decision owner(s): SESTRAV finalization team

## Rubric (0-5 per category)

### 1) Performance Evidence

- AUC-ROC quality and stability:
  - A: 0.7536 +/- 0.0365 (H2 Tier A summary, v2 dataset, 30-feature integrated model).
  - B: 0.726 +/- 0.042 (legacy 5-fold benchmark summary, v1 dataset).
- AUC-PR quality and stability:
  - A: 0.8497 +/- 0.0287 (H2 Tier A summary, v2 dataset). Above-trivial AUC-PR: +0.147 (+41% over prevalence baseline).
  - B: 0.919 +/- 0.021 (legacy benchmark summary, v1 dataset). Above-trivial AUC-PR: +0.105.
- ISSR@10 and ISSR@25 operational value:
  - A: ISSR@10 0.8571, ISSR@25 0.8514 (v2 dataset).
  - B: ISSR@10 0.911, ISSR@25 0.947 (v1 dataset).
- External/holdout behavior quality:
  - A: Full stage recovery validated (15/15 Stage 1 and Stage 2 recovery); gold-standard negative discrimination 9/10; H2 ratio criterion not supported (R10 0.9836, R25 1.0276).
  - B: Strongly documented legacy CV narrative with known limitations and historical continuity.
- Score (0-5): A=4.5, B=3.8

### 2) Biological Validity and Defensibility

- Alignment to known immunology constraints:
  - A: Explicit multi-allele panel support and integrated binding+physicochemical framing.
  - B: Historically constrained to sequence-only legacy mode and older single-allele framing.
- Honest limitation handling:
  - Both tracks clearly document gold-standard bias, rank tie density, score saturation, and binding decorrelation limitations.
- Interpretability and feature plausibility:
  - Both preserve feature-level interpretability; A retains stronger translational relevance by pairing immunogenicity signal with binding context.
- Sensitivity to known biases:
  - A and B both inherit dataset bias risks; A has better operational framing for candidate filtering under multi-allele use.
- Score (0-5): A=4.3, B=3.9

### 3) Finalization and Reproducibility

- End-to-end run stability:
  - A: Verified in isolated clean environment with tests + Snakemake dry-run + final validation.
  - B: Stable legacy artifacts available but not chosen as primary path.
- Artifact completeness:
  - A: Canonical final validation bundle regenerated (`gold_standard_validation.csv`, `baseline_comparison.csv`, `h2_tier_a_summary.csv`, `final_validation_report.md`).
  - B: Legacy benchmark docs remain available as comparator context.
- Version/environment consistency:
  - A: Repro path now pinned in docs (Python 3.11 + NumPy 1.26.4 + TensorFlow-compatible stack).
- Documentation clarity:
  - A: Canonical vs progress framing is now explicit across release docs.
- Score (0-5): A=4.6, B=3.6

### 4) Proposal Alignment

- Match to originally stated scope:
  - A: Better aligns with integrated biological signal framing and multi-factor immunogenicity prioritization.
  - B: Useful historical baseline but underrepresents current finalized architecture.
- Completion of required deliverables:
  - A: Meets finalization checklist gates with current pipeline defaults.
  - B: Completed as legacy baseline evidence line.
- Consistency with stated hypotheses:
  - A: Supports updated canonical narrative with transparent H2 decision status.
  - B: Supports legacy hypothesis narrative with lower current representativeness.
- Readiness for public presentation claims:
  - A: Stronger representation of current platform capabilities and release defaults.
- Score (0-5): A=4.4, B=3.7

## Totals

- Candidate A total (max 20): 17.8
- Candidate B total (max 20): 15.0

## Tie-Break Rule

If totals are within 1 point, select the candidate with:

1. Better biological defensibility under live Q&A.
2. Fewer unresolved reproducibility risks.
3. Cleaner, auditable release evidence.

## Final Decision Statement

- Canonical public release: Candidate A (30-feature integrated track)
- Progress/future track: Candidate B (21-feature legacy comparator)
- Why canonical won (3-5 bullets):
  - Higher discriminative and enrichment metrics in current validated outputs.
  - Better alignment with the current multi-allele integrated architecture used in production config.
  - Stronger reproducibility readiness after environment and runtime hardening.
  - More representative of the proposal's biologically integrated final direction.
- What remains for progress track (next milestones):
  - Maintain legacy benchmark reproducibility for historical comparison.
  - Use as ablation/comparator in future reports, not as release default.
  - Keep legacy docs synchronized and clearly scoped to avoid narrative ambiguity.
