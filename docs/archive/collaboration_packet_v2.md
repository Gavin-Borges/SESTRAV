# SESTRAV v2 Collaboration Packet

## 1) One-Page Technical Abstract

SESTRAV (Structural Epitope Scoring via TCR Recognition And Vaccinology) is an open-source computational pipeline for ranking candidate therapeutic vaccine epitopes in HPV and EBV proteomes. The pipeline combines peptide generation, multi-allele MHC presentation prediction, TCR-facing feature extraction, and immunogenicity scoring into a reproducible workflow with explicit validation artifacts.

SESTRAV v2 introduces an updated IEDB training dataset (720 peptides, 2.36:1 class ratio) with significantly better class balance than v1 (928 peptides, 5.58:1). The v2 analysis includes the first quantitative evidence that TCR-contact physicochemical features add discriminative value beyond binding prediction alone.

## 2) What Changed in v2

### Dataset Update
- Source: Updated IEDB T-cell assay exports (April 2026) for EBV and HPV16
- 720 unique peptides (506 positive, 214 negative) after majority-vote deduplication
- 2.36:1 positive-to-negative ratio (v1 was 5.58:1)
- 512 peptides shared with v1; 69 label flips between versions

### New Diagnostic Capabilities
- **Gold standard negatives:** 10 curated strong-binder experimentally-negative peptides; the integrated model correctly pushes 9/10 below the binding-only baseline ranking
- **30-feature SHAP analysis:** Binding features account for 60% and TCR-contact features for 40% of total model explanation power
- **Calibration analysis:** Brier skill scores of 0.20 (v2) and 0.27 (v1), both showing moderate skill above trivial baselines
- **Negative class deep dive:** Correctly classified negatives show distinctive physicochemical signatures at TCR-contact positions

## 3) Reproducibility Evidence

Canonical source and track:

- Repository: this repository (`main` branch)
- Canonical mode: 30-feature integrated track
- Dataset version: v2 (IEDB-20260424-EBV_HPV16_UPDATED)

Frozen evidence and release bundle:

- `docs/colloquium_evidence_freeze_v2.md`
- `release_artifacts/sestrav-v2-20260425T040154Z.manifest.json`
- `release_artifacts/sestrav-v2-20260425T040154Z.zip`
- `results/multi_run_stability_report.md` (3x v1 + 3x v2 pipeline cycle evidence)

v1 baseline preserved in:

- `models/v1_backup/`
- `results/v1_backup/`
- `docs/colloquium_evidence_freeze.md`

## 4) Current Results Highlights (v2)

- Pipeline runs end-to-end reproducibly (3x full cycles, zero variance).
- Gold-standard positive recovery: 15/15 found at all stages; 9/15 in top 25%.
- Gold-standard negative discrimination: 9/10 pushed down (TCR features add value).
- Feature signal: 40/60 TCR/binding SHAP split — TCR features are not noise.
- Class balance: 2.36:1 produces more defensible metrics than v1's 5.58:1.
- Above-trivial AUC-PR: +0.147 (v2) vs +0.105 (v1) — 41% improvement.
- Platt calibrator refit on v2 class distribution (2026-04-25).

Primary files:

- `results/final_validation_report.md`
- `results/v1_v2_quality_comparison.md`
- `results/gold_standard_negative_validation.csv`
- `results/calibration_metrics.csv`
- `results/negative_class_analysis.csv`

## 5) Limitations (Required)

SESTRAV v2 does **not** claim:

- wet-lab biological validation,
- clinical efficacy,
- broad population generalizability beyond current scope.

Additional v2-specific caveats:

- Smaller total dataset than v1 (720 vs 928 peptides)
- 69 label flips between v1 and v2 (9.6% of shared peptides)
- High negative misclassification rate at current operating point (82% at threshold 0.3559) reflects the 99% recall target, not model failure
- Platt calibrator needs refitting for v2 class distribution

Use standardized language in:

- `docs/limitations_statement_v1.md` (updated for v2 caveats)

## 6) Collaboration Menu (How Experts Can Engage)

### Immunology and Vaccine Biology Experts

- Review candidate-ranking biological plausibility.
- Co-design prospective validation criteria and ranking interpretation thresholds.
- Advise on immunodominance and assay-priority candidate subsets.
- Evaluate gold-standard negative set quality and suggest additional curated negatives.

### Wet-Lab Collaborators

- Define pilot validation panel from top-ranked candidates.
- Run assay campaigns (for example ELISPOT/multimer/ICS) on prioritized peptides.
- Return structured outcomes for model feedback loops.
- Validate gold-standard negative predictions experimentally.

### Bioinformatics / Data Curation Collaborators

- Expand IEDB input pool under dataset-versioning policy.
- Add external datasets and QC harmonization for robustness testing.
- Improve metadata completeness for allele/context-aware training.
- Investigate v3 union dataset (v2 labels + v1-only peptides).

### Clinical/Translational Advisors

- Help frame practical triage criteria and deployment constraints.
- Map computational outputs to realistic translational milestones.

## 7) Requested Contributions

1. Domain review of current assumptions and claim boundaries.
2. Access or guidance for additional validation-grade datasets.
3. Pilot wet-lab validation partnership design.
4. Feedback on candidate prioritization criteria and downstream utility.
5. Review of gold-standard negative selection methodology.

## 8) Expected Outputs from Collaboration

- Agreed validation protocol and success criteria.
- Expanded and versioned dataset release with documented QC.
- Prospective evaluation report linking computational ranks to biological assays.
- Updated evidence freeze and decisions record with revised support level.

## 9) Suggested Collaboration Timeline

- Weeks 1-2: protocol alignment, data audit, candidate shortlist review.
- Weeks 3-6: pilot validation execution planning and expanded-data reruns.
- Weeks 7-12: first assay-linked model assessment and publication-grade reporting outline.

## 10) Contact/Onboarding Checklist

For new collaborators:

1. Read `README.md`.
2. Read `docs/master_walkthrough_v1.md`.
3. Read `docs/master_decisions_v1.md`.
4. Review `results/v1_v2_quality_comparison.md` for dataset version context.
5. Review frozen evidence and bundle manifest.
6. Confirm limitation language before external communication.
