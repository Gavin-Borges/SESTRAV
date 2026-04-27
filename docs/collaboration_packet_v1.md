# SESTRAV v1 Collaboration Packet

## 1) One-Page Technical Abstract

SESTRAV (Structural Epitope Scoring via TCR Recognition And Vaccinology) is an open-source computational pipeline for ranking candidate therapeutic vaccine epitopes in HPV and EBV proteomes. The pipeline combines peptide generation, multi-allele MHC presentation prediction, TCR-facing feature extraction, and immunogenicity scoring into a reproducible workflow with explicit validation artifacts.

SESTRAV v1 is positioned as a **computational prioritization prototype**. It is designed to improve candidate triage and hypothesis generation while clearly separating computational evidence from biological validation claims.

## 2) Reproducibility Evidence

Canonical source and track:

- Repository: this repository (`main` branch)
- Canonical mode: 30-feature integrated track

Frozen evidence and release bundle:

- `docs/colloquium_evidence_freeze.md`
- `release_artifacts/sestrav-v1-20260424T203715Z.manifest.json`
- `release_artifacts/sestrav-v1-20260424T203715Z.zip`

Repeated full-run status:

- Documented in `docs/reproducibility_finalization_status.md`
- Includes repeated full forced reruns and validation regeneration in pinned conda environment.

## 3) Current Results Highlights (v1)

- Pipeline runs end-to-end reproducibly from canonical repository assets.
- Gold-standard recovery and baseline comparison outputs are generated and frozen.
- H2 Tier A integrated-over-binding ratio is currently **NOT SUPPORTED** in frozen snapshot.
- Current evidence supports use as a computational workbench and ranking prototype.

Primary files:

- `results/final_validation_report.md`
- `results/h2_tier_a_summary.csv`
- `results/gold_standard_validation.csv`
- `results/baseline_comparison.csv`

## 4) Limitations (Required)

SESTRAV v1 does **not** claim:

- wet-lab biological validation,
- clinical efficacy,
- broad population generalizability beyond current scope.

Use standardized language in:

- `docs/limitations_statement_v1.md`

## 5) Collaboration Menu (How Experts Can Engage)

### Immunology and Vaccine Biology Experts

- Review candidate-ranking biological plausibility.
- Co-design prospective validation criteria and ranking interpretation thresholds.
- Advise on immunodominance and assay-priority candidate subsets.

### Wet-Lab Collaborators

- Define pilot validation panel from top-ranked candidates.
- Run assay campaigns (for example ELISPOT/multimer/ICS) on prioritized peptides.
- Return structured outcomes for model feedback loops.

### Bioinformatics / Data Curation Collaborators

- Expand IEDB input pool under dataset-versioning policy.
- Add external datasets and QC harmonization for robustness testing.
- Improve metadata completeness for allele/context-aware training.

### Clinical/Translational Advisors

- Help frame practical triage criteria and deployment constraints.
- Map computational outputs to realistic translational milestones.

## 6) Requested Contributions

1. Domain review of current assumptions and claim boundaries.
2. Access or guidance for additional validation-grade datasets.
3. Pilot wet-lab validation partnership design.
4. Feedback on candidate prioritization criteria and downstream utility.

## 7) Expected Outputs from Collaboration

- Agreed validation protocol and success criteria.
- Expanded and versioned dataset release with documented QC.
- Prospective evaluation report linking computational ranks to biological assays.
- Updated evidence freeze and decisions record with revised support level.

## 8) Suggested Collaboration Timeline

- Weeks 1-2: protocol alignment, data audit, candidate shortlist review.
- Weeks 3-6: pilot validation execution planning and expanded-data reruns.
- Weeks 7-12: first assay-linked model assessment and publication-grade reporting outline.

## 9) Contact/Onboarding Checklist

For new collaborators:

1. Read `README.md`.
2. Read `docs/master_walkthrough_v1.md`.
3. Read `docs/master_decisions_v1.md`.
4. Review frozen evidence and bundle manifest.
5. Confirm limitation language before external communication.
