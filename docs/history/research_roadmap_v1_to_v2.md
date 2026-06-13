# SESTRAV Research Roadmap (v1 -> v2)

This roadmap defines the post-v1 path to improve biological credibility, validation depth, and collaboration readiness.

## Roadmap Principles

1. Preserve reproducibility while expanding scope.
2. Separate exploratory improvements from canonical claims.
3. Prioritize expert-guided biological validation design.
4. Keep claim language matched to evidence maturity.

## Phase 1 (1-2 Months): Controlled Expansion and Validation Readiness

## Objectives

- Expand and version IEDB inputs without breaking baseline comparability.
- Improve robustness and uncertainty reporting.
- Prepare external collaboration workflows and protocol drafts.

## Deliverables

1. Dataset versioned expanded run:
   - New Mode B dataset version ID
   - Updated `docs/iedb_data_counts.md`
   - Side-by-side baseline vs expanded metrics
2. Reproducibility hardening:
   - Repeat full run validation in pinned environment
   - Updated freeze and release bundle manifest
3. Model and validation analysis updates:
   - Refreshed `results/h2_tier_a_summary.csv`
   - Refreshed `results/baseline_comparison.csv`
   - Data-bias and sensitivity reruns
4. Collaboration onboarding package:
   - finalized `docs/collaboration_packet_v1.md` distribution version
   - shortlist of domain reviewers and target labs

## Success Criteria

- Expanded-data results are transparently versioned and reproducible.
- No cross-mode claim mixing in public materials.
- At least one expert feedback loop is operational.

## Phase 2 (3-6 Months): Biological Validation Path and Project Scaling

## Objectives

- Transition from computational-only prototype to biologically informed validation pathway.
- Improve generalization confidence across broader data conditions.
- Position project for publication/funding/collaboration opportunities.

## Deliverables

1. External validation design package:
   - Candidate selection protocol for assay testing
   - Defined endpoints and interpretation rubric
2. Broader data and scope extension:
   - Additional pathogens and/or broader HLA coverage (as feasible)
   - Explicit registry of added datasets and QC gates
3. Evidence maturity upgrades:
   - Updated decision record on supported vs unsupported hypotheses
   - Revised claim-boundary statement based on new evidence
4. Opportunity outputs:
   - Research-facing technical brief
   - Collaboration proposal templates
   - Preprint-ready methods and limitations narrative draft

## Success Criteria

- Clear expert-reviewed validation plan exists.
- At least one external evidence stream is integrated into evaluation strategy.
- Project outputs are ready for broader academic/research dissemination.

## Collaboration Strategy by Role

### Immunology Experts

- Validate biological plausibility assumptions.
- Advise candidate prioritization criteria and assay interpretation.

### Wet-Lab Partners

- Co-design practical pilot assay plan for top-ranked candidates.
- Return structured assay outputs for iterative model assessment.

### Computational/ML Collaborators

- Improve calibration, robustness checks, and reproducibility automation.
- Support scalable data integration and model benchmarking.

## Risk Controls Across Both Phases

1. Maintain a canonical evidence freeze for every major claim update.
2. Keep limitation language synchronized across README, reports, and slides.
3. Require dataset/version provenance for every rerun in external-facing materials.
4. Avoid biological overclaiming until validation evidence supports promotion.

## Promotion Criteria Toward v2 Positioning

SESTRAV can be repositioned beyond v1 prototype framing only when:

1. Repeated expanded-data runs remain reproducible.
2. Validation evidence improves with transparent uncertainty bounds.
3. External expert feedback is incorporated into protocol and interpretation.
4. Documentation and evidence bundles are refreshed to match new support levels.
