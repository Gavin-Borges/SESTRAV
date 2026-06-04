# SESTRAV Master Decisions Record (v1 Prototype)

This document records major decisions, rationale, and tradeoffs for SESTRAV v1.
It is the authoritative decision log for explaining why the current prototype is shaped the way it is.

## 1) Repository and Execution Scope

### Decision

Use this repository (`main` branch) as the sole authoritative v1 codebase.

### Why

- Multiple workspace copies existed and could cause drift.
- A single canonical path is required for reproducibility and public claims.

### Tradeoff

- Historical work in other folders remains useful context but is not directly claimable without revalidation.

### References

- `docs/canonical_source_of_truth.md`
- `docs/reproducibility_finalization_status.md`

## 2) Canonical Track Selection (30-feature integrated)

### Decision

Set the 30-feature integrated configuration as canonical; keep 21-feature as historical comparator.

### Why

- Better aligns with integrated SESTRAV design intent (TCR physicochemical features plus multi-allele binding context).
- Matches current default in `config.yaml` and Stage 4 model compatibility logic.

### Tradeoff

- Legacy documents and scripts may still reference 21-feature assumptions and require explicit labeling to avoid confusion.

### References

- `config.yaml`
- `docs/canonical_selection_scorecard.md`
- `README.md`

## 3) Validation Narrative and Claim Boundaries

### Decision

Use current frozen evidence to communicate computational performance honestly, including unsupported hypotheses.

### Why

- Current H2 Tier A row is not supportive (`R10 < 2`), and baseline binding-only is stronger in committed snapshot.
- Transparent reporting increases scientific credibility and collaboration readiness.

### Tradeoff

- Narrative is more conservative and less headline-optimized.

### References

- `results/final_validation_report.md`
- `results/h2_tier_a_summary.csv`
- `results/baseline_comparison.csv`
- `docs/H2_ISSR_evaluation_protocol.md`

## 4) Reproducibility-First Release Gating

### Decision

Require repeated full reruns and a checksum-based release bundle before final v1 sharing.

### Why

- Ensures outputs are stable and traceable.
- Supports external reviewer confidence and downstream collaboration.

### Tradeoff

- Adds operational overhead and time to each release cycle.

### References

- `docs/reproducibility_finalization_status.md`
- `docs/colloquium_evidence_freeze.md`
- `release_artifacts/sestrav-v1-20260424T203715Z.manifest.json`

## 5) IEDB Data-Pool Policy

### Decision

Formalize two modes:

- Mode A frozen baseline (release claims)
- Mode B expanded IEDB exploratory reruns (future improvement path)

### Why

- You explicitly need the ability to increase/change the IEDB pool in this prototype phase.
- Dataset changes can materially alter metrics and must be tracked/versioned.

### Tradeoff

- Additional documentation burden for each expanded rerun.

### References

- `docs/iedb_mode_policy.md`
- `docs/training_data_strategy.md`
- `docs/iedb_data_counts.md`

## 6) Limitations and Biological Accuracy Position

### Decision

Position v1 as a computational prioritization prototype, not biologically validated truth.

### Why

- No wet-lab validation in this version.
- Dataset and cohort limitations constrain biological generalization claims.

### Tradeoff

- Must avoid overclaiming in outreach materials; may reduce immediate perceived impact while improving scientific integrity.

### References

- `README.md`
- `docs/sestrav_final_clarity_accuracy_audit.md`
- `05_Risk and Obstacles/SESTRAV_Discrepancy_Analysis_and_Forward_Plan.md`
- `08_Future/FUTURE.txt`

## 7) Collaboration-Oriented Product Direction

### Decision

Package v1 as a reproducible prototype ready for expert review and co-development rather than final biological endpoint delivery.

### Why

- Aligns with current evidence maturity.
- Creates concrete opportunities for immunology/research partnerships.

### Tradeoff

- Requires tighter communication artifacts and clear asks for collaborators.

### References

- `docs/master_walkthrough_v1.md`
- `docs/colloquium_evidence_freeze.md`
- `docs/reproducibility_finalization_status.md`

## 8) Key Changes Over Time (Condensed)

1. Initial project scope emphasized integrated TCR-aware ranking and multiple benchmark aspirations.
2. Final closeout prioritized reproducibility, canonical path clarity, and transparent hypothesis reporting.
3. The v1 decision stack now favors claim discipline and collaboration readiness over aggressive performance positioning.
