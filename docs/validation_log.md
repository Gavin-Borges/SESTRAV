# SESTRAV 2.0 Wet-Lab Validation Ledger

This document serves as the canonical record linking SESTRAV computational predictions to their downstream wet-lab assay outcomes. This forms the closed-loop feedback mechanism necessary for Phase 3 Horizon.

## Wet-Lab Assay Cycles

### Round 1: [Status: Planned]
- **Date Submitted to Lab:** YYYY-MM-DD
- **Target Proteome / Virus:** (e.g., HPV16 E6/E7)
- **Model Version:** (e.g., 30-feature RF canonical)
- **Assay Type:** (e.g., ELISPOT IFN-γ)
- **Collaborating Lab:** (Name of lab or PI)

#### Top Candidates Submitted
| Rank | Peptide Sequence | Target Allele | SESTRAV Score | Predicted Class (Top 25%) | Wet-Lab Assay Result | True Label assigned |
| ---- | ---------------- | ------------- | ------------- | ------------------------- | -------------------- | ------------------- |
| 1    |                  |               |               |                           |                      |                     |
| 2    |                  |               |               |                           |                      |                     |
| 3    |                  |               |               |                           |                      |                     |

#### Error Auditing Notes
*(Record any observations where high-scoring candidates failed the assay (False Positives) or known binders failed to trigger a response. Feed these observations back to `scripts/scoring_error_audit.py`.)*

---

### Instructions for Updates
1. When candidates are submitted for experimental validation, add a new round section above.
2. Once assay results are received, fill out the "Wet-Lab Assay Result" and "True Label" columns.
3. Import the newly labeled peptides back into the SESTRAV validation datasets (bumping the dataset version).
