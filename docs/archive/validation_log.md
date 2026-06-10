# SESTRAV Stage 3 Pre-Registration Log and Wet-Lab Validation Ledger

This document serves as the canonical record and pre-registration contract for the evaluation of the **SESTRAV** model on independent viral cohorts, followed by the downstream wet-lab validation ledger.

---

## Part 1: Stage 3 Computational Pre-Registration Protocol

This protocol is committed and locked *prior* to unblinding or scoring any peptides in the independent SARS-CoV-2 and Influenza A validation sets.

### 1.1 Hypotheses
*   **Primary Hypothesis (H1):** The SESTRAV 30-feature Random Forest model generalizes out-of-distribution to independent viral cohorts (SARS-CoV-2 and Influenza A) and achieves superior discrimination compared to prevalence baselines.
*   **Secondary Hypothesis (H2):** SESTRAV's joint sequence-physicochemical and binding-affinity features achieve performance gains (increased enrichment) over binding-only baselines (`binding_max`) in top ranks.

### 1.2 Evaluation Cohorts
*   **SARS-CoV-2 Cohort:** Cleaned CD8+ T-cell reactivity ELISPOT assay entries.
*   **Influenza A Cohort:** Cleaned Influenza A virus ELISPOT assay entries.

### 1.3 Exclusion and Filtering Criteria (Zero-Overlap Gate)
To prevent selection bias and claim inflation:
*   A pre-validation overlap check using the Aho-Corasick automaton will compare all validation peptides against the 720-peptide training pool.
*   Any validation peptide showing an **exact sequence match** or a **substring match** (e.g., a 9-mer training peptide embedded inside a 10-mer validation peptide, or vice versa) will be discarded.
*   Only the remaining zero-overlap peptides will be included in the evaluation.

### 1.4 Primary & Secondary Endpoints
*   **Primary Endpoint:** AUC-PR $\ge 0.75$ on the clean (overlap-excluded) validation datasets.
*   **Secondary Endpoints:**
    1.  **ISSR@10 Ratio:** The ratio of the Integrated Symbol Success Rate in the top 10% scored candidates of SESTRAV relative to binding-only baseline:
        $$\text{ISSR Ratio (R10)} = \frac{\text{ISSR@10}_{\text{SESTRAV}}}{\text{ISSR@10}_{\text{binding\_max}}} \ge 2.0$$
    2.  **Length-Stratified AUC-PR:** Evaluation of 9-mer vs. non-9-mer performance to assess length-based prediction biases.

### 1.5 Statistical Robustness
*   All metrics (AUC-ROC, AUC-PR, ISSR@10) will be reported with **95% bootstrap confidence intervals** (N=2,000 resamples, $\alpha=0.05$).
*   P-values for model metrics vs. baselines will be adjusted using Benjamini-Hochberg False Discovery Rate (FDR) correction where multiple comparators are run.

---

## Part 2: Downstream Wet-Lab Validation Ledger

This ledger links computational prioritization lists with experimental PBMC assay readouts.

### Wet-Lab Assay Cycles

#### Round 1: [Status: Planned]
*   **Date Submitted to Lab:** YYYY-MM-DD
*   **Target Proteome / Virus:** (e.g., HPV16 E6/E7)
*   **Model Version:** (e.g., 30-feature RF canonical)
*   **Assay Type:** (e.g., ELISPOT IFN-γ)
*   **Collaborating Lab:** (Name of lab or PI)

##### Top Candidates Submitted
| Rank | Peptide Sequence | Target Allele | SESTRAV Score | Predicted Class (Top 25%) | Wet-Lab Assay Result | True Label assigned |
| ---- | ---------------- | ------------- | ------------- | ------------------------- | -------------------- | ------------------- |
| 1    |                  |               |               |                           |                      |                     |
| 2    |                  |               |               |                           |                      |                     |
| 3    |                  |               |               |                           |                      |                     |

##### Error Auditing Notes
*(Record any observations where high-scoring candidates failed the assay (False Positives) or known binders failed to trigger a response. Feed these observations back to `scripts/scoring_error_audit.py`.)*

---

### Instructions for Updates
1. When candidates are submitted for experimental validation, add a new round section above.
2. Once assay results are received, fill out the "Wet-Lab Assay Result" and "True Label" columns.
3. Import the newly labeled peptides back into the SESTRAV validation datasets (bumping the dataset version).
