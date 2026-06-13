# SESTRAV Strategic Roadmap: Future Milestones & Execution Plan

This strategic document details the remaining computational, validation, and professional objectives of the SESTRAV (Structural Epitope Scoring via TCR Recognition and Vaccinology) ecosystem, drawing on the content within the `08_Future` folder.

---

## 1. Immediate Execution short (Weeks 17–18)

These represent the final pending checkpoints to close the current release cycle and lock the repository's security posture.

### Task 1.1: OpenSSF Passing Badge Submission (Target: Week 17)
- **Objective:** Register the repository and obtain the "Passing" level security certification.
- **Checklist:**
  1. Navigate to [bestpractices.coreinfrastructure.org](https://bestpractices.coreinfrastructure.org/) and log in via GitHub.
  2. Register `https://github.com/Gavin-Borges/SESTRAV` as a new project.
  3. Fill out the questionnaire using the completed justifications mapped in [openssf_best_practices_readiness.md](docs/openssf_best_practices_readiness.md) (referencing Pydantic type checking, secure dependency hash locking in `requirements.txt`, private vulnerability channels in `SECURITY.md`, and Bandit/CodeQL static scans).
  4. Replace the pending OpenSSF badge markdown placeholders in the root [README.md](README.md) with the generated badge ID.

### Task 1.2: Release v2.0.0-rc1 Tagging & Packaging (Target: Week 18)
- **Objective:** Freeze the audited codebase, build source/wheel targets, and tag the release state.
- **Checklist:**
  1. Compile release artifacts using `python -m src.release_bundle` (this generates checksums and packages the models).
  2. Verify the project build sequence using the PyPA build system:
     ```bash
     conda run -n sestrav python -m pip install --upgrade build
     conda run -n sestrav python -m build
     ```
  3. Tag the frozen repository commit:
     ```bash
     git tag -a v2.0.0-rc1 -m "Freeze release candidate 1 with Pydantic configurations and dynamic GNN validation"
     git push origin v2.0.0-rc1
     ```

---

## 2. Near-Term Horizon (Months 1–4)

This phase prioritizes validation infrastructure and feature expansion to eliminate biological modeling blindspots.

### Stage 3: Independent Validation Cohort (Pre-Registered)
- **Objective:** Establish generalizability on datasets strictly separated from training cohorts.
- **Candidate Cohorts (Priority Order):**
  1. **SARS-CoV-2 MHC-I (Post-2024 IEDB):** High N, 9-mer enriched, zero overlap by design.
  2. **Influenza A (H1N1/H3N2) ELISPOT:** Large published cohorts, HLA-diverse.
  3. **Grifoni et al. (Cell 2020) SARS-CoV-2:** Gold-standard curated benchmark.
- **Execution Rules:**
  - **Zero-Overlap Check:** Programmatically verify zero exact or substring overlap between candidate cohorts and the SESTRAV training set.
  - **Pre-Registration:** Write and commit a pre-registration entry to `docs/validation_log.md` detailing the primary endpoint (AUC-PR ≥ 0.75 target), bootstrap resamples configuration ($N \ge 2000$), and false discovery rate correction (Benjamini-Hochberg) **before** unblinding model scores.

### Stage 4: Allele-Aware Ingestion & Antigen Processing Features
- **Objective:** Move beyond "allele-blind" models and bridge the 9-mer prediction gap with competitor tools.
- **Step 4.1: IEDB T-cell Assay Ingestion:**
  - Migrate raw data ingestion from name-based epitope lists to row-level T-cell Assay exports (`data/raw/iedb_tcell_assay/`).
  - Map alleles to Sette & Sidney supertypes (`src/hla_supertypes.py`) to handle underpopulated alleles.
- **Step 4.2: Add Cleavage and Transport Features (Mode B):**
  - **Feature 31 (C-terminal Cleavage):** Query proteasomal cleavage likelihood via the NetChop 3.1 web API, requiring a 6-residue protein flanking context.
  - **Feature 32 (TAP Transport):** Calculate transporter efficiency scores using IEDB's Class I Processing Prediction tool.
  - **Continuity Guard:** Write a `pytest` continuity rule in `tests/test_data_integrity.py` to ensure that all 15 positive gold-standard epitopes are maintained and correctly labeled in the new allele-aware database.

---

## 3. Medium-Term Horizon (Months 5–9)

This phase builds translation artifacts, completes deep learning evaluations, and launches local interfaces.

### Stage 5: Advanced Model Promotion Review (Month 9)
- **Objective:** Audit the experimental PyTorch ANN and GNN models against the promotion rubric in `docs/ann_gnn_promotion_rubric.md`.
- **Checklist:**
  - Run logistic regression and XGBoost as baseline linear comparators on the new allele-aware dataset.
  - Test if GNN/ANN out-of-fold metrics achieve an enrichment ratio lower-bound $R_{10} \ge 2.0$ over binding-only baselines.
  - Verify if promoted architectures correctly down-rank $\ge 80\%$ of hard-binder negative decoy controls.

### Stage 8: Streamlit App, REST API & Container Deployment (Month 10)
- **Objective:** Provide a lightweight interface for biology collaborators to score candidates without coding.
- **Deliverables:**
  - **Streamlit Demo (`app/demo.py`):** Accepts peptide strings, plots SHAP waterfalls for the top 10 features, and provides plain-language immunogenicity scoring interpretations.
  - **FastAPI HTTP Microservice (`api/main.py`):** Exposes `POST /score` (returns RF and GNN scores + SHAP vectors) and `GET /provenance` (returns Zenodo DOI, checksums, and model cards).
  - **Docker Compose:** Configures local uvicorn execution and loopback binding (`127.0.0.1`) to prevent shared network data leakage.

---

## 4. Long-Term Horizon (Months 9–12+)

This phase targets preprint dissemination, software packaging, and establishing prospective biological partnerships.

### Stage 6: Peer-Reviewed Publication & Dissemination
- **Step 6.1: Zenodo Preservation:**
  - Archive all model weights, frozen benchmarking runs (`extval_20260520_1607_gb_tierA`), and requirements with a permanent Zenodo DOI.
  - Update `CITATION.cff` with the minted DOI.
- **Step 6.2: Publication Outlets (Priority Order):**
  1. **Journal of Open Source Software (JOSS):** Focuses on code quality, testing, and documentation. Submit here first to establish software validation.
  2. **Oxford Bioinformatics / PLOS Computational Biology:** Submit methodology and contamination findings (the 36.9% external competitor overlap) once Stage 3 independent validation is complete.
  3. **bioRxiv Preprint:** Post manuscript 2–4 weeks prior to journal submission to assert priority and solicit computational biology feedback.

### Stage 7: Prospective Wet-Lab Validation (Collaboration-Contingent)
- **Objective:** Conduct ELISpot confirmation on donor PBMCs for new prioritizations.
- **Validation Protocol:**
  - OSF pre-registration of cohort testing metrics.
  - Cohort design: 60–80 peptides partitioned into Arm A (SESTRAV top prioritizations), Arm B (binding-only matching controls), and positive/negative controls (Arms C/D).
  - Execute IFN-gamma ELISpot assays on HLA-matched donor PBMCs and re-ingest results to retrain the feature models.

---

## 5. Gavin Borges: Career & URI Networking Strategy

To secure the collaboration and funding required for Stage 7 wet-lab assays, activate local university connections.

```
                  [Gavin Borges (URI Developer)]
                                │
       ┌────────────────────────┼────────────────────────┐
       ▼                        ▼                        ▼
[Iris Schellenberg]     [Emine Byers]           [Charles Jouaneh]
   (Shih Lab member)       (Former MIT)            (Shih Lab member)
       │                        │                        │
       └───────────┬────────────┘                        │
                   ▼                                     ▼
        [Shih Lab (saRNA Vaccines)]             [URI Legorreta Center]
        (PI Introduction Target)                 (Cold-to-Warm Pitch)
```

1. **Activate Shih Lab (saRNA cancer vaccines):**
   - *Target:* Reconnect with co-authors **Iris Schellenberg** and **Charles Jouaneh** (both members of the Shih Lab).
   - *Action:* Request a PI introduction to discuss how SESTRAV’s high-precision epitope scoring can inform saRNA antigen selection, bypassing traditional binding-only inflation.
2. **Review with Emine Byers:**
   - *Target:* Co-author and former MIT Technical Associate.
   - *Action:* Send the finalized model cards and collaboration packet to Emine for technical feedback. Her experience with peer-reviewed computational biology reviews will refine your submission arguments.
3. **Engage Legorreta Cancer Center (URI):**
   - *Target:* Faculty investigating HPV16/EBV therapeutic vaccine designs.
   - *Action:* Draft a short, 3-sentence introduction linking your pre-registered validation plan and the Zenodo DOI, offering computational target screening support.
