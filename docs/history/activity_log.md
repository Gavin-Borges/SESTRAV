# SESTRAV Chronological Activity & Implementation Log

This log chronicles the major technical milestones, model evaluation runs, security audits, and packaging finalizations executed in the SESTRAV repository. Use this ledger to trace project progression, audit past actions, and establish starting coordinates for future development sprints.

---

## 📅 June 10, 2026 — Release v2.0.1 & Workspace Hardening

### 🛠️ GNN Ingestion & Verification Remediation
- **Refactoring:** Upgraded [src/train_gnn.py](file:///C:/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev/src/train_gnn.py) to aggregate validation predictions natively across Stratified 5-Fold cross-validation. Genuine predictions are exported to `models/gnn_oof_predictions.csv`. Checkpoint path standardized to `models/gnn/structural_gnn_v2.pth`.
- **Hardened Gates:** Refactored [src/verify/promote_gnn.py](file:///C:/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev/src/verify/promote_gnn.py) to remove mocked evaluations and enforce a mathematical promotion gate requiring genuine OOF AUC-PR $\ge 0.85$.
- **Validation Run:** Verified that training the GNN on a small validation cohort correctly generates a real AUC-PR of **0.512**, causing the promotion orchestrator to correctly **REJECT** the GNN model from canonical status (as documented in the [GNN Remediation Report](file:///C:/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev/docs/history/GNN_Remediation_and_Verification_Report.md)).
- **PyPI Build Warning Fix:** Updated [pyproject.toml](file:///C:/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev/pyproject.toml) to replace the deprecated license table syntax with a standard SPDX expression: `license = "MIT"`. Package compile (`python -m build`) completes with zero warnings.

### 📁 Workspace Restructuring & Archiving
- **Consolidated Documentation:** Standardized `docs/` by creating `docs/archive/` (obsolete design plans and migration drafts) and `docs/history/` (detailed run reports and evaluation logs) to maintain context economy.
- **Wet-Lab Pre-Registration:** Moved the Phase 7 prospective clinical protocol from the legacy root folder `11_External-Testing` to [docs/Wet_Lab_Protocol_v1.md](file:///C:/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev/docs/Wet_Lab_Protocol_v1.md).
- **Roadmap Archiving:** Archived the successfully implemented technical roadmap plans (`phase1` through `phase7`) to [08_Future/archive/](file:///C:/Users/gavin/.gemini/antigravity/scratch/SESTRAV/08_Future/archive).
- **Git Release Synced:** Pushed workspace updates to `release/2.0-rc1` and synchronized the production repository branch `fix/sestrav-v2-finalization`. Successfully tagged release `v2.0.1` on GitHub.

---

## 📅 June 8, 2026 — Checksum Generation & Model Verification
- **Checksums:** Compiled [models/model_artifact_checksums.json](file:///C:/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev/model_artifact_checksums.json) containing SHA-256 hashes for RF (30/21/50 features), XGBoost (30/21/50 features), and ANN checkpoints.
- **Verification Tests:** Configured `tests/test_model_load.py` and `tests/test_artifact_integrity.py` to auto-verify weights integrity and check bounds during initialization.

---

## 📅 June 3, 2026 — Forensic Security Audit & CI Workflows
- **Dependency Hardening:** Upgraded the dependency registry via `pip-compile --generate-hashes`. Pinned security versions for `keras >= 3.13.2` (resolving 6 CVEs) and `protobuf >= 5.29.6` (resolving DoS CVE).
- **CI Workflows:** Pinned GitHub Actions dependencies to full commit SHAs. Implemented `permissions: read-all` policies across:
  - `security.yml`: AST security scans (Bandit + CodeQL + Semgrep python).
  - `ci.yml`: Automated pytest, dataset QC gates, benchmark runs, and Quarto rendering.
  - `fuzzing.yml`: Hypothesis property-based testing (200 standard, 1000 weekly examples).
  - `pr-review-check.yml`: Blocks external PRs failing peer review checks.

---

## 📅 May 24, 2026 — Pre-Publish Decisions & Evidence Freeze
- **Decisions Locked:** Executed pre-publish validation runs under strict `freeze_mode: true`. Confirmed Random Forest (30-feature) OOF configuration as the authoritative canonical release (AUC-PR **0.828** Tier A).
- **Snapshots:** Generated the [Latest Publish Gate Report](file:///C:/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev/docs/history/final_publish_gate_report_20260524.md) and [Colloquium Evidence Freeze](file:///C:/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev/docs/history/colloquium_evidence_freeze_v2_20260524.md), writing a successful validation flag (`"valid": true`) to `results/freeze_status.json`.

---

## 📅 May 20, 2026 — Horizon 0 External Validation
- **Benchmarking:** Executed the computational benchmarking cycle against state-of-the-art predictors PredIG-Path and PRIME 2.1 across Tier A and Tier B.
- **Contamination Quantification:** Discovered **36.9%** exact and substring overlap in external tools' evaluation sets. On overlap-excluded clean holdouts (N=451), SESTRAV RF leads (RF 0.822 vs. PRIME 0.720, PredIG 0.688) as logged in [results/external_benchmark_comparison.md](file:///C:/Users/gavin/.gemini/antigravity/scratch/SESTRAV/SESTRAV-Dev/results/external_benchmark_comparison.md).
