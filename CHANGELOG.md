# Changelog

All notable changes to the SESTRAV project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed
- **Pooled same-pathogen AUC-ROC 0.9368 retracted (2026-07-11)**: The pooled within-virus
  "same-pathogen AUC-ROC 0.9368" reported for the b5ffe37 build was decoy-inflated - it only
  reproduces when synthetic / cross-pathogen decoys (incl. the vaccinia panel) are mixed in as
  if they were same-pathogen negatives - and is RETRACTED. Same-pathogen discrimination is now
  reported per-virus (within-CV mean AUC-ROC 0.751; `results/per_virus_eval_v5_mode31.csv`).
  The honest pooled same-pathogen ROC on real IEDB negatives is 0.712 (pooled AUC-PR is
  base-rate-inflated and not a headline). Self-proteome Gate 1 AUC-PR 0.8897 is unaffected.
  The historical b5ffe37 entry below is left intact as the record of what was reported then;
  see `docs/claims_register.md` D12.
- **Per-virus within-CV metrics regenerated (session 70, 2026-07-10)**: The committed
  `results/per_virus_eval_v5_mode31.{csv,json}` lagged the current 35,597-row v5 dataset and
  were regenerated. New within-CV AUC-ROC: CMV 0.819, DENV 0.859, EBV 0.790, HBV 0.708,
  HCV 0.575, HIV-1 0.894, HPV 0.561, IAV 0.856, SARS-CoV-2 0.699 (mean 0.751). HPV within-CV
  (0.561) now falls below the 0.58 Amendment-6 threshold. Leave-one-virus-out (LOO) figures are
  unchanged (mean 0.463; `results/loo_cross_virus_v5_clean.csv`). The earlier Amendment-6
  within-CV values (HPV 0.598, EBV 0.667) recorded below remain the accurate record for the
  b5ffe37 snapshot at which they were achieved.
- **v5 feature ablation added** (`models/v5/ablation/`, `models/v5/training_results_ablation.csv`):
  RF modes 21/31/33/35. Binding scores (mode 21->31) add +0.008 AUC-ROC / +0.015 AUC-PR; modes
  33 and 35 add nothing measurable. Confirms mode-31 as the production configuration.

### Added
- **v5 dataset (31,999 active rows / 46,386 total)**: Rebuilt from merged IEDB API negatives
  pipeline. Key numbers: 36,689 IEDB viral negatives, 4,219 net-new experimentally confirmed
  non-immunogenic peptides added via `scripts/merge_iedb_api_negatives.py` (bridges Pipeline A
  IEDB REST API downloads to Pipeline B v5 schema). Provenance sidecar:
  `data/immunogenicity_dataset_v5_provenance.json`. 17 singleton viruses quarantined (<50 rows
  or <10 real negatives).
- **v5 RF model (mode-31 canonical)**: Retrained on v5 dataset. Evaluation results:
  AUC-PR 0.7678 within-virus (harder same-pathogen discrimination context) / AUC-PR 0.8897
  self-proteome Gate 1 (viral epitopes vs. self-peptide background; Gate 1 threshold protocol) /
  AUC-ROC 0.9368. Per-virus Amendment 6 thresholds met: HPV >= 0.58 (achieved 0.598), EBV >= 0.57
  (achieved 0.667 post-quarantine). OOF predictions: `models/rf_oof_predictions.csv`,
  `models/rf_oof_predictions_mode31.csv`.
- **B*27 EBV conflict quarantine**: 3 rows (FRKAQIQGL x2, RRARSLSAERY) transferred to
  `data/holding/conflicts_v5_preaudit.csv`. These share sequences with label=1 rows for
  other B*27 subtypes; the allele-subtype-specific conflict is documented in claims_register
  Section 5 (ES1). EBV within-virus AUC-ROC: 0.553 (FAIL) -> 0.656 (PASS, beats 0.57 threshold).
- **`scripts/merge_iedb_api_negatives.py`**: New pipeline bridge connecting IEDB REST API
  negative downloads (`data/iedb/*.csv`) to the v5 build schema.
- **ESM-2 embedding cache**: 27,376 peptides pre-computed for GNN v5 training
  (`data/esm2_cache_v5/`). GNN t12 baseline training pending GPU availability.
- **`scripts/download_tcr3d_structures.py`**: Downloads TCR3d 2.0 TSV, applies 4 quality
  filters, downloads ~100 PDB files from RCSB with retry/backoff logic.
- **`scripts/update_contact_weights.py`**: Patches `ALLELE_CONTACT_WEIGHTS` and
  `POPULATION_AVG_CONTACT_WEIGHTS` in `src/features.py` from TCR3d-derived contact frequency
  matrices with `--dry-run` support.

### Changed
- **PyPI publish migrated to OIDC Trusted Publishers**: removed `twine` and the
  `PYPI_API_TOKEN` secret from `release.yml`; replaced with
  `pypa/gh-action-pypi-publish@release/v1` using short-lived GitHub OIDC tokens.
  No static credential is stored anywhere. The `pypi` GitHub environment is
  protected by required-reviewer approval before any upload proceeds.
  Deleted orphaned `environments/requirements-ci-twine.{in,txt}`.
- **`pyproject.toml`**: removed personal email from author metadata; name retained.
- **Library coverage gate raised** `90 -> 95` in `.coveragerc.library`. Verified
  library-scope coverage is **98.91% combined** (~99% statement / ~98% branch;
  measured 2026-06-22), comfortably clearing the new gate and the OpenSSF Gold
  targets. Synced the stale `fail_under` references in `docs/security_compliance.md`,
  `ROADMAP.md`, and the `ci.yml` step comment to the authoritative config value.

### Added
- **HBV/HCV proteome ingestion path** (issue #78): `scripts/fetch_viral_proteomes.py`
  downloads 8 UniProt sequences (HBV panel4: HBcAg P03147, HBx P03165, HBsAg-S P03138,
  HBpol P03157; HCV panel4: Core P26664, NS3 O92972, NS5A O92975, NS5B O92976) with
  HTTP retry logic and writes provenance JSON. `src/naming.py` exposes canonical IDs and
  short aliases; `config.yaml` wires `antigens` + `proteome_files` for both panels.
  Snakemake dry-run passes end-to-end. Suite: **524 tests, 2 skipped, 0 failures**.
- **`docs/antigen_accessions.md`**: sections 3 (HBV) and 4 (HCV) added with full
  UniProt accession table, strain notes, and provenance file references.

### Fixed
- **`external_predictors.py` coverage 88% → 100%** (issue #77): 13 targeted tests
  covering proline/PDE/RKYFW mock-score paths, OOB index in `parse_netchop_html`,
  poll-success return, TAPreg threshold kwarg, and parse success/empty-parse branches.
  Removed dead conditional `if mock_fallback or True:` in `query_netchop`; deleted
  unreachable `raise RuntimeError`; annotated structurally unreachable
  `except (ValueError, IndexError)` with `# pragma: no cover`.

### Security
- **Alert #51 (HIGH - Token-Permissions)**: Fixed - `dco.yml` top-level
  `permissions: contents: read / pull-requests: read` added (commit `f99ae34`).
- **Alert #52 (MEDIUM - Pinned-Dependencies)**: Dismissed false positive - pip
  smoke-test install cannot be hash-pinned by design.
- **Alert #50 (HIGH - Token-Permissions)**: Dismissed won't-fix - `contents: write`
  required by `gh release create`; top-level `permissions: read` already restricts
  all other jobs.
- **Alert #15 (HIGH - CVE-2025-3000, torch)**: Dismissed won't-fix - no upstream
  patch; `torch.jit.script` not exposed to untrusted input; EPSS 0.08%; will reopen
  when PyTorch releases a fix.
- **Dependabot #35 (torch CVE-2025-3000)**: Dismissed `tolerable_risk` - same
  rationale as alert #15.
- **Dependabot #99-#103 (5x HIGH - GitPython URL/config injection and env-var
  expansion, secret exfiltration on fetch)**: Fixed - `gitpython` bumped 3.1.52
  -> 3.1.54 (PR #157, clears GHSA-r9mr-m37c-5fr3 / GHSA-6p8h-3wgx-97gf /
  GHSA-fjr4-x663-mwxc / GHSA-3rp5-jjmw-4wv2) -> 3.1.55 (PR #158/#159, clears
  GHSA-94p4-4cq8-9g67) in both hash-pinned lockfiles (`environments/requirements-ci.txt`,
  `environments/requirements.lock`). CI-only transitive dependency (`# via snakemake`);
  not on the runtime peptide-scoring path.

## [2.0.3] - 2026-06-17

This release delivers the next test-coverage and CI hardening pass: 154 new
unit tests, a library-coverage ratchet advance to 90%, and a complete PyPI
publish pipeline.

### Added
- **154 new unit tests** across 9 test modules, bringing the total suite to
  **476 passing tests**:
  - `test_features_advanced` (23): `compute_sample_weights` virus/length
    correction, `compute_features_for_dataset` vectorised batch extraction,
    and `compute_weisfeiler_lehman_features` WL kernel.
  - `test_iedb_extractor` (43): full branch coverage of
    `src/verify/iedb_multi_virus_extractor.py` - REST mocking, VDJdb TSV
    parsing, decoy generation, `process_target`, and `main()` entry point.
  - `test_promote_gnn_runner` (15): `check_promotion_gates` short-circuit and
    aggregation logic; `promote_model` config-mutation and checksum behaviour.
  - `test_sestrav_evaluator_extended` (15): pipeline-runner paths in
    `src/verify/sestrav_evaluator.py`.
  - `test_statistical_bootstrap` (9): 98% coverage of
    `src/statistical_bootstrap.py` including the joblib worker path via
    `_inline_parallel` mock.
  - `test_train_gnn_dataset` (16): `GraphPeptideDataset` and `set_seed`
    reproducibility in `src/train_gnn.py`.
  - `test_external_predictors_extended` (13): proline/PDE/RKYFW mock-score
    paths, OOB index in `parse_netchop_html`, successful poll return, TAPreg
    threshold kwarg, and parse success/empty-parse branches.
  - `test_model_extended` (7): CUDA mock paths for `get_device` and
    `set_seeds`; `device=None` auto-detect in `train_one_fold`, `run_cv`, and
    `train_final_model`; epoch-exhaustion and `best_state=None` branches.
  - `test_features_graph` (13): `get_cb_cb_edges` for 9/10/11-mers, ERAP
    short-flanking-sequence padding, and `compute_sample_weights` without a
    peptide column.
- **PyPI publish job** in `.github/workflows/release.yml`: runs after the
  existing build/attest/GitHub-Release job; installs twine from a hash-pinned
  lockfile (`environments/requirements-ci-twine.{in,txt}`); uploads sdist and
  wheel; smoke-tests the published package from the live PyPI index. Skipped
  automatically when `PYPI_API_TOKEN` is not configured.
  - Includes a `checkout` step (without which `requirements-ci-twine.txt`
    would not be present on the runner - a bug caught pre-commit).

### Fixed
- **7 mypy type errors** resolved across four source files:
  - `src/data_curation_qc.py`: renamed lambda capture variable to eliminate a
    late-binding closure error.
  - `src/expand_negatives.py`, `src/external_benchmark_comparison.py`:
    `type: ignore[index]` annotations for typed-dict list subscripts on
    `GOLD_STANDARD_NEGATIVES`.
  - `src/external_benchmark_comparison.py`: renamed a list variable that was
    shadowed by a later `np.array` assignment of the same name.
  - `src/verify/structural_gnn.py`: `type: ignore[misc]` for the dynamic
    `Dataset if HAS_PYG else object` base class.

### Changed
- **Library coverage ratchet** advanced: `fail_under` in `.coveragerc.library`
  raised from 85 → **90**. Actual library coverage is **96.03%** combined
  statement+branch (≈96% statement, ≈94% branch) - both above the OpenSSF
  Gold targets (≥90% statement, ≥80% branch).
- **Whole-repo coverage floor** unchanged at 33 (`pyproject.toml`); actual is
  33.74%. Executable research scripts (those with a `__main__` guard) are
  validated by integration tests and CI gates, not unit statement coverage.

---

## [2.0.2] - 2026-06-16

This release completes the OpenSSF Best Practices hardening pass: governance and
assurance documentation, automated signed releases with build provenance, and a
two-scope test-coverage regime meeting the Gold coverage targets.

### Added
- **OpenSSF governance & assurance documentation**: `GOVERNANCE.md`, `ROADMAP.md`, `BUS_FACTOR.md`, `CONTRIBUTORS.md`, `docs/threat_model.md`, and `docs/security_review.md`.
- **Signed releases with provenance**: `.github/workflows/release.yml` builds the distribution on a version tag and publishes a keyless SLSA build-provenance attestation (Sigstore via GitHub OIDC), guarded by a fail-fast tag/version consistency check. Verification and the release procedure are documented in `docs/releasing.md` and `SECURITY.md`.
- **Two-scope test-coverage measurement**: library-scope coverage via `.coveragerc.library` (OpenSSF Silver `test_statement_coverage80`), kept in sync mechanically by `tools/check_library_coverage.py`, with a subprocess-coverage hook (`tools/coverage_subprocess`). Library coverage raised to ≈91% statement / ≈81% branch (OpenSSF Gold targets) with new unit tests.

### Fixed
- **Stage 4 MC-dropout path**: corrected a missing `import torch` on the uncertainty-scoring branch.
- **IEDB data loader**: added a missing `import sys`.
- **PRIME wrapper**: corrected a `temp_peptides_file` reference.
- **Dependency Security Vulnerabilities**: Additional dependency hardening applied after the v2.0.0-rc1 tag and re-compiled with `pip-compile --generate-hashes --allow-unsafe`:
  - `tornado==6.5.6` (mitigates four advisories surfaced by the OSSF Scorecard OSV scan): GHSA-fqwm-6jpj-5wxc (cookie attribute injection, high), GHSA-qjxf-f2mg-c6mc (DoS via multipart parts, high), GHSA-78cv-mqj4-43f7 (incomplete cookie validation, medium), and GHSA-cx3h-4qpv-8hc9 (out-of-bounds memory access, low). The 6.5.6 release also restores `manylinux_2_28` wheel availability (absent from 6.5.5).
  - `protobuf==7.35.1` (patch bump over the 7.35.0 baseline shipped in rc1).

### Changed
- **License detection**: `LICENSE` now opens with the canonical `MIT License` text so GitHub and automated tooling identify it as MIT (the SPDX identifier is retained in `pyproject.toml`).
- **Dependency updates** (Dependabot): `starlette` 1.1.0→1.3.1 (#75), `aiohttp` 3.14.0→3.14.1 (#74), and a Python minor/patch group of six updates (#73).

### Security
- **Hash-pinned security-scanner installs**: The `semgrep` and `pip-audit` jobs in `security.yml` now install from hash-pinned lockfiles (`environments/requirements-semgrep.txt`, `environments/requirements-pip-audit.txt`) via `pip install --require-hashes`, resolving the OpenSSF Scorecard *Pinned-Dependencies* findings. Lockfiles are generated from `.in` sources with `pip-compile --generate-hashes`.

---

## [2.0.0-rc1] - 2026-06-10

This release candidate for SESTRAV v2.0 focuses on API & frontend demo containerization, OpenSSF Scorecard compliance, and strict security and reproducibility gating.

### Added
- **FastAPI prediction microservice**: Deployed a scalable FastAPI backend (`api/main.py`) with strict input schema validation (amino acid IUPAC constraints) and cached singleton model loading.
- **Streamlit interactive interface**: Built a frontend GUI (`app/demo.py`) supporting single-peptide predictions, real-time SHAP waterfall visualizations (headless-compatible via matplotlib Agg backend), and dynamic PDF report generation.
- **Docker Compose orchestration**: Standardized deployments via a two-service compose stack (`Dockerfile.api`, `Dockerfile.demo`, `docker-compose.yml`) utilizing local-only loopback binds (`127.0.0.1`) for local research environment safety.
- **PII & absolute path gatekeeper**: Added a pre-merge action workflow (`.github/workflows/pii_scan.yml`) to block commits containing machine-specific filesystem path leaks (Windows user-profile or WSL mount paths) or unresolved TODO placeholders.
- **Hypothesis Property-Based Fuzzing**: Integrated standard property-based fuzz tests in CI (`.github/workflows/fuzzing.yml` and `tests/test_fuzz.py`) with customizable test example ranges (200 for standard pushes, 1000 for weekly schedules).
- **Consensus Rank Aggregation**: Added Borda count-based rank aggregation ensemble inside `src/consensus_ensemble.py` as a robust alternative to geometric mean pooling to bypass zero-cancellation issues.
- **Aho-Corasick Contamination Gate**: Deployed a dedicated verification step in Stage 3 to screen IEDB evaluation records against the training corpus for exact and substring contamination.
- **License SPDX Identifier**: Added machine-readable `SPDX-License-Identifier: MIT` tag to `LICENSE` for automated OSSF Scorecard detection.

### Changed
- **Cross-Platform Path Standardization**: Standardized absolute Windows filesystem paths (`C:\Users\gavin\...`) to relative, POSIX-compliant expressions (`Path` bindings, `.relative_to().as_posix()`, and relative markdown paths) across `README.md`, `src/verify/sestrav_evaluator.py`, and `scripts/benchmark_runner.py` to allow execution on UNIX/Linux/WSL hosts.
- **GitHub Actions Security Hardening**: Pinned all upstream action runners to secure, verified commit SHAs rather than mutable version tags. Locked down workflow run tokens to a strict `permissions: read-all` default state.
- **Branch Rulesets & Review Gating**: Applied automated branch protection configurations via Git credential tokens:
  - Required PR reviews for external contributors while allowing frictionless self-merge bypasses for the repo owner.
  - Enforced strict merge gates requiring status checks (`Require human review` and `SESTRAV CI / test (3.13)`) to pass on clean branches.
  - Restricted branch deletions and force pushes on `main`.

### Removed
- **Unused Stub Codes**: Removed orphaned empty duplicate stubs of `run_evaluation_pipeline` in `src/verify/sestrav_evaluator.py` to prevent naming clashes.

### Fixed
- **Dependency Security Vulnerabilities**: Upgraded minimum versions for vulnerable libraries in `requirements.in` and compiled the hashes using `pip-compile --generate-hashes --allow-unsafe`:
  - `keras==3.14.1` (mitigates GHSA-36fq-jgmw-4r9c, GHSA-4f3f-g24h-fr8m, GHSA-cjgq-5qmw-rcj6, GHSA-hjqc-jx6g-rwp9, GHSA-mq84-hjqx-cwf2, GHSA-7gcm-g887-7qv7).
  - `protobuf==7.35.1` (mitigates Any-message DoS recursion vulnerability GHSA-m2f8-v8q4-3m59).
  - Pinned `nvidia-nccl-cu12==2.30.4` transitive hash matching on Linux systems.
- **Git Index Cleanup**: Cleaned up the tracking index by appending transient test/CI output artifacts (e.g. `ci_install_test.log`, `temp_test_out.txt`, and `bandit_text.txt`) to `.gitignore` to prevent tracking of local runtime logs.
- **Semgrep Custom Rules**: Restructured rules in `semgrep-rules/sestrav-custom.yml` to remove overly-broad match patterns triggering false positives on safe `load_verified_joblib` operations.

---

## Version 2.0.0 (2026-06-04)

### Release Summary

SESTRAV v2.0.0 finalizes the semester core pipeline and integrates advanced computational biology models and validation tracks for public release using the **v2.0.0-alpha dataset** (expansion_alpha).

- **Canonical release track**: **30-feature integrated model/config** (20 physicochemical + 10 multi-allele MHC binding features)
- **Secondary/Optional track**: Neural Network (FlexibleMLP) and Graph Neural Network (GCN/GAT) benchmark modules
- **Legacy comparator track**: **21-feature sequence-only configuration** (for historical comparison)
- **Training dataset**: **v2.0.0-alpha** (1004 peptides, 3.35:1 class ratio)

### What Is Included

- **Four-stage pipeline**: Peptide generation, multi-allele binding prediction, feature extraction, and immunogenicity scoring (RF and XGBoost)
- **FlexibleMLP Extension**: PyTorch ANN classification with 14-configuration hyperparameter architecture search
- **GNN Benchmark Suite**: GCN, GAT, and Bipartite Peptide-Allele graphs for structure-based benchmarking
- **Ablation Studies**: Multi-group feature ablation analyses to quantify contact-residue contribution
- **Final validation bundle generation**:
  - `results/gold_standard_validation.csv`
  - `results/baseline_comparison.csv`
  - `results/h2_tier_a_summary.csv`
  - `results/final_validation_report.md`
- **Security & Dependency Hardening**:
  - Refactored scripts clean of `bandit` security findings (such as shell injections, path handling, and try-catch safety)
  - Upgraded dependencies inside `environments/requirements.lock` resolving 9 CVEs/vulnerabilities
- **Multi-run stability evidence**: `results/multi_run_stability_report.md` demonstrating perfect deterministic reproducibility
- **Platt calibrator refit** on the v2 class distribution to output calibrated probabilities

### Key Results (v2)

- RF AUC-ROC: `0.5684` | AUC-PR: `0.8047` | ISSR@10: `0.7895` | ISSR@25: `0.8285`
- Gold-standard positive recovery: `15/15` found, `7/15` in top 25% (R10 = 0.9494)
- Gold-standard negative discrimination: `9/10` pushed down (TCR features add value)
- SHAP feature split: 60% binding / 40% TCR-contact features
- H2 Tier A decision: **NOT SUPPORTED** ($R_{10} = 0.9494$, below standard threshold)

### Reproducibility Commands

```bash
conda env create -f environment.yml
conda activate sestrav
pip install snakemake
mhcflurry-downloads fetch models_class1_presentation
python -m src.train_classifier --data data/immunogenicity_dataset_v3.csv --feature-mode 30 --binding-matrix models/peptide_binding_matrix_v3.csv
python -m src.train_classifier --data data/immunogenicity_dataset_v3.csv --feature-mode 21
python -m pytest tests/ -v
snakemake --snakefile pipeline.smk --cores 4
python -m src.final_validation_report --results-dir results --model-dir models --data data/immunogenicity_dataset_v3.csv --binding-matrix models/peptide_binding_matrix_v3.csv --model-path models/rf_30feature_integrated.joblib --dataset-mode expansion_alpha --dataset-version 2.0.0-alpha
python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v2
```

### Known Environment Notes

- Base Python 3.13 is not compatible with this `mhcflurry` stack. Use the project conda env with Python 3.11.
- `setuptools==80.9.0` is required for `pkg_resources` compatibility with the `mhcflurry` release.
- Model serialization warnings may appear if scikit-learn versions differ from the model training environment (models should be trained fresh each cycle).
- XGBoost SHAP TreeExplainer has compatibility issues with the `shap` library version; RF SHAP (the canonical model) works correctly.

### Canonical Decision Statement

The 30-feature integrated track is selected as canonical because it best balances:
- predictive performance evidence,
- biological defensibility,
- reproducibility readiness, and
- alignment with proposal scope.

The v2 dataset is selected over v1 because:
- Class balance is more honest (3.35:1 vs 5.58:1)
- 63% more negative training examples (231 vs 141)
- Gold-standard negative discrimination (9/10 pushed down) is a novel capability
- TCR features contribute 40% of model explanation power (SHAP)

The 21-feature track remains documented as a legacy comparator.

### Limitation and Claim Boundary (Required)

SESTRAV v2 should be communicated as a reproducible computational prioritization prototype. It should not be described as biologically or clinically validated in this release.

Use:
- `docs/limitations_statement_v1.md`
- `docs/archive/colloquium_evidence_freeze_v2_20260524.md`
- `results/final_validation_report.md`

for standardized non-overclaim language and current supported statements.

---

## Version 1.0.0 (2026-04-01)

### Release Summary

Historical baseline release of SESTRAV using the v1 dataset (928 peptides, 5.58:1 class ratio) with the legacy 21-feature sequence-only comparator track.

### Key Results (v1)

- RF AUC-ROC: `0.820` | AUC-PR: `0.953` | Above-trivial AUC-PR: `+0.105`
