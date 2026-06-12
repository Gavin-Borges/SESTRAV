# Changelog

All notable changes to the SESTRAV project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0-rc1] - 2026-05-26

This release marks the release candidate for SESTRAV v2.0, focusing on hardening reproducibility, OSSF Scorecard security remediations, cross-platform portability, and dataset quality gates.

### Added
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
  - `protobuf==7.35.0` (mitigates Any-message DoS recursion vulnerability GHSA-m2f8-v8q4-3m59).
  - Pinned `nvidia-nccl-cu12==2.30.4` transitive hash matching on Linux systems.
- **Git Index Cleanup**: Cleaned up the tracking index by appending transient test/CI output artifacts (e.g. `ci_install_test.log`, `temp_test_out.txt`, and `bandit_text.txt`) to `.gitignore` to prevent tracking of local runtime logs.
- **Semgrep Custom Rules**: Restructured rules in `semgrep-rules/sestrav-custom.yml` to remove overly-broad match patterns triggering false positives on safe `load_verified_joblib` operations.
