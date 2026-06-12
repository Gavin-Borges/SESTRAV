# SESTRAV Security & Compliance Posture

This document tracks SESTRAV's readiness for the [OpenSSF Best Practices Badge](https://bestpractices.coreinfrastructure.org/en) (formerly CII Best Practices). SESTRAV targets the **Passing** level criteria as of version 2.0.

## 1. Basics
- **Project Description:** SESTRAV is a T-cell epitope immunogenicity prediction pipeline.
- **URL / Repository:** Hosted on GitHub (https://github.com/Gavin-Borges/SESTRAV).
- **License:** Open Source (specified in repo).

## 2. Change Control
- **Version Control:** Git/GitHub is strictly utilized.
- **Release Tracking:** Releases are tagged (e.g., `release/2.0-rc1`).
- **Review:** All pull requests to `main` require a successful GitHub Actions CI check before merging.

## 3. Reporting
- **Vulnerability Reporting:** Described in `SECURITY.md`. Issues can be reported confidentially; the maintainers pledge to respond to vulnerabilities promptly.
- **Bug Tracking:** GitHub Issues is used as the primary issue tracker.

## 4. Quality
- **Automated Testing:** 
  - Over 100 pytest test cases (`tests/`) validate core logic, schema conformity, and feature store deterministic behavior.
  - Test suite executes cleanly with no critical failures.
- **CI/CD Integration:** Tests are executed via `.github/workflows/ci.yml` on every pull request.
- **Strict Data Typing:** Pydantic is utilized via `SestravConfig` to enforce configuration schema, preventing runtime type coercion errors and hardcoded path injection.

## 5. Security
- **Secure Architecture:** 
  - Avoidance of `eval()`/`exec()` and unsafe `shell=True` subprocesses.
  - Transitioned from unsafe file loading (`json.loads(open(path).read())`) to context-managed explicit IO logic.
  - Safe model unpickling via `ModelRegistry` validating expected features.
- **Dependency Management:** Dependencies are rigorously pinned with `--require-hashes` using `pip-compile` to prevent supply chain injection.
- **Static Analysis (SAST):** CodeQL, Bandit, and Semgrep are integrated into GitHub actions.

## 6. Analysis
- **Dynamic Analysis / Fuzzing:** The pipeline supports `freeze_mode` constraints to ensure data immutability.
- **Hygiene:** 0 known vulnerabilities exist in the core python codebase (verified by Bandit).

## Future Upgrades
We plan to introduce automated dependency updates (e.g. Dependabot/Renovate) and Hypothesis-based fuzz testing for scientific artifacts.
