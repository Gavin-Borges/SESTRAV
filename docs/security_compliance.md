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
  - 200+ pytest test cases (`tests/`) validate core logic, schema conformity, and feature store deterministic behavior.
  - Test suite executes cleanly with no critical failures.
- **CI/CD Integration:** Tests are executed via `.github/workflows/ci.yml` on every pull request.
- **Statement coverage (OpenSSF Silver `test_statement_coverage80`):** Coverage is
  measured on two scopes, both gated in CI:
  - *Library scope* — the importable library surface (`src`/`functions` modules
    with no `__main__` CLI entry point), measured via `.coveragerc.library` and
    gated at `fail_under=80`. Currently ≈83% statement / ≈81% branch-inclusive.
    The omit list is generated mechanically from the presence of a `__main__`
    guard (`tools/check_library_coverage.py --check` enforces it stays in sync),
    so the scope is objective rather than hand-picked.
  - *Whole-repo floor* — a regression floor across the entire tree
    (`pyproject.toml`), currently ≈30% statement. Executable research/pipeline
    scripts (those with `__main__`) are validated by integration tests and the CI
    data/benchmark gates rather than by unit statement coverage.
- **Strict Data Typing:** Pydantic is utilized via `SestravConfig` to enforce configuration schema, preventing runtime type coercion errors and hardcoded path injection.

## 5. Security
- **Secure Architecture:** 
  - Avoidance of `eval()`/`exec()` and unsafe `shell=True` subprocesses.
  - Transitioned from unsafe file loading (`json.loads(open(path).read())`) to context-managed explicit IO logic.
  - Safe model unpickling via `ModelRegistry` validating expected features.
- **Dependency Management:** Dependencies are rigorously pinned with `--require-hashes` using `pip-compile` to prevent supply chain injection.
- **Static Analysis (SAST):** CodeQL, Bandit, and Semgrep are integrated into GitHub actions.

## 6. Analysis
- **Dynamic Analysis / Fuzzing:** Hypothesis property-based fuzz testing is integrated in CI via `.github/workflows/fuzzing.yml`. Tests in `tests/test_fuzz.py` exercise `compute_features` and `get_tcr_positions` under adversarial and edge-case amino acid inputs. Standard runs use 200 examples per push; weekly scheduled runs use 1000 examples. The Hypothesis database is persisted as an artifact to retain failure examples across runs.
- **Pipeline Data Integrity:** `freeze_mode` constraints enforce data immutability during reproducibility benchmarking — this is a correctness guard, not a security control.
- **Hygiene:** 0 known vulnerabilities exist in the core python codebase (verified by Bandit). Dependency-review Action blocks PRs introducing new vulnerable packages.

## Future Upgrades
Automated dependency updates are in place via Dependabot (`.github/dependabot.yml`)
alongside the Dependency-review Action and OSSF Scorecard. Remaining planned work:
publish the package to PyPI with a signed release attestation and cryptographically
sign release tags/artifacts (the outstanding OpenSSF Silver `signed_releases`
criterion). See `ROADMAP.md` for the test-coverage ratchet toward the Silver
`test_statement_coverage80` target.
