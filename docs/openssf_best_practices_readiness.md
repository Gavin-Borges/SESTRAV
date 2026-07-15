# SESTRAV OpenSSF Best Practices Badge Readiness Assessment (Passing Level)

This document provides a comprehensive readiness checklist and evidence mapping for registering SESTRAV at the **Passing Level** on the [OpenSSF Best Practices Badge Program](https://bestpractices.coreinfrastructure.org/) (formerly CII Best Practices). 

---

## 1. Basics

### 1.1 FLOSS (Free/Libre and Open Source Software)
*   **Requirement:** The project MUST be released as Free/Libre and Open Source Software (FLOSS).
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** The project is released under the permissive MIT License. The full license text is located in [LICENSE](../LICENSE) in the root directory.

### 1.2 OSI-Approved License
*   **Requirement:** The project's license MUST be approved by the Open Source Initiative (OSI).
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** The MIT License is OSI-approved. The license file begins with a machine-readable SPDX identifier (`SPDX-License-Identifier: MIT`) as required by standard automated compliance scanners (remediated in v2.0.0-rc1).

### 1.3 Documentation (Description, Installation, Usage)
*   **Requirement:** The project website/repo MUST describe what the software does, how to install it, and how to execute basic commands.
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** [README.md](../README.md) details the biological context, pipeline architecture, release tracks, command-line usage for training/reproduction, and quick-start instructions for virtual environments and containers.

### 1.4 Contribution Guidelines
*   **Requirement:** The project MUST explain how to contribute code and submit pull requests.
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** [CONTRIBUTING.md](../CONTRIBUTING.md) details development environment setup (conda, requirements), code style formatting rules (`ruff`), local test suite execution, containerized testing configurations (Docker and Singularity), and the mandatory PR checklist.

### 1.5 Public Discussion Forum
*   **Requirement:** The project MUST support a public forum or mailing list for discussion.
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** GitHub Issues and Pull Requests serve as the primary public forum for technical discussion and user feedback.
*   *Recommendation:* Enable **GitHub Discussions** in the repository settings to separate user Q&A from bug tracking.

---

## 2. Change Control

### 2.1 Version Control
*   **Requirement:** The project MUST use a version control system (VCS) to track all changes.
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** All code is tracked publicly via Git in the [SESTRAV GitHub Repository](https://github.com/Gavin-Borges/SESTRAV).

### 2.2 Version Numbering
*   **Requirement:** The project MUST use unique, sequentially increasing version numbers.
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** Follows Semantic Versioning (v2.0.0, v2.0.0-rc1) configured in [CITATION.cff](../CITATION.cff) and documented in the changelog.

### 2.3 Changelog / Release Notes
*   **Requirement:** The project MUST document changes in release notes or a changelog.
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** [CHANGELOG.md](../CHANGELOG.md) maps all major additions, changes, bug fixes, and security patches following the Keep a Changelog convention.

---

## 3. Reporting & Vulnerabilities

### 3.1 Security Policy & Vulnerability Reporting Channel
*   **Requirement:** The project MUST document how to report security vulnerabilities privately.
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** [SECURITY.md](../SECURITY.md) explicitly warns against reporting vulnerabilities via public issues, directing reporters to:
    1.  A private, confidential email address (`gavinmborges1104@gmail.com`).
    2.  GitHub's built-in **Private Vulnerability Reporting** mechanism.

### 3.2 Response SLA
*   **Requirement:** The project MUST commit to responding to private vulnerability reports within a specified timeframe.
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** [SECURITY.md](../SECURITY.md) commits to:
    *   Acknowledge receipt of vulnerability reports within **3-5 business days**.
    *   Provide an initial assessment (severity, scope) within **10 business days** of acknowledgement.

---

## 4. Quality & Testing

### 4.1 Working Build System
*   **Requirement:** The project MUST provide a reliable way to build/install the software on standard systems.
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** Build and dependency configuration is handled via:
    *   [environment.yml](../environment.yml) (Conda environment)
    *   [requirements.txt](../requirements.txt) (pip compiled hashes)
    *   [pyproject.toml](../pyproject.toml) (standard Python project package layout)
    *   [Dockerfile](../Dockerfile) and [singularity.def](../singularity.def) (container isolation)

### 4.2 Automated Test Suite
*   **Requirement:** The project MUST possess an automated test suite.
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** Features 1214 unit and integration tests (verified via `pytest --collect-only`) under the [tests/](../tests) folder, covering feature extraction, pipeline stages, consensus ensemble scoring, API schema validations, dataset build pipeline, and security hardening paths.

### 4.3 Automated Testing in CI
*   **Requirement:** The test suite MUST run automatically on new commits (CI).
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** The GitHub Actions workflow [.github/workflows/ci.yml](../.github/workflows/ci.yml) executes the `pytest` test suite, validates Snakemake wiring, runs a dataset curation QC gate, and renders reports on all pushes and pull requests targeting the `main` branch.

### 4.4 New Functionality Test Policy
*   **Requirement:** The project MUST enforce a policy requiring new code contributions to be covered by tests.
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** Documented in [CONTRIBUTING.md](../CONTRIBUTING.md) under "Development Guidelines" and enforced in the pull request template checklist.

### 4.5 Addressing Warning Flags
*   **Requirement:** Warnings generated by compiling or running the project MUST be addressed or documented.
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** [pytest.ini](../pytest.ini) handles deprecation and system warnings during test runs to ensure a clean execution output. Upstream deprecation warnings from external packages (e.g. `torch_geometric` or `torch.jit`) are documented in the release notes.

---

## 5. Security Practices

### 5.1 Secure Design Principles
*   **Requirement:** The project MUST be designed with security and data privacy in mind (e.g., least privilege, private-by-default).
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** 
    *   **Data Privacy by Design:** Documented in [SECURITY.md](../SECURITY.md). SESTRAV operates entirely offline on the host machine; it does not collect, log, or transmit sequences, user queries, or outputs.
    *   **Secure Network Binds:** The microservice and Streamlit frontend are hard-bound to `127.0.0.1` (loopback only) in [docker-compose.yml](../docker-compose.yml) to prevent accidental network exposure on shared university machines.

### 5.2 Cryptographic Best Practices
*   **Requirement:** Cryptographic operations MUST use secure algorithms (e.g., PBKDF2/bcrypt, SHA-256/SHA-512) and avoid weak algorithms (e.g., MD5/SHA-1) for security-critical checks.
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** 
    *   SESTRAV does not store user credentials or perform custom payload encryption.
    *   For data integrity, [release_bundle.py](../src/release_bundle.py) implements standard **SHA-256** checksum generation for validating release archives.
    *   WL graph subtree calculations utilize MD5 strictly for non-cryptographic feature index mapping (with `usedforsecurity=False` flagged in `hashlib` to prevent warning triggers).

### 5.3 Pinned Dependencies & Secure Supply Chain
*   **Requirement:** Third-party dependencies MUST be pinned and audited for known vulnerabilities.
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:**
    *   **Lockfiles with Hashes:** Dependency lists are compiled and locked with SHA-256 verification hashes using `pip-compile` inside [requirements.txt](../requirements.txt) to prevent dependency hijacking.
    *   **Weekly Dependency Scans:** Automated dependency review workflows are active ([dependency-review.yml](../.github/workflows/dependency-review.yml)) to block vulnerable imports on new PRs.

---

## 6. Security Analysis (SAST, DAST, Fuzzing)

### 6.1 Static Application Security Testing (SAST)
*   **Requirement:** The project MUST run static analysis tools (linters/security scanners) on code changes.
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** The [.github/workflows/security.yml](../.github/workflows/security.yml) workflow executes three distinct SAST scanners:
    1.  **Bandit:** Scans Python source files recursively for common security flaws (shell injection, path validation, unsecure serializers).
    2.  **CodeQL:** Performs deep semantic analysis on code changes via GitHub's integrated security engine.
    3.  **Semgrep:** Performs semantic structural scanning using Python security rulesets.

### 6.2 Dynamic Analysis & Property-Based Fuzzing
*   **Requirement:** The project MUST use dynamic analysis tools or techniques (e.g. memory sanitizers, input fuzzing) to identify runtime edge cases.
*   **SESTRAV Status:** ✅ **PASSING**
*   **Evidence:** Property-based input fuzzing is integrated using the Hypothesis framework inside [tests/test_fuzz.py](../tests/test_fuzz.py) and executed automatically in the CI loop [.github/workflows/fuzzing.yml](../.github/workflows/fuzzing.yml). The suite validates:
    *   **Sequence length boundaries:** `get_tcr_positions` is fuzzed across the full integer range to ensure no crashes on edge-case peptide lengths.
    *   **Non-standard amino acid characters:** `compute_features` is fuzzed with arbitrary Unicode text and boundary floating-point binding scores to verify structural integrity of the feature output dictionary.

---

## 7. Registration Protocol - ✅ Complete

SESTRAV is registered at the OpenSSF Best Practices Badge Program as
**[project 13191](https://www.bestpractices.dev/projects/13191)** and has attained
the **Passing** level. The questionnaire was answered using the justifications and
file links mapped in this document.

The live badge is embedded at the top of `README.md`:
```markdown
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/13191/badge)](https://www.bestpractices.dev/projects/13191)
```

Higher tiers are tracked in `ROADMAP.md`: Silver is essentially met (signed
releases ship via the `release.yml` Sigstore attestation workflow), with the
remaining Silver/Gold gaps being the multi-person criteria (`bus_factor`,
`two_person_review`, `contributors_unassociated`) that require a second
maintainer/contributor.
