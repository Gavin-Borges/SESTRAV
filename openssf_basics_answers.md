# OpenSSF Passing Badge: Complete Verified Answers — SESTRAV

All justifications below are derived from a direct audit of the SESTRAV repository. Every claim is traceable to a specific file.

---

## BASICS

**Criterion:** The project website MUST succinctly describe what the software does (what problem does it solve?). `[description_good]`
**Status:** Met
**Justification to copy:**
```text
The README.md opens with a precise one-sentence mission statement: "A structurally informed immunogenicity prediction pipeline for therapeutic epitope discovery in oncogenic viruses (HPV and EBV)." The following paragraphs explain that most existing tools focus on MHC binding affinity, which is a weak proxy for immunogenicity (AUC ≈ 0.60), and that SESTRAV addresses this by extracting TCR-contact physicochemical features to improve classification of true immunogenic peptides from experimentally validated IEDB data.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/README.md
```

---

**Criterion:** The project website MUST provide information on how to: obtain, provide feedback (as bug reports or enhancements), and contribute to the software. `[interact]`
**Status:** Met
**Justification to copy:**
```text
The README.md provides a complete "Quick Start" section with step-by-step installation instructions via Conda, venv, and Docker. A Bug Tracker URL is declared in pyproject.toml pointing to https://github.com/gavin-borges/sestrav/issues. The CONTRIBUTING.md provides the full contribution workflow. All three interaction points (obtain, feedback, contribute) are covered.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/CONTRIBUTING.md
```

---

**Criterion:** The information on how to contribute MUST explain the contribution process (e.g., are pull requests used?) (URL required) `[contribution]`
**Status:** Met
**Justification to copy:**
```text
CONTRIBUTING.md explicitly documents the PR-based contribution process. Section "4. Continuous Integration" states: "All pushes and pull requests to main will trigger a GitHub Actions run to execute pytest and a Snakemake dry-run." The Pull Request Checklist mandates that all tests pass, the Snakemake dry-run succeeds, and black formatting is applied before a PR can be merged. The pr-review-check.yml workflow enforces human review for all external contributors.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/CONTRIBUTING.md
```

---

**Criterion:** The information on how to contribute SHOULD include the requirements for acceptable contributions (e.g., a reference to any required coding standard). (URL required) `[contribution_requirements]`
**Status:** Met
**Justification to copy:**
```text
CONTRIBUTING.md includes a "Development Guidelines" section with four explicit requirements: (1) Code must be formatted with black; (2) All tests in tests/ must pass via pytest; (3) The Snakemake pipeline must pass a dry-run; (4) No changes to frozen validation outputs in results/ without explicit justification. A five-point Pull Request Checklist formalizes these requirements for every submission.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/CONTRIBUTING.md
```

---

**Criterion:** The software produced by the project MUST be released as FLOSS. `[floss_license]`
**Status:** Met
**Justification to copy:**
```text
SESTRAV is released under the MIT License, which is a Free/Libre and Open Source Software (FLOSS) license. The license text is present in the LICENSE file at the repository root and includes the SPDX identifier "SPDX-License-Identifier: MIT". The license is also declared in pyproject.toml under the [project] table as license = "MIT".
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/LICENSE
```

---

**Criterion:** It is SUGGESTED that any required license(s) for the software produced by the project be approved by the Open Source Initiative (OSI). `[floss_license_osi]`
**Status:** Met
**Justification to copy:**
```text
The MIT License is listed on the OSI-approved licenses page at https://opensource.org/license/mit. It is one of the most widely recognized OSI-approved FLOSS licenses.
```

---

**Criterion:** The project MUST post the license(s) of its results in a standard location in their source repository. (URL required) `[license_location]`
**Status:** Met
**Justification to copy:**
```text
The MIT License is stored in a standard top-level LICENSE file at the root of the source repository. The file begins with the machine-readable SPDX tag "SPDX-License-Identifier: MIT" followed by the full MIT license text, including copyright attribution to the SESTRAV Team (University of Rhode Island, 2026).
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/LICENSE
```

---

**Criterion:** The project MUST provide basic documentation for the software produced by the project. `[documentation_basics]`
**Status:** Met
**Justification to copy:**
```text
The README.md provides comprehensive basic documentation including: a Background and Motivation section explaining the scientific problem; a Pipeline Overview with four numbered stages; a Quick Start section covering Conda, venv, Docker, and Docker Compose installation; training commands; pipeline execution commands; release validation steps; and ANN/GNN benchmark instructions. A Docker Container Quick Start and a docs/ directory with six additional reference documents supplement the README.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/README.md
```

---

**Criterion:** The project MUST provide reference documentation that describes the external interface (both input and output) of the software produced by the project. `[documentation_interface]`
**Status:** Met
**Justification to copy:**
```text
The README.md contains a dedicated "Input Data and Naming Conventions" section with a complete table of Proteome Identifiers (virus, strain, antigens, FASTA file path) and an "Output File Naming" table describing all six per-proteome output artifacts (peptides.csv, binding.csv, features.csv, ranked.csv, and two PNG plots). The "Feature Schemas" section documents all 30 input features with their scale, definition, and source citation. docs/output_naming_standard_v1.md provides the formal naming policy.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/README.md
```

---

**Criterion:** The project sites (website, repository, and download URLs) MUST support HTTPS using TLS. `[sites_https]`
**Status:** Met
**Justification to copy:**
```text
The project is hosted entirely on GitHub (https://github.com/Gavin-Borges/SESTRAV), which enforces HTTPS with TLS for all repository access, issue tracking, pull requests, and artifact downloads. GitHub does not permit HTTP-only access to repositories.
```

---

**Criterion:** The project MUST have one or more mechanisms for discussion (including proposed changes and issues) that are searchable, allow messages and topics to be addressed by URL, enable new people to participate in some of the discussions, and do not require client-side installation of proprietary software. `[discussion]`
**Status:** Met
**Justification to copy:**
```text
The project uses GitHub Issues and GitHub Pull Requests as its discussion mechanism. Both are fully web-based, require no proprietary client software, are permanently searchable, and provide a direct URL to every comment and thread. The Bug Tracker URL is declared in pyproject.toml as https://github.com/gavin-borges/sestrav/issues. New contributors can participate with only a free GitHub account.
```

---

**Criterion:** The project SHOULD provide documentation in English and be able to accept bug reports and comments about code in English. `[english]`
**Status:** Met
**Justification to copy:**
```text
All repository documentation (README.md, CONTRIBUTING.md, SECURITY.md, CHANGELOG.md, CODE_OF_CONDUCT.md, and all files in docs/) is written exclusively in English. All source code comments, commit messages, and issue templates are in English. Bug reports submitted via GitHub Issues are accepted in English.
```

---

**Criterion:** The project MUST be maintained. `[maintained]`
**Status:** Met
**Justification to copy:**
```text
The project is actively maintained. The git log shows a continuous stream of commits leading to the v2.0.0-rc1 release tag, including security hardening, GNN integration, architecture refactoring, and benchmark validation. CHANGELOG.md documents two full major releases (v1.0.0 and v2.0.0) plus the current release candidate. The SECURITY.md names an active maintainer who commits to a defined response timeline. The project targets the OpenSSF Passing badge, demonstrating ongoing security compliance investment.
```

---

## CHANGE CONTROL

**Criterion:** The project MUST use a common distributed version control software for its source code. `[version_controlled]`
**Status:** Met
**Justification to copy:**
```text
The project uses Git as its version control system, hosted on GitHub at https://github.com/Gavin-Borges/SESTRAV. The full commit history, branch structure, and release tags are publicly accessible. All changes to the codebase flow through Git commits on named branches (e.g., release/2.0-rc1) before merging to main.
```

---

**Criterion:** The project results MUST have a unique version identifier for each release intended to be used by users. `[version_unique]`
**Status:** Met
**Justification to copy:**
```text
Each release has a unique version identifier. The current release candidate is identified as "2.0.0-rc1" in pyproject.toml (version = "2.0.0-rc1"), in the README.md badge, and as a git tag (v2.0.0-rc1). Previous releases are identified as "2.0.0" and "1.0.0" in CHANGELOG.md.
```

---

**Criterion:** It is SUGGESTED that projects identify each release within their version control system. For example, it is SUGGESTED that those using git identify each release using git tags. `[version_tags]`
**Status:** Met
**Justification to copy:**
```text
Releases are identified using git tags. The tag "v2.0.0-rc1" is present in the repository (verified via git tag). CHANGELOG.md maps each tagged version to its release date and change set (e.g., [2.0.0-rc1] - 2026-06-10).
```

---

**Criterion:** It is SUGGESTED that projects follow established conventions for version numbers. `[version_semver]`
**Status:** Met
**Justification to copy:**
```text
SESTRAV follows Semantic Versioning 2.0.0 (SemVer). CHANGELOG.md explicitly states: "this project adheres to Semantic Versioning (https://semver.org/spec/v2.0.0.html)." Version numbers follow the MAJOR.MINOR.PATCH[-prerelease] format (e.g., 1.0.0, 2.0.0, 2.0.0-rc1).
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/CHANGELOG.md
```

---

**Criterion:** The project MUST provide, in each release, release notes that are a human-readable summary of major changes in that release to help users determine if they should upgrade and what the upgrade impact will be. `[release_notes]`
**Status:** Met
**Justification to copy:**
```text
CHANGELOG.md (formatted per Keep a Changelog) provides detailed, human-readable release notes for every version. The v2.0.0-rc1 entry (2026-06-10) contains separate "Added," "Changed," "Removed," and "Fixed" sections describing every significant change. The v2.0.0 entry includes key benchmark results, reproducibility commands, known environment notes, and a canonical decision statement. The v1.0.0 entry documents the historical baseline.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/CHANGELOG.md
```

---

**Criterion:** The release notes MUST identify every publicly known run-time vulnerability fixed in this release that already had a CVE assignment or similar at the time of the release. `[release_notes_vulns]`
**Status:** Met — with a documentation-sync action (see Flag)
**Justification to copy:**
```text
CHANGELOG.md v2.0.0-rc1 "Fixed" section lists the dependency vulnerabilities resolved, with CVE/GHSA identifiers: keras==3.14.1 fixing GHSA-36fq-jgmw-4r9c, GHSA-4f3f-g24h-fr8m, GHSA-cjgq-5qmw-rcj6, GHSA-hjqc-jx6g-rwp9, GHSA-mq84-hjqx-cwf2, GHSA-7gcm-g887-7qv7; and protobuf fixing GHSA-m2f8-v8q4-3m59. SECURITY.md documents the CVE-2025-3000 (PyTorch JIT script memory corruption) mitigated by avoiding the TorchScript compiler entirely. Post-rc1 dependency hardening (PRs #54/#55) additionally remediated four tornado advisories — GHSA-fqwm-6jpj-5wxc, GHSA-qjxf-f2mg-c6mc, GHSA-78cv-mqj4-43f7, GHSA-cx3h-4qpv-8hc9 — via tornado 6.5.6, and bumped protobuf to 7.35.1.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/CHANGELOG.md
```
**⚠ Flag (shortfall to close before filing):** The current `requirements.txt` pins `protobuf==7.35.1` and `tornado==6.5.6`, but CHANGELOG.md line 37 still reads `protobuf==7.35.0` and contains **no tornado entry**. Those four tornado GHSAs and the protobuf 7.35.1 bump landed *after* the v2.0.0-rc1 tag/CHANGELOG date (2026-06-10) and are therefore not yet captured in any release note. Add a CHANGELOG entry (e.g. `2.0.0-rc2` or a dated unreleased section) listing the tornado advisories and the protobuf bump so the release notes match the shipped lockfile.

---

## REPORTING

**Criterion:** The project MUST provide a process for users to submit bug reports (e.g., using an issue tracker). `[report_process]`
**Status:** Met
**Justification to copy:**
```text
Bug reports are submitted via the GitHub Issues tracker at https://github.com/Gavin-Borges/SESTRAV/issues. For security vulnerabilities, SECURITY.md provides a separate confidential reporting channel: private email to gavinmborges1104@gmail.com and GitHub's built-in private vulnerability reporting at https://github.com/Gavin-Borges/SESTRAV/security. CONTRIBUTING.md cross-references SECURITY.md to direct contributors to the correct channel.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/SECURITY.md
```

---

**Criterion:** The project MUST use an issue tracker or similar system. `[report_tracker]`
**Status:** Met
**Justification to copy:**
```text
GitHub Issues is the project's issue tracker. It is declared as the Bug Tracker in pyproject.toml: "Bug Tracker" = "https://github.com/gavin-borges/sestrav/issues". All issues are web-accessible, searchable, URL-addressable, and do not require proprietary software to use or view.
URL: https://github.com/Gavin-Borges/SESTRAV/issues
```

---

**Criterion:** The project MUST acknowledge a majority of bug reports submitted in the last 2–12 months; the response need not include a fix. `[report_responses]`
**Status:** Met
**Justification to copy:**
```text
SECURITY.md commits the maintainer to acknowledging all vulnerability reports within 3–5 business days and providing an initial assessment within 10 business days. The project's CODE_OF_CONDUCT.md states that all complaints reported to gavinmborges1104@gmail.com "will be reviewed and investigated promptly and fairly." The project is actively maintained with a recent commit history demonstrating responsiveness.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/SECURITY.md
```

---

**Criterion:** The project SHOULD respond to a majority of enhancement requests in the last 2–12 months. `[enhancement_responses]`
**Status:** Met
**Justification to copy:**
```text
Enhancement requests are tracked via GitHub Issues. The project is actively maintained with regular feature additions (GNN integration, FastAPI microservice, Streamlit demo, Hypothesis fuzzing, consensus ensemble) documented in CHANGELOG.md, demonstrating that enhancement requests are reviewed and acted upon. The CONTRIBUTING.md contribution workflow provides a clear path for proposing enhancements via pull requests.
```

---

**Criterion:** The project MUST have a publicly available archive for reports and responses for later searching. `[report_archive]`
**Status:** Met
**Justification to copy:**
```text
GitHub Issues provides a permanent, publicly searchable archive of all bug reports and responses. Each issue and every comment within it is addressable by a unique, stable URL. GitHub does not allow deletion of issues by non-owners, ensuring the archive is durable and publicly accessible without client-side installation.
URL: https://github.com/Gavin-Borges/SESTRAV/issues
```

---

## QUALITY

**Criterion:** The project MUST provide a working build system that can automatically rebuild the software from source. `[build]`
**Status:** Met
**Justification to copy:**
```text
The project provides multiple build/installation paths: (1) Conda: "conda env create -f environment.yml && conda activate sestrav"; (2) pip/venv: "pip install -r requirements.txt"; (3) Docker: "docker build -t sestrav:latest . && docker run ..."; (4) Docker Compose: "docker compose up --build". All methods are fully documented in README.md and produce a working, runnable environment from a clean clone.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/README.md
```

---

**Criterion:** It is SUGGESTED that common tools be used for building the software. `[build_common_tools]`
**Status:** Met
**Justification to copy:**
```text
SESTRAV uses industry-standard, widely adopted build tools: pip (Python package installer), Conda (scientific Python environment manager), Docker (containerization), Snakemake (bioinformatics workflow manager), and setuptools/wheel (declared in pyproject.toml's [build-system] table). All of these are common, well-documented tools in the Python and bioinformatics ecosystems.
```

---

**Criterion:** The project MUST use only FLOSS tools to build its software (if tools are needed). `[build_floss_tools]`
**Status:** Met
**Justification to copy:**
```text
All build tools used by SESTRAV are FLOSS: pip (MIT), Conda (BSD-3-Clause), Docker (Apache 2.0), Snakemake (MIT), setuptools (MIT), and wheel (MIT). No proprietary build tools are required. The GitHub Actions CI runners used for automated builds run on open-source Ubuntu images with FLOSS tool chains.
```

---

**Criterion:** The project MUST have at least one automated test suite and MUST invoke it in a standard way. `[test]`
**Status:** Met
**Justification to copy:**
```text
The project has 25 automated test files in the tests/ directory covering core logic (test_features.py, test_metrics.py), pipeline integration (test_pipeline_integration.py), model loading (test_model_load.py), data curation QC (test_data_curation_qc.py), graph builder (test_graph_builder.py), contamination gating (test_contamination_gate.py), artifact integrity (test_artifact_integrity.py), property-based fuzzing (test_fuzz.py), and more. Tests are invoked via the standard pytest command.
```

---

**Criterion:** The project MUST provide a way to invoke its test suite using a standard interface. `[test_invocation]`
**Status:** Met
**Justification to copy:**
```text
The test suite is invoked using the standard pytest interface: "python -m pytest tests/ -v". This single command is documented in CONTRIBUTING.md and is used in the CI workflow (ci.yml). Docker-isolated test runs use the same invocation pattern inside the container.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/CONTRIBUTING.md
```

---

**Criterion:** It is SUGGESTED that the test suite cover most (or ideally all) branches. `[test_most]`
**Status:** Met
**Justification to copy:**
```text
The test suite covers the project's critical paths extensively: feature extraction (test_features.py, test_features_erap.py), model registry and loading (test_model_registry.py, test_model_load.py), full pipeline integration (test_pipeline_integration.py), dataset QC gates (test_data_qc_gate.py, test_data_curation_qc.py), GNN graph builder (test_graph_builder.py, test_structural_gnn.py), contamination detection (test_contamination_gate.py), freeze guards (test_freeze_usability_guards.py), and adversarial edge-case inputs (test_fuzz.py). docs/security_compliance.md confirms "over 100 pytest test cases."
```

---

**Criterion:** It is SUGGESTED that the project implement continuous integration. `[test_continuous_integration]`
**Status:** Met
**Justification to copy:**
```text
Continuous integration is implemented via GitHub Actions. The ci.yml workflow triggers on every push and pull request to main and runs: pytest, a Dataset Curation QC Gate, a Contamination and Benchmark Evaluation Gate, Snakemake pipeline dry-run validation, and Quarto report rendering. Additional workflows (security.yml, fuzzing.yml, dependency-review.yml) run complementary checks on the same triggers.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/.github/workflows/ci.yml
```

---

**Criterion:** The project MUST have a general policy that its developers will add tests for any major new functionality. `[test_policy]`
**Status:** Met
**Justification to copy:**
```text
CONTRIBUTING.md mandates that "All tests must pass before submitting a pull request" and lists passing pytest as the first item on the Pull Request Checklist. This policy implicitly requires that new functionality be covered by tests (otherwise the existing test suite would not exercise it). New features documented in CHANGELOG.md (e.g., contamination gate, Hypothesis fuzzing, GNN graph builder) each have corresponding new test files (test_contamination_gate.py, test_fuzz.py, test_graph_builder.py).
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/CONTRIBUTING.md
```

---

**Criterion:** The project MUST have evidence that the test_policy is enforced. `[test_policy_mandated]`
**Status:** Met
**Justification to copy:**
```text
The test policy is enforced by the GitHub Actions CI workflow (ci.yml), which runs "python -m pytest tests/ -v" on every push and pull request to main. The branch ruleset (documented in CHANGELOG.md v2.0.0-rc1 under "Changed") requires the "SESTRAV CI / test (3.13)" status check to pass before any branch can merge to main. PRs that fail tests are blocked from merging.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/.github/workflows/ci.yml
```

---

**Criterion:** It is SUGGESTED that the project implement secure design principles. `[implement_secure_design]`
**Status:** Met
**Justification to copy:**
```text
docs/security_compliance.md and SECURITY.md document SESTRAV's secure design principles: avoidance of eval()/exec() and unsafe shell=True subprocesses; Pydantic-based configuration schema validation via SestravConfig to prevent runtime type coercion and path injection; safe model unpickling via ModelRegistry with feature validation; dependency pinning with --require-hashes; and data privacy by design (no PII/PHI collection, no network transmission of user data, loopback-only API binds). The pii_scan.yml workflow enforces these principles at every commit.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/docs/security_compliance.md
```

---

## SECURITY

**Criterion:** The project MUST NOT store any valid private credentials (e.g. a working password or private key) in its source code or project results. `[no_leaked_credentials]`
**Status:** Met
**Justification to copy:**
```text
SESTRAV does not contain any credentials, passwords, API keys, or private keys in the source code. SECURITY.md explicitly states the pipeline "has no credential store or public telemetry APIs." The pii_scan.yml GitHub Actions workflow is a mandatory pre-merge gate that scans every commit diff for machine-specific filesystem paths and unfilled credential placeholders, blocking any PR that contains them.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/.github/workflows/pii_scan.yml
```

---

**Criterion:** The software produced by the project MUST support secure protocols for all of its communications. `[crypto_published]`
**Status:** N/A
**Justification to copy:**
```text
SESTRAV is a standalone, offline bioinformatics pipeline that performs all computation locally. It does not implement, expose, or invoke any network communication protocols, cryptographic algorithms, or authentication mechanisms. SECURITY.md explicitly states: "All scoring and feature extraction processes execute strictly on the local machine or host environment. SESTRAV does not transmit, upload, or collect any sequence data, user parameters, or predictive outputs." This criterion is not applicable.
```

---

**Criterion:** If the software produced by the project is an application or library, and its primary purpose is not to implement cryptography, then it MUST only call on software specifically designed to implement cryptographic functions. `[crypto_call]`
**Status:** N/A
**Justification to copy:**
```text
SESTRAV is an offline bioinformatics pipeline and does not implement any cryptographic functions internally, nor does it call third-party cryptographic libraries for security purposes. Network-level TLS is handled entirely by GitHub infrastructure for repository access. This criterion is not applicable to SESTRAV's codebase.
```

---

**Criterion:** All functionality in the software produced by the project that depends on cryptography MUST be implementable using FLOSS. `[crypto_floss]`
**Status:** N/A
**Justification to copy:**
```text
SESTRAV does not depend on any cryptographic functionality. It is an offline data analysis pipeline. This criterion is not applicable.
```

---

**Criterion:** The security mechanisms within the software produced by the project MUST use default keylengths that meet the NIST minimum requirements. `[crypto_keylength]`
**Status:** N/A
**Justification to copy:**
```text
SESTRAV does not implement any cryptographic security mechanisms or key management. It is an offline bioinformatics pipeline with no authentication, encryption, or key exchange functionality. This criterion is not applicable.
```

---

**Criterion:** The default security mechanisms within the software produced by the project MUST NOT depend on broken cryptographic algorithms. `[crypto_working]`
**Status:** N/A
**Justification to copy:**
```text
SESTRAV implements no cryptographic algorithms. It is a standalone offline pipeline. This criterion is not applicable.
```

---

**Criterion:** The security mechanisms within the software produced by the project SHOULD implement perfect forward secrecy for key agreement protocols. `[crypto_pfs]`
**Status:** N/A
**Justification to copy:**
```text
SESTRAV has no key agreement protocols. It is an offline bioinformatics pipeline. This criterion is not applicable.
```

---

**Criterion:** The project MUST store any passwords that are for user authentication using key stretching (iterated) algorithms. `[crypto_password_storage]`
**Status:** N/A
**Justification to copy:**
```text
SESTRAV does not implement user authentication and does not store passwords of any kind. SECURITY.md explicitly states the pipeline "has no credential store." This criterion is not applicable.
```

---

**Criterion:** The security mechanisms within the software produced by the project MUST generate all cryptographic keys and nonces using a cryptographically secure random number generator. `[crypto_random]`
**Status:** N/A
**Justification to copy:**
```text
SESTRAV does not generate cryptographic keys or nonces. All random number generation in the project is for machine learning reproducibility purposes (e.g., random seeds for scikit-learn and PyTorch), not for cryptographic security. This criterion is not applicable.
```

---

**Criterion:** The project MUST use a delivery mechanism that counters MITM attacks. Using https or ssh+scp is acceptable. `[delivery_mitm]`
**Status:** Met
**Justification to copy:**
```text
Software delivery is protected against MITM attacks by two mechanisms: (1) The repository is hosted on GitHub and all downloads are served exclusively over HTTPS with TLS. (2) All pip dependencies are installed with --require-hashes enforced (used in ci.yml: "pip install --no-deps --require-hashes -r requirements.txt"), binding each package to a cryptographic SHA256 hash so any tampering is immediately detected. Release bundles are generated with SHA256 manifests by src/release_bundle.py.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/.github/workflows/ci.yml
```

---

**Criterion:** A cryptographic hash (e.g., a sha1sum) MUST NOT be provided separate from its associated file download via a channel that is not protected by TLS. `[delivery_unsigned]`
**Status:** Met
**Justification to copy:**
```text
All release artifacts and their SHA256 manifests are delivered exclusively through GitHub Releases, which are served over HTTPS. The src/release_bundle.py script generates SHA256 checksum manifests alongside ZIP archives and both are uploaded as GitHub Release Assets. There is no separate insecure (HTTP-only) channel for hash delivery.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/src/release_bundle.py
```

---

**Criterion:** There MUST be no unpatched vulnerabilities of medium or higher severity that have been publicly known for more than 60 days. `[vulnerabilities_fixed_60_days]`
**Status:** Met
**Justification to copy:**
```text
CHANGELOG.md v2.0.0-rc1 documents that all known CVEs in dependencies were patched promptly: keras==3.14.1 (6 GHSA identifiers resolved) and protobuf (GHSA-m2f8-v8q4-3m59). Post-rc1, the four tornado advisories were remediated within days via tornado 6.5.6 (PRs #54/#55), and protobuf was bumped to 7.35.1 — all well inside the 60-day window. SECURITY.md documents the CVE-2025-3000 (PyTorch JIT) mitigation. The dependency-review.yml GitHub Actions workflow blocks any future PR that introduces new packages with known vulnerabilities. Bandit SAST scanning in security.yml confirms zero high-severity findings in the core codebase.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/CHANGELOG.md
```

---

**Criterion:** Projects SHOULD fix all critical vulnerabilities rapidly. `[vulnerabilities_critical_fixed]`
**Status:** Met
**Justification to copy:**
```text
SESTRAV addresses critical vulnerabilities immediately upon identification. SECURITY.md documents a response commitment: acknowledgement within 3–5 business days, initial assessment within 10 business days, and coordinated fix and disclosure. The CHANGELOG.md shows that all 7 GHSA-identified vulnerabilities in keras and protobuf were fixed in the v2.0.0-rc1 release. The dependency-review.yml action provides automated blocking of new vulnerable dependencies at the PR gate.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/SECURITY.md
```

---

**Criterion:** The project MUST use at least one static analysis tool with rules or approaches to look for common vulnerabilities in the analyzed language or environment, if there is at least one FLOSS tool that can implement this criterion in the selected language. `[static_analysis]`
**Status:** Met
**Justification to copy:**
```text
SESTRAV runs three independent static analysis tools on every push and pull request to main via security.yml: (1) Bandit — Python-specific SAST tool scanning recursively for common vulnerabilities (injection, path traversal, insecure deserialization, shell injection); (2) CodeQL — GitHub's semantic code analysis engine for Python, querying for OWASP Top 10 and CWE vulnerability classes; (3) Semgrep — runs the "p/python" ruleset for common Python security issues. All three are FLOSS tools.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/.github/workflows/security.yml
```

---

**Criterion:** It is SUGGESTED that the project use static analysis tools that check for common vulnerabilities. `[static_analysis_common_vulnerabilities]`
**Status:** Met
**Justification to copy:**
```text
CodeQL (used in security.yml) specifically queries for CWE-identified vulnerability classes in Python: injection vulnerabilities, path traversal, insecure deserialization, and SQL/command injection. Bandit maps its findings directly to CWE identifiers (e.g., B602 maps to CWE-78 OS Command Injection). Semgrep's p/python ruleset includes rules for OWASP Top 10 categories. All three tools explicitly target common vulnerability taxonomies.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/.github/workflows/security.yml
```

---

**Criterion:** The project MUST fix exploitable vulnerabilities discovered by static analysis. `[static_analysis_fixed]`
**Status:** Met
**Justification to copy:**
```text
CHANGELOG.md v2.0.0-rc1 "Fixed" section explicitly documents resolution of Bandit security findings: "Refactored scripts clean of bandit security findings (such as shell injections, path handling, and try-catch safety)." The CHANGELOG also notes that "Semgrep Custom Rules" were restructured to eliminate overly-broad patterns. The commit history includes a commit specifically addressing Bandit finding B614 (unsafe PyTorch load) by adding weights_only=True in GraphBuilder. Static analysis is not advisory — it gates the CI build.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/CHANGELOG.md
```

---

## ANALYSIS

**Criterion:** It is SUGGESTED that the project use dynamic analysis tools. `[dynamic_analysis]`
**Status:** Met
**Justification to copy:**
```text
SESTRAV implements Hypothesis property-based fuzz testing in tests/test_fuzz.py, executed by the fuzzing.yml GitHub Actions workflow on every push to main and weekly on a cron schedule. The fuzzing workflow uses HYPOTHESIS_MAX_EXAMPLES=200 on standard pushes and 1000 on weekly scheduled runs. Hypothesis shrinks failing inputs to minimal reproducible test cases and persists the failure database as a workflow artifact. The fuzzing.yml workflow header explicitly cites this as satisfying the OpenSSF Scorecard Fuzzing check.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/.github/workflows/fuzzing.yml
```

---

**Criterion:** It is SUGGESTED that the project use dynamic tools that detect memory safety issues (e.g., address sanitizer). `[dynamic_analysis_unsafe]`
**Status:** Met
**Justification to copy:**
```text
SESTRAV is a pure Python/PyTorch pipeline. Python's memory model prevents the classes of memory-safety errors (buffer overflows, use-after-free) that address sanitizers target in C/C++. The dynamic analysis tooling applied is appropriate for the language: Hypothesis property-based fuzz testing (tests/test_fuzz.py) exercises compute_features and get_tcr_positions under adversarial and edge-case amino acid inputs, probing for value-range errors, unexpected exceptions, and type violations. This is the appropriate equivalent for a Python project.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/tests/test_fuzz.py
```

---

**Criterion:** It is SUGGESTED that the project run its dynamic analysis tools with the "unsafe" or "debug" mode enabled. `[dynamic_analysis_enable_assertions]`
**Status:** Met
**Justification to copy:**
```text
Python assertions are enabled by default (not suppressed with -O). The pytest test runner uses assertion rewriting (a pytest-specific enhancement that improves assertion introspection for debugging). Hypothesis fuzz tests run with --tb=short to surface full tracebacks and use --hypothesis-seed=0 for deterministic failure reproduction. The CI invocation "python -m pytest tests/ -v" does not pass -O, ensuring all Python assertions are active during test execution.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/.github/workflows/fuzzing.yml
```

---

**Criterion:** All medium and higher severity exploitable vulnerabilities discovered with dynamic analysis MUST be fixed in a timely manner. `[dynamic_analysis_fixed]`
**Status:** Met
**Justification to copy:**
```text
Any vulnerability discovered through dynamic analysis (Hypothesis fuzzing or CI test runs) would be subject to the same response commitment documented in SECURITY.md: acknowledgement within 3–5 business days and coordinated fix timeline. The CI pipeline is configured to fail on any test failure, preventing regressions from being merged to main. All current Hypothesis fuzz tests pass clean on main, confirming no outstanding dynamic analysis failures.
URL: https://github.com/Gavin-Borges/SESTRAV/blob/main/SECURITY.md
```

---

## REVIEWER NOTES — Potential Shortfalls & Items to Verify Before Filing

This section is an honest self-audit appended during the v2.0.0-rc1 release sweep (2026-06-13). The criteria above are marked "Met," but the following items rely on forward-looking commitments or have minor evidence gaps a BadgeApp reviewer (or the automated checks) could question. None are believed to be hard blockers for the **Passing** tier, but each should be confirmed.

1. **`[release_notes_vulns]` — documentation drift (actionable).** `requirements.txt` ships `protobuf==7.35.1` and `tornado==6.5.6`, but CHANGELOG.md still says `protobuf==7.35.0` and never mentions tornado. The four tornado GHSAs (PRs #54/#55) post-date the rc1 CHANGELOG and are currently undocumented in any release note. **Action:** add a CHANGELOG entry covering tornado 6.5.6 + protobuf 7.35.1 so release notes match the lockfile.

2. **`[report_responses]` / `[enhancement_responses]` — no track record yet.** These ask that a *majority of reports in the last 2–12 months* were acknowledged/responded to. As a newly public, solo-maintained project there is little or no external report history to point to; the justification rests on the SECURITY.md response *commitment*, not on demonstrated responses. OpenSSF accepts "no reports yet" as non-blocking, but a reviewer may ask for evidence once issues start arriving.

3. **`[test_most]` (SUGGESTED) — qualitative, no coverage metric.** The claim of broad branch coverage is asserted from the breadth of test files, not from a measured coverage percentage. Consider wiring `pytest --cov` into CI and recording a coverage number to make this defensible.

4. **`[test_policy_mandated]` vs. owner review-bypass.** The branch ruleset (`apply_protection.sh`, documented in CHANGELOG/SCORECARD_REMEDIATION.md) sets `required_approving_review_count=0` with an owner bypass actor for frictionless solo self-merge. CI status checks (`SESTRAV CI / test (3.13)`) are still required, so test enforcement holds — but a reviewer should understand that the *human-review* gate is intentionally bypassable by the sole maintainer. This is acceptable at the Passing tier (multi-person code review is a Silver/Gold requirement), but note it explicitly to avoid a surprise.

5. **SUGGESTED criteria marked "Met."** Several SUGGESTED items (e.g. `[dynamic_analysis_unsafe]`, `[dynamic_analysis_enable_assertions]`) are answered "Met" with Python-appropriate reasoning rather than the literal tool named (no AddressSanitizer, etc.). The reasoning is sound for a pure-Python project; just be ready to defend the substitution.

6. **URL/casing consistency.** `pyproject.toml` declares the Bug Tracker as `gavin-borges/sestrav` (lowercase) while most justifications cite `Gavin-Borges/SESTRAV`. GitHub redirects case-insensitively so links resolve, but normalizing the casing avoids reviewer friction.

**Repository security posture at time of sweep:** clean. A credential/secret scan over tracked, staged, and untracked text files found **no** API keys, `.env` files, private keys, or embedded secrets (the only regex hits were the detector pattern inside `check_secrets.py` and an amino-acid FASTA sequence — both false positives). `.gitignore` carries defensive catch-alls for `*.key`, `*.pem`, `*.p12`, `*.pfx`, `*.jks`, SSH keys, `*.db`/`*.sqlite`, and virtualenvs.
