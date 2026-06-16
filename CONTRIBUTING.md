# Contributing to SESTRAV

Thank you for your interest in contributing to SESTRAV! This document provides instructions for setting up the development environment, running tests, submitting pull requests, and following our repository guidelines.

## Development Environment Setup

We recommend using Conda to manage environment dependencies.

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Gavin-Borges/SESTRAV.git
   cd SESTRAV
   ```

2. **Create the Conda Environment:**
   ```bash
   conda env create -f environment.yml
   conda activate sestrav
   ```

3. **Download MHCflurry Presentation Models:**
   ```bash
   mhcflurry-downloads fetch models_class1_presentation
   ```

4. **Train Canonical Local Models:**
   Models must be trained before the production pipeline can execute.
   ```bash
   python -m src.train_classifier --data data/immunogenicity_dataset_v3.csv --feature-mode 30 --binding-matrix models/peptide_binding_matrix_v3.csv
   ```

## Development Guidelines

### 1. Code Style
The project follows **PEP 8**, enforced automatically with [Ruff](https://docs.astral.sh/ruff/).
The exact ruleset and any documented exceptions live in `pyproject.toml`
(`[tool.ruff]`). CI runs `ruff check .` as a required gate, so please lint and
format before submitting:
```bash
ruff check . --fix     # lint (and auto-fix what is safely fixable)
ruff format .          # format (black-compatible)
```

### 2. Running Tests
Tests are located in the `tests/` directory. All tests must pass before submitting a pull request:
```bash
python -m pytest tests/ -v
```

To measure whole-repo coverage locally (statement + branch):
```bash
python -m pytest tests/ --cov=src --cov=functions --cov-branch --cov-report=term-missing
```

CI gates the **library** coverage scope used for the OpenSSF Silver
`test_statement_coverage80` criterion (the importable `src`/`functions` modules,
excluding executable scripts that carry a `__main__` entry point). To reproduce
that gate locally:
```bash
# fails if the omit list has drifted from the source tree
python tools/check_library_coverage.py --check
python -m pytest tests/ --cov=src --cov=functions \
    --cov-config=.coveragerc.library --cov-report=term-missing
```
Executable research/pipeline scripts are validated by the integration tests and
CI data/benchmark gates rather than by unit statement coverage. When adding a
new script, ensure it has a `__main__` guard (so the scope check classifies it
correctly) or add unit tests if it is importable library code.

New functionality MUST be accompanied by tests, and bug fixes SHOULD add a
regression test.

#### Container-Isolated Testing (Docker)
To verify that changes run successfully in a clean, container-isolated environment, you can run the test suite within Docker:
```bash
# Build the test-ready container
docker build -t sestrav:test .

# Run the pytest suite inside the container
docker run --rm -v "$(pwd)/data:/app/data:ro" sestrav:test -m pytest tests/ -q --basetemp=tmp_pytest
```

#### Container-Isolated Testing (Singularity / Apptainer)
For High-Performance Computing (HPC) environments where Docker is not available:
```bash
# Build the Singularity image file (SIF)
singularity build sestrav.sif singularity.def

# Run the test suite within the Singularity container
singularity exec sestrav.sif pytest tests/ -q
```


### 3. Snakemake Validation
Validate the Snakemake pipeline structure with a dry-run:
```bash
snakemake --snakefile pipeline.smk --dry-run --cores 1
```

For full validation report generation (including H2 and gold-standard checks):
```bash
snakemake --snakefile pipeline.smk full_validation_report --cores 4 --forceall
```
Ensure that `results/freeze_status.json` has `"valid": true` before releasing any changes.

### 4. Continuous Integration
All pushes and pull requests to `main` will trigger a GitHub Actions run to execute `pytest` and a Snakemake dry-run.

---

## Pull Request Checklist

When submitting a pull request, ensure the following checklist is completed:

- [ ] All tests pass locally using `pytest`.
- [ ] Snakemake dry-run succeeds.
- [ ] Coding style follows PEP 8 / formatting via `black` is applied.
- [ ] No changes are made to frozen validation outputs in `results/` unless explicitly requested.
- [ ] `freeze_mode: true` is enabled in `config.yaml` for release-grade validation runs.

---

## Security and Vulnerability Reporting

If you discover a security vulnerability or critical compliance issue within SESTRAV, please do NOT open a public issue.
Instead, follow the confidential reporting instructions outlined in `SECURITY.md`. We aim to resolve and publicly disclose patches within a responsible timeframe.

---

## Release Protocol

To prepare a new release bundle for upload as a GitHub Release Asset:
```bash
python -m src.release_bundle --output-dir release_artifacts
```
This generates a ZIP archive and a SHA256 checksum manifest inside the `release_artifacts/` directory.

### Reproducible builds

SESTRAV is pure Python with no compilation step, and dependencies are hash-pinned
(`environments/requirements.lock`). To produce a reproducible source/wheel build,
set `SOURCE_DATE_EPOCH` so timestamps are deterministic:
```bash
SOURCE_DATE_EPOCH=$(git log -1 --format=%ct) python -m build
```
Building twice from the same commit should yield byte-identical artifacts.

### Signing releases

Release tags and artifacts are cryptographically signed so consumers can verify
authenticity (not just integrity). Sign tags with `git tag -s vX.Y.Z`; the
verification procedure for release artifacts is documented in `SECURITY.md`.

---

## Code Review Process

- Every change reaches `main` through a Pull Request; direct pushes to `main` are
  avoided.
- Each PR must pass all **required CI checks** (lint, tests, security, Snakemake
  dry-run) before merge.
- A reviewer (the maintainer or a designated backup reviewer; see `GOVERNANCE.md`)
  checks each PR for: correctness, adequate tests, adherence to the coding
  standards, security implications, and documentation/CHANGELOG updates.
- The project's goal is for the majority of changes to be reviewed by someone
  other than the author before release; as the maintainer base grows this becomes
  a strict requirement (see `ROADMAP.md`).

## Developer Certificate of Origin (DCO)

By contributing, you certify that you wrote the contribution or otherwise have the
right to submit it under the project's MIT license, per the
[Developer Certificate of Origin](https://developercertificate.org/). Please add a
`Signed-off-by` line to each commit (your real name and email):
```bash
git commit -s -m "Your message"
```

## Ways to Make a Significant Contribution

We actively welcome substantial contributions, especially from collaborators
outside the core institution. High-impact areas include:

- **New pathogen/allele adapters** — extend curation/proteomes to additional
  viruses or HLA alleles (see `ROADMAP.md`).
- **External-dataset validation** — add independent benchmark datasets and
  comparison harnesses.
- **Documentation** — tutorials, worked examples, and API reference improvements.
- **Independent review** — review PRs, reproduce results, and audit methodology.

Issues labelled **`good first issue`** and **`help wanted`** are good entry points.
Sustained, significant contributors may be invited to become maintainers per
`GOVERNANCE.md`, and are credited in `CONTRIBUTORS.md`.
