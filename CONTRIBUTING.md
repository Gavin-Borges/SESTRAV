# Contributing to SESTRAV

Thank you for your interest in contributing to SESTRAV! This document provides instructions for setting up the development environment, running tests, submitting pull requests, and following our repository guidelines.

## Development Environment Setup

We recommend using Conda to manage environment dependencies.

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YOUR_ORG/SESTRAV.git
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
We use [black](https://github.com/psf/black) to enforce a consistent code style. Before submitting code, format it:
```bash
black src/ functions/ tests/
```

### 2. Running Tests
Tests are located in the `tests/` directory. All tests must pass before submitting a pull request:
```bash
python -m pytest tests/ -v
```

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

## Release Protocol

To prepare a new release bundle for upload as a GitHub Release Asset:
```bash
python -m src.release_bundle --output-dir release_artifacts
```
This generates a ZIP archive and a SHA256 checksum manifest inside the `release_artifacts/` directory.
