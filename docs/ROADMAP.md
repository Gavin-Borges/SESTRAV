# SESTRAV v2.0 Finalization and Strategic Roadmap

This document defines the steps required to freeze SESTRAV v2.0 as a publication-grade release, and establishes the post-release research agenda, collaboration framework, and validation pathway.

## Repository Identity

**GitHub About Description (final, 133 characters)**
A computational immunology framework integrating MHC binding with TCR-facing physicochemistry to predict viral epitope immunogenicity.

**GitHub Topics**
`computational-immunology`, `bioinformatics`, `vaccinology`, `epitope-prediction`, `machine-learning`, `snakemake`, `hpv`, `ebv`

**README Badges**
Add immediately below the title: CI status, Python 3.11, License MIT, Code style black.

---

## 1. Verified Current State (2026-06-03)

| Component | Status |
|-----------|--------|
| `CITATION.cff` | v2.0.0 metadata dated 2026-05-22, five URI authors, MIT license |
| `LICENSE` | Full MIT text present |
| `.gitignore` | Excludes `models/*.joblib`, `models/*.pt`, caches; preserves frozen validation CSVs |
| GitHub Actions | ci, dependency-review, fuzzing, pr-review-check, scorecard, security workflows present |
| Security posture | bandit and pip-audit clean; semgrep rules configured |
| Pipeline stability | 3x execution zero variance; Brier Skill Score 0.20; Platt calibrator refit complete |
| Release bundling | `src/release_bundle.py` produces deterministic archives with SHA256 manifest |

---

## 2. Immediate Finalization Tasks (branch: fix/sestrav-v2-finalization)

### 2.1 Repository configuration
- Create `.github/repository-metadata.json` with the exact description and topics above
- Set description and topics manually via GitHub repository settings (no automation required for a one-time action)

### 2.2 README professionalization
- Insert badge markdown (CI, Python, license)
- Ensure the opening paragraph matches the finalized About description
- Replace any remaining conversational phrasing with declarative, scientific language
- Confirm that all em dashes, placeholder citations (“Standard biochemistry”), and ASCII art are removed

### 2.3 Governance files
- **`CONTRIBUTING.md`**: conda setup, `pytest tests/ -q` requirement, PR checklist, release protocol using `src.release_bundle`
- **`CODE_OF_CONDUCT.md`**: Contributor Covenant 2.1
- **`SECURITY.md`**: disclosure contact, reference to semgrep-rules and existing security workflow

### 2.4 Issue & Pull Request templates
- `.github/ISSUE_TEMPLATE/bug_report.md`: require OS, Python, scikit-learn, mhcflurry versions, traceback, and a config.yaml snippet
- `.github/ISSUE_TEMPLATE/feature_request.md`: require use case, proposed model class, dataset size, and expected performance delta
- `.github/pull_request_template.md`: checklist for tests pass, Snakemake dry-run, freeze_mode true, no changes to frozen results

### 2.5 Version history
- `CHANGELOG.md` using Keep a Changelog format; seed with v2.0.0 entry

### 2.6 Final verification before tagging v2.0.1
```bash
python -m pytest tests/ -v
snakemake --snakefile pipeline.smk --dry-run --cores 1
snakemake --snakefile pipeline.smk full_validation_report --cores 4 --forceall
python -m src.release_bundle --output-dir release_artifacts
```
Gate: `results/freeze_status.json` must contain `"valid": true`

---

## 3. Post-Release Strategic Roadmap

### 3.1 Stakeholder alignment (Weeks 1–4)

| Activity | Goal | Owner |
|:---|:---|:---|
| URI Foundation Team sync | Formalize maintainer roles, contribution policy, and IP stewardship | Gavin Borges |
| Academic mentor review | Audit statistical rigor and SHAP interpretation, validate ANN/GNN benchmarks | Dr. Aura G., Dr. Liu, Dr. Eme |

### 3.2 Wet-lab validation pathway (Months 1–12, contingent on partnership)

**Objective:** Demonstrate that the 30-feature model enriches true immunogenic epitopes beyond binding-only (contingent on partnership for assay execution).

- Synthesize 80 peptides: top 40 SESTRAV-ranked and 40 binding-only controls, balanced across HPV16, HPV18, EBV
- Assay: IFN-gamma ELISpot on HLA-typed donor PBMCs (target >= 10 donors covering as many alleles in config.yaml as feasible)
- Success criterion: SESTRAV top 25 percent achieves R10 >= 2.0 versus binding-only
- Pre-register protocol and analysis plan in `11_External-Testing/Wet_Lab_Protocol_v1.md` before the first assay run

### 3.3 Pathogen and allele expansion (Months 6–18)

| Workstream | Actions | Success criteria |
|:---|:---|:---|
| Pathogen expansion | Curate IEDB-derived training data for HBV, HCV, KSHV; add corresponding proteomes | AUC-PR >= 0.80 on new taxa without regression on HPV/EBV |
| Pan-allele modeling | Integrate 166-feature allele-aware pocket pseudo-sequences | Improved allele-stratified recall; documented in `docs/allele_aware_training_report.md` |
| Bias mitigation | Update `data_bias_audit_v3.md` and recompute sample weights | Balanced recall across taxa and peptide lengths |

### 3.4 Deep-learning promotion criteria

ANN and GNN benchmarks remain **optional** until all quantitative gates are met:

- Training set > 5,000 peptides from at least 3 distinct viruses
- 5-fold CV AUC-PR mean >= 0.85
- Cross-run standard deviation < 0.02 (across three independent seeds)
- Inference time <= 2× Random Forest on a single CPU
- Expected Calibration Error (ECE) < 0.05
- Interpretability: SHAP or surrogate model available

Upon satisfying all gates, the architecture may be promoted to a **second canonical track**, with corresponding model card and `CITATION.cff` update.

### 3.5 Packaging and CI enhancements
- Publish `sestrav` to PyPI as a pip-installable package
- Push a pre-built Docker image (containing the canonical 30-feature model) to GitHub Container Registry
- Automate release-bundle attachment and checksum verification in the GitHub Release workflow

---

## 4. Governance and Decision Rights

- Canonical track: 30-feature RF/XGBoost. Changes to `config.yaml`, `pipeline.smk`, or frozen `results/` artifacts require maintainer review.
- Experimental features start in feature branches. Promotion to canonical requires meeting the Section 3.4 gates.
- `freeze_mode` must be `true` for all release-critical runs.
- Contributions follow `CONTRIBUTING.md`; all PRs must pass CI and a human review.

---

## 5. Success Metrics

| Metric | Target |
|:---|:---|
| OpenSSF Scorecard | >= 7.0 within 30 days of v2.0.1 |
| Peer-review feedback | All items triaged in GitHub Issues within 14 days |
| Wet-lab data analysis | Completed by Month 12 (dependent on partnership) |

---

*Maintainer: Gavin Borges, SESTRAV Lead Developer*  
*Version: 2.0 – 2026-06-03*
