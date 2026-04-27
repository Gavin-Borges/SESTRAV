# SESTRAV Naming Migration Validation Gates (2026-04-24)

This record confirms that naming hardening changes passed pre-release safety checks.

## Gate 1: Test suite

- Command:
  - `conda run -n sestrav python -m pytest tests/ -q`
- Result: **PASS** (`20 passed`)
- Note: sklearn serialization version warnings persist (known, non-blocking for this pass).

## Gate 2: Snakemake DAG resolution

- Command:
  - `conda run -n sestrav snakemake --snakefile pipeline.smk --dry-run --cores 1`
- Result: **PASS**
- Confirmation:
  - DAG resolves canonical proteome outputs including `EBV_B95_8_panel8`.

## Gate 3: Full pipeline rerun

- Command:
  - `conda run -n sestrav snakemake --snakefile pipeline.smk --cores 4 --forceall`
- Result: **PASS**
- Confirmation:
  - Canonical outputs generated for:
    - `HPV16_18_panel8`
    - `EBV_B95_8_panel8`

## Gate 4: Full validation bundle

- Command:
  - `conda run -n sestrav snakemake --snakefile pipeline.smk full_validation_report --cores 4 --forceall`
- Result: **PASS**
- Confirmation:
  - Canonical validation artifacts regenerated.
  - Mode/version-tagged summary aliases generated:
    - `gold_standard_validation__modeA_baseline__IEDB-20260424-EBV_HPV16_BASELINE-v1.csv`
    - `baseline_comparison__modeA_baseline__IEDB-20260424-EBV_HPV16_BASELINE-v1.csv`
    - `h2_tier_a_summary__modeA_baseline__IEDB-20260424-EBV_HPV16_BASELINE-v1.csv`

## Gate 5: Release bundle manifest

- Command:
  - `conda run -n sestrav python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v1`
- Result: **PASS**
- Generated:
  - `release_artifacts/sestrav-v1-20260424T211212Z.manifest.json`
  - `release_artifacts/sestrav-v1-20260424T211212Z.zip`

## Gate Conclusion

Naming hardening changes passed required safety gates for a balanced migration pass with one-release compatibility behavior.
