## Pull Request Description

Provide a brief summary of the changes introduced by this PR and the rationale behind them.

## Related Issue
Fixes # (issue number)

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Documentation update
- [ ] Optimization / Code cleanup

## Pull Request Checklist

Please ensure all of the following requirements are met before submitting the PR:

- [ ] All unit tests pass locally (`python -m pytest tests/ -v`).
- [ ] Snakemake dry-run succeeds (`snakemake --snakefile pipeline.smk --dry-run --cores 1`).
- [ ] Validation report runs and yields correct results (`snakemake --snakefile pipeline.smk full_validation_report --cores 4 --forceall`).
- [ ] `results/freeze_status.json` has `"valid": true` (freeze mode enabled in `config.yaml`).
- [ ] No changes are made to frozen validation outputs in `results/` unless explicitly planned.
- [ ] Code formatting has been run using `black`.
- [ ] `CITATION.cff` or documentation is updated if new authors/contributors are introduced.
