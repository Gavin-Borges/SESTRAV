## Summary

<!-- One paragraph: what this PR does and why -->

## Type of Change

- [ ] Bug fix
- [ ] New feature / capability
- [ ] Refactor (no behavior change)
- [ ] Test coverage improvement
- [ ] Documentation update
- [ ] New virus panel or dataset
- [ ] Model / feature change

---

## PR Checklist

### Code Quality
- [ ] `ruff check .` passes (no lint errors)
- [ ] `mypy src/` passes (0 new errors)
- [ ] `pytest` passes (all tests green, no regressions)
- [ ] New functionality has accompanying tests

### Biological Accuracy (required for any model, feature, or claims change)
- [ ] **Q1 Mechanism** — Any new biological claim has a primary literature citation (not just a tool reference)
- [ ] **Q2 Scope** — New claims specify which viruses, alleles, peptide lengths, and assay types they apply to
- [ ] **Q3 Limitation** — New claims include corresponding limitation language
- [ ] **Q4 Fairness** — Any benchmark comparison uses equivalent evaluation conditions, or asymmetry is disclosed
- [ ] `docs/claims_register.md` updated with any new public-facing claims

### Model / Feature Changes (if applicable)
- [ ] `docs/feature_glossary.md` updated if feature set changed
- [ ] `docs/model_cards/` updated if model performance changed
- [ ] `config.yaml` `feature_mode` and `model_path` updated if a new model is canonical
- [ ] `docs/model_evaluation_summary.md` updated with new AUC-PR result

### Data Changes (if applicable)
- [ ] New dataset validated against `data/immunogenicity_dataset_v4_schema.json`
- [ ] `docs/data_registry.md` updated with row counts, class balance, and provenance
- [ ] Overlap analysis run vs existing training set
- [ ] `antigen_processing_cache_path` updated if new peptides introduced

### Documentation
- [ ] `CHANGELOG.md` updated (under `[Unreleased]`)
- [ ] `docs/limitations_statement_v1.md` updated if a new limitation is introduced
- [ ] `docs/antigen_accessions.md` updated if new viruses or accessions added

---

## Testing Evidence

```
pytest -q
... passed, ... skipped in ...s
```

## Before / After (for model changes)

| Metric | Before | After |
|--------|--------|-------|
| AUC-PR | | |
| AUC-ROC | | |
| ISSR@10 | | |

## Reviewer Notes

<!-- Anything the reviewer should pay particular attention to -->
