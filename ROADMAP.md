# SESTRAV Roadmap

This roadmap describes the project's intended direction for at least the next
12 months. It is a statement of intent, not a guarantee; priorities may shift
with research findings and contributor availability. Progress is tracked in
GitHub Issues and reflected in `CHANGELOG.md`.

_Last updated: 2026-06._

## Near term (0–3 months)

- **OpenSSF Best Practices badge.** Attain the Passing badge, then complete the
  Silver criteria (governance, coverage measurement, signed releases, threat
  model/assurance case). See `docs/threat_model.md` and `GOVERNANCE.md`.
- **Test coverage.** Coverage is measured on two scopes, both gated in CI:
  - **Library scope** (OpenSSF Silver `test_statement_coverage80`): the importable
    library surface — `src`/`functions` modules without a `__main__` CLI entry
    point. Measured via `.coveragerc.library` (omit list generated mechanically by
    `tools/check_library_coverage.py`, kept in sync by `--check`). Currently
    **≈91% statement / ≈81% branch** (combined ≈89%), gated at `fail_under=85`.
    This clears the OpenSSF **Gold** targets (≥90% statement, ≥80% branch).
  - **Whole-repo floor**: `pyproject.toml`'s `fail_under` blocks regressions across
    the entire tree (research/CLI scripts included), currently ≈30% statement.
    Executable scripts (those with `__main__`) are validated by the integration
    tests and CI data/benchmark gates rather than unit statement coverage.

  Raise both floors only as real tests land — never by padding — stepping toward
  ≥90% statement / ≥80% branch on the library for Gold. Subprocess-launched
  modules are measured via the `tools/coverage_subprocess` hook so they are not
  undercounted as 0%.
- **Signed releases.** Cryptographically sign release artifacts and tags and
  document verification in `SECURITY.md`.
- **Packaging.** Publish `sestrav` to PyPI as a pip-installable package and push a
  pre-built Docker image (with the canonical 30-feature model) to a container
  registry.

## Mid term (3–9 months)

- **Pathogen expansion.** Curate IEDB-derived training data for additional
  oncogenic viruses (e.g. HBV, HCV, KSHV) and add the corresponding proteomes.
  Target: AUC-PR ≥ 0.80 on new taxa without regression on HPV/EBV.
- **Pan-allele modeling.** Integrate allele-aware pocket pseudo-sequence features
  to improve allele-stratified recall.
- **Bias mitigation.** Refresh the data bias audit and recompute sample weights
  for balanced recall across taxa and peptide lengths.
- **Release automation.** Automate release-bundle attachment and checksum/signature
  verification in the GitHub Release workflow.

## Longer term (9–18 months)

- **Deep-learning promotion.** ANN/GNN tracks remain optional benchmarks until they
  meet published quantitative gates (sufficient multi-virus training data,
  5-fold CV AUC-PR ≥ 0.85, cross-run SD < 0.02, calibration ECE < 0.05, and
  interpretability via SHAP/surrogate). On passing, a track may be promoted to a
  second canonical model with its own model card.
- **Wet-lab validation (contingent on partnership).** Pre-register and execute an
  IFN-γ ELISpot validation comparing SESTRAV-ranked epitopes against binding-only
  controls across HPV16/HPV18/EBV.
- **Governance growth.** Recruit and onboard additional maintainers and
  independent contributors to raise the project's bus factor and enable
  two-person review (see `GOVERNANCE.md`, `BUS_FACTOR.md`).
- **Per-file licensing (Gold `license_per_file` / `copyright_per_file`).**
  The repository is licensed as a whole (see `LICENSE`); per-file SPDX and
  copyright headers are intentionally deferred until a second contributor lands,
  so authorship attribution is recorded accurately rather than retroactively
  assigned to a single author. When that happens the headers will be applied in
  one isolated, reviewable commit using a [REUSE](https://reuse.software/)-style
  workflow:
  - Add `SPDX-License-Identifier:` and `SPDX-FileCopyrightText:` headers via an
    **idempotent** script (re-running it must be a no-op), driven by a
    `.reuse/dep5` (or `REUSE.toml`) config so binary/data assets are covered by
    declaration rather than inline edits.
  - Preserve file preambles exactly: keep any encoding cookie or shebang on
    lines 1–2, and insert headers **after** `from __future__ import ...` lines so
    import ordering and `__future__` semantics are unaffected.
  - Verify with `reuse lint` in CI before merging the headers commit.

## How to help

Contributions are welcome — see `CONTRIBUTING.md` for the workflow and for the
kinds of significant contributions the project is actively seeking. Issues
labelled `good first issue` and `help wanted` are good entry points.
