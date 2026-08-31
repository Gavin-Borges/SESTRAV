# SESTRAV Roadmap

This roadmap describes the project's intended direction for at least the next
12 months. It is a statement of intent, not a guarantee; priorities may shift
with research findings and contributor availability. Progress is tracked in
GitHub Issues and reflected in `CHANGELOG.md`.

_Last updated: 2026-08._

## Near term (0-3 months)

- **OpenSSF Best Practices badge - Passing attained, and it is the terminal tier.**
  ([project 13191](https://www.bestpractices.dev/projects/13191)). **Silver and Gold
  are formally declined as of 2026-08-17**, not scheduled: both require the
  multi-person criteria (`bus_factor`, `two_person_review`,
  `contributors_unassociated`), SESTRAV has one maintainer, and there is no plan to
  add a second (`BUS_FACTOR.md`). The many Silver/Gold criteria the project already
  satisfies - governance, two-scope coverage measurement, Sigstore-signed releases,
  threat model/assurance case - stand on their own merits and continue to be
  maintained. See `docs/threat_model.md` and `GOVERNANCE.md`.
- **Test coverage.** Coverage is measured on two scopes; only the library scope is
  gated in CI:
  - **Library scope** (OpenSSF Silver `test_statement_coverage80`): the importable
    library surface - `src`/`functions` modules without a `__main__` CLI entry
    point. Measured via `.coveragerc.library` (omit list generated mechanically by
    `tools/check_library_coverage.py`, kept in sync by `--check`). Currently
    **~99% statement / ~98% branch** (combined 98.91%), gated at `fail_under=95`.
    This clears the OpenSSF **Gold** targets (>=90% statement, >=80% branch).
  - **Whole-repo floor**: `pyproject.toml`'s `fail_under` blocks regressions across
    the entire tree (research/CLI scripts included), gated at `fail_under=35`,
    currently **49.36%** (branch-inclusive, re-measured 2026-08-16 at `a336360`) -
    comfortably above the floor. This is a local-DX regression gate, not a CI/badge
    gate. **Re-measure rather than carry this figure forward:** it has now moved
    twice in nine days (47.88% on 2026-08-08, 49.71% on 2026-08-15, 49.36% here),
    so a restatement without a fresh run and its own date is not evidence.
    *(Supersedes the previously published 34.37% "measured 2026-06-22", which had
    gone seven weeks stale and described the figure as "a hair under the floor";
    both the number and that characterization were out of date.)*
    Executable scripts (those with `__main__`) are validated by the integration
    tests and CI data/benchmark gates rather than unit statement coverage.

  Raise both floors only as real tests land - never by padding. The library scope
  already clears the Gold targets (>=90% statement / >=80% branch); the whole-repo
  scope is the one with headroom to grow. Subprocess-launched
  modules are measured via the `tools/coverage_subprocess` hook so they are not
  undercounted as 0%.
- **Signed release artifacts - shipped.** Release artifacts carry a keyless
  Sigstore/SLSA build-provenance attestation (`.github/workflows/release.yml`),
  with verification documented in `SECURITY.md`'s "Release Integrity &
  Verification" section. Version tags remain annotated but **unsigned**
  (`version_tags_signed`, SUGGESTED, currently Unmet - see `docs/releasing.md`).
- **Container image.** A publish-to-`ghcr.io` workflow with provenance and SBOM
  is in place (`.github/workflows/docker.yml`); it fires on the next version tag.
  It has not run yet - the workflow was added after `v2.0.3`, so no image is
  published.
- **Packaging.** Publish `sestrav` to PyPI as a pip-installable package.
  **Installation is from source today - nothing has been published yet** and the
  package name is still unclaimed. The publish job in
  `.github/workflows/release.yml` is enabled (`PYPI_PUBLISH` is `true`) and is
  scheduled by any `v*` tag; it then pauses for approval under the `pypi`
  environment's required-reviewer rule. The pending Trusted Publisher was confirmed
  registered on 2026-08-17. Two caveats before the first publishing tag: the publish
  path has **never run end to end** (PR CI cannot exercise `release.yml`, which is
  tag-triggered only), and the publisher is bound to the `Gavin-Borges` personal
  account, so the planned GitHub-organization migration should happen first - see the
  ordering constraint in `docs/releasing.md`.

## Mid term (3-9 months)

- **Pathogen expansion.** Curate IEDB-derived training data for additional
  oncogenic viruses (e.g. HBV, HCV, KSHV) and add the corresponding proteomes.
  Target: **pooled AUC-PR >= 0.65 under peptide-grouped 5-fold CV**
  (`src.ml_utils.PeptideGroupedKFold`) on new taxa, without regression on
  HPV/EBV. **Re-anchored 2026-08-10.** This gate previously read ">= 0.80" with
  no splitter stated; that threshold was set against the pre-remediation
  ungrouped baseline (pooled AUC-PR 0.8312), which is retracted as
  peptide-leakage-inflated (`docs/claims_register.md` D15). Against the current
  certified peptide-grouped baseline of 0.6058, a 0.80 target was not a stretch
  goal but an unreachable one. 0.65 is set as a meaningful improvement over the
  current baseline on the honest scale. **Always state the splitter when quoting
  this gate** - a number without one is not comparable across the D15 boundary.
- **Pan-allele modeling - BUILT AND EVALUATED, NOT ADOPTED.** This entry previously
  read "integrate allele-aware pocket pseudo-sequence features to improve
  allele-stratified recall". That promised a future gain which has since been
  measured and did not appear, so the promise is withdrawn rather than restated.
  The 166-feature allele-aware set (20 physicochemical + 10 binding + 136 HLA
  pocket pseudo-sequence features) is implemented - see `FEATURE_COLUMNS_ALLELE`
  in `src/features.py` and `prepare_features_166` in `src/train_classifier.py` -
  is reachable as `--feature-mode 166`, and has tracked artifacts under
  `models/allele_aware/`. **Corrected 2026-08-27: the cited screen statistic is
  retracted, not merely stale.** A 2026-07-19 paired-bootstrap screen (full
  provenance in `docs/claims_register.md` D36) reported the mode-166-vs-mode-31
  AUC-PR delta as not excluding zero (95% CI [-0.0118, 0.0108]), run under a
  comparison deliberately tilted toward mode-166 (its leak-contaminated OOF
  against mode-31's then-honest OOF). That comparison no longer holds under
  currently tracked inputs: the 2026-08-10 peptide-leakage remediation
  (`30f1b76`, `docs/claims_register.md` D15) re-baselined mode-31's out-of-fold
  scores under a peptide-grouped splitter, but mode-166's OOF
  (`models/allele_aware/rf_oof_predictions_mode166.csv`) was never regenerated
  and still reflects the pre-remediation splitter it always had. Re-running the
  identical screen against current tracked files now gives delta +0.0586, 95%
  CI [0.0457, 0.0716] (excludes zero, nominally favoring mode-166) - but
  mode-166's own AUC-PR in that comparison is unchanged from the 2026-07-19 run
  (its OOF was never re-touched); the shift is entirely mode-31 losing its own
  leak inflation. This is a leak-correction asymmetry between arms, not a
  reversal in mode-166's favor, and it does not show allele-conditioning helps.
  Independently (`docs/claims_register.md` D30), the 136 HLA-pocket columns are
  degenerate as currently built: only 10 of 136 vary at all, collapsing to
  exactly 2 distinct joint vectors (one per HLA locus, A vs B, not one per
  allele), and together they carry about 4% of total RF feature importance.
  Allele-conditioning **as currently featurized and evaluated** is an
  evaluated-but-not-adopted extension, not pending work; re-opening it needs
  both a corrected featurization (D30) and a fair, symmetric, peptide-grouped
  re-evaluation of both arms, not a re-run of this retracted comparison.
- **Bias mitigation.** Refresh the data bias audit and recompute sample weights
  for balanced recall across taxa and peptide lengths.
- **Release automation.** Attach the `src.release_bundle` ZIP to the GitHub
  Release automatically. Checksum generation (`SHA256SUMS.txt`) and artifact
  provenance attestation are already automated in `.github/workflows/release.yml`.

## Longer term (9-18 months)

- **Deep-learning promotion.** ANN/GNN tracks remain optional benchmarks until they
  meet published quantitative gates. The canonical, machine-checked definitions live
  in `src/verify/promote_gnn.py`; this list mirrors them and must stay in sync:
  - Gate 1 - Generalization: **peptide-grouped 5-fold CV AUC-PR >= 0.65** on the full
    training dataset (**re-anchored 2026-08-10**, was `>= 0.85` with no splitter
    stated; see below). The splitter is a **hard precondition, not a footnote**:
    `gate1_generalization` calls `grouped_splitter_violation` first, and an OOF frame
    that does not carry a `splitter` column naming `PeptideGroupedKFold` on every row
    fails the gate outright, with no AUC-PR computed and none printed. The 0.65
    threshold is absolute, not relative to any other model's score.
  - Gate 2 - Stability: cross-fold AUC-PR std **<= 0.02** across the CV folds,
    computed per fold from the `fold` column the OOF artifact now carries. A frame
    without that column fails, because cross-fold spread is not measurable without
    per-row fold identity; folds containing a single class are reported rather than
    dropped, since dropping them shrinks the std.
  - Gate 3 - Latency: GNN CPU inference <= 2x RF CPU inference (per batch).
  - Gate 4 - Calibration: Expected Calibration Error (ECE) < 0.05.
  - Gate 5 - Escape sensitivity: >= 80% of ALL out-of-fold positives score
    above the median out-of-fold negative, pool-wide - despite its name, NOT
    restricted to IEDB gold-standard epitopes (both training entry points
    exclude those from the pool before any fold is built; see
    `docs/claims_register.md` D26).

  The scorecard can be exercised without side effects: `python -m src.verify.promote_gnn
  --dry-run` evaluates all five gates and reports the mutations that would follow,
  leaving `config.yaml` and `models/model_artifact_checksums.json` untouched.

  On passing all five, a track may be promoted to a second canonical model with its
  own model card. **Two corrections logged 2026-08-10:** (1) the 0.85 AUC-PR
  threshold was anchored to the pre-remediation ungrouped RF baseline (0.8312), which
  is retracted as peptide-leakage-inflated (`docs/claims_register.md` D15); against
  the certified peptide-grouped RF baseline of 0.6058 it was unreachable rather than
  ambitious, so it is re-anchored to 0.65 on the honest scale, matching the pathogen
  expansion gate above. Any promotion candidate must be measured under
  `src.ml_utils.PeptideGroupedKFold` for the comparison to be meaningful. (2) This
  list previously omitted Gate 5 entirely and stated Gate 2 as `< 0.02` where the
  code enforces `<= 0.02`; both are corrected here against
  `src/verify/promote_gnn.py`. **A third correction logged 2026-08-12:** the GNN
  track itself was still splitting with an ungrouped `StratifiedKFold` when the
  re-anchor was written, so the gate asked for a peptide-grouped number that no GNN
  run could produce. `src/train_gnn.py` now builds folds with
  `src.ml_utils.PeptideGroupedKFold` (`build_cv_splits`) at both training entry
  points and stamps `fold` and `splitter` onto every out-of-fold row
  (`build_oof_records`). The tracked `models/gnn_oof_predictions.csv` predates that
  repair, so it fails Gate 1 by precondition and Gate 2 for want of fold identity,
  and **no GNN figure sourced from it is comparable to a peptide-grouped one**.

  **Outcome, and the re-run policy that follows from it (recorded 2026-08-16; run
  executed 2026-08-13).** The v5 GNN was evaluated against these gates and **returned
  a null result on the pre-registered bar**. Because the bar is an AND-conjunction,
  Gate 1 alone settles it: pooled peptide-grouped AUC-PR **0.6458 against a >= 0.65
  threshold, failing by 0.0042**; Gate 2 also failed (cross-fold std 0.0234 against
  <= 0.02); Gates 3, 4 and 5 passed. The track is **not promoted**. Reported alongside
  it, because omitting either half would misrepresent the run: against RF mode-31 the
  GNN improved AUC-PR by **+0.0402, 95% CI [0.0286, 0.0520], p < 0.0001** (paired
  bootstrap, 10,000 resamples). The architecture is measurably better on
  discrimination and still misses the promotion bar.

  **This evaluation must not be re-run with different hyperparameters against the same
  held-out set.** Tuning until a 0.0042 shortfall closes is precisely the leakage this
  project flags in every other model, and a threshold that moves after the result is
  known is not a threshold. Re-opening the track requires a bar pre-registered *before*
  the run and evaluated on data not used to produce the result above. That applies to a
  new architecture or feature set too: the risk being controlled is selection across
  repeated attempts, not any one model. The same statement is recorded in
  `src/verify/promote_gnn.py`'s module docstring, next to the gates that enforce it.
- **Wet-lab validation (contingent on partnership).** Pre-register and execute an
  IFN-gamma ELISpot validation comparing SESTRAV-ranked epitopes against binding-only
  controls across HPV16/HPV18/EBV.
- **Governance growth - NOT planned, and removed from this roadmap 2026-08-17.**
  SESTRAV is solo-maintained with no backup candidate and no recruitment underway.
  The multi-person OpenSSF criteria (`bus_factor`, `two_person_review`,
  `contributors_unassociated`) are therefore not on the roadmap. `MAINTAINERS.md`
  documents the process should a qualified co-maintainer ever appear unprompted;
  the project is not pursuing one. The realistic bus-factor mitigations for a solo
  project are archival (Zenodo DOI) and making ownership transferable (GitHub
  organization migration) - both tracked in `BUS_FACTOR.md`.
- **Per-file licensing (Gold `license_per_file` / `copyright_per_file`) - not
  pursued.** The repository is licensed as a whole (see `LICENSE`). Per-file SPDX
  and copyright headers were previously deferred "until a second contributor
  lands"; since no second contributor is expected, that deferral was indefinite in
  substance and is now recorded as a decline. Should the position change, the
  headers would be applied in one isolated, reviewable commit using a
  [REUSE](https://reuse.software/)-style workflow:
  - Add `SPDX-License-Identifier:` and `SPDX-FileCopyrightText:` headers via an
    **idempotent** script (re-running it must be a no-op), driven by a
    `.reuse/dep5` (or `REUSE.toml`) config so binary/data assets are covered by
    declaration rather than inline edits.
  - Preserve file preambles exactly: keep any encoding cookie or shebang on
    lines 1-2, and insert headers **after** `from __future__ import ...` lines so
    import ordering and `__future__` semantics are unaffected.
  - Verify with `reuse lint` in CI before merging the headers commit.

## How to help

Contributions are welcome - see `CONTRIBUTING.md` for the workflow and for the
kinds of significant contributions the project is actively seeking. Issues
labelled `good first issue` and `help wanted` are good entry points.
