# Changelog

All notable changes to the SESTRAV project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Seven runtime dependencies that the packaged code has always imported but never declared.**
  An AST audit of every module-scope import across the three packaged trees (`sestrav/`, `src/`,
  `functions/`) found `biopython`, `mhcflurry`, `PyYAML`, `scipy`, `xgboost`, `openpyxl` and
  `networkx` imported but absent from `[project].dependencies`. Two of them are reached by
  `sestrav predict`, the flagship command: `src/cli.py`'s `cmd_predict` imports all four
  `functions/stage*.py` modules, of which stage 1 imports biopython and stage 2 imports mhcflurry.
  A `pip install sestrav` would therefore have failed at stage 1 on a clean environment. The gap
  was in the metadata a PyPI consumer resolves, not in CI - all seven are already pinned in
  `environments/requirements.lock`, which is why no test ever caught it.
- **A `CITATION.cff` consistency gate on the release workflow.** Nothing in CI, the tests, the
  scripts or the integrity harness referenced that file, which is how the W6 "phantom release"
  arose: `CITATION.cff` advertised a version and date for which no tag existed. The tag job now
  fails unless `CITATION.cff`'s `version` matches the tag and its `date-released` is a real ISO
  date that is not in the future. Both `pyproject.toml` and `CITATION.cff` are now reported
  together, so a mismatched release surfaces every offending field in one run rather than one
  per attempt.

### Fixed
- **The PyPI publish path could not have succeeded, and had never once run.** The publish job was
  gated on a repository variable set one day after the only tagged release, so it was `skipped`
  on v2.0.3 and has never executed. Three independent defects were found in it:
  - `dist/SHA256SUMS.txt` was shipped to the publish job alongside the distributions, and
    `gh-action-pypi-publish` runs `twine upload dist/*` unfiltered. twine raises
    `InvalidDistribution` on any extension outside `.whl`/`.tar.gz`/`.zip`, so the upload would
    have failed. The artifact handoff now carries distributions only; the checksum manifest is
    still attached to the GitHub Release, which reads it from the build job's own `dist/`.
  - The post-upload smoke test used `pip install --no-deps` and then imported `sestrav` and
    `functions.stage1_peptide_generation`. Neither import can resolve without dependencies, so
    the step could only ever have failed. `--no-deps` is removed, and the step now retries, since
    a fresh upload can lag the CDN and a failure at that point is unfixable by re-running.
  - Most consequentially, verification happened only *after* `gh-action-pypi-publish`. PyPI
    permanently refuses a re-upload of the same version, so any failure there would have burned
    the version number and left a broken release published. A PRE-PUBLISH GATE now installs the
    freshly built wheel with its dependencies into a clean virtualenv, asserts the installed
    version matches the tag, and exercises the imports and the console script - all before the
    provenance attestation and the GitHub Release, while a bad build is still recoverable.
- **The Docker image installed a third of the package.** The build stage copied `src/` alone,
  though `pyproject.toml` declares `packages.find` over `sestrav*`, `src*` and `functions*`, and
  setuptools can only find what is in the build context. `sestrav --help` still worked, because
  the console script is `src.cli:main`, which is why the image's smoke test would not have caught
  it - but `import sestrav` failed outright and `sestrav predict` died at stage 1. `functions/`
  and `sestrav/` are now copied. Note that `docker.yml` fires on the same `v*.*.*` tag as the
  release workflow and has never run either, so this would have reddened the first release
  alongside PyPI.
- **`docs/paper.md` cited its references out of first-appearance order, which Vancouver style
  requires (claims register D25's own deferred note, "Fix that before submission").** D25 and D27
  appended three new references to the end of an already-drafted list instead of renumbering, so
  the Abstract cited [25] and [26] before [1] ever appeared. All 27 references and all 42 in-text
  citation tokens were renumbered into strict first-appearance order in a single atomic pass.
  - Verified by reconstruction rather than by spot check: applying the derived permutation to the
    pre-renumber body reproduces the post-renumber body with a zero-line diff, which proves no
    prose changed, no citation was dropped, added or remapped inconsistently, and no
    number-to-source binding was touched. First-appearance order in the result is exactly 1..27.
  - Three lookalike non-citation brackets - a `[0, 1]` normalized range and two confidence
    intervals - were confirmed byte-identical, and the reference entries themselves were reordered,
    not rewritten.
  - Cross-document pointers that cited the old numbering by value were swept in the same pass
    (`docs/claims_register.md` D25/D27, and this file). Where a pointer had to survive future
    renumbering it now names the papers instead of their numbers.
- **The manuscript's central novelty claim was false, and so was a checkable claim about a
  competitor (claims register D25).** The Abstract asserted, unhedged, that no published
  immunogenicity predictor had reported a systematic leave-one-pathogen-out benchmark on
  assay-confirmed negatives. Two published falsifiers, both verified by reading the primary
  sources: **Bravi et al. (eLife 2023)** satisfies all four conjuncts - MHC class I,
  leave-one-organism-out with per-fold retraining, strictly assay-confirmed IEDB negatives,
  mean AUC 0.68 - and **TRAP (Genome Med 2023)** trains explicitly on non-SARS-CoV-2 and
  non-vaccinia splits, its only gap being unassayed thymic self ligands among its negatives.
  - **The durable lesson is the search method, not the claim.** Both falsifiers use different
    vocabulary ("leave-one-organism-out", "cross-species manner"), so a phrase search on this
    project's own coined term returned nothing. A negative claim searched only by its author's
    phrasing reads as engineered to survive, and must be searched by concept.
  - **The headline finding is also preempted.** Buckley et al. (2022) benchmarked nine published
    models on a compiled panel of assay-confirmed SARS-CoV-2 CD8+ T-cell epitopes and found none
    beat random appreciably, so the mean LOO AUC-ROC of 0.463 generalises that result rather than
    discovering it. The distinct contribution is methodological: the quantified test-partition
    contamination effect.
    - *Corrected 2026-08-14 (claims register D27): this entry said "eight models on SARS-CoV-2
      megapool peptides". Both were checkably wrong against the primary source - the paper states
      "we evaluated the performance of nine models", and "megapool" appears once, used informally
      by the authors for their own compiled IEDB/VIPR dataset, not as an assay methodology. D27
      corrected `docs/paper.md` and `docs/claims_register.md` on 2026-08-13 but did not sweep this
      file.*
  - **A false statement about BigMHC, corrected in the same pass.** Section 1 said its
    "training data composition was not fully disclosed". BigMHC discloses it explicitly
    (1,580 positive / 5,293 negative, with the train/validation split, public Mendeley data
    and a public GitHub repository). Replaced with the accurate criticism: its immunogenicity
    training set is predominantly neoepitopes, 5,279 of 6,873 examples.
  - All three novelty loci now describe the prior art and cite it (Bravi et al., Lee et al./TRAP,
    and Buckley et al. 2022 - added as references [25]-[27], renumbered to [1], [2] and [10] by the
    2026-08-14 first-appearance pass; named here rather than numbered so the pointer cannot rot again)
    rather than
    asserting a vacuum, since a negative literature claim can be falsified but never proven.
- **`docs/paper.md` Section 3.2 reported a calibration ECE pair that was stale, cited to a file
  that did not contain it, and explained by a mechanism its own data refutes (claims register
  D24).** The section claimed the isotonic layer improves global ECE "from about 0.028 to about
  0.003" and that per-virus ECE improves for 8 of the 9 target viruses.
  - **The pair was stale, and it was stale because nothing was bound to it.** It was computed
    2026-07-11/12 against the pre-D15 out-of-fold scores. `30f1b76` ("re-baseline every certified
    v5 number") regenerated `models/v5/rf_oof_predictions_mode31.csv` under the peptide-grouped
    splitter and never reached this pair, because no integrity-manifest claim pointed at it.
    Recomputed against the current artifact: raw pooled ECE is **0.060, not 0.028**, and per-virus
    ECE improves for **1 of 9 (HBV only), not 8 of 9**.
  - **The cited source never contained the numbers.** `_local/staging/calibration_assessment.csv`
    is gitignored, so no reader could open it, and it holds only nine per-virus rows with no
    pooled row. The previously planned remedy - promote that CSV - would therefore not have made
    the pair reproducible, and is retracted.
  - **The first replacement draft was itself wrong.** It attributed the low pooled ECE to the
    out-of-panel rows being "trivially easy to calibrate". Two independent adversarial audits
    caught it: those rows have `ece_cal` **0.140**. The real mechanism is **cancellation** - in
    every score bin the target viruses are under-confident and the out-of-panel rows
    over-confident, so the pooled 0.005 lands *below* both components (0.235 and 0.140) instead of
    between them.
  - **A fourth claim in the same paragraph was false and is corrected.** Isotonic calibration is
    monotone *non-decreasing*, so it creates ties (10,119 distinct raw scores collapse to 274) and
    is not AUC-preserving: pooled AUC-ROC moves 0.8136 to 0.8107. The manuscript's AUC-ROC values
    are unaffected because they are computed on raw scores, which is now what the text says.
  - **Remediation.** New `scripts/assess_calibration.py` cross-fits the calibration layer over
    peptide-grouped folds and writes tracked `results/calibration_assessment_v5_mode31.csv` with a
    provenance sidecar (`.gitignore` negations added so both actually ship). Five values are bound
    in the integrity manifest, including the `off_panel` row that makes the cancellation argument
    checkable. Section 3.2 now reports calibration as a limitation rather than a benefit.
  - **Not fixed here:** `models/isotonic_calibrator.joblib` (2026-07-12, untracked) is still
    fitted on the retired score distribution. Refitting changes deployed scoring behaviour and is
    tracked as a separate decision.
- **The integrity harness's citation check passed any `_local/` citation unconditionally while
  still counting it as verified, and could not see data-file citations at all.** Its PASS line read
  "622 prose citation(s) checked, all resolve to tracked files", which was false for five of them.
  Two independent holes: the `_local/` guard passed such citations in *any* document, and the
  citation regex matched only `.md`/`.py` targets, so `docs/paper.md`'s citation of a gitignored
  `.csv` (the D24 defect above) was invisible for a second, unrelated reason. Fixing only the first
  would not have caught D24. The `_local/` allowance is now scoped to provenance ledgers
  (`docs/claims_register.md`, `docs/data_registry.md`, `CHANGELOG.md`) and fails closed everywhere
  else; the regex is widened for `_local/` targets only, so the `data/*.csv` artifact references
  the repo has already ruled false positives do not flood the check; and the PASS line now reports
  resolved, pair-exempted and ledger-allowed counts separately. Adversarial selftest coverage went
  from 35 to 42 cases, mutation-tested so each probe is proven to catch the specific regression it
  claims to.
- **The Tier A external benchmark (0.828) was disclosed as peptide-leakage-exposed for four days on
  a mechanism that does not apply to it. Ruled on and corrected across ~24 sites (claims register
  D22).** D15 established real exact-duplicate CV leakage on the **v5** corpus and, by a code trace
  through `src/prepare_external_validation_inputs.py`, extended that finding to the Tier A arm.
  D16 had separately established that the Tier A figures come from a 2026-05 run against the
  720-row root `immunogenicity_dataset.csv` at `69e0e5c`. Both could not be right: that corpus has
  **720 rows and 720 unique peptides**, so an ungrouped splitter is already peptide-disjoint on it
  and grouping is a mathematical no-op - the same argument the repo already accepts for the v3
  corpus at `src/h2_tier_a_evaluation.py` and in D17.
  - **The provenance was traced, not assumed.** `results/external_validation_input.csv` and
    `results/table3_tier_a_metrics.csv` were each committed exactly once (`f360b90` 2026-05-23,
    `f5153152` 2026-06-21) and never regenerated. The `rf_oof_score` column that actually feeds the
    benchmark matches the small zero-duplicate corpus by scored population (704 = 720 minus the 16
    `GOLD_STANDARD_EPITOPES`, exactly), by holdout-exclusion pattern, and by `n_estimators`
    fingerprint (704/704 values are multiples of 1/200, versus 671/988 for the larger
    duplicate-bearing OOF file sitting in the same commit's tree).
  - **D15's own cited evidence for residual Tier A exposure measures a different question.**
    The "-0.0176 AUC-PR" figure came from `scripts/audit_cv_leakage.py`'s `_tier_a_ab`, which trains
    a **fresh model on the current v5 corpus** and re-scores a 414-peptide subset; it never reads
    the certified `rf_oof_score` column. Its docstring claimed otherwise and has been corrected
    (docstring only, no logic changed).
  - **A different, previously undisclosed leakage channel was found and is now disclosed instead.**
    Zero *exact* duplicates does not mean zero *near* duplicates. An all-pairs substring-containment
    scan of the 704-peptide pool - the same containment test `src/h2_tier_a_evaluation.py` already
    applies to its own corpus - found **185 overlapping pairs across 226 distinct peptides, 32.1%
    of the pool**, at length differences of 1-3 residues and 77.8% same-label concordance,
    consistent with the same epitope tested at different registration boundaries. **This is
    disclosed as a real risk of unmeasured effect, not as proven leakage**: the historical fold
    assignment is unrecoverable, so whether it moved the reported 0.828 cannot be established.
    **0.828 is not void and is not corrected** - unlike the D17 all-zeros case.
  - Corrected at `README.md`, `ARCHITECTURE.md`, `USAGE.md`, `docs/paper.md` (abstract, Section 2.4,
    Section 3.5, front-matter status), `docs/model_cards/rf_30feature.md`,
    `docs/model_cards/rf_31feature_integrated.md`, `docs/claims_register.md` (D22 plus D15's own
    addendum and two Section 2 disclosure rows), `docs/proposals/2026_feature_upgrade_roadmap.md`,
    and the two generator scripts `src/external_validation_fairness.py` /
    `src/external_validation_finalize.py`, which would otherwise have re-emitted the retracted
    claim into artifacts on their next run. **This supersedes the earlier `[Unreleased]` note
    describing that code-trace as "the chain D15 uses to establish the Tier A leakage exposure".**
    The ANN and XGB 30-feature cards are a deliberate exception - their language describes the
    v3-era corpus, which D22 does not cover.
- **`docs/paper.md` asserted that `allele_matched_nonbinder` decoys were "selected for low MHC
  binding affinity". They are selected for HIGH affinity (claims register D21).** The sole tracked
  generator, `scripts/generate_hard_decoys.py`, keeps only peptides scoring
  `>= PRESENTATION_THRESHOLD` (0.5) and names the retained set `strong`; `--negative-origin` is a
  cosmetic label with no effect on that filter. Measured directly: of the 218 such peptides carrying
  a score in `models/peptide_binding_matrix_v5.csv`, the median max-per-allele presentation score is
  **0.761** (range 0.503-0.982) - *higher* than the 0.705 median for true positives (n=7,037). Both
  occurrences corrected; the replacement causal mechanism for the associated LOO AUC-ROC inflation
  is deliberately **not** asserted, because the low-affinity story is now empirically excluded and
  no verified alternative exists.
- **Two tracked artifacts disagree on the `protein` annotation for an identical row set (claims
  register D20 - root cause located and the generator fixed 2026-08-12, see the entry below; tracked
  artifacts NOT regenerated, so the disagreement below is still observable in the committed tree).**
  For the 693 HIV-1 `allele_matched_nonbinder` rows,
  `data/immunogenicity_dataset_v5.csv` records `ViralProteome_HXB2` while
  `models/v5/rf_oof_predictions_mode31.csv` records `SYNTH_<peptide>`; the OOF export carries the
  same placeholder for 22,456 `tested_negative` and 7,655 `label=1` rows. Found when an audit and
  the author each queried a different artifact and both got real, reproducible, disagreeing answers.
  The manuscript no longer asserts source-protein provenance for these rows.
- **`docs/results_report.qmd` presented RETRACTED metrics as current results under a false
  "Leakage-Proof" attestation (claims register D23).** The tracked, CI-rendered 2026-06-08 report
  asserted `[x] Leakage-Proof CV splits: Implemented Leave-One-Protein-Out (LOPO) cross-validation to
  isolate proteins`. False four ways: LOPO is not the splitter that produced its artifact (5-fold
  `MultiStratifiedKFold`); that splitter is the *ungrouped* one D15 retracts; LOPO could not isolate
  proteins anyway, because it groups on the `protein` column `train_models()` overwrites with
  `SYNTH_<peptide>` for 97.9% of rows (D20), collapsing it to leave-one-*peptide*-out; and the run is
  31-feature, not the "30-feature" it claims. Its Section 2 renders `models/training_results.csv`
  live, which holds the D15-retracted AUC-ROC 0.9429 / AUC-PR 0.8312. **The durable finding is a gate
  limit:** the source contained no retracted literal - the numbers are read from a CSV at render time
  and exist only in rendered output - so no text-scanning gate could catch it, the same shape as
  D17's `freeze_status.json` reaching run manifests through a generation step. **Severity bounded:**
  CI renders but does not upload, deploy or publish (only the `.qmd` is tracked), so nothing reached a
  public URL. Remediated by annotation, not rewriting: the document keeps its three numbered claims
  verbatim and gains a four-point DO-NOT-CITE banner. The retracted-token sweep's glob was widened to
  `docs/**/*.qmd`, which closes the `.qmd` gap but **not** the generated-output class.
- **D20's root cause located and the generator fixed** (supersedes the "root cause not located; no
  generator fixed" status this entry carried until 2026-08-12). `load_all_proteins()`
  (`src/train_classifier.py`) hardcodes a FASTA list that is a **second source of truth alongside
  `config.yaml`'s `proteome_files` map, and the two silently diverged**: the list named only the 4
  EBV/HPV files, so peptides from the other target viruses could not resolve to a real parent
  protein even though `config.yaml` registered their proteomes and `src/naming.py` carried their
  canonical IDs. The 7 missing FASTAs are now read (78 -> 103 protein sequences), which resolves all
  693/693 HIV-1 `ViralProteome_HXB2` rows (HIV-1 has 3,269 active rows in total, of which 2,006
  resolve post-fix - the 693 figure is that one decoy subset, not the virus), and
  `tests/test_train_classifier.py::test_config_proteomes_are_all_loaded`
  FAILs if `config.yaml` ever registers a proteome the list omits. **Scope stated honestly: this is
  not the explanation for the 97.9% placeholder rate** - 21,432 of those rows are
  `Orthopoxvirus vaccinia`, which has no proteome FASTA in the repository, so 77.7% of active rows
  still resolve to `SYNTH_` after the fix. Resolution on the population the mapper actually sees
  rises 2.24% -> 22.32% (798 -> 7,945 of 35,597 rows). **No tracked artifact is regenerated**, so the
  two-artifact disagreement remains observable in the committed tree; regeneration changes no CERTIFIED
  metric (the column feeds no feature and no splitter) but is NOT change-free: it takes
  `training_subgroup_metrics.csv`'s protein dimension from 31,656 to 22,592 rows and its computable
  protein subgroups from 28 to 208. An open owner decision. Also disclosed: the
  fix introduces 9 cross-virus mis-attributions by substring homology (on the 35,597 active rows the
  mapper processes), and changes `--lopo` fold count 15,738 -> 11,111. See D20 for the full enumeration, including two tracked
  `models/allele_aware/` exports and three `training_subgroup_metrics.csv` files an earlier sweep
  missed.
- Six further `docs/paper.md` defects closed, each contradicted by a tracked artifact: the
  feature-importance split (stated 63.5/33.9/2.6, actual **55.8/41.7/2.5**, overstating binding
  dominance by 7.7 points); an HIV-1 out-of-fold claim whose two figures reproduced nothing and
  whose ranking **inverts** against assay-confirmed negatives, contradicting the paper's own
  Section 3 six hundred lines away; a quarantine example citing "three HLA-B*27 EBV rows carrying
  contradictory IEDB records" that does not exist in the shipped data (EBV's real 31 quarantined
  rows are 27 `HLA class I` + 4 `HLA-B57`, and "contradictory IEDB records" is not a mechanism
  anywhere in the codebase); two stated ranges that excluded their own extremes (class imbalance
  "0.6:1 to 1.8:1" omitting HIV-1's **3.3:1**; decoy gap "0.03 to 0.23" omitting DENV's **0.37**,
  reported two sentences later); and a methods claim that DeLong's paired test was used, when
  `src/statistical_bootstrap.py`'s own docstring states paired bootstrap (N=10,000) was chosen
  **instead of** DeLong precisely because DeLong's normality and independence assumptions are
  violated here.
- **The retracted-token sweep never scanned 19 of the 45 markdown files under `docs/`, including
  all five model cards and the Phase 0 roadmap.** `live_doc_globs` used `docs/*.md`, and
  `pathlib.glob` does not cross a path separator, so everything in `docs/model_cards/`,
  `docs/proposals/`, `docs/architecture/`, `docs/data/`, `docs/pipeline/`, `docs/external_testing/`
  and `docs/outreach/` was invisible while the harness reported PASS on every token. Widened to
  `docs/**/*.md`.
  - **`docs/paper.md` was separately whitelisted out entirely, on a false premise.** Its exemption
    read "SUPERSEDED preprint - banner-flagged, body frozen". The file carries no SUPERSEDED banner,
    its front matter reads `status: manuscript draft` with `last_updated: 2026-08-10`, and both D18
    item (3) and D19 landed corrections in it at `96ab220`. (D17 did **not** touch it - an earlier
    draft of this entry said it did, and that claim was retracted after checking
    `git log -- docs/paper.md`.)
  - **The gap was measured, not asserted, and it was not theoretical.** Under the exemption the
    three D11 LOO tokens - `0.870`, `0.824`, `0.784` - each reported **"0 occurrence(s)" PASS**,
    because their only live occurrences sit in that one file. The harness was certifying a vacuous
    check. With `paper.md` scanned they report 2/1/1 occurrences and still PASS, all in genuine
    retraction context. Verified by A/B against the real sweep, then the probe was reverted and the
    file re-checked by sha256.
  - Widening the glob immediately surfaced **seven live citations of the void R10 = 0.9494**
    (D17); narrowing the false-PASS marker below surfaced **three more**, ten in total. A
    subsequent hand sweep of the same two documents found **seven live citations of the
    D15-retracted within-virus mean 0.751**. **The harness could not have found those seven, and
    that is a finding in itself:** `0.751` has no `[[token]]` in `retracted.toml`, and neither
    does any other member of the D15-retracted family (`0.712`, `0.8312`, `0.9429`). An earlier
    draft of this entry credited the glob widening with surfacing them; that is mechanically
    impossible and is retracted. All ten-plus-seven sit in two **gitignored, untracked** planning
    documents - LinkedIn "About" copy, a proposed article title, a draft manuscript abstract, and
    interview talking points. **No tracked reader-facing document carried either number as
    a live claim.** Two tracked *code* comments did name a retracted figure as "certified", and
    both are corrected here: `tests/test_entry_point_help_smoke.py` (0.9494, void per D17) and
    `scripts/audit_cv_leakage.py` (0.751 and 0.712, retracted by D15). An earlier draft of this
    entry conceded only the first, which made the sentence false as written. All corrected
    locally; the null-result narrative is unaffected, since the corrected R10 = 1.0588 still fails
    the R10 >= 2.0 gate.
    - Worth recording: the first correction pass caught only 3 of the 10 R10 sites and none of the
      0.751 sites, and then **asserted in a banner that the file had been corrected** - a false
      all-clear on exactly the axis it was not. The adversarial audit found the remainder. A
      partial fix advertised as complete is worse than no fix.
  - **A real false-PASS marker was found and narrowed: `"not report"` -> `"do not report"`.** The
    old form matched innocuous prose - "a benchmark **not report**ed by any published MHC-I
    predictor" - and was single-handedly rescuing three of those live void-R10 citations, so the
    harness printed "all in retraction context" over a genuine leak.
  - `"amendment"` and `"inflat"` were briefly added to `markers` to guard a hypothetical false-FAIL,
    then **reverted** once measured: zero occurrences anywhere depended on either, so they were
    pure gate relaxation buying nothing. The asymmetry is the lesson - a false FAIL is loud and
    self-correcting, a false PASS is silent. Speculative markers are not worth their cost.
- **D19's "two live code docstrings" follow-up is closed, and its enumeration was incomplete.**
  Both named sites were corrected (`scripts/compute_pooled_honest_metric.py`'s docstring and inline
  comment; `src/train_classifier.py`'s `_excluded_bloc_cv_metrics` docstring). **The D19
  enumeration was incomplete, and a repo-wide sweep found two classes, not one.**
  - **Class 1 - wording (the vaccinia bloc called a decoy), ten sites:**
    `scripts/audit_cv_leakage.py` `_vaccinia_ablation`; four in
    `docs/proposals/2026_feature_upgrade_roadmap.md`; **`README.md`'s per-virus footnote** - the
    front door - which read "the trivially separable vaccinia decoys" while attached to the
    *current* certified 0.733 / 0.670; `README.md`'s 0.9368 retraction note;
    `docs/model_evaluation_summary.md`; `docs/validation_summary.md`; and
    `docs/model_cards/rf_31feature_integrated.md`.
  - **Class 2 - substance: self-proteome decoys asserted to be inside the SCORED pool.** This is
    D19's root claim rather than its wording tail, and it sits on the most-read surfaces.
    `ARCHITECTURE.md` described the 35,597-row active set as "central-tolerance self-binder plus
    IEDB viral negatives" and explained the certified 0.658 as caused by "the hard decoys";
    `README.md` billed Paradigm 2 as a "hard-decoy generalization set" including
    "central-tolerance decoys"; and `docs/model_cards/rf_31feature_integrated.md` said the negative
    class behind the certified 0.8137 / 0.6058 is "dominated by central-tolerance self-peptides",
    citing 8,811 all-negative `Self` rows. **All 8,811 are quarantined and zero reach the active
    pool**, verified by the same crosstab D19 used. The corpus-construction and LOO-training
    statements that legitimately DO include the 5,000 decoys were left alone, exactly as D19 ruled.
  - **Class 3 - a withdrawn argument.** `docs/proposals/2026_feature_upgrade_roadmap.md` Section
    4.6 rejected `feature_mode=35` partly on a "structural reason": that `generate_hard_decoys.py`
    builds most hard negatives from the human self-proteome, so an is-this-human feature would
    predict "negative" by construction. **Withdrawn** - `scripts/audit_cv_leakage.py` scores on
    `_load_active()`, which drops every quarantined row, so *zero* self-proteome decoys sit in the
    frame that produced the cited 0.6055. The null result stands on its own evidence; the
    explanation offered for it did not. That section's ablation citation was corrected in the same
    pass: it read "ungrouped ... -0.0002" and is now the peptide-grouped 0.6085 -> 0.6069 =
    **-0.0016** (`models/v5/training_results_ablation.csv`, regenerated under `PeptideGroupedKFold`
    at `30f1b76`).
  - **Ten is the wording-class count; no cross-class total is certified**, deliberately. Three
    successive counts (five, then seven, then ten) were each wrong, every time because the sweep
    was scoped to the sites a prior report named instead of being run repo-wide. **The durable finding is that no mechanical gate covers either
    class:** the harness matches retracted *numbers*, and neither "decoy" nor a false claim about
    what a pool contains is a number. Compositional claims should be bound to a crosstab the way
    figures are bound to CSV cells.
  - Left alone deliberately: `scripts/generate_hard_decoys.py` and the `prepare_features_35`
    docstring's "hard decoys" both refer to the genuine self-proteome decoy set, which **is**
    synthetic. Only the *Orthopoxvirus vaccinia* bloc was ever mislabelled.
- **A stale internal comment in `tests/test_entry_point_help_smoke.py` named R10 = 0.9494 as the
  "certified" result.** It is void (D17); README certifies 1.0588. Corrected. `tests/` is outside
  `live_doc_globs`, which is why the sweep never saw it.

- **`results/h2_tier_a_summary.md` now states WHY its ungrouped splitter is harmless**, closing the
  item the prior session plan called "the largest remaining honesty gap". That framing turned out to
  overstate it: `src/h2_tier_a_evaluation.py` already emitted the splitter name into the artifact, so
  D15's disclosure duty was met. What was missing was the justification - a reader could not tell a
  benign ungrouped splitter from a D15-class defect. The generator now emits, alongside the splitter
  line, that the v3 corpus is **1,004 rows over 1,004 unique peptides** (verified directly, not
  carried over from D17), so no peptide can straddle a fold boundary and a grouped re-run is a no-op.
  Written at the generator rather than into the artifact so the two cannot drift apart.
  **Regenerating produced a 6-line diff and zero numeric change** - both H2 CSVs are bit-identical,
  which independently re-confirms the D17 corrected values reproduce.
  - **That 6-line insertion immediately broke a live citation, and the incident is worth recording:**
    it shifted the R10 row in the generated summary from line 17 to line 23, and
    `docs/Wet_Lab_Protocol_v1.md` cited `results/h2_tier_a_summary.md:17` for its powering prior -
    so a reader following it landed on the integrated-arm ISSR@10 instead of the ratio, **in a
    pre-registration document**. Caught before commit and fixed by de-line-numbering the citation
    (it now names the "Enrichment ratios" anchor, with a note that generated files shift whenever
    the template changes). This is the **tenth** instance of line-citation rot handled here, and it
    was created by the same commit that fixes the other nine - which is the strongest available
    argument that prose discipline is not sufficient and a mechanical, symbol-anchored gate is
    needed. Generated artifacts are the sharpest case: their line numbers move whenever a generator
    template changes, so they should never be cited by line at all.
- **Two rotten `src/train_classifier.py` line citations re-anchored by symbol, across nine sites in
  seven tracked files.** Both were correct when written and drifted silently as the file grew; this
  is the failure mode recorded in the prior entry as having recurred three times.
  - `:781-786` (cited for the `rf_oof_predictions*.csv` writes) - the real writes were at
    **941/946** *as of this commit*. **They moved again in the very next commit** - `42de845` is
    a direct child of this one (`git rev-list --parents` confirms the single parent) and made a net
    +3 change upstream of them (four lines added, one removed), putting the writes at 944/949 and
    `gs_mask` at 678. That is exactly the recurrence this entry predicts, one commit later.
    **Every citing site outside this changelog is now symbol-anchored with no line number**; the
    numbers retained here are historical record, not citations.
    - Worth recording how the count above was first got wrong: an earlier draft said "three commits
      later", read off `git log --oneline`, which is date-ordered and interleaved two dependabot
      merges that are **not ancestors** of `42de845`. Commit distance is an ancestry question and
      needs an ancestry-aware instrument (`git rev-list --count A..B`). Same shape as the
      `resave_checkpoint.py` error in this same pass: an instrument that could not answer the
      question it was asked.
    This one is **load-bearing**: it is the code-traced provenance chain D15 uses to establish the
    Tier A leakage exposure, so a reader following it landed on unrelated sample-weight code. Five
    citing sites, not the two previously recorded: `docs/claims_register.md` (D15 **and** the
    Section 2 Tier A row), `docs/proposals/2026_feature_upgrade_roadmap.md`,
    `scripts/audit_cv_leakage.py`, and `CHANGELOG.md`.
  - `:555` (cited for the `GOLD_STANDARD_EPITOPES` exclusion) - the real `gs_mask` line was
    **675** *as of this commit*, and moved to 678 at `42de845`. Now symbol-anchored.
    Four citing sites: `docs/holdout_and_qc_policy.md`, both RF model cards, and `CHANGELOG.md`.
  - Each now names the **symbol** as well as the line, so the next drift is detectable by reading
    rather than silent. A mechanical gate for this class is still worth building: note that a naive
    exists-plus-EOF checker finds **none** of these, because every rotten citation still points at a
    real file and a line that exists - the rot is semantic, so the check must be symbol-anchored.
- **`config.yaml`'s `thresholds_path` was dead config - nothing read it.** The key was declared on
  `SestravConfig` (`src/core/config.py`) and set in `config.yaml`, and `SestravConfig.load()`
  populated it correctly, but no consumer ever asked for it:
  `functions/stage4_immunogenicity_scoring.py` located `optimal_thresholds.json` purely by
  convention from the model's directory. **Repointing `thresholds_path` was therefore a silent
  no-op, while repointing `model_path` silently moved the operating point with it** - the opposite
  of what the two keys look like they do.
  - Added `_resolve_thresholds_path(model_dir, thresholds_path=None)`, mirroring the preference
    order the adjacent `_resolve_calibrator_path` has always used: an explicit config path when it
    exists, else the model-directory copy. Threaded through `score_immunogenicity` and passed from
    `pipeline.py` exactly as `calibration_path` already was. `src/cli.py` passes neither and so
    keeps the model-dir fallback unchanged.
  - **No behavior change, asserted rather than assumed:** under the live config both the old
    convention and the new resolver return `models/optimal_thresholds.json`, so no score, threshold,
    or shortlist moves. Four unit tests pin the resolver, including that a configured-but-missing
    path does not mask the model-dir copy, and that an explicit path actually drives the cut.
  - **Deliberately NOT changed** (each a separate, owner-visible decision): `_apply_thresholds`
    still silently no-ops when no thresholds file is found, including under `freeze_mode`; and the
    shipped root file still carries the leakage-era `overall_f1: 0.7316` alongside
    `min_subgroup_f1: 0.0`, despite "maximize minimum subgroup F1" being its stated primary
    objective. Regenerating the root operating point remains out of scope here.

### Disclosed, not fixed
- **`results/freeze_status.json` is a tracked freeze-governance artifact that still carries the
  VOID H2 ratios and the placeholder input hash.** It records `h2.r10 = 0.9493670886075949`,
  `h2.r25 = 1.0207514198339884`, and `inputs.binding_matrix_path.sha256 = c7bb5ea1...` - the
  all-zeros placeholder D17 identified as the root cause, superseded by `78aa3db8...`.
  - **Scoped precisely:** its `h2.status` already reads `"NOT SUPPORTED"`, which matches the
    corrected decision, so the recorded *conclusion* is not wrong - only the two ratios and the
    hash. Its `"valid": true` is **not** a scientific certification either:
    `src/final_validation_report.py:217` hardcodes it on the success path (`false` only in the
    exception handler), so it means "the generator run completed". An earlier draft of this entry
    called it "certifying a void result as valid"; that over-read the field and is retracted.
  - It is written by the same `src/final_validation_report.py` as
    `results/final_validation_report.md`; because D17's remediation hand-patched that report rather
    than re-running the generator (a deliberate choice, to avoid disturbing unrelated certified
    artifacts the same script writes), this JSON was never updated and the omission was never
    disclosed until now.
  - **It evades the retracted-token sweep twice over:** `live_doc_globs` matches only `.md`, so no
    `.json` artifact is ever scanned; and the value is stored at full precision, so the 4-decimal
    token `0.9494` would not match as a substring even if it were.
  - Deliberately **not** hand-edited: `results/` artifacts are script-generated and the standing
    rule is regenerate-or-annotate, never hand-patch. Regeneration also rewrites
    `gold_standard_validation.csv` and `baseline_comparison.csv`, so it is not a drive-by edit.
    **Owner decision required:** regenerate, or annotate the file as historical.

### Changed
- **CORRECTED: `docs/paper.md` attributed the pooled figure's inflation to self-proteome hard
  decoys; the pooled background contains none** (`docs/claims_register.md` **D19**). No reported
  number changes - only the stated explanation for one.
  - **Verified false.** All 5,000 `self_proteome_decoy` rows in `data/immunogenicity_dataset_v5.csv`
    carry `is_quarantined = True` (crosstab, zero exceptions). `_filter_quarantined`
    (`src/train_classifier.py`) drops them right after the corpus read in `train_models`, and no
    unfiltered re-read occurs before the `training_results_mode{N}.csv` write - so both the OOF
    frame and the cited results file are computed on the filtered pool.
    `models/v5/rf_oof_predictions_mode31.csv` contains **zero** such rows.
  - **What actually dominates the pooled negative background:** of 27,534 negatives, **21,432
    (77.8%) are *Orthopoxvirus vaccinia*** - outside the nine-virus target panel, carried under
    `negative_origin = tested_negative` - and **3,112 (11.3%) are `allele_matched_nonbinder`**.
    Only **1,851** are same-pathogen `iedb_api` negatives (the 1,956 `iedb_api` rows include 105
    RSV); across the nine target viruses the real-negative pool is **2,201**.
  - **The vaccinia rows are NOT decoys, and an intermediate draft of this entry wrongly called
    them one.** They are genuine IEDB tier-1 assay records (21,432/21,432 `database_source = IEDB`;
    21,425 ELISPOT + 7 multimer), and `results/per_virus_eval_v5_mode31.csv` reports them as
    `n_neg_real = 21432, n_neg_decoy = 0`. They are *out-of-panel*, not synthetic - which is a
    different and weaker claim than the one being retracted, and the honest one. The error was
    inherited from `scripts/compute_pooled_honest_metric.py`'s docstring and
    `src/train_classifier.py`'s comments, both of which call this a "hard-decoy panel". **Both
    were corrected on 2026-08-10** - see the D19 closure entry above; this sentence described them
    as a deferred follow-up and that is no longer true.
  - **The allele-matched non-binders do not explain the pooled-vs-per-virus gap** and the
    manuscript no longer claims they do. They sit on both sides of the comparison and are a far
    larger share of most per-virus negative sets (Table 3b: 0.988 DENV, 0.920 HIV-1) than of the
    pooled background (0.113), so if anything they inflate the per-virus side more. Only the
    vaccinia term is both unique to the pooled side and measured.
  - **The inflation was already measured, in the very file the paper cites.** Re-slicing the same
    out-of-fold predictions without vaccinia gives AUC-ROC **0.670** (`rf_cv_mean_no_vaccinia`)
    against 0.814 pooled. The manuscript now quotes that instead of asserting inflation
    unquantified. AUC-PR moves the *other* way on the re-slice (0.606 -> 0.733) purely because
    dropping 78% of the negatives raises the base rate - a prevalence effect, flagged in the text
    so it is not misread as better discrimination.
  - **Two carry-caveats preserved:** the re-slice re-partitions existing predictions rather than
    refitting, so it is **not** the corpus-refit counterpart in `results/cv_leakage_audit.csv`
    (0.7693 / 0.7427); and the paper's own Methods already stated the correct composition
    elsewhere, so the document previously contradicted itself.
  - This is a D12-class error inverted - attributing decoy inflation to an absent decoy category
    while the categories actually present went unnamed, despite
    `scripts/compute_pooled_honest_metric.py`'s docstring already naming them.
  - **Three sites corrected, and five deliberately left alone.** A sweep found the same wording in
    a Discussion paragraph that also cites the pooled 0.81 figure, so three sites were corrected,
    not two. The manuscript's other self-proteome statements (LOO training set, corpus
    construction, dedup priority) are **correct**: `scripts/run_loo_cross_virus_v5.py` separates
    `source_type == "Self"` rows *before* the quarantine filter and applies that filter to viral
    rows only - "Self-proteome decoys are quarantined by design ... but are always included in
    training regardless of quarantine status". LOO training genuinely includes all 5,000. Only the
    `train_classifier.py` pooled path drops them, so the defect is scoped to the pooled figure.
- **`docs/paper.md` now carries the D18 mock-antigen-processing disclosure, closing D18 item (3)**
  - the last surface where mode-33 numbers were presented with no mock qualifier. It was scheduled
  for the manuscript submission pass; it is closed here instead because D19 required editing the
  same Results section, and leaving a known disclosure gap in a paragraph being touched anyway
  would have been indefensible. The D18 register row has been updated to say CLOSED rather than
  STILL OPEN, so the register and this changelog agree.
  The manuscript contained zero occurrences of "mock", "NetChop", or "TAPreg" before this change.
  Added: a bold-lead caveat after Table 1 in the paper's established house style (matching the D15
  precedents in the same section), a `MOCK features - see below` marker on Table 1's mode-33 row,
  and an amended Methods 2.5, which had justified excluding mode-33 from LOO on cache-*availability*
  grounds. That rationale is retained as the original reason - the mock status was established later
  and is now stated as an additional, after-the-fact ground, not retrofitted as the decisive one.
  The YAML front matter's blanket "all reported numbers are current and claims-audited" was
  reconciled: it now names **three** qualifications (Tier A, D18, D19) as an explicitly open list
  ("at least these three"), scopes the currency claim to the v5 cross-validation figures, and
  discloses that the Section 3.2 calibration ECE pair binds to a gitignored path and is therefore
  not reader-reproducible pending promotion to a tracked artifact.
- **The `feature_mode=33` code surfaces now disclose that the antigen-processing features are MOCK**
  (`docs/claims_register.md` **D18**, items (1) and (2), previously recorded there as open).
  - `src/features.py`: the `FEATURE_COLUMNS_33` comment no longer describes the features as
    "orthogonal ... (NetChop 3.1, TAPreg)"; it now states they are mock, not reproducible, and
    circular as evidence for proteasomal cleavage preference. `load_antigen_processing_cache`'s
    docstring carries the same warning.
  - `src/train_classifier.py`: **the `--feature-mode` `--help` text no longer reads
    "33 (31+NetChop+TAPreg)"** - the string a user sees now names the scores as MOCK and cites D18.
    The `--antigen-processing-cache` help, the printed `mode_label`, and `prepare_features_33`'s
    docstring were given the same disclosure.
  - **The "deterministic" mischaracterization is corrected** at all four
    `scripts/precompute_antigen_processing.py` sites and in `docs/limitations_statement_v1.md`.
    The mocks are stable *within* one process but differ *between* processes, because the
    generators use `hash()`, which CPython salts per process; re-running the script therefore does
    not reproduce the shipped `data/antigen_processing_cache.csv`. Empirically confirmed: three
    processes scoring `SIINFEKL` gave netchop means 0.43625 / 0.42000 / 0.42750. Those specific
    figures are single draws and will not themselves reproduce; the disagreement, not the values,
    is the evidence. The reproducible statement is analytic, not sampled: for `SIINFEKL` the
    generator's netchop mean has exact support `[0.36875, 0.45875]` on a `0.00125` lattice, and
    every value quoted in D18 lies inside it.
  - **Deliberately NOT changed:** `src/features.py`'s ESM-2 fallback also says "deterministic mock
    vector", and that claim is accurate - it seeds `numpy.random.default_rng` from a
    `hashlib.sha256` digest, which is stable across processes. Different generator, different
    guarantee; the correction is scoped to the antigen-processing mock only.
  - **Two further surfaces found by the adversarial pass on this very diff, in the retraction
    ledger itself** (`docs/claims_register.md` Section 3, Biological Accuracy Claims - both dated
    `2026-06-18`, both still listed as *approved corrective language*, neither carrying a mock
    note). Both are now amended in place under disclosure rather than silently edited:
    - The MHCflurry row asserted "NetChop 3.1 and TAPreg features provide **orthogonal**,
      tool-independent processing signals" - verbatim the clause D18 retracts, and verbatim the
      word this same commit removed from `src/features.py`. Retracted; first sentence retained.
    - The Rock & Goldberg 1999 row asserted "**Predicted via NetChop 3.1** (Nielsen et al. 2005)".
      The pipeline does not predict via NetChop 3.1. Retracted and replaced with language that
      also names the circularity.
    - Both rows' `Location` pointers (`docs/paper.md` Section 2.2; `docs/feature_glossary.md`) were
      already orphaned - neither file still contains the clause - and are now marked historical.
  - `docs/model_cards/rf_33feature_integrated.md`'s cache note called the values "high-fidelity
    mock scores calibrated to literature ranges"; that overstates a hand-coded rule plus `hash()`
    jitter and carried no non-reproducibility note. Corrected, and the pre-existing non-ASCII
    `U+2248` / `U+00A7` characters on that line replaced per `.claude/rules/encoding.md`.
  - `docs/proposals/2026_feature_upgrade_roadmap.md`'s D18 note still read "Code surfaces still
    present the features as real" in the present tense - the doc D18 names as its own remedy
    tracker. Updated, while keeping the distinction that **disclosure is not repair**: the mock
    still feeds `data/antigen_processing_cache.csv` and the actual fix remains Phase 1 step 8.
  - `scripts/batch_experiment_runner.py`'s mode-33 comment given the same note.
  - **A citation this commit invalidated itself:** D18 cited
    `scripts/precompute_antigen_processing.py:139,148` for the `mock_fallback=True` call sites, but
    this commit's own edits shifted them to 144/153. Re-anchored by symbol as well as line.
  - **KNOWN, PRE-EXISTING, NOT FIXED HERE - two `file.py:NNN` citations into
    `src/train_classifier.py` were already wrong before this commit** and are shifted further by
    it. Recorded rather than silently carried: (a) `docs/holdout_and_qc_policy.md:20` cites `:555`
    for the `GOLD_STANDARD_EPITOPES` exclusion; the real site is the `gs_mask = ...isin(...)` line
    (671 before this commit, 675 after). (b) `docs/claims_register.md` D15,
    `docs/proposals/2026_feature_upgrade_roadmap.md:227`, `scripts/audit_cv_leakage.py:307`, and an
    earlier `CHANGELOG.md` entry all cite `:781-786` for where `rf_oof_predictions*.csv` is
    written; the real writes are at 934/939 before this commit, 941/946 after. **(b) is
    load-bearing** - it is the code-traced provenance chain D15 uses to establish the Tier A
    leakage exposure, so a reader following it lands on unrelated code. Left for a dedicated pass
    rather than widened into this one. **This is the third instance of the same failure mode**
    (line-number citations rotting as files change), so the fix worth making is systemic: anchor
    by symbol, and add a CI check that resolves `file.py:NNN` citations in tracked docs the way
    `scripts/check_doc_commit_refs.py` already resolves commit SHAs.
  - Item (3) of D18 remains open, and is **understated in the original row**: `docs/paper.md`
    carries **three** mode-33 surfaces with no mock note (ablation prose, the Table 1 row, and the
    imputation-coverage sentence), not one. Note that exposure is a *missing disclosure, not a
    false attribution* - `docs/paper.md` never names NetChop or TAPreg. Deferred to the manuscript
    submission pass. No model, feature value, metric, or test-assertion changes anywhere in this
    entry; verified by an AST comparison showing all **four** touched `.py` files
    (`src/features.py`, `src/train_classifier.py`, `scripts/precompute_antigen_processing.py`,
    `scripts/batch_experiment_runner.py`) differ from their prior state only in
    comment/docstring/string-literal text.
- **RETRACTED: the H2 Tier A primary-hypothesis result R10 = 0.9494 is VOID** (`docs/claims_register.md`
  **D17**). The binding-only arm of that comparison was computed against an **all-zeros binding
  matrix**, so the denominator was a constant, not a baseline.
  - Three independent signatures of a constant score vector in `results/h2_tier_a_fold_metrics.csv`:
    for `method == binding_only_max`, `auc_roc` is **exactly 0.5000 with std 0.0 in all five folds and every subgroup that has both classes**; `auc_pr`
    equals the fold base rate exactly (146/192, 146/191); `ndcg_10` equals `auc_pr` to within
    floating-point rounding.
  - Root cause is `f360b90`, which introduced the placeholder. `37d1d67` (2026-06-18) is the FIX,
    and states it in its own message: "peptide_binding_matrix_v3.csv was an all-zeros placeholder
    since commit f360b90." The H2 artifact was generated against the placeholder.
  - **The integrated arm was degraded by the same file** - `prepare_features_30` sources 10 of its
    30 features from it, so the "integrated model" ran with 10 dead features.
  - **Corrected by controlled re-run** (identical script, data, model and seed; only the matrix
    differs): binding-only AUC-ROC **0.6636** / ISSR@10 **0.8947**; integrated AUC-ROC **0.7563** /
    ISSR@10 **0.9474**; **R10 = 1.0588** (95% CI [0.9778, 1.1220], sign-flip p = 0.1875);
    **R25 = 1.0331**. Verification control: re-running against the `f360b90` matrix reproduces the
    pre-regeneration `results/h2_tier_a_summary.csv` **byte-for-byte**, establishing the matrix as
    the sole cause.
  - **The H2 decision is UNCHANGED: NOT SUPPORTED.** Note the correction moves R10 *upward* past
    1.0 - the void artifact reported a lower ratio than a valid measurement does.
  - **Not a D15 leakage defect.** The v3 corpus is 1,004 rows / 1,004 unique peptides, so
    `StratifiedKFold` is already peptide-disjoint on it and a grouped re-run would be a no-op.
  - `docs/Wet_Lab_Protocol_v1.md` pre-registered 0.9494 as its powering prior and is marked
    **do not run, submit, or cite until re-derived**. A separate error corrected in the same pass:
    three of this repo's own annotations had conflated the H2 computational gate (R10 >= 2.0) with
    that protocol's Primary criterion, which pre-commits **no fixed multiple** and instead requires
    R10 > 1.0 with a bootstrap CI lower bound above 1.0. The corrected figure passes the first and
    fails the second (0.9778), so the disclosed null stands - on the protocol's own terms.
  - **Regenerated 2026-08-10:** `results/h2_tier_a_*` and `results/final_validation_report.md` now
    carry the corrected values, reproducing the figures above byte-for-byte. The corrected R10 is
    bound in the local claims manifest against `results/h2_tier_a_summary.csv`.
- **DISCLOSED: the `feature_mode=33` antigen-processing features are MOCK, not NetChop 3.1 / TAPreg
  output** (`docs/claims_register.md` **D18**). `scripts/precompute_antigen_processing.py` calls both
  predictors with `mock_fallback=True`, which short-circuits before any network call.
  - **The values are not reproducible**: the generators use `hash()`, which CPython salts per
    process and the original seed was never recorded, so the shipped
    `data/antigen_processing_cache.csv` cannot be reproduced.
  - **Two biological inferences retracted.** `netchop_score` ranking first was cited as "confirming
    independent proteasomal processing signal" and as consistent with Rock & Goldberg 1999. The
    generator **already assumes** hydrophobic/basic C-terminal cleavage preference, so its
    importance cannot be evidence *for* that mechanism - the inference is circular. Same error
    class as D13. Corroboration: under the v5 peptide-grouped splitter, mode 33 exceeds mode 31 by
    **+0.0027 AUC-PR** (0.6085 vs 0.6058).
  - **"Best v3 model" and "recommended for production" designations withdrawn** across
    `docs/data_registry.md` AD-9 (a LOCKED row, amended under disclosure rather than silently
    edited), `docs/model_evaluation_summary.md`, and both the 31- and 33-feature model cards.
    **`mode_31` remains the canonical production track.**
  - Disclosure was partial, not absent - `docs/limitations_statement_v1.md` and the 33-card's cache
    note named the mock - but seven further tracked surfaces presented it as real, including
    `README.md` and `ARCHITECTURE.md`, and the 33-card's own feature schema cited Nielsen 2005 and
    Peters 2003 for values neither tool produced. All are now corrected.
  - Still open: `docs/limitations_statement_v1.md` and `scripts/precompute_antigen_processing.py`
    describe the mock as "deterministic", which the per-process salting refutes.
- **BREAKING (reported metrics): peptide-level CV leakage remediated and every certified v5
  number re-baselined** (`docs/claims_register.md` D15, Phase 0 of
  `docs/proposals/2026_feature_upgrade_roadmap.md`). The leakage disclosed in the audit below
  is now closed at the source rather than only documented.
  - `src/ml_utils.py` gains **`PeptideGroupedKFold`**, a sibling of `MultiStratifiedKFold`
    sharing its composite stratification key (label|origin|supertype|length) but holding every
    row of a given peptide in exactly one fold. A separate class rather than a `groups=` kwarg,
    because `StratifiedKFold` raises on sparse strata while `StratifiedGroupKFold` only warns -
    one `min_stratum_size` knob cannot mean the same thing for both.
  - **`_bin_origin` now recognizes `iedb_api` as a real negative**, matching
    `scripts/build_dataset_v5.py` and `scripts/analyze_hiv1_binding_bias.py`. `src/ml_utils.py`
    was the only site that did not. This was the root cause of the silent label-only
    stratification fallback firing on *every* v5 split: it moved the rarest composite stratum
    from 2 rows to 6. Fixing it is metric-neutral (ungrouped AUC-PR 0.8347 -> 0.8343, inside
    fold noise), which is how it was verified in isolation before anything else landed.
  - The single global `min_stratum_size` pre-check is replaced by an explicit **coarsening
    ladder** (label|origin|supertype|length -> ... -> label) that records the rung actually used
    as `.stratification_components_` and prints it, and **raises** rather than degrading further
    if even label-only stratification is too sparse. Degradation is no longer silent.
  - `MultiStratifiedKFold(shuffle=False)` no longer raises (`random_state` was forwarded to
    sklearn unconditionally).
  - `src/train_classifier.py` gains `--cv-group-by {none,peptide}` (**default `peptide`**) and
    `--no-fold-impute`, threaded through `train_models`/`_cross_validate` as keyword-only
    parameters. The **Python API defaults to legacy behavior** (`cv_group_by=None`,
    `fold_impute=False`) so `src/cli.py`, `src/bias_skew_finalization.py`,
    `scripts/regenerate_shareout_pngs.py` and the Colab notebook are unchanged; the **CLI
    defaults to honest**, because the CLI is what produces certified artifacts.
  - **The shipped model artifact and its operating point both changed, and both were also
    stale before this run.** `models/v5/model_artifact_checksums.json` and
    `models/v5/optimal_thresholds.json` were last written by `58bbc15` (2026-06-26), i.e. against
    the *earlier, smaller* v5 build - not the 35,597-row corpus shipped since `d3972f7`
    (2026-07-05). Regenerating them here corrects that drift as well as reflecting the grouped
    splitter. `rf_31feature_integrated.joblib` moves sha256 `a5cec4c7...` -> `e2acd332...` and
    62,716,233 -> 128,089,513 bytes (the size roughly doubles because the 2026-06-26 manifest
    described a model trained on the pre-merge corpus, so this is a corpus-driven change, not a
    hyperparameter one; `n_estimators` is 200 in both). **The v5 operating-point ledger moves materially, and
    a THIRD copy is stale:** `models/v5/optimal_thresholds.json` goes from `threshold` 0.25
    (precision 0.730 / recall 0.846 / F1 0.784) to **0.329 (precision 0.516 / recall 0.664 /
    F1 0.581)**. That F1 drop is the expected consequence of picking an operating point on honest,
    non-leaked out-of-fold predictions - the earlier one was tuned on leakage-inflated scores.
    **This is NOT the file production reads.** `config.yaml:86` (`thresholds_path`) points at
    `models/optimal_thresholds.json`, and `functions/stage4_immunogenicity_scoring.py` resolves
    `optimal_thresholds.json` under the configured model dir - i.e. the root copy, which this run
    did not regenerate and which sits at a third value again (`threshold` 0.325 / F1 0.732).
    **Production scoring is therefore still on a pre-remediation operating point.** Regenerating
    the root copy is deliberately left as a separate, owner-visible step rather than folded into
    this re-baseline, because it changes shortlist sizes for every downstream consumer.
  - **Antigen-processing median imputation moved inside the fold** for feature modes 33/35
    (`src/features.py` `load_antigen_processing_cache(..., impute=False)` plus
    `antigen_processing_cache_medians()`). The whole-cache median leaked held-out peptides into
    the training features. Modes 21/30/31/50/166 are provably unaffected - mode-31 OOF scores
    are bit-identical across the flag (max abs diff 0.0). The final full-pool refit still uses
    whole-cache medians, since the full pool is that model's own training set, so the shipped
    `.joblib` is unchanged by this repair.
  - **Certified ledgers regenerated** under the grouped splitter. `models/v5/training_results_mode31.csv`:
    RF **AUC-ROC 0.9429 -> 0.8137**, **AUC-PR 0.8312 -> 0.6058** (XGB 0.9096 -> 0.8093,
    0.7672 -> 0.5597). `models/v5/training_results_ablation.csv` retrained across modes
    21/31/33/35 (AUC-PR 0.8158 -> 0.5047, 0.8312 -> 0.6058, 0.8313 -> 0.6085, 0.8311 -> 0.6069).
    `results/per_virus_eval_v5_mode31.csv` mean AUC-ROC **0.751 -> 0.658**.
    `results/pooled_honest_same_pathogen.csv` **0.712 -> 0.6015** ROC / 0.917 -> 0.8711 PR.
    `results/loo_binding_confound_decomposition.csv` regenerated (R1/R2/transfer-gap means
    0.751/0.654/0.191 -> 0.658/0.551/0.088); its `loo_cross_virus` column is unchanged at 0.463,
    confirming that column was never exposed to this defect.
  - **New `*_no_vaccinia` columns** on the training-results ledgers: an OOF re-slice with the
    `Orthopoxvirus vaccinia` bloc (77.8% of active negatives) dropped from validation, at zero
    extra model fits. Mode-31 RF AUC-PR 0.7328 / AUC-ROC 0.6702. **This is not the same quantity
    as a corpus refit without vaccinia** (`results/cv_leakage_audit.csv`,
    `peptide_grouped_splitter_no_vaccinia`, AUC-PR 0.7693 / AUC-ROC 0.7427) - the re-slice scores
    the shipped model on target-virus rows, the refit answers how much of the headline the bloc
    accounts for. Both move the same way (AUC-PR up, AUC-ROC down) but by different magnitudes,
    so they are not interchangeable. AUC-PR rises partly mechanically, since dropping vaccinia
    lifts the validation base rate from 0.226 to 0.568; AUC-ROC falls because the vaccinia decoys
    were trivially separable and removing them leaves a harder negative set (AUC-ROC is
    prevalence-invariant, so the base rate does not explain that fall).
  - **Tier A deliberately not re-run.** Only 414 of its 704 peptides resolve to an active v5
    row, so a grouped re-run would score a smaller, non-comparable field rather than correct
    this one. `results/table3_tier_a_metrics.csv` (0.828) stands as a labeled 2026-05 /
    30-feature / unweighted / 200-tree historical figure (D16), and
    `scripts/verify_tier_a_provenance.py` still reproduces its certified cells.
  - `scripts/audit_cv_leakage.py` gains `production_splitter_repaired` and
    `production_grouped_splitter` arms that use the real `src.ml_utils` classes with full
    composite-key arguments. The two original D15-anchor arms are **unchanged in code and still
    reproduce 0.8347 / 0.6092 to 4 decimal places**; the repaired arms measure 0.8343 / 0.6079,
    so the 0.0004-0.0014 deltas are demonstrably noise against the 0.0065-0.0229 fold standard
    deviations. (The anchor arms are not bit-frozen: `_fold_overlap` passes the full composite-key
    arguments, so the `_bin_origin` fix above shifted its per-fold overlap percentages, and the
    reported overall peptide overlap moves 71.02% -> 71.05%. The AUC cells move only in the 7th
    decimal.)
- **GNN promotion Gate 1 re-anchored from AUC-PR >= 0.85 to >= 0.65** under a peptide-grouped
  splitter (`src/verify/promote_gnn.py` `GATE1_AUC_PR_MIN`, mirrored in `ROADMAP.md` and
  `docs/architecture/gnn_models.md`). The 0.85 threshold was set against the now-retracted
  ungrouped RF baseline of 0.8312; against the certified 0.6058 it was unreachable rather than
  ambitious. The `ROADMAP.md` gate list also previously omitted Gate 5 entirely and stated
  Gate 2 as `< 0.02` where the code enforces `<= 0.02`; both corrected against the code.
  The **pathogen-expansion gate** in `ROADMAP.md` is likewise re-anchored from `AUC-PR >= 0.80`
  to `>= 0.65`, and both gates now state the splitter explicitly.
- Public metric disclosures updated to the re-baselined figures with their retracted
  predecessors named: `README.md`, `docs/paper.md` (Sections 2.4, 3.2-3.5, Tables 1/2/2b/3/3b),
  `docs/model_evaluation_summary.md`, `docs/validation_summary.md`, `ARCHITECTURE.md`,
  `USAGE.md`, `docs/data_registry.md`, `figures/captions.md`, and the live
  `/model-info` (`api/main.py`) and Streamlit (`app/demo.py`) contamination disclosures.
  `docs/model_cards/ann_30feature.md` and `xgb_30feature.md` gain the D15 disclosure they
  previously lacked entirely (their v3-era numbers are labeled as not re-measured).
  `ARCHITECTURE.md`'s description of the Tier A figure as "v3-era" is corrected to the
  720-row root corpus at `69e0e5c` per D16's second correction.

### Added
- **Peptide-level cross-validation leakage audit** (`scripts/audit_cv_leakage.py`, output
  `results/cv_leakage_audit.csv` with a provenance sidecar recording the dataset SHA-256,
  seed, and estimator count). `src/train_classifier.py` cross-validates with
  `MultiStratifiedKFold` (`src/ml_utils.py`), which accepts a `peptides=` argument but uses
  it only to bin length for stratification - never as a fold group. The v5 corpus is
  deduplicated on `(peptide, hla_allele)` rather than on peptide, and every
  `feature_mode=31` feature is a pure function of the peptide string, so rows sharing a
  peptide are feature-identical and land on opposite sides of a fold boundary: **71.0% of
  held-out test rows have their exact peptide present in that fold's training set** (as first
  measured; the same figure reads 71.1% after the Phase 0 `_bin_origin` fix changed fold
  composition - see the Changed entry above). Holding
  the RF configuration fixed at production's own settings (`n_estimators=200`,
  `random_state=42`, `class_weight=balanced`) and changing only the splitter moves AUC-PR
  from 0.8347 to 0.6092 (+0.2255, +37.0%). The production-splitter arm reproduces the
  certified ledger cell (`models/v5/training_results_mode31.csv`, AUC-PR 0.8312) to within
  0.0035, so the gap is attributable to leakage rather than to a modeling difference.
  Recorded as `docs/claims_register.md` D15.
- **`docs/proposals/2026_feature_upgrade_roadmap.md`** - ranks seven 2026-era feature-schema
  upgrades against the leakage-corrected baseline, and proposes a Phase 0 that repairs the
  evaluation harness (peptide-grouped splitter, fold-disjointness test, in-fold imputation)
  before any feature work is measured. The honest feature-mode deltas span -0.0037 to +0.0096,
  roughly 23x smaller than the leakage inflation itself.

### Fixed
- **Tier A measured under both splitters.** [**SUPERSEDED IN PART - read the two corrections
  below this bullet's original text before citing anything in it.** Its original heading said
  Tier A's "real defect is reproducibility, not leakage"; that framing, and the v3-generation
  detail in this bullet, were retracted by later commits recorded further down this same section.
  The current account: 0.828 is a **2026-05, 30-feature, unweighted, 200-tree** measurement whose
  provenance IS established - the defect is a wrong label, not irreproducibility. The corpus was
  the 720-row root `immunogenicity_dataset.csv` at `69e0e5c` (recoverable from history, not
  tracked at HEAD), **not** `data/immunogenicity_dataset_v3.csv`; and it used
  `prepare_features_30` with `n_estimators=200`, **not** `prepare_features_31` with 500. The
  v5-coverage figures below were also mispaired and are corrected in the second note. Original
  text retained for the record:]
  The SESTRAV arm of the Tier A external benchmark is the `rf_oof_score` column, which traces
  to the same ungrouped OOF output as everything else (`src/train_classifier.py` (the `rf_oof_predictions*.csv` writes in `train_models`) ->
  `src/prepare_external_validation_inputs.py:100` -> `scripts/run_tier_a_benchmarks.py:269`),
  so Tier A was never an independent held-out field. Measured on the 414 field peptides
  resolvable to an active (non-quarantined) v5 row, changing only the splitter moves AUC-PR
  0.8932 -> 0.8756 (-0.0176),
  AUC-ROC 0.5924 -> 0.5680, ISSR@10 1.0000 -> 0.9024 - an order of magnitude milder than the
  CV metrics above. Two limits are recorded: that subset is not representative of the certified
  n=704 field (positive rate 0.838 vs 0.696), so only the within-subset delta is interpretable;
  and those 414 are the maximally-exposed peptides, so the delta reads closer to an upper bound.
  Recorded as `docs/claims_register.md` **D16**: the certified 0.828 is a **v3-era** measurement
  presented as the canonical v5 `mode_31` result. The number is sound for what it is; its label is
  wrong. All 704 field peptides resolve against the tracked `data/immunogenicity_dataset_v3.csv`,
  while only 414 exist in v5 and 236 exist in neither v4 nor v5. Re-running from tracked v3 inputs
  (`prepare_features_31`, 5-fold `StratifiedKFold`, RF 500 estimators) recovers 704/704 coverage
  and matches the certified ISSR@10 (0.8429) exactly, with AUC-ROC within 0.004 and AUC-PR within
  0.009; mean deviation from the stored column is 0.084 against v3 versus 0.371 against v5,
  confirming the generation. **The v5 mode-31 model has never been evaluated on the full Tier A
  field and cannot be** - 290 of its peptides are absent from v5 - so re-running Tier A under v5
  yields a different, smaller (n=414) field whose numbers are not comparable to the published
  ones. That is a replacement, not a refresh.
  [**Second correction (2026-08-09), on the v5-coverage figures in the paragraph above.** It read
  "only 414 exist in v5 and 236 exist in neither v4 nor v5" - arithmetically impossible, since
  704 - 414 = 290, not 236; the pairing silently required 54 v4-only peptides and there are zero.
  The two figures measure different things. Measured directly: **468** of the 704 exist somewhere
  in v5 and **236** exist in neither v4 nor v5 (none are v4-only); only **414** resolve to an
  **active, non-quarantined** v5 row, the other **54** appearing solely in quarantined rows. So
  236 + 54 = **290** unscoreable, and n=414 remains the correct evaluable field size - the
  conclusion of the paragraph stands, only its justification was misstated.]
- **Whole-repo coverage re-measured: 47.88%, not the published 34.37%.** `ROADMAP.md`,
  `docs/security_compliance.md`, and `docs/claims_register.md` all carried "34.37%
  (branch-inclusive, measured 2026-06-22) - a hair under the floor" against a `fail_under=35`
  gate. Both halves were wrong: the number by 13.5 points, and the characterization in the
  opposite direction (the gate reports "Required test coverage of 35.0% reached"). All three
  locations updated with an explicit supersedes note.
- **Wet-lab pre-registration retargeted to the production model and its success bar grounded.**
  `docs/Wet_Lab_Protocol_v1.md` was written against the GNN - a deferred, GPU-gated research
  track never promoted through `src/verify/promote_gnn.py` - so it could not have been run
  against the system that exists; it now targets the RF mode-31 production scorer. Its
  pre-committed `R10 >= 2.0` success criterion exceeded SESTRAV's own certified computational
  analog of that ratio (0.9494, a null) by more than 2x; the criterion is now `R10 > 1.0` with
  a bootstrap CI lower bound above 1.0, the disclosed prior stated up front, and the expected
  null pre-registered as a publishable result. Peptide selection moved from absolute probability
  thresholds to rank, since the model emits a triage score rather than a calibrated probability.
  Added an explicit status banner: not funded, not IRB-approved, not scheduled.
- **The two metrics cited as the leakage-honest corrective are themselves inflated.**
  `results/per_virus_eval_v5_mode31.csv` (per-virus mean AUC-ROC 0.751) and
  `results/pooled_honest_same_pathogen.csv` (Def A pooled AUC-ROC 0.712) have been cited as
  the honest antidote to the retracted pooled 0.9368/0.7678 (`docs/claims_register.md` D12).
  Both are computed downstream of `models/v5/rf_oof_predictions_mode31.csv`, which the same
  `MultiStratifiedKFold` path writes - not an independent splitter, as had been assumed.
  Measured under a matched peptide-grouped splitter: the per-virus mean falls to
  0.6587 +/- 0.0908 (+0.0925, +14.0% inflation) and the pooled honest figure to 0.5989
  (+0.1135, +19.0%). Both production-splitter reproductions match their certified values
  almost exactly, validating the measurement. The LOO cross-virus table
  (`results/loo_cross_virus_v5_clean.csv`) is confirmed genuinely unaffected: it trains a
  fresh model per held-out virus with explicit virus-level partitioning and never uses
  `MultiStratifiedKFold`. No published number is changed here - this records the finding and
  its scope; the re-baseline itself is a separate, owner-sequenced decision.
- **False held-out-cohort claim removed from the served API and demo.** `api/main.py`'s
  `/model-card` endpoint and `app/demo.py`'s disclosure expander both claimed SESTRAV "is
  evaluated on a held-out independent validation cohort (SARS-CoV-2 and Influenza A) ... with
  no overlap with the training set" - inside the field literally named
  `contamination_disclosure`. Both are trained viruses in v5 (SARS-CoV-2 n_pos 2473, IAV
  n_pos 342, `results/per_virus_eval_v5_mode31.csv`); stranded v4-era copy that survived the
  v5 migration and contradicted `api/main.py`'s own correct training-set description three
  lines above it. Replaced with the true v5 posture and the D15 splitter disclosure.
- **16-peptide holdout scope corrected, not "Tier A/B quarantine."** `GOLD_STANDARD_EPITOPES`
  (`src/iedb_data_loader.py:24`) excludes exactly 16 peptides from training
  (`src/train_classifier.py` (the `gs_mask` gold-standard exclusion in `train_models`)); `docs/holdout_and_qc_policy.md`, both RF model cards, and
  `docs/paper.md` all overstated this as "Tier A and Tier B Gold Standard validation peptides
  ... strictly excluded" / "permanently quarantined." 414 of the 704 Tier A peptides are
  present in the v5 training corpus (D16). This is the root cause of D16's "the SESTRAV Tier A
  arm was never independent"; corrected at all four locations, which also supplies the
  replacement language the "conservative" framing fixes below depend on.
- **RETRACTED SHAP 60/40 split removed from `ARCHITECTURE.md`.** `ARCHITECTURE.md:179-181`
  still restated the pre-D13 "roughly 60% MHC binding / 40% TCR-contact" figure verbatim,
  contradicting `README.md:157` ("Retracted") and the register itself. Replaced with an
  explicit RETRACTED note citing D13 and the all-zero `bind_*` SHAP columns.
- **QC policy amended to require peptide grouping.** `docs/holdout_and_qc_policy.md` mandated
  "strict stratified 5-fold cross-validation" - the exact splitter D15 indicts - as binding
  policy. New certified runs must now group by peptide; documented as a hard prerequisite for
  any future re-baseline (H6).
- **Hardcoded baseline printout corrected to read its cited source.**
  `scripts/compute_ann_baseline_summary.py:194` printed a literal `RF AUC-ROC 0.7268 AUC-PR
  0.8317` attributed to `training_results.csv`, which actually holds 0.9429/0.8312. Now reads
  the RF row from `--results-file` at call time, with an explicit D15 caveat printed alongside.
- **Test-count claim corrected.** `docs/security_compliance.md:21` said "200+ tests" against
  an actual, much higher floor (1,200 elsewhere, 1809 current); now states "More than 1,200
  pytest test cases" with a dated correction note.
- **`results/pooled_honest_same_pathogen.csv` tracked.** Cited by D15, its Section-4 row, this
  CHANGELOG, and the roadmap as the binding source for the 0.712 honest pooled same-pathogen
  figure, but gitignored - a reader could not open the evidence for a number the claims
  register asserts. Same defect class `ad65a21` closed for `External_Validation_Sign_Off.md`
  hours earlier, reintroduced by the same day's own D15 edit. Un-ignored (277 bytes, one row,
  regenerates from `scripts/compute_pooled_honest_metric.py`).
- **D16 corrected a second time: 0.828 is 30-feature, unweighted, 200 trees - not 500.**
  `results/external_validation_input.csv`'s only commit (`f360b90`, 2026-05-23) predates
  `feature_mode=31`'s introduction (`27cdc61`, 2026-06-18) by 26 days, so 0.828 cannot be a
  mode-31 measurement; it belongs to the 30-feature track. All 704 stored `rf_oof_score`
  values are exact multiples of 1/200 (704/704, vs 362/704 for 1/500), fingerprinting
  `n_estimators=200` and refuting the 500 the first D16 pass asserted. Root cause of the
  original mislabel identified: the 31-feature v3 weighted CV mean (0.8275628) sits 0.0002
  from this 30-feature field metric (0.8277666) by coincidence; treating that coincidence as
  agreement licensed collapsing two different measurements into one headline. `README.md`,
  `USAGE.md`, and both RF model cards' "conservative by construction"/"strictly out-of-fold"
  framing withdrawn (not softened) at this point - D15 shows the leakage runs toward SESTRAV,
  so the framing was backwards. `README.md`'s "self-proteome Gate 1 AUC-PR 0.8897" also
  corrected here: not a self-proteome metric (no such artifact exists; "Gate 1" is a GNN
  promotion threshold), and not the current corpus (which reports 0.8312).
- **The 0.828/0.8897 relabel propagated from footnotes to the public headline.** The corrections
  immediately above landed as fine-print footnotes without updating the text three lines above
  them: `README.md`'s top comparison table (:30) and Tier A table (:105) still read
  "full_31/mode_31" for the 30-feature figure; `ARCHITECTURE.md:164-165` asserted "canonical
  full_31 AUC-PR 0.828 (OOF)" with no D15/D16 disclosure at all; `docs/model_evaluation_summary.md`
  and `docs/validation_summary.md` still stated "Self-proteome Gate 1 AUC-PR 0.8897" as live
  fact; `docs/data_registry.md`'s v5 Build Log was missing the row for the `d3972f7` rebuild
  (2026-07-05) that produced the shipped 51,185/35,597-row corpus entirely (H7). All propagated
  to match the corrected internal record. Separately, `docs/model_evaluation_summary.md:76-80`
  had called the 0.8276-vs-0.828 coincidence a "reconciliation" - the documented root cause of
  the original mislabel - and is now corrected to state plainly that they are not the same
  measurement (H2). `rf_31feature_integrated.md`'s 500-vs-200-estimators mismatch against the
  tracked `src/train_classifier.py` (H3) is flagged rather than resolved, pending archaeological
  confirmation of which generation "500" ever described.
- **"Conservative by construction" / "strictly out-of-fold" framing retired from `docs/paper.md`**
  (Abstract, Section 2.4 Evaluation Methodology, Section 3.5 Table 4 caption and narrative) -
  the 7 locations in this file the register had flagged as still carrying the withdrawn framing
  after `README.md`/`USAGE.md`/the model cards were already fixed. Section 3.5's SESTRAV row
  relabeled per D16 to match the front-door fix above. The paired-bootstrap paragraph now
  discloses that both significance tests were computed on the leakage-inflated OOF arm and that
  the binding-only result (p=0.04) clears zero by only 0.0018 within a 0.069-wide CI, so should
  be treated as unconfirmed pending a peptide-grouped re-run.
- **A leakage-explained gap misread as biology, retracted.** `docs/paper.md` Section 3.4 stated
  the pooled model (AUC-ROC 0.943) exceeding most per-virus values "indicat[es] that cross-viral
  training provides complementary discriminatory signal." Pooled AUC-PR is +37.0%
  leakage-inflated versus the per-virus mean's +14.0% (D15); the gap is substantially explained
  by that differential inflation rather than by validated cross-viral signal. Inference
  retracted explicitly, not deleted silently.
- **`rf_33feature_integrated.md` given the D15 leakage disclosure the other two RF model cards
  already carried** - it was the only one with zero corrective language despite reporting the
  same stratified-but-ungrouped 5-fold OOF metric. Scoped honestly: only mode-31's
  splitter-leakage delta has been directly measured (+0.2255 AUC-PR, `results/cv_leakage_audit.csv`);
  mode-33's own delta has not. Same 500-vs-200-estimators flag added as `rf_31feature_integrated.md`.
- **`docs/claims_register.md` D12 marked superseded-in-part by D15.** D12's own "honest"
  corrective figures (per-virus mean 0.751, pooled same-pathogen 0.712) are themselves
  peptide-leakage-inflated - D15 measures them at 0.6587 and 0.5989 under a peptide-grouped
  splitter - but D12 carried no cross-reference to that fact. Added the same SUPERSEDED banner
  pattern already used on Section 2's D15-affected rows.
- **"Honest" disambiguated at four locations that used it for a decoy-corrected figure without
  flagging it is not also leakage-corrected.** README.md, `docs/model_evaluation_summary.md`,
  `docs/validation_summary.md`, and `docs/model_cards/rf_31feature_integrated.md` all stated
  "the honest pooled same-pathogen ROC is 0.712" (or the 0.751 per-virus mean) with no leakage
  caveat nearby. Added "(decoy-corrected)" qualifiers and the D15 peptide-grouped reproduction
  values at all four.
- **`CITATION.cff`'s abstract described "Version 2" GNN work as "graph neural network
  benchmarks (GCN, GAT, bipartite peptide-allele)."** Per D1 the actual v2.3 GNN research track
  is GINEConv+ESM-2 (`GraphPredictorV2`); GCN/GAT/bipartite are a real but separate historical
  benchmark. Citation metadata now names the shipped architecture and scopes the older
  benchmark as retained-for-reference.
- **README's citation of `results/external_benchmark_comparison.md` corrected to match its
  actual content.** README described it as methodology for the certified 5-tool Tier A
  comparison; it is dated 2026-05-22, compares only SESTRAV against one binding-only baseline,
  and itself carries the pre-D16 "31-feat" mislabel. Both citing locations now describe it as
  historical reference, not a citable source for the current comparison.
- **Wet-lab protocol's disclosed prior traced to a different model generation than the one
  under test.** `docs/Wet_Lab_Protocol_v1.md`'s R10 success criterion is grounded in a 0.9494
  prior from the Tier A family, whose SESTRAV arm (D16) is a 30-feature, 200-tree, 2026-05
  measurement - not the production mode-31 model the protocol's own Objective and Section 2
  target for the physical assay. Added an addendum disclosing the mismatch; not resolved, since
  no mode-31-substrate R10 exists without a grouped-splitter re-run (D15).
- **README's comparison table claimed "pip-installable"; the package is not yet on PyPI**
  (`ROADMAP.md`: "Installation is from source today"), and README's own Quick Start already said
  so. Table cell corrected.
- **Stale "v4" reference in README's Paradigm 1/2 framing corrected to "v5"** - the
  generalization-set corpus moved to v5 well before this sentence was last touched; Paradigm 2
  is explicitly labeled v5 two sections below it.
- **An arithmetically impossible coverage claim in the certified claims register, corrected.**
  `docs/claims_register.md` D16 stated "Only 414 of the 704 peptides exist in v5 and 236 exist in
  neither v4 nor v5" - both cannot hold, since 704 - 414 = 290, not 236, and the pairing silently
  required 54 v4-only peptides where there are **zero**. The figures measure different things and
  had been conflated. Measured directly, and matching what `scripts/verify_tier_a_provenance.py`
  has printed all along: **468** exist somewhere in v5; **236** exist in neither v4 nor v5;
  **414** resolve to an **active, non-quarantined** v5 row; the other **54** appear only in
  quarantined rows; so **290** cannot be scored by the v5 model. Only the register and the
  script's docstring were wrong. n=414 remains the correct evaluable field size, so every
  downstream conclusion is unchanged. The "exists in v5" versus "resolves to an active v5 row"
  distinction was propagated to `docs/paper.md`, the feature-upgrade roadmap, D15's Tier A subset
  wording, and this file.
- **Two claims-register rows still asserted retired numbers as live fact.** The D3 (GNN promotion
  gates) row cited the D12-retracted 0.7678 without retraction framing, asserted a "Gate 1
  self-proteome 0.8897" that does not exist (Gate 1 is a GNN promotion threshold,
  `src/verify/promote_gnn.py:8`; no self-proteome artifact exists in this repository), and
  miscomputed both of its differences (0.85 - 0.7678 = 0.082, not 0.12; 0.85 is 0.04 *below*
  0.8897, not above). The Section 4 row attributed 0.8897 to "commit e6aafe2" - `git log --
  models/v5/training_results.csv` returns only `58bbc15` and `7656b8f`, and `e6aafe2` touched a
  different path whose values at that commit are the D12-retracted 0.7678/0.9368. Both rewritten.
  D15's row also still carried the retracted D16-v2 text ("a v3-era result ... reproduces from
  tracked v3 inputs at 704/704"), superseded by `dd5a356` but missed there at the time.
- **The withdrawn "conservative out-of-fold" framing survived in two tracked source files**, which
  the B3 pass never reached because its enumeration listed only markdown.
  `src/external_validation_fairness.py` emitted "SESTRAV RF uses 5-fold out-of-fold predictions
  (conservative)" into a report section headed "## Mandatory Disclosure", and
  `src/external_validation_finalize.py` emitted "SESTRAV RF uses conservative OOF scoring" into
  every MCDA verdict block - the same defect class as the API/demo `contamination_disclosure`
  (fixed earlier) and the hardcoded baseline print (H8). Both now state the D15-accurate
  direction: the arm is optimistic, not conservative.
- **`docs/model_cards/rf_30feature.md` compared 0.864 against 0.828** - the 31-feature ablation
  mean (n=1,004) against this model's Tier A *field* metric (n=704). The like-for-like counterpart
  is 0.825 (`combined_30`, same ablation table), as `docs/claims_register.md` D8 already stated.
  This is the card D16 singles out as correct all along, so the mispairing was unusually likely to
  be trusted. Its "Metrics on v3 dataset" heading was also stale: the corpus is the 720-row root
  `immunogenicity_dataset.csv` at `69e0e5c`, not the 1,004-row v3 dataset.
- **`README.md`'s Track Definitions bound the 21-feature row to 0.772**, which is `physico_20`'s
  ablation value; `sestrav_21` is **0.784**. All six rows re-checked against their source.
- **`ARCHITECTURE.md` and `USAGE.md` quoted the per-virus mean 0.751 with no splitter
  disclosure**, which D15 makes mandatory wherever that figure appears. Both now record that it
  reproduces at 0.6587 under a peptide-grouped splitter (+0.0925, +14.0%).
- **This CHANGELOG's own D16 bullet, and a released `[2.0.x]` entry, still asserted retracted
  accounts** (respectively the v3-corpus/500-estimator first draft, and "Self-proteome Gate 1
  AUC-PR 0.8897 is unaffected"). Both marked superseded in place with the corrected account rather
  than rewritten, since released entries record what was reported at the time.

---

> **Everything below this line was staged as `## [2.1.0] - 2026-08-07` and has been folded back
> into `[Unreleased]` (2026-08-10).** No `v2.1.0` tag was ever pushed, and `CITATION.cff`'s
> `date-released: 2026-08-07` predated the D15/D16 disclosure, so releasing it would have
> shipped a version that structurally could not disclose findings already public on `main`.
> The version identifier is back at `2.0.3` across `pyproject.toml`, `CITATION.cff`, `README.md`,
> `USAGE.md`, and `api/main.py`. These entries are unreleased, not retracted - subsection
> headings therefore repeat within `[Unreleased]`, and the block above this divider is the
> Phase 0 work of 2026-08-10.

### Security
- **PredIG Docker image pinned off the mutable `:latest` tag**
  (`scripts/run_predig_wrapper.py`). This wrapper was the last of four PredIG call sites
  still pulling `bsceapm/predig:latest`; it now uses the same content digest already pinned
  in `scripts/run_predig_batched.py`, `scripts/run_external_tier_a.ps1`, and
  `scripts/run_external_tier_b.ps1`, so all four resolve to one validated image instead of
  whatever `:latest` happens to point at on a given day.
- **Dependabot cooldown window added** (`.github/dependabot.yml`): `cooldown.default-days: 7`
  on all three `package-ecosystem` blocks (`pip` root, `pip` `/environments`,
  `github-actions`). Newly published versions were previously eligible for a bump with zero
  waiting period, which narrows the window in which a compromised release could be proposed
  before the advisory databases catch it. Closes semgrep `dependabot-missing-cooldown`.

- **`LICENSE` now detected as MIT by GitHub instead of "Other".** The copyright block
  spanned two lines, but only the first began with `Copyright`; GitHub's `licensee`
  detector strips copyright lines before template matching, so the second line
  (the five author names) was treated as license BODY text and pushed the file below
  the similarity threshold for the MIT template. The author line is now its own
  `Copyright (c) 2026` line - the idiomatic multi-holder form. No legal substance
  changed: the grant, conditions, and warranty disclaimer are byte-identical.
- **Dead citations in tracked docs re-pointed at targets a reader can actually open.**
  Seven reader-facing references pointed at paths that are not in the repository.
  `README.md`'s certified-headline footnote and `USAGE.md:178` cited the gitignored
  `docs/external_testing/External_Validation_Sign_Off.md`; both now cite the tracked
  `results/external_benchmark_comparison.md` for methodology plus
  `results/table3_tier_a_metrics.csv` and `docs/claims_register.md` for the certified
  metrics and their scope boundaries, and
  `docs/claims_register.md` gained an explicit note that the sign-off file is an internal
  artifact named for provenance completeness, not a document a reader can open. Two
  typo-class defects fixed in the claims register itself: `docs/model_cards/rf_31feature.md
  (pending)` -> `docs/model_cards/rf_31feature_integrated.md`, and `docs/naming.py` ->
  `src/naming.py`. `docs/model_evaluation_summary.md`'s provenance note cited
  `docs/nn_gnn_project2_sync_matrix.md`, which has never existed; it now cites the tracked
  `docs/nn_gnn_optional_module_guide.md`. `docs/model_cards/rf_30feature.md` told the reader
  to "See `results/scoring_error_audit.md`", a generated artifact that is not tracked; it now
  names the generator, `scripts/scoring_error_audit.py`. `docs/architecture/gnn_alphafold_debate.md`
  proposed `src/generate_structural_cache.py`; that script was never created under that name,
  and the design record now points at what was actually built,
  `scripts/run_pandora_structures.py` (PANDORA/MODELLER rather than AlphaFold).
- **Contributor-facing instruction to run a script that does not exist, removed.**
  `.github/ISSUE_TEMPLATE/data_contribution.md` instructed data contributors to run
  `python scripts/check_overlap.py`, which is absent repo-wide (`git ls-files '*overlap*'`
  returns nothing), against `data/immunogenicity_dataset_v3.csv`, two dataset generations
  stale. Replaced with a plain request to state known overlap against the current v5 dataset,
  noting that maintainers re-run a full overlap and contamination check before any merge, so
  an approximate answer does not block a submission.
- **Tier-1 `results/` silent-overwrite guard closed for LOO cross-virus benchmarks**
  (`scripts/run_loo_cross_virus_v4.py`, `scripts/run_loo_cross_virus_v5.py`).
  `--output-json`/`--output-csv` are now required with no default at both the
  CLI and Python-API layers (`run_loo()`'s two output parameters are
  keyword-only with no default), guarded via `src/artifact_guard.py` before
  any work starts. Closes Tier-1 enumeration items #4 and #5.
- **Tier-1 `results/` silent-overwrite guard closed for the last 4 enumeration items**
  (`scripts/compute_loo_binding_confound.py`, `scripts/compute_tier_a_paired_bootstrap.py`,
  `scripts/eval_tsnadb_crossdomain.py`, `scripts/run_tier_a_benchmarks.py`), none of which
  had a CLI at all before - a bare invocation always silently rewrote a git-tracked
  artifact. Each script now takes an optional `--output` flag (or
  `--scores-output`/`--metrics-output` for `run_tier_a_benchmarks.py`, which writes two
  independent tracked artifacts) with no default: omitting it prints results without
  writing anything, matching `scripts/evaluate_per_virus.py`'s existing convention, rather
  than erroring or guessing a destination. `eval_tsnadb_crossdomain.py`'s undisclosed
  second write site (`data/tsnadb_crossdomain_binding.csv`) was confirmed gitignored and
  left unguarded deliberately, out of scope for the tracked-artifact defect class. Closes
  the Tier-1 enumeration completely: 15 of 15 modules now guarded.
- **`docs/model_evaluation_summary.md`'s Pipeline Gold-Standard Recovery table corrected**
  to match its own cited source, `results/baseline_comparison.csv` (Combined row): RF
  4/15 top-10% / 7/15 top-25% / 34.7% mean rank (was stale at 6/15 / 8/15 / 27.1%);
  XGBoost 1/15 / 3/15 / 52.4% (was 2/15 / 6/15 / 35.6%); ANN (MLP) 4/15 / 6/15 / 47.3%
  (was 0/15 / 3/15 / 36.0%). The Binding-only baseline row already matched and is
  unchanged. This drift was disclosed but not fixed in the 2026-07-31 `results/` guard
  batch; the source CSV itself was already current, so this is a transcription
  correction only, not a new pipeline run. Logged as `docs/claims_register.md` D14.
- **`results/v1_v2_quality_comparison.md` given the same "SUPERSEDED HISTORICAL
  SNAPSHOT... DO NOT CITE" disclosure banner already carried by its sibling file**,
  `results/multi_run_stability_report.md` - both are the same 2026-04-24/25 v1/v2
  diagnostic era, but only one had been banner-ed; this one was missed. No content
  below the banner changed.

### Changed
- **Consolidated the last 6 bespoke `results/`/`models/` overwrite guards onto
  the shared `src/artifact_guard.py` template** (`src/train_classifier.py`,
  `src/train_gnn.py`, `src/ann_benchmark.py`, `src/gnn_benchmark.py`,
  `src/ablation_study.py`, `scripts/compute_ann_baseline_summary.py`).
  `guard_planned_paths()` gained `noun`, `trailing`, and `single_path`
  parameters (each defaulting to reproducing the exact prior message) to
  cover message shapes the shared template didn't previously support.
  `single_path=True` now raises `ValueError` if a caller passes more than
  one planned path, and the module docstring's stale module count is
  corrected. All 19 modules using this pattern now delegate to one
  implementation instead of 13 delegating and 6 carrying their own copy.

### Security
- **Fail-closed freshness gate for the committed SBOM artifacts**
  (`tools/check_sbom_freshness.py`, wired into the `python-sbom` job in
  `.github/workflows/security.yml`). `docs/sbom.json` and
  `docs/DEPENDENCY_LICENSES.md` are `pip-licenses` output committed into the repo, but
  CI only ever uploaded freshly generated copies as a 90-day artifact and never
  compared them against what was committed, so the committed pair drifted unnoticed:
  last generated 2026-06-26, and **46 of the 124 packages they share with
  `environments/requirements.lock` disagreed with it**. Four disagreed in the
  security-relevant direction, advertising `torch 2.12.0`, `cryptography 48.0.0`,
  `gitpython 3.1.46` and `aiohttp 3.13.5` when the repo already pinned `2.13.0`,
  `50.0.0`, `3.1.57` and `3.14.3` - so the artifacts understated the project's own
  posture, which is worse than publishing none, because downstream tooling parses them
  as authoritative. The new steps compare the just-generated artifacts against the
  committed ones and fail the job on **either** a version disagreement **or** a package
  the generated set contains that the committed artifact omits. That second direction
  matters: an earlier revision of this tool compared only the package intersection, which
  meant deleting a row outright from `docs/sbom.json` passed as "in sync" - confirmed by
  deleting the `torch`, `cryptography`, `gitpython` and `aiohttp` rows and watching it
  exit 0. Silent deletion is the strongest form of the understates-its-own-posture
  failure this gate exists to prevent, so it now fails. The reverse direction (committed
  but not regenerated) stays a notice, because the lockfile install in that job is
  `continue-on-error` and platform-specific wheels (CUDA and similar) legitimately fail
  on a GitHub runner. Only the `Version` field is compared: license strings can be
  reformatted between `pip-licenses` releases, and more importantly PyPI versions are
  immutable, so a genuine relicensing ships with a version bump that trips the version
  check anyway and the mandated wholesale-file fix refreshes the license column with it.
  The gate is placed after the artifact upload, which retains `if: always()`, so the
  regenerated files stay downloadable on a failing run - that download is the documented
  fix path, and the failure message states it explicitly rather than inviting anyone to
  hand-edit individual rows. The two artifact checks are separate steps because GitHub
  runs `run:` blocks under `bash -e`, so chaining them would skip the second whenever the
  first failed. 24 unit tests (`tests/test_check_sbom_freshness.py`) drive the
  parse/compare logic from fixtures, since `pip-licenses` cannot install the
  Linux-compiled production lockfile on a Windows dev box (`nvidia-cufile` ships no
  Windows wheel); the tool was additionally validated end-to-end against the real
  committed `docs/sbom.json` and `docs/DEPENDENCY_LICENSES.md`, reproducing the 46-row
  drift and flagging all four security-relevant packages, and against a deliberately
  tampered copy with those four rows deleted.
  Generation now passes `--ignore-packages pip-licenses prettytable wcwidth pip wheel`,
  excluding the SBOM tooling and the runner's own bootstrap - none of which is in
  `environments/requirements.lock` (which does pin `setuptools`, deliberately kept).
  Without that, the mandated fix path would bake the SBOM tooling into the production
  SBOM, over-reporting the dependency set: the exact mirror of the under-reporting this
  gate exists to catch.
  **This gate is expected to fail on its first run** until the regenerated artifacts are
  downloaded and committed once. That correction will also **shrink** the committed
  artifacts, which currently carry 214 packages captured in a local dev environment,
  down to the runner-installable production set - an expected drop, not tampering.
  Afterwards it stays green until the audited dependency set changes; usually a lockfile
  bump, but a runner image refresh can also move a preinstalled package and turn it red
  with the lockfile untouched. Every such failure is resolved by the same
  regenerate-and-commit path.

- **`torch` upgraded `2.12.0` -> `2.13.0`** to close CVE-2025-3000 / GHSA-rrmf-rvhw-rf47 /
  PYSEC-2025-194 (low, CVSS v3 5.3): memory corruption in `torch.jit.script`, affecting
  `<= 2.12.1`, first patched in `2.13.0` (published 2026-07-08). This **replaces a standing
  risk acceptance with an actual fix.** The prior entry in `SECURITY.md` rested on two
  claims - that SESTRAV never calls `torch.jit.script` (still true) and that no upstream
  patch existed (false since 2026-07-08). Its own re-review trigger was "publication of a
  patched release"; that trigger fired and went unnoticed for four weeks because nothing
  re-evaluates the register on a schedule.
  Pinned at `2.13.0` in `requirements.in`, `requirements.txt`,
  `environments/requirements.lock` and `environments/requirements-ci-torch-cpu.txt`, and
  floored at `>=2.13.0` in `pyproject.toml`. **The `pyproject.toml` floor is not
  redundant:** the lockfiles only govern hash-pinned installs, so without it a plain
  `pip install -e .` or any downstream consumer could still resolve into the affected
  range. Both compiled lockfiles moved exactly three pins, all torch-transitive: `torch`,
  `cuda-toolkit` `13.0.2` -> `13.0.3.0`, `triton` `3.7.0` -> `3.7.1`.
- **`pip-audit` suppressions removed.** The `--ignore-vuln PYSEC-2025-194` /
  `GHSA-rrmf-rvhw-rf47` flags are deleted from all three `pip-audit` invocations in
  `.github/workflows/security.yml`, so the advisory is reported again if it ever
  reappears rather than being permanently muted. That suppression list is now empty.
- **`overrides.txt` retired.** torch `2.12.0` declared a `setuptools<82` build-metadata cap
  that collided with this repo's `setuptools>=83.0.0` security floor (GHSA-h35f-9h28-mq5c)
  and made both application specs unsatisfiable for any resolver, forcing every recompile
  through a `uv` override file. torch `2.13.0` declares `setuptools>=77.0.3`, meeting the
  override's own documented exit condition, so the file is deleted and both specs now
  compile unaided. **The `setuptools` floor itself is unchanged** - it was the security
  constraint, not the workaround. `tests/test_dependency_tooling.py` now asserts both
  halves of the retirement so the workaround cannot quietly return and mask a genuine
  resolution conflict.

### Fixed
- **`docs/security_compliance.md`'s pip-audit table no longer asserts resolved advisories
  are open.** Correcting the `torch` row surfaced the same defect on two others: `aiohttp`
  read "tolerable risk ... will upgrade when mhcflurry releases a compatible version" and
  `pyjwt` read "tolerable risk - transitive dependency", when
  `environments/requirements.lock` already pins `aiohttp==3.14.3` and `pyjwt==2.13.0`,
  satisfying the `>=3.14.1` and `>=2.13.0` fixes recorded in the table's own Fix column.
  Both re-annotated RESOLVED, with the original disposition retained as the record of what
  was believed at the time of the run. The observed-version column is left untouched
  throughout: **a version number ageing is harmless, a false status assertion is not.**
- **Five documents contradicted the torch upgrade and were corrected.**
  `docs/security_compliance.md` (linked from `README.md` as the compliance front door) and
  `docs/SCORECARD_REMEDIATION.md` both still asserted "no upstream patch"; the latter also
  rated the advisory *critical* in its summary table where GitHub rates it *low*, and gave
  the affected range as `<= 2.12.0` rather than `<= 2.12.1`. `SECURITY.md` was rewritten
  from a risk acceptance to a resolution. `docs/threat_model.md` (linked from `README.md`
  as the governance and assurance evidence) listed CVE-2025-3000 under "Residual risks
  (accepted)" as a live example of an advisory with "no available patch"; the example is
  now the `mcp` SDK transport advisories, which are genuinely still accepted.
  `.github/workflows/security.yml` justified auditing the installed set rather than
  `pip-audit -r environments/requirements.lock` on the grounds that the direct form dies
  with `ResolutionImpossible` from the setuptools/torch conflict - a reason the upgrade
  eliminated. The installed-set form is kept deliberately, for the reason that is still
  true.
  **Three of these five were found only by successive pre-push claims audits**, not by the
  initial sweep. Separately, and worth recording on its own: three of the five were
  falsified by the very first commit of this branch (the upgrade itself invalidated their
  status assertions), and this file then developed two *internal* self-contradictions
  because writing a changelog entry about the upgrade falsified pre-existing text
  elsewhere in the same file. Correcting one document repeatedly invalidated another,
  across four successive audit rounds.

### Documentation
- **`docs/DEPENDENCY_LICENSES.md` labelled as stale rather than partially patched.** It
  carried `torch 2.12.0`, but the file turned out to be broadly stale: generated by
  `pip-licenses` against an installed environment, last regenerated 2026-06-26, not derived
  from the lockfiles, with **46 of its 124 lockfile-matched rows** now disagreeing with
  `environments/requirements.lock`. It names `torch 2.12.0`, `cryptography 48.0.0`,
  `gitpython 3.1.46` and `aiohttp 3.13.5` where the repo pins `2.13.0`, `50.0.0`,
  `3.1.57` and `3.14.3` - all four older versions carrying advisories already closed, so
  the table *understates* the project's posture. Updating only the torch row would have
  manufactured the appearance of freshness across the other 45, so the version column is
  left untouched and a provenance banner added instead. `docs/sbom.json` is co-generated
  by the same tool and stale in the same way, but being JSON cannot carry such a banner;
  it is disclosed in the licenses banner. Regenerating both was disclosed here as
  unscheduled at the time this entry was written; **the gating half is now implemented in
  this same release** - see the fail-closed freshness gate under Security above. The
  one-time regeneration of the committed artifacts remains outstanding.
- **Recorded a standing lesson in `SECURITY.md`:** a risk acceptance with no scheduled
  re-review is a claim that decays silently. Treat an advisory's `firstPatchedVersion`
  field as the authoritative test of whether a patch exists, not prose in a checked-in
  document.
- **Noted the Dependabot lockfile blind spot** in `.github/workflows/security.yml`:
  Dependabot parses `.in` sources, never the compiled `.lock`/`.txt` artifacts, so a
  resolved-but-vulnerable pin that exists only in a lockfile raises no alert. That job is
  currently the only thing that sees such a pin.

- **Fail-closed advisory gate added for the production lockfile**
  (`tools/check_lockfile_advisories.py`, `environments/accepted_advisories.toml`),
  retiring pip-audit's `--ignore-vuln` CLI flags as a suppression mechanism in
  `.github/workflows/security.yml`. Two structural gaps compounded to let
  CVE-2025-3000 sit patched-but-unnoticed for four weeks after torch 2.13.0 shipped:
  Dependabot parses `.in` sources, never compiled `.lock`/`.txt` artifacts (across
  every alert this repo has ever had, zero carry a `manifest_path` ending in
  `.lock`), and every pip-audit step in CI was `continue-on-error` with permanent
  `--ignore-vuln` flags applied *before* any report was written - invisible to any
  tool that reads pip-audit's own output, including one meant to notice when a
  suppressed finding gets fixed upstream. The new step consumes the same
  `pip-audit --format json` report the tooling-assert step already produces
  (now generated unfiltered), fails the job on any finding not explicitly listed
  in `environments/accepted_advisories.toml`, and emits a `::notice::` when an
  accepted advisory no longer appears in the audit - the fixed-but-forgotten
  direction of the same problem. **Scoped to packages this repo actually pins**
  in `environments/requirements.lock`, not everything an audited venv contains:
  a real `pip-audit` run against a live dev environment (not just a fixture)
  surfaced findings on `pip` and `mcp`, neither of which is a lockfile pin -
  `pip` is whatever the CI runner's Python bootstrap ships, `mcp` lives only in
  `environments/requirements-semgrep.txt`, a separate CI/dev-only manifest this
  job never audits. Gating on either would have blocked every PR on a finding
  with no pin here to bump. Out-of-scope findings are reported as `::notice::`
  for visibility and never require an acceptance entry, matching Dependabot's
  own scope (it never alerts on a package absent from a tracked manifest). 28
  new unit tests (`tests/test_check_lockfile_advisories.py`), since pip-audit
  cannot run against `environments/requirements.lock` on Windows (it is
  Linux-compiled; `nvidia-cufile` has no Windows wheel), so the gate is
  exercised against fixture reports rather than a live audit; schema
  assumptions were independently verified against a real `pip-audit --format
  json` report before being locked into fixtures. Built against a branch point
  before PR #205 (torch 2.13.0) had merged, so it was briefly seeded with a
  temporary allowlist entry for `PYSEC-2025-194` / torch; rebased onto main
  after that PR merged, and the now-unnecessary entry was removed before this
  branch was pushed rather than left for the tool's own stale-acceptance
  notice to catch later.

- **`cryptography` floored at `>=50.0.0`** to close GHSA-g6cj-pr64-35w5 / CVE-2026-69247
  (high, CVSS 8.2): `pkcs7_decrypt_der` / `_pem` / `_smime` reported the outcome of decrypting
  a `RecipientInfo`'s `encryptedKey` in distinguishable ways, one of which disclosed the exact
  recovered length, giving an attacker a Bleichenbacher oracle against the content-encryption
  key. Affects `>=44.0.0,<50.0.0`; the repo was resolved at `49.0.0`.
  `environments/requirements-lock.in` already carried a `cryptography>=48.0.1` floor for the
  earlier GHSA-537c-gmf6-5ccf, so the floor was raised in place and both affected lockfiles
  recompiled: `environments/requirements.lock` (where `cryptography` is a direct security
  override) and `environments/requirements-semgrep.txt` (where it arrives via `pyjwt[crypto]`).
  **Both source specs are floored, so neither lockfile can be walked back by a later
  recompile.** The two arrive at `50.0.0` by different routes and each needed its own
  constraint: `environments/requirements.lock` compiles from
  `environments/requirements-lock.in`, where `cryptography` was already an explicit security
  override and the existing floor was raised in place. `environments/requirements-semgrep.txt`
  compiles from `environments/requirements-semgrep.in`, which previously declared nothing but
  the `semgrep` pin - `cryptography` arrived transitively via `pyjwt[crypto]` and resolved to
  `50.0.0` only because that is currently latest, not because anything required it. A matching
  `cryptography>=50.0.0` override (with its own exit condition) was therefore added there too,
  which is why that file's `# via` trailer now names the spec alongside `pyjwt`. Without it the
  remediation would have held by coincidence rather than by constraint.
  **Both recompiles were verified surgical rather than assumed:** each file changed exactly one
  pin (`cryptography 49.0.0 -> 50.0.0`), with package count, total hash-line count and
  per-package hash uniqueness identical before and after - the check that catches the
  in-place-compile hash-duplication defect this repo hit once before (a recompile that silently
  doubled 262 hash lines across 21 packages).
  **Scope note - this is a defence-in-depth floor, not an exploitable-path fix.** The vulnerable
  API is unreachable from SESTRAV: `cryptography` is never imported by tracked source (the only
  tracked `.py` occurrence is a synthetic fixture string in
  `tests/test_check_lockfile_freshness.py`), and there are zero `pkcs7` / `EnvelopedData` /
  S-MIME call sites repo-wide. Exploitation additionally requires a service that auto-decrypts
  untrusted `EnvelopedData`, which SESTRAV does not provide.
  **Disclosed, not fixed here:** GitHub raised this as Dependabot alert #107 against
  `environments/requirements-semgrep.txt` only. It did **not** raise an alert for
  `environments/requirements.lock`, which pinned the same vulnerable `49.0.0` - Dependabot
  tracks `environments/requirements-lock.in` (the source spec) rather than the compiled
  `.lock` artifact, so a vulnerable *resolved* pin in that file is invisible to alerting
  whenever the `.in` floor permits it. This branch closes the instance; the alerting blind
  spot itself remains open.
  `docs/security_compliance.md`'s pip-audit table records the resolution on its
  `cryptography` row. That table is a dated snapshot of a 2026-06-18 run and its other
  rows still quote that run's versions. Those version numbers are **deliberately left as
  the historical record** - correcting a past run's observed versions is a separate
  editorial decision from recording a remediation. (Superseded on 2026-08-05: the
  *disposition* text on the remaining three rows was found to be a different case
  entirely. A version number ageing is harmless; a row asserting "tolerable risk, will
  upgrade later" when the upgrade has already shipped is a false status claim. All three
  were re-annotated RESOLVED at that point. See the torch entry at the top of this
  section.)

### Added
- **`src/artifact_guard.py` - one shared overwrite guard, and one contract the four
  `results/`-family entry points must satisfy.** The `FileExistsError` guard that closed the
  `models/` and `results/` silent-overwrite defect class had been copy-pasted into ten modules.
  This change extracts and unifies **four** of them - the `results/` family:
  `scripts/run_analysis.py`, `src/final_validation_report.py`, `src/bias_skew_finalization.py`
  and `src/h2_tier_a_evaluation.py`. The duplicated implementation was cheap; the *divergent
  per-module tests written against it* were not. During the `--results-dir` repair a dedicated
  16-test file passed in full against a live regression it existed to catch, because its
  assertions searched all of stderr for a flag name the guard's own error message also contains.
  Extraction is what makes a single generic, parametrized contract test possible, so a guarded
  module cannot quietly be held to a weaker standard than its siblings.
  **This is a behaviour-preserving refactor, and that was verified rather than asserted:** each
  module's `FileExistsError` message, planned-path enumeration, empty-directory silence,
  partial-collision behaviour and `allow_overwrite` disarm were captured before and after the
  change and diffed byte-for-byte (15 captured behaviours across the three importable modules,
  identical). `scripts/run_analysis.py` cannot be imported on the dev machine at all (its
  `src/shap_analysis.py` import hard-crashes the interpreter via `shap`), so its message was
  instead reproduced from the shared helper using its exact arguments and compared against the
  pre-refactor literal recovered from git at `227ed6c` - full collision, partial collision and
  empty-directory cases all match. Each module keeps its own `planned_*_paths()` and thin
  `_guard_*_dir()` wrapper, so every existing per-module test still runs unchanged and is itself
  evidence the behaviour did not move. Enumeration deliberately stays per-module: it is the part
  that requires reading that specific module's writes, including derived filenames and delegate
  writes, and the part most likely to drift.
  `tests/test_artifact_guard_contract.py` adds 23 cases - 6 parametrized checks across the 3
  registered modules (planned paths stay under the output directory, planned paths are unique,
  silence on an empty directory, *every* planned artifact individually triggers the guard rather
  than only the first, `allow_overwrite` disarms a full collision, and the message names the flag,
  the escape hatch, the API hint and every colliding file), plus 5 covering the shared helper
  directly (name joining, directories sharing an artifact name are not treated as overwrites,
  sorted collision listing, and the optional detail clause both present and absent).
  An 11-mutation battery confirms the tests are load-bearing: dead escape hatch, guard inspecting
  only the first planned path, guard never raising, unsorted listing, dropped detail clause,
  dropped flag name, dropped API hint, a same-prefix API hint, under-reported artifact count, a
  module dropping a planned artifact, and a module's guard becoming defined-but-inert.
  **11 of 11 detected.** Full suite
  1451 passed, 0 failed, 0 errors, 2 skipped under the standing local exclusion, from 1453
  collected (reconciling as 1430 + 23). *(These are the figures measured for this change.
  The torch 2.13.0 entry at the top of this section subsequently adds a net +1 test, so
  they no longer describe the current suite. Left as the record of this run rather than
  recomputed, since the two runs use different exclusion sets and the newer figure was
  measured under the pre-push gate, not this one.)*
  **Scope note - this does not put every guarded entry point under the contract.** Six further
  modules carry their own copy of this guard and are deliberately left outside both the shared
  helper and `tests/test_artifact_guard_contract.py`, so the coverage claim above is four of ten,
  not ten of ten. Four are the same three-piece pattern and could be folded in later:
  `src/train_classifier.py`, `src/train_gnn.py`, `src/ann_benchmark.py` and
  `src/gnn_benchmark.py`. Two are single-path variants whose message has a different shape
  entirely (no collision count, no listing): `src/ablation_study.py` and
  `scripts/compute_ann_baseline_summary.py`. Extraction was not attempted for any of the six here
  because the template would first need to grow: `src/ann_benchmark.py` and `src/ablation_study.py`
  both add a "(for example `models/scratch/<run-name>`)" clause the current message cannot express,
  and folding the single-path variants in would change their user-facing text, which would no
  longer be a behaviour-preserving refactor. Each remains individually tested by its own
  per-module file; what they do not yet get is the uniform contract.

### Removed
- **`requirements-ann.txt` and `requirements-gnn.txt` retired** - both were broken installs and
  almost entirely redundant. Each began with `-r requirements.txt`, which is fully hash-pinned,
  and then added unhashed pins; pip turns on `--require-hashes` automatically as soon as any
  requirement carries a hash, so `pip install -r requirements-ann.txt` failed with "Hashes are
  required in --require-hashes mode" - the same defect class as the README install command
  repaired in #177. The pins they added were redundant anyway: `torch` and
  `torch-geometric` are both already pinned and hashed in `requirements.txt`, leaving only
  `transformers`, which `pyproject.toml`'s existing `[gnn]` extra already declares. Callers now
  point at the two paths that work: the base install for `src/ann_benchmark.py` and
  `src/verify/sestrav_evaluator.py` (which need only what `requirements.txt` already pins -
  note the evaluator's non-mock scoring path also uses `torch_geometric`, which the base
  install supplies), and `pip install ".[gnn]"` for
  `src/gnn_benchmark.py` (which needs `torch_geometric`) - the command README already
  recommended for the GNN track. Updated `README.md`, `docs/nn_gnn_optional_module_guide.md`,
  3 messages in `src/gnn_benchmark.py`, and 1 in `src/verify/sestrav_evaluator.py`.

### Changed
- **Training runs can no longer silently overwrite published artifacts** (breaking CLI change):
  `src/train_classifier.py` defaulted `--model-dir` (and `train_models(model_dir=...)`) to the
  production `models/` directory, which is how the v5 retrain overwrote
  `models/training_results_mode31.csv` and `models/rf_oof_predictions_mode31.csv` in place and
  left v4 metrics quoted against v5 files. `--model-dir` is now required with no default on
  `python -m src.train_classifier`, `sestrav validate`, and `python -m
  src.bias_skew_finalization` (all three retrain and write the same tracked release-artifact set
  through `train_models`), and `train_models` aborts before training (`FileExistsError`, listing
  the offending paths) if any artifact it would write already exists, unless
  `--allow-overwrite` / `allow_overwrite=True` is passed. The checksum manifest is exempt from
  the guard because it is upserted, not replaced. Callers that used to train into `models/` now
  target their own directories: `scripts/regenerate_shareout_pngs.py` retrains into
  `models/shareout_20260426_retrain/`, and `notebooks/SESTRAV_Colab_Pipeline.py` trains into
  `models/scratch/colab_v3_mode30/` (a Colab run clones this repository, so the tracked mode-30
  artifacts are already on disk and the old `models/` target would now abort on the guard; the
  notebook's `train_ann` and `run_ablation` cells were retargeted alongside it so they keep
  writing into the same scratch directory rather than back into `models/`). The `sestrav --help`
  usage example and every documented invocation of these three commands now include
  `--model-dir`. Per-mode artifact filenames and all training math are unchanged.
- **`src/ann_benchmark.py` closes the same silent-overwrite trap** (breaking CLI change): this
  was the fourth entry point of the class above and was explicitly scoped out of that change.
  `--model-dir` defaulted to `models`, so a run that omitted the flag rewrote the tracked
  `models/training_results.csv` in place, adding `ann_cv_mean`/`ann_cv_std` columns
  to a published release artifact. `--model-dir` is now required with no default, `train_ann`
  takes `allow_overwrite`, and a run aborts with `FileExistsError` before any training work if it
  would replace an artifact already on disk. Two files are deliberately exempt from the guard:
  the checksum manifest (upserted) and `training_results.csv`, which this step *merges into*
  rather than replaces - the `train_classifier` step of the same run writes it moments earlier,
  so guarding it would make every legitimate RF-then-ANN run into one directory fail. The
  required flag, not the guard, is what removes the accidental path to `models/`.
  `src/bias_skew_finalization.py` now threads its `allow_overwrite` through to the ANN step, the
  three usage examples in the module docstring carry `--model-dir`, and the two docs that named
  `models/` as the ANN output destination (`docs/nn_gnn_optional_module_guide.md`,
  `docs/model_evaluation_summary.md`) now describe the destination as whatever `--model-dir` is
  pointed at. `src/train_ann.py` is a separate entry point and is correctly untouched: its
  `models/ann` default contains no tracked artifacts. The two instances this entry previously
  listed as still open - `src/train_gnn.py` and `src/gnn_benchmark.py` - are now closed as well;
  see the following entry. Training math is unchanged.
- **The GNN entry points close the same trap for the two widest `models/` writers** (breaking
  CLI change):
  `src/train_gnn.py` defaulted `--model-dir` to `models/gnn` and `src/gnn_benchmark.py` defaulted
  `--output-dir` to `models`, so a run that omitted the flag overwrote tracked release artifacts.
  This was the widest instance of the family: a default `python -m src.train_gnn` run (v2
  architecture, mean pooling) rewrote **four** tracked files - `models/gnn/gnn_config.json`,
  `models/gnn/gnn_config_mean.json`, `models/gnn_oof_predictions.csv` and
  `models/gnn_oof_predictions_mean.csv` - and `python -m src.gnn_benchmark` rewrote
  `models/gnn_sequence_benchmark.csv` and `models/gnn_bipartite_benchmark.csv`.
  Both flags are now required with no default, both entry points take `--allow-overwrite`, and
  both abort with `FileExistsError` before any training or benchmarking work.
  **The GNN guard is deliberately wider than its output directory:** `train_gnn` writes the OOF
  predictions into the *parent* of `--model-dir`, so a guard scoped to `--model-dir` alone would
  have missed the two tracked `gnn_oof_predictions*.csv` files entirely. Pointing `--model-dir`
  at a scratch directory redirects those writes too. The guard is also pooling-aware: only a
  mean-pooling run writes the untagged canonical copies, so an `--pooling attention` experiment
  is not blocked by canonical artifacts sitting alongside it. `pipeline.smk`'s `train_gnn` rule
  passes `--allow-overwrite` explicitly, because that rule *is* the reproduction path for those
  published artifacts - regenerating them there is the intent, not an accident. Training math,
  architectures and artifact filenames are unchanged.
- **`scripts/compute_ann_baseline_summary.py` and `src/ablation_study.py` close two further
  instances of the same trap** (breaking CLI change): `--output-summary` defaulted to the tracked
  `models/ann_cv_summary.csv` and `--results-file` defaulted to the tracked
  `models/training_results.csv` (merged into on every default run); `--output-dir` defaulted to
  `models`, threatening `models/ablation_study_results.csv` (currently untracked, unlike the
  other six instances in this line - see below). `--output-summary` and `--output-dir` are now
  required with no default and gain `--allow-overwrite` plus a `FileExistsError` guard that runs
  before any work. `--results-file` is different from every other flag in this family: it is a
  merge target, not a fresh artifact, so it is NOT guarded with `FileExistsError` (same exemption
  reasoning as `ann_benchmark.py`'s `training_results.csv`) - instead it simply lost its dangerous
  default. It is now optional with no default at all: omitting it skips the merge outright, so a
  bare invocation can no longer touch `models/training_results.csv` by accident. `run_ablation()`
  also lost its `output_dir="models"` Python-API default, matching every other function in this
  line (`train_ann`, `train_gnn`, `train_gnn_v2` all require it too) - the notebook and CLI
  callers already passed an explicit path, so nothing breaks. Severity note: unlike the other six
  entry points in this line, `models/ablation_study_results.csv` is not currently a tracked file,
  so this specific instance could not have silently corrupted a published artifact the way the
  other six could have - it is closed anyway for consistency of the defect class and because
  nothing prevents the file from being tracked later.
- **The same silent-overwrite trap also existed for `results/`, not just `models/`, and is now
  closed for `scripts/run_analysis.py` and `src/final_validation_report.py`** (breaking CLI
  change): `--results-dir` defaulted to `results` on both, with no guard. This is a
  higher-severity instance than any single one in the `models/` line above:
  `results/h2_tier_a_summary.md` is the source behind the certified R10 = 0.9494 H2 Tier A null
  result published in `README.md`, not just a training artifact. `--results-dir` is now required
  with no default on both, and both gain `--allow-overwrite` plus a `FileExistsError` guard
  that runs before any work. `run_analysis.py`'s guard covers all 9 files a run writes,
  including the 7 delegated through `src/shap_analysis.py`'s `run_shap_analysis` (not just its
  own 2 direct CSVs). `final_validation_report.py`'s guard covers all 10 atomically-published
  files, including the 3 mode/version-tagged aliases `canonical_output_filename` produces - the
  guard and the real publish step now draw from one shared `planned_validation_paths()` helper
  so they cannot drift apart. `src/bias_skew_finalization.py` is fixed alongside them: it called
  `run_final_validation()` but also, redundantly, computed and wrote `baseline_comparison.csv`
  itself one step earlier - once `final_validation_report.py` gained a guard, that redundant
  pre-write would have made even a first-ever run into an empty directory trip the guard, since
  the file would always already exist by the time `run_final_validation`'s guard checked for it.
  The redundant write is removed (nothing downstream depended on it existing early) rather than
  forcing `allow_overwrite=True` unconditionally on that call site, which would have silently
  disabled the guard for all 10 of `run_final_validation`'s files. `allow_overwrite` now threads
  through normally, matching the existing `train_models`/`train_ann` pattern in the same
  function. **`run_bias_skew_finalization`'s own `--results-dir` is a further, larger instance
  of this same defect (writes or delegates 8 more files: `immunogenicity_provenance.csv`,
  `data_bias_audit_summary.csv`, `data_bias_audit.md`, `data_bias_audit_summary_virus_label_counts.csv`,
  `gold_standard_sensitivity.csv`, `gold_standard_sensitivity.md`,
  `gold_standard_sensitivity_deltas.csv`, `release_readiness_summary.md`) and was left open at
  the time** - not fixed in this entry, since each delegate needed the same enumerate-every-write
  rigor this entry used before a guard could be built correctly. Now closed - see the following
  entry.
- **`run_bias_skew_finalization`'s own `--results-dir` closes the last disclosed instance of the
  `results/` line** (breaking CLI change): flagged but deliberately not fixed in the entry above,
  pending a per-delegate write enumeration. That enumeration is now done by reading
  `src/data_bias_audit.py`'s `refresh_dataset`/`write_audit_reports` and
  `src/gold_standard_sensitivity.py`'s `run_gold_standard_sensitivity` in full, and the originally
  logged 8-file list is confirmed exactly correct, no 9th file and no false entry:
  `immunogenicity_provenance.csv` (`refresh_dataset`), `data_bias_audit_summary.csv` and
  `data_bias_audit.md` (`write_audit_reports`), the derived
  `data_bias_audit_summary_virus_label_counts.csv` (`write_audit_reports`,
  `output_csv.replace(".csv", "_virus_label_counts.csv")`), `gold_standard_sensitivity.csv` and
  `gold_standard_sensitivity.md` (`run_gold_standard_sensitivity`), the derived
  `gold_standard_sensitivity_deltas.csv` (`run_gold_standard_sensitivity`,
  `output_csv.replace(".csv", "_deltas.csv")`), and `release_readiness_summary.md`, written
  directly at the end of `run_bias_skew_finalization` itself. `--results-dir` is now required
  with no default at both the CLI and `run_bias_skew_finalization(results_dir=...)` layers (moved
  ahead of the already-defaulted `data_csv` parameter in the signature, since Python does not
  allow a no-default parameter after a defaulted one; every existing caller already passed it by
  keyword, so nothing breaks), gains `--allow-overwrite`, and a new `planned_bias_skew_paths()` +
  `_guard_results_dir()` pair raises `FileExistsError` listing every colliding file before any
  work starts - the guard call is the literal first statement of the function, ahead of even
  `os.makedirs`. No write-before-guard collision was found this time (unlike the
  `baseline_comparison.csv` case above): the guard is the first statement in the function, so
  nothing in `run_bias_skew_finalization` writes any of these 8 files before the guard checks for
  them, and none of the 8 filenames overlap with `run_final_validation`'s own, separately-guarded
  10-file list. Tests in `tests/test_bias_skew_finalization_results_guard.py` cover planned-path
  enumeration, guard-pass-on-empty-dir, guard-refuses-on-an-existing-file (one case per delegate),
  allow-overwrite disarming the guard, a monkeypatched wiring test proving the guard is actually
  called by the real `run_bias_skew_finalization` (not just defined), and a CLI-level check that
  `--results-dir` has no default. No `pipeline.smk` rule invokes
  `src.bias_skew_finalization` (checked directly), so no Snakemake rule needed an
  `--allow-overwrite` addition.
  **Scope note - this does not close the `results/` defect class.** It closes the last instance
  *previously disclosed in this changelog*, which is a narrower claim than it may read as. An
  independent sweep at this commit finds 36 further `default="results..."` call sites across 25
  tracked `.py` files under `src/` and `scripts/`. At least six overwrite git-tracked artifacts on
  a bare invocation, with no required flag, no `FileExistsError` guard and no `--allow-overwrite`:
  `src/h2_tier_a_evaluation.py` (`--output-dir`, writing `h2_tier_a_summary.md`,
  `h2_tier_a_summary.csv` and `h2_tier_a_fold_metrics.csv` - the same certified R10 = 0.9494
  source this line cites above as its own justification, and so the most severe instance now
  known); `src/data_bias_audit.py` and `src/gold_standard_sensitivity.py`, whose own `__main__`
  CLIs default 7 of the 8 filenames guarded above straight back into `results/` (every one except
  `release_readiness_summary.md`), leaving them reachable unguarded through the siblings; `src/calibration_analysis.py`
  (`results/calibration_metrics.csv`); `src/shap_analysis.py` (`shap_values_{tag}.csv`, i.e. the
  tracked `results/shap_values_rf.csv`); `scripts/compute_population_coverage.py`
  (`results/population_coverage_v5.json`); and `src/external_validation_cross_virus.py`
  (`results/external_validation_cross_virus.csv`, since closed - see the Tier-1 #2/#3/#9 entry
  below). These were disclosed here, not fixed at the time: each needed the same per-file write
  enumeration this entry used before a guard could be built correctly, and that list was not
  assumed exhaustive.
- **`src/h2_tier_a_evaluation.py` closes the most severe instance disclosed in the entry above**
  (breaking CLI change): `--output-dir` defaulted to `results`, with no guard, so a bare
  `python -m src.h2_tier_a_evaluation` run rewrote `h2_tier_a_summary.md` in place - the source
  behind the certified R10 = 0.9494 H2 Tier A null result published in `README.md`. `--output-dir`
  is now required with no default at both the CLI and `run_h2_tier_a(output_dir=...)` layers, gains
  `--allow-overwrite`, and a new `planned_h2_tier_a_paths()` + `_guard_output_dir()` pair raises
  `FileExistsError` listing every colliding file before any work starts. All three writes
  (`h2_tier_a_fold_metrics.csv`, `h2_tier_a_summary.csv`, `h2_tier_a_summary.md`) are direct, with
  no derived filenames and no delegate writes - `evaluate_subgroups` was checked and returns
  DataFrames without touching disk. `src/final_validation_report.py:137` already calls
  `run_h2_tier_a` with these same three filenames guarded a second time, and stays compatible only
  because it passes a `tempfile.mkdtemp()` directory rather than `results/` itself, so this new
  guard finds nothing there; a test locks that property down since it is load-bearing and
  non-obvious. Tests: `tests/test_h2_tier_a_results_guard.py` adds 15 cases (planned-path
  enumeration, guard-pass-on-empty-dir, a per-file parametrized clobber check, a wiring test
  proving the guard is called by `run_h2_tier_a` and not merely defined, an
  allow-overwrite-passthrough test, the `final_validation_report` temp-dir interaction lock, and 3
  CLI-level checks anchored on argparse's required-arguments line rather than a bare stderr
  substring - the guard's own error message names `--output-dir` and would otherwise satisfy that
  check while a regression was live). `tests/test_entry_point_help_smoke.py` registers this module
  into the existing `OUTPUT_DIR_REQUIRED_ENTRY_POINTS` list, adding 4 more cases across the file's
  existing parametrized checks. Full suite 1428 passed, 0 failed, 0 errors, 2 skipped, from 1430
  collected under this box's standing local exclusion of
  `tests/test_run_analysis_results_guard.py` (7 tests; a
  deterministic Windows `shap`-import crash on this machine, reproduced on 4 separate attempts
  including with `KMP_DUPLICATE_LIB_OK=TRUE` set - not a regression from this change: it reproduces
  byte-identically on `main` (`d136942`) too, with the same `0xc06d007f` / `scipy.linalg.inv` /
  `shap/plots/colors/_colorconv.py` signature), the collected figure reconciling as this box's
  1411-collected baseline + 15 (new guard file) + 4 (smoke file). **CI confirms the unexcluded
  totals and that the crash is
  local to this machine:** `test (3.13)` on `ubuntu-latest` collected 1437 (1428 passed, 9 skipped,
  0 failed), reconciling exactly as `main`'s 1418 + 19 on both the total and the passed count, which
  places the 7 otherwise-uncollectable tests among CI's passes rather than its skips. The 9-vs-2
  skip difference is unrelated to this change: a fresh checkout has no gitignored `data/` or
  `models/` fixtures, so 7 more data-dependent tests skip there than in a populated local tree,
  reproduced identically in a `main` worktree. **Scope note, unchanged from the entry above:**
  `src/data_bias_audit.py`, `src/gold_standard_sensitivity.py`, `src/calibration_analysis.py`,
  `src/shap_analysis.py`, `scripts/compute_population_coverage.py`, and (at the time of this entry)
  `src/external_validation_cross_virus.py` remained open instances of the same defect class.
  `src/external_validation_cross_virus.py` is since closed - see the Tier-1 #2/#3/#9 entry below.
- **`src/data_bias_audit.py` and `src/gold_standard_sensitivity.py` close step 8 of the
  `results/` silent-overwrite defect-class line** (breaking CLI change): per the Tier-1
  enumeration in `_local/notes/results-dir-tier1-enumeration-2026-07-30.md` (15 modules, ~26
  tracked artifacts, `src/h2_tier_a_evaluation.py` above closed item #1), this closes item #8,
  bringing the closed count to **2 of 15**. `src/data_bias_audit.py` protects exactly **one**
  git-tracked artifact: `results/data_bias_audit.md` (un-ignored at `.gitignore:259`); its other
  writes (`immunogenicity_provenance.csv`, `data_bias_audit_summary.csv`, and the derived
  `data_bias_audit_summary_virus_label_counts.csv`) are untracked. `src/gold_standard_sensitivity.py`
  protects **zero** tracked files - `gold_standard_sensitivity.*` is untracked, unlike the
  similarly-named, tracked `gold_standard_validation.csv` written by a different module (a Tier 2
  entry in the enumeration, not Tier 1) - and is closed here for consistency with the rest of the
  defect-class family and CHANGELOG-disclosure closure, not to protect a published result.
  `--provenance-csv`, `--audit-csv` and `--audit-md` on `data_bias_audit.py`, and `--output-csv`
  and `--output-md` on `gold_standard_sensitivity.py`, are now required with no default (all five
  previously defaulted into `results/`); both modules gain `--allow-overwrite`.
  `data_bias_audit.py`'s `--output-csv` (the `data/immunogenicity_dataset_v4.csv` dataset path)
  and `gold_standard_sensitivity.py`'s `--results-dir` (a read-only input, only ever joined with
  `{prefix}_ranked.csv` and read) are both deliberately untouched by this fix.
  Neither module fit `src/artifact_guard.py`'s existing template as-is: neither has a single
  output directory - both take independent file-path flags rather than one
  `--output-dir`/`--results-dir` - so the shared `"under '{output_dir}'"` /
  `"Point {flag} at a fresh directory"` message clauses would have been actively wrong advice.
  `guard_planned_paths` gained optional `scope`/`remedy` keyword parameters so these two callers
  can substitute accurate clauses (`scope="among this run's planned artifacts"` and a remedy
  naming the actual flags to redirect) while every existing caller's message stays byte-identical
  when the new parameters are omitted - verified directly, not assumed: the four existing callers'
  `FileExistsError` messages (`scripts/run_analysis.py`, reproduced via the shared helper with its
  exact arguments since it still cannot be imported on this machine; `src/final_validation_report.py`;
  `src/bias_skew_finalization.py`; `src/h2_tier_a_evaluation.py`) were captured across
  full-collision, partial-collision, empty-location and `allow_overwrite`-disarm cases both before
  and after this change and diffed identical.
  Three hazards drove the design, all handled. **Hazard A**: `data_bias_audit`'s `output_csv` (the
  `data/` dataset path) exists on disk right now, is gitignored, is `refresh_dataset`'s declared
  rewrite target, and is read back intra-run by `write_audit_reports` moments later - guarding it
  would abort every run unconditionally, so `planned_data_bias_audit_paths` never includes it
  (same exemption shape as the earlier `training_results.csv` merge-target precedent), locked down
  by a dedicated regression test. **Hazard B**: `src/bias_skew_finalization.py` is the only caller
  of either module and, unlike the `h2_tier_a`/`final_validation_report` interaction, passes the
  real `results_dir` rather than a `tempfile.mkdtemp()` sandbox; `allow_overwrite` was already
  threaded to `train_models`, `train_ann` and `run_final_validation` but not to `refresh_dataset`
  (`:120`), `write_audit_reports` (`:126`) or `run_gold_standard_sensitivity` (`:169`), which would
  have let a legitimate `--allow-overwrite` rerun do expensive training, let `run_final_validation`
  overwrite its 10 files, and only then abort - partial and destructive. All three call sites now
  forward `allow_overwrite` explicitly. **Hazard C**: each guarded function checks only its own
  writes - `refresh_dataset`'s guard covers only `provenance_csv`, `write_audit_reports`'s guard
  covers only its own 3 files - so a union guard cannot make the second call in a pipeline abort
  because the first call already wrote its own file moments earlier in the same run.
  Guards sit at the public function, matching `run_h2_tier_a`/`run_bias_skew_finalization`
  precedent: `refresh_dataset`, `write_audit_reports` and `run_gold_standard_sensitivity` each
  gained an `allow_overwrite: bool = False` parameter and guard themselves as their first
  statement. `data_bias_audit.py`'s `__main__` also gets a defense-in-depth preflight guard
  (`_guard_data_bias_audit_cli`, the union of all 4 tracked-risk paths) placed above the
  `refresh_dataset` call, so a blocked run fails before paying the cost of parsing every IEDB xlsx
  file rather than after. `run_gold_standard_sensitivity`'s call in
  `bias_skew_finalization.py:169` stays positional and unreordered; `allow_overwrite` was appended
  as a trailing keyword argument.
  Tests: `tests/test_data_bias_audit_guard.py` (31 cases) and
  `tests/test_gold_standard_sensitivity_guard.py` (16 cases) cover planned-path enumeration
  (including the derived-filename behaviour, built with the identical `.replace()` expression the
  writer uses, not `os.path.splitext`), guard silence on an empty/nonexistent location, a per-file
  parametrized clobber check, a wiring test proving each guard is actually called by its real
  function and not merely defined, `allow_overwrite` disarming and threading through each
  function, a Hazard-A regression test, and CLI-level checks anchored on argparse's `the following
  arguments are required:` line rather than a bare stderr substring (the guard's own message also
  names these flags, which is exactly what let a prior regression in this line pass a naive
  substring check while live). `tests/test_bias_skew_finalization_results_guard.py` gained 3
  threading tests locking down Hazard B across all three call sites. Both new modules are
  registered in `tests/test_entry_point_help_smoke.py`'s `REQUIRED_OUTPUT_FLAGS` /
  `ALL_ENTRY_POINTS` via a new `MULTI_FLAG_REQUIRED_ENTRY_POINTS` list, since each carries multiple
  required flags rather than the one uniform flag name the existing lists assume.
  A 12-mutation battery (`_local/mutations/step8_data_bias_audit_gold_standard_sensitivity.json`,
  gitignored, local dev tool) targeting this diff specifically - guard-never-called for each of the
  three delegate functions, Hazard A reintroduced, Hazard B dropped from each of the three
  `bias_skew_finalization.py` call sites, Hazard C reintroduced in both directions, both
  derived-filename drops, and the `__main__` preflight guard dropped entirely - confirms the tests
  are load-bearing. **12 of 12 detected.**
  Full suite 1515 passed, 0 failed, 0 errors, 2 skipped under the standing local exclusion of
  `tests/test_run_analysis_results_guard.py`, from 1517 collected, reconciling exactly as the
  pre-branch baseline (1451 passed / 1453 collected) + 64 new cases (14 in
  `tests/test_entry_point_help_smoke.py`, 31 in `tests/test_data_bias_audit_guard.py`, 16 in
  `tests/test_gold_standard_sensitivity_guard.py`, 3 in
  `tests/test_bias_skew_finalization_results_guard.py`).
- **`src/external_benchmark_comparison.py`, `src/prepare_external_validation_inputs.py` and
  `src/external_validation_cross_virus.py` close Tier-1 items #2, #3 and #9 of the `results/`
  silent-overwrite defect-class line** (breaking CLI change): per the Tier-1 enumeration in
  `_local/notes/results-dir-tier1-enumeration-2026-07-30.md` (15 modules; `h2_tier_a_evaluation.py`
  closed item #1 and `data_bias_audit.py`/`gold_standard_sensitivity.py` closed item #8 above), this
  closes items #2, #3 and #9, bringing the closed count to **5 of 15**.
  `src/prepare_external_validation_inputs.py` was already required with no default going into this
  branch (its config-indirection fallback `cfg.get("output_dir", "results")` was removed entirely
  and `run_prepare()` extracted from the old inline `main()`); this branch closes the remaining two.
  `src/external_benchmark_comparison.py`'s `--results-dir` defaulted to `results`, so a bare
  `python -m src.external_benchmark_comparison` run could silently rewrite up to 6 tracked
  comparison artifacts in place, including `external_benchmark_comparison.md` - the report backing
  the head-to-head metric table published in `README.md`. `--results-dir` is now required with no
  default at both the CLI and `run_comparison(results_dir=...)` layers; `results_dir` moved to the
  front of the signature, ahead of the already-defaulted `predig_path`/`prime_path` parameters,
  since Python does not allow a no-default parameter after a defaulted one - the only caller
  (this module's own `main()`) already passed everything by keyword, so nothing breaks. A new
  `planned_external_benchmark_paths()` + `_guard_results_dir()` pair raises `FileExistsError`
  listing every colliding file before any work starts; the guard's message discloses that
  `external_benchmark_comparison.md` is also independently appended to (not overwritten) by
  `external_validation_finalize.py`, a separate write path this guard does not cover.
  `src/external_validation_cross_virus.py`'s `--output` defaulted to
  `results/external_validation_cross_virus.csv`, threatening the single tracked cross-virus
  transfer table. `--output` is now required with no default at both the CLI and
  `run_cross_virus(output_path=...)` layers; unlike the `--results-dir`-shaped guards elsewhere in
  this line, `--output` names a file, not a directory, so `planned_cross_virus_paths()` and
  `_guard_output_path()` use the `scope`/`remedy` override on `guard_planned_paths()` (same shape as
  `src/data_bias_audit.py`'s single-file guards) so the message does not claim a directory-shaped
  destination that does not exist. The only other caller of `run_cross_virus` in the repo,
  `src/external_validation_finalize.py` (`:611-623`), already passes `output_path` explicitly as a
  positional argument, so removing the default does not break it - reconfirmed by grep before this
  change, not just assumed from the prior disclosure, and locked down by a dedicated regression
  test. `README.md`'s `external_benchmark_comparison` example command gained the now-required
  `--results-dir results`, matching its `prepare_external_validation_inputs` neighbor on the
  preceding line.
  Tests: `tests/test_prepare_external_validation_inputs_results_guard.py` (16 cases),
  `tests/test_external_benchmark_comparison_results_guard.py` (19 cases) and
  `tests/test_external_validation_cross_virus_results_guard.py` (15 cases) each cover planned-path
  enumeration, guard-pass-on-empty-location, a per-file parametrized clobber check, a wiring test
  proving the guard is actually called by the real `run_*` function before any read happens (rather
  than merely defined - each guard is the literal first statement, so a bogus/nonexistent input
  path deterministically distinguishes `FileExistsError` from the guard against a
  `FileNotFoundError` the real work would raise instead), an allow-overwrite-passthrough test, a
  Python-level `TypeError`-on-missing-argument test, and CLI-level checks anchored on argparse's
  required-arguments line rather than a bare stderr substring - the guard's own error message names
  the flag too and would give a false pass on a substring check while a regression was live.
  `tests/test_entry_point_help_smoke.py` registers all three: `prepare_external_validation_inputs`
  and `external_benchmark_comparison` join `RESULTS_DIR_REQUIRED_ENTRY_POINTS` (12 new parametrized
  cases across the file's existing checks), and `external_validation_cross_virus` gets a new
  `OUTPUT_REQUIRED_ENTRY_POINTS` list for its `--output` flag, wired into `REQUIRED_OUTPUT_FLAGS`
  and `ALL_ENTRY_POINTS` the same way the existing lists are (49 -> 61 collected cases in this file).
  **Scope note - this does not close the `results/` defect class.** Per the Tier-1 enumeration doc,
  `src/calibration_analysis.py`, `src/shap_analysis.py` and `scripts/compute_population_coverage.py`
  remain open instances and are explicitly out of scope for this branch. The two locations above
  that previously listed `src/external_validation_cross_virus.py` as still open are now stale and
  superseded by this entry - it is closed as of here. Three bare, flagless invocations of
  `python -m src.external_benchmark_comparison` and `python -m src.external_validation_cross_virus`
  survive in prose inside `docs/external_testing/SESTRAV_External_Validation_PRIME_PredIG_Plan.md`
  (historical planning narrative, not a runnable script or CI step); disclosed here rather than
  edited, since rewriting a planning-document's prose is a different kind of change than closing a
  guard gap.
  Full suite 1577 passed, 0 failed, 0 errors, 2 skipped under the standing local exclusion of
  `tests/test_run_analysis_results_guard.py` (7 tests), from 1579 collected, reconciling exactly as
  the pre-branch baseline (1515 passed / 1517 collected) + 62 new cases (16 in
  `tests/test_prepare_external_validation_inputs_results_guard.py`, 19 in
  `tests/test_external_benchmark_comparison_results_guard.py`, 15 in
  `tests/test_external_validation_cross_virus_results_guard.py`, 12 in
  `tests/test_entry_point_help_smoke.py`). `ruff check .` and `mypy src/` are clean on every
  changed and added file.
- **`src/calibration_analysis.py`, `scripts/compute_population_coverage.py`,
  `src/shap_analysis.py` and `src/baseline_comparison.py` close Tier-1 items #7, #11, #10 and #6
  of the `results/` silent-overwrite defect-class line** (breaking CLI change): per the Tier-1
  enumeration in `_local/notes/results-dir-tier1-enumeration-2026-07-30.md` (15 modules; items #1,
  #8, #2, #3 and #9 closed in the entries above), this closes four more, bringing the closed count
  to **9 of 15**. Together the four modules make 12 planned writes, of which **4 are git-tracked**:
  `results/calibration_metrics.csv`, `results/population_coverage_v5.json`,
  `results/shap_values_rf.csv` and `results/baseline_comparison.csv`. **All four are cited in
  tracked documents**, so every one of them sits behind something a reader can see:

  - `results/population_coverage_v5.json` is the source of the panel-coverage prose and table in
    `docs/paper.md:216-233` (EUR 0.919, AFR 0.621, AMR 0.813, EAS 0.742, SAS 0.847, global mean
    0.789 - all six match the artifact exactly). A bare `python scripts/compute_population_coverage.py`
    on `main` silently rewrote the file behind a manuscript table.
  - `results/baseline_comparison.csv` is the source of the "Pipeline Gold-Standard Recovery" table
    at `docs/model_evaluation_summary.md:169-174`, whose row labels ("RF (SESTRAV)", "ANN (MLP)",
    "Binding-only baseline") are emitted by `src/baseline_comparison.py` and by no other module in
    the repo. It is also summarized at `README.md:90` and named by path in
    `results/final_validation_report.md:5`.
  - `results/calibration_metrics.csv` is the source of the **v1** Brier column of the table at
    `results/v1_v2_quality_comparison.md:33-35` (Brier 0.096, Trivial 0.131, BSS 0.265 match the CSV
    exactly). The same table's **v2** column does not match the current artifact, and neither does
    `results/multi_run_stability_report.md:147-148` - see the drift disclosure below.
  - `results/shap_values_rf.csv` is cited by name at `README.md:162` as the supporting evidence for
    the 60/40 SHAP attribution split published at `README.md:157-158` - but see below: the artifact
    does not reproduce that split.

  **This count was wrong three times before it was right, and the sequence is recorded rather than
  quietly fixed.** Successive drafts of this entry said none of the four backed a published number,
  then two, then three. Each figure was written from a partial sweep and revised only where the
  previous round had looked. The correct answer, reached by tracing every artifact's citations
  rather than by adjusting the previous count, is four.

  **Two pre-existing drift/binding defects surfaced while tracing those citations. Neither is
  introduced or repaired by this branch**, and both are disclosed here rather than silently patched,
  because correcting a published number is a separate change that needs its own evidence trail:

  - **`README.md`'s 60/40 SHAP attribution split is unbound.** `README.md:162` names
    `results/shap_values_rf.csv` as its evidence, but all ten `bind_*` columns in that artifact are
    identically `0.0` across all 2000 rows - a 0/100 binding-to-physicochemical attribution, not
    60/40 - and the file carries 2000 rows against the README's stated 720 samples at `README.md:156`.
    The other cited source, `results/external_benchmark_comparison.md`, contains no SHAP methodology
    at all. The same figure appears again at `README.md:92`. So the guard added here protects a file
    that a public claim points at while not actually supporting it.
  - **Three tracked reports carry numbers their source artifacts no longer produce.**
    `results/v1_v2_quality_comparison.md:33,35` publishes v2 Brier 0.170 / BSS 0.198 against the
    current CSV's 0.199 / 0.059; `results/multi_run_stability_report.md:147-148` mirrors the same
    mismatch; and `docs/model_evaluation_summary.md:172-174` publishes RF 6/15, 8/15, 27.1% against
    the current `baseline_comparison.csv`'s 4, 7, 34.65% (XGBoost and ANN rows likewise). This is
    ledger-independent drift of exactly the kind an unguarded regeneration produces, which is the
    argument for the guards in this entry, not against them.

  What remains true, and is the honest severity comparison: **none of the four carries a
  certified-ledger headline metric** the way `h2_tier_a_summary.md` carries the certified
  R10 = 0.9494 result (item #1) - verified against the ledger rather than assumed. This batch is
  closer to defect-class completeness and OpenSSF posture than to a claim-integrity repair of item
  #1's severity, but it is not free of claim-integrity exposure, and an earlier draft that described
  it as such was wrong.
  These were queued as "the mechanical sub-class (a) remainder," and that framing turned out to be
  only partly right: all four do use an argparse default rather than a module-level constant, but
  they have four different internal shapes and two needed a documented deviation from the template.
  `src/calibration_analysis.py` is the clean case: `--output-dir` is now required with no default
  at both the CLI and the `run_calibration_analysis(output_dir=...)` layers, and
  `planned_calibration_paths()` enumerates all 3 writes
  (`calibration_reliability_diagram.png`, `calibration_score_distribution.png`,
  `calibration_metrics.csv`), guarded ahead of `os.makedirs`.
  `scripts/compute_population_coverage.py`'s `--output` names a **file**, not a directory, so like
  `src/external_validation_cross_virus.py` (item #9) it uses the `scope`/`remedy` override on
  `guard_planned_paths()` rather than claiming a directory-shaped destination. The load-bearing
  subtlety: this module resolves a relative `--output` against `PROJECT_ROOT`, not the working
  directory, so the guard runs on the **resolved** path. Guarding `args.output` directly would have
  checked the wrong location on any relative invocation and silently passed while a collision
  existed; a dedicated regression test plants a collision under a monkeypatched `PROJECT_ROOT`,
  changes directory elsewhere, and asserts the planted absolute path appears in the message.
  `src/shap_analysis.py` is the derived-filename case this line has been bitten by twice before.
  Three of its four filename templates are f-string interpolated over a model-tag loop, so
  `planned_shap_paths()` enumerates **7** paths, not 4: `shap_values_{tag}.csv`,
  `shap_summary_{tag}.png` and `shap_bar_{tag}.png` each expand over `rf` and `xgb`, plus
  `shap_waterfall_top_gs.png`, which is written not by `run_shap_analysis` itself but by its
  `_shap_gold_standard_waterfall` delegate and is included for that reason. The tag loop was
  refactored to iterate the same `SHAP_MODEL_TAGS` constant the enumeration reads, so the two
  cannot drift apart. **`--results-dir` on this module is an input flag and was deliberately left
  optional**: it is where the feature CSVs are read from, and making it required would have been a
  regression dressed as a fix. Only `--output-dir` is the output. The tracked
  `results/shareout_20260426/` copies of three of these PNGs live in a frozen share-out directory
  this script never writes to and are correctly **not** enumerated.
  `src/baseline_comparison.py` needed the widest deviation, and it is disclosed rather than hidden.
  Its `--results-dir` is the one **dual-purpose** flag in this line: the run reads its
  `{prefix}_features.csv` inputs out of the same directory it writes `baseline_comparison.csv`
  into. The family's standard remedy clause, "point the flag at a fresh directory," is therefore
  actively wrong advice here, since a fresh directory has no inputs to read and produces a
  different failure; the guard carries a custom remedy saying so explicitly, and a test asserts the
  wrong advice is **absent** from the message. Second, this module has no public function that
  writes: `compare_methods()` returns a DataFrame and the write happens inline under `__main__`. The
  guard is therefore placed in `__main__`, after the pre-existing directory validation and before
  `compare_methods()` is called, so it still aborts ahead of the expensive work. The consequence,
  stated plainly: **this module gets a CLI-level guard only, with no Python-API-level
  `allow_overwrite` parameter**, because it has no Python-level write API to attach one to, and the
  guard message says so rather than advertising an escape hatch that does not exist. Note also that
  `src/baseline_comparison.py` is **named in this changelog as an instance for the first time
  here** - unlike items #7, #10 and #11 it appears in none of the earlier disclosure sweeps or
  scope notes. It was, however, already enumerated as item #6 in
  `_local/notes/results-dir-tier1-enumeration-2026-07-30.md`, the same internal document this
  entry cites for its numbering, so it was known and tracked, just never surfaced publicly. The
  reason it escaped the changelog sweeps is **not** a difference in spelling: its pre-change flag
  name and `default="results"` were identical to `src/shap_analysis.py`'s, which those sweeps did
  catch (the two differ only in help text - "pipeline output CSVs" versus "pipeline feature CSVs").
  It is that the sweeps read `--results-dir` as an input-directory flag, which for this module it
  also genuinely is; the write is the second, less visible half of a dual-purpose flag.
  **Two breaking Python-API signature reorders, called out because they fail silently rather than
  loudly.** `run_shap_analysis(results_dir, model_dir=, output_dir=, ...)` became
  `(results_dir, output_dir, model_dir=, ...)`, and
  `run_calibration_analysis(v2_oof_path, v1_oof_path=, output_dir=, ...)` became
  `(v2_oof_path, output_dir, v1_oof_path=, ...)`. Python does not permit a no-default parameter
  after a defaulted one, so making `output_dir` required forced it forward. Any **positional**
  third-argument caller therefore now binds a different parameter instead of raising. Every caller
  in this repo was converted to keyword form and the conversion is locked by source-reading tests,
  so nothing in-tree breaks, but an external caller could. `scripts/run_analysis.py` additionally
  threads `allow_overwrite` through to `run_shap_analysis`, which it previously did not: without
  that, an `--allow-overwrite` rerun would have completed the gold-standard and baseline stages,
  written both CSVs, and only then aborted on the SHAP guard, leaving partial and destructive
  output. `scripts/regenerate_shareout_pngs.py` passes `allow_overwrite=True` to both
  `run_shap_analysis` and `run_calibration_analysis` deliberately, with an inline comment:
  regenerating exactly those artifacts is that script's declared purpose, so it *is* their
  reproduction path, and without the flag its guard would abort every run after the first.
  Tests: `tests/test_shap_analysis_results_guard.py` (25 cases),
  `tests/test_baseline_comparison_results_guard.py` (17),
  `tests/test_calibration_analysis_results_guard.py` (17) and
  `tests/test_compute_population_coverage_results_guard.py` (15) each cover planned-path
  enumeration, guard-pass-on-empty-location, a per-file parametrized clobber check, and a wiring
  test proving the guard is actually called by the real entry point before any work starts rather
  than merely defined. As in the entries above, the CLI required-flag checks are anchored on
  argparse's own required-arguments line rather than a bare stderr substring, since the guard's
  error message names the flag too and would give a false pass while a regression was live.
  `tests/test_entry_point_help_smoke.py` registers all four (61 -> 77 collected cases in that
  file): `src.calibration_analysis` and `src.shap_analysis` join `OUTPUT_DIR_REQUIRED_ENTRY_POINTS`,
  `src.baseline_comparison` joins `RESULTS_DIR_REQUIRED_ENTRY_POINTS`, and
  `scripts.compute_population_coverage` joins `OUTPUT_REQUIRED_ENTRY_POINTS`.
  **Scope note - this does not close the `results/` defect class.** Six Tier-1 instances remain
  (#4 `scripts/run_loo_cross_virus_v5.py`, #5 `scripts/run_loo_cross_virus_v4.py`,
  #12 `compute_loo_binding_confound.py`, #13 `compute_tier_a_paired_bootstrap.py`,
  #14 `eval_tsnadb_crossdomain.py`, #15 `run_tier_a_benchmarks.py`). **Four of those six**, not all
  six, are the genuinely wider sub-class (b) repair where no output flag exists at all and one must
  be introduced: #12, #13 and #14 do not import `argparse`, and #15 parses only `--smoke` while its
  output path is an inline literal. The remaining two are closer to this batch than a first reading
  of the enumeration suggests: `scripts/run_loo_cross_virus_v5.py:256-257` and
  `scripts/run_loo_cross_virus_v4.py:187-188` already expose `--output-json` and `--output-csv`,
  with the module-level constants supplying only their defaults, so they are the same
  `required=True`-plus-guard shape this batch just applied. An earlier draft of this entry
  described all six as flagless and therefore overstated the remaining work; corrected here after
  reading each of the six rather than inheriting the enumeration's summary column.
  **Three scope notes above are stale as of here and superseded by this entry**, each listing
  `src/calibration_analysis.py`, `src/shap_analysis.py` and `scripts/compute_population_coverage.py`
  as remaining open instances: the one closing the `src/bias_skew_finalization.py` entry, the one
  in the `src/h2_tier_a_evaluation.py` entry, and the one in the item #2/#3/#9 entry. The first two
  are explicitly maintained for supersession and already carry corrections for
  `src/external_validation_cross_virus.py`. An earlier draft of this paragraph named only two of
  the three; the third was found by a full re-sweep of this file rather than by re-checking the
  locations already known.
  Full suite on this branch: **1669 collected, 1663 passed, 4 failed, 0 errors, 2 skipped** under
  the standing local exclusion of `tests/test_run_analysis_results_guard.py`. The 4 failures are
  all four `src.shap_analysis` cases in `tests/test_entry_point_help_smoke.py` and are a property
  of this Windows development machine, not of this branch: importing the real `shap` library
  hard-crashes the interpreter there (`Windows fatal exception: code 0xc06d007f`, raised inside
  `scipy.linalg.inv` from `shap/plots/colors/_colorconv.py` at import time). Because those cases
  spawn a subprocess the crash is contained and they fail rather than taking the run down; the
  guard-side coverage for that module lives in `tests/test_shap_analysis_results_guard.py`, which
  stubs the import out and passes everywhere. **CI on `ubuntu-latest` is the designated authority
  for those 4** - `.github/workflows/ci.yml` runs the suite with no `--ignore`, so it exercises
  both them and the locally excluded file. Stated as the designated authority rather than a
  completed check: at the time of writing this entry no CI run exists for this branch yet, so CI
  green for those 4 is an expectation, not a measurement. The crash itself is confirmed
  pre-existing - `python -m src.shap_analysis --help` returns 127 on an unmodified `main` checkout
  as well as on this branch.
  The count reconciles exactly against a freshly measured baseline: `main` at `2850bab` collects
  1579 under the same exclusion, plus 74 new cases across the four new files, plus 16 in
  `tests/test_entry_point_help_smoke.py` (4 new entry points x 4 parametrized checks each), equals
  1669. **Two corrections to earlier entries in this file, both measured rather than assumed:** the
  1453 collected figure stated in the `src/artifact_guard.py` entry above is stale (the current
  `main` baseline is 1579), and the standing exclusion has been described as skipping "7 tests" -
  in fact, without the exclusion pytest does not report failures at all, it dies during collection
  and **zero** tests run. The same crash reproduces identically on an unmodified `main` checkout,
  verified this session in a throwaway worktree rather than inferred. `ruff check .` and
  `mypy src/` are clean; `bandit` is clean on `src/`, `scripts/` and `tests/`.
- **Pooled same-pathogen AUC-ROC 0.9368 retracted (2026-07-11)**: The pooled within-virus
  "same-pathogen AUC-ROC 0.9368" reported for the e6aafe2 build was decoy-inflated - it only
  reproduces when synthetic / cross-pathogen decoys (incl. the vaccinia panel) are mixed in as
  if they were same-pathogen negatives - and is RETRACTED. Same-pathogen discrimination is now
  reported per-virus (within-CV mean AUC-ROC 0.751; `results/per_virus_eval_v5_mode31.csv`).
  The honest pooled same-pathogen ROC on real IEDB negatives is 0.712 (pooled AUC-PR is
  base-rate-inflated and not a headline). Self-proteome Gate 1 AUC-PR 0.8897 is unaffected.
  The historical e6aafe2 entry below is left intact as the record of what was reported then;
  see `docs/claims_register.md` D12.
  **[Superseded 2026-08-09: the sentence "Self-proteome Gate 1 AUC-PR 0.8897 is unaffected" is
  withdrawn. There is no self-proteome evaluation artifact in this repository and "Gate 1" is a
  GNN promotion threshold (`src/verify/promote_gnn.py:8`), not an RF metric; 0.8897 is the pooled
  CV `auc_pr` of the 2026-06-26 `58bbc15` build, superseded by 0.8312 on the current corpus, and
  both are peptide-leakage-inflated (D15). The bare "AUC-PR 0.7678 ... / 0.8897 self-proteome
  Gate 1" restatement further down this released section carries the same two defects; it is
  annotated here rather than rewritten, because released entries are the record of what was
  reported at the time. See `docs/claims_register.md` D3, D12, D15, D16.]**
- **Per-virus within-CV metrics regenerated (session 70, 2026-07-10)**: The committed
  `results/per_virus_eval_v5_mode31.{csv,json}` lagged the current 35,597-row v5 dataset and
  were regenerated. New within-CV AUC-ROC: CMV 0.819, DENV 0.859, EBV 0.790, HBV 0.708,
  HCV 0.575, HIV-1 0.894, HPV 0.561, IAV 0.856, SARS-CoV-2 0.699 (mean 0.751). HPV within-CV
  (0.561) now falls below the 0.58 Amendment-6 threshold. Leave-one-virus-out (LOO) figures are
  unchanged (mean 0.463; `results/loo_cross_virus_v5_clean.csv`). The earlier Amendment-6
  within-CV values (HPV 0.598, EBV 0.667) recorded below remain the accurate record for the
  e6aafe2 snapshot at which they were achieved.
- **v5 feature ablation added** (`models/v5/ablation/`, `models/v5/training_results_ablation.csv`):
  RF modes 21/31/33/35. Binding scores (mode 21->31) add +0.008 AUC-ROC / +0.015 AUC-PR; modes
  33 and 35 add nothing measurable. Confirms mode-31 as the production configuration.

### Added
- **Dependency-update tooling** (`tools/update_dependencies.py`): CLI wrapper around
  `uv pip compile` that encodes the per-lockfile conventions (interpreter version,
  `--generate-hashes`, `--no-emit-index-url`, pip-compile's unsafe-package handling)
  for all 10 `.in`-sourced manifests. Defaults to `--python-platform linux`
  so a lockfile compiled on a Windows workstation matches the Ubuntu CI runners.
  Supports `--target <pkg>` (single-package bump), `--ci-env <name>` and `--all`.
- **Hash-pin CI gate** (`tools/check_hash_pins.py`, wired into the `lint` job): fails the
  build if any requirement in `requirements.txt` or `environments/requirements-ci-*.txt`
  is missing a `--hash=`, instead of waiting for a `pip --require-hashes` install to
  discover it. Skips comments, `-r`/`-c` includes and option lines, and joins backslash
  continuations so multi-line pip-compile entries are evaluated as one requirement.
- **v5 dataset (31,999 active rows / 46,386 total)**: Rebuilt from merged IEDB API negatives
  pipeline. Key numbers: 36,689 IEDB viral negatives, 4,219 net-new experimentally confirmed
  non-immunogenic peptides added via `scripts/merge_iedb_api_negatives.py` (bridges Pipeline A
  IEDB REST API downloads to Pipeline B v5 schema). Provenance sidecar:
  `data/immunogenicity_dataset_v5_provenance.json`. 17 singleton viruses quarantined (<50 rows
  or <10 real negatives).
- **v5 RF model (mode-31 canonical)**: Retrained on v5 dataset. Evaluation results:
  AUC-PR 0.7678 within-virus (harder same-pathogen discrimination context) / AUC-PR 0.8897
  self-proteome Gate 1 (viral epitopes vs. self-peptide background; Gate 1 threshold protocol) /
  AUC-ROC 0.9368. Per-virus Amendment 6 thresholds met: HPV >= 0.58 (achieved 0.598), EBV >= 0.57
  (achieved 0.667 post-quarantine). OOF predictions: `models/rf_oof_predictions.csv`,
  `models/rf_oof_predictions_mode31.csv`.
- **B*27 EBV conflict quarantine**: 3 rows (FRKAQIQGL x2, RRARSLSAERY) transferred to
  `data/holding/conflicts_v5_preaudit.csv`. These share sequences with label=1 rows for
  other B*27 subtypes; the allele-subtype-specific conflict is documented in claims_register
  Section 5 (ES1). EBV within-virus AUC-ROC: 0.553 (FAIL) -> 0.656 (PASS, beats 0.57 threshold).
- **`scripts/merge_iedb_api_negatives.py`**: New pipeline bridge connecting IEDB REST API
  negative downloads (`data/iedb/*.csv`) to the v5 build schema.
- **ESM-2 embedding cache**: 27,376 peptides pre-computed for GNN v5 training
  (`data/esm2_cache_v5/`). GNN t12 baseline training pending GPU availability.
- **`scripts/download_tcr3d_structures.py`**: Downloads TCR3d 2.0 TSV, applies 4 quality
  filters, downloads ~100 PDB files from RCSB with retry/backoff logic.
- **`scripts/update_contact_weights.py`**: Patches `ALLELE_CONTACT_WEIGHTS` and
  `POPULATION_AVG_CONTACT_WEIGHTS` in `src/features.py` from TCR3d-derived contact frequency
  matrices with `--dry-run` support.

### Changed
- **PyPI publish migrated to OIDC Trusted Publishers**: removed `twine` and the
  `PYPI_API_TOKEN` secret from `release.yml`; replaced with
  `pypa/gh-action-pypi-publish@release/v1` using short-lived GitHub OIDC tokens.
  No static credential is stored anywhere. The `pypi` GitHub environment is
  protected by required-reviewer approval before any upload proceeds.
  Deleted orphaned `environments/requirements-ci-twine.{in,txt}`.
- **`pyproject.toml`**: removed personal email from author metadata; name retained.
- **Library coverage gate raised** `90 -> 95` in `.coveragerc.library`. Verified
  library-scope coverage is **98.91% combined** (~99% statement / ~98% branch;
  measured 2026-06-22), comfortably clearing the new gate and the OpenSSF Gold
  targets. Synced the stale `fail_under` references in `docs/security_compliance.md`,
  `ROADMAP.md`, and the `ci.yml` step comment to the authoritative config value.

### Added
- **HBV/HCV proteome ingestion path** (issue #78): `scripts/fetch_viral_proteomes.py`
  downloads 8 UniProt sequences (HBV panel4: HBcAg P03147, HBx P03165, HBsAg-S P03138,
  HBpol P03157; HCV panel4: Core P26664, NS3 O92972, NS5A O92975, NS5B O92976) with
  HTTP retry logic and writes provenance JSON. `src/naming.py` exposes canonical IDs and
  short aliases; `config.yaml` wires `antigens` + `proteome_files` for both panels.
  Snakemake dry-run passes end-to-end. Suite: **524 tests, 2 skipped, 0 failures**.
- **`docs/antigen_accessions.md`**: sections 3 (HBV) and 4 (HCV) added with full
  UniProt accession table, strain notes, and provenance file references.

### Fixed
- **Lockfile freshness gate reported phantom failures on developer machines**
  (`tools/check_lockfile_freshness.py`): `discover_unmapped_in_files` walked the filesystem
  with a hand-maintained directory denylist that omitted `.claude/`, so every gitignored
  agent worktree - each a full copy of the repo - contributed its 10 `requirements*.in`
  files as unmapped-file errors. Locally this meant 30 phantom errors and exit 1 on a clean
  tree, while CI passed because a fresh checkout has no such copies, so a fail-closed gate
  was green in the one place it ran and unusable everywhere else. Discovery is now scoped to
  git-tracked files (`git ls-files`), matching what the gate is actually about: a lockfile
  pair can only drift in a commit, and an untracked file can never appear in a PR diff. The
  filesystem walk is retained as a fallback for non-git contexts (unpacked sdist) with
  `.claude` added to its denylist. `tests/test_check_lockfile_freshness.py::test_discover_unmapped_in_files_is_empty_for_real_repo`
  was failing on any working copy with worktrees present; 2 regression tests added. The
  repo-data test now drives off `LOCKFILE_PAIRS` and covers all 10 pairs
  (`test_current_repo_hash_completeness_for_all_pairs`): it previously hand-listed 8, excluding
  `requirements.txt` and `environments/requirements.lock` because of accumulated Dependabot
  drift that PR #177's relock has since cleared.
- **`train_one_fold` device-placement bug** (`src/model.py`): `train_one_fold` never
  guaranteed the model was on the resolved device itself - it silently relied on the
  caller. `run_cv` and `train_final_model` already place the model via `.to(device)`
  at construction, so they were unaffected, but any caller constructing a model
  without pre-placement and passing an explicit `device=` argument would hit a
  CPU/GPU tensor mismatch. Found during Windows/Blackwell (RTX 5070 Ti, sm_120) GPU
  bring-up; fixed with `model = model.to(device)` immediately after device
  resolution. Full suite: 1213 passed / 2 skipped / 0 failed.
- **`external_predictors.py` coverage 88% -> 100%** (issue #77): 13 targeted tests
  covering proline/PDE/RKYFW mock-score paths, OOB index in `parse_netchop_html`,
  poll-success return, TAPreg threshold kwarg, and parse success/empty-parse branches.
  Removed dead conditional `if mock_fallback or True:` in `query_netchop`; deleted
  unreachable `raise RuntimeError`; annotated structurally unreachable
  `except (ValueError, IndexError)` with `# pragma: no cover`.

### Security
- **Orphaned doc-cited commit SHAs (20 dead citations across 12 files)**: Fixed at
  the root cause - a prior `git rebase --signoff` backfill rewrote every SHA in
  the rebased range, silently breaking commit citations in tracked docs and
  provenance sidecars. Added a `prepare-commit-msg` hook that appends the DCO
  `Signed-off-by` trailer at commit time (removing the reason to ever backfill),
  and a CI gate (`.github/workflows/doc_commit_refs.yml`,
  `scripts/check_doc_commit_refs.py`) that fails a PR if any cited commit SHA is
  dead or unreachable from the base branch. Repointed the 7 live successor SHAs
  (`data/iedb_negatives_v5_provenance.json`, `lanl_hiv_v5`, `vdjdb_v5`,
  `viral_decoys_{denv,ebv,hiv1,iav}_provenance.json`) by tree-identity match;
  `models/peptide_binding_matrix_v3.provenance.json`'s `git_sha` was set to
  `null` (object unrecoverable) with a note pointing to the verifiable
  commit that added the file instead of a guessed SHA.
- **Alert #51 (HIGH - Token-Permissions)**: Fixed - `dco.yml` top-level
  `permissions: contents: read / pull-requests: read` added (commit `eef10c7`).
- **Alert #52 (MEDIUM - Pinned-Dependencies)**: Dismissed false positive - pip
  smoke-test install cannot be hash-pinned by design.
- **Alert #50 (HIGH - Token-Permissions)**: Dismissed won't-fix - `contents: write`
  required by `gh release create`; top-level `permissions: read` already restricts
  all other jobs.
- **Alert #15 (HIGH - CVE-2025-3000, torch)**: Dismissed won't-fix - no upstream
  patch; `torch.jit.script` not exposed to untrusted input; EPSS 0.08%; will reopen
  when PyTorch releases a fix. **(SUPERSEDED 2026-08-05 - torch 2.13.0 shipped the
  fix and the advisory is now resolved by upgrade, not dismissal. See the torch
  entry in the Security section at the top of this release.)**
- **Dependabot #35 (torch CVE-2025-3000)**: Dismissed `tolerable_risk` - same
  rationale as alert #15. **(SUPERSEDED 2026-08-05, same as above.)**
- **Dependabot #99-#103 (5x HIGH - GitPython URL/config injection and env-var
  expansion, secret exfiltration on fetch)**: Fixed - `gitpython` bumped 3.1.52
  -> 3.1.54 (PR #157, clears GHSA-r9mr-m37c-5fr3 / GHSA-6p8h-3wgx-97gf /
  GHSA-fjr4-x663-mwxc / GHSA-3rp5-jjmw-4wv2) -> 3.1.55 (PR #158/#159, clears
  GHSA-94p4-4cq8-9g67) in both hash-pinned lockfiles (`environments/requirements-ci.txt`,
  `environments/requirements.lock`). CI-only transitive dependency (`# via snakemake`);
  not on the runtime peptide-scoring path.

## [2.0.3] - 2026-06-17

This release delivers the next test-coverage and CI hardening pass: 154 new
unit tests, a library-coverage ratchet advance to 90%, and a complete PyPI
publish pipeline.

### Added
- **154 new unit tests** across 9 test modules, bringing the total suite to
  **476 passing tests**:
  - `test_features_advanced` (23): `compute_sample_weights` virus/length
    correction, `compute_features_for_dataset` vectorised batch extraction,
    and `compute_weisfeiler_lehman_features` WL kernel.
  - `test_iedb_extractor` (43): full branch coverage of
    `src/verify/iedb_multi_virus_extractor.py` - REST mocking, VDJdb TSV
    parsing, decoy generation, `process_target`, and `main()` entry point.
  - `test_promote_gnn_runner` (15): `check_promotion_gates` short-circuit and
    aggregation logic; `promote_model` config-mutation and checksum behaviour.
  - `test_sestrav_evaluator_extended` (15): pipeline-runner paths in
    `src/verify/sestrav_evaluator.py`.
  - `test_statistical_bootstrap` (9): 98% coverage of
    `src/statistical_bootstrap.py` including the joblib worker path via
    `_inline_parallel` mock.
  - `test_train_gnn_dataset` (16): `GraphPeptideDataset` and `set_seed`
    reproducibility in `src/train_gnn.py`.
  - `test_external_predictors_extended` (13): proline/PDE/RKYFW mock-score
    paths, OOB index in `parse_netchop_html`, successful poll return, TAPreg
    threshold kwarg, and parse success/empty-parse branches.
  - `test_model_extended` (7): CUDA mock paths for `get_device` and
    `set_seeds`; `device=None` auto-detect in `train_one_fold`, `run_cv`, and
    `train_final_model`; epoch-exhaustion and `best_state=None` branches.
  - `test_features_graph` (13): `get_cb_cb_edges` for 9/10/11-mers, ERAP
    short-flanking-sequence padding, and `compute_sample_weights` without a
    peptide column.
- **PyPI publish job** in `.github/workflows/release.yml`: runs after the
  existing build/attest/GitHub-Release job; installs twine from a hash-pinned
  lockfile (`environments/requirements-ci-twine.{in,txt}`); uploads sdist and
  wheel; smoke-tests the published package from the live PyPI index. Skipped
  automatically when `PYPI_API_TOKEN` is not configured.
  - Includes a `checkout` step (without which `requirements-ci-twine.txt`
    would not be present on the runner - a bug caught pre-commit).

### Fixed
- **7 mypy type errors** resolved across four source files:
  - `src/data_curation_qc.py`: renamed lambda capture variable to eliminate a
    late-binding closure error.
  - `src/expand_negatives.py`, `src/external_benchmark_comparison.py`:
    `type: ignore[index]` annotations for typed-dict list subscripts on
    `GOLD_STANDARD_NEGATIVES`.
  - `src/external_benchmark_comparison.py`: renamed a list variable that was
    shadowed by a later `np.array` assignment of the same name.
  - `src/verify/structural_gnn.py`: `type: ignore[misc]` for the dynamic
    `Dataset if HAS_PYG else object` base class.

### Changed
- **Library coverage ratchet** advanced: `fail_under` in `.coveragerc.library`
  raised from 85 -> **90**. Actual library coverage is **96.03%** combined
  statement+branch (~96% statement, ~94% branch) - both above the OpenSSF
  Gold targets (>=90% statement, >=80% branch).
- **Whole-repo coverage floor** unchanged at 33 (`pyproject.toml`); actual is
  33.74%. Executable research scripts (those with a `__main__` guard) are
  validated by integration tests and CI gates, not unit statement coverage.

---

## [2.0.2] - 2026-06-16

This release completes the OpenSSF Best Practices hardening pass: governance and
assurance documentation, automated signed releases with build provenance, and a
two-scope test-coverage regime meeting the Gold coverage targets.

### Added
- **OpenSSF governance & assurance documentation**: `GOVERNANCE.md`, `ROADMAP.md`, `BUS_FACTOR.md`, `CONTRIBUTORS.md`, `docs/threat_model.md`, and `docs/security_review.md`.
- **Signed releases with provenance**: `.github/workflows/release.yml` builds the distribution on a version tag and publishes a keyless SLSA build-provenance attestation (Sigstore via GitHub OIDC), guarded by a fail-fast tag/version consistency check. Verification and the release procedure are documented in `docs/releasing.md` and `SECURITY.md`.
- **Two-scope test-coverage measurement**: library-scope coverage via `.coveragerc.library` (OpenSSF Silver `test_statement_coverage80`), kept in sync mechanically by `tools/check_library_coverage.py`, with a subprocess-coverage hook (`tools/coverage_subprocess`). Library coverage raised to ~91% statement / ~81% branch (OpenSSF Gold targets) with new unit tests.

### Fixed
- **Stage 4 MC-dropout path**: corrected a missing `import torch` on the uncertainty-scoring branch.
- **IEDB data loader**: added a missing `import sys`.
- **PRIME wrapper**: corrected a `temp_peptides_file` reference.
- **Dependency Security Vulnerabilities**: Additional dependency hardening applied after the v2.0.0-rc1 tag and re-compiled with `pip-compile --generate-hashes --allow-unsafe`:
  - `tornado==6.5.6` (mitigates four advisories surfaced by the OSSF Scorecard OSV scan): GHSA-fqwm-6jpj-5wxc (cookie attribute injection, high), GHSA-qjxf-f2mg-c6mc (DoS via multipart parts, high), GHSA-78cv-mqj4-43f7 (incomplete cookie validation, medium), and GHSA-cx3h-4qpv-8hc9 (out-of-bounds memory access, low). The 6.5.6 release also restores `manylinux_2_28` wheel availability (absent from 6.5.5).
  - `protobuf==7.35.1` (patch bump over the 7.35.0 baseline shipped in rc1).

### Changed
- **License detection**: `LICENSE` now opens with the canonical `MIT License` text so GitHub and automated tooling identify it as MIT (the SPDX identifier is retained in `pyproject.toml`).
- **Dependency updates** (Dependabot): `starlette` 1.1.0->1.3.1 (#75), `aiohttp` 3.14.0->3.14.1 (#74), and a Python minor/patch group of six updates (#73).

### Security
- **Hash-pinned security-scanner installs**: The `semgrep` and `pip-audit` jobs in `security.yml` now install from hash-pinned lockfiles (`environments/requirements-semgrep.txt`, `environments/requirements-pip-audit.txt`) via `pip install --require-hashes`, resolving the OpenSSF Scorecard *Pinned-Dependencies* findings. Lockfiles are generated from `.in` sources with `pip-compile --generate-hashes`.

---

## [2.0.0-rc1] - 2026-06-10

This release candidate for SESTRAV v2.0 focuses on API & frontend demo containerization, OpenSSF Scorecard compliance, and strict security and reproducibility gating.

### Added
- **FastAPI prediction microservice**: Deployed a scalable FastAPI backend (`api/main.py`) with strict input schema validation (amino acid IUPAC constraints) and cached singleton model loading.
- **Streamlit interactive interface**: Built a frontend GUI (`app/demo.py`) supporting single-peptide predictions, real-time SHAP waterfall visualizations (headless-compatible via matplotlib Agg backend), and dynamic PDF report generation.
- **Docker Compose orchestration**: Standardized deployments via a two-service compose stack (`Dockerfile.api`, `Dockerfile.demo`, `docker-compose.yml`) utilizing local-only loopback binds (`127.0.0.1`) for local research environment safety.
- **PII & absolute path gatekeeper**: Added a pre-merge action workflow (`.github/workflows/pii_scan.yml`) to block commits containing machine-specific filesystem path leaks (Windows user-profile or WSL mount paths) or unresolved TODO placeholders.
- **Hypothesis Property-Based Fuzzing**: Integrated standard property-based fuzz tests in CI (`.github/workflows/fuzzing.yml` and `tests/test_fuzz.py`) with customizable test example ranges (200 for standard pushes, 1000 for weekly schedules).
- **Consensus Rank Aggregation**: Added Borda count-based rank aggregation ensemble inside `src/consensus_ensemble.py` as a robust alternative to geometric mean pooling to bypass zero-cancellation issues.
- **Aho-Corasick Contamination Gate**: Deployed a dedicated verification step in Stage 3 to screen IEDB evaluation records against the training corpus for exact and substring contamination.
- **License SPDX Identifier**: Added machine-readable `SPDX-License-Identifier: MIT` tag to `LICENSE` for automated OSSF Scorecard detection.

### Changed
- **Cross-Platform Path Standardization**: Standardized absolute Windows filesystem paths (`C:\Users\gavin\...`) to relative, POSIX-compliant expressions (`Path` bindings, `.relative_to().as_posix()`, and relative markdown paths) across `README.md`, `src/verify/sestrav_evaluator.py`, and `scripts/benchmark_runner.py` to allow execution on UNIX/Linux/WSL hosts.
- **GitHub Actions Security Hardening**: Pinned all upstream action runners to secure, verified commit SHAs rather than mutable version tags. Locked down workflow run tokens to a strict `permissions: read-all` default state.
- **Branch Rulesets & Review Gating**: Applied automated branch protection configurations via Git credential tokens:
  - Required PR reviews for external contributors while allowing frictionless self-merge bypasses for the repo owner.
  - Enforced strict merge gates requiring status checks (`Require human review` and `SESTRAV CI / test (3.13)`) to pass on clean branches.
  - Restricted branch deletions and force pushes on `main`.

### Removed
- **Unused Stub Codes**: Removed orphaned empty duplicate stubs of `run_evaluation_pipeline` in `src/verify/sestrav_evaluator.py` to prevent naming clashes.

### Fixed
- **Dependency Security Vulnerabilities**: Upgraded minimum versions for vulnerable libraries in `requirements.in` and compiled the hashes using `pip-compile --generate-hashes --allow-unsafe`:
  - `keras==3.14.1` (mitigates GHSA-36fq-jgmw-4r9c, GHSA-4f3f-g24h-fr8m, GHSA-cjgq-5qmw-rcj6, GHSA-hjqc-jx6g-rwp9, GHSA-mq84-hjqx-cwf2, GHSA-7gcm-g887-7qv7).
  - `protobuf==7.35.1` (mitigates Any-message DoS recursion vulnerability GHSA-m2f8-v8q4-3m59).
  - Pinned `nvidia-nccl-cu12==2.30.4` transitive hash matching on Linux systems.
- **Git Index Cleanup**: Cleaned up the tracking index by appending transient test/CI output artifacts (e.g. `ci_install_test.log`, `temp_test_out.txt`, and `bandit_text.txt`) to `.gitignore` to prevent tracking of local runtime logs.
- **Semgrep Custom Rules**: Restructured rules in `semgrep-rules/sestrav-custom.yml` to remove overly-broad match patterns triggering false positives on safe `load_verified_joblib` operations.

---

## Version 2.0.0 (2026-06-04)

### Release Summary

SESTRAV v2.0.0 finalizes the semester core pipeline and integrates advanced computational biology models and validation tracks for public release using the **v2.0.0-alpha dataset** (expansion_alpha).

- **Canonical release track**: **30-feature integrated model/config** (20 physicochemical + 10 multi-allele MHC binding features)
- **Secondary/Optional track**: Neural Network (FlexibleMLP) and Graph Neural Network (GCN/GAT) benchmark modules
- **Legacy comparator track**: **21-feature sequence-only configuration** (for historical comparison)
- **Training dataset**: **v2.0.0-alpha** (1004 peptides, 3.35:1 class ratio)

### What Is Included

- **Four-stage pipeline**: Peptide generation, multi-allele binding prediction, feature extraction, and immunogenicity scoring (RF and XGBoost)
- **FlexibleMLP Extension**: PyTorch ANN classification with 14-configuration hyperparameter architecture search
- **GNN Benchmark Suite**: GCN, GAT, and Bipartite Peptide-Allele graphs for structure-based benchmarking
- **Ablation Studies**: Multi-group feature ablation analyses to quantify contact-residue contribution
- **Final validation bundle generation**:
  - `results/gold_standard_validation.csv`
  - `results/baseline_comparison.csv`
  - `results/h2_tier_a_summary.csv`
  - `results/final_validation_report.md`
- **Security & Dependency Hardening**:
  - Refactored scripts clean of `bandit` security findings (such as shell injections, path handling, and try-catch safety)
  - Upgraded dependencies inside `environments/requirements.lock` resolving 9 CVEs/vulnerabilities
- **Multi-run stability evidence**: `results/multi_run_stability_report.md` demonstrating perfect deterministic reproducibility
- **Platt calibrator refit** on the v2 class distribution to output calibrated probabilities

### Key Results (v2)

- RF AUC-ROC: `0.5684` | AUC-PR: `0.8047` | ISSR@10: `0.7895` | ISSR@25: `0.8285`
- Gold-standard positive recovery: `15/15` found, `7/15` in top 25% (R10 = 0.9494)
- Gold-standard negative discrimination: `9/10` pushed down (TCR features add value)
- SHAP feature split: 60% binding / 40% TCR-contact features
- H2 Tier A decision: **NOT SUPPORTED** ($R_{10} = 0.9494$, below standard threshold)

### Reproducibility Commands

These are the commands as run for the 2.0.0 release, kept verbatim as a historical record.
They no longer execute as written: `--model-dir` later became a required flag on
`src.train_classifier`, so the two `train_classifier` lines now exit with an argparse error.
The release wrote into `models/`, which today holds published artifacts that the overwrite
guard refuses to replace without `--allow-overwrite`. To re-run this sequence now, pass
`--model-dir models/local` on both `train_classifier` lines and repoint the
`final_validation_report` line's `--model-dir` and `--model-path` at `models/local` to read
back what those two lines just wrote.

```bash
conda env create -f environment.yml
conda activate sestrav
pip install snakemake
mhcflurry-downloads fetch models_class1_presentation
python -m src.train_classifier --data data/immunogenicity_dataset_v3.csv --feature-mode 30 --binding-matrix models/peptide_binding_matrix_v3.csv
python -m src.train_classifier --data data/immunogenicity_dataset_v3.csv --feature-mode 21
python -m pytest tests/ -v
snakemake --snakefile pipeline.smk --cores 4
python -m src.final_validation_report --results-dir results --model-dir models --data data/immunogenicity_dataset_v3.csv --binding-matrix models/peptide_binding_matrix_v3.csv --model-path models/rf_30feature_integrated.joblib --dataset-mode expansion_alpha --dataset-version 2.0.0-alpha
python -m src.release_bundle --output-dir release_artifacts --bundle-name sestrav-v2
```

### Known Environment Notes

- Base Python 3.13 is not compatible with this `mhcflurry` stack. Use the project conda env with Python 3.11.
- `setuptools==80.9.0` is required for `pkg_resources` compatibility with the `mhcflurry` release.
- Model serialization warnings may appear if scikit-learn versions differ from the model training environment (models should be trained fresh each cycle).
- XGBoost SHAP TreeExplainer has compatibility issues with the `shap` library version; RF SHAP (the canonical model) works correctly.

### Canonical Decision Statement

The 30-feature integrated track is selected as canonical because it best balances:
- predictive performance evidence,
- biological defensibility,
- reproducibility readiness, and
- alignment with proposal scope.

The v2 dataset is selected over v1 because:
- Class balance is more honest (3.35:1 vs 5.58:1)
- 63% more negative training examples (231 vs 141)
- Gold-standard negative discrimination (9/10 pushed down) is a novel capability
- TCR features contribute 40% of model explanation power (SHAP)

The 21-feature track remains documented as a legacy comparator.

### Limitation and Claim Boundary (Required)

SESTRAV v2 should be communicated as a reproducible computational prioritization prototype. It should not be described as biologically or clinically validated in this release.

Use:
- `docs/limitations_statement_v1.md`
- `docs/archive/colloquium_evidence_freeze_v2_20260524.md`
- `results/final_validation_report.md`

for standardized non-overclaim language and current supported statements.

---

## Version 1.0.0 (2026-04-01)

### Release Summary

Historical baseline release of SESTRAV using the v1 dataset (928 peptides, 5.58:1 class ratio) with the legacy 21-feature sequence-only comparator track.

### Key Results (v1)

- RF AUC-ROC: `0.820` | AUC-PR: `0.953` | Above-trivial AUC-PR: `+0.105`
