# Changelog

All notable changes to the SESTRAV project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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
  collected (reconciling as 1430 + 23).
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
  repaired in #177. The pins they added were redundant anyway: `torch==2.12.0` and
  `torch-geometric==2.7.0` are already pinned and hashed in `requirements.txt`, leaving only
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
  (`results/external_validation_cross_virus.csv`). These are disclosed here, not fixed: each needs
  the same per-file write enumeration this entry used before a guard can be built correctly, and
  that list is not assumed exhaustive.
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
  `src/shap_analysis.py`, `scripts/compute_population_coverage.py`, and
  `src/external_validation_cross_virus.py` remain open instances of the same defect class.
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
- **Pooled same-pathogen AUC-ROC 0.9368 retracted (2026-07-11)**: The pooled within-virus
  "same-pathogen AUC-ROC 0.9368" reported for the e6aafe2 build was decoy-inflated - it only
  reproduces when synthetic / cross-pathogen decoys (incl. the vaccinia panel) are mixed in as
  if they were same-pathogen negatives - and is RETRACTED. Same-pathogen discrimination is now
  reported per-virus (within-CV mean AUC-ROC 0.751; `results/per_virus_eval_v5_mode31.csv`).
  The honest pooled same-pathogen ROC on real IEDB negatives is 0.712 (pooled AUC-PR is
  base-rate-inflated and not a headline). Self-proteome Gate 1 AUC-PR 0.8897 is unaffected.
  The historical e6aafe2 entry below is left intact as the record of what was reported then;
  see `docs/claims_register.md` D12.
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
- **`external_predictors.py` coverage 88% → 100%** (issue #77): 13 targeted tests
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
  when PyTorch releases a fix.
- **Dependabot #35 (torch CVE-2025-3000)**: Dismissed `tolerable_risk` - same
  rationale as alert #15.
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
  raised from 85 → **90**. Actual library coverage is **96.03%** combined
  statement+branch (≈96% statement, ≈94% branch) - both above the OpenSSF
  Gold targets (≥90% statement, ≥80% branch).
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
- **Two-scope test-coverage measurement**: library-scope coverage via `.coveragerc.library` (OpenSSF Silver `test_statement_coverage80`), kept in sync mechanically by `tools/check_library_coverage.py`, with a subprocess-coverage hook (`tools/coverage_subprocess`). Library coverage raised to ≈91% statement / ≈81% branch (OpenSSF Gold targets) with new unit tests.

### Fixed
- **Stage 4 MC-dropout path**: corrected a missing `import torch` on the uncertainty-scoring branch.
- **IEDB data loader**: added a missing `import sys`.
- **PRIME wrapper**: corrected a `temp_peptides_file` reference.
- **Dependency Security Vulnerabilities**: Additional dependency hardening applied after the v2.0.0-rc1 tag and re-compiled with `pip-compile --generate-hashes --allow-unsafe`:
  - `tornado==6.5.6` (mitigates four advisories surfaced by the OSSF Scorecard OSV scan): GHSA-fqwm-6jpj-5wxc (cookie attribute injection, high), GHSA-qjxf-f2mg-c6mc (DoS via multipart parts, high), GHSA-78cv-mqj4-43f7 (incomplete cookie validation, medium), and GHSA-cx3h-4qpv-8hc9 (out-of-bounds memory access, low). The 6.5.6 release also restores `manylinux_2_28` wheel availability (absent from 6.5.5).
  - `protobuf==7.35.1` (patch bump over the 7.35.0 baseline shipped in rc1).

### Changed
- **License detection**: `LICENSE` now opens with the canonical `MIT License` text so GitHub and automated tooling identify it as MIT (the SPDX identifier is retained in `pyproject.toml`).
- **Dependency updates** (Dependabot): `starlette` 1.1.0→1.3.1 (#75), `aiohttp` 3.14.0→3.14.1 (#74), and a Python minor/patch group of six updates (#73).

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
