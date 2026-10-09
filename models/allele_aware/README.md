# models/allele_aware/ - feature-mode 166 (allele-aware) artifacts

**Status: superseded inputs. These artifacts are retained as the record of an
evaluated-but-not-adopted track, not as a current model.** They are not the
production model, and no number in them is comparable to a current mode-31 figure.

For the status of the allele-aware track itself, read the pan-allele modeling entry in
`ROADMAP.md` and rows D30 and D36 of `docs/claims_register.md`. This file records only
what the directory is and why it is stale; it makes no performance claim and supersedes
no published number.

## What produced these files

Every data artifact here was last written by commit `50322d5` (2026-07-04) from the
166-feature allele-aware set (`FEATURE_COLUMNS_ALLELE` in `src/features.py`, assembled by
`prepare_features_166` in `src/train_classifier.py`, reachable as `--feature-mode 166`).
The only later content change to this directory was `55dd895` and `7ffce7e` (both
2026-08-23), which re-recorded portable blob digests in the checksum manifest. No data
artifact here has been regenerated since 2026-07-04.

The training corpus was `data/allele_aware/IEDB-20260704-MULTI_VIRUS_ALLELE_AWARE-v2.csv`,
whose tracked provenance sidecar in the same directory records 4,567 training pairs,
4,169 unique peptides, 10 alleles and 136 pseudo-sequence dimensions.
`rf_oof_predictions_mode166.csv` carries exactly 4,567 rows, which is how the binding is
established.

## What has moved since

1. **The corpus.** The shipped `config.yaml` has pointed at the v5 corpus since `bcc1ea0`
   (2026-09-02). The production out-of-fold record `models/v5/rf_oof_predictions.csv`
   carries 35,555 rows against this directory's 4,567. The two tracks are not evaluated on
   the same data.
2. **The splitter.** The peptide-level cross-validation leakage fix landed in `30f1b76`
   (2026-08-10), after these artifacts were written. It re-baselined the mode-31 arm; the
   mode-166 out-of-fold scores here were never regenerated and still reflect the
   pre-remediation splitter. Any mode-166-versus-mode-31 comparison drawn from these files
   therefore compares two different evaluation protocols. `ROADMAP.md` states the
   consequence in full.
3. **The producing code.** `src/train_classifier.py` and `src/features.py` have both
   changed repeatedly since 2026-07-06, and nothing here has been regenerated against any
   of it. A commit count is deliberately not quoted: that question answers differently
   under default history simplification than with `--full-history`, and both move with
   every push. Re-measure if you need a figure, and name the instrument.

## Two things to know before reading a filename here

- **The unsuffixed files are mode-166 artifacts too.** `rf_oof_predictions.csv` and
  `rf_oof_predictions_mode166.csv` are the same git blob, as are `training_results.csv`
  and `training_results_mode166.csv`. This directory holds one track, not two, and
  `optimal_thresholds.json` records `"feature_mode": 166`. Basenames repeat across
  `models/`, `models/v5/` and this directory with different contents, which is why
  `src/artifact_integrity.py` refuses to match an artifact by basename.
- **The checksum manifest does not cover everything here.** `model_artifact_checksums.json`
  carries no entry for `rf_oof_predictions_mode166.csv` or `training_results_mode166.csv`.
  Both are tracked, so their integrity rests on git alone.

## Reproducing or retiring this track

Re-opening allele-aware modeling needs a corrected featurization (see D30) and a
symmetric, peptide-grouped re-evaluation of BOTH arms on the current corpus. Re-running the
old comparison against these files does not answer the question.
`scripts/batch_experiment_runner.py` excludes mode 166 from its sweep on the grounds that
it has no released scoring artifact; that exclusion is consistent with this status and is
not a defect to fix.
