#!/usr/bin/env python3
"""Compute the honest pooled same-pathogen discrimination metric (Def A).

Background: the retired headline "same-pathogen AUC-ROC 0.9368" only reproduced when
easy negatives (the out-of-panel Orthopoxvirus vaccinia bloc and the synthetic
allele-matched non-binders) were pooled in AND the folds were left ungrouped. The
splitter's contribution is measurable directly: on the same corpus, the previously
reported pooled mode-31 AUC-ROC was 0.9429 before Phase 0 and is 0.8137 after (compare
`git show 30f1b76^:models/v5/training_results_mode31.csv` against the current file).
That commit also carried the `_bin_origin` stratification fix, which D15 measured as
metric-neutral on the audit's corpus-refit frame (AUC-PR 0.8347 -> 0.8343,
results/cv_leakage_audit.csv - note both the metric and the frame differ from the
AUC-ROC pair above and must not be conflated with it), so the move above is
attributable to grouping. Peptide
leakage therefore contributed materially AS WELL AS the easy negatives - this is not
a decomposition and the two are not orthogonal: restricting the grouped frame to
Def A negatives moves 0.8137 -> 0.6015 separately. Do not read either delta as that
factor's isolated share. D12 is FULLY SUPERSEDED BY D15 - neither 0.9368 nor the "honest"
correctives this script's row once carried (0.712 / 0.751) may be cited as current.
Scope note: every figure named here is computed on the ACTIVE (non-quarantined)
pool. v5 quarantines 15,588 of 51,185 rows, including all 5,000
self_proteome_decoy rows, so none of this is a claim about every negative in the
corpus. Note that the vaccinia bloc is NOT a
hard-decoy panel, as this docstring asserted until 2026-08-10: those 21,432 rows are
genuine IEDB tier-1 assay records (database_source == IEDB for all of them), so they
are out-of-panel rather than synthetic. That earlier wording was the proximate source
of a false attribution that reached the manuscript - see docs/claims_register.md D19.
The honest same-pathogen figure restricts negatives to real IEDB REST-API
viral negatives (negative_origin == 'iedb_api') for the 9 target viruses - this is
"Def A" in _local/notes/pooled_within_virus_metric_recompute.md and the number cited
in docs/model_evaluation_summary.md.

This script makes that number reproducible and binds it to a canonical results/ CSV so
the integrity harness can reconcile it (retiring the last unbound-orphan of its class).

Reproduce:  python scripts/compute_pooled_honest_metric.py \
                --output results/pooled_honest_same_pathogen.csv

--output has no default, matching scripts/compute_loo_binding_confound.py and
scripts/compute_tier_a_paired_bootstrap.py: the output is git-tracked and is bound by the
integrity harness, so a bare run prints the row and writes nothing rather than silently
rewriting certified output. Note this file embeds `generated_utc` as a DATA COLUMN, so a
regeneration is never a no-op even when every metric is unchanged - do not re-run it just
to confirm reproducibility, compare the printed row instead.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

TRACKED_OUTPUT = "results/pooled_honest_same_pathogen.csv"

TARGET_VIRUSES = {"CMV", "DENV", "EBV", "HBV", "HCV", "HIV-1", "HPV", "IAV", "SARS-CoV-2"}
# Def A real-negative origin: real IEDB REST-API viral negatives only. tested_negative is
# dominated by the out-of-panel Orthopoxvirus vaccinia bloc (real IEDB assay negatives,
# not decoys - see D19) and allele_matched_nonbinder is synthetic; both are excluded so
# the metric reflects genuine same-pathogen discrimination.
REAL_NEG_ORIGIN = "iedb_api"


def compute(oof_path: Path) -> dict:
    df = pd.read_csv(oof_path)
    is_pos = df["label"] == 1
    is_real_neg = df["negative_origin"] == REAL_NEG_ORIGIN
    subset = df[df["virus"].isin(TARGET_VIRUSES) & (is_pos | is_real_neg)]
    y = subset["label"].to_numpy()
    s = subset["score"].to_numpy()
    p, r, _ = precision_recall_curve(y, s)
    return {
        "metric": "honest_same_pathogen",
        "definition": "Def A: positives + real IEDB (iedb_api) negatives, 9 target viruses",
        "auc_roc": roc_auc_score(y, s),
        "auc_pr": auc(r, p),
        "n_pos": int((subset["label"] == 1).sum()),
        "n_neg": int((subset["label"] == 0).sum()),
        "source_oof": str(oof_path).replace("\\", "/"),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", default="models/v5/rf_oof_predictions_mode31.csv")
    ap.add_argument(
        "--output",
        default=None,
        help=(
            f"Output CSV path (optional). No default: {TRACKED_OUTPUT} is a git-tracked "
            "artifact bound by the integrity harness, so a bare run prints the row and "
            "writes nothing rather than silently rewriting it."
        ),
    )
    args = ap.parse_args(argv)

    row = compute(Path(args.oof))
    print(f"honest same-pathogen (Def A): AUC-ROC {row['auc_roc']:.4f} "
          f"(-> {row['auc_roc']:.3f}), AUC-PR {row['auc_pr']:.3f}, "
          f"n_pos={row['n_pos']} n_neg={row['n_neg']}")

    if args.output is None:
        print("No --output given: nothing written.")
        return 0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # lineterminator pinned to LF. .gitattributes pins results/*.csv to eol=lf and the
    # integrity harness hashes RAW BYTES, so a bare to_csv() on Windows writes CRLF and
    # records a digest that cannot reproduce from a clean clone. Same pin as
    # scripts/compute_loo_binding_confound.py and scripts/compute_d29_corpus_composition.py.
    pd.DataFrame([row]).to_csv(out, index=False, lineterminator="\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
