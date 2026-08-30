"""Compute the pooled (single-pass) cross-validation metrics for the canonical mode-31 RF.

WHY THIS EXISTS
---------------
`models/v5/training_results_mode31.csv` carries `rf_cv_mean` = 0.605827, which is the
FOLD-MEAN of five per-fold AUC-PR values. The POOLED figure - one precision-recall
computation over all out-of-fold predictions concatenated - is 0.605524, and PR #303
propagated it to eight reader-facing files while no artifact in the repository held it.
That makes it an unbound number the integrity harness cannot see drift (the D16 class).

This script materializes it so a `kind = "cell"` claim can bind it.

THE TWO QUANTITIES ARE NOT INTERCHANGEABLE AND BOTH ARE CORRECT:
    pooled    = average_precision_score(all_labels, all_oof_scores)   -> 0.6055
    fold-mean = mean(per-fold average_precision_score)                -> 0.6058
A genuine single-pass pooled computation has no fold spread to report, which is why no
`std` is emitted for the pooled rows. That convention already exists in
`results/cv_leakage_audit.csv`, where `pooled_honest_same_pathogen_auc_pr` carries
std = NaN while every fold-mean carries a value.

The fold-mean is emitted alongside deliberately, as a CONTROL that the OOF frame and the
training summary describe the same run.

BUT THE CONTROL HOLDS ONLY TO A TOLERANCE, and this paragraph overstated it until
2026-08-31. It read "it MUST reproduce `rf_cv_mean`. If it does not, the OOF frame and the
training summary describe different runs and nothing here is publishable." Measured:

    this script's fold-mean AUC-PR : 0.6058259889375138
    training_results_mode31.csv    : 0.6058273241425536   (delta 1.34e-06)
    ROC analogue                   : 0.8137012751 vs 0.8137012070 (delta 6.8e-08)

Both round to the 4 dp every document publishes (0.6058, 0.8137), and the deltas are
ordinary floating-point noise between two code paths over the same folds - NOT evidence of
different runs. Acting on the old wording at the 6 dp it quoted would therefore have
condemned a sound artifact. Compare at 4 dp, or state an explicit tolerance; do not assert
exact equality. Note also that nothing here or in the test suite actually performs this
comparison against the real ledger - the tests use synthetic fixtures - so the control is
descriptive, not enforced.

--output HAS NO DEFAULT, by the same "no-default explicit path" rule as
`scripts/compute_loo_binding_confound.py` and `scripts/evaluate_per_virus.py`:
`results/pooled_cv_metrics_mode31.csv` is git-tracked, and a script that defaults to it
silently rewrites certified output on any bare invocation. A bare run prints and writes
nothing.

CRLF HAZARD: `core.autocrlf` is true on Windows checkouts and the integrity harness hashes
raw bytes. `.gitattributes` pins `results/*.csv text eol=lf`, so the CSV MUST be written
with lineterminator="\n" or the recorded digest is non-portable and the harness FAILs it as
such. See `src/h2_tier_a_evaluation.py` for the same guard.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from typing import NamedTuple, Sequence

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OOF = "models/v5/rf_oof_predictions_mode31.csv"
TRACKED_OUTPUT = "results/pooled_cv_metrics_mode31.csv"


class Row(NamedTuple):
    metric: str
    value: float
    kind: str


def compute(oof_path: pathlib.Path) -> list[Row]:
    df = pd.read_csv(oof_path)
    missing = {"label", "score"} - set(df.columns)
    if missing:
        raise SystemExit("FATAL: {} lacks required column(s) {}".format(oof_path, sorted(missing)))
    y = df["label"].to_numpy()
    s = df["score"].to_numpy()

    rows = [
        Row("mode31_pooled_auc_pr", float(average_precision_score(y, s)), "pooled_single_pass"),
        Row("mode31_pooled_auc_roc", float(roc_auc_score(y, s)), "pooled_single_pass"),
        Row("mode31_pooled_n_rows", float(len(df)), "count"),
        Row("mode31_pooled_n_positive", float(int(y.sum())), "count"),
    ]
    if "fold" in df.columns:
        per_fold = [
            float(average_precision_score(g["label"], g["score"]))
            for _, g in df.groupby("fold")
        ]
        rows.append(Row("mode31_fold_mean_auc_pr", sum(per_fold) / len(per_fold), "fold_mean"))
    return rows


def _resolve(rel_or_abs: str) -> pathlib.Path:
    p = pathlib.Path(rel_or_abs)
    return p if p.is_absolute() else REPO_ROOT / p


def _sidecar_key(path: pathlib.Path) -> str:
    """Portable key for the provenance `inputs` map: repo-relative, forward slashes.

    Never record the raw CLI string. Doing so let a Windows-style relative argument write
    BACKSLASHES into a git-tracked sidecar, and an absolute argument write this
    workstation's home-directory path into one, so identical inputs produced different
    sidecar bytes on Windows and Linux. Neither form is caught by `.gitattributes` (it
    normalizes line endings, not path separators); only the absolute case is caught, by
    pre-commit Gate 4 - which, fittingly, rejected the first draft of this very docstring
    for quoting such a path as an example. Mirrors the normalization in
    `scripts/compute_pooled_honest_metric.py`.
    """
    resolved = path.resolve()
    try:
        rel: pathlib.Path = resolved.relative_to(REPO_ROOT)
    except ValueError:
        rel = resolved
    return str(rel).replace("\\", "/")


def write(rows: Sequence[Row], out_path: pathlib.Path) -> pathlib.Path:
    """Write the metric table as LF CSV and return its path. The sidecar is main()'s job."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" plus lineterminator="\n" => LF on every platform. Do not "simplify" this.
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "value", "kind"], lineterminator="\n")
        w.writeheader()
        w.writerows(r._asdict() for r in rows)
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compute pooled (single-pass) mode-31 cross-validation metrics."
    )
    ap.add_argument(
        "--oof",
        default=DEFAULT_OOF,
        help="Out-of-fold prediction CSV to pool (default: {}).".format(DEFAULT_OOF),
    )
    ap.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path (optional). No default: {} is git-tracked and certified, "
            "so a bare run prints the table and writes nothing rather than silently "
            "rewriting it.".format(TRACKED_OUTPUT)
        ),
    )
    args = ap.parse_args(argv)

    oof_path = _resolve(args.oof)
    rows = compute(oof_path)
    for r in rows:
        print("{:<26} {:.6f}  ({})".format(r.metric, r.value, r.kind))

    if args.output is None:
        print("\nNo --output given: nothing written.")
        return 0

    out_path = _resolve(args.output)
    write(rows, out_path)

    sys.path.insert(0, str(REPO_ROOT))
    from src.artifact_integrity import sha256_file, write_provenance_sidecar

    sidecar = write_provenance_sidecar(
        out_path,
        script="scripts/compute_pooled_cv_metrics.py",
        extra={"inputs": {_sidecar_key(oof_path): sha256_file(oof_path)}},
    )
    print("\nwrote {}\nwrote {}".format(out_path, sidecar))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
