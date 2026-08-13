#!/usr/bin/env python3
"""Measure what the isotonic calibration layer actually does to mode-31 scores.

Why this exists separately from `scripts/fit_calibrator.py`: that script FITS the
production calibrator and evaluates it **in-sample**, calling `.predict()` on the
very scores it just fitted on. That produces a global ECE of 0.00000, which is an
overfit artifact and not an estimate of deployed behaviour. It also prints its
numbers and persists none of them, so nothing downstream can reconcile against it.

This script answers the assessment question instead, and writes the answer to a
tracked artifact:

  1. It CROSS-FITS the isotonic layer over peptide-grouped folds, so no row is
     calibrated by a model that saw its peptide. Peptide grouping (rather than
     plain row folds) matches the D15 re-baseline discipline: mode-31 features
     are a pure function of the peptide string, so two rows sharing a peptide are
     feature-identical and an ungrouped split leaks.
  2. It reports the pooled figure, the nine-target-virus figure AND the
     out-of-panel figure separately, because the pooled number is an artifact of
     CANCELLATION and is unusable on its own. The pool mixes two populations that
     are miscalibrated in opposite directions - in every score bin the target
     viruses are under-confident and the out-of-panel rows over-confident - so
     the errors offset and the pooled ECE lands BELOW both components rather
     than between them. Emitting the off_panel scope is what makes that
     checkable: quoting pooled ECE without it invites the reading that the pool
     is simply easier to calibrate, which its own ece_cal refutes.
     Caveat on the off_panel row: at a positive rate near 0.0008 the ECE
     degenerates towards the mean predicted score and carries little bin
     resolution. It is a diagnostic for the pooled figure, not a result.

It writes NO model artifact. `scripts/fit_calibrator.py` remains the only writer
of `models/isotonic_calibrator.joblib`.

The ECE and Brier definitions are IMPORTED from `scripts/fit_calibrator.py`
rather than restated here. A second copy of a metric definition is a second
source of truth, which is exactly the divergence D20 was.

Reproduce:  python scripts/assess_calibration.py --allow-overwrite
Output:     results/calibration_assessment_v5_mode31.csv (+ .provenance.json)

The --allow-overwrite flag is required for a plain re-run because --out defaults
to the committed artifact, and the overwrite guard aborts before doing any work
rather than silently replacing a file the manuscript cites. Point --out at a
fresh path instead when comparing against the committed copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fit_calibrator import (  # noqa: E402
    TARGET_VIRUSES,
    brier_score,
    expected_calibration_error,
)
from src.artifact_guard import guard_planned_paths, planned_paths_under  # noqa: E402

RANDOM_STATE = 42
N_SPLITS = 5

DEFAULT_OOF_PATH = PROJECT_ROOT / "models" / "v5" / "rf_oof_predictions_mode31.csv"
DEFAULT_OUT_PATH = PROJECT_ROOT / "results" / "calibration_assessment_v5_mode31.csv"


def _sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as fb:
        for chunk in iter(lambda: fb.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _load_rf_oof(oof_path: Path) -> pd.DataFrame:
    """Load OOF predictions, keeping only RandomForest rows with valid scores.

    Mirrors `scripts.fit_calibrator._load_rf_oof` so the assessed population is
    the same one the production calibrator is fitted on.
    """
    df = pd.read_csv(oof_path)
    df = df[df["method"] == "RandomForest"].copy()
    df = df.dropna(subset=["score", "label"])
    df["score"] = df["score"].astype(np.float64).clip(0.0, 1.0)
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def cross_fitted_calibration(df: pd.DataFrame) -> np.ndarray:
    """Calibrated score for every row, from a fold that never saw its peptide.

    Returns an array aligned to df's row order. Uses StratifiedGroupKFold so each
    fold keeps the label balance while the peptide group constraint holds - the
    same splitter arrangement `scripts/audit_cv_leakage.py` uses for its
    peptide-grouped arm.
    """
    scores = df["score"].to_numpy(dtype=np.float64)
    labels = df["label"].to_numpy(dtype=np.float64)
    groups = df["peptide"].to_numpy()

    calibrated = np.full(len(df), np.nan)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    for train_idx, test_idx in splitter.split(
        scores.reshape(-1, 1), labels, groups=groups
    ):
        fold = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        fold.fit(scores[train_idx], labels[train_idx])
        calibrated[test_idx] = fold.predict(scores[test_idx])
    if np.isnan(calibrated).any():
        raise RuntimeError(
            f"{int(np.isnan(calibrated).sum())} row(s) never appeared in a test "
            "fold; the cross-fit is incomplete and its ECE would be meaningless"
        )
    return calibrated


def _scope_row(scope: str, sub: pd.DataFrame) -> dict:
    labels = sub["label"].to_numpy(dtype=np.float64)
    raw = sub["score"].to_numpy(dtype=np.float64)
    cal = sub["calibrated"].to_numpy(dtype=np.float64)
    ece_raw = expected_calibration_error(labels, raw)
    ece_cal = expected_calibration_error(labels, cal)
    return {
        "scope": scope,
        "n": len(sub),
        "n_positive": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "ece_raw": ece_raw,
        "ece_cal": ece_cal,
        "delta": ece_cal - ece_raw,
        "brier_raw": brier_score(labels, raw),
        "brier_cal": brier_score(labels, cal),
    }


def run(oof_path: Path) -> pd.DataFrame:
    df = _load_rf_oof(oof_path)
    df["calibrated"] = cross_fitted_calibration(df)

    is_target = df["virus"].isin(TARGET_VIRUSES)
    rows = [
        # The pooled figure. Majority out-of-panel negatives - see module docstring.
        _scope_row("pooled_all", df),
        # The population the manuscript's per-virus results are actually about.
        _scope_row("target_viruses", df[is_target]),
        # The complement, reported so the pooled/target gap is attributable rather
        # than merely asserted.
        _scope_row("off_panel", df[~is_target]),
    ]
    for virus in TARGET_VIRUSES:
        sub = df[df["virus"] == virus]
        if sub.empty:
            continue
        rows.append(_scope_row(virus, sub))
    return pd.DataFrame(rows)


def _provenance_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".provenance.json")


def _planned_paths(output_path: Path) -> list[str]:
    return planned_paths_under(
        str(output_path.parent), [output_path.name, _provenance_path(output_path).name]
    )


def _guard_output_path(output_path: Path, allow_overwrite: bool) -> None:
    """Abort before doing any work if this run would replace an existing artifact.

    Extracted from `main` so the generic contract test in
    `tests/test_artifact_guard_contract.py` can exercise the guard this module
    actually uses, rather than restating its keyword arguments in the test and
    then slowly drifting from them. Every sibling guarded module exposes its
    guard the same way.
    """
    guard_planned_paths(
        str(output_path.parent),
        _planned_paths(output_path),
        allow_overwrite,
        flag="--out",
        api_hint="main(['--allow-overwrite'])",
        scope="among this run's planned artifacts",
        remedy="Point --out at a fresh path, ",
        detail=": these numbers are cited by docs/paper.md Section 3.2",
    )


def _write_provenance(output_path: Path, oof_path: Path, result: pd.DataFrame) -> None:
    try:
        oof_rel = oof_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        oof_rel = oof_path.name
    pooled = result[result["scope"] == "pooled_all"].iloc[0]
    try:
        out_rel = output_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        out_rel = output_path.name
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/assess_calibration.py",
        # The artifact's OWN hash, and an explicit pointer to it. Without these
        # the integrity harness's provenance check cannot verify this sidecar at
        # all: it looks for `sha256`/`checksum`, and it resolves the artifact
        # from `file`/`artifact` before falling back to mangling the sidecar
        # filename. Recording only the INPUT hash (oof_sha256, below) left the
        # check with nothing to check, which is how it came to report zero PASS
        # results across all 40 sidecars in the repo.
        "artifact": out_rel,
        "sha256": _sha256_file(output_path),
        "oof_path": oof_rel,
        "oof_sha256": _sha256_file(oof_path),
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "splitter": "StratifiedGroupKFold(groups=peptide)",
        "calibrator": "IsotonicRegression(out_of_bounds=clip, y_min=0.0, y_max=1.0)",
        "in_sample": False,
        "n_rows_assessed": int(pooled["n"]),
        "n_rows_emitted": len(result),
    }
    provenance_path = _provenance_path(output_path)
    # newline="" keeps the LF that json.dumps produces, instead of letting the
    # platform rewrite it to CRLF. Matches the eol=lf pin in .gitattributes, so
    # the file git stores and the file on disk are the same bytes.
    with provenance_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {provenance_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", default=str(DEFAULT_OOF_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace the assessment CSV and its provenance sidecar at --out if "
        "they already exist. Without this flag the run aborts before any work.",
    )
    args = parser.parse_args(argv)

    oof_path = Path(args.oof)
    if not oof_path.is_absolute():
        oof_path = PROJECT_ROOT / args.oof
    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / args.out

    _guard_output_path(output_path, args.allow_overwrite)

    result = run(oof_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # lineterminator is pinned to LF rather than left to the platform. pandas
    # writes CRLF on Windows, git stores LF, and the sidecar records a sha256 of
    # whichever one it happened to see - so an unpinned terminator makes the
    # recorded hash platform-dependent and the provenance check unverifiable
    # anywhere but the machine that wrote it. .gitattributes pins the same file
    # from git's side; this pins it from the writer's side, so the two agree
    # even before the file is ever committed.
    result.to_csv(output_path, index=False, lineterminator="\n")
    _write_provenance(output_path, oof_path, result)
    print(f"wrote {output_path} ({len(result)} rows)")

    pooled = result[result["scope"] == "pooled_all"].iloc[0]
    target = result[result["scope"] == "target_viruses"].iloc[0]
    per_virus = result[result["scope"].isin(TARGET_VIRUSES)]
    improved = int((per_virus["delta"] < 0).sum())
    print(
        f"pooled ECE {pooled['ece_raw']:.4f} -> {pooled['ece_cal']:.4f} "
        f"({pooled['delta']:+.4f}) over n={int(pooled['n'])}"
    )
    print(
        f"target-virus ECE {target['ece_raw']:.4f} -> {target['ece_cal']:.4f} "
        f"({target['delta']:+.4f}) over n={int(target['n'])}"
    )
    print(f"per-virus ECE improves for {improved} of {len(per_virus)} target viruses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
