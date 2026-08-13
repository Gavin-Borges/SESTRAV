"""Fit and save FINAL, deployable per-virus isotonic calibrators (A1-B).

WHAT THIS IS, AND WHAT IT IS NOT. scripts/fit_per_virus_calibrator.py is a
three-arm research COMPARISON: cross-fitted (peptide-grouped, held-out) so its
numbers are honest generalization estimates, scratch-only by hard-coded
guard, and its own docstring states plainly that it "fits no artifact for
deployment". This script is the deployment-fitting counterpart - the thing
that comparison was measuring the merit OF. It fits one
sklearn.isotonic.IsotonicRegression PER TARGET VIRUS on ALL of that virus's
available OOF rows (no held-out split, matching how the existing global
models/isotonic_calibrator.joblib was fit by scripts/fit_calibrator.py - a
final calibrator is refit on all available data once cross-validation has
already answered the "does this generalize" question).

WHY THE DEFAULT OUTPUT IS SCRATCH, UNLIKE fit_calibrator.py. Writing a real
per-virus calibrator to models/calibration/per_virus/ is the actual PROMOTION
act: functions/stage4_immunogenicity_scoring.py's _apply_calibration falls
back to the global calibrator purely from file ABSENCE, so populating that
directory for real changes deployed scoring behaviour for any caller that
passes --virus. That is A1-promote in the open-item register, a separate,
still-open decision from A1-B (Stage 4 integration - the code path this
script exists to exercise and test). Promote for real only with an explicit
override once that decision is made:

    python scripts/fit_final_per_virus_calibrators.py \
        --out-dir models/calibration/per_virus

Reproduce (scratch, safe by default):
    python scripts/fit_final_per_virus_calibrators.py
Output:
    models/scratch/per_virus_calibrators_final/<VIRUS>.joblib  (one per virus)
    models/scratch/per_virus_calibrators_final/model_artifact_checksums.json
    models/scratch/per_virus_calibrators_final/summary.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fit_calibrator import (  # noqa: E402
    TARGET_VIRUSES,
    brier_score,
    expected_calibration_error,
)
from src.artifact_guard import guard_planned_paths, planned_paths_under  # noqa: E402
from src.artifact_integrity import (  # noqa: E402
    default_manifest_path_for,
    update_checksum_manifest,
    write_provenance_sidecar,
)

DEFAULT_OOF_PATH = PROJECT_ROOT / "models" / "v5" / "rf_oof_predictions_mode31.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "models" / "scratch" / "per_virus_calibrators_final"
SUMMARY_NAME = "summary.md"

# Any real deployable calibrator lives under models/ (this script's whole
# purpose), so unlike fit_per_virus_calibrator.py's hard scratch-only guard,
# --out-dir here is intentionally NOT restricted to models/scratch/ - the
# guard below (_guard_output_dir) is the only thing standing between a normal
# run and overwriting a promoted set of calibrators, exactly as it does for
# every other model-writing script in this repo.

MIN_ROWS_TO_FIT = 20  # below this, isotonic regression is not meaningfully fit


def _load_rf_oof(oof_path: Path) -> pd.DataFrame:
    """Mirrors scripts.fit_calibrator._load_rf_oof so both read one schema."""
    df = pd.read_csv(oof_path)
    df = df[df["method"] == "RandomForest"].copy()
    df = df.dropna(subset=["score", "label"])
    df["score"] = df["score"].astype(np.float64).clip(0.0, 1.0)
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def _sanitize_name(name: str) -> str:
    """Mirrors functions.stage4_immunogenicity_scoring._sanitize_name exactly -
    the calibrator filename this script writes must match the filename
    _resolve_calibrator_path looks up, or a promoted calibrator would silently
    never be found."""
    import re

    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


def fit_final_calibrators(df: pd.DataFrame) -> dict[str, dict]:
    """Fit one final IsotonicRegression per target virus with enough rows.

    Returns {virus: {"calibrator": IsotonicRegression, "n": int, "ece_raw": float,
    "ece_cal": float, "brier_raw": float, "brier_cal": float}}, skipping any
    virus with fewer than MIN_ROWS_TO_FIT rows (reported, not silently dropped).
    """
    results: dict[str, dict] = {}
    for virus in TARGET_VIRUSES:
        sub = df[df["virus"] == virus]
        if len(sub) < MIN_ROWS_TO_FIT:
            print(f"[skip] {virus}: only {len(sub)} row(s), below MIN_ROWS_TO_FIT={MIN_ROWS_TO_FIT}")
            continue
        scores = sub["score"].to_numpy(dtype=np.float64)
        labels = sub["label"].to_numpy(dtype=np.float64)
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(scores, labels)
        calibrated = np.asarray(calibrator.predict(scores), dtype=np.float64)
        results[virus] = {
            "calibrator": calibrator,
            "n": len(sub),
            "ece_raw": expected_calibration_error(labels, scores),
            "ece_cal": expected_calibration_error(labels, calibrated),
            "brier_raw": brier_score(labels, scores),
            "brier_cal": brier_score(labels, calibrated),
        }
    return results


def _planned_paths(out_dir: Path, viruses: list[str]) -> list[str]:
    names = [f"{_sanitize_name(v)}.joblib" for v in viruses] + [SUMMARY_NAME]
    return planned_paths_under(str(out_dir), names)


def _guard_output_dir(out_dir: Path, viruses: list[str], allow_overwrite: bool) -> None:
    """Abort before doing any work if this run would replace an existing artifact."""
    guard_planned_paths(
        str(out_dir),
        _planned_paths(out_dir, viruses),
        allow_overwrite,
        flag="--out-dir",
        api_hint="main(['--allow-overwrite'])",
        scope="among this run's planned artifacts",
        remedy="Point --out-dir at a fresh path, ",
        detail=": a prior fit may still be in use",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", default=str(DEFAULT_OOF_PATH))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace existing calibrator artifacts in --out-dir instead of aborting "
        "before any work.",
    )
    args = parser.parse_args(argv)

    oof_path = Path(args.oof)
    out_dir = Path(args.out_dir)

    df = _load_rf_oof(oof_path)
    print(f"[fit] Loaded {len(df)} RandomForest OOF rows from {oof_path}")

    fitted = fit_final_calibrators(df)
    if not fitted:
        print("[fit] No virus had enough rows to fit a calibrator - nothing to write.")
        return 0

    _guard_output_dir(out_dir, sorted(fitted), args.allow_overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)

    from joblib import dump as joblib_dump

    written: list[Path] = []
    summary_lines = [
        "# Final per-virus isotonic calibrators",
        "",
        f"Fit from `{oof_path.name}`, one IsotonicRegression per virus on ALL available "
        "OOF rows (no held-out split - this is the deployment fit, not an evaluation).",
        "",
        "| Virus | n | ECE raw | ECE cal | Brier raw | Brier cal |",
        "|---|---|---|---|---|---|",
    ]
    for virus in sorted(fitted):
        info = fitted[virus]
        joblib_path = out_dir / f"{_sanitize_name(virus)}.joblib"
        joblib_dump(info["calibrator"], joblib_path)
        written.append(joblib_path)
        summary_lines.append(
            f"| {virus} | {info['n']} | {info['ece_raw']:.4f} | {info['ece_cal']:.4f} "
            f"| {info['brier_raw']:.4f} | {info['brier_cal']:.4f} |"
        )
        print(f"[fit] {virus}: n={info['n']}, wrote {joblib_path}")

    manifest_path = update_checksum_manifest(default_manifest_path_for(written[0]), written)
    print(f"[save] Checksum manifest updated at {manifest_path}")

    summary_path = out_dir / SUMMARY_NAME
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(summary_lines) + "\n")
    print(f"[save] Summary written to {summary_path}")

    for artifact in (*written, summary_path):
        write_provenance_sidecar(
            artifact,
            script="scripts/fit_final_per_virus_calibrators.py",
            extra={"oof_path": oof_path.name, "viruses_fit": sorted(fitted)},
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
