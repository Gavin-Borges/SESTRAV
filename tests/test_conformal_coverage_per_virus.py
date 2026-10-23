"""models/v5/conformal_coverage_mode31.json: its per_virus block is bound to its source.

Why this test exists
--------------------
The coverage artifact carries a nine-virus `per_virus` block that no test read
(to-do S37i: `git grep per_virus` over the conformal tests returned 0). The
artifact also names its own source, `source_oof`, with a sha256. So the block
can be checked against data rather than against a snapshot of itself:

- the recorded source path is tracked, and its recorded sha256 identifies that
  content (in either line-ending rendering; see the test's own docstring);
- the nine viruses are exactly the fitting script's TARGET_VIRUSES that occur
  in that source;
- every field the source alone determines - n_samples, mean_raw_score,
  mean_label and brier_raw per virus, and the corresponding overall metrics -
  recomputes from the source with the fitting script's own row filter
  (`_load_rf_oof`: RandomForest rows, non-null score, label and peptide, score
  clipped to [0, 1]); measured 2026-10-15, every difference is exactly 0.0;
- the fields that need the gitignored calibrator (calibrated midpoints,
  interval widths, brier_mid) are only range-checked here, because CI cannot
  rebuild them.

A regenerated artifact passes as long as it still describes its source; a
hand edit, a changed filter, or a stale source does not.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "models" / "v5" / "conformal_coverage_mode31.json"
FIT_SCRIPT = REPO_ROOT / "scripts" / "fit_conformal_calibrator.py"
EXPECTED_VIRUSES = {"CMV", "DENV", "EBV", "HBV", "HCV", "HIV-1", "HPV", "IAV", "SARS-CoV-2"}
SOURCE_FIELDS = ("n_samples", "mean_raw_score", "mean_label", "brier_raw")
TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source(report: dict) -> pd.DataFrame:
    """The source OOF rows, filtered exactly as `_load_rf_oof` filters them."""
    df = pd.read_csv(REPO_ROOT / report["source_oof"]["path"])
    df = df[df["method"] == "RandomForest"]
    df = df.dropna(subset=["score", "label", "peptide"]).copy()
    df["score"] = df["score"].astype(np.float64).clip(0.0, 1.0)
    df["label"] = df["label"].astype(int)
    return df


def _target_viruses() -> tuple[str, ...]:
    spec = importlib.util.spec_from_file_location("fit_conformal_calibrator_under_test", FIT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.TARGET_VIRUSES)


def source_fields(rows: pd.DataFrame) -> dict[str, float]:
    scores = rows["score"].to_numpy(dtype=np.float64)
    labels = rows["label"].to_numpy(dtype=np.float64)
    return {
        "n_samples": int(len(rows)),
        "mean_raw_score": float(np.mean(scores)),
        "mean_label": float(np.mean(labels)),
        "brier_raw": float(np.mean((scores - labels) ** 2)),
    }


def mismatches(recorded: dict, recomputed: dict) -> list[str]:
    """Each source-determined field whose recorded value differs from the source's."""
    found = []
    for field in SOURCE_FIELDS:
        if field == "n_samples":
            if recorded[field] != recomputed[field]:
                found.append(f"{field}: recorded {recorded[field]}, source {recomputed[field]}")
        elif not math.isclose(recorded[field], recomputed[field], rel_tol=0.0, abs_tol=TOLERANCE):
            found.append(f"{field}: recorded {recorded[field]!r}, source {recomputed[field]!r}")
    return found


def test_the_recorded_source_is_tracked_and_its_content_unchanged(report: dict) -> None:
    """The recorded sha256 must identify the tracked content, whatever its line endings.

    Measured 2026-10-15: the recorded value is the sha256 of the file's CRLF
    rendering, while .gitattributes pins the file to LF, so the committed blob
    hashes differently and a byte comparison would pass or fail with the
    checkout. Both renderings are therefore derived from the canonical LF bytes;
    a different source fails either way.
    """
    rel = report["source_oof"]["path"]
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", rel], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=False)
    assert tracked.returncode == 0, f"{rel} is not tracked"
    canonical = (REPO_ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
    renderings = {
        hashlib.sha256(canonical).hexdigest(),
        hashlib.sha256(canonical.replace(b"\n", b"\r\n")).hexdigest(),
    }
    assert report["source_oof"]["sha256"] in renderings


def test_per_virus_covers_exactly_the_target_viruses_present(report: dict, source: pd.DataFrame) -> None:
    targets = _target_viruses()
    present = {v for v in targets if (source["virus"] == v).any()}
    assert set(targets) == EXPECTED_VIRUSES
    assert set(report["per_virus"]) == present == EXPECTED_VIRUSES


@pytest.mark.parametrize("virus", sorted(EXPECTED_VIRUSES))
def test_per_virus_source_fields_recompute(report: dict, source: pd.DataFrame, virus: str) -> None:
    recomputed = source_fields(source[source["virus"] == virus])
    assert mismatches(report["per_virus"][virus], recomputed) == []


def test_overall_source_fields_recompute(report: dict, source: pd.DataFrame) -> None:
    metrics = report["metrics"]
    recorded = {
        "n_samples": report["n_samples"],
        "mean_raw_score": metrics["mean_raw_score"],
        "mean_label": metrics["mean_label"],
        "brier_raw": metrics["brier_score_raw"],
    }
    assert mismatches(recorded, source_fields(source)) == []
    assert report["n_groups"] == source["peptide"].nunique()


@pytest.mark.parametrize("virus", sorted(EXPECTED_VIRUSES))
def test_calibrator_fields_are_in_range(report: dict, virus: str) -> None:
    rec = report["per_virus"][virus]
    assert 0.0 <= rec["mean_calibrated_mid"] <= 1.0
    assert 0.0 <= rec["median_interval_width"] and 0.0 <= rec["mean_interval_width"] < 1.0
    assert 0.0 <= rec["brier_mid"] <= 1.0


def test_a_tampered_value_is_caught(report: dict, source: pd.DataFrame) -> None:
    """Anti-vacuity: the comparison fails on a 1e-9 edit to a mean and on a count edit."""
    rec = dict(report["per_virus"]["CMV"])
    recomputed = source_fields(source[source["virus"] == "CMV"])
    assert mismatches(rec, recomputed) == []
    rec["mean_label"] += 1e-9
    rec["n_samples"] += 1
    assert len(mismatches(rec, recomputed)) == 2
