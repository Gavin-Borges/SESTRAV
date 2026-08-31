"""results/ guard tests for scripts/compute_pooled_honest_metric.py.

Same guard contract as tests/test_compute_loo_binding_confound_results_guard.py.
`results/pooled_honest_same_pathogen.csv` is git-tracked and its auc_roc cell is bound by
the integrity harness (claim pooled.honest_same_pathogen.auc_roc), so --output has no
default and a bare invocation must print the row and write nothing.

This script was the last writer of a tracked results/ artifact still carrying a default
output path, and it also wrote CRLF: a bare `to_csv()` on Windows emits \r\n, and
check_provenance hashes RAW BYTES, so the recorded digest would not reproduce from a clean
clone. That is the NON-PORTABLE digest FAIL class. Both are covered below.

Fixtures are synthetic; the real committed OOF frame is not read, so these tests neither
depend on nor re-certify the published Def A figures.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts import compute_pooled_honest_metric as cphm

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def synthetic_oof(tmp_path: Path) -> Path:
    """Positives and real-IEDB negatives across two target viruses, plus rows the Def A
    filter must EXCLUDE: an off-panel virus and a non-`iedb_api` negative origin."""
    rows = []
    for i in range(20):
        rows.append(
            {
                "virus": "CMV" if i % 2 else "DENV",
                "label": i % 2,
                "negative_origin": cphm.REAL_NEG_ORIGIN,
                "score": 0.4 + 0.1 * (i % 2) + 0.02 * (i % 7),
            }
        )
    # Must be filtered out: off-panel virus, and a synthetic negative origin.
    rows.append({"virus": "Vaccinia", "label": 0, "negative_origin": cphm.REAL_NEG_ORIGIN, "score": 0.9})
    rows.append({"virus": "CMV", "label": 0, "negative_origin": "allele_matched_nonbinder", "score": 0.9})
    path = tmp_path / "oof.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_compute_applies_the_def_a_filter(synthetic_oof):
    """The two excluded rows must not reach the metric: 10 pos + 10 real negs only."""
    row = cphm.compute(synthetic_oof)
    assert row["n_pos"] == 10
    assert row["n_neg"] == 10
    assert row["metric"] == "honest_same_pathogen"


def test_bare_invocation_writes_nothing(synthetic_oof, capsys):
    assert cphm.main(["--oof", str(synthetic_oof)]) == 0
    captured = capsys.readouterr()
    assert "honest same-pathogen" in captured.out  # row still printed
    assert "nothing written" in captured.out


def test_bare_invocation_does_not_touch_the_tracked_default_path(
    synthetic_oof, tmp_path, monkeypatch
):
    """A bare run must never create results/ from whatever cwd it happens to run in."""
    monkeypatch.chdir(tmp_path)
    cphm.main(["--oof", str(synthetic_oof)])
    assert not (tmp_path / "results").exists()


def test_output_flag_writes_the_given_path(synthetic_oof, tmp_path):
    out = tmp_path / "honest.csv"
    cphm.main(["--oof", str(synthetic_oof), "--output", str(out)])
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) == 1
    assert df.iloc[0]["metric"] == "honest_same_pathogen"


def test_output_flag_creates_parent_directory_if_missing(synthetic_oof, tmp_path):
    out = tmp_path / "new_subdir" / "honest.csv"
    assert not out.parent.exists()
    cphm.main(["--oof", str(synthetic_oof), "--output", str(out)])
    assert out.exists()


def test_csv_is_written_with_lf_endings(synthetic_oof, tmp_path):
    """The regression this file exists for: a bare to_csv() emits CRLF on Windows and the
    harness hashes raw bytes, so the recorded digest would not reproduce elsewhere."""
    out = tmp_path / "honest.csv"
    cphm.main(["--oof", str(synthetic_oof), "--output", str(out)])
    raw = out.read_bytes()
    assert b"\r\n" not in raw
    assert raw.count(b"\n") == 2  # header + one data row


def test_cli_help_advertises_no_default_output():
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.compute_pooled_honest_metric", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert "No default" in proc.stdout
    assert cphm.TRACKED_OUTPUT in proc.stdout
