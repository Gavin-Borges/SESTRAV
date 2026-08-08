"""Tier-1 results/ guard tests for scripts/compute_tier_a_paired_bootstrap.py.

Closes Tier-1 enumeration item #13: this script had no CLI at all, so a bare
invocation always silently rewrote the git-tracked
results/tier_a_paired_bootstrap.csv. --output is now optional with no
default (the "no-default explicit path" pattern, matching
scripts/evaluate_per_virus.py): a bare run prints the table and writes
nothing; passing --output writes there.

compute_bootstrap_table() reads its two inputs via inline relative-path
literals (results/external_validation_merged_scores.csv and
data/tier_a_external_benchmarks.csv), matching the script's pre-existing
design - these tests use monkeypatch.chdir() to a tmp directory holding
small synthetic fixtures at those same relative paths, rather than
refactoring the read paths into module constants (out of scope for a
guard-only change) or depending on the real committed results/ data.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import compute_tier_a_paired_bootstrap as ctpb

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_fixtures(tmp_path: Path, n: int = 40) -> None:
    rng = np.random.default_rng(0)
    peptides = [f"PEP{i:03d}" for i in range(n)]
    (tmp_path / "results").mkdir()
    (tmp_path / "data").mkdir()
    pd.DataFrame(
        {
            "peptide": peptides,
            "label": rng.integers(0, 2, size=n),
            "rf_oof_score": rng.random(n),
            "binding_max": rng.random(n),
        }
    ).to_csv(tmp_path / "results" / "external_validation_merged_scores.csv", index=False)
    pd.DataFrame(
        {"peptide": peptides, "bigmhc_score": rng.random(n)}
    ).to_csv(tmp_path / "data" / "tier_a_external_benchmarks.csv", index=False)


@pytest.fixture()
def fixture_cwd(monkeypatch, tmp_path):
    _write_fixtures(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ctpb, "B", 25)  # keep the bootstrap fast in tests
    return tmp_path


def test_compute_bootstrap_table_has_both_comparisons(fixture_cwd):
    out = ctpb.compute_bootstrap_table()
    assert set(out["comparison"]) == {"rf_vs_bigmhc", "rf_vs_binding_only"}


def test_bare_invocation_writes_nothing(fixture_cwd, capsys):
    ctpb.main([])
    assert not (fixture_cwd / "results" / "tier_a_paired_bootstrap.csv").exists()
    captured = capsys.readouterr()
    assert "rf_vs_bigmhc" in captured.out


def test_output_flag_writes_the_given_path(fixture_cwd):
    output_path = fixture_cwd / "out" / "bootstrap.csv"
    ctpb.main(["--output", str(output_path)])
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert len(df) == 2


def test_cli_help_advertises_no_default_output():
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.compute_tier_a_paired_bootstrap", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert "No default" in proc.stdout
    assert ctpb.TRACKED_OUTPUT in proc.stdout
