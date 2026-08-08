"""Tier-1 results/ guard tests for scripts/run_tier_a_benchmarks.py.

Closes Tier-1 enumeration item #15. This script writes TWO independent
git-tracked artifacts - data/tier_a_external_benchmarks.csv (per-peptide
external-tool scores) and results/table3_tier_a_metrics.csv (the certified
Tier-A headline table, source of the published 0.828 AUC-PR figure) - and
previously had no CLI for either (--smoke was the only flag). Both
--scores-output and --metrics-output are now optional with no default (the
"no-default explicit path" pattern, matching scripts/evaluate_per_virus.py):
a bare run performs the benchmark and prints results without writing either
file; passing either flag writes that one.

The write-or-skip decision for both artifacts is shared via the standalone
maybe_write_csv() function specifically so it is testable without running
DeepImmuno/BigMHC/MixMHCpred (external tools requiring separate conda envs,
a BigMHC venv, and a Perl installation - none available in a standard test
environment) - these tests exercise only that function and the CLI parsing,
not the benchmark itself.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_tier_a_benchmarks as rtab

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "peptide": ["AAA", "BBB", "CCC"],
            "label": [1, 0, 1],
            "deepimmuno_score": [0.1, 0.2, 0.3],
            "extra_col_not_selected": [9, 9, 9],
        }
    )


def test_maybe_write_csv_does_nothing_when_path_is_none(tmp_path, sample_df):
    rtab.maybe_write_csv(sample_df, None, ["peptide", "label"])
    assert list(tmp_path.iterdir()) == []


def test_maybe_write_csv_writes_only_the_given_columns(tmp_path, sample_df):
    output_path = tmp_path / "out.csv"
    rtab.maybe_write_csv(sample_df, str(output_path), ["peptide", "label", "deepimmuno_score"])
    written = pd.read_csv(output_path)
    assert list(written.columns) == ["peptide", "label", "deepimmuno_score"]
    assert len(written) == 3


def test_maybe_write_csv_creates_parent_directory_if_missing(tmp_path, sample_df):
    output_path = tmp_path / "new_subdir" / "out.csv"
    assert not output_path.parent.exists()
    rtab.maybe_write_csv(sample_df, str(output_path), ["peptide", "label"])
    assert output_path.exists()


def test_parse_args_both_outputs_default_to_none():
    args = rtab.parse_args([])
    assert args.scores_output is None
    assert args.metrics_output is None
    assert args.smoke is False


def test_parse_args_accepts_both_outputs_independently():
    args = rtab.parse_args(
        ["--scores-output", "custom/scores.csv", "--metrics-output", "custom/metrics.csv"]
    )
    assert args.scores_output == "custom/scores.csv"
    assert args.metrics_output == "custom/metrics.csv"


def test_cli_help_advertises_no_default_for_both_outputs():
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.run_tier_a_benchmarks", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert proc.stdout.count("No default") == 2
    assert rtab.TRACKED_SCORES_OUTPUT in proc.stdout
    assert rtab.TRACKED_METRICS_OUTPUT in proc.stdout
