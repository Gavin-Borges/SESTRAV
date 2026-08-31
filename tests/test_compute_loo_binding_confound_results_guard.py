"""Tier-1 results/ guard tests for scripts/compute_loo_binding_confound.py.

Closes Tier-1 enumeration item #12: this script had no CLI at all, so a bare
invocation always silently rewrote the git-tracked
results/loo_binding_confound_decomposition.csv. --output is now optional
with no default (the "no-default explicit path" pattern, matching
scripts/evaluate_per_virus.py): a bare run prints the table and writes
nothing; passing --output writes there.

These tests monkeypatch the module's PV_SRC/LOO_SRC constants to point at
small synthetic fixtures rather than the real committed results/ CSVs, so
they exercise the write-or-skip guard behavior in isolation from the
(already-published, already-trusted) science those files carry.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts import compute_loo_binding_confound as clbc

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    """Minimal-but-valid PV_SRC/LOO_SRC fixtures covering all 9 CANON viruses."""
    pv_rows = []
    loo_rows = []
    for i, virus in enumerate(clbc.CANON):
        pv_rows.append(
            {
                "virus": virus,
                "auc_roc": 0.7 + i * 0.01,
                "auc_roc_real_neg_only": 0.6 + i * 0.01,
                "n_neg_real": 10,
                "n_neg_decoy": 5,
            }
        )
        loo_rows.append({"test_virus": virus, "auc_roc": 0.5 + i * 0.01})

    pv_path = tmp_path / "per_virus_eval_v5_mode31.csv"
    loo_path = tmp_path / "loo_cross_virus_v5_clean.csv"
    pd.DataFrame(pv_rows).to_csv(pv_path, index=False)
    pd.DataFrame(loo_rows).to_csv(loo_path, index=False)
    return pv_path, loo_path


@pytest.fixture()
def patched_sources(monkeypatch, tmp_path):
    pv_path, loo_path = _write_fixtures(tmp_path)
    monkeypatch.setattr(clbc, "PV_SRC", str(pv_path))
    monkeypatch.setattr(clbc, "LOO_SRC", str(loo_path))


def test_compute_decomposition_has_one_row_per_canon_virus_plus_mean(patched_sources):
    out = clbc.compute_decomposition()
    assert len(out) == len(clbc.CANON) + 1
    assert out.iloc[-1]["virus"] == "Mean"


def test_bare_invocation_writes_nothing(patched_sources, tmp_path, capsys, monkeypatch):
    # The previous form asserted that `tmp_path / "would_not_be_written.csv"` did not
    # exist. That string appears nowhere in scripts/ or src/, so no code path could ever
    # have created it and the assertion held for every possible implementation - including
    # one that wrote the tracked artifact. Only the "Mean" check below ever bit.
    #
    # Two real anchors replace it. compute_loo_binding_confound.py:178-179 prints
    # "wrote <path>" for the CSV and its sidecar, so that token is the script's own report
    # of every write it makes. And the chdir gives a relative write somewhere observable:
    # this module anchors nothing to REPO_ROOT, so a reinstated default would be resolved
    # against the current directory and land in tmp_path.
    # Snapshot rather than assert-empty: patched_sources writes its own input fixtures into
    # tmp_path, so the directory is legitimately non-empty before main() runs. What must
    # not change is its contents.
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    clbc.main([])
    captured = capsys.readouterr()
    assert "Mean" in captured.out  # table is still printed
    assert "wrote " not in captured.out
    assert set(tmp_path.iterdir()) == before


def test_output_flag_writes_the_given_path(patched_sources, tmp_path):
    output_path = tmp_path / "decomposition.csv"
    clbc.main(["--output", str(output_path)])
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert len(df) == len(clbc.CANON) + 1


def test_output_flag_creates_parent_directory_if_missing(patched_sources, tmp_path):
    """--output is now an arbitrary user-supplied path, unlike the original
    hardcoded results/ constant whose parent directory always already
    existed - a fresh, nested destination must not fail."""
    output_path = tmp_path / "new_subdir" / "decomposition.csv"
    assert not output_path.parent.exists()
    clbc.main(["--output", str(output_path)])
    assert output_path.exists()


def test_output_flag_does_not_touch_the_tracked_default_path(
    patched_sources, tmp_path, monkeypatch
):
    """A bare run must never write results/loo_binding_confound_decomposition.csv
    from whatever the current working directory happens to be."""
    monkeypatch.chdir(tmp_path)
    clbc.main([])
    assert not (tmp_path / "results").exists()


def test_cli_help_advertises_no_default_output():
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.compute_loo_binding_confound", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert "No default" in proc.stdout
    assert clbc.TRACKED_OUTPUT in proc.stdout
