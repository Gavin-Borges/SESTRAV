"""Overwrite-guard tests for src/bias_skew_finalization.py's own --results-dir.

This is the last disclosed instance of the results/ silent-overwrite defect
class. run_bias_skew_finalization() had its own --results-dir CLI flag,
separate from the already-guarded run_final_validation() call it makes (see
test_final_validation_results_guard.py), that still defaulted to "results"
with no guard. It writes or delegates writes for 8 files directly into
results_dir: immunogenicity_provenance.csv (refresh_dataset), the pair
data_bias_audit_summary.csv / data_bias_audit.md plus the derived
data_bias_audit_summary_virus_label_counts.csv (all three from
write_audit_reports), the pair gold_standard_sensitivity.csv /
gold_standard_sensitivity.md plus the derived
gold_standard_sensitivity_deltas.csv (all three from
run_gold_standard_sensitivity), and release_readiness_summary.md, written
directly at the end of run_bias_skew_finalization itself.

results_dir is now required with no default at both the CLI and Python-API
layers - it moved ahead of the already-defaulted data_csv parameter in the
signature, since Python does not allow a plain no-default parameter after a
defaulted one. Every existing caller already passed it by keyword, so nothing
breaks. planned_bias_skew_paths() and _guard_results_dir() are the same
required-flag + FileExistsError + --allow-overwrite pattern used throughout
this defect-class line (see test_final_validation_results_guard.py and
test_gnn_model_dir_guard.py for the two closest precedents).

No write-before-guard collision exists here (unlike the baseline_comparison.csv
case fixed earlier in this line): the new guard is the literal first statement
of run_bias_skew_finalization, so nothing in the function writes any of these
8 files before the guard checks for them, and none of the 8 filenames overlap
with run_final_validation's own, separately-guarded 10-file list.

All heavy dependencies (dataset refresh, RF/ANN training, H2 evaluation via
run_final_validation, gold-standard sensitivity) are faked in the wiring test
below: this file is about the call graph and file-write surface, not about
running the real, slow, data-dependent pipeline.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.bias_skew_finalization import (
    _guard_results_dir as guard_bias_skew,
    planned_bias_skew_paths,
    run_bias_skew_finalization,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# planned path set
# ---------------------------------------------------------------------------


def test_bias_skew_plans_all_eight_files():
    names = {p.name for p in map(Path, planned_bias_skew_paths("results"))}
    assert names == {
        "immunogenicity_provenance.csv",
        "data_bias_audit_summary.csv",
        "data_bias_audit.md",
        "data_bias_audit_summary_virus_label_counts.csv",
        "gold_standard_sensitivity.csv",
        "gold_standard_sensitivity.md",
        "gold_standard_sensitivity_deltas.csv",
        "release_readiness_summary.md",
    }


def test_bias_skew_plans_paths_inside_the_given_results_dir(tmp_path):
    paths = planned_bias_skew_paths(str(tmp_path))
    for p in paths:
        assert Path(p).parent == tmp_path


# ---------------------------------------------------------------------------
# guard behavior
# ---------------------------------------------------------------------------


def test_bias_skew_guard_passes_on_an_empty_directory(tmp_path):
    guard_bias_skew(str(tmp_path), allow_overwrite=False)


def test_bias_skew_guard_refuses_to_clobber_the_provenance_table(tmp_path):
    """refresh_dataset's output - the provenance/audit trail underpinning the
    dataset's own honesty claims, per the note that flagged this instance."""
    sentinel = tmp_path / "immunogenicity_provenance.csv"
    sentinel.write_text("source_file,n_records\nfoo.csv,10\n")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        guard_bias_skew(str(tmp_path), allow_overwrite=False)

    assert "10" in sentinel.read_text()


def test_bias_skew_guard_refuses_to_clobber_a_write_audit_reports_file(tmp_path):
    sentinel = tmp_path / "data_bias_audit.md"
    sentinel.write_text("# published audit")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        guard_bias_skew(str(tmp_path), allow_overwrite=False)

    assert sentinel.read_text() == "# published audit"


def test_bias_skew_guard_catches_the_derived_virus_label_counts_file(tmp_path):
    """The derived filename that was missed on the first enumeration pass."""
    sentinel = tmp_path / "data_bias_audit_summary_virus_label_counts.csv"
    sentinel.write_text("virus,label,n\nEBV,1,5\n")

    with pytest.raises(FileExistsError) as excinfo:
        guard_bias_skew(str(tmp_path), allow_overwrite=False)

    assert "data_bias_audit_summary_virus_label_counts.csv" in str(excinfo.value)


def test_bias_skew_guard_refuses_to_clobber_a_gold_standard_sensitivity_file(tmp_path):
    sentinel = tmp_path / "gold_standard_sensitivity.csv"
    sentinel.write_text("virus,set_name,recovery_top25\nEBV,current_standard,0.8\n")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        guard_bias_skew(str(tmp_path), allow_overwrite=False)

    assert "0.8" in sentinel.read_text()


def test_bias_skew_guard_catches_the_derived_sensitivity_deltas_file(tmp_path):
    sentinel = tmp_path / "gold_standard_sensitivity_deltas.csv"
    sentinel.write_text("virus,set_name,delta_recovery_top25\nEBV,high_confidence_only,-0.01\n")

    with pytest.raises(FileExistsError) as excinfo:
        guard_bias_skew(str(tmp_path), allow_overwrite=False)

    assert "gold_standard_sensitivity_deltas.csv" in str(excinfo.value)


def test_bias_skew_guard_refuses_to_clobber_the_release_readiness_summary(tmp_path):
    sentinel = tmp_path / "release_readiness_summary.md"
    sentinel.write_text("# Release ready: True")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        guard_bias_skew(str(tmp_path), allow_overwrite=False)

    assert sentinel.read_text() == "# Release ready: True"


def test_bias_skew_guard_names_the_blocking_file_and_the_escape_hatch(tmp_path):
    (tmp_path / "release_readiness_summary.md").write_text("published")

    with pytest.raises(FileExistsError) as excinfo:
        guard_bias_skew(str(tmp_path), allow_overwrite=False)

    message = str(excinfo.value)
    assert "release_readiness_summary.md" in message
    assert "--allow-overwrite" in message


def test_bias_skew_allow_overwrite_disarms_the_guard(tmp_path):
    (tmp_path / "release_readiness_summary.md").write_text("published")
    guard_bias_skew(str(tmp_path), allow_overwrite=True)


# ---------------------------------------------------------------------------
# the guard is actually wired into run_bias_skew_finalization()
# ---------------------------------------------------------------------------
#
# The guard is the first statement in run_bias_skew_finalization, before even
# os.makedirs(model_dir) runs, so a blocked run costs nothing. All heavy
# dependencies are monkeypatched out so this exercises only the call graph
# and file-write surface, not the real pipeline.


def _monkeypatch_heavy_dependencies(monkeypatch):
    import src.bias_skew_finalization as bsf

    monkeypatch.setattr(bsf, "_collect_raw_records", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(bsf, "refresh_dataset", lambda *a, **k: None)
    monkeypatch.setattr(
        bsf,
        "write_audit_reports",
        lambda *a, **k: (pd.DataFrame(), {"n_total": 0, "raw_n_records": 0}),
    )
    monkeypatch.setattr(bsf, "train_models", lambda *a, **k: None)
    monkeypatch.setattr(bsf, "train_ann", lambda *a, **k: None)
    monkeypatch.setattr(bsf, "run_final_validation", lambda **k: ("gs.csv", "baseline.csv", "f.md"))
    monkeypatch.setattr(bsf, "run_gold_standard_sensitivity", lambda *a, **k: None)
    return bsf


def test_guard_is_wired_into_run_bias_skew_finalization(tmp_path, monkeypatch):
    bsf = _monkeypatch_heavy_dependencies(monkeypatch)

    model_dir = tmp_path / "models"
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    sentinel = results_dir / "release_readiness_summary.md"
    sentinel.write_text("published")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        bsf.run_bias_skew_finalization(
            source_data_dir=str(tmp_path / "raw"),
            model_dir=str(model_dir),
            results_dir=str(results_dir),
            allow_overwrite=False,
        )

    assert sentinel.read_text() == "published"
    # The guard fired before any delegate ran.
    assert not model_dir.exists()


def test_bias_skew_finalization_allow_overwrite_lets_a_blocked_run_proceed(tmp_path, monkeypatch):
    bsf = _monkeypatch_heavy_dependencies(monkeypatch)

    model_dir = tmp_path / "models"
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "release_readiness_summary.md").write_text("stale")

    out = bsf.run_bias_skew_finalization(
        source_data_dir=str(tmp_path / "raw"),
        model_dir=str(model_dir),
        results_dir=str(results_dir),
        allow_overwrite=True,
    )

    assert out == str(results_dir / "release_readiness_summary.md")
    assert (results_dir / "release_readiness_summary.md").read_text() != "stale"


def test_run_bias_skew_finalization_requires_an_explicit_results_dir():
    """No default destination: omitting results_dir is a TypeError."""
    with pytest.raises(TypeError):
        run_bias_skew_finalization(source_data_dir="data/raw", model_dir="models")


# ---------------------------------------------------------------------------
# Hazard B (step 8): allow_overwrite must thread through all three call sites
# this step added guards to - refresh_dataset, write_audit_reports and
# run_gold_standard_sensitivity - the same way it already threads through
# train_models, train_ann and run_final_validation. Before this fix, a
# legitimate `--results-dir results --allow-overwrite` rerun into a populated
# directory would: pass the outer guard -> run train_models + train_ann
# (expensive) -> let run_final_validation overwrite its 10 files -> then abort
# at refresh_dataset's or write_audit_reports's own guard. Partial and
# destructive. Pattern matches
# tests/test_final_validation_results_guard.py's
# test_bias_skew_finalization_threads_allow_overwrite_and_writes_no_baseline_csv_itself.
# ---------------------------------------------------------------------------


def test_bias_skew_finalization_threads_allow_overwrite_into_refresh_dataset_and_write_audit_reports(
    tmp_path, monkeypatch
):
    bsf = _monkeypatch_heavy_dependencies(monkeypatch)

    calls = {}

    def _fake_refresh_dataset(**kwargs):
        calls["refresh_dataset"] = kwargs
        return None

    def _fake_write_audit_reports(**kwargs):
        calls["write_audit_reports"] = kwargs
        return (pd.DataFrame(), {"n_total": 0, "raw_n_records": 0})

    monkeypatch.setattr(bsf, "refresh_dataset", _fake_refresh_dataset)
    monkeypatch.setattr(bsf, "write_audit_reports", _fake_write_audit_reports)

    model_dir = tmp_path / "models"
    results_dir = tmp_path / "results"

    bsf.run_bias_skew_finalization(
        source_data_dir=str(tmp_path / "raw"),
        model_dir=str(model_dir),
        results_dir=str(results_dir),
        allow_overwrite=True,
    )

    assert calls["refresh_dataset"]["allow_overwrite"] is True
    assert calls["write_audit_reports"]["allow_overwrite"] is True


def test_bias_skew_finalization_threads_allow_overwrite_into_gold_standard_sensitivity(
    tmp_path, monkeypatch
):
    bsf = _monkeypatch_heavy_dependencies(monkeypatch)

    calls = {}

    def _fake_run_gold_standard_sensitivity(*args, **kwargs):
        calls["run_gold_standard_sensitivity"] = {"args": args, "kwargs": kwargs}
        return None

    monkeypatch.setattr(
        bsf, "run_gold_standard_sensitivity", _fake_run_gold_standard_sensitivity
    )

    model_dir = tmp_path / "models"
    results_dir = tmp_path / "results"

    bsf.run_bias_skew_finalization(
        source_data_dir=str(tmp_path / "raw"),
        model_dir=str(model_dir),
        results_dir=str(results_dir),
        allow_overwrite=True,
    )

    call = calls["run_gold_standard_sensitivity"]
    assert call["kwargs"].get("allow_overwrite") is True
    # results_dir, gs_sens_csv and gs_sens_md must still be positional and in
    # order - the enumeration note flags this call site as one that must not
    # be reordered.
    assert call["args"][0] == str(results_dir)
    assert call["args"][1] == str(results_dir / "gold_standard_sensitivity.csv")
    assert call["args"][2] == str(results_dir / "gold_standard_sensitivity.md")


def test_bias_skew_finalization_threads_allow_overwrite_false_too(tmp_path, monkeypatch):
    """allow_overwrite must thread through as False just as faithfully as
    True - a call site that only forwards a True value would still leave a
    default-False run silently assuming something different from what the
    caller asked for."""
    bsf = _monkeypatch_heavy_dependencies(monkeypatch)

    calls = {}

    def _fake_refresh_dataset(**kwargs):
        calls["refresh_dataset"] = kwargs
        return None

    def _fake_write_audit_reports(**kwargs):
        calls["write_audit_reports"] = kwargs
        return (pd.DataFrame(), {"n_total": 0, "raw_n_records": 0})

    def _fake_run_gold_standard_sensitivity(*args, **kwargs):
        calls["run_gold_standard_sensitivity"] = kwargs
        return None

    monkeypatch.setattr(bsf, "refresh_dataset", _fake_refresh_dataset)
    monkeypatch.setattr(bsf, "write_audit_reports", _fake_write_audit_reports)
    monkeypatch.setattr(
        bsf, "run_gold_standard_sensitivity", _fake_run_gold_standard_sensitivity
    )

    model_dir = tmp_path / "models"
    results_dir = tmp_path / "results"

    bsf.run_bias_skew_finalization(
        source_data_dir=str(tmp_path / "raw"),
        model_dir=str(model_dir),
        results_dir=str(results_dir),
        allow_overwrite=False,
    )

    assert calls["refresh_dataset"]["allow_overwrite"] is False
    assert calls["write_audit_reports"]["allow_overwrite"] is False
    assert calls["run_gold_standard_sensitivity"]["allow_overwrite"] is False


# ---------------------------------------------------------------------------
# CLI-level check
# ---------------------------------------------------------------------------


def test_bias_skew_finalization_cli_requires_results_dir_explicitly():
    """--results-dir must be rejected by argparse itself, before any work starts.

    The assertion is anchored on argparse's required-arguments line rather than
    searching all of stderr. --model-dir is supplied here, so a --results-dir
    that quietly regained a default would parse cleanly, start the run, and hit
    _guard_results_dir, whose own message reads "Point --results-dir at a fresh
    directory". That puts the flag name in stderr and satisfies a bare substring
    check while the regression is live, which is the opposite of what this test
    is for.
    """
    result = subprocess.run(
        [sys.executable, "-m", "src.bias_skew_finalization", "--model-dir", "models/scratch"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    required_lines = [
        ln for ln in result.stderr.splitlines() if "the following arguments are required:" in ln
    ]
    assert required_lines, (
        f"expected argparse to reject the run outright, got: {result.stderr[:400]}"
    )
    assert "--results-dir" in required_lines[0], (
        f"--results-dir was not named as required; it may have regained a default: "
        f"{required_lines[0][:200]}"
    )


def test_bias_skew_finalization_cli_advertises_allow_overwrite():
    result = subprocess.run(
        [sys.executable, "-m", "src.bias_skew_finalization", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "--allow-overwrite" in result.stdout
