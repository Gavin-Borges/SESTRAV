"""Overwrite-guard tests for src/final_validation_report.py and src/bias_skew_finalization.py.

These are results/ instances of the silent-overwrite defect class the
train_classifier / ann_benchmark / train_gnn / gnn_benchmark /
compute_ann_baseline_summary / ablation_study line already closed for models/.
Found by the claims-auditor while gating that line's last PR, and higher
severity than any single instance in it: results/h2_tier_a_summary.md is the
certified R10=0.9494 binding source in docs/claims_register.md, not just a
training artifact. scripts/run_analysis.py is the third instance in this same
results/ sub-line; its tests live in test_run_analysis_results_guard.py
instead, split out because importing it pulls in `shap`, which crashes the
Python process natively on at least one Windows dev machine in this project
(pre-existing, unrelated to this fix - see that file's docstring).

final_validation_report.py's guard has to know about the three mode/version
-tagged aliases canonical_output_filename produces, in addition to the 7 fixed
names - both the guard and the real atomic-publish step draw from the same
planned_validation_paths() helper, so they can never drift apart.

src/bias_skew_finalization.py is fixed alongside it: run_bias_skew_finalization
called run_final_validation() but ALSO, redundantly, computed and wrote
baseline_comparison.csv itself one step earlier - once final_validation_report
gained a guard, that redundant pre-write would have made even a first-ever run
into an empty directory trip the guard, since the file would always already
exist by the time run_final_validation's guard checked it. The redundant
write is removed rather than forcing allow_overwrite=True on that call site,
which would have silently disabled the guard for all 10 of
run_final_validation's files, not just the one redundant one.

Further instances of the same defect class in src/bias_skew_finalization.py's
own --results-dir (immunogenicity_provenance.csv, data_bias_audit*.{csv,md},
gold_standard_sensitivity*.{csv,md}, release_readiness_summary.md) are
deliberately NOT covered here - see
_local/notes/results-dir-silent-overwrite-2026-07-30.md for why, and for what
a follow-up PR needs to enumerate first.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.final_validation_report import (
    _guard_results_dir as guard_final_validation,
    planned_validation_paths,
    run_final_validation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# src/final_validation_report.py - planned path set
# ---------------------------------------------------------------------------


def test_final_validation_plans_all_ten_publish_names(tmp_path):
    names = {
        p.name for p in map(Path, planned_validation_paths(str(tmp_path), "expansion_v4", "4.0.0"))
    }
    assert names == {
        "gold_standard_validation.csv",
        "baseline_comparison.csv",
        "h2_tier_a_fold_metrics.csv",
        "h2_tier_a_summary.csv",
        "h2_tier_a_summary.md",
        "final_validation_report.md",
        "freeze_status.json",
        "gold_standard_validation__expansion_v4__4.0.0.csv",
        "baseline_comparison__expansion_v4__4.0.0.csv",
        "h2_tier_a_summary__expansion_v4__4.0.0.csv",
    }


def test_final_validation_tagged_names_change_with_dataset_mode(tmp_path):
    """A different dataset_mode/version must plan different tagged filenames."""
    v4_names = {p.name for p in map(Path, planned_validation_paths(str(tmp_path), "a", "1.0.0"))}
    v5_names = {p.name for p in map(Path, planned_validation_paths(str(tmp_path), "b", "2.0.0"))}
    assert v4_names != v5_names
    assert "gold_standard_validation__a__1.0.0.csv" in v4_names
    assert "gold_standard_validation__b__2.0.0.csv" in v5_names


# ---------------------------------------------------------------------------
# src/final_validation_report.py - guard behavior
# ---------------------------------------------------------------------------


def test_final_validation_guard_passes_on_an_empty_directory(tmp_path):
    guard_final_validation(str(tmp_path), "expansion_v4", "4.0.0", allow_overwrite=False)


def test_final_validation_guard_refuses_to_clobber_the_certified_ledger_file(tmp_path):
    """h2_tier_a_summary.md is the certified R10=0.9494 binding source - this is the case."""
    sentinel = tmp_path / "h2_tier_a_summary.md"
    sentinel.write_text("R10: 0.9494")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        guard_final_validation(str(tmp_path), "expansion_v4", "4.0.0", allow_overwrite=False)

    assert "0.9494" in sentinel.read_text()


def test_final_validation_guard_catches_a_tagged_alias_too(tmp_path):
    sentinel = tmp_path / "baseline_comparison__expansion_v4__4.0.0.csv"
    sentinel.write_text("published")

    with pytest.raises(FileExistsError) as excinfo:
        guard_final_validation(str(tmp_path), "expansion_v4", "4.0.0", allow_overwrite=False)

    assert "baseline_comparison__expansion_v4__4.0.0.csv" in str(excinfo.value)


def test_final_validation_allow_overwrite_disarms_the_guard(tmp_path):
    (tmp_path / "freeze_status.json").write_text("{}")
    guard_final_validation(str(tmp_path), "expansion_v4", "4.0.0", allow_overwrite=True)


# ---------------------------------------------------------------------------
# src/final_validation_report.py - the guard is wired into run_final_validation()
# ---------------------------------------------------------------------------
#
# The guard is the first statement in run_final_validation, before
# tempfile.mkdtemp or any of the three stage functions run, so a nonexistent
# --data / --binding-matrix / --model-path never gets touched if it raises.


def test_guard_is_wired_into_run_final_validation(tmp_path):
    sentinel = tmp_path / "final_validation_report.md"
    sentinel.write_text("published")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_final_validation(
            results_dir=str(tmp_path),
            data_path=str(tmp_path / "does_not_exist.csv"),
            binding_matrix_path=str(tmp_path / "does_not_exist_matrix.csv"),
            model_path=str(tmp_path / "does_not_exist_model.joblib"),
            allow_overwrite=False,
        )

    assert sentinel.read_text() == "published"


def test_run_final_validation_requires_an_explicit_results_dir():
    """No default destination: omitting results_dir is a TypeError."""
    with pytest.raises(TypeError):
        run_final_validation()


def test_final_validation_cli_requires_results_dir_explicitly():
    result = subprocess.run(
        [sys.executable, "-m", "src.final_validation_report"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert "--results-dir" in result.stderr


def test_final_validation_cli_advertises_allow_overwrite():
    result = subprocess.run(
        [sys.executable, "-m", "src.final_validation_report", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "--allow-overwrite" in result.stdout


# ---------------------------------------------------------------------------
# src/bias_skew_finalization.py - the redundant pre-write is gone, and
# allow_overwrite threads through to the now-guarded run_final_validation call
# ---------------------------------------------------------------------------
#
# Before this fix, run_bias_skew_finalization computed and wrote
# baseline_comparison.csv directly, one step before calling
# run_final_validation - which computes and publishes the identical file
# itself. Once run_final_validation gained a guard, that redundant pre-write
# would have made even a first-ever run into an empty directory trip the
# guard. All heavy dependencies (training, dataset refresh, H2 evaluation) are
# faked here: this test is about the call graph and file-write surface, not
# about running the real pipeline.


def test_bias_skew_finalization_no_longer_imports_compare_methods():
    """Structural proof the redundant direct call site is gone, not just untested."""
    import src.bias_skew_finalization as bsf

    assert not hasattr(bsf, "compare_methods")


def test_bias_skew_finalization_threads_allow_overwrite_and_writes_no_baseline_csv_itself(
    tmp_path, monkeypatch
):
    import src.bias_skew_finalization as bsf

    calls = {}

    monkeypatch.setattr(bsf, "_collect_raw_records", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(bsf, "refresh_dataset", lambda *a, **k: None)
    monkeypatch.setattr(
        bsf,
        "write_audit_reports",
        lambda *a, **k: (pd.DataFrame(), {"n_total": 0, "raw_n_records": 0}),
    )
    monkeypatch.setattr(bsf, "train_models", lambda *a, **k: None)
    monkeypatch.setattr(bsf, "train_ann", lambda *a, **k: None)
    monkeypatch.setattr(bsf, "run_gold_standard_sensitivity", lambda *a, **k: None)

    def _fake_run_final_validation(**kwargs):
        calls["run_final_validation"] = kwargs
        return ("gs.csv", "baseline.csv", "final.md")

    monkeypatch.setattr(bsf, "run_final_validation", _fake_run_final_validation)

    model_dir = tmp_path / "models"
    results_dir = tmp_path / "results"

    bsf.run_bias_skew_finalization(
        source_data_dir=str(tmp_path / "raw"),
        model_dir=str(model_dir),
        results_dir=str(results_dir),
        allow_overwrite=True,
    )

    assert calls["run_final_validation"]["allow_overwrite"] is True
    assert not (results_dir / "baseline_comparison.csv").exists()
