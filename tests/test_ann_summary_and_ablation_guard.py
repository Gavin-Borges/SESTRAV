"""Overwrite-guard tests for scripts/compute_ann_baseline_summary.py and src/ablation_study.py.

These are the two remaining instances of the silent-overwrite defect class after
train_classifier / ann_benchmark / train_gnn / gnn_benchmark were fixed elsewhere.
`compute_ann_baseline_summary.py` defaulted `--output-summary` to the tracked
`models/ann_cv_summary.csv` and `--results-file` to the tracked
`models/training_results.csv` (silently merged into on every default run);
`ablation_study.py` defaulted `--output-dir` to `models`, threatening
`models/ablation_study_results.csv` (currently untracked, but the same class of
defect regardless - see CHANGELOG for the severity note).

`--results-file` is deliberately NOT guarded with a FileExistsError, matching
ann_benchmark's own exemption for the same reason: it is a merge target this
script only ever updates when explicitly given a path, never a fresh artifact it
silently creates. The fix there is that it no longer defaults into models/ at
all - omitting it now skips the merge outright. Both guarded call sites raise
before any read work, so the wiring tests below cost nothing even with a
nonexistent input path.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.compute_ann_baseline_summary import _guard_summary_path, main as ann_summary_main
from src.ablation_study import _guard_output_dir as guard_ablation, run_ablation

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# scripts/compute_ann_baseline_summary.py - guard behavior
# ---------------------------------------------------------------------------


def test_ann_summary_guard_passes_on_a_fresh_path(tmp_path):
    _guard_summary_path(str(tmp_path / "ann_cv_summary.csv"), allow_overwrite=False)


def test_ann_summary_guard_refuses_to_clobber_an_existing_summary(tmp_path):
    target = tmp_path / "ann_cv_summary.csv"
    target.write_text("metric,ann_cv_mean,ann_cv_std\nauc_roc,0.73,0.01\n")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _guard_summary_path(str(target), allow_overwrite=False)

    assert "0.73" in target.read_text()


def test_ann_summary_guard_names_the_blocking_file_and_the_escape_hatch(tmp_path):
    target = tmp_path / "ann_cv_summary.csv"
    target.write_text("published")

    with pytest.raises(FileExistsError) as excinfo:
        _guard_summary_path(str(target), allow_overwrite=False)

    message = str(excinfo.value)
    assert str(target) in message
    assert "--allow-overwrite" in message


def test_ann_summary_allow_overwrite_disarms_the_guard(tmp_path):
    target = tmp_path / "ann_cv_summary.csv"
    target.write_text("published")
    _guard_summary_path(str(target), allow_overwrite=True)


# ---------------------------------------------------------------------------
# scripts/compute_ann_baseline_summary.py - the guard is wired into main()
# ---------------------------------------------------------------------------
#
# The guard call precedes the --oof-file existence check, so a nonexistent
# --oof-file never gets touched if the guard raises first - that is exactly
# what this test pins.


def test_guard_is_wired_into_main_before_any_oof_read(tmp_path, monkeypatch):
    target = tmp_path / "ann_cv_summary.csv"
    target.write_text("published")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_ann_baseline_summary.py",
            "--oof-file",
            str(tmp_path / "does_not_exist.csv"),
            "--output-summary",
            str(target),
        ],
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        ann_summary_main()

    assert target.read_text() == "published"


def test_ann_summary_cli_requires_output_summary_explicitly():
    """No default destination: omitting --output-summary must fail argparse, not run."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.compute_ann_baseline_summary"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert "--output-summary" in result.stderr


def test_ann_summary_cli_results_file_has_no_default():
    """--results-file must stay optional (bracketed) so a bare run never merges into models/."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.compute_ann_baseline_summary", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "[--results-file RESULTS_FILE]" in result.stdout
    assert "--output-summary OUTPUT_SUMMARY" in result.stdout
    assert "[--output-summary" not in result.stdout


# ---------------------------------------------------------------------------
# src/ablation_study.py - guard behavior
# ---------------------------------------------------------------------------


def test_ablation_guard_passes_on_an_empty_directory(tmp_path):
    guard_ablation(str(tmp_path), allow_overwrite=False)


def test_ablation_guard_refuses_to_clobber_a_tracked_result(tmp_path):
    sentinel = tmp_path / "ablation_study_results.csv"
    sentinel.write_text("feature_set,auc_pr_mean\nfull_31,0.8639\n")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        guard_ablation(str(tmp_path), allow_overwrite=False)

    assert "0.8639" in sentinel.read_text()


def test_ablation_guard_names_the_blocking_file_and_the_escape_hatch(tmp_path):
    sentinel = tmp_path / "ablation_study_results.csv"
    sentinel.write_text("published")

    with pytest.raises(FileExistsError) as excinfo:
        guard_ablation(str(tmp_path), allow_overwrite=False)

    message = str(excinfo.value)
    assert "ablation_study_results.csv" in message
    assert "--allow-overwrite" in message


def test_ablation_allow_overwrite_disarms_the_guard(tmp_path):
    (tmp_path / "ablation_study_results.csv").write_text("published")
    guard_ablation(str(tmp_path), allow_overwrite=True)


# ---------------------------------------------------------------------------
# src/ablation_study.py - the guard is wired into run_ablation()
# ---------------------------------------------------------------------------
#
# The guard is the first statement in run_ablation, before pd.read_csv(data_path),
# so a nonexistent data/binding-matrix path never gets touched if it raises.


def test_guard_is_wired_into_run_ablation(tmp_path):
    sentinel = tmp_path / "ablation_study_results.csv"
    sentinel.write_text("published")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_ablation(
            str(tmp_path / "does_not_exist.csv"),
            str(tmp_path / "does_not_exist_matrix.csv"),
            str(tmp_path),
            allow_overwrite=False,
        )

    assert sentinel.read_text() == "published"


def test_run_ablation_requires_an_explicit_output_dir():
    """No default destination: omitting output_dir is a TypeError, not a write to models/."""
    with pytest.raises(TypeError):
        run_ablation("does_not_exist.csv", "does_not_exist_matrix.csv")
