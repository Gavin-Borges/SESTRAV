"""Overwrite-guard tests for src/h2_tier_a_evaluation.py's --output-dir.

Tier-1 instance of the results/ silent-overwrite defect class. --output-dir
defaulted to "results" with no guard, so a bare
`python -m src.h2_tier_a_evaluation` rewrote three git-tracked release
artifacts in place. One of them, results/h2_tier_a_summary.md, is the source
behind the certified R10 H2 Tier A null result reported in README.md (R10 =
1.0588 as of the 2026-08-10 D17 regeneration; previously 0.9494, retracted as
void), so a silent rewrite could change a published number with no commit
showing a science change.

These cover the same ground as the guard tests for the other entry points in
this line: planned-path enumeration, the guard's own behaviour, that the guard
is actually wired into run_h2_tier_a rather than merely defined, and the CLI
contract. The CLI checks anchor on argparse's required-arguments line rather
than searching all of stderr, because the guard's own error message names
--output-dir and would otherwise satisfy a bare substring check while the
regression was live.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src import h2_tier_a_evaluation as h2

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_NAMES = [
    "h2_tier_a_fold_metrics.csv",
    "h2_tier_a_summary.csv",
    "h2_tier_a_summary.md",
]


# ---------------------------------------------------------------------------
# Planned-path enumeration
# ---------------------------------------------------------------------------


def test_planned_paths_lists_all_three_artifacts():
    paths = h2.planned_h2_tier_a_paths("out")
    assert [Path(p).name for p in paths] == EXPECTED_NAMES


def test_planned_paths_are_scoped_to_the_given_output_dir(tmp_path):
    paths = h2.planned_h2_tier_a_paths(str(tmp_path))
    for p in paths:
        assert Path(p).parent == tmp_path


# ---------------------------------------------------------------------------
# Guard behaviour
# ---------------------------------------------------------------------------


def test_guard_passes_on_an_empty_directory(tmp_path):
    h2._guard_output_dir(str(tmp_path), allow_overwrite=False)


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_guard_refuses_to_clobber_each_artifact(tmp_path, name):
    (tmp_path / name).write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        h2._guard_output_dir(str(tmp_path), allow_overwrite=False)
    assert name in str(exc.value)


def test_guard_names_the_blocking_file_and_the_escape_hatch(tmp_path):
    (tmp_path / "h2_tier_a_summary.md").write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        h2._guard_output_dir(str(tmp_path), allow_overwrite=False)
    message = str(exc.value)
    assert "h2_tier_a_summary.md" in message
    assert "--allow-overwrite" in message


def test_guard_reports_every_colliding_file_not_just_the_first(tmp_path):
    for name in EXPECTED_NAMES:
        (tmp_path / name).write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        h2._guard_output_dir(str(tmp_path), allow_overwrite=False)
    message = str(exc.value)
    for name in EXPECTED_NAMES:
        assert name in message
    assert "3 existing artifact(s)" in message


def test_allow_overwrite_disarms_the_guard(tmp_path):
    for name in EXPECTED_NAMES:
        (tmp_path / name).write_text("existing", encoding="utf-8")
    h2._guard_output_dir(str(tmp_path), allow_overwrite=True)


# ---------------------------------------------------------------------------
# Wiring: the guard must actually run inside run_h2_tier_a
# ---------------------------------------------------------------------------


def test_guard_is_wired_into_run_h2_tier_a(tmp_path, monkeypatch):
    """A defined-but-uncalled guard would pass every test above."""
    (tmp_path / "h2_tier_a_summary.csv").write_text("published", encoding="utf-8")

    def _should_not_run(*args, **kwargs):
        raise AssertionError("run_h2_tier_a did work before the guard rejected the run")

    # Everything after the guard must never execute.
    monkeypatch.setattr(h2.os, "makedirs", _should_not_run)

    with pytest.raises(FileExistsError):
        h2.run_h2_tier_a(
            data_path="does_not_matter.csv",
            model_path="does_not_matter.joblib",
            binding_matrix_path="does_not_matter.csv",
            output_dir=str(tmp_path),
        )


def test_allow_overwrite_passes_through_run_h2_tier_a(tmp_path, monkeypatch):
    """With the escape hatch set, the guard must not be what stops the run."""
    (tmp_path / "h2_tier_a_summary.csv").write_text("published", encoding="utf-8")

    sentinel = RuntimeError("got past the guard")

    def _stop_after_guard(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(h2.os, "makedirs", _stop_after_guard)

    with pytest.raises(RuntimeError) as exc:
        h2.run_h2_tier_a(
            data_path="does_not_matter.csv",
            model_path="does_not_matter.joblib",
            binding_matrix_path="does_not_matter.csv",
            output_dir=str(tmp_path),
            allow_overwrite=True,
        )
    assert exc.value is sentinel


# ---------------------------------------------------------------------------
# The run_final_validation interaction
# ---------------------------------------------------------------------------


def test_final_validation_calls_h2_into_a_fresh_temp_dir_not_results():
    """run_final_validation must keep passing a temp dir, or this guard bites it.

    src/final_validation_report.py already guards these same three filenames and
    then calls run_h2_tier_a. It stays compatible only because it passes a
    per-run tempfile.mkdtemp() directory rather than results/ itself, so the new
    guard finds nothing there. If that ever changes to write straight into
    results/, the pipeline.smk --allow-overwrite reproduction path would start
    failing on the guard, so lock the property down here.
    """
    source = (REPO_ROOT / "src" / "final_validation_report.py").read_text(encoding="utf-8")
    assert "tmp_dir = tempfile.mkdtemp(" in source
    assert "output_dir=tmp_dir," in source


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------

REQUIRED_ARGS_MARKER = "the following arguments are required:"


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "src.h2_tier_a_evaluation", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_cli_requires_output_dir_explicitly():
    result = _run_module("--data", "does_not_exist.csv")
    assert result.returncode != 0
    required_lines = [ln for ln in result.stderr.splitlines() if REQUIRED_ARGS_MARKER in ln]
    assert required_lines, (
        f"expected argparse to reject the run outright, got: {result.stderr[:400]}"
    )
    assert "--output-dir" in required_lines[0], (
        f"--output-dir was not named as required; it may have regained a default: "
        f"{required_lines[0][:200]}"
    )


def test_cli_advertises_allow_overwrite():
    result = _run_module("--help")
    assert result.returncode == 0
    assert "--allow-overwrite" in result.stdout


def test_cli_actually_threads_allow_overwrite_into_run_h2_tier_a():
    """Advertising the flag is not the same as wiring it.

    An --allow-overwrite that argparse accepts but __main__ never forwards is a
    silent no-op: --help looks right, the flag is accepted, and the guard still
    aborts the run it was meant to permit. Checking the source of the __main__
    call is cheap; the alternative is executing the real evaluation.
    """
    source = (REPO_ROOT / "src" / "h2_tier_a_evaluation.py").read_text(encoding="utf-8")
    call_start = source.index("    run_h2_tier_a(")
    call_block = source[call_start : source.index("    )", call_start)]
    assert "allow_overwrite=args.allow_overwrite," in call_block, (
        "__main__ parses --allow-overwrite but never forwards it to run_h2_tier_a"
    )
