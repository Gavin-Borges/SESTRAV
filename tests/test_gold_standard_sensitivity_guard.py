"""Overwrite-guard tests for src/gold_standard_sensitivity.py's --output-csv
and --output-md.

Step 8 (second half) of the results/ silent-overwrite defect-class repair
line. Unlike every other instance in this line, src/gold_standard_sensitivity.py
writes ZERO git-tracked files: gold_standard_sensitivity.*  is untracked (a
different artifact from the similarly-named, tracked
gold_standard_validation.csv written by a different module). This guard exists
for consistency with the rest of the defect-class family and for CHANGELOG
disclosure closure, not to protect a published result - the module docstring
on _guard_gold_standard_sensitivity and the CHANGELOG entry for this change
both say so explicitly, and this file does not claim otherwise either.

--results-dir is a READ-ONLY input here (joins {prefix}_ranked.csv, only ever
read), not a write target, and is deliberately left untouched: still optional,
still defaulting to "results", not part of the guarded set.
test_results_dir_flag_keeps_its_existing_default locks that down.

This module has no single output directory - --output-csv and --output-md are
independent file-path flags - so it does not fit
tests/test_artifact_guard_contract.py's planned_paths_under registration
shape, matching src/data_bias_audit.py's situation. See
src/artifact_guard.py's module docstring for the scope/remedy template
extension that made this guard possible.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src import gold_standard_sensitivity as gss

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_NAMES = [
    "gold_standard_sensitivity.csv",
    "gold_standard_sensitivity_deltas.csv",
    "gold_standard_sensitivity.md",
]

REQUIRED_ARGS_MARKER = "the following arguments are required:"


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "src.gold_standard_sensitivity", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# Planned-path enumeration
# ---------------------------------------------------------------------------


def test_planned_paths_lists_all_three_artifacts(tmp_path):
    output_csv = str(tmp_path / "gold_standard_sensitivity.csv")
    output_md = str(tmp_path / "gold_standard_sensitivity.md")
    paths = gss.planned_gold_standard_sensitivity_paths(output_csv, output_md)
    assert [Path(p).name for p in paths] == EXPECTED_NAMES


def test_planned_paths_derived_name_matches_the_writer_exactly(tmp_path):
    """The derived name must come from the identical output_csv.replace(".csv",
    "_deltas.csv") expression run_gold_standard_sensitivity itself uses
    (str.replace, not os.path.splitext), including on a path containing ".csv"
    twice, where the two diverge."""
    output_csv = str(tmp_path / "sensitivity.csv.backup.csv")
    expected_derived = output_csv.replace(".csv", "_deltas.csv")
    paths = gss.planned_gold_standard_sensitivity_paths(output_csv, "out.md")
    assert expected_derived in paths
    assert Path(expected_derived).name == "sensitivity_deltas.csv.backup_deltas.csv"


# ---------------------------------------------------------------------------
# Guard behaviour
# ---------------------------------------------------------------------------


def test_guard_passes_on_a_nonexistent_location(tmp_path):
    gss._guard_gold_standard_sensitivity(
        str(tmp_path / "gold_standard_sensitivity.csv"),
        str(tmp_path / "gold_standard_sensitivity.md"),
        False,
    )


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_guard_refuses_to_clobber_each_artifact(tmp_path, name):
    output_csv = tmp_path / "gold_standard_sensitivity.csv"
    output_md = tmp_path / "gold_standard_sensitivity.md"
    (tmp_path / name).write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        gss._guard_gold_standard_sensitivity(str(output_csv), str(output_md), False)
    assert name in str(exc.value)


def test_guard_reports_every_colliding_file_not_just_the_first(tmp_path):
    output_csv = tmp_path / "gold_standard_sensitivity.csv"
    output_md = tmp_path / "gold_standard_sensitivity.md"
    for name in EXPECTED_NAMES:
        (tmp_path / name).write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        gss._guard_gold_standard_sensitivity(str(output_csv), str(output_md), False)
    message = str(exc.value)
    for name in EXPECTED_NAMES:
        assert name in message
    assert "3 existing artifact(s)" in message


def test_guard_names_the_flag_and_the_escape_hatch(tmp_path):
    output_csv = tmp_path / "gold_standard_sensitivity.csv"
    output_md = tmp_path / "gold_standard_sensitivity.md"
    output_md.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        gss._guard_gold_standard_sensitivity(str(output_csv), str(output_md), False)
    message = str(exc.value)
    assert "--output-csv" in message
    assert "--output-md" in message
    assert "--allow-overwrite" in message
    assert "run_gold_standard_sensitivity(..., allow_overwrite=True)" in message


def test_allow_overwrite_disarms_the_guard(tmp_path):
    output_csv = tmp_path / "gold_standard_sensitivity.csv"
    output_md = tmp_path / "gold_standard_sensitivity.md"
    for name in EXPECTED_NAMES:
        (tmp_path / name).write_text("existing", encoding="utf-8")
    gss._guard_gold_standard_sensitivity(str(output_csv), str(output_md), True)


# ---------------------------------------------------------------------------
# Wiring: the guard must actually run inside run_gold_standard_sensitivity
# ---------------------------------------------------------------------------


def test_guard_is_wired_into_run_gold_standard_sensitivity(tmp_path, monkeypatch):
    """A defined-but-uncalled guard would pass every test above."""
    output_csv = tmp_path / "gold_standard_sensitivity.csv"
    output_md = tmp_path / "gold_standard_sensitivity.md"
    output_md.write_text("published", encoding="utf-8")

    def _should_not_run():
        raise AssertionError(
            "run_gold_standard_sensitivity did work before the guard rejected the run"
        )

    monkeypatch.setattr(gss, "_sets", _should_not_run)

    with pytest.raises(FileExistsError):
        gss.run_gold_standard_sensitivity(str(tmp_path), str(output_csv), str(output_md))


def test_allow_overwrite_passes_through_run_gold_standard_sensitivity(tmp_path, monkeypatch):
    """With the escape hatch set, the guard must not be what stops the run."""
    output_csv = tmp_path / "gold_standard_sensitivity.csv"
    output_md = tmp_path / "gold_standard_sensitivity.md"
    output_md.write_text("published", encoding="utf-8")

    sentinel = RuntimeError("got past the guard")

    def _stop_after_guard():
        raise sentinel

    monkeypatch.setattr(gss, "_sets", _stop_after_guard)

    with pytest.raises(RuntimeError) as exc:
        gss.run_gold_standard_sensitivity(
            str(tmp_path), str(output_csv), str(output_md), allow_overwrite=True
        )
    assert exc.value is sentinel


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--output-csv", "--output-md"])
def test_cli_requires_flag_explicitly(flag):
    """Anchored on argparse's required-arguments line, not a bare stderr
    substring - the guard's own error message also names these flags, which
    would satisfy a substring check even during a live regression."""
    result = _run_module()
    assert result.returncode != 0
    required_lines = [ln for ln in result.stderr.splitlines() if REQUIRED_ARGS_MARKER in ln]
    assert required_lines, (
        f"expected argparse to reject the run outright, got: {result.stderr[:400]}"
    )
    assert flag in required_lines[0], (
        f"{flag} was not named as required; it may have regained a default: "
        f"{required_lines[0][:200]}"
    )


def test_cli_advertises_allow_overwrite():
    result = _run_module("--help")
    assert result.returncode == 0
    assert "--allow-overwrite" in result.stdout


def test_results_dir_flag_keeps_its_existing_default():
    """--results-dir is a read-only input (only ever joined with
    {prefix}_ranked.csv and read), not a write target - it must stay optional
    with its existing 'results' default, unlike --output-csv/--output-md."""
    result = _run_module("--help")
    assert result.returncode == 0
    assert "[--results-dir" in result.stdout


def test_cli_actually_threads_allow_overwrite_into_run_gold_standard_sensitivity():
    """Advertising the flag is not the same as wiring it."""
    source = (REPO_ROOT / "src" / "gold_standard_sensitivity.py").read_text(encoding="utf-8")
    call_start = source.index("    run_gold_standard_sensitivity(")
    call_block = source[call_start : source.index("    )", call_start)]
    assert "allow_overwrite=args.allow_overwrite," in call_block
