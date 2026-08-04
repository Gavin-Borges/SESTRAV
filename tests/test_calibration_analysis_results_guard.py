"""Overwrite-guard tests for src/calibration_analysis.py's --output-dir.

Tier-1 instance #7 of the results/ silent-overwrite defect class (see
_local/notes/results-dir-tier1-enumeration-2026-07-30.md). --output-dir
defaulted to "results" with no guard, so a bare
`python -m src.calibration_analysis` rewrote three artifacts in place, one of
them the git-tracked results/calibration_metrics.csv.

--output-dir is now required with no default at both the CLI and the
run_calibration_analysis(output_dir=...) layers. It moved ahead of the
defaulted v1_oof_path/method parameters in the signature, because Python does
not allow a no-default parameter after a defaulted one; that reorder is why the
caller-shape test at the bottom of this file exists, since a positional caller
would silently pass its v1 OOF path as the output directory rather than fail.

Mirrors tests/test_h2_tier_a_results_guard.py's structure: planned-path
enumeration, guard behaviour in isolation, a wiring test proving the guard
actually runs inside run_calibration_analysis rather than merely being defined,
an allow-overwrite-passthrough test, and CLI-level checks anchored on argparse's
required-arguments line rather than a bare stderr substring - the guard's own
FileExistsError message names --output-dir too and would give a false pass on a
substring check while a regression was live.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src import calibration_analysis as ca

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_NAMES = [
    "calibration_reliability_diagram.png",
    "calibration_score_distribution.png",
    "calibration_metrics.csv",
]


# ---------------------------------------------------------------------------
# Planned-path enumeration
# ---------------------------------------------------------------------------


def test_planned_paths_lists_all_three_artifacts():
    paths = ca.planned_calibration_paths("out")
    assert [Path(p).name for p in paths] == EXPECTED_NAMES


def test_planned_paths_are_scoped_to_the_given_output_dir(tmp_path):
    paths = ca.planned_calibration_paths(str(tmp_path))
    for p in paths:
        assert Path(p).parent == tmp_path


def test_planned_paths_are_unique():
    paths = ca.planned_calibration_paths("out")
    assert len(paths) == len(set(paths))


# ---------------------------------------------------------------------------
# Guard behaviour
# ---------------------------------------------------------------------------


def test_guard_passes_on_an_empty_directory(tmp_path):
    ca._guard_output_dir(str(tmp_path), allow_overwrite=False)


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_guard_refuses_to_clobber_each_artifact(tmp_path, name):
    (tmp_path / name).write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        ca._guard_output_dir(str(tmp_path), allow_overwrite=False)
    assert name in str(exc.value)


def test_guard_names_the_blocking_file_and_the_escape_hatch(tmp_path):
    (tmp_path / "calibration_metrics.csv").write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        ca._guard_output_dir(str(tmp_path), allow_overwrite=False)
    message = str(exc.value)
    assert "calibration_metrics.csv" in message
    assert "--allow-overwrite" in message
    assert "run_calibration_analysis(..., allow_overwrite=True)" in message


def test_guard_reports_every_colliding_file_not_just_the_first(tmp_path):
    for name in EXPECTED_NAMES:
        (tmp_path / name).write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        ca._guard_output_dir(str(tmp_path), allow_overwrite=False)
    message = str(exc.value)
    for name in EXPECTED_NAMES:
        assert name in message
    assert "3 existing artifact(s)" in message


def test_allow_overwrite_disarms_the_guard(tmp_path):
    for name in EXPECTED_NAMES:
        (tmp_path / name).write_text("existing", encoding="utf-8")
    ca._guard_output_dir(str(tmp_path), allow_overwrite=True)


# ---------------------------------------------------------------------------
# Wiring: the guard must actually run inside run_calibration_analysis
# ---------------------------------------------------------------------------


def test_guard_is_wired_into_run_calibration_analysis(tmp_path):
    """A defined-but-uncalled guard would pass every test above.

    v2_oof_path points at a file that does not exist, so without the guard the
    run would fail with FileNotFoundError while reading it. Getting
    FileExistsError instead proves the guard runs before any read.
    """
    (tmp_path / "calibration_metrics.csv").write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ca.run_calibration_analysis(
            v2_oof_path=str(tmp_path / "does_not_exist.csv"),
            output_dir=str(tmp_path),
        )


def test_allow_overwrite_passes_through_run_calibration_analysis(tmp_path):
    """With the escape hatch set, the guard must not be what stops the run."""
    (tmp_path / "calibration_metrics.csv").write_text("published", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        ca.run_calibration_analysis(
            v2_oof_path=str(tmp_path / "does_not_exist.csv"),
            output_dir=str(tmp_path),
            allow_overwrite=True,
        )


def test_run_calibration_analysis_requires_an_explicit_output_dir():
    """No default destination: omitting output_dir is a TypeError."""
    with pytest.raises(TypeError):
        ca.run_calibration_analysis("models/rf_oof_predictions.csv")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Caller shape: the parameter reorder is silent-failure-shaped
# ---------------------------------------------------------------------------


def test_regenerate_shareout_calls_calibration_by_keyword():
    """output_dir moved ahead of v1_oof_path, so positional callers would break
    silently rather than loudly: the old third positional argument was the
    output directory and is now the v1 OOF path. The one in-repo caller,
    scripts/regenerate_shareout_pngs.py, passes everything by keyword; lock that
    down, along with the deliberate allow_overwrite=True it needs (regenerating
    into results/ is that script's declared purpose).
    """
    source = (REPO_ROOT / "scripts" / "regenerate_shareout_pngs.py").read_text(encoding="utf-8")
    call_start = source.index("    run_calibration_analysis(")
    call_block = source[call_start : source.index("    )", call_start)]
    assert "v2_oof_path=" in call_block
    assert "output_dir=" in call_block
    assert "v1_oof_path=" in call_block
    assert "allow_overwrite=True," in call_block


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------

REQUIRED_ARGS_MARKER = "the following arguments are required:"


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "src.calibration_analysis", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_cli_requires_output_dir_explicitly():
    result = _run_module("--method", "RandomForest")
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


def test_cli_actually_threads_allow_overwrite_into_run_calibration_analysis():
    """Advertising the flag is not the same as wiring it.

    An --allow-overwrite that argparse accepts but __main__ never forwards is a
    silent no-op: --help looks right, the flag is accepted, and the guard still
    aborts the run it was meant to permit.
    """
    source = (REPO_ROOT / "src" / "calibration_analysis.py").read_text(encoding="utf-8")
    call_start = source.index("    run_calibration_analysis(")
    call_block = source[call_start : source.index("    )", call_start)]
    assert "allow_overwrite=args.allow_overwrite," in call_block, (
        "__main__ parses --allow-overwrite but never forwards it to run_calibration_analysis"
    )
