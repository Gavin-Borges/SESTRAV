"""Overwrite-guard tests for src/baseline_comparison.py's --results-dir.

Tier-1 instance #6 of the results/ silent-overwrite defect class (see
_local/notes/results-dir-tier1-enumeration-2026-07-30.md). --results-dir
defaulted to "results" with no guard, so a bare
`python -m src.baseline_comparison` rewrote the git-tracked
results/baseline_comparison.csv in place.

Two things make this module's repair different from its siblings, and both
shape this file.

1. **--results-dir is dual-purpose.** The per-virus {prefix}_features.csv
   inputs are read out of it and baseline_comparison.csv is written back into
   it. Requiring the flag is still the right fix, but the family's standard
   "Point --results-dir at a fresh directory" remedy is actively wrong advice
   here: a fresh directory has no feature CSVs to read, so following it trades
   this error for a different one. Only `remedy` is overridden, not `scope` -
   unlike the file-path flags elsewhere in this line there really is a single
   directory to name, so the default "under '<dir>'" clause stays accurate.
2. **There is no public write API.** compare_methods() returns a DataFrame; the
   __main__ block writes it. The guard therefore sits in __main__, immediately
   after the existing directory validation and before compare_methods() - which
   is where all the work is. The consequence is that this module gets a
   CLI-level guard only, with no Python-level allow_overwrite parameter,
   because there is no Python-level write API to attach one to. The wiring test
   below is correspondingly a subprocess test rather than a direct call.

CLI checks anchor on argparse's required-arguments line rather than a bare
stderr substring: the guard's own FileExistsError message names --results-dir
too and would give a false pass on a substring check while a regression was
live.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src import baseline_comparison as bc

REPO_ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_NAME = "baseline_comparison.csv"


# ---------------------------------------------------------------------------
# Planned-path enumeration
# ---------------------------------------------------------------------------


def test_planned_paths_lists_the_single_artifact():
    paths = bc.planned_baseline_comparison_paths("out")
    assert [Path(p).name for p in paths] == [ARTIFACT_NAME]


def test_planned_paths_are_scoped_to_the_given_results_dir(tmp_path):
    paths = bc.planned_baseline_comparison_paths(str(tmp_path))
    assert [Path(p) for p in paths] == [tmp_path / ARTIFACT_NAME]


# ---------------------------------------------------------------------------
# Guard behaviour
# ---------------------------------------------------------------------------


def test_guard_passes_on_an_empty_directory(tmp_path):
    bc._guard_results_dir(str(tmp_path), allow_overwrite=False)


def test_guard_ignores_the_feature_csvs_it_reads(tmp_path):
    """The inputs share the directory but are not planned writes."""
    (tmp_path / "EBV_B95_8_panel8_features.csv").write_text("peptide\n", encoding="utf-8")
    (tmp_path / "HPV16_18_panel8_features.csv").write_text("peptide\n", encoding="utf-8")
    bc._guard_results_dir(str(tmp_path), allow_overwrite=False)


@pytest.mark.parametrize("name", [ARTIFACT_NAME])
def test_guard_refuses_to_clobber_each_artifact(tmp_path, name):
    (tmp_path / name).write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        bc._guard_results_dir(str(tmp_path), allow_overwrite=False)
    assert name in str(exc.value)


def test_guard_names_the_blocking_file_and_the_escape_hatch(tmp_path):
    (tmp_path / ARTIFACT_NAME).write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        bc._guard_results_dir(str(tmp_path), allow_overwrite=False)
    message = str(exc.value)
    assert ARTIFACT_NAME in message
    assert "--allow-overwrite" in message


def test_guard_does_not_advise_pointing_a_dual_purpose_flag_at_a_fresh_directory(tmp_path):
    """--results-dir is read from as well as written into, so the family's
    default remedy would send the caller to a directory with no inputs."""
    (tmp_path / ARTIFACT_NAME).write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        bc._guard_results_dir(str(tmp_path), allow_overwrite=False)
    message = str(exc.value)
    assert "Point --results-dir at a fresh directory" not in message
    assert "read from as well as written into" in message
    assert "move or rename the existing file" in message


def test_guard_still_names_the_directory_it_checked(tmp_path):
    """`scope` is deliberately NOT overridden: there is one real directory."""
    (tmp_path / ARTIFACT_NAME).write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        bc._guard_results_dir(str(tmp_path), allow_overwrite=False)
    assert f"under '{tmp_path}'" in str(exc.value)


def test_guard_discloses_that_the_escape_hatch_is_cli_only(tmp_path):
    """No Python-level allow_overwrite exists to point the reader at, so the
    message must not invent one."""
    (tmp_path / ARTIFACT_NAME).write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        bc._guard_results_dir(str(tmp_path), allow_overwrite=False)
    assert "CLI only" in str(exc.value)


def test_allow_overwrite_disarms_the_guard(tmp_path):
    (tmp_path / ARTIFACT_NAME).write_text("existing", encoding="utf-8")
    bc._guard_results_dir(str(tmp_path), allow_overwrite=True)


# ---------------------------------------------------------------------------
# Wiring: __main__ must guard before compare_methods() does any work
# ---------------------------------------------------------------------------


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "src.baseline_comparison", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def _fixture_dirs(tmp_path) -> tuple[Path, Path]:
    """A results dir and a model dir that both exist and are both empty.

    Both have to exist or __main__'s pre-existing parser.error() validation
    fires first and the run would abort for the wrong reason - which would let
    this test pass with the guard removed. The model dir is created inside
    tmp_path rather than reusing the repo's models/, which is gitignored and
    absent from a fresh CI checkout.
    """
    results_dir = tmp_path / "results"
    model_dir = tmp_path / "models"
    results_dir.mkdir()
    model_dir.mkdir()
    return results_dir, model_dir


def test_guard_is_wired_into_main_before_any_work(tmp_path):
    """A defined-but-uncalled guard would pass every test above.

    With an empty results dir, compare_methods() finds no feature CSVs, returns
    an empty frame and the run writes a header-only baseline_comparison.csv over
    whatever was there - i.e. without the guard this invocation succeeds and
    clobbers. Getting a refusal instead, with none of compare_methods()'s
    "[Baseline]" progress output on stdout, proves the guard runs first.
    """
    results_dir, model_dir = _fixture_dirs(tmp_path)
    artifact = results_dir / ARTIFACT_NAME
    artifact.write_text("published", encoding="utf-8")

    result = _run_module("--results-dir", str(results_dir), "--model-dir", str(model_dir))

    assert result.returncode != 0
    assert "Refusing to overwrite" in result.stderr
    assert "[Baseline]" not in result.stdout, (
        "compare_methods() ran before the guard rejected the run"
    )
    assert artifact.read_text(encoding="utf-8") == "published"


def test_allow_overwrite_passes_through_main(tmp_path):
    """With the escape hatch set, the guard must not be what stops the run."""
    results_dir, model_dir = _fixture_dirs(tmp_path)
    artifact = results_dir / ARTIFACT_NAME
    artifact.write_text("published", encoding="utf-8")

    result = _run_module(
        "--results-dir",
        str(results_dir),
        "--model-dir",
        str(model_dir),
        "--allow-overwrite",
    )

    assert result.returncode == 0, result.stderr[-800:]
    assert artifact.read_text(encoding="utf-8") != "published"


def test_a_clean_results_dir_is_not_blocked(tmp_path):
    results_dir, model_dir = _fixture_dirs(tmp_path)

    result = _run_module("--results-dir", str(results_dir), "--model-dir", str(model_dir))

    assert result.returncode == 0, result.stderr[-800:]
    assert (results_dir / ARTIFACT_NAME).is_file()


def test_guard_call_precedes_compare_methods_in_main():
    """Placement is the whole point: guarding after compare_methods() would
    still abort the run, but only after paying for every model load and scoring
    pass. Source order is the cheap check; the subprocess test above is the
    behavioural one.
    """
    source = (REPO_ROOT / "src" / "baseline_comparison.py").read_text(encoding="utf-8")
    main_start = source.index('if __name__ == "__main__":')
    main_block = source[main_start:]
    guard_at = main_block.index("_guard_results_dir(args.results_dir")
    work_at = main_block.index("compare_methods(args.results_dir")
    assert guard_at < work_at, "__main__ calls compare_methods() before guarding"


def test_compare_methods_is_still_a_pure_read_and_return():
    """The reason the guard lives in __main__ rather than in a run_* function.

    If compare_methods ever starts writing, the guard's placement stops being
    sufficient and this module needs the public-function treatment its siblings
    got.
    """
    source = (REPO_ROOT / "src" / "baseline_comparison.py").read_text(encoding="utf-8")
    body_start = source.index("def compare_methods(")
    body = source[body_start : source.index("\ndef ", body_start + 1)]
    assert "to_csv(" not in body
    assert "open(" not in body


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------

REQUIRED_ARGS_MARKER = "the following arguments are required:"


def test_cli_requires_results_dir_explicitly():
    result = _run_module("--model-dir", "models")
    assert result.returncode != 0
    required_lines = [ln for ln in result.stderr.splitlines() if REQUIRED_ARGS_MARKER in ln]
    assert required_lines, (
        f"expected argparse to reject the run outright, got: {result.stderr[:400]}"
    )
    assert "--results-dir" in required_lines[0], (
        f"--results-dir was not named as required; it may have regained a default: "
        f"{required_lines[0][:200]}"
    )


def test_cli_advertises_allow_overwrite():
    result = _run_module("--help")
    assert result.returncode == 0
    assert "--allow-overwrite" in result.stdout
