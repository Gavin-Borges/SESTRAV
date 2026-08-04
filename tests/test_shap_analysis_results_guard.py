"""Overwrite-guard tests for src/shap_analysis.py's --output-dir.

Tier-1 instance #10 of the results/ silent-overwrite defect class (see
_local/notes/results-dir-tier1-enumeration-2026-07-30.md). --output-dir
defaulted to "results" with no guard, so a bare `python -m src.shap_analysis`
rewrote 7 artifacts in place, one of them the git-tracked
results/shap_values_rf.csv.

Two traps this file exists to lock down.

1. **7 planned paths, not 4.** Three of the four filename templates are
   f-string interpolated over the model tag loop variable in
   run_shap_analysis, so each template is two files. The seventh,
   shap_waterfall_top_gs.png, is written by the _shap_gold_standard_waterfall
   delegate rather than by run_shap_analysis itself. The equivalent
   derived-filename trap has already bitten this line twice.
2. **--results-dir must NOT become required here.** On this module it is an
   input flag: it is only ever read from (*_features.csv, *_ranked.csv). Every
   sibling module in this defect-class line made its --results-dir required
   because there it was the output destination, so a future maintainer
   pattern-matching across the family could plausibly "fix" this one wrongly.

Import note: this file installs a stub `shap` module into sys.modules before
importing src.shap_analysis, but only when the real one has not already been
imported. Nothing under test here touches SHAP itself - the guard and its
enumeration are pure path arithmetic - and on at least one Windows dev machine
in this project `import shap` hard-crashes the interpreter natively (Windows
fatal exception 0xc06d007f inside scipy.linalg.inv, called from
shap/plots/colors/_colorconv.py at import time), which would otherwise take the
whole pytest process down at collection. On a machine where shap imports
cleanly, tests/test_run_analysis_results_guard.py is collected first
(alphabetically) and has already imported the real library, so the stub is not
installed and these tests run against the real import path.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

if "shap" not in sys.modules:  # pragma: no cover - depends on collection order
    # src/shap_analysis.py touches no shap attribute at import time, so an empty
    # module object is enough to satisfy its `import shap`.
    sys.modules["shap"] = types.ModuleType("shap")

from src import shap_analysis as sa  # noqa: E402
from src.artifact_integrity import ArtifactIntegrityError  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_NAMES = [
    "shap_values_rf.csv",
    "shap_summary_rf.png",
    "shap_bar_rf.png",
    "shap_values_xgb.csv",
    "shap_summary_xgb.png",
    "shap_bar_xgb.png",
    "shap_waterfall_top_gs.png",
]


# ---------------------------------------------------------------------------
# Planned-path enumeration
# ---------------------------------------------------------------------------


def test_planned_paths_lists_all_seven_artifacts():
    paths = sa.planned_shap_paths("out")
    assert [Path(p).name for p in paths] == EXPECTED_NAMES


def test_planned_paths_cover_both_model_tags_for_every_template():
    """The f-string-over-a-loop-variable trap: one template, two files."""
    names = {Path(p).name for p in sa.planned_shap_paths("out")}
    for template in ["shap_values_{tag}.csv", "shap_summary_{tag}.png", "shap_bar_{tag}.png"]:
        for tag in sa.SHAP_MODEL_TAGS:
            assert template.format(tag=tag) in names


def test_planned_paths_include_the_delegate_write():
    """shap_waterfall_top_gs.png is written by _shap_gold_standard_waterfall,
    not by run_shap_analysis, so a guard scoped to the visible loop misses it."""
    names = {Path(p).name for p in sa.planned_shap_paths("out")}
    assert "shap_waterfall_top_gs.png" in names


def test_planned_paths_are_scoped_to_the_given_output_dir(tmp_path):
    for p in sa.planned_shap_paths(str(tmp_path)):
        assert Path(p).parent == tmp_path


def test_planned_paths_are_unique():
    paths = sa.planned_shap_paths("out")
    assert len(paths) == len(set(paths))


def test_planned_paths_exclude_the_frozen_shareout_snapshot():
    """results/shareout_20260426/ holds tracked copies of three of these
    filenames, but this module never writes into that directory - only
    scripts/regenerate_shareout_pngs.py copies files there. Enumerating them
    would block every run on artifacts that are not at risk."""
    for p in sa.planned_shap_paths("results"):
        assert "shareout" not in p


# ---------------------------------------------------------------------------
# Guard behaviour
# ---------------------------------------------------------------------------


def test_guard_passes_on_an_empty_directory(tmp_path):
    sa._guard_output_dir(str(tmp_path), allow_overwrite=False)


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_guard_refuses_to_clobber_each_artifact(tmp_path, name):
    (tmp_path / name).write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        sa._guard_output_dir(str(tmp_path), allow_overwrite=False)
    assert name in str(exc.value)


def test_guard_names_the_blocking_file_and_the_escape_hatch(tmp_path):
    (tmp_path / "shap_values_rf.csv").write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        sa._guard_output_dir(str(tmp_path), allow_overwrite=False)
    message = str(exc.value)
    assert "shap_values_rf.csv" in message
    assert "--allow-overwrite" in message
    assert "run_shap_analysis(..., allow_overwrite=True)" in message


def test_guard_reports_every_colliding_file_not_just_the_first(tmp_path):
    for name in EXPECTED_NAMES:
        (tmp_path / name).write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        sa._guard_output_dir(str(tmp_path), allow_overwrite=False)
    message = str(exc.value)
    for name in EXPECTED_NAMES:
        assert name in message
    assert "7 existing artifact(s)" in message


def test_allow_overwrite_disarms_the_guard(tmp_path):
    for name in EXPECTED_NAMES:
        (tmp_path / name).write_text("existing", encoding="utf-8")
    sa._guard_output_dir(str(tmp_path), allow_overwrite=True)


# ---------------------------------------------------------------------------
# Wiring: the guard must actually run inside run_shap_analysis
# ---------------------------------------------------------------------------


def test_guard_is_wired_into_run_shap_analysis(tmp_path):
    """A defined-but-uncalled guard would pass every test above.

    model_dir points at a directory with no model files, so without the guard
    the run would fail in load_verified_joblib with ArtifactIntegrityError.
    Getting FileExistsError instead proves the guard runs before any load.
    """
    (tmp_path / "shap_values_rf.csv").write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError):
        sa.run_shap_analysis(
            results_dir=str(tmp_path / "no_such_inputs"),
            output_dir=str(tmp_path),
            model_dir=str(tmp_path / "no_such_models"),
        )


def test_allow_overwrite_passes_through_run_shap_analysis(tmp_path):
    """With the escape hatch set, the guard must not be what stops the run."""
    (tmp_path / "shap_values_rf.csv").write_text("published", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        sa.run_shap_analysis(
            results_dir=str(tmp_path / "no_such_inputs"),
            output_dir=str(tmp_path),
            model_dir=str(tmp_path / "no_such_models"),
            allow_overwrite=True,
        )


def test_run_shap_analysis_requires_an_explicit_output_dir():
    """No default destination: omitting output_dir is a TypeError."""
    with pytest.raises(TypeError):
        sa.run_shap_analysis("results")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# --results-dir is an INPUT flag and must stay optional
# ---------------------------------------------------------------------------


def test_results_dir_stays_an_optional_input_flag_in_the_source():
    """Regression lock: --results-dir on this module is read-only input.

    Every sibling module in this defect-class line made its --results-dir
    required because there the flag named the output destination. Here it names
    the pipeline feature/ranked CSVs the run reads, so requiring it would be a
    pattern-match error rather than a fix, and would break the two in-repo
    callers that rely on the default.
    """
    source = (REPO_ROOT / "src" / "shap_analysis.py").read_text(encoding="utf-8")
    start = source.index('"--results-dir"')
    block = source[start : source.index("    )", start)]
    assert 'default="results"' in block
    assert "required=True" not in block


# ---------------------------------------------------------------------------
# In-repo callers: the parameter reorder is silent-failure-shaped
# ---------------------------------------------------------------------------


def test_run_analysis_calls_shap_by_keyword_and_threads_allow_overwrite():
    """output_dir moved ahead of model_dir, so the old third positional
    argument now lands on a different parameter - a silent failure, not a loud
    one. scripts/run_analysis.py must pass by keyword.

    It must also forward allow_overwrite: its own guard already covers all 7 of
    these filenames, and it writes gold_standard_validation.csv and
    baseline_comparison.csv before reaching this call. Without the forward, an
    --allow-overwrite rerun would do that work, write both CSVs, and only then
    abort on SHAP's guard - partial and destructive.
    """
    source = (REPO_ROOT / "scripts" / "run_analysis.py").read_text(encoding="utf-8")
    call_start = source.index("    run_shap_analysis(")
    call_block = source[call_start : source.index("    )", call_start)]
    assert "results_dir=results_dir," in call_block
    assert "output_dir=results_dir," in call_block
    assert "model_dir=model_dir," in call_block
    assert "allow_overwrite=allow_overwrite," in call_block


def test_regenerate_shareout_calls_shap_by_keyword_with_allow_overwrite():
    """Regenerating these PNGs into results/ is that script's declared purpose,
    so it is one of the few callers that legitimately disarms the guard."""
    source = (REPO_ROOT / "scripts" / "regenerate_shareout_pngs.py").read_text(encoding="utf-8")
    call_start = source.index("    run_shap_analysis(")
    call_block = source[call_start : source.index("    )", call_start)]
    assert "results_dir=" in call_block
    assert "output_dir=" in call_block
    assert "allow_overwrite=True," in call_block


def test_run_analysis_planned_paths_still_match_this_module():
    """scripts/run_analysis.py enumerates SHAP's 7 filenames in its own guard.

    The two lists are maintained separately, so they can drift. Reading
    run_analysis.py's source rather than importing it keeps this check
    available on machines where importing it crashes on the shap import.
    """
    source = (REPO_ROOT / "scripts" / "run_analysis.py").read_text(encoding="utf-8")
    for name in EXPECTED_NAMES:
        assert f'"{name}"' in source, (
            f"scripts/run_analysis.py no longer plans {name}, which it still writes "
            "by delegating to run_shap_analysis"
        )


# ---------------------------------------------------------------------------
# CLI contract
#
# The subprocess-level CLI checks for this module (--help parses, --output-dir
# is rejected when omitted, --allow-overwrite is advertised) live in
# tests/test_entry_point_help_smoke.py, which registers src.shap_analysis in
# OUTPUT_DIR_REQUIRED_ENTRY_POINTS. They are not duplicated here: spawning
# `python -m src.shap_analysis` imports the real shap library, which crashes
# the interpreter natively on at least one Windows dev machine in this project,
# so those cases are CI-authoritative and are kept in one place.
# ---------------------------------------------------------------------------


def test_cli_actually_threads_allow_overwrite_into_run_shap_analysis():
    """Advertising the flag is not the same as wiring it.

    An --allow-overwrite that argparse accepts but __main__ never forwards is a
    silent no-op: --help looks right, the flag is accepted, and the guard still
    aborts the run it was meant to permit.
    """
    source = (REPO_ROOT / "src" / "shap_analysis.py").read_text(encoding="utf-8")
    call_start = source.index("    run_shap_analysis(")
    call_block = source[call_start : source.index("    )", call_start)]
    assert "allow_overwrite=args.allow_overwrite," in call_block, (
        "__main__ parses --allow-overwrite but never forwards it to run_shap_analysis"
    )
