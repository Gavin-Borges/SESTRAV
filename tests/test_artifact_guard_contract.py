"""One contract every guarded entry point must satisfy.

This is the reason `src/artifact_guard.py` exists. Before it, each guarded
module carried its own copy of the guard and its own bespoke tests, and those
test files were not equivalent to one another. That gap was not theoretical: a
dedicated 16-test file passed in full against a live regression it existed to
catch, because its assertions searched all of stderr for a flag name that the
guard's own error message also contains.

The registry below is the enforcement point. Adding a module to it subjects that
module to every check in this file at once, so a new guarded entry point cannot
quietly be held to a weaker standard than its siblings. Per-module files remain
the right home for module-specific facts (which filenames are planned, which
delegate writes exist, which callers must keep passing a temp directory); this
file only covers the properties that must hold identically everywhere.

Two kinds of absence from the registry, both deliberate.

`scripts/run_analysis.py` DOES delegate to the shared helper but cannot be
registered: importing it pulls in `src/shap_analysis.py` -> `shap`, which
hard-crashes the interpreter on some Windows machines, and a module-scope import
here would take the whole file down with it. Its guard is the same shared
implementation, exercised through `tests/test_run_analysis_results_guard.py`,
which CI runs with no `--ignore`.

`src/train_classifier.py` and `src/ann_benchmark.py` now delegate too: both
matched the shared three-piece shape exactly once `remedy` carried their
"(for example models/scratch/<run-name>)" clause, verified byte-identical
against their pre-refactor messages.

Four other modules still carry their own copy of the guard and do not delegate
here at all, so this contract does not speak for them. Two are genuine
three-piece variants blocked on the template growing further:
`src/train_gnn.py` omits the `under '{output_dir}'` scope clause entirely and
appends a trailing sentence about OOF predictions the shared message has no
slot for; `src/gnn_benchmark.py` also omits the scope clause, calls its
artifacts "result CSV(s)" instead of "artifact(s)" (the noun is not a shared
parameter), and omits the Python-API-hint parenthesis altogether. Folding
either in by editing its message to fit would not be behaviour-preserving.
The remaining two, `src/ablation_study.py` and
`scripts/compute_ann_baseline_summary.py`, are single-path variants whose
message omits the count and the listing. See `src/artifact_guard.py`'s module
docstring for more. This file covers the delegating modules; it is not a
sweep of everything that guards an artifact.
"""

from __future__ import annotations

import os

import pytest

from src import ann_benchmark as ab
from src import bias_skew_finalization as bsf
from src import final_validation_report as fvr
from src import h2_tier_a_evaluation as h2
from src import train_classifier as tc
from src.artifact_guard import guard_planned_paths, planned_paths_under

# planned_paths(dir), guard(dir, allow_overwrite), flag, api_hint. The pytest id
# is passed last as a keyword. api_hint is the full parenthesised string the
# message must contain, not a prefix, so a hint mutated into a different-but-
# similarly-prefixed call still fails.
GUARDED_MODULES = [
    pytest.param(
        lambda d: fvr.planned_validation_paths(d, "expansion_v4", "4.0.0"),
        lambda d, allow: fvr._guard_results_dir(d, "expansion_v4", "4.0.0", allow),
        "--results-dir",
        "run_final_validation(..., allow_overwrite=True)",
        id="final_validation_report",
    ),
    pytest.param(
        bsf.planned_bias_skew_paths,
        bsf._guard_results_dir,
        "--results-dir",
        "run_bias_skew_finalization(..., allow_overwrite=True)",
        id="bias_skew_finalization",
    ),
    pytest.param(
        h2.planned_h2_tier_a_paths,
        h2._guard_output_dir,
        "--output-dir",
        "run_h2_tier_a(..., allow_overwrite=True)",
        id="h2_tier_a_evaluation",
    ),
    pytest.param(
        lambda d: tc.planned_artifact_paths(d, 21),
        lambda d, allow: tc._guard_output_dir(d, 21, allow),
        "--model-dir",
        "train_models(..., allow_overwrite=True)",
        id="train_classifier",
    ),
    pytest.param(
        lambda d: ab.planned_ann_artifact_paths(d, 21, False),
        lambda d, allow: ab._guard_output_dir(d, 21, False, allow),
        "--model-dir",
        "train_ann(..., allow_overwrite=True)",
        id="ann_benchmark",
    ),
]


@pytest.mark.parametrize(("planned", "guard", "flag", "api_hint"), GUARDED_MODULES)
def test_planned_paths_are_all_under_the_given_directory(planned, guard, flag, api_hint, tmp_path):
    """A planned path escaping the output directory would defeat redirection."""
    paths = planned(str(tmp_path))
    assert paths, "module planned no paths at all"
    for path in paths:
        assert str(path).startswith(str(tmp_path)), (
            f"{path} is not under the requested output directory, so pointing "
            f"{flag} somewhere safe would not redirect it"
        )


@pytest.mark.parametrize(("planned", "guard", "flag", "api_hint"), GUARDED_MODULES)
def test_planned_paths_are_unique(planned, guard, flag, api_hint, tmp_path):
    """A duplicated planned path inflates the count reported to the user."""
    paths = planned(str(tmp_path))
    assert len(paths) == len(set(paths)), f"duplicate entries in planned paths: {paths}"


@pytest.mark.parametrize(("planned", "guard", "flag", "api_hint"), GUARDED_MODULES)
def test_guard_is_silent_on_an_empty_directory(planned, guard, flag, api_hint, tmp_path):
    """A first-ever run into a fresh directory must not be blocked."""
    guard(str(tmp_path), False)


@pytest.mark.parametrize(("planned", "guard", "flag", "api_hint"), GUARDED_MODULES)
def test_guard_refuses_when_any_single_planned_artifact_exists(
    planned, guard, flag, api_hint, tmp_path
):
    """Every planned path must be load-bearing, not just the first one.

    Checked one file at a time rather than all at once: a guard that only
    inspected planned_paths[0] would pass a whole-directory check.
    """
    for path in planned(str(tmp_path)):
        target = tmp_path / os.path.basename(path)
        target.write_text("published", encoding="utf-8")
        with pytest.raises(FileExistsError) as exc:
            guard(str(tmp_path), False)
        assert target.name in str(exc.value), (
            f"{target.name} is planned but does not trigger the guard"
        )
        target.unlink()


@pytest.mark.parametrize(("planned", "guard", "flag", "api_hint"), GUARDED_MODULES)
def test_allow_overwrite_disarms_the_guard(planned, guard, flag, api_hint, tmp_path):
    """The escape hatch must work even when every planned artifact collides."""
    for path in planned(str(tmp_path)):
        (tmp_path / os.path.basename(path)).write_text("published", encoding="utf-8")
    guard(str(tmp_path), True)


@pytest.mark.parametrize(("planned", "guard", "flag", "api_hint"), GUARDED_MODULES)
def test_error_names_the_flag_the_escape_hatch_and_every_collision(
    planned, guard, flag, api_hint, tmp_path
):
    """The message has to be actionable, and must not under-report the damage."""
    names = []
    for path in planned(str(tmp_path)):
        name = os.path.basename(path)
        (tmp_path / name).write_text("published", encoding="utf-8")
        names.append(name)

    with pytest.raises(FileExistsError) as exc:
        guard(str(tmp_path), False)
    message = str(exc.value)

    assert flag in message, f"message does not say which flag to redirect: {message}"
    assert "--allow-overwrite" in message, "message does not document its own escape hatch"
    assert api_hint in message, "message does not name the Python-level escape hatch"
    assert f"{len(names)} existing artifact(s)" in message, (
        "message under-reports how many artifacts would be replaced"
    )
    for name in names:
        assert name in message, f"{name} would be overwritten but is not listed"


# ---------------------------------------------------------------------------
# The shared helper's own behaviour
# ---------------------------------------------------------------------------


def test_planned_paths_under_joins_every_name(tmp_path):
    paths = planned_paths_under(str(tmp_path), ["a.csv", "b.md"])
    assert [os.path.basename(p) for p in paths] == ["a.csv", "b.md"]


def test_guard_ignores_directories_that_share_an_artifact_name(tmp_path):
    """Only files count. A directory named like an artifact is not an overwrite."""
    (tmp_path / "report.md").mkdir()
    guard_planned_paths(
        str(tmp_path),
        planned_paths_under(str(tmp_path), ["report.md"]),
        False,
        flag="--results-dir",
        api_hint="run(..., allow_overwrite=True)",
    )


def test_guard_lists_collisions_in_sorted_order(tmp_path):
    """Stable ordering keeps the message diffable between runs."""
    names = ["c.csv", "a.csv", "b.csv"]
    for name in names:
        (tmp_path / name).write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        guard_planned_paths(
            str(tmp_path),
            planned_paths_under(str(tmp_path), names),
            False,
            flag="--results-dir",
            api_hint="run(..., allow_overwrite=True)",
        )
    message = str(exc.value)
    assert message.index("a.csv") < message.index("b.csv") < message.index("c.csv")


def test_detail_clause_is_interpolated_when_given(tmp_path):
    (tmp_path / "a.csv").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        guard_planned_paths(
            str(tmp_path),
            planned_paths_under(str(tmp_path), ["a.csv"]),
            False,
            flag="--output-dir",
            api_hint="run(..., allow_overwrite=True)",
            detail=": a.csv backs a published number",
        )
    assert "These may be published results: a.csv backs a published number." in str(exc.value)


def test_no_detail_clause_leaves_the_sentence_intact(tmp_path):
    (tmp_path / "a.csv").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        guard_planned_paths(
            str(tmp_path),
            planned_paths_under(str(tmp_path), ["a.csv"]),
            False,
            flag="--results-dir",
            api_hint="run(..., allow_overwrite=True)",
        )
    assert "These may be published results. Point --results-dir" in str(exc.value)
