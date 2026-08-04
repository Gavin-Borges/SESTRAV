"""Overwrite-guard tests for scripts/compute_population_coverage.py's --output.

Tier-1 instance #11 of the results/ silent-overwrite defect class (see
_local/notes/results-dir-tier1-enumeration-2026-07-30.md). --output defaulted to
"results/population_coverage_v5.json" with no guard, so a bare
`python scripts/compute_population_coverage.py` silently rewrote that
git-tracked artifact in place.

--output is now required with no default. Like
src/external_validation_cross_virus.py's flag of the same name it names a file,
not a directory, so the guard uses the scope/remedy override on
guard_planned_paths() and reuses that module's wording, rather than telling the
caller to point a file-path flag at a fresh directory.

The load-bearing detail specific to this module: main() resolves a relative
--output against PROJECT_ROOT, not the working directory, so the guard has to
run on the resolved path. Guarding args.output directly would check the working
directory while the write landed somewhere else - the guard would pass and the
tracked artifact would still be replaced. That is locked down below.

Mirrors tests/test_h2_tier_a_results_guard.py's structure: planned-path
enumeration, guard behaviour in isolation, a wiring test proving the guard
actually runs inside main() before any work, an allow-overwrite-passthrough
test, and CLI-level checks anchored on argparse's required-arguments line
rather than a bare stderr substring - the guard's own FileExistsError message
names --output too and would give a false pass on a substring check while a
regression was live.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import compute_population_coverage as cpc

REPO_ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_NAME = "population_coverage_v5.json"


# ---------------------------------------------------------------------------
# Planned-path enumeration
# ---------------------------------------------------------------------------


def test_planned_paths_lists_the_single_artifact(tmp_path):
    output_path = tmp_path / ARTIFACT_NAME
    paths = cpc.planned_population_coverage_paths(output_path)
    assert [Path(p).name for p in paths] == [ARTIFACT_NAME]


def test_planned_paths_resolve_to_the_given_output_path(tmp_path):
    output_path = tmp_path / ARTIFACT_NAME
    paths = cpc.planned_population_coverage_paths(output_path)
    assert [Path(p) for p in paths] == [output_path]


def test_planned_paths_handle_a_bare_filename_with_no_directory():
    paths = cpc.planned_population_coverage_paths(ARTIFACT_NAME)
    assert [Path(p) for p in paths] == [Path(".") / ARTIFACT_NAME]


# ---------------------------------------------------------------------------
# Guard behaviour
# ---------------------------------------------------------------------------


def test_guard_passes_when_the_output_path_does_not_exist(tmp_path):
    cpc._guard_output_path(tmp_path / ARTIFACT_NAME, allow_overwrite=False)


def test_guard_refuses_to_clobber_the_existing_output_file(tmp_path):
    output_path = tmp_path / ARTIFACT_NAME
    output_path.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        cpc._guard_output_path(output_path, allow_overwrite=False)
    assert ARTIFACT_NAME in str(exc.value)


def test_guard_names_the_blocking_file_and_the_escape_hatch(tmp_path):
    output_path = tmp_path / ARTIFACT_NAME
    output_path.write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        cpc._guard_output_path(output_path, allow_overwrite=False)
    message = str(exc.value)
    assert ARTIFACT_NAME in message
    assert "--allow-overwrite" in message


def test_guard_does_not_claim_a_directory_shaped_destination(tmp_path):
    """--output names a file: "Point --output at a fresh directory" would be
    wrong advice, so scope/remedy replace those clauses."""
    output_path = tmp_path / ARTIFACT_NAME
    output_path.write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        cpc._guard_output_path(output_path, allow_overwrite=False)
    message = str(exc.value)
    assert "fresh directory" not in message
    assert "Point --output at a fresh path" in message


def test_allow_overwrite_disarms_the_guard(tmp_path):
    output_path = tmp_path / ARTIFACT_NAME
    output_path.write_text("existing", encoding="utf-8")
    cpc._guard_output_path(output_path, allow_overwrite=True)


# ---------------------------------------------------------------------------
# Wiring: the guard must actually run inside main(), before any work
# ---------------------------------------------------------------------------


def test_guard_is_wired_into_main(tmp_path, monkeypatch):
    """A defined-but-uncalled guard would pass every test above.

    Unlike the other modules in this line there is no input file to point
    somewhere bogus - the coverage numbers are computed from module constants -
    so the work itself is booby-trapped instead. compute_all_coverage must
    never run once the guard has something to refuse.
    """
    output_path = tmp_path / ARTIFACT_NAME
    output_path.write_text("published", encoding="utf-8")

    def _should_not_run(*args, **kwargs):
        raise AssertionError("main() did work before the guard rejected the run")

    monkeypatch.setattr(cpc, "compute_all_coverage", _should_not_run)

    with pytest.raises(FileExistsError):
        cpc.main(["--output", str(output_path)])

    assert output_path.read_text(encoding="utf-8") == "published"


def test_allow_overwrite_passes_through_main(tmp_path):
    """With the escape hatch set, the guard must not be what stops the run."""
    output_path = tmp_path / ARTIFACT_NAME
    output_path.write_text("published", encoding="utf-8")

    rc = cpc.main(["--output", str(output_path), "--no-markdown", "--allow-overwrite"])

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "panel_coverage" in payload


def test_a_clean_destination_still_runs_end_to_end(tmp_path):
    output_path = tmp_path / ARTIFACT_NAME
    rc = cpc.main(["--output", str(output_path), "--no-markdown"])
    assert rc == 0
    assert output_path.is_file()


# ---------------------------------------------------------------------------
# The PROJECT_ROOT resolution: the guard must check where the write lands
# ---------------------------------------------------------------------------


def test_guard_checks_the_project_root_resolved_path_not_the_raw_flag(tmp_path, monkeypatch):
    """A relative --output is rebased onto PROJECT_ROOT before the write.

    Guarding args.output would resolve against the working directory instead,
    so the guard would find nothing while the write replaced the real artifact.
    Here PROJECT_ROOT is redirected at tmp_path, the collision is planted under
    it, and the working directory deliberately does not contain the file.
    """
    monkeypatch.setattr(cpc, "PROJECT_ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    planted = tmp_path / "results" / ARTIFACT_NAME
    planted.write_text("published", encoding="utf-8")

    workdir = tmp_path / "elsewhere"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    with pytest.raises(FileExistsError) as exc:
        cpc.main(["--output", f"results/{ARTIFACT_NAME}", "--no-markdown"])

    assert str(planted) in str(exc.value)
    assert planted.read_text(encoding="utf-8") == "published"


def test_an_absolute_output_path_is_left_alone(tmp_path, monkeypatch):
    """Only relative paths are rebased; an absolute --output is guarded as given."""
    monkeypatch.setattr(cpc, "PROJECT_ROOT", tmp_path / "not_here")
    output_path = tmp_path / ARTIFACT_NAME
    output_path.write_text("published", encoding="utf-8")

    with pytest.raises(FileExistsError) as exc:
        cpc.main(["--output", str(output_path), "--no-markdown"])

    assert str(output_path) in str(exc.value)


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------

REQUIRED_ARGS_MARKER = "the following arguments are required:"


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.compute_population_coverage", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_cli_requires_output_explicitly():
    result = _run_module("--no-markdown")
    assert result.returncode != 0
    required_lines = [ln for ln in result.stderr.splitlines() if REQUIRED_ARGS_MARKER in ln]
    assert required_lines, (
        f"expected argparse to reject the run outright, got: {result.stderr[:400]}"
    )
    assert "--output" in required_lines[0], (
        f"--output was not named as required; it may have regained a default: "
        f"{required_lines[0][:200]}"
    )


def test_cli_advertises_allow_overwrite():
    result = _run_module("--help")
    assert result.returncode == 0
    assert "--allow-overwrite" in result.stdout
