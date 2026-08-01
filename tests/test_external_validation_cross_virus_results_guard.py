"""Overwrite-guard tests for src/external_validation_cross_virus.py's --output.

Tier-1 instance #9 of the results/ silent-overwrite defect class (see
_local/notes/results-dir-tier1-enumeration-2026-07-30.md). --output defaulted
to "results/external_validation_cross_virus.csv" with no guard, so a bare
`python -m src.external_validation_cross_virus` could silently rewrite the
tracked cross-virus transfer table in place.

--output is now required with no default at both the CLI and Python-API
layers. Unlike the --results-dir-shaped guards elsewhere in this line,
--output names a file, not a directory: planned_cross_virus_paths() and
_guard_output_path() use the scope/remedy override on guard_planned_paths()
(same shape as src/data_bias_audit.py's single-file guards) so the error
message does not claim a directory-shaped destination that does not exist.

The only other caller of run_cross_virus in the repo is
src/external_validation_finalize.py (around line 611-623), which already
calls it with output_path passed explicitly as a positional argument, so
removing the default does not break that caller - confirmed by grep before
this change and locked down by test_finalize_calls_cross_virus_with_an_
explicit_output_path below.

Mirrors tests/test_h2_tier_a_results_guard.py's structure: planned-path
enumeration, guard behaviour in isolation, a wiring test that proves the guard
actually runs inside run_cross_virus rather than merely being defined, an
allow-overwrite-passthrough test, and CLI-level checks anchored on argparse's
required-arguments line rather than a bare stderr substring - the guard's own
FileExistsError message names --output too and would give a false pass on a
substring check while a regression was live.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src import external_validation_cross_virus as evcv

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Planned-path enumeration
# ---------------------------------------------------------------------------


def test_planned_paths_lists_the_single_tracked_artifact(tmp_path):
    output_path = str(tmp_path / "external_validation_cross_virus.csv")
    paths = evcv.planned_cross_virus_paths(output_path)
    assert [Path(p).name for p in paths] == ["external_validation_cross_virus.csv"]


def test_planned_paths_resolve_to_the_given_output_path(tmp_path):
    output_path = str(tmp_path / "external_validation_cross_virus.csv")
    paths = evcv.planned_cross_virus_paths(output_path)
    assert paths == [output_path]


def test_planned_paths_handle_a_bare_filename_with_no_directory():
    import os

    paths = evcv.planned_cross_virus_paths("external_validation_cross_virus.csv")
    assert paths == [os.path.join(".", "external_validation_cross_virus.csv")]


# ---------------------------------------------------------------------------
# Guard behaviour
# ---------------------------------------------------------------------------


def test_guard_passes_when_the_output_path_does_not_exist(tmp_path):
    output_path = str(tmp_path / "external_validation_cross_virus.csv")
    evcv._guard_output_path(output_path, allow_overwrite=False)


def test_guard_refuses_to_clobber_the_existing_output_file(tmp_path):
    output_path = tmp_path / "external_validation_cross_virus.csv"
    output_path.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        evcv._guard_output_path(str(output_path), allow_overwrite=False)
    assert "external_validation_cross_virus.csv" in str(exc.value)


def test_guard_names_the_blocking_file_and_the_escape_hatch(tmp_path):
    output_path = tmp_path / "external_validation_cross_virus.csv"
    output_path.write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        evcv._guard_output_path(str(output_path), allow_overwrite=False)
    message = str(exc.value)
    assert "external_validation_cross_virus.csv" in message
    assert "--allow-overwrite" in message


def test_guard_does_not_claim_a_directory_shaped_destination(tmp_path):
    """--output names a file, not a directory: the message must not read
    "Point --output at a fresh directory" - that would be wrong advice."""
    output_path = tmp_path / "external_validation_cross_virus.csv"
    output_path.write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        evcv._guard_output_path(str(output_path), allow_overwrite=False)
    message = str(exc.value)
    assert "fresh directory" not in message
    assert "Point --output at a fresh path" in message


def test_allow_overwrite_disarms_the_guard(tmp_path):
    output_path = tmp_path / "external_validation_cross_virus.csv"
    output_path.write_text("existing", encoding="utf-8")
    evcv._guard_output_path(str(output_path), allow_overwrite=True)


# ---------------------------------------------------------------------------
# Wiring: the guard must actually run inside run_cross_virus, before any read
# ---------------------------------------------------------------------------


def test_guard_is_wired_into_run_cross_virus(tmp_path):
    """A defined-but-uncalled guard would pass every test above.

    data_path points at a file that does not exist, so if the guard did not
    run first, run_cross_virus would fail with FileNotFoundError while trying
    to read data_path instead of FileExistsError from the guard. Getting
    FileExistsError here proves the guard runs before any read.
    """
    output_path = tmp_path / "external_validation_cross_virus.csv"
    output_path.write_text("published", encoding="utf-8")
    bogus_data_path = str(tmp_path / "does_not_exist.csv")

    with pytest.raises(FileExistsError):
        evcv.run_cross_virus(
            bogus_data_path,
            "models/peptide_binding_matrix_v4.csv",
            str(output_path),
            allow_overwrite=False,
        )


def test_allow_overwrite_passes_through_run_cross_virus(tmp_path):
    """With the escape hatch set, the guard must not be what stops the run."""
    output_path = tmp_path / "external_validation_cross_virus.csv"
    output_path.write_text("published", encoding="utf-8")
    bogus_data_path = str(tmp_path / "does_not_exist.csv")

    with pytest.raises(FileNotFoundError):
        evcv.run_cross_virus(
            bogus_data_path,
            "models/peptide_binding_matrix_v4.csv",
            str(output_path),
            allow_overwrite=True,
        )


def test_run_cross_virus_requires_an_explicit_output_path():
    """No default destination: omitting output_path is a TypeError."""
    with pytest.raises(TypeError):
        evcv.run_cross_virus(  # type: ignore[call-arg]
            "data/immunogenicity_dataset_v4.csv",
            "models/peptide_binding_matrix_v4.csv",
        )


# ---------------------------------------------------------------------------
# The external_validation_finalize.py caller
# ---------------------------------------------------------------------------


def test_finalize_calls_cross_virus_with_an_explicit_output_path():
    """external_validation_finalize.py must keep passing output_path
    positionally, or this required-argument change breaks it. It already did
    at the time this guard was added (verified by grep, not just this note),
    so lock the call shape down here.
    """
    source = (REPO_ROOT / "src" / "external_validation_finalize.py").read_text(encoding="utf-8")
    call_start = source.index("_run_cross_virus(")
    call_block = source[call_start : source.index(")", call_start) + 1]
    assert "data/immunogenicity_dataset_v4.csv" in call_block
    assert "models/peptide_binding_matrix_v4.csv" in call_block
    assert "external_validation_cross_virus.csv" in call_block


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------

REQUIRED_ARGS_MARKER = "the following arguments are required:"


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "src.external_validation_cross_virus", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_cli_requires_output_explicitly():
    result = _run_module()
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


def test_cli_actually_threads_allow_overwrite_into_run_cross_virus():
    """Advertising the flag is not the same as wiring it.

    An --allow-overwrite that argparse accepts but __main__ never forwards is a
    silent no-op: --help looks right, the flag is accepted, and the guard
    still aborts the run it was meant to permit. Checking the source of the
    __main__ call is cheap; the alternative is executing the real training
    pipeline.
    """
    source = (REPO_ROOT / "src" / "external_validation_cross_virus.py").read_text(
        encoding="utf-8"
    )
    call_start = source.index("    run_cross_virus(")
    call_block = source[call_start : source.index("\n\n", call_start)]
    assert "allow_overwrite=args.allow_overwrite," in call_block, (
        "__main__ parses --allow-overwrite but never forwards it to run_cross_virus"
    )
