"""Overwrite-guard tests for src/prepare_external_validation_inputs.py's --results-dir.

Tier-1 instance #2 of the results/ silent-overwrite defect class (see
_local/notes/results-dir-tier1-enumeration-2026-07-30.md). --results-dir used
to fall through a config-indirection default (`cfg.get("output_dir",
"results")`) with no guard, so a bare invocation could silently rewrite the 4
tracked external-validation sidecar artifacts in place. The fix removed the
config-indirection default entirely: --results-dir is now required at both the
CLI and Python-API layers, and run_prepare() enumerates and guards all 4 paths
it writes before any read (of config.yaml or the data/models inputs) happens.

Mirrors tests/test_h2_tier_a_results_guard.py's structure: planned-path
enumeration, guard behaviour in isolation, a wiring test that proves the guard
actually runs inside run_prepare rather than merely being defined, an
allow-overwrite-passthrough test, and CLI-level checks anchored on argparse's
required-arguments line rather than a bare stderr substring - the guard's own
FileExistsError message names --results-dir too and would give a false pass on
a substring check while a regression was live.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src import prepare_external_validation_inputs as pev

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_NAMES = [
    "external_validation_input.csv",
    "external_predig_peptide_allele_pairs.csv",
    "external_prime_peptides.txt",
    "external_prime_alleles_compact.txt",
]


# ---------------------------------------------------------------------------
# Planned-path enumeration
# ---------------------------------------------------------------------------


def test_planned_paths_lists_all_four_artifacts():
    paths = pev.planned_prepare_paths("out")
    assert [Path(p).name for p in paths] == EXPECTED_NAMES


def test_planned_paths_are_scoped_to_the_given_results_dir(tmp_path):
    paths = pev.planned_prepare_paths(str(tmp_path))
    for p in paths:
        assert Path(p).parent == tmp_path


# ---------------------------------------------------------------------------
# Guard behaviour
# ---------------------------------------------------------------------------


def test_guard_passes_on_an_empty_directory(tmp_path):
    pev._guard_results_dir(str(tmp_path), allow_overwrite=False)


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_guard_refuses_to_clobber_each_artifact(tmp_path, name):
    (tmp_path / name).write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        pev._guard_results_dir(str(tmp_path), allow_overwrite=False)
    assert name in str(exc.value)


def test_guard_names_the_blocking_file_and_the_escape_hatch(tmp_path):
    (tmp_path / "external_validation_input.csv").write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        pev._guard_results_dir(str(tmp_path), allow_overwrite=False)
    message = str(exc.value)
    assert "external_validation_input.csv" in message
    assert "--allow-overwrite" in message


def test_guard_reports_every_colliding_file_not_just_the_first(tmp_path):
    for name in EXPECTED_NAMES:
        (tmp_path / name).write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        pev._guard_results_dir(str(tmp_path), allow_overwrite=False)
    message = str(exc.value)
    for name in EXPECTED_NAMES:
        assert name in message
    assert "4 existing artifact(s)" in message


def test_allow_overwrite_disarms_the_guard(tmp_path):
    for name in EXPECTED_NAMES:
        (tmp_path / name).write_text("existing", encoding="utf-8")
    pev._guard_results_dir(str(tmp_path), allow_overwrite=True)


# ---------------------------------------------------------------------------
# Wiring: the guard must actually run inside run_prepare, before any read
# ---------------------------------------------------------------------------


def test_guard_is_wired_into_run_prepare(tmp_path):
    """A defined-but-uncalled guard would pass every test above.

    root points at a directory with no config.yaml, so if the guard did not
    run first, run_prepare would fail with FileNotFoundError while trying to
    open config.yaml instead of FileExistsError from the guard. Getting
    FileExistsError here proves the guard runs before any read.
    """
    (tmp_path / "external_validation_input.csv").write_text("published", encoding="utf-8")
    bogus_root = str(tmp_path / "does_not_exist_as_a_repo_root")

    with pytest.raises(FileExistsError):
        pev.run_prepare(bogus_root, str(tmp_path), allow_overwrite=False)


def test_allow_overwrite_passes_through_run_prepare(tmp_path):
    """With the escape hatch set, the guard must not be what stops the run."""
    (tmp_path / "external_validation_input.csv").write_text("published", encoding="utf-8")
    bogus_root = str(tmp_path / "does_not_exist_as_a_repo_root")

    with pytest.raises(FileNotFoundError):
        pev.run_prepare(bogus_root, str(tmp_path), allow_overwrite=True)


def test_run_prepare_requires_an_explicit_results_dir():
    """No default destination: omitting results_dir is a TypeError."""
    with pytest.raises(TypeError):
        pev.run_prepare(".")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------

REQUIRED_ARGS_MARKER = "the following arguments are required:"


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "src.prepare_external_validation_inputs", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_cli_requires_results_dir_explicitly():
    result = _run_module()
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


def test_cli_actually_threads_allow_overwrite_into_run_prepare():
    """Advertising the flag is not the same as wiring it.

    An --allow-overwrite that argparse accepts but __main__ never forwards is a
    silent no-op: --help looks right, the flag is accepted, and the guard
    still aborts the run it was meant to permit. Checking the source of the
    __main__ call is cheap; the alternative is executing the real pipeline.
    """
    source = (REPO_ROOT / "src" / "prepare_external_validation_inputs.py").read_text(
        encoding="utf-8"
    )
    call_start = source.index("    run_prepare(")
    call_block = source[call_start : source.index("\n\n", call_start)]
    assert "allow_overwrite=args.allow_overwrite" in call_block, (
        "__main__ parses --allow-overwrite but never forwards it to run_prepare"
    )
