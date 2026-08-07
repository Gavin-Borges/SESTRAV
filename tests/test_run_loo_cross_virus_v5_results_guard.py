"""Overwrite-guard tests for scripts/run_loo_cross_virus_v5.py's --output-json
and --output-csv.

Tier-1 instance #4 of the results/ silent-overwrite defect class (see
_local/notes/results-dir-tier1-enumeration-2026-07-30.md and the follow-up
_local/notes/tier1-mechanical-four-enumeration-2026-07-31.md:180-184, which
notes this module already exposed --output-json/--output-csv with the module
constants supplying only the defaults - i.e. the required=True-plus-guard
shape used throughout this line, not a wider repair). A bare
`python scripts/run_loo_cross_virus_v5.py` previously defaulted both flags to
results/loo_cross_virus_v5_clean.json and results/loo_cross_virus_v5_clean.csv
and wrote them with no guard - the Amendment 7 clean-test-partition artifact
that README.md and docs/claims_register.md (D11) cite for the corrected LOO
AUC-ROC numbers.

--output-json and --output-csv are now required with no default at both the
CLI and Python-API layers. Like src/data_bias_audit.py's guards, each names a
file, not a directory, so planned_loo_paths() and _guard_output_paths() use
the scope/remedy override on guard_planned_paths() rather than the default
"under '{output_dir}'" / "Point {flag} at a fresh directory" clauses, which
would be wrong advice for a flag that names a file.

Mirrors tests/test_h2_tier_a_results_guard.py's structure: planned-path
enumeration, guard behaviour in isolation, a wiring test that proves the guard
actually runs inside run_loo rather than merely being defined, an
allow-overwrite-passthrough test, and CLI-level checks anchored on argparse's
required-arguments line rather than a bare stderr substring - the guard's own
FileExistsError message names both flags too and would give a false pass on a
substring check while a regression was live.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_loo_cross_virus_v5 as loo_v5

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Planned-path enumeration
# ---------------------------------------------------------------------------


def test_planned_paths_lists_both_tracked_artifacts(tmp_path):
    output_json = str(tmp_path / "loo_cross_virus_v5_clean.json")
    output_csv = str(tmp_path / "loo_cross_virus_v5_clean.csv")
    paths = loo_v5.planned_loo_paths(output_json, output_csv)
    assert [Path(p).name for p in paths] == [
        "loo_cross_virus_v5_clean.json",
        "loo_cross_virus_v5_clean.csv",
    ]


def test_planned_paths_resolve_to_the_given_output_paths(tmp_path):
    output_json = str(tmp_path / "loo_cross_virus_v5_clean.json")
    output_csv = str(tmp_path / "loo_cross_virus_v5_clean.csv")
    paths = loo_v5.planned_loo_paths(output_json, output_csv)
    assert paths == [output_json, output_csv]


# ---------------------------------------------------------------------------
# Guard behaviour
# ---------------------------------------------------------------------------


def test_guard_passes_when_neither_output_path_exists(tmp_path):
    output_json = str(tmp_path / "loo_cross_virus_v5_clean.json")
    output_csv = str(tmp_path / "loo_cross_virus_v5_clean.csv")
    loo_v5._guard_output_paths(output_json, output_csv, allow_overwrite=False)


def test_guard_refuses_to_clobber_an_existing_json(tmp_path):
    output_json = tmp_path / "loo_cross_virus_v5_clean.json"
    output_csv = tmp_path / "loo_cross_virus_v5_clean.csv"
    output_json.write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        loo_v5._guard_output_paths(str(output_json), str(output_csv), allow_overwrite=False)
    assert "loo_cross_virus_v5_clean.json" in str(exc.value)


def test_guard_refuses_to_clobber_an_existing_csv(tmp_path):
    output_json = tmp_path / "loo_cross_virus_v5_clean.json"
    output_csv = tmp_path / "loo_cross_virus_v5_clean.csv"
    output_csv.write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        loo_v5._guard_output_paths(str(output_json), str(output_csv), allow_overwrite=False)
    assert "loo_cross_virus_v5_clean.csv" in str(exc.value)


def test_guard_names_both_blocking_files_and_the_escape_hatch(tmp_path):
    output_json = tmp_path / "loo_cross_virus_v5_clean.json"
    output_csv = tmp_path / "loo_cross_virus_v5_clean.csv"
    output_json.write_text("published", encoding="utf-8")
    output_csv.write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        loo_v5._guard_output_paths(str(output_json), str(output_csv), allow_overwrite=False)
    message = str(exc.value)
    assert "loo_cross_virus_v5_clean.json" in message
    assert "loo_cross_virus_v5_clean.csv" in message
    assert "--allow-overwrite" in message


def test_guard_does_not_claim_a_directory_shaped_destination(tmp_path):
    """--output-json/--output-csv name files, not a directory: the message
    must not read "Point --output-json/--output-csv at a fresh directory" -
    that would be wrong advice."""
    output_json = tmp_path / "loo_cross_virus_v5_clean.json"
    output_csv = tmp_path / "loo_cross_virus_v5_clean.csv"
    output_json.write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        loo_v5._guard_output_paths(str(output_json), str(output_csv), allow_overwrite=False)
    message = str(exc.value)
    assert "fresh directory" not in message
    assert "Point --output-json and --output-csv at fresh paths" in message


def test_allow_overwrite_disarms_the_guard(tmp_path):
    output_json = tmp_path / "loo_cross_virus_v5_clean.json"
    output_csv = tmp_path / "loo_cross_virus_v5_clean.csv"
    output_json.write_text("published", encoding="utf-8")
    output_csv.write_text("published", encoding="utf-8")
    loo_v5._guard_output_paths(str(output_json), str(output_csv), allow_overwrite=True)


# ---------------------------------------------------------------------------
# Wiring: the guard must actually run inside run_loo, before any read
# ---------------------------------------------------------------------------


def test_guard_is_wired_into_run_loo(tmp_path):
    """A defined-but-uncalled guard would pass every test above.

    dataset_path points at a file that does not exist, so if the guard did not
    run first, run_loo would fail with FileNotFoundError while trying to read
    dataset_path instead of FileExistsError from the guard. Getting
    FileExistsError here proves the guard runs before any read.
    """
    output_json = tmp_path / "loo_cross_virus_v5_clean.json"
    output_csv = tmp_path / "loo_cross_virus_v5_clean.csv"
    output_json.write_text("published", encoding="utf-8")
    bogus_dataset_path = str(tmp_path / "does_not_exist.csv")

    with pytest.raises(FileExistsError):
        loo_v5.run_loo(
            dataset_path=bogus_dataset_path,
            output_json=str(output_json),
            output_csv=str(output_csv),
            allow_overwrite=False,
        )


def test_allow_overwrite_passes_through_run_loo(tmp_path):
    """With the escape hatch set, the guard must not be what stops the run."""
    output_json = tmp_path / "loo_cross_virus_v5_clean.json"
    output_csv = tmp_path / "loo_cross_virus_v5_clean.csv"
    output_json.write_text("published", encoding="utf-8")
    bogus_dataset_path = str(tmp_path / "does_not_exist.csv")

    with pytest.raises(FileNotFoundError):
        loo_v5.run_loo(
            dataset_path=bogus_dataset_path,
            output_json=str(output_json),
            output_csv=str(output_csv),
            allow_overwrite=True,
        )


def test_run_loo_has_no_default_for_output_json_or_output_csv():
    """The Python-API layer must not silently fall back to the tracked paths.

    Omitting either kwarg has to fail loudly (TypeError, missing required
    keyword-only argument) rather than quietly defaulting to
    results/loo_cross_virus_v5_clean.{json,csv} - the exact defect this guard
    exists to close. Asserted two ways: the call itself, and the signature
    directly, so a future refactor that reintroduces a default fails this
    test even if it happens not to hit the omitted-kwarg call path.
    """
    import inspect

    with pytest.raises(TypeError):
        loo_v5.run_loo(dataset_path="unused.csv")  # missing output_json/output_csv

    sig = inspect.signature(loo_v5.run_loo)
    assert sig.parameters["output_json"].default is inspect.Parameter.empty
    assert sig.parameters["output_csv"].default is inspect.Parameter.empty


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------

REQUIRED_ARGS_MARKER = "the following arguments are required:"


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_loo_cross_virus_v5", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_cli_requires_output_json_and_output_csv_explicitly():
    result = _run_module()
    assert result.returncode != 0
    required_lines = [ln for ln in result.stderr.splitlines() if REQUIRED_ARGS_MARKER in ln]
    assert required_lines, (
        f"expected argparse to reject the run outright, got: {result.stderr[:400]}"
    )
    assert "--output-json" in required_lines[0], (
        f"--output-json was not named as required; it may have regained a default: "
        f"{required_lines[0][:200]}"
    )
    assert "--output-csv" in required_lines[0], (
        f"--output-csv was not named as required; it may have regained a default: "
        f"{required_lines[0][:200]}"
    )


def test_cli_advertises_allow_overwrite():
    result = _run_module("--help")
    assert result.returncode == 0
    assert "--allow-overwrite" in result.stdout


def test_cli_actually_threads_allow_overwrite_into_run_loo():
    """Advertising the flag is not the same as wiring it.

    An --allow-overwrite that argparse accepts but main() never forwards is a
    silent no-op: --help looks right, the flag is accepted, and the guard
    still aborts the run it was meant to permit. Checking the source of the
    main() call is cheap; the alternative is executing the real training run.
    """
    source = (REPO_ROOT / "scripts" / "run_loo_cross_virus_v5.py").read_text(encoding="utf-8")
    call_start = source.index("    run_loo(")
    call_block = source[call_start : source.index("\n\n", call_start)]
    assert "allow_overwrite=args.allow_overwrite," in call_block, (
        "main() parses --allow-overwrite but never forwards it to run_loo"
    )
