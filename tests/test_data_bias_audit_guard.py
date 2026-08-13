"""Overwrite-guard tests for src/data_bias_audit.py's --provenance-csv,
--audit-csv and --audit-md.

Step 8 of the results/ silent-overwrite defect-class repair line. This module
has exactly ONE git-tracked artifact at risk: results/data_bias_audit.md
(un-ignored by the explicit `!results/data_bias_audit.md` negation in
.gitignore - cited by the negation itself rather than by line number, because
the line moved from 259 to 263 when 6e75fda added four negations above it and
the pinned citation drifted). A bare `python -m src.data_bias_audit
--source-data-dir ...` rewrote it in place before this fix, because
--provenance-csv, --audit-csv and --audit-md all defaulted into results/ with
no guard.

Unlike every prior module in this line, this one has no single output
directory - its writes straddle data/ (the exempt dataset CSV) and results/
(the guarded provenance/audit files) via independent file-path flags - so it
does not fit tests/test_artifact_guard_contract.py's planned_paths_under
registration shape. See src/artifact_guard.py's module docstring for the
scope/remedy template extension that made this module's guard possible
without changing the message shape for the four existing delegates.

Three hazards drove this module's design (see
_local/notes/step8-enumeration-2026-07-31.md for the full write enumeration):

  Hazard A: --output-csv (default data/immunogenicity_dataset_v4.csv) MUST
  NOT be guarded. It exists on disk right now, is gitignored, is
  refresh_dataset's declared rewrite target, and is read back intra-run by
  write_audit_reports moments later. Guarding it would abort every run
  unconditionally. test_hazard_a_output_csv_is_never_guarded locks this down.

  Hazard B: src/bias_skew_finalization.py is the only caller and passes real
  results_dir paths (no tempfile.mkdtemp() sandbox unlike the h2_tier_a/
  final_validation_report interaction). allow_overwrite now threads through
  all three of its call sites into this module - covered in
  tests/test_bias_skew_finalization_results_guard.py, not here.

  Hazard C: refresh_dataset and write_audit_reports each guard only their own
  writes. A union guard would make write_audit_reports abort because
  refresh_dataset already wrote immunogenicity_provenance.csv moments earlier
  in the same run. test_refresh_dataset_guard_is_indifferent_to_audit_files
  and test_write_audit_reports_guard_is_indifferent_to_provenance_file lock
  this down directly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import data_bias_audit as dba

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_ARGS_MARKER = "the following arguments are required:"


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "src.data_bias_audit", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# Planned-path enumeration
# ---------------------------------------------------------------------------


def test_planned_paths_lists_all_four_and_excludes_output_csv(tmp_path):
    provenance_csv = str(tmp_path / "immunogenicity_provenance.csv")
    audit_csv = str(tmp_path / "data_bias_audit_summary.csv")
    audit_md = str(tmp_path / "data_bias_audit.md")

    paths = dba.planned_data_bias_audit_paths(provenance_csv, audit_csv, audit_md)

    assert [Path(p).name for p in paths] == [
        "immunogenicity_provenance.csv",
        "data_bias_audit_summary.csv",
        "data_bias_audit_summary_virus_label_counts.csv",
        "data_bias_audit.md",
    ]
    assert not any("immunogenicity_dataset_v4.csv" in p for p in paths)


def test_planned_paths_derived_name_matches_the_writer_exactly(tmp_path):
    """The derived name must come from the identical audit_csv.replace(".csv",
    "_virus_label_counts.csv") expression write_audit_reports itself uses
    (str.replace, not os.path.splitext), including on a path containing
    ".csv" twice, where the two diverge."""
    audit_csv = str(tmp_path / "audit.csv.backup.csv")
    expected_derived = audit_csv.replace(".csv", "_virus_label_counts.csv")
    paths = dba.planned_data_bias_audit_paths("prov.csv", audit_csv, "audit.md")
    assert expected_derived in paths
    assert Path(expected_derived).name == "audit_virus_label_counts.csv.backup_virus_label_counts.csv"


# ---------------------------------------------------------------------------
# Hazard A: output_csv must never be part of any guarded set
# ---------------------------------------------------------------------------


def test_hazard_a_output_csv_is_never_guarded(tmp_path):
    """output_csv (the data/ dataset path) is refresh_dataset's declared
    rewrite target, gitignored, and read back intra-run by write_audit_reports
    moments later. Guarding it would abort every run unconditionally, since
    the real dataset already exists on disk right now."""
    output_csv = tmp_path / "immunogenicity_dataset_v4.csv"
    output_csv.write_text("peptide,label\nAAA,1\n", encoding="utf-8")

    provenance_csv = tmp_path / "immunogenicity_provenance.csv"
    audit_csv = tmp_path / "data_bias_audit_summary.csv"
    audit_md = tmp_path / "data_bias_audit.md"

    planned = dba.planned_data_bias_audit_paths(
        str(provenance_csv), str(audit_csv), str(audit_md)
    )
    assert str(output_csv) not in planned
    assert output_csv.name not in {Path(p).name for p in planned}

    # And the guards themselves stay silent even though output_csv already
    # exists, as long as none of the 4 guarded paths do.
    dba._guard_refresh_dataset(str(provenance_csv), allow_overwrite=False)
    dba._guard_write_audit_reports(str(audit_csv), str(audit_md), allow_overwrite=False)
    dba._guard_data_bias_audit_cli(
        str(provenance_csv), str(audit_csv), str(audit_md), allow_overwrite=False
    )


# ---------------------------------------------------------------------------
# Guard behaviour: refresh_dataset's own guard (provenance_csv only)
# ---------------------------------------------------------------------------


def test_refresh_dataset_guard_passes_on_a_nonexistent_path(tmp_path):
    dba._guard_refresh_dataset(str(tmp_path / "immunogenicity_provenance.csv"), False)


def test_refresh_dataset_guard_refuses_to_clobber_provenance_csv(tmp_path):
    provenance_csv = tmp_path / "immunogenicity_provenance.csv"
    provenance_csv.write_text("source_file,n_records\nfoo.csv,10\n", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        dba._guard_refresh_dataset(str(provenance_csv), False)
    assert "immunogenicity_provenance.csv" in str(exc.value)
    assert provenance_csv.read_text(encoding="utf-8").startswith("source_file")


def test_refresh_dataset_guard_is_indifferent_to_audit_files(tmp_path):
    """Hazard C: refresh_dataset's guard must not trip on write_audit_reports'
    own files - a union guard would make a legitimate two-step run abort on
    its second call even though refresh_dataset itself writes nothing that
    collides."""
    (tmp_path / "data_bias_audit_summary.csv").write_text("x", encoding="utf-8")
    (tmp_path / "data_bias_audit.md").write_text("x", encoding="utf-8")
    dba._guard_refresh_dataset(str(tmp_path / "immunogenicity_provenance.csv"), False)


def test_refresh_dataset_allow_overwrite_disarms_its_guard(tmp_path):
    provenance_csv = tmp_path / "immunogenicity_provenance.csv"
    provenance_csv.write_text("existing", encoding="utf-8")
    dba._guard_refresh_dataset(str(provenance_csv), True)


# ---------------------------------------------------------------------------
# Guard behaviour: write_audit_reports' own guard (its 3 files)
# ---------------------------------------------------------------------------

AUDIT_REPORT_NAMES = [
    "data_bias_audit_summary.csv",
    "data_bias_audit_summary_virus_label_counts.csv",
    "data_bias_audit.md",
]


def test_write_audit_reports_guard_passes_on_nonexistent_paths(tmp_path):
    dba._guard_write_audit_reports(
        str(tmp_path / "data_bias_audit_summary.csv"),
        str(tmp_path / "data_bias_audit.md"),
        False,
    )


@pytest.mark.parametrize("name", AUDIT_REPORT_NAMES)
def test_write_audit_reports_guard_refuses_to_clobber_each_file(tmp_path, name):
    audit_csv = tmp_path / "data_bias_audit_summary.csv"
    audit_md = tmp_path / "data_bias_audit.md"
    (tmp_path / name).write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        dba._guard_write_audit_reports(str(audit_csv), str(audit_md), False)
    assert name in str(exc.value)


def test_write_audit_reports_guard_is_indifferent_to_provenance_file(tmp_path):
    """Hazard C, the other direction: write_audit_reports' guard must not trip
    on refresh_dataset's own provenance_csv, which is normally already on disk
    by the time write_audit_reports runs in the same pipeline."""
    (tmp_path / "immunogenicity_provenance.csv").write_text("x", encoding="utf-8")
    dba._guard_write_audit_reports(
        str(tmp_path / "data_bias_audit_summary.csv"),
        str(tmp_path / "data_bias_audit.md"),
        False,
    )


def test_write_audit_reports_allow_overwrite_disarms_its_guard(tmp_path):
    audit_csv = tmp_path / "data_bias_audit_summary.csv"
    audit_md = tmp_path / "data_bias_audit.md"
    for name in AUDIT_REPORT_NAMES:
        (tmp_path / name).write_text("published", encoding="utf-8")
    dba._guard_write_audit_reports(str(audit_csv), str(audit_md), True)


def test_write_audit_reports_guard_names_the_flag_and_escape_hatch(tmp_path):
    audit_csv = tmp_path / "data_bias_audit_summary.csv"
    audit_md = tmp_path / "data_bias_audit.md"
    audit_md.write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        dba._guard_write_audit_reports(str(audit_csv), str(audit_md), False)
    message = str(exc.value)
    assert "--audit-csv" in message
    assert "--audit-md" in message
    assert "--allow-overwrite" in message
    assert "write_audit_reports(..., allow_overwrite=True)" in message


# ---------------------------------------------------------------------------
# Guard behaviour: the __main__ defense-in-depth guard (union of all 4)
# ---------------------------------------------------------------------------


def test_cli_guard_passes_on_an_empty_location(tmp_path):
    dba._guard_data_bias_audit_cli(
        str(tmp_path / "immunogenicity_provenance.csv"),
        str(tmp_path / "data_bias_audit_summary.csv"),
        str(tmp_path / "data_bias_audit.md"),
        False,
    )


@pytest.mark.parametrize(
    "name",
    [
        "immunogenicity_provenance.csv",
        "data_bias_audit_summary.csv",
        "data_bias_audit_summary_virus_label_counts.csv",
        "data_bias_audit.md",
    ],
)
def test_cli_guard_refuses_to_clobber_each_planned_artifact(tmp_path, name):
    provenance_csv = tmp_path / "immunogenicity_provenance.csv"
    audit_csv = tmp_path / "data_bias_audit_summary.csv"
    audit_md = tmp_path / "data_bias_audit.md"
    (tmp_path / name).write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        dba._guard_data_bias_audit_cli(
            str(provenance_csv), str(audit_csv), str(audit_md), False
        )
    assert name in str(exc.value)


def test_cli_guard_allow_overwrite_disarms_it(tmp_path):
    provenance_csv = tmp_path / "immunogenicity_provenance.csv"
    audit_csv = tmp_path / "data_bias_audit_summary.csv"
    audit_md = tmp_path / "data_bias_audit.md"
    for name in [
        "immunogenicity_provenance.csv",
        "data_bias_audit_summary.csv",
        "data_bias_audit.md",
    ]:
        (tmp_path / name).write_text("published", encoding="utf-8")
    dba._guard_data_bias_audit_cli(
        str(provenance_csv), str(audit_csv), str(audit_md), True
    )


# ---------------------------------------------------------------------------
# Wiring: the guards must actually run inside the real functions
# ---------------------------------------------------------------------------


def test_guard_is_wired_into_refresh_dataset(tmp_path, monkeypatch):
    """A defined-but-uncalled guard would pass every guard-behaviour test above."""
    provenance_csv = tmp_path / "immunogenicity_provenance.csv"
    provenance_csv.write_text("published", encoding="utf-8")

    def _should_not_run(*args, **kwargs):
        raise AssertionError("refresh_dataset did work before the guard rejected the run")

    monkeypatch.setattr(dba, "load_and_clean_iedb", _should_not_run)

    with pytest.raises(FileExistsError):
        dba.refresh_dataset(
            source_data_dir=str(tmp_path),
            output_csv=str(tmp_path / "immunogenicity_dataset_v4.csv"),
            provenance_csv=str(provenance_csv),
        )


def test_allow_overwrite_passes_through_refresh_dataset(tmp_path, monkeypatch):
    provenance_csv = tmp_path / "immunogenicity_provenance.csv"
    provenance_csv.write_text("published", encoding="utf-8")

    sentinel = RuntimeError("got past the guard")

    def _stop_after_guard(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(dba, "load_and_clean_iedb", _stop_after_guard)

    with pytest.raises(RuntimeError) as exc:
        dba.refresh_dataset(
            source_data_dir=str(tmp_path),
            output_csv=str(tmp_path / "immunogenicity_dataset_v4.csv"),
            provenance_csv=str(provenance_csv),
            allow_overwrite=True,
        )
    assert exc.value is sentinel


def test_guard_is_wired_into_write_audit_reports(tmp_path, monkeypatch):
    """A defined-but-uncalled guard would pass every guard-behaviour test above."""
    audit_md = tmp_path / "data_bias_audit.md"
    audit_md.write_text("published", encoding="utf-8")

    def _should_not_run(*args, **kwargs):
        raise AssertionError(
            "write_audit_reports did work before the guard rejected the run"
        )

    monkeypatch.setattr(dba.pd, "read_csv", _should_not_run)

    with pytest.raises(FileExistsError):
        dba.write_audit_reports(
            dataset_csv="does_not_matter.csv",
            raw_records=pd.DataFrame(),
            output_csv=str(tmp_path / "data_bias_audit_summary.csv"),
            output_md=str(audit_md),
        )


def test_allow_overwrite_passes_through_write_audit_reports(tmp_path, monkeypatch):
    audit_md = tmp_path / "data_bias_audit.md"
    audit_md.write_text("published", encoding="utf-8")

    sentinel = RuntimeError("got past the guard")

    def _stop_after_guard(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(dba.pd, "read_csv", _stop_after_guard)

    with pytest.raises(RuntimeError) as exc:
        dba.write_audit_reports(
            dataset_csv="does_not_matter.csv",
            raw_records=pd.DataFrame(),
            output_csv=str(tmp_path / "data_bias_audit_summary.csv"),
            output_md=str(audit_md),
            allow_overwrite=True,
        )
    assert exc.value is sentinel


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--provenance-csv", "--audit-csv", "--audit-md"])
def test_cli_requires_flag_explicitly(flag):
    """Omitting any of the 3 newly-required flags must fail fast via argparse's
    own required-arguments line, not merely produce some non-zero exit. The
    guard's own error message also names these flags, so a bare substring
    check against stderr would pass for the wrong reason during a live
    regression - this anchors on the argparse usage line instead."""
    result = _run_module("--source-data-dir", "does_not_exist_dir")
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


def test_cli_output_csv_keeps_its_existing_default():
    """--output-csv (Hazard A's exempt path) is deliberately untouched by this
    fix - it must stay optional with its existing data/ default."""
    result = _run_module("--help")
    assert result.returncode == 0
    assert "[--output-csv" in result.stdout


def test_cli_actually_threads_allow_overwrite_into_both_delegates():
    """Advertising the flag is not the same as wiring it into both call sites."""
    source = (REPO_ROOT / "src" / "data_bias_audit.py").read_text(encoding="utf-8")

    refresh_call_start = source.index("    refreshed_df, _ = refresh_dataset(")
    refresh_call_block = source[refresh_call_start : source.index("    )", refresh_call_start)]
    assert "allow_overwrite=args.allow_overwrite," in refresh_call_block

    audit_call_start = source.index("    write_audit_reports(")
    audit_call_block = source[audit_call_start : source.index("    )", audit_call_start)]
    assert "allow_overwrite=args.allow_overwrite," in audit_call_block


def test_cli_guard_runs_before_refresh_dataset_in_source_order():
    """Ordering constraint from the step-8 enumeration note: the __main__
    defense-in-depth guard must sit above the refresh_dataset call, since
    refresh_dataset parses every IEDB xlsx before writing and a blocked run
    should not pay that cost first."""
    source = (REPO_ROOT / "src" / "data_bias_audit.py").read_text(encoding="utf-8")
    main_start = source.index('if __name__ == "__main__":')
    guard_pos = source.index("_guard_data_bias_audit_cli(", main_start)
    refresh_pos = source.index("refresh_dataset(", main_start)
    assert guard_pos < refresh_pos
