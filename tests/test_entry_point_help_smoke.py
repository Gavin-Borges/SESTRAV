"""`--help` smoke tests for every entry point with a required output directory.

`src/train_ann.py` is deliberately excluded: its `models/ann` default holds no
tracked artifacts, so it is the one writer of the family that never needed the
required flag.

Two regressions motivate this file:

1. A `--help` epilog string that fails to parse breaks the command for every
   user while every unit test still passes, because nothing else exercises
   argparse construction. That exact regression was caught by hand during the
   model-dir repair work and could silently return.
2. The output-directory flag was made required on the entry points that write
   the tracked release artifacts under `models/` and `results/`. A default
   quietly reappearing would restore the silent-overwrite trap, and no unit
   test would notice.

These run the real CLI in a subprocess, so they cover argparse construction,
module-level import side effects and the epilog together. `--help` neither
trains nor writes anything.

Matches the existing subprocess pattern in tests/test_features_mode31.py.

Machine note: the 4 cases parametrized on `src.shap_analysis` spawn a process
that imports the real `shap` library, which hard-crashes the interpreter on at
least one Windows dev machine in this project (Windows fatal exception
0xc06d007f inside scipy.linalg.inv, called from
shap/plots/colors/_colorconv.py at import time - the same pre-existing crash
that makes tests/test_run_analysis_results_guard.py uncollectable there). They
fail rather than crash, because the crash is contained in the subprocess, and
CI on Linux is the authoritative check for them. The guard-side coverage for
that module lives in tests/test_shap_analysis_results_guard.py, which stubs the
import out and therefore runs everywhere.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Entry points whose --model-dir must stay required with no default, because
# they write into the published models/ directory.
MODEL_DIR_REQUIRED_ENTRY_POINTS = [
    "src.train_classifier",
    "src.ann_benchmark",
    "src.bias_skew_finalization",
    "src.train_gnn",
]

# Same defect class, different flag name: these write their result CSVs to
# --output-dir, which used to default to models/.
OUTPUT_DIR_REQUIRED_ENTRY_POINTS = [
    "src.gnn_benchmark",
    "src.ablation_study",
    # Writes into results/, not models/: the 3 tracked H2 Tier A artifacts,
    # one of which backs the certified R10 = 0.9494 result in README.md.
    "src.h2_tier_a_evaluation",
    # Tier-1 #7: 3 calibration artifacts, one of them the tracked
    # results/calibration_metrics.csv.
    "src.calibration_analysis",
    # Tier-1 #10: 7 SHAP artifacts (3 of the filename templates are
    # interpolated over 2 model tags), one of them the tracked
    # results/shap_values_rf.csv. Its --results-dir is an INPUT flag and stays
    # optional on purpose, so it is deliberately absent from
    # RESULTS_DIR_REQUIRED_ENTRY_POINTS below.
    "src.shap_analysis",
]

# Same defect class again, a third flag name: this one writes its CV summary
# to --output-summary, which used to default to models/ann_cv_summary.csv.
OUTPUT_SUMMARY_REQUIRED_ENTRY_POINTS = ["scripts.compute_ann_baseline_summary"]

# Same defect class, a fourth flag name, and the only one pointing at results/
# rather than models/: this guards the 8 files run_bias_skew_finalization writes
# into --results-dir. src.bias_skew_finalization is the one entry point carrying
# two independently required output flags, so it appears both here and in
# MODEL_DIR_REQUIRED_ENTRY_POINTS. The checks below are keyed on (module, flag)
# pairs, so listing it twice checks each flag rather than double-checking one.
# src.prepare_external_validation_inputs guards its 4 external-validation
# sidecar artifacts and src.external_benchmark_comparison guards its 6
# comparison-report artifacts (see
# _local/notes/results-dir-tier1-enumeration-2026-07-30.md, Tier-1 #2 and #3);
# both use the same --results-dir flag name as bias_skew_finalization.
# src.baseline_comparison (Tier-1 #6) guards the tracked
# results/baseline_comparison.csv. Its --results-dir is the one dual-purpose
# flag in this list: the run reads its {prefix}_features.csv inputs out of the
# same directory it writes into, which is why its guard carries a custom remedy
# clause. The required-flag contract checked here is identical either way.
RESULTS_DIR_REQUIRED_ENTRY_POINTS = [
    "src.bias_skew_finalization",
    "src.prepare_external_validation_inputs",
    "src.external_benchmark_comparison",
    "src.baseline_comparison",
]

# Same defect class, a fifth flag name: src.external_validation_cross_virus
# guards the single tracked results/external_validation_cross_virus.csv
# artifact at --output (Tier-1 #9 in the enumeration doc above), and
# scripts.compute_population_coverage guards the tracked
# results/population_coverage_v5.json at the same flag name (Tier-1 #11). Both
# flags name a file rather than a directory. Like
# scripts.compute_ann_baseline_summary above, the second entry is a `scripts.`
# module rather than a `src.` one; `python -m` reaches it the same way.
OUTPUT_REQUIRED_ENTRY_POINTS = [
    "src.external_validation_cross_virus",
    "scripts.compute_population_coverage",
]

# Same defect class, step 8: these two modules have no single output
# directory - both take independent required file-path flags rather than one
# --output-dir/--results-dir, so they don't fit any of the single-flag-name
# lists above (each of which applies one flag name uniformly across every
# module it lists). Listed as explicit (module, [flags]) pairs instead. See
# _local/notes/step8-enumeration-2026-07-31.md for the write enumeration:
# src.data_bias_audit protects one tracked file (results/data_bias_audit.md),
# src.gold_standard_sensitivity protects zero tracked files and is guarded for
# family-consistency/disclosure reasons only.
MULTI_FLAG_REQUIRED_ENTRY_POINTS = [
    ("src.data_bias_audit", ["--provenance-csv", "--audit-csv", "--audit-md"]),
    ("src.gold_standard_sensitivity", ["--output-csv", "--output-md"]),
]

# (module, flag) pairs for the checks that do not care which name the flag has.
REQUIRED_OUTPUT_FLAGS = [
    *((m, "--model-dir") for m in MODEL_DIR_REQUIRED_ENTRY_POINTS),
    *((m, "--output-dir") for m in OUTPUT_DIR_REQUIRED_ENTRY_POINTS),
    *((m, "--output-summary") for m in OUTPUT_SUMMARY_REQUIRED_ENTRY_POINTS),
    *((m, "--results-dir") for m in RESULTS_DIR_REQUIRED_ENTRY_POINTS),
    *((m, "--output") for m in OUTPUT_REQUIRED_ENTRY_POINTS),
    *((m, flag) for m, flags in MULTI_FLAG_REQUIRED_ENTRY_POINTS for flag in flags),
]

# Deliberately does not splat RESULTS_DIR_REQUIRED_ENTRY_POINTS in full: its
# first member (src.bias_skew_finalization) already appears via
# MODEL_DIR_REQUIRED_ENTRY_POINTS, and the checks keyed on this list are
# per-module, not per-flag, so it is only added once below via dict.fromkeys.
ALL_ENTRY_POINTS = list(
    dict.fromkeys(
        [
            *MODEL_DIR_REQUIRED_ENTRY_POINTS,
            *OUTPUT_DIR_REQUIRED_ENTRY_POINTS,
            *OUTPUT_SUMMARY_REQUIRED_ENTRY_POINTS,
            *RESULTS_DIR_REQUIRED_ENTRY_POINTS,
            *OUTPUT_REQUIRED_ENTRY_POINTS,
            *(m for m, _ in MULTI_FLAG_REQUIRED_ENTRY_POINTS),
            "src.cli",
        ]
    )
)

# Per-module, so an entry point with two required flags is still checked once.
ALLOW_OVERWRITE_ENTRY_POINTS = list(dict.fromkeys(m for m, _ in REQUIRED_OUTPUT_FLAGS))

# argparse's wording for a missing required argument. Anchoring on this line
# keeps the rejection check from being satisfied by the echoed usage line.
REQUIRED_ARGS_MARKER = "the following arguments are required:"


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def _run_help(module: str) -> subprocess.CompletedProcess[str]:
    return _run_module(module, "--help")


@pytest.mark.parametrize("module", ALL_ENTRY_POINTS)
def test_help_parses_and_exits_clean(module: str) -> None:
    """`python -m <module> --help` must exit 0 and print a usage block.

    This is what catches a malformed epilog: argparse raises while building
    the parser, so the command dies before it can print usage.
    """
    result = _run_help(module)
    assert result.returncode == 0, f"{module} --help exited {result.returncode}: {result.stderr}"
    assert "usage:" in result.stdout.lower(), f"{module} --help printed no usage block"


@pytest.mark.parametrize(("module", "flag"), REQUIRED_OUTPUT_FLAGS)
def test_output_dir_flag_is_advertised_as_required(module: str, flag: str) -> None:
    """The output-directory flag must appear in help and must not advertise a default.

    argparse renders an optional argument in the usage line wrapped in
    brackets; a required one is unbracketed. Checking the usage line rather
    than the help prose keeps this honest if the help text is reworded.
    """
    result = _run_help(module)
    assert flag in result.stdout, f"{module} does not expose {flag}"
    assert f"[{flag}" not in result.stdout, (
        f"{module} advertises {flag} as optional; it must stay required so a run "
        "cannot silently overwrite the published artifacts under models/ or results/"
    )


@pytest.mark.parametrize(("module", "flag"), REQUIRED_OUTPUT_FLAGS)
def test_missing_output_dir_flag_is_rejected(module: str, flag: str) -> None:
    """Omitting the output-directory flag must fail fast, before any work starts.

    The flag has to be named on argparse's "the following arguments are
    required" line, not merely somewhere in stderr: argparse also echoes the
    usage line, which names every flag whether it is required or defaulted. On
    an entry point carrying more than one required flag, a plain substring
    check against all of stderr passes even when the flag under test has
    silently regained a default, because a *different* missing flag is what
    produced the non-zero exit.
    """
    result = _run_module(module, "--data", "does_not_exist.csv")
    assert result.returncode != 0, f"{module} accepted a run with no {flag}"
    required_lines = [ln for ln in result.stderr.splitlines() if REQUIRED_ARGS_MARKER in ln]
    assert required_lines, (
        f"{module} failed without an argparse required-arguments error: {result.stderr[:400]}"
    )
    assert any(flag in ln for ln in required_lines), (
        f"{module} failed without naming {flag} as required; it may have regained a "
        f"default: {' '.join(required_lines)[:400]}"
    )


@pytest.mark.parametrize("module", ALLOW_OVERWRITE_ENTRY_POINTS)
def test_allow_overwrite_escape_hatch_is_advertised(module: str) -> None:
    """The guard must document its own escape hatch, or it reads as a hard block."""
    result = _run_help(module)
    assert "--allow-overwrite" in result.stdout, (
        f"{module} exposes no --allow-overwrite flag, so a deliberate retrain into an "
        "existing directory has no documented path"
    )
