"""`--help` smoke tests for every entry point whose `--model-dir` is required.

Deliberately not a sweep of everything that writes artifacts: `src/train_ann.py`,
`src/train_gnn.py` and `src/gnn_benchmark.py` also write, and are not covered here.

Two regressions motivate this file:

1. A `--help` epilog string that fails to parse breaks the command for every
   user while every unit test still passes, because nothing else exercises
   argparse construction. That exact regression was caught by hand during the
   model-dir repair work and could silently return.
2. `--model-dir` was made required on the entry points that write the tracked
   release artifacts under `models/`. A default quietly reappearing would
   restore the silent-overwrite trap, and no unit test would notice.

These run the real CLI in a subprocess, so they cover argparse construction,
module-level import side effects and the epilog together. `--help` neither
trains nor writes anything.

Matches the existing subprocess pattern in tests/test_features_mode31.py.
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
]

ALL_ENTRY_POINTS = [*MODEL_DIR_REQUIRED_ENTRY_POINTS, "src.cli"]


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


@pytest.mark.parametrize("module", MODEL_DIR_REQUIRED_ENTRY_POINTS)
def test_model_dir_is_advertised_as_required(module: str) -> None:
    """--model-dir must appear in help and must not advertise a default.

    argparse renders an optional argument in the usage line wrapped in
    brackets; a required one is unbracketed. Checking the usage line rather
    than the help prose keeps this honest if the help text is reworded.
    """
    result = _run_help(module)
    assert "--model-dir" in result.stdout, f"{module} does not expose --model-dir"
    assert "[--model-dir" not in result.stdout, (
        f"{module} advertises --model-dir as optional; it must stay required so a run "
        "cannot silently overwrite the published artifacts under models/"
    )


@pytest.mark.parametrize("module", MODEL_DIR_REQUIRED_ENTRY_POINTS)
def test_missing_model_dir_is_rejected(module: str) -> None:
    """Omitting --model-dir must fail fast, before any training work starts."""
    result = _run_module(module, "--data", "does_not_exist.csv")
    assert result.returncode != 0, f"{module} accepted a run with no --model-dir"
    assert "--model-dir" in result.stderr, (
        f"{module} failed without naming --model-dir as the cause: {result.stderr[:400]}"
    )


@pytest.mark.parametrize("module", MODEL_DIR_REQUIRED_ENTRY_POINTS)
def test_allow_overwrite_escape_hatch_is_advertised(module: str) -> None:
    """The guard must document its own escape hatch, or it reads as a hard block."""
    result = _run_help(module)
    assert "--allow-overwrite" in result.stdout, (
        f"{module} exposes no --allow-overwrite flag, so a deliberate retrain into an "
        "existing directory has no documented path"
    )
