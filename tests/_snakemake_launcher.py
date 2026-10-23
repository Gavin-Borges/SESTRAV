"""Resolve the snakemake launcher for a test, PATH first and the running env second.

Shared by every test that runs snakemake as a program, so the lookup cannot differ from
file to file. It did once: `tests/test_snakemake_shell_quoting.py` had both steps below
while `tests/test_pipeline_smoke_executes.py`, the one test that executes the pipeline,
had PATH only, so it skipped on a workstation where the other ran.
`tests/test_snakemake_launcher.py` pins this behaviour.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest


def snakemake_launcher() -> str:
    """Resolve the snakemake launcher, PATH first and the running env second.

    PATH alone is not enough. On a workstation where the conda environment is active
    for the interpreter but not exported to the shell, `shutil.which` returns None while
    snakemake is installed and importable, and a test that skipped on that would certify
    nothing while reading as green. The second lookup is the launcher directory beside
    the interpreter actually running the tests, which is the environment snakemake was
    installed into. Skips only when neither lookup finds one.
    """
    found = shutil.which("snakemake")
    if found is None:
        launcher_dir = Path(sys.executable).parent
        for name in ("Scripts/snakemake.exe", "snakemake.exe", "bin/snakemake", "snakemake"):
            candidate = launcher_dir / name
            if candidate.exists():
                found = str(candidate)
                break
    if found is None:
        pytest.skip("snakemake launcher not found on PATH or beside the running interpreter")
    return str(found)
