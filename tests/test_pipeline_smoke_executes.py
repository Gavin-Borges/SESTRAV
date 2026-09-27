"""Execute one real Snakemake rule against the tracked smoke fixture.

Both CI Snakemake legs are dry-runs, which resolve file names and never run a
job body. This test runs generate_peptides for real, in a git-archive
extraction of HEAD, so it sees committed content only: an uncommitted edit is
invisible to it, and it needs a git checkout to run at all.
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _extract_tracked_tree(destination: Path) -> None:
    result = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            assert target.is_relative_to(destination.resolve())
        archive.extractall(destination)  # noqa: S202 - members validated above


def test_tracked_smoke_fixture_executes_generate_peptides(tmp_path):
    snakemake = shutil.which("snakemake")
    if snakemake is None:
        pytest.skip("snakemake executable is not installed in this test environment")

    tree = tmp_path / "tracked-tree"
    _extract_tracked_tree(tree)
    result = subprocess.run(
        [
            snakemake,
            "--snakefile",
            "pipeline.smk",
            "--configfile",
            "tests/fixtures/dag_smoke/config.smoke.yaml",
            "--cores",
            "1",
            "results/SMOKE_peptides.csv",
        ],
        cwd=tree,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    output = tree / "results/SMOKE_peptides.csv"
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == ["protein_id", "peptide", "length", "start", "end"]
    assert len(rows) == 25
    for row in rows:
        length = int(row["length"])
        assert len(row["peptide"]) == length == 9
        assert int(row["end"]) - int(row["start"]) + 1 == length
