"""Behavioural tests for scripts/check_panel_duplicates.py.

The script shipped with no test coverage at all. It is a curation aid that
answers one question - is this panel already inside the training corpus - and
the answer feeds a contamination decision, so a wrong all-clear is the failure
that matters. The blank-peptide case below is a regression test for exactly
that: two tracked panels carry rows whose peptide column is entirely empty, and
the script used to certify all of them as new and safe for held-out use.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_panel_duplicates.py"


def _run(panel: Path, dataset: Path, out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--panel-file",
            str(panel),
            "--dataset-file",
            str(dataset),
            "--output-dir",
            str(out),
        ],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.csv"
    path.write_text("peptide,hla_allele\nSIINFEKL,HLA-A*02:01\nGILGFVFTL,HLA-A*02:01\n", encoding="utf-8")
    return path


def test_blank_peptide_column_refuses_a_verdict(tmp_path: Path, dataset: Path) -> None:
    """A present-but-empty peptide column must not read as 'all rows are new'."""
    panel = tmp_path / "blank.csv"
    panel.write_text("peptide,hla_allele\n,HLA-A*02:01\n,HLA-B*07:02\n", encoding="utf-8")

    result = _run(panel, dataset, tmp_path)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "every value is empty" in result.stderr
    # The false all-clear this guards against must not appear.
    assert "safe for held-out use" not in result.stdout


def test_panel_with_novel_peptides_passes(tmp_path: Path, dataset: Path) -> None:
    panel = tmp_path / "novel.csv"
    panel.write_text("peptide,hla_allele\nKLVALGINAV,HLA-A*02:01\n", encoding="utf-8")

    result = _run(panel, dataset, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr


def test_overlapping_peptide_is_reported(tmp_path: Path, dataset: Path) -> None:
    """A peptide already in the corpus is the thing the tool exists to find."""
    panel = tmp_path / "overlap.csv"
    panel.write_text("peptide,hla_allele\nSIINFEKL,HLA-A*02:01\n", encoding="utf-8")

    result = _run(panel, dataset, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1" in result.stdout


def test_header_only_panel_is_not_an_error(tmp_path: Path, dataset: Path) -> None:
    """Zero rows is honestly nothing to check, and must stay distinct from blank rows."""
    panel = tmp_path / "empty.csv"
    panel.write_text("peptide,hla_allele\n", encoding="utf-8")

    result = _run(panel, dataset, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
