"""Smoke tests for the ``sestrav`` CLI (MASTER_STRATEGIC_PLAN.md Part 12, obj #14).

Verifies the dependency-light entry points run and produce parseable output
without requiring a model, FASTA, or heavy stage modules: ``info``, the no-arg
help banner, and ``--help`` for the program and each subcommand.
"""

import pytest

from src.cli import main


def test_info_returns_zero_and_prints_summary(capsys):
    rc = main(["info"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SESTRAV Environment Info" in out
    assert "torch" in out  # provenance summary lists torch (or 'not installed')


def test_no_subcommand_prints_help_and_returns_zero(capsys):
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Subcommands:" in out
    assert "predict" in out and "validate" in out and "benchmark" in out


def test_top_level_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("sub", ["predict", "validate", "benchmark", "info"])
def test_subcommand_help_exits_zero(sub):
    with pytest.raises(SystemExit) as exc:
        main([sub, "--help"])
    assert exc.value.code == 0


def test_validate_requires_model_dir(capsys):
    """validate retrains, so its destination is explicit - no fallback to models/."""
    with pytest.raises(SystemExit) as exc:
        main(["validate", "--dataset", "does_not_matter.csv"])
    assert exc.value.code != 0
    assert "--model-dir" in capsys.readouterr().err
