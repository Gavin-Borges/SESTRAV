"""Tier-1 results/ guard tests for scripts/eval_tsnadb_crossdomain.py.

Closes Tier-1 enumeration item #14: this script had no CLI at all, so a bare
invocation always silently rewrote the git-tracked
results/tsnadb_crossdomain_benchmark.json. --output is now optional with no
default (the "no-default explicit path" pattern, matching
scripts/evaluate_per_virus.py): a bare run performs the benchmark and prints
metrics without writing anything; passing --output writes there.

The write-or-skip decision lives in the standalone maybe_write_json()
function specifically so it is testable without running the full
MHCflurry/model-scoring pipeline (which needs a real trained model,
MHCflurry's downloaded weights, and the tracked data/models/ inputs) - these
tests exercise only that function and the CLI parsing, not the benchmark
itself.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts import eval_tsnadb_crossdomain as etc

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_maybe_write_json_does_nothing_when_path_is_none(tmp_path, capsys, monkeypatch):
    # chdir for the same reason as the empty-string case below: without it, an empty
    # tmp_path says nothing, because tmp_path is never handed to the function.
    monkeypatch.chdir(tmp_path)
    etc.maybe_write_json({"a": 1}, None)
    assert list(tmp_path.iterdir()) == []
    assert "written" not in capsys.readouterr().out


def test_maybe_write_json_does_nothing_when_path_is_empty_string(tmp_path, capsys, monkeypatch):
    # Two anchors, because neither alone is load-bearing.
    #
    # `monkeypatch.chdir` is what makes the directory assertion mean anything at all.
    # maybe_write_json never receives tmp_path, so a bare `assert list(tmp_path.iterdir())
    # == []` was vacuously true - a fresh empty directory nothing could have written to.
    # After the chdir, a relative path (which is what a reinstated default would be, since
    # this module anchors nothing to REPO_ROOT) lands inside tmp_path and is caught.
    #
    # The stdout check is the independent one: maybe_write_json, the function under test,
    # prints "Results written to ..." on every write it performs, so absence of that token
    # is a real signal about the function's behaviour rather than about the directory.
    # Named by symbol rather than pinned to a line number, which would rot on the next
    # edit upstream of it.
    monkeypatch.chdir(tmp_path)
    etc.maybe_write_json({"a": 1}, "")
    assert list(tmp_path.iterdir()) == []
    assert "written" not in capsys.readouterr().out


def test_maybe_write_json_writes_given_path(tmp_path):
    output_path = tmp_path / "out.json"
    etc.maybe_write_json({"a": 1, "b": [1, 2, 3]}, str(output_path))
    assert output_path.exists()
    assert json.loads(output_path.read_text()) == {"a": 1, "b": [1, 2, 3]}


def test_maybe_write_json_creates_parent_directory_if_missing(tmp_path):
    output_path = tmp_path / "new_subdir" / "out.json"
    assert not output_path.parent.exists()
    etc.maybe_write_json({"a": 1}, str(output_path))
    assert output_path.exists()


def test_parse_args_output_defaults_to_none():
    args = etc.parse_args([])
    assert args.output is None


def test_parse_args_output_accepts_explicit_path():
    args = etc.parse_args(["--output", "custom/out.json"])
    assert args.output == "custom/out.json"


def test_cli_help_advertises_no_default_output():
    # COLUMNS is pinned wide because this assertion is otherwise decided by where the
    # repository happens to be checked out, not by the help text. argparse wraps at the
    # terminal width (80 when COLUMNS is unset and stdout is not a tty), and this script's
    # TRACKED_OUTPUT is an ABSOLUTE path - unlike its sibling in
    # compute_loo_binding_confound.py, which is relative and short. On a deep checkout the
    # wrap lands mid-filename ("tsnadb_crossdomain_be" / "nchmark.json") and the substring
    # check fails against a help text that is perfectly correct. Measured here: the
    # filename survives intact at COLUMNS=100 and 200, and is split at 80.
    #
    # Normalizing whitespace afterwards does not fix it - argparse breaks inside the word,
    # so rejoining still yields "tsnadb_crossdomain_be nchmark.json".
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.eval_tsnadb_crossdomain", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "COLUMNS": "200"},
    )
    assert proc.returncode == 0
    assert "No default" in proc.stdout
    assert "tsnadb_crossdomain_benchmark.json" in proc.stdout


def test_tracked_output_constant_unchanged():
    """Sanity check the doc-reference constant still names the real tracked path."""
    assert etc.TRACKED_OUTPUT.replace("\\", "/").endswith(
        "results/tsnadb_crossdomain_benchmark.json"
    )
