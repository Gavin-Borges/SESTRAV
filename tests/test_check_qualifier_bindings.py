from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import check_qualifier_bindings as gate


ROOT = Path(__file__).parents[1]

# Stripped again here although the root conftest.py already does it, so the
# throwaway repos below can never be resolved through the real repository.
_GIT_DISCOVERY_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX")


def _git(repo: Path, *args: str) -> None:
    env = {key: value for key, value in os.environ.items() if key not in _GIT_DISCOVERY_VARS}
    subprocess.run(["git", "-C", str(repo), *args], env=env, check=True, capture_output=True)


def _binding(**overrides):
    binding = {
        "id": "example",
        "value": "0.6458",
        "decimals": 4,
        "qualifier_patterns": [r"best\s+of\s+eight"],
        "window_lines": 1,
        "scan_globs": ["docs/**/*.md"],
        "violation_ceiling": 0,
        "carriers": [{"path": "docs/result.md", "anchor_pattern": r"score.*0\.6458"}],
        "exemptions": [],
    }
    binding.update(overrides)
    return binding


def _declared_ceilings(config) -> dict[str, int]:
    return {binding["id"]: binding["violation_ceiling"] for binding in config["bindings"]}


def test_repository_config_declares_the_debt_it_still_carries():
    """Pin the DECLARED ceilings exactly - and pin them only here.

    The script is a one-sided ratchet: ``audit`` raises ``over_ceiling`` only when
    a binding's violations EXCEED its ceiling, so repairing a carrier passes. The
    two live-tree tests below therefore assert ``<= ceiling`` rather than a count,
    because a two-sided ``==`` pin on the MEASURED count made this gate and the
    separate best-of-eight repair branch mutually unlandable: inserting the
    missing qualifier in ARCHITECTURE.md's window drops the measured total from 7
    to 6, which the script accepts and a ``== 7`` assertion rejects, so whichever
    landed second would have reddened CI. The exact debt stays pinned in this test, so
    lowering a ceiling remains a deliberate, reviewed edit here plus
    ``check_qualifier_bindings.py --update`` on the config in the same commit.
    """
    config = gate._load_config(ROOT / "docs/qualifier_bindings.json")
    assert _declared_ceilings(config) == {
        "gnn-gate1-best-of-eight": 7,
        "per-virus-mean-nine-virus-scope": 0,
    }


def test_repository_config_passes_its_ratchet():
    config = gate._load_config(ROOT / "docs/qualifier_bindings.json")
    ceilings = _declared_ceilings(config)
    result = gate.audit(ROOT, config)
    assert result.bindings_checked == 2
    assert result.carriers_checked == 9
    assert len(result.violations) <= sum(ceilings.values())
    # Per binding, because that is the unit the script's ratchet is computed over.
    for binding_id, measured in result.per_binding.items():
        assert len(measured.violations) <= ceilings[binding_id]
    assert not result.over_ceiling


def test_strict_mode_exposes_the_seed_debt():
    config = gate._load_config(ROOT / "docs/qualifier_bindings.json")
    ceilings = _declared_ceilings(config)
    result = gate.audit(ROOT, config, strict=True)
    assert result.over_ceiling
    # The lower bound keeps the `all(...)` below from passing vacuously, which is
    # the non-vacuity the retired `== 7` used to supply. NOTE: it also means a
    # COMPLETE repair of every seed carrier reds this test by design - retire it
    # in the commit that clears the last violation, and drop the ceiling to 0 in
    # the pin above.
    assert 1 <= len(result.violations) <= sum(ceilings.values())
    assert all("gnn-gate1-best-of-eight" in item for item in result.violations)


def test_counterfactual_qualifier_removal_names_file_and_line(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    carrier = docs / "result.md"
    carrier.write_text("score 0.6458, best of eight runs\n", encoding="utf-8")
    config = {"version": 1, "bindings": [_binding()]}
    assert not gate.audit(tmp_path, config, strict=True).violations

    carrier.write_text("score 0.6458\n", encoding="utf-8")
    result = gate.audit(tmp_path, config, strict=True)
    assert result.over_ceiling
    # The expected text is ASSEMBLED rather than written as one literal.
    # scripts/check_doc_line_citations.py would read a contiguous fixture path
    # plus a line number here as a real citation (its CITATION_RE matches a
    # listed extension followed by a colon and digits), and would then fail
    # the required context "Cited lines still hold their content" on a
    # tmp_path fixture no reader can follow.
    expected = "docs/result.md" + ":1: example: value 0.6458 lacks a"
    expected += " required qualifier within 1 line(s)"
    assert result.violations == [expected]


def test_longer_float_and_file_path_are_not_claims(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "result.md").write_text("score 0.6458, best of eight runs\n", encoding="utf-8")
    (docs / "near_miss.md").write_text(
        "The diagnostic was 0.64581. See artifacts/0.6458/report.json.\n",
        encoding="utf-8",
    )
    result = gate.audit(tmp_path, {"version": 1, "bindings": [_binding()]}, strict=True)
    assert not result.violations
    assert result.carriers_checked == 1


def test_a_value_that_ends_a_sentence_is_still_a_claim(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "result.md").write_text("score 0.6458, best of eight runs\n", encoding="utf-8")
    (docs / "summary.md").write_text("The structural scorer reached 0.6458.\n", encoding="utf-8")
    result = gate.audit(tmp_path, {"version": 1, "bindings": [_binding()]}, strict=True)
    assert len(result.violations) == 1
    assert result.violations[0].startswith("docs/summary.md" + ":1:")


def test_untracked_and_gitignored_files_do_not_change_the_result(tmp_path: Path):
    """Only TRACKED files are scanned when the root is a git work tree's top level.

    A working checkout can hold gitignored notes that restate a bound value
    bare. Scanning the filesystem would count them, so a local run (and the
    pre-push check) would disagree with a CI checkout of the same commit.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "result.md").write_text("score 0.6458, best of eight runs\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("docs/ignored.md\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "docs/result.md", ".gitignore")
    config = {"version": 1, "bindings": [_binding()]}
    baseline = gate.audit(tmp_path, config, strict=True)
    assert baseline.violations == []

    (docs / "ignored.md").write_text("score 0.6458\n", encoding="utf-8")
    (docs / "untracked.md").write_text("score 0.6458\n", encoding="utf-8")
    result = gate.audit(tmp_path, config, strict=True)
    assert result.violations == baseline.violations
    assert result.carriers_checked == baseline.carriers_checked == 1

    # Non-vacuity: the same bare value is caught once git tracks the file.
    _git(tmp_path, "add", "docs/untracked.md")
    tracked = gate.audit(tmp_path, config, strict=True)
    assert len(tracked.violations) == 1
    assert tracked.violations[0].startswith("docs/untracked.md" + ":1:")


def test_an_untracked_registered_carrier_counts_as_missing(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "result.md").write_text("score 0.6458, best of eight runs\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    config = {"version": 1, "bindings": [_binding()]}
    result = gate.audit(tmp_path, config, strict=True)
    assert result.violations == ["docs/result.md" + ":0: example: carrier is missing"]

    _git(tmp_path, "add", "docs/result.md")
    assert gate.audit(tmp_path, config, strict=True).violations == []


def test_every_exemption_requires_a_note(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "result.md").write_text("score 0.6458\n", encoding="utf-8")
    binding = _binding(exemptions=[{"path": "docs/history.md", "note": ""}])
    with pytest.raises(gate.ConfigError, match="needs a non-empty note"):
        gate.audit(tmp_path, {"version": 1, "bindings": [binding]})


def test_update_records_line_hash_and_measured_ceiling(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "result.md").write_text("score 0.6458\n", encoding="utf-8")
    config_path = docs / "bindings.json"
    config_path.write_text(
        json.dumps({"version": 1, "bindings": [_binding(violation_ceiling=4)]}),
        encoding="utf-8",
    )

    assert gate.main(["--root", str(tmp_path), "--config", str(config_path), "--update"]) == 0
    updated = json.loads(config_path.read_text(encoding="utf-8"))
    binding = updated["bindings"][0]
    assert binding["violation_ceiling"] == 1
    assert binding["carriers"][0]["observed"]["line"] == 1
    assert len(binding["carriers"][0]["observed"]["window_sha256"]) == 64
