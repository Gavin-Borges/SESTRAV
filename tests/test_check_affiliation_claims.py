"""Regression tests for the affiliation-allowlist gate.

`scripts/check_affiliation_claims.py` is the CI gate
(`.github/workflows/affiliation_claims.yml`) and pre-push Check 3 that fails
when a tracked file names an institution nobody has reviewed.

The motivating defect is `docs/claims_register.md` D35. A documentation
hygiene pass rewrote a README line that named NO institution into one that
claimed the project's coursework was "at NC State" - an institution SESTRAV
has never had any connection to. It reached the public README and stood there
for roughly five weeks. Every gate this repo had was blind to it: the claim
carried no number and no retracted token, so the retracted-token sweep, the
reconcile check and the citation gate could not see it by construction.

The gate's whole value is classifying correctly in BOTH directions, so these
tests pin both. `test_the_original_fabrication_is_caught` uses the exact
published string, because a gate that would not have caught the bug that
caused it is theatre.

Two subtleties are pinned deliberately, since both were real bugs during
development and both are the kind a later "simplification" would reintroduce:

* A retracted name must remain quotable in the files whose job is to record
  that it is false, and must still fail anywhere else. A blanket allow, or a
  whole-line suppression, would let the original bug back into README.md.
* `str.lstrip` takes a character SET, not a prefix, so `lstrip("./")` turned
  ".claude/rules/..." into "claude/rules/...", and no dotfile path ever
  matched its own allowlist entry.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_affiliation_claims.py"

# The exact text that was published in README.md and lived on origin/main.
ORIGINAL_FABRICATION = (
    "*Developed by Gavin Borges. Academic acknowledgements: bioinformatics "
    "coursework at NC State (BPS 542 / CMB 522 / CSC 522 / STA 522; CMB 523) "
    "provided foundational grounding; SESTRAV is an independently maintained "
    "research tool.*"
)

CORRECTED = ORIGINAL_FABRICATION.replace(
    "at NC State", "at the University of Rhode Island"
)


def _load_module():
    """Import the checker by path - `scripts/` is not an installed package."""
    spec = importlib.util.spec_from_file_location("check_affiliation_claims", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _unreviewed(line: str, path: str = "README.md") -> list[str]:
    return [n for n in mod.find_institutions(line) if not mod.is_allowed(n, path)]


def test_the_original_fabrication_is_caught():
    """The exact published D35 string must fail. This is the whole point."""
    assert _unreviewed(ORIGINAL_FABRICATION) == ["NC State"]


def test_the_correction_passes():
    assert _unreviewed(CORRECTED) == []


def test_real_affiliation_records_pass():
    for line in (
        "| Gavin Borges | @Gavin-Borges | Lead maintainer | University of Rhode Island |",
        'affiliation: "University of Rhode Island"',
        "Copyright (c) 2026 SESTRAV Team - University of Rhode Island",
        "**Original SESTRAV 1.0 Foundation Team (University of Rhode Island)**",
    ):
        assert _unreviewed(line) == [], line


def test_other_fabricated_institutions_are_caught():
    """The gate is an allowlist, not a denylist keyed on "NC State"."""
    for line, expected in (
        ("coursework at Ohio State provided grounding", "Ohio State"),
        ("Gavin Borges, University of Delaware", "University of Delaware"),
        ("Affiliation: Stanford University", "Stanford University"),
        ("a fellow at Imperial College", "Imperial College"),
    ):
        assert expected in _unreviewed(line), line


def test_mit_the_licence_is_not_the_institute():
    """MIT names this project's licence far more often than any affiliation.

    Flagging the bare token unconditionally produced 100+ findings on a repo
    whose only real fabrication was "NC State", and a gate that cries wolf
    stops being read. An abbreviation counts only near an affiliation word.
    """
    for line in (
        "MIT License. See `LICENSE` for details.",
        "| PuLP | 3.3.2 | MIT |",
        "![License: MIT](https://img.shields.io/badge/License-MIT-green)",
    ):
        assert mod.find_institutions(line) == [], line


def test_state_as_an_ordinary_word_is_not_an_institution():
    for line in (
        "The model State dict is saved.",
        "## Current State",
        "Hidden State vectors are cached; see STATE.md",
    ):
        assert mod.find_institutions(line) == [], line


def test_retracted_name_is_quotable_only_in_the_retraction_record():
    """D35 has to be able to name the claim it retracts, and only there."""
    line = 'the prior version read "coursework at NC State", which was false'

    assert _unreviewed(line, "docs/claims_register.md") == []
    assert _unreviewed(line, ".claude/rules/third-party-claims.md") == []
    # Anywhere else it is the original defect returning.
    assert _unreviewed(line, "README.md") == ["NC State"]
    assert _unreviewed(line, "docs/paper.md") == ["NC State"]


def test_generated_rule_mirror_is_exempt_only_for_the_retracted_name():
    """The .agents mirror may quote D35, and nothing more.

    The narrowness is the load-bearing half. Exempting the mirror DIRECTORY, or
    keying the exemption on the "GENERATED FILE - DO NOT EDIT" banner, would let
    a file win exemption by its own content.
    """
    mirror = ".agents/rules/third-party-claims.md"
    line = 'the prior version read "coursework at NC State", which was false'

    assert _unreviewed(line, mirror) == []
    # A DIFFERENT name in the SAME exempt file is still unreviewed.
    assert _unreviewed("Affiliation: Stanford University", mirror) == [
        "Stanford University"
    ]
    # A DIFFERENT file in the SAME mirror directory is still scanned and caught.
    assert mod.should_scan(".agents/rules/gnn.md") is True
    assert _unreviewed(line, ".agents/rules/gnn.md") == ["NC State"]


def test_mirror_exemption_tracks_its_canonical_source():
    """Any .claude/rules exemption must also name its generated .agents mirror.

    Pure string logic over the table, with no filesystem access: .claude/ and
    .agents/ are gitignored, so a CI checkout has neither and an existence
    assertion would fail there for reasons unrelated to the gate.
    """
    for name, paths in mod.RETRACTED_INSTITUTIONS.items():
        for rel in paths:
            if rel.startswith(".claude/rules/"):
                mirror = rel.replace(".claude/rules/", ".agents/rules/", 1)
                assert mirror in paths, (
                    f"{name}: {rel} is exempt but its generated mirror "
                    f"{mirror} is not; re-read sync_agent_rules.py"
                )


def test_self_exemption_covers_the_gate_and_its_suite_and_nothing_else():
    """The gate must not fail on its own machinery, and must stay narrow.

    This file and the gate's own source necessarily contain the names the
    gate screens for - the allowlist, the retracted-name table, and the
    fixtures above that prove detection works. Both are exempt.

    Discovered the hard way: this file was NOT exempt when written, and the
    gate blocked the very commit that introduced it, with 17 findings that
    were all fixtures. The exemption is undiscoverable before the file is
    tracked, because an untracked file is not scanned in the default mode.

    The narrowness is the load-bearing half. Exempting all of tests/, or
    matching on a "test_" prefix, would let a fabricated affiliation sit
    unchallenged in any test file - so the assertions below pin that a
    DIFFERENT test file is still scanned.
    """
    assert mod.should_scan("scripts/check_affiliation_claims.py") is False
    assert mod.should_scan("tests/test_check_affiliation_claims.py") is False

    # Everything else stays scanned - especially other test files.
    assert mod.should_scan("tests/test_something_else.py") is True
    assert mod.should_scan("README.md") is True
    assert mod.should_scan("docs/claims_register.md") is True


def test_dotfile_paths_normalise_correctly():
    """Guards the lstrip-takes-a-character-set bug described in the docstring."""
    assert mod.normalise_path("./.claude/rules/x.md") == ".claude/rules/x.md"
    assert mod.normalise_path(".claude\\rules\\x.md") == ".claude/rules/x.md"
    assert mod.normalise_path("docs/claims_register.md") == "docs/claims_register.md"


def test_name_is_not_matched_across_a_sentence_boundary():
    """"...Rhode Island. Corresponding-author..." is one name, not five words."""
    line = "Schellenberg, Jouaneh, Byers, all University of Rhode Island. Corresponding-author TBD"
    assert _unreviewed(line) == []


def test_quoted_and_line_wrapped_names_still_resolve():
    assert _unreviewed("Confirm README still carries 'University of Rhode Island'") == []
    # A line-based scan sees a wrapped name truncated; a prefix of an allowed
    # name is not an unreviewed institution.
    assert _unreviewed("OpenSSF Passing; MIT; University of Rhode") == []


def test_the_live_repository_passes_its_own_gate():
    """The tracked tree must be clean, or the gate is not actually enforced."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=str(_SCRIPT.resolve().parents[1]),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
