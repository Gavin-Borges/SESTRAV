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


def test_tools_exclusion_is_a_path_prefix_not_a_bare_directory_name():
    """A bare directory name unscans that name at EVERY depth.

    "tools" sat in EXCLUDED_DIR_PARTS to skip the vendored competitor source
    under `_local/tools/`, whose own READMEs name their own institutions
    correctly. Because the test was a bare-name set intersection, it also
    unscanned the tracked top-level `tools/`, which holds 8 tracked files the
    CI gate is meant to cover and never opened.

    `.claude/tools/` was unscanned too, but that directory is gitignored and
    holds no tracked file, so it is reachable only under `--all`. It is
    asserted below as a path-predicate property, NOT as recovered CI coverage.

    Both directions are load-bearing. The vendored tree must stay excluded,
    including the backslash-separated form that `--all` produces on Windows
    via os.walk, and the tracked tree must now be scanned.
    """
    assert "tools" not in mod.EXCLUDED_DIR_PARTS

    # Still excluded: the vendored competitor source the entry was added for.
    assert mod.should_scan("_local/tools/DeepImmuno/README.md") is False
    assert mod.should_scan("_local\\tools\\DeepImmuno\\README.md") is False
    assert mod.should_scan("./_local/tools/BigMHC/setup.py") is False

    # The directory NODE itself, carrying no trailing slash. This is the form
    # the os.walk prune tests, and it is the only thing the equality half of
    # is_excluded_prefix covers, so without these two the clause could be
    # deleted with every other assertion here still passing. Losing it would
    # not change any verdict, but the walk would descend into roughly 1840
    # vendored files and discard them one at a time instead of pruning.
    assert mod.is_excluded_prefix("_local/tools") is True
    assert mod.is_excluded_prefix("./_local/tools") is True
    assert mod.is_excluded_prefix("_local\\tools") is True

    # No longer excluded: the tracked tree the bare name also caught.
    assert mod.should_scan("tools/check_hash_pins.py") is True
    assert mod.should_scan("tools/coverage_subprocess/sitecustomize.py") is True

    # Gitignored, so this one is reachable only under --all. Asserted as a
    # path-predicate property, not as coverage the CI gate regains.
    assert mod.should_scan(".claude/tools/sync_agent_rules.py") is True

    # A directory merely named "tools" elsewhere is not the vendored one.
    assert mod.should_scan("docs/tools/overview.md") is True


def test_name_is_not_matched_across_a_sentence_boundary():
    """"...Rhode Island. Corresponding-author..." is one name, not five words."""
    line = "Schellenberg, Jouaneh, Byers, all University of Rhode Island. Corresponding-author TBD"
    assert _unreviewed(line) == []


def test_quoted_and_line_wrapped_names_still_resolve():
    assert _unreviewed("Confirm README still carries 'University of Rhode Island'") == []
    # A line-based scan sees a wrapped name truncated; a prefix of an allowed
    # name is not an unreviewed institution.
    assert _unreviewed("OpenSSF Passing; MIT; University of Rhode") == []


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


def _run_all(cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--all"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def test_nested_checkout_detection_is_structural_not_a_name_glob():
    """A worktree is detected by carrying .git, never by being called wt_*.

    Keying on a name prefix would match a NAME rather than the property
    (.claude/rules/git-instruments-diffs.md rule 14), so a checkout parked
    under any other name would keep flooding the report while a plain
    directory that happened to be named wt_something would go unscanned.

    The scan root itself is the load-bearing negative case: `os.walk(".")`
    starts at the repository root, which of course carries .git. Treating it
    as nested would unscan every tracked file and make --all report LESS than
    the default mode.
    """
    # A `git worktree add` checkout carries .git as a FILE, a clone as a DIR.
    assert mod.is_nested_checkout("./_local/wt_x", [], [".git", "a.md"]) is True
    assert mod.is_nested_checkout("./_local/clone", [".git"], ["a.md"]) is True

    # The scan root is never nested, whichever form it carries.
    assert mod.is_nested_checkout(".", [".git"], []) is False
    assert mod.is_nested_checkout(".", [], [".git"]) is False

    # A directory that merely looks like a worktree by name is not one.
    assert mod.is_nested_checkout("./_local/wt_stale", ["docs"], ["notes.md"]) is False


def test_untracked_file_in_a_nested_checkout_is_still_scanned_and_still_fails(tmp_path):
    """THE anti-regression test. Excluding worktrees wholesale was rejected.

    Wholesale exclusion measured identically on this workstation, but it
    blinds every untracked file inside those checkouts - 858 of them when
    this was measured, 815 under a checkout's own nested `_local/`. Untracked
    files under `_local/` are the unpublished outreach and manuscript copy
    that `--all` exists to read, so blinding them would remove the mode's
    entire reason for existing while leaving it apparently healthier.

    So: the file git TRACKS in the nested checkout is skipped (CI and
    pre-push already scan that branch's tracked content), and the untracked
    one beside it is still reported.
    """
    _git("init", "-q", cwd=tmp_path)
    (tmp_path / "outer_tracked.md").write_text("coursework at NC State\n")
    _git("add", "outer_tracked.md", cwd=tmp_path)

    nested = tmp_path / "nested"
    nested.mkdir()
    _git("init", "-q", cwd=nested)
    (nested / "nested_tracked.md").write_text("coursework at NC State\n")
    (nested / "nested_untracked.md").write_text("coursework at NC State\n")
    _git("add", "nested_tracked.md", cwd=nested)

    result = _run_all(tmp_path)
    out = result.stdout.replace("\\", "/")

    assert result.returncode == 1, out + result.stderr
    # Skipped: tracked inside the nested checkout.
    assert "nested/nested_tracked.md" not in out
    # Kept: untracked inside the nested checkout. This is the whole point.
    assert "nested/nested_untracked.md" in out
    # Kept: the SCAN ROOT's own tracked files. --all stays a superset of the
    # default mode; the root is not itself a "nested" checkout.
    assert "outer_tracked.md" in out


def test_a_broken_nested_gitdir_falls_back_to_scanning_everything(tmp_path):
    """git ls-files failing must fail toward MORE scanning, never less.

    A stale `git worktree` whose gitdir has been deleted leaves a .git
    pointer file that resolves to nothing. Silently treating that as "no
    tracked files here" is the safe direction; treating it as "skip the
    directory" would be a hole opened by an error path.
    """
    nested = tmp_path / "broken"
    nested.mkdir()
    (nested / ".git").write_text("gitdir: /nonexistent/path/to/gitdir\n")
    (nested / "note.md").write_text("coursework at NC State\n")

    assert mod.nested_tracked_paths(str(nested)) == set()

    result = _run_all(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "note.md" in result.stdout.replace("\\", "/")


def test_findings_are_grouped_one_line_per_file_and_name(tmp_path):
    """Grouping is presentation only: it must lose no occurrence and no name.

    The D35 retraction row puts the same name on ONE line five times, and
    that line is replicated across every checkout, so per-occurrence
    reporting turned a single reviewed sentence into a hundred-odd errors.
    Per (file, name) is the readable unit. The occurrence count is still
    printed so nothing is hidden, and a SECOND name in the same file must
    still get its own line.
    """
    doc = tmp_path / "notes.md"
    doc.write_text(
        "coursework at NC State\n"
        "still at NC State\n"
        "NC State and NC State on one line\n"
        "also Affiliation: Stanford University\n"
    )

    result = _run_all(tmp_path)
    out = result.stdout
    errors = [ln for ln in out.splitlines() if ln.startswith("ERROR ")]

    assert result.returncode == 1, out + result.stderr
    # Two names in one file: two lines, not five.
    assert len(errors) == 2, errors
    nc = next(ln for ln in errors if "NC State" in ln)
    assert "4 occurrence(s)" in nc
    assert "line(s) 1, 2, 3" in nc
    assert any("Stanford University" in ln and "1 occurrence(s)" in ln for ln in errors)
    # The occurrence total survives the grouping and is reported alongside it.
    assert "5 unreviewed institution reference(s) in 2 (file, name) group(s)." in out


def test_the_live_repository_passes_its_own_gate():
    """The tracked tree must be clean, or the gate is not actually enforced."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=str(_SCRIPT.resolve().parents[1]),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
