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
import os
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


def test_directory_exemption_requires_a_trailing_slash_and_stops_at_it():
    """Prefix entries are opt-in per entry, and must not leak past the slash.

    `_local/state/` is exempt as a DIRECTORY because its filenames are dated:
    every session that discusses D35 creates a new one, so an exact-path list
    would leave this gate permanently red, and a permanently red gate is
    ignored. The narrowness is the load-bearing half. Without the trailing
    slash a plain `str.startswith` would also exempt `_local/statement...`,
    silently widening a security allowlist by string coincidence.
    """
    line = 'the earlier draft said "coursework at NC State", now retracted'

    # Dated and nested names nobody can enumerate in advance.
    assert _unreviewed(line, "_local/state/session_plan_2026-12-31.md") == []
    assert _unreviewed(line, "_local/state/nested/deep.md") == []
    # The exemption stops at the slash: a sibling merely sharing the prefix
    # characters is NOT beneath the directory and stays gated.
    assert _unreviewed(line, "_local/statement_of_work.md") == ["NC State"]
    # An exact entry stays exact: no implied directory, no implied suffix.
    assert _unreviewed(line, "STATE.md") == []
    assert _unreviewed(line, "STATE.md.bak") == ["NC State"]


def test_outreach_drafts_are_never_blanket_exempt():
    """`_local/drafts/` is where copy is written that cannot be edited later.

    Outreach publishes to places with no edit button, so a verbatim D35
    recurrence there is the worst case this gate exists for. Only four exact
    files are exempt, two in each of two dated, frozen packets, and only
    because they are snapshots of records that document the retraction. A
    packet regenerated under a new date trips the gate again and is exempted
    only after every occurrence in it has been read; the 2026-09-09 packet is
    that mechanism having fired once. The assertions below cover the
    2026-09-06 packet and a future date, not the 2026-09-09 pair.
    """
    line = 'SESTRAV grew out of coursework at NC State'

    assert _unreviewed(line, "_local/drafts/linkedin_post_3.md") == ["NC State"]
    assert _unreviewed(line, "_local/drafts/SESTRAV_manuscript_draft.md") == ["NC State"]
    # The frozen packet snapshots that legitimately carry the retraction row.
    packet = "_local/drafts/mountain_view_packet_2026-09-06"
    assert _unreviewed(line, f"{packet}/07_claims_register.md") == []
    assert _unreviewed(line, f"{packet}/11_PRIVATE_brain_map.md") == []
    # A later packet is a new directory and is deliberately not carried over.
    assert _unreviewed(
        line, "_local/drafts/mountain_view_packet_2026-10-01/07_claims_register.md"
    ) == ["NC State"]


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
    """Run the gate's ``--all`` mode with the output format pinned.

    The script prints GitHub's ``::error::`` annotation form when
    ``GITHUB_ACTIONS=true`` and a plain ``ERROR `` prefix otherwise, so a test
    asserting on the prefix passes locally and fails on the runner. Dropping
    the variable makes the format a property of the test rather than of where
    it happens to run.
    """
    env = dict(os.environ)
    env.pop("GITHUB_ACTIONS", None)
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--all"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
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


# ---------------------------------------------------------------------------
# Every RETRACTED_INSTITUTIONS exemption must carry its OWN note.
#
# Guards the shape of incident #10 in .claude/rules/third-party-claims-cases.md:
# PR #448 added "STATE.md" to this allowlist under a comment reading "The two
# entries below are exact paths into a dated, frozen packet" - with FOUR entries
# following it. An unexplained allowlist entry silently widens a security gate.
#
# The rule is deliberately PER-ENTRY. An earlier draft let one comment cover two
# consecutive entries, which accepted an unjustified entry inserted directly
# beneath an annotated one - the exact insertion position incident #10 used.
# ---------------------------------------------------------------------------

import ast as _ast
import pathlib as _pathlib
import re as _re

_GATE_SOURCE = _pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_affiliation_claims.py"


def _retracted_allowlist_entries(tree):
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, _ast.AnnAssign)
        and isinstance(node.target, _ast.Name)
        and node.target.id == "RETRACTED_INSTITUTIONS"
    )
    entries = []
    for paths in assignment.value.values:
        for path in paths.elts:
            entries.append((path.value, path.lineno))
    return entries


def _has_own_note(lines, lineno):
    """A note on the entry's own line, or on the line directly above it.

    Requires a word character after the '#', so a bare '#' does not qualify, and
    splits on the closing quote so a '#' inside the path string does not either.
    """
    same_line_tail = lines[lineno - 1].split('",')[-1]
    if _re.search(r"#\s*\w", same_line_tail):
        return True
    previous = lines[lineno - 2] if lineno >= 2 else ""
    return bool(_re.match(r"\s*#\s*\w", previous))


def test_every_retracted_allowlist_entry_carries_its_own_note():
    source = _GATE_SOURCE.read_text(encoding="utf-8")
    lines = source.splitlines()
    entries = _retracted_allowlist_entries(_ast.parse(source, filename=str(_GATE_SOURCE)))
    assert entries, "RETRACTED_INSTITUTIONS parsed to zero entries; the test is vacuous"

    missing = [
        f"{path} (line {lineno})" for path, lineno in entries if not _has_own_note(lines, lineno)
    ]
    assert not missing, (
        "Allowlist entries with no note of their own:\n"
        + "\n".join(missing)
        + "\n\nEvery exemption must say why it is exempt. A comment above a "
        "NEIGHBOURING entry does not carry over."
    )


def test_a_bare_hash_is_not_a_note():
    assert not _has_own_note(['    "docs/x.md",  #'], 1)
    assert _has_own_note(['    "docs/x.md",  # frozen packet'], 1)


def test_a_hash_inside_the_path_is_not_a_note():
    assert not _has_own_note(['    "docs/a#b.md",'], 1)


# ---------------------------------------------------------------------------
# Verbatim copies of tracked files.
#
# A `git worktree` checkout carries .git and the tests above pin its handling.
# A plain directory COPY of this repository carries none. Measured 2026-09-20:
# a 737-file copy sat under `_local/tmp/`, all 663 tracked files byte-identical
# once line endings are normalised, and its copy of `docs/claims_register.md` -
# the D35 retraction row, which has to QUOTE the fabricated institution in
# order to retract it - was reported and BLOCKED A PUSH. The tracked original
# is allowlisted by exact path; the copy sits at a different path.
#
# The predicate is CONTENT IDENTITY per file, never a directory test, because
# treating a directory as a checkout means its files are not scanned at all and
# that is a blind spot in a security gate. The fixture below therefore gives
# the copy a pyproject.toml and a src/ tree: anything keying on "this looks
# like a checkout" would exempt it, and must not.
#
# `test_a_copy_edited_into_a_fabrication_is_still_reported` is the
# load-bearing one. Without it this fix is indistinguishable from simply
# widening the blind spot.
# ---------------------------------------------------------------------------

# Shaped like the D35 row: it quotes the fabricated name in order to retract
# it, which is why the tracked path carries an allowlist entry.
RETRACTION_ROW = (
    '| D35 | README.md read "coursework at NC State" | RETRACTED: this '
    "project's affiliation is the University of Rhode Island |"
)

# The same row edited into an ASSERTION of the fabrication. The gate cannot
# tell a record from a claim - that distinction is carried by the PATH - which
# is exactly why an inherited allowance has to stop at the first changed byte.
FABRICATION_ROW = (
    "| D35 | SESTRAV grew out of coursework at NC State, which provided the "
    "foundational grounding for this project |"
)


def _run_default(cwd: Path) -> subprocess.CompletedProcess[str]:
    """The gate's DEFAULT mode, the one CI and pre-push Check 3 run."""
    env = dict(os.environ)
    env.pop("GITHUB_ACTIONS", None)
    return subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
    )


def _repo_with_a_plain_directory_copy(tmp_path, copied_row, newline="\n"):
    """A tracked, allowlisted claims register plus a plain COPY of it.

    The copy carries no .git - that is the whole gap - and carries the markers
    a looser predicate might key on.
    """
    _git("init", "-q", cwd=tmp_path)
    tracked = tmp_path / "docs" / "claims_register.md"
    tracked.parent.mkdir(parents=True)
    tracked.write_bytes((RETRACTION_ROW + "\n").encode("utf-8"))
    _git("add", "docs/claims_register.md", cwd=tmp_path)

    copy_root = tmp_path / "_local" / "tmp" / "sestrav_x2_d064dbc"
    (copy_root / "docs").mkdir(parents=True)
    (copy_root / "src").mkdir()
    (copy_root / "pyproject.toml").write_bytes(b'[project]\nname = "sestrav"\n')
    (copy_root / "docs" / "claims_register.md").write_bytes(
        (copied_row + "\n").replace("\n", newline).encode("utf-8")
    )
    return copy_root


def test_a_verbatim_directory_copy_of_a_tracked_file_is_suppressed(tmp_path):
    """The reported class: a .git-less copy of an allowlisted tracked file."""
    copy_root = _repo_with_a_plain_directory_copy(tmp_path, RETRACTION_ROW)

    # Pinned so the fix cannot quietly become "widen nested-checkout detection".
    assert (
        mod.is_nested_checkout(str(copy_root), ["docs", "src"], ["pyproject.toml"])
        is False
    )

    result = _run_all(tmp_path)
    out = result.stdout.replace("\\", "/")
    assert result.returncode == 0, out + result.stderr
    assert "sestrav_x2_d064dbc" not in out


def test_a_copy_edited_into_a_fabrication_is_still_reported(tmp_path):
    """THE load-bearing test. A copy is not a licence to fabricate.

    The directory is indistinguishable from the suppressed one above by every
    structural signal - same name, same pyproject.toml, same src/ tree, same
    path beneath _local/ - and differs only in the CONTENT of the flagged
    file, which now asserts the fabrication instead of retracting it. A
    directory-shaped predicate, or a marker-file heuristic, would report
    nothing here. That is the blind spot this test exists to refuse.
    """
    _repo_with_a_plain_directory_copy(tmp_path, FABRICATION_ROW)

    result = _run_all(tmp_path)
    out = result.stdout.replace("\\", "/")
    errors = [ln for ln in out.splitlines() if ln.startswith("ERROR ")]

    assert result.returncode == 1, out + result.stderr
    assert "_local/tmp/sestrav_x2_d064dbc/docs/claims_register.md" in out
    assert "'NC State'" in out
    # The tracked original stays clean: exactly the copy is reported.
    assert len(errors) == 1, errors


def test_one_trailing_space_revokes_the_inherited_allowance(tmp_path):
    """Identity is byte-for-byte, so the smallest possible edit revokes it.

    A predicate that tolerated whitespace, or compared only the flagged LINE,
    would let an edited copy keep its original's allowance.
    """
    _repo_with_a_plain_directory_copy(tmp_path, RETRACTION_ROW + " ")

    result = _run_all(tmp_path)
    out = result.stdout.replace("\\", "/")
    assert result.returncode == 1, out + result.stderr
    assert "_local/tmp/sestrav_x2_d064dbc/docs/claims_register.md" in out


def test_line_endings_are_the_only_difference_a_copy_may_carry(tmp_path):
    """A CRLF copy of an LF original is still the same content.

    This is not cosmetic: the measured 737-file copy differed from its
    originals in line endings alone, so a raw byte comparison would have
    suppressed nothing and left the gate blocking pushes.
    """
    _repo_with_a_plain_directory_copy(tmp_path, RETRACTION_ROW, newline="\r\n")

    result = _run_all(tmp_path)
    out = result.stdout.replace("\\", "/")
    assert result.returncode == 0, out + result.stderr
    assert "sestrav_x2_d064dbc" not in out


def test_a_tracked_duplicate_is_reported_in_BOTH_modes(tmp_path):
    """--all must stay a superset of the default mode over the tracked set.

    `docs/register_copy.md` is tracked and byte-identical to the allowlisted
    `docs/claims_register.md`, but nothing allows IT to carry the retracted
    name, so the default mode - the CI gate and pre-push Check 3 - reports it.
    An "any twin is allowed" predicate would silence it under --all while the
    default mode stayed red, which is why the predicate requires EVERY twin:
    a tracked file is always a twin of itself.
    """
    _git("init", "-q", cwd=tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "claims_register.md").write_bytes((RETRACTION_ROW + "\n").encode("utf-8"))
    (docs / "register_copy.md").write_bytes((RETRACTION_ROW + "\n").encode("utf-8"))
    _git("add", "docs/claims_register.md", "docs/register_copy.md", cwd=tmp_path)

    default = _run_default(tmp_path)
    assert default.returncode == 1, default.stdout + default.stderr
    assert "register_copy.md" in default.stdout.replace("\\", "/")

    every = _run_all(tmp_path)
    assert every.returncode == 1, every.stdout + every.stderr
    assert "register_copy.md" in every.stdout.replace("\\", "/")


def test_content_fingerprint_folds_line_endings_and_nothing_else(tmp_path):
    lf = tmp_path / "lf.md"
    lf.write_bytes(b"a\nb\n")
    crlf = tmp_path / "crlf.md"
    crlf.write_bytes(b"a\r\nb\r\n")
    cr = tmp_path / "cr.md"
    cr.write_bytes(b"a\rb\r")
    changed = tmp_path / "changed.md"
    changed.write_bytes(b"a\nB\n")

    assert mod.content_fingerprint(str(lf)) == mod.content_fingerprint(str(crlf))
    assert mod.content_fingerprint(str(lf)) == mod.content_fingerprint(str(cr))
    assert mod.content_fingerprint(str(lf)) != mod.content_fingerprint(str(changed))
    # Unreadable suppresses nothing.
    assert mod.content_fingerprint(str(tmp_path / "absent.md")) is None


def test_an_empty_index_suppresses_nothing():
    """The default mode passes {}, and a git failure produces {}.

    Both must read as "no twin known", never as "already reviewed" - the same
    fail-toward-MORE-scanning direction as the broken-gitdir fallback.
    """
    assert mod.verbatim_tracked_twins("_local/copy/x.md", {}, {}) == ()
    assert mod.is_reviewed_verbatim_copy("NC State", "_local/copy/x.md", {}, {}) is False


def test_every_tracked_twin_must_allow_the_name_not_merely_one():
    """Pins the `all`, not `any`, directly on the predicate.

    The cache is pre-seeded so the twins are fixtures rather than files; the
    index only has to be non-empty for the lookup to be enabled.
    """
    index = {"f" * 64: ("docs/claims_register.md",)}

    allowed_only = {"_local/copy/x.md": ("docs/claims_register.md",)}
    assert (
        mod.is_reviewed_verbatim_copy("NC State", "_local/copy/x.md", index, allowed_only)
        is True
    )

    # One unreviewed carrier of the same content is enough to report the copy.
    mixed = {"_local/copy/x.md": ("docs/claims_register.md", "docs/paper.md")}
    assert (
        mod.is_reviewed_verbatim_copy("NC State", "_local/copy/x.md", index, mixed)
        is False
    )

    # A name no twin is allowed to carry is never inherited.
    assert (
        mod.is_reviewed_verbatim_copy(
            "Ohio State", "_local/copy/x.md", index, allowed_only
        )
        is False
    )
