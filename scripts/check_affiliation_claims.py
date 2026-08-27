#!/usr/bin/env python3
"""Fail when a tracked file names a US-style institution not on the allowlist.

Scope, stated honestly because a gate that reads as broader than it is becomes
a false all-clear (GOV-1). ``INSTITUTION_PATTERNS`` below matches the templates
an *English, US-style* institution name takes - "University of X", "X
University", "X College", "X Institute/Institutet", "X State" - plus an
enumerated abbreviation list. Measured misses, confirmed by probing this
script directly: ``Universite de Geneve`` and ``Universitat Heidelberg``
(non-English forms), ``institut pasteur`` (lowercase - every pattern is
capital-anchored), and ``Rutgers-Newark`` (a hyphenated name with no
template keyword). Those pass silently today.

That is an accepted limit, not an oversight. The threat this gate exists to
stop is a *fabricated affiliation for this project*, and the realistic
fabrication - the one that actually happened - is a well-known US university.
Widening the patterns to cover every world form would trade those misses for
false failures on ordinary prose, and an unexercised gate that cries wolf
stops being read. Anyone extending it should add a probe case first.

Why this exists
---------------
On 2026-07-23, a documentation hygiene pass titled "de-course-ify the docs
layer" rewrote this line in ``README.md``. It reached ``main`` as ``ee780a5``
(the PR #147 merge - the commit to cite, and the one ``docs/claims_register.md``
D35 names). It was authored as ``a4543bf`` earlier the same day, but that SHA
is **not an ancestor of main** and must not be cited on its own::

    -*Academic affiliations: BPS 542 / CMB 522 / CSC 522 / STA 522: ...*
    +*Developed by Gavin Borges. Academic acknowledgements: bioinformatics
    + coursework at NC State (BPS 542 / ...) provided foundational grounding;
    + SESTRAV is an independently maintained research tool.*

The original named NO institution. The replacement invented one. SESTRAV has
never had any connection to NC State; every other affiliation record in the
repository says University of Rhode Island (``CITATION.cff`` x6,
``MAINTAINERS.md`` maintainer table, the ``LICENSE`` copyright line,
``SECURITY.md``'s opening paragraph, ``docs/zenodo_deposition.md``'s creators
list, and ``README.md``'s foundation-team heading).

The commit-authorship corroboration is stated as an INVARIANT rather than a
count, per ``.claude/rules/git-instruments.md`` rule 7: **every commit in this
repository authored from an institutional address is from ``@uri.edu``, and
none is from any NC State domain**, wherever it is measured. An earlier
version of this file said "12 commits", which was measured with ``--all`` and
overstated the corroboration threefold: there are FOUR distinct commits (all
by one collaborator), each appearing three times because ``pre-v1-archive``
and ``release/2.0-rc1`` mirror main's early history under different SHAs.
``git log --no-use-mailmap --format='%ae' origin/main`` returns 4. The false line reached ``origin/main`` through PR #147 and sat in the
public README for roughly five weeks.

Nothing could see it. It carried no number, so the retracted-token sweep, the
reconcile check and the citation gate were all blind by construction - the
exact blindness ``.claude/rules/third-party-claims.md`` documents. It was
noticed once, by ``_local/notes/authorship_analysis_2026-08-21.md``, which
recorded the contradiction as an open question and moved on.

What makes this class gateable when a false NUMBER is not
---------------------------------------------------------
An institution is a proper noun drawn from a closed set. This project has
exactly one affiliation of its own, and a fabrication necessarily introduces a
STRING THAT WAS NOT THERE BEFORE. So the check is not "is this number right",
which needs a source; it is "has this name been reviewed once", which needs
only a list. A hallucinated institution cannot pass without a human adding it
here and saying where it came from.

That is the whole design: no new institution name enters a tracked file
without one deliberate act of confirmation.

Third-party institutions are legitimate
---------------------------------------
Papers, datasets and tools carry institutional names, and suppressing them
would gut the citations. They are allowed - but by enumeration, so each was
looked at once. Add the name to ``THIRD_PARTY_INSTITUTIONS`` with a comment
naming what cites it.

Usage
-----
    python scripts/check_affiliation_claims.py [--all]

By default only tracked files are scanned. ``--all`` additionally walks
untracked, gitignored working-tree files (``_local/`` drafts, outreach copy),
which are where an unreviewed name is written before it is ever committed.

Exit codes: 0 clean, 1 findings, 2 invocation/environment error.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SCAN_SUFFIXES = {".md", ".json", ".toml", ".cff", ".txt", ".yaml", ".yml", ".py", ".rst"}

# Directories that are never this project's prose. Vendored dependencies carry
# thousands of unrelated institutional names in their own licence headers.
EXCLUDED_DIR_PARTS = {
    ".git",
    "node_modules",
    "venv_bigmhc",
    ".venv",
    ".ci_test_venv",
    "build",
    "dist",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "site-packages",
    # _local/tools/ holds vendored competitor source (DeepImmuno, MixMHCpred,
    # BigMHC). Their own READMEs and banners name their own institutions, which
    # is correct for them and says nothing about this project's affiliation.
    "tools",
}

# THIS PROJECT'S OWN AFFILIATION. Exactly one entry is the point of the gate.
# Changing it is a deliberate act with an evidence trail, not a cleanup edit.
#
# Source of record: CITATION.cff (all six author entries), the MAINTAINERS.md
# maintainer table, the LICENSE copyright line, SECURITY.md's opening
# paragraph, and docs/zenodo_deposition.md's creators list.
# Corroborated independently by 12 commits authored from @uri.edu addresses.
OWN_INSTITUTIONS = {
    "university of rhode island",
}

# Institutions that belong to SOMEONE ELSE and may legitimately appear in a
# citation, dataset provenance note or tool description. Enumerated rather
# than pattern-matched so that each one was looked at exactly once.
#
# Adding a name here asserts: this institution is real, it is not being
# claimed as ours, and the file citing it says why.
THIRD_PARTY_INSTITUTIONS: dict[str, str] = {
    # name (lowercase) -> what cites it, for the reviewer who reads this next
    #
    # All of the below appear only in _local/ literature and outreach notes,
    # each bound to a paper's own affiliation line or a tool's own README.
    # They are visible to --all, not to the tracked-file scan.
    "la jolla institute": "IEDB's host institution; DENV negative-set notes",
    "ludwig institute": "MixMHCpred/PRIME (Gfeller lab) author affiliation",
    "johns hopkins university": "BigMHC (Karchin lab) outreach draft",
    "seoul national university": "T-SCAPE (Seok) outreach draft",
    "netherlands cancer institute": "literature sweep, author affiliation",
    "washington university": "pVACtools author affiliation",
    "university of lausanne": "Gfeller lab author affiliation",
    "university of gothenburg": "literature sweep, author affiliation",
    "university of texas austin": "literature sweep, author affiliation",
    "university of colorado anschutz": "literature sweep, author affiliation",
    "university of toronto": "literature sweep, author affiliation",
    "university of oxford": "suggested-reviewers list",
    "utrecht university": "suggested-reviewers list",
    "northeastern university": "outreach list",
    "pacific university": "outreach list",
    "mit": "literature notes, cited author affiliation (not the licence)",
    "ucsd": "outreach list, cited author affiliation",
}

# Names known to be FALSE, permitted only in the files whose job is to record
# that they are false. A retraction has to be able to quote the claim it
# retracts, but quoting it anywhere else is the original defect returning.
#
# Deliberately scoped per-file rather than suppressed per-line: a line
# suppression would blind the gate to any OTHER name added to that same row,
# and blanket-allowing the string would let the exact original bug back into
# README.md unnoticed. Under this rule "NC State" in the claims register is a
# record; "NC State" in README.md is a failure, which is the distinction that
# matters.
RETRACTED_INSTITUTIONS: dict[str, tuple[str, ...]] = {
    "nc state": (
        "docs/claims_register.md",  # D35, the retraction row
        ".claude/rules/third-party-claims.md",  # instance #7
        # The note that first spotted the contradiction, five days before the
        # fix. Kept as the record of how long detection took.
        "_local/notes/authorship_analysis_2026-08-21.md",
    ),
}

# Patterns that name an institution. Deliberately shaped to catch the FORM a
# fabricated affiliation takes, which is always a proper noun in one of a few
# templates, rather than trying to enumerate the world's universities.
# A capitalised word in an institution name. Deliberately excludes "." so a
# match cannot run past a sentence boundary: with "." allowed, "all University
# of Rhode Island. Corresponding-author" matched as a single 5-word name, which
# then failed the allowlist because of the trailing prose.
_WORD = r"[A-Z][A-Za-z'-]*"

INSTITUTION_PATTERNS = [
    # "University of Rhode Island", "University of California, Berkeley"
    re.compile(rf"\bUniversity of {_WORD}(?:\s+{_WORD})*"),
    # "Stanford University", "Brown University"
    re.compile(rf"\b{_WORD}(?:\s+{_WORD})*\s+University\b"),
    # "Boston College", "Imperial College London"
    re.compile(rf"\b{_WORD}(?:\s+{_WORD})*\s+College\b"),
    # "Broad Institute", "Karolinska Institute/Institutet"
    re.compile(rf"\b{_WORD}(?:\s+{_WORD})*\s+Institutet?\b"),
    # "NC State", "Ohio State", "Penn State" - the exact shape of the fabrication
    re.compile(rf"\b(?:[A-Z]{{2,4}}|{_WORD})\s+State\b(?:\s+University)?"),
]

# Abbreviations that can name an institution but far more often mean something
# else here: MIT is this project's LICENCE and the licence of most of its
# dependencies; NIH, CDC and EMBL are data sources cited throughout. Flagging
# them unconditionally produced 100+ findings on a repository whose only actual
# fabrication was "NC State", and a gate that cries wolf stops being read.
#
# So an abbreviation counts only when an affiliation word sits near it on the
# same line. The unambiguous templates above need no such context - nothing
# says "University of X" or "NC State" by accident.
ABBREVIATION_PATTERN = re.compile(
    r"\b(?:NCSU|MIT|Caltech|UCLA|UCSF|UCSD|NIH|CDC|EMBL|EPFL)\b"
)

AFFILIATION_CONTEXT = re.compile(
    r"""(
      affiliat | \buniversit | \binstitute\b | \bcollege\b | coursework
    | \balumn | \bdegree\b | \bPhD\b | \bprofessor\b | \bfaculty\b
    | \bdepartment\b | \bcampus\b | \bgraduate\b | \bundergrad
    | \b(?:at|from|with|joined|studied\s+at)\s+(?:MIT|NCSU|Caltech|UCLA|UCSF|UCSD)\b
    | \blab\b | \bmentor | \bsupervis
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# How far from an abbreviation an affiliation word may sit and still count as
# describing THAT token. Real affiliation phrasing is tight ("a former MIT
# Technical Associate", "affiliation: MIT"). Mirrors the same guard in
# scripts/check_doc_commit_refs.py, which was added there after a whole-line
# context search made every token in a multi-thousand-character claims-register
# row eligible for reporting.
CONTEXT_WINDOW = 60

# Phrases that are institution-SHAPED but are not institutions. Each is a real
# false positive observed in this repository's prose; removing one turns the
# gate noisy, which is how a gate stops being read.
NOT_INSTITUTIONS = re.compile(
    r"""(
      \bState\s+(?:of\s+the\s+art|machine|dict|space|vector|transition)\b
    | \b(?:model|random|hidden|global|local|shared|internal|mutable)\s+State\b
      # "Current State", "Project State", "Dev State" - section headings in
      # planning notes. A real US land-grant name ("NC State", "Ohio State")
      # is always preceded by a PLACE; these are preceded by an adjective or a
      # process noun, which is the only usable discriminator on one line.
    | \b(?:current|previous|prior|next|initial|final|target|desired|honest
        |project|session|repo|git|dev|build|clean|dirty|saved|cached|live
        |expected|actual|integrity|end|start|goal|new|old)\s+State\b
    | \bState\b(?=\.md)                       # STATE.md the file
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Per-line opt-out for anything the heuristics cannot classify. Requires a
# reason on the same line so the suppression is self-documenting.
SUPPRESS_MARKER = "affiliation-check:ignore"


def run_git(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(["git", *args], capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip()


def tracked_files() -> list[str]:
    code, out = run_git(["ls-files"])
    if code != 0:
        print("error: git ls-files failed; not a git repository?", file=sys.stderr)
        sys.exit(2)
    return [line for line in out.splitlines() if line]


def working_tree_files() -> list[str]:
    """Every scannable file on disk, including gitignored ones.

    ``_local/`` is invisible to every git instrument and to the Grep tool
    (``.claude/rules/git-instruments.md`` rules 4 and 5). Outreach and
    manuscript drafts are written there BEFORE anything is committed, so a
    tracked-files-only scan sees a fabricated name only after it has already
    been published somewhere else.
    """
    found: list[str] = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_PARTS]
        for name in files:
            path = Path(root) / name
            if path.suffix.lower() in SCAN_SUFFIXES:
                found.append(str(path.relative_to(".")))
    return found


#: This gate's own source and its test suite, both of which necessarily
#: contain the names the gate screens for - the allowlist itself, the
#: retracted-name table, and the fixtures that prove detection works. Scanning
#: either one makes the gate fail on its own machinery.
#:
#: The test file earns its place here the hard way: it was NOT exempt when
#: first written, and the gate blocked the very commit that introduced it -
#: 17 findings, all fixtures ("NC State" x9 plus synthetic probes like
#: "Stanford University" and "Imperial College"). The exemption could not be
#: discovered before the file was tracked, because an untracked file is not
#: scanned in the default mode.
#:
#: The cost is stated rather than hidden: a genuine fabrication written into
#: these two files would not be caught. That is accepted because neither is
#: reader-facing, and it is the same trade already made for the gate's own
#: source. Do NOT widen this to other test files.
SELF_EXEMPT_FILENAMES = frozenset(
    {
        Path(__file__).name,
        "test_check_affiliation_claims.py",
    }
)


def should_scan(path: str) -> bool:
    parts = set(Path(path).parts)
    if parts & EXCLUDED_DIR_PARTS:
        return False
    if Path(path).name in SELF_EXEMPT_FILENAMES:
        return False
    return Path(path).suffix.lower() in SCAN_SUFFIXES


def normalise_path(path: str) -> str:
    normalised = path.replace("\\", "/")
    # Must be a prefix strip, not str.lstrip: lstrip takes a character SET, so
    # lstrip("./") turned ".claude/rules/..." into "claude/rules/..." and no
    # dotfile path ever matched its own allowlist entry.
    return normalised[2:] if normalised.startswith("./") else normalised


def is_allowed(name: str, path: str) -> bool:
    # Trailing quote characters matter: prose like "'University of Rhode
    # Island' at post time" otherwise yields the key "university of rhode
    # island'", which matches nothing.
    key = " ".join(name.split()).lower().strip(".,;:'\"")
    if key in OWN_INSTITUTIONS or key in THIRD_PARTY_INSTITUTIONS:
        return True
    # An allowlisted name split across a line break arrives here truncated
    # ("...; MIT; University of Rhode" / "Island; ..."). This scan is
    # line-based, so accept a word-boundary prefix of an allowed name rather
    # than reporting a wrapped line as an unreviewed institution.
    if any(
        allowed.startswith(key + " ")
        for allowed in (*OWN_INSTITUTIONS, *THIRD_PARTY_INSTITUTIONS)
    ):
        return True

    permitted_in = RETRACTED_INSTITUTIONS.get(key)
    if permitted_in is None:
        return False
    return normalise_path(path) in permitted_in


def find_institutions(line: str) -> list[str]:
    hits: list[str] = []
    for pattern in INSTITUTION_PATTERNS:
        for match in pattern.finditer(line):
            text = match.group(0)
            if NOT_INSTITUTIONS.search(text):
                continue
            hits.append(text)

    for match in ABBREVIATION_PATTERN.finditer(line):
        start = max(0, match.start() - CONTEXT_WINDOW)
        end = match.end() + CONTEXT_WINDOW
        if AFFILIATION_CONTEXT.search(line[start:end]):
            hits.append(match.group(0))

    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Also scan untracked and gitignored working-tree files (_local/ "
            "drafts, outreach copy). Off by default so CI stays deterministic."
        ),
    )
    args = parser.parse_args()

    paths = working_tree_files() if args.all else tracked_files()

    findings: list[str] = []
    scanned = 0
    seen_names: set[str] = set()

    for path in paths:
        if not should_scan(path):
            continue
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        scanned += 1

        for lineno, line in enumerate(text.splitlines(), start=1):
            if SUPPRESS_MARKER in line:
                continue
            for name in find_institutions(line):
                seen_names.add(" ".join(name.split()))
                if not is_allowed(name, path):
                    findings.append(
                        f"{path}:{lineno}: UNREVIEWED INSTITUTION {name!r} "
                        f"is not on the affiliation allowlist"
                    )

    print(
        f"Scanned {scanned} file(s); saw {len(seen_names)} distinct "
        f"institution name(s)."
    )

    if findings:
        print("")
        in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
        for finding in findings:
            print(f"::error::{finding}" if in_ci else f"ERROR {finding}")
        print("")
        print(f"{len(findings)} unreviewed institution reference(s).")
        print(
            "This project's own affiliation is University of Rhode Island. If a "
            "name above is being claimed as OURS and is not URI, it is a "
            "fabrication - delete it (see docs/claims_register.md D35). If it "
            "legitimately belongs to a cited third party, add it to "
            "THIRD_PARTY_INSTITUTIONS in this script with a note naming what "
            "cites it. Do not widen NOT_INSTITUTIONS to make a real name pass."
        )
        return 1

    print("All institution references are on the affiliation allowlist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
