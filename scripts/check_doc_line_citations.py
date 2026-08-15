#!/usr/bin/env python3
"""Verify that each PINNED ``path/file.py:NNN`` citation in tracked docs still
points at the content it was pinned to.

Not every citation in the tree is pinned. Historical ledgers are exempt by
design and hold most of them; the gate prints the checked and exempt counts
side by side on every check run so the coverage ratio is never implied.

Why this exists
---------------
Line-number citations rot silently. A doc cites a line for the
``GOLD_STANDARD_EPITOPES`` exclusion; an unrelated edit inserts lines upstream;
the citation now points at whatever happened to slide into that position. The
file still exists and the line still exists, so nothing complains.

(Stated generically on purpose. The real ``:555`` citation drifted to 675, a
120-line move, while the +3 shift below is a different incident; an earlier
draft of this docstring merged the two into one tidier example, which is how
composite illustrations quietly become false.)

The repository's own running count reads "tenth instance" at CHANGELOG.md:118
as of d971947, and this change fixes two more of the same class, so twelve is
the number this file is written against.

The clearest specimen is the pair 96ab220 re-anchored to lines 675 and 941/946:
its DIRECT CHILD 42de845 made a net +3 change upstream of both, moving them to
678 and 944/949. Renumbering only resets the clock. Note what that example is
and is not - by 96ab220 those two citations had already been rewritten into
prose ("line 675", "lines 941/946"), which the pattern below does not match.
That instance MOTIVATES this gate; it is not one this gate would have caught.
Coverage is `path:NNN` syntax only, and prose line references stay uncovered.

Why the obvious checks do not work
----------------------------------
An exists-plus-EOF checker finds none of the SEMANTIC instances. Measured on
this repository at d971947: 77 line citations under the pattern below, of which
0 pointed past end-of-file. Those rotten citations still name a real file and a
line that exists, which is what makes them invisible.

The quantifier is deliberately scoped. A plain existence check DOES catch a
different sub-class: 5 of those 77 name a file that does not resolve at all,
one of which this change fixes (an omitted src/ prefix). Existence checking is
necessary and insufficient, not useless.

Both numbers come from ONE counting rule - the iter_citations logic below,
with fence skipping, suppression and the "now line" redirect applied. Counting
raw regex matches instead gives 79 and 6. Quoting 77 alongside 6 would silently
mix the two, and the single citation that separates them is real: a line in
docs/security_compliance.md carrying two citations plus an annotation, which
the single-match guard drops.

Inferring the claim from prose does not work either. A prototype extracted
backticked "anchor" identifiers near each citation and checked whether they
appeared in the cited range. Sweeping the proximity window from whole-line down
to 25 characters, the flag count wandered (10, 13, 13, 11, 8, 9, 7) while the
number of citations it could actually validate collapsed (18, 11, 9, 9, 9, 5,
2). At a 40-character window it validated 5 citations and flagged 9, mostly on
backticked English words such as ``already``, ``writes`` and ``uses``. Tuning
that until the output looked clean would have produced a gate that passes
silently while checking almost nothing - a false PASS, which is the failure
mode this repository treats as most dangerous, because a false FAIL is loud and
self-correcting while a false PASS is silent.

What this does instead
----------------------
It does not guess what a citation means. It pins what the cited line CONTAINS.

``docs/line_citations.json`` records, for each citation, the current text of the
cited line(s). That snapshot is verified by a human once, when the entry is
added. Afterwards an edit that moves those lines USUALLY makes the pinned text
stop matching, and the gate fires. The MATCHING RULE is exact string
comparison: deterministic, with no heuristic and no threshold to tune.

Three things do NOT fire, all measured rather than assumed. Two are deliberate;
the third is a genuine false PASS and is recorded as such.

1. Reindentation, because pins are whitespace-collapsed. Moving a citation is
   the defect; reformatting is not. Deliberate.
2. An edit confined to the part of a line PAST character 160, because pins are
   truncated there to stay reviewable. See ``normalize``. A real blind spot,
   accepted knowingly in exchange for a diffable baseline.
3. A citation that MOVES while text identical to the pin slides into the cited
   position. Built and measured: inserting one line above a cited line whose
   neighbour is an identical ``return None`` leaves the gate green even though
   the citation now points at a different statement. This is the false-PASS
   direction, which this file elsewhere calls the more dangerous one, so it is
   named rather than left for a reader to discover. It needs the surrounding
   lines to be duplicates, which is why symbol-anchoring - citing a unique
   name instead of a line - remains the durable fix rather than this gate.

What counts as a failure
------------------------
DRIFTED      the cited line no longer contains the pinned text. This is the rot.
UNPINNED     a line citation with no baseline entry. New citations must be
             pinned (``--update``) so a human confirms what they point at.
OUT-OF-RANGE the cited line number is past the end of the target file.
MISSING      the cited file is not tracked, so a reader cannot open it.
STALE-PIN    a baseline entry whose citation no longer appears in the doc.
             Reported so the baseline cannot silently accumulate dead weight.

Deliberate exclusions
---------------------
Historical ledgers quote stale line numbers ON PURPOSE, as a record of where
something used to be. ``CHANGELOG.md`` is the archetype: its entries describe
past states and must not be rewritten to match the present. Those files are
listed in ``EXEMPT_CITING_FILES``.

That exemption is about DRIFT ONLY, and was narrowed on 2026-08-14 after five
drifted citations were found inside it BY HAND during an audit. Two checks need
no live/historical judgement and now run on the ledgers too (see
``audit_exempt_ledgers``): a cited file that does not RESOLVE, and a line PAST
END-OF-FILE, are broken for every reader whatever the intent. Measured when the
narrowing landed: 3 of 63 were MISSING, all in ``CHANGELOG.md``, two of them
created the same day by quoting an old citation inside a retraction.

Drift inside the ledgers stays uncovered, and this file will not pretend
otherwise. What replaces it is a RATCHET: the ledgers keep the citations they
already have, but the count may not grow (``exempt_ledger_citation_ceiling`` in
the baseline). An unbounded blind spot becomes a shrinking one with no
heuristic, and new references are pushed toward symbol anchors, which cannot
rot. Lowering the ceiling is free; raising it is a deliberate, reviewable act.

A line carrying a ``now line NNN`` self-annotation is also skipped.
``docs/security_compliance.md`` uses that form to preserve dated scan-time line
numbers as forensic fact while still telling a reader where the code lives
today ("2026-06-18 scan; now line 178"). The scan-time number is the record;
rewriting it would destroy the evidence.

Fenced code blocks in markdown are skipped: a line citation inside example
output is illustration, not a citation a reader is meant to follow.

``--update`` rewrites the baseline from the current tree. It is a human action:
run it, then READ THE DIFF. An entry whose pinned text changes in that diff is
either a citation you deliberately repointed or a rot you just papered over,
and only a human can tell those apart.

Usage
-----
    python scripts/check_doc_line_citations.py
    python scripts/check_doc_line_citations.py --update

Exit codes: 0 clean, 1 findings, 2 invocation/environment error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

BASELINE_PATH = "docs/line_citations.json"

SCAN_SUFFIXES = {".md", ".py", ".toml", ".json", ".yml", ".yaml", ".txt", ".cff"}

# Historical ledgers: these quote past line numbers as a deliberate record, so
# their citations cannot be pinned against HEAD without either fabricating a
# correct-looking pin or manufacturing a future false FAIL.
#
# CHANGELOG.md is the obvious case - an entry describes the tree as it stood.
#
# docs/claims_register.md is here for a measured reason, not by analogy. Its 19
# Section 1 rows are single physical lines averaging 3,351 characters (longest
# 11,547), and one row mixes live and historical citations of the SAME target.
# Row D16 is the proof: it cites docs/model_cards/rf_30feature.md:26 twice,
# once under an explicit "[Historical - ... line numbers are as they stood
# before that pass ...]" banner and once as live guidance. Line 26 held
# "- AUC-PR: **0.828**" at f515315 (verified with git show), so the historical
# citation is CORRECT AS WRITTEN, while the live one now lands on corpus
# provenance. Separating them needs sentence-scope attribution inside one
# physical line - the same proximity heuristic this gate rejects elsewhere.
#
# The cost of this exemption is stated rather than hidden: one confirmed live
# rot inside that file (the D18 row's pointer into the Phase 0 roadmap) was
# found by hand and fixed by hand, because this gate cannot see it.
EXEMPT_CITING_FILES = {
    "CHANGELOG.md",
    "docs/claims_register.md",
}

# (citing file, target as written) pairs whose target is untracked BY DESIGN.
#
# Scoped to the exact pair rather than allowed by prefix. A blanket "_local/ is
# fine" rule would let any tracked file cite anything in the permanently
# gitignored workspace without review; naming the pair keeps each one a
# deliberate, greppable decision. The integrity harness uses the same shape for
# the same reason.
EXEMPT_CITATIONS = {
    (
        "tests/test_run_loo_cross_virus_v4_results_guard.py",
        "_local/notes/tier1-mechanical-four-enumeration-2026-07-31.md",
    ),
    (
        "tests/test_run_loo_cross_virus_v5_results_guard.py",
        "_local/notes/tier1-mechanical-four-enumeration-2026-07-31.md",
    ),
    # scripts/resave_checkpoint.py is a real 17KB file, gitignored at
    # .gitignore:366, so a fresh clone cannot open it. The bandit waiver rows
    # that cite it are a dated disposition record and are kept as written; the
    # doc already discloses the reproducibility caveat in its own text.
    # Recorded here because git cannot see a gitignored file, and a previous
    # session asserted from `git log --all` that this file "has never existed".
    ("docs/security_compliance.md", "scripts/resave_checkpoint.py"),
}

# Directories whose contents are not prose citations of this repository.
#
# The gate and its tests exclude THEMSELVES, and the reason is not cosmetic.
# Left in scope, this gate FAILS ON ITS OWN INTRODUCING COMMIT with fifteen
# UNPINNED findings - the files were untracked while the baseline was built, so
# `collect()` never saw them, and the failure appears only once they are
# staged. `--update` cannot rescue it either: five of those fifteen name paths
# that do not resolve from the citing file (docstring illustrations, and
# fixtures such as
# src/nonexistent.py:5), so they route to `problems` and re-report forever,
# while illustrative citations that happen to resolve - .gitignore:2, and a
# fabricated src/train_classifier.py:5 - would be pinned to whatever unrelated
# content sits there. That is the "fabricating a correct-looking pin" outcome
# EXEMPT_CITING_FILES above refuses to accept.
#
# THE COST, STATED RATHER THAN GLOSSED: four of those fifteen are REAL, live,
# currently-correct citations that this exclusion gives up checking -
# CHANGELOG.md:118, docs/model_cards/rf_30feature.md:26 (which is load-bearing
# for the exemption argument above), .gitignore:366 and .gitignore:259. They
# are knowingly traded away, because the alternative is a gate that cannot pass
# its own commit. It is a real loss, not an empty set.
#
# All three figures (15 / 5 / 4) were re-derived by simulating the pre-fix
# state, not carried over from a report.
EXCLUDED_PREFIXES = (
    ".github/",
    "docs/line_citations.json",
    "scripts/check_doc_line_citations.py",
    "tests/test_check_doc_line_citations.py",
)

# A repo-relative path with a real source extension, followed by :NNN, an
# inclusive range (:NNN-NNN), or a comma list (:NNN,NNN).
#
# The leading lookbehind rejects a path already embedded in a longer token, so
# a URL fragment or an already-matched prefix does not produce a second hit.
#
# The basename alternation admits a LEADING-DOT name (.gitignore) as well as
# name.ext. Requiring a basename before the dot made three real citations
# invisible - two of .gitignore:259 and one of .gitignore:366 - a silent blind
# spot rather than a false alarm, which is the direction that matters.
#
# Honest accounting of the yield: only ONE of those three becomes checked
# (tests/test_data_bias_audit_guard.py). The other two sit in an exempt ledger
# and in a multi-citation annotated line. The pattern is still widened, because
# the next dotfile citation someone writes should not be invisible by default.
CITATION_RE = re.compile(
    r"(?<![\w/.-])"
    r"((?:[\w.-]+/)*"
    r"(?:[\w.-]+\.(?:py|md|toml|json|yml|yaml|cfg|ini|txt|sh|lock)|\.gitignore))"
    r":(\d+(?:\s*-\s*\d+)?(?:\s*,\s*\d+)*)"
)

# Preserves a dated line number while naming the current one. See module docs.
#
# The captured number is REDIRECTED TO, not skipped. docs/security_compliance.md
# writes "`tests/test_sestrav_evaluator_extended.py:165` (2026-06-18 scan; now
# line 178)": 165 is the dated forensic record and must not be rewritten, but
# 178 is the pointer a reader
# actually follows, so 178 is what gets pinned. Skipping the whole line instead
# would leave those pointers permanently unchecked - a blanket relaxation,
# which is the shape of gate weakening this repo treats as most dangerous.
#
# Of the 7 annotated lines in that file the redirect recovers 5, not 7: two
# cite the gitignored scripts/resave_checkpoint.py (exempt above), and one of
# those carries two citations, which the single-match guard skips anyway.
SELF_ANNOTATION_RE = re.compile(r"now\s+(?:at\s+)?line\s+(\d+)", re.IGNORECASE)

# Per-line opt-out, mirroring check_doc_commit_refs.py's sha-check:ignore.
SUPPRESS_MARKER = "line-cite:ignore"

# Pinned text is stored stripped and truncated: enough to identify the line,
# short enough to keep the baseline reviewable.
PIN_MAX_CHARS = 160


def run_git(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(["git", *args], capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip()


def repo_root() -> Path:
    code, out = run_git(["rev-parse", "--show-toplevel"])
    if code != 0 or not out:
        print("error: not a git repository", file=sys.stderr)
        sys.exit(2)
    return Path(out)


def tracked_files() -> list[str]:
    code, out = run_git(["ls-files"])
    if code != 0:
        print("error: git ls-files failed; not a git repository?", file=sys.stderr)
        sys.exit(2)
    return [line for line in out.splitlines() if line]


def should_scan(path: str) -> bool:
    if path.startswith(EXCLUDED_PREFIXES):
        return False
    if path in EXEMPT_CITING_FILES:
        return False
    return Path(path).suffix.lower() in SCAN_SUFFIXES


def parse_spec(spec: str) -> list[int]:
    """Expand '139,148' or '781-786' into the list of cited line numbers.

    A reversed or absurd range yields no numbers rather than raising, so a
    malformed citation is reported as OUT-OF-RANGE instead of crashing the gate.
    """
    nums: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            head, tail = part.split("-", 1)
            try:
                lo, hi = int(head.strip()), int(tail.strip())
            except ValueError:
                continue
            if lo <= hi and hi - lo <= 200:
                nums.extend(range(lo, hi + 1))
        else:
            try:
                nums.append(int(part))
            except ValueError:
                continue
    return nums


def normalize(text: str) -> str:
    """Collapse whitespace so reindentation alone is not reported as drift.

    The trailing .strip() makes this IDEMPOTENT, which it must be: stored pins
    are normalized again when they are read back, so normalize(normalize(x))
    has to equal normalize(x). Without it, truncation landing on a space left a
    trailing space that the second pass removed, and the pinned line reported as
    DRIFTED against itself. Found by running the gate on this repository:
    docs/claims_register.md's citation of a long README table row differed from
    its own pin by exactly one trailing space.

    The trigger is narrow - the collapsed text must be longer than
    PIN_MAX_CHARS AND the cut must land on whitespace - not every long line.
    (That discovery predates exempting claims_register.md, so the shipped gate
    no longer reads the file that surfaced it.)

    KNOWN BLIND SPOT, and the direct cost of truncating: an edit that rewrites
    a cited line only BEYOND character PIN_MAX_CHARS does not change the pin
    and does not fire. Measured, not theorised. Widening the cap shrinks the
    blind spot and makes the baseline less reviewable; 160 is the trade, and it
    is a tunable, which is why the module docstring's "no threshold to tune" is
    scoped to the MATCHING RULE rather than claimed of the whole gate.
    """
    return " ".join(text.split())[:PIN_MAX_CHARS].strip()


def iter_citations(text: str):
    """Yield (lineno, target, spec) for each line citation in one file.

    Fenced markdown blocks are skipped: a citation inside example output is
    illustration, not a pointer a reader follows.
    """
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if SUPPRESS_MARKER in line:
            continue
        matches = list(CITATION_RE.finditer(line))
        annot = SELF_ANNOTATION_RE.search(line)
        if annot:
            # Redirect to the current line the annotation names, but only when
            # the association is unambiguous. Two citations plus one "now line"
            # gives no way to know which it belongs to, so the line is skipped
            # rather than guessed at.
            if len(matches) == 1:
                yield lineno, matches[0].group(1), annot.group(1)
            continue
        for match in matches:
            yield lineno, match.group(1), match.group(2)


def pin_key(citing: str, target: str, spec: str) -> str:
    return f"{citing}|{target}|{spec}"


def read_lines(root: Path, target: str) -> list[str] | None:
    p = root / target
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8", errors="replace").splitlines()


def collect(root: Path, tracked: set[str]) -> dict[str, dict]:
    """Walk tracked docs and return the current citation state, keyed by pin_key."""
    found: dict[str, dict] = {}
    for rel in sorted(tracked):
        if not should_scan(rel):
            continue
        p = root / rel
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, target, spec in iter_citations(text):
            key = pin_key(rel, target, spec)
            entry = found.setdefault(
                key,
                {
                    "citing": rel,
                    "target": target,
                    "lines": spec,
                    "occurrences": [],
                },
            )
            entry["occurrences"].append(lineno)
    return found


def resolve_target(citing: str, target: str, tracked: set[str]) -> str | None:
    """Resolve a cited path the way a reader would, or None if it does not exist.

    Repo-root-relative is tried first, then relative to the citing file's own
    directory. The second form is how sibling references are actually written:
    docs/claims_register.md cites 'limitations_statement_v1.md:72' for a file
    that really is docs/limitations_statement_v1.md. Resolving only from the
    root would report that as dead when a reader would find it immediately.
    (That example predates exempting claims_register.md and is quoted because
    it is the clearest one; the behaviour applies to every scanned file.)

    The fallback is NOT a blanket rescue. scripts/audit_cv_leakage.py cited
    'prepare_external_validation_inputs.py:122' for a file that lives in src/,
    so neither form resolved and the gate reported it - correctly, because a
    reader following it from either place found nothing. THAT CITATION IS FIXED
    BY THIS CHANGE (the src/ prefix was added), so it resolves and is pinned
    today; it is described here in the past tense because it is the specimen
    that motivated the rule, not a live finding.
    """
    if target in tracked:
        return target
    sibling = f"{Path(citing).parent.as_posix()}/{target}".lstrip("./")
    if sibling in tracked:
        return sibling
    return None


def current_pin(root: Path, tracked: set[str], citing: str, target: str, spec: str):
    """Return (pinned_text_list, error_kind). Exactly one is non-None."""
    resolved = resolve_target(citing, target, tracked)
    if resolved is None:
        return None, "MISSING"
    lines = read_lines(root, resolved)
    if lines is None:
        return None, "MISSING"
    nums = parse_spec(spec)
    if not nums:
        return None, "OUT-OF-RANGE"
    out: list[str] = []
    for n in nums:
        if n < 1 or n > len(lines):
            return None, "OUT-OF-RANGE"
        out.append(normalize(lines[n - 1]))
    return out, None


RATCHET_KEY = "exempt_ledger_citation_ceiling"


def load_ratchet_ceiling(root: Path) -> int | None:
    """The recorded ceiling on line citations inside the exempt ledgers.

    None means unset, which happens exactly once - on the commit that introduces
    the ratchet, before the first --update seeds it. Returning None skips the
    check rather than defaulting to 0 (which would fail every tree) or to
    infinity (which would pass silently, the failure direction this gate treats
    as worse).
    """
    p = root / BASELINE_PATH
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = raw.get(RATCHET_KEY)
    return value if isinstance(value, int) else None


def load_baseline(root: Path) -> dict[str, dict]:
    p = root / BASELINE_PATH
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read {BASELINE_PATH}: {exc}", file=sys.stderr)
        sys.exit(2)
    out: dict[str, dict] = {}
    for entry in raw.get("citations", []):
        key = pin_key(entry["citing"], entry["target"], entry["lines"])
        out[key] = entry
    return out


def write_baseline(root: Path, entries: list[dict], ratchet_ceiling: int) -> None:
    payload = {
        RATCHET_KEY: ratchet_ceiling,
        "_ratchet_comment": (
            "Number of path:NNN citations currently inside EXEMPT_CITING_FILES. "
            "Their CONTENT is not checked (see the module docstring), so this "
            "blind spot is allowed to shrink but never to grow. A rise fails the "
            "gate. Lowering it is free and welcome: re-anchor a citation to its "
            "SYMBOL, which does not rot, then run --update."
        ),
        "_comment": (
            "Pinned content for the path:NNN citations this gate CHECKS - not "
            "for every citation in the tree. Most live in historical ledgers "
            "(see EXEMPT_CITING_FILES) and are deliberately absent here; the "
            "gate prints both counts on every check run. "
            "Written by scripts/check_doc_line_citations.py --update. "
            "Each 'pinned' array is the text the cited line(s) held when the "
            "entry was last confirmed by a human. If an edit moves those "
            "lines, the pinned text stops matching and CI reports DRIFTED. "
            "Review every pinned-text change in the diff: it is either a "
            "citation you deliberately repointed, or a rot you just hid."
        ),
        "citations": sorted(
            entries, key=lambda e: (e["citing"], e["target"], e["lines"])
        ),
    }
    p = root / BASELINE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def iter_exempt_citations(root: Path, tracked: set[str]):
    """Yield (citing, lineno, target, spec) for every citation in an exempt ledger.

    The exemption in EXEMPT_CITING_FILES is about DRIFT, which needs a pin and a
    live/historical judgement no line-scoped gate can make. It was never meant to
    exempt these files from the two checks that need neither.
    """
    for rel in sorted(EXEMPT_CITING_FILES):
        if rel not in tracked:
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, target, spec in iter_citations(text):
            yield rel, lineno, target, spec


def audit_exempt_ledgers(root: Path, tracked: set[str]) -> tuple[list[str], dict]:
    """Check the two things about an exempt-ledger citation that ARE decidable.

    Why this exists
    ---------------
    The ledgers are exempt from DRIFT checking for a measured reason recorded in
    EXEMPT_CITING_FILES: one physical row can carry a historical citation and a
    live one to the SAME target, and separating them needs sentence-scope
    attribution inside a 14,000-character line - the proximity heuristic this
    gate rejects everywhere else.

    None of that reasoning applies to a citation that names a file which does not
    resolve at all, or a line past the end of one. Those are broken for every
    reader, historical intent or not, so they are hard failures here. Measured
    when this was added: 3 of 63, all in CHANGELOG.md, two of them introduced the
    same day by quoting an old citation inside a retraction.

    THE RATCHET, and why it is the real fix
    ---------------------------------------
    Drift inside these files stays uncovered, and pretending otherwise would be
    the false-PASS this gate's docstring calls the more dangerous direction. What
    IS enforceable is that the blind spot must not GROW: the ledgers may keep the
    line citations they already carry, but a new one cannot be added. That turns
    an unbounded liability into a shrinking one without a single heuristic, and
    it pushes new citations toward symbol anchors, which do not rot at all.

    Ratchet ceiling lives in the baseline as `exempt_ledger_citation_ceiling` so
    it is reviewed in the same diff as everything else this gate pins.
    """
    notices: list[str] = []
    stats = {"total": 0, "resolves": 0, "missing": 0, "out_of_range": 0, "exempt_pair": 0}

    for citing, lineno, target, spec in iter_exempt_citations(root, tracked):
        stats["total"] += 1
        if (citing, target) in EXEMPT_CITATIONS:
            stats["exempt_pair"] += 1
            continue
        _, err = current_pin(root, tracked, citing, target, spec)
        if err == "MISSING":
            stats["missing"] += 1
            notices.append(
                f"{citing}:{lineno}: cited file '{target}' does not resolve. If "
                f"the file was renamed or the path prefix is missing, fix it; if "
                f"it was deleted, that is legitimate history."
            )
        elif err == "OUT-OF-RANGE":
            stats["out_of_range"] += 1
            notices.append(
                f"{citing}:{lineno}: '{target}:{spec}' points past the end of the "
                f"file. Expected if the target shrank since the entry was written."
            )
        else:
            stats["resolves"] += 1

    return notices, stats


def in_github_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def emit(finding: str) -> None:
    print(f"::error::{finding}" if in_github_actions() else f"ERROR {finding}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check pinned line citations.")
    parser.add_argument(
        "--update",
        action="store_true",
        help=(
            "Rewrite the baseline from the current tree, then READ THE DIFF. "
            "A changed pinned text is either a deliberate repoint or a rot."
        ),
    )
    args = parser.parse_args()

    root = repo_root()
    tracked = set(tracked_files())
    found = collect(root, tracked)

    if args.update:
        entries: list[dict] = []
        problems: list[str] = []
        for key, item in sorted(found.items()):
            if (item["citing"], item["target"]) in EXEMPT_CITATIONS:
                continue
            pinned, err = current_pin(
                root, tracked, item["citing"], item["target"], item["lines"]
            )
            if err:
                problems.append(
                    f"{item['citing']}: cannot pin {item['target']}:{item['lines']} ({err})"
                )
                continue
            entries.append(
                {
                    "citing": item["citing"],
                    "target": item["target"],
                    "lines": item["lines"],
                    "pinned": pinned,
                }
            )
        _, exempt_stats = audit_exempt_ledgers(root, tracked)
        write_baseline(root, entries, exempt_stats["total"])
        print(f"Wrote {len(entries)} pinned citation(s) to {BASELINE_PATH}.")
        print(
            f"Ratchet ceiling seeded at {exempt_stats['total']} exempt-ledger "
            f"citation(s). If that number went UP, say why in the commit message."
        )
        if problems:
            print("")
            print(f"{len(problems)} citation(s) could NOT be pinned and were omitted:")
            for p in problems:
                print(f"  {p}")
            print("Fix these citations; they are unresolvable as written.")
            return 1
        print("Review the diff before committing.")
        return 0

    baseline = load_baseline(root)
    findings: list[str] = []
    checked = 0

    for key, item in sorted(found.items()):
        citing, target, spec = item["citing"], item["target"], item["lines"]
        if (citing, target) in EXEMPT_CITATIONS:
            continue
        where = f"{citing}:{item['occurrences'][0]}"
        entry = baseline.get(key)
        if entry is None:
            findings.append(
                f"{where}: UNPINNED - citation '{target}:{spec}' has no baseline "
                f"entry. Run 'python scripts/check_doc_line_citations.py --update' "
                f"and confirm what it points at."
            )
            continue

        pinned, err = current_pin(root, tracked, citing, target, spec)
        if err == "MISSING":
            findings.append(
                f"{where}: MISSING - cited file '{target}' is not tracked, so a "
                f"reader cannot open it."
            )
            continue
        if err == "OUT-OF-RANGE":
            findings.append(
                f"{where}: OUT-OF-RANGE - '{target}:{spec}' points past the end "
                f"of the file."
            )
            continue

        checked += 1
        recorded = [normalize(x) for x in entry.get("pinned", [])]
        if pinned != recorded:
            detail = _first_difference(recorded, pinned)
            findings.append(
                f"{where}: DRIFTED - '{target}:{spec}' no longer holds the "
                f"pinned content. {detail} Re-anchor the citation (prefer naming "
                f"the symbol over the line number), then --update."
            )

    for key, entry in sorted(baseline.items()):
        if key not in found:
            findings.append(
                f"{BASELINE_PATH}: STALE-PIN - baseline still pins "
                f"'{entry['target']}:{entry['lines']}' for '{entry['citing']}', "
                f"but that citation no longer appears there. Run --update."
            )

    # Report the exempted volume alongside the checked volume. A bare checked
    # count reads as broad coverage; most line citations in this tree sit in the
    # two historical ledgers and are NOT checked. A gate that hides the size of
    # its own blind spot is a gate that will be over-trusted. (Both figures are
    # computed here rather than quoted, so the two FIGURES cannot go stale -
    # which matters, since this file is the one the gate cannot check. The
    # surrounding prose, including "two historical ledgers", is still hand-
    # written and would need updating if EXEMPT_CITING_FILES grew.)
    # ADVISORY, not findings. A historical ledger may legitimately cite a line
    # past end-of-file or a since-deleted file - that is what "where it used to
    # be" means, and tests/test_check_doc_line_citations.py pins that behaviour
    # deliberately. Surfacing the list is what closes the blind spot; failing on
    # it would be a false-FAIL factory over legitimate history.
    exempt_notices, exempt_stats = audit_exempt_ledgers(root, tracked)

    # The ratchet. Existing ledger citations are grandfathered; new ones are not.
    # A missing ceiling is not treated as "unlimited" - it is seeded from the
    # current count on the next --update, and until then the check is skipped
    # rather than silently passing an unbounded file.
    ceiling = load_ratchet_ceiling(root)
    exempt_total = exempt_stats["total"]
    if ceiling is not None and exempt_total > ceiling:
        findings.append(
            f"{BASELINE_PATH}: RATCHET - the exempt historical ledgers now carry "
            f"{exempt_total} line citation(s), above the recorded ceiling of "
            f"{ceiling}. Drift inside those files is NOT checked, so the blind "
            f"spot must not grow. Cite the SYMBOL instead of the line, or write "
            f"the reference as prose. Raising the ceiling is a deliberate act: "
            f"run --update and justify it in the diff."
        )

    print(
        f"Checked {checked} pinned line citation(s) across tracked docs. "
        f"Exempt historical ledgers hold {exempt_total} more "
        f"(ceiling {ceiling if ceiling is not None else 'unset'}): "
        f"{exempt_stats['resolves']} resolve, {exempt_stats['missing']} missing, "
        f"{exempt_stats['out_of_range']} out-of-range, "
        f"{exempt_stats['exempt_pair']} exempt by pair. Their CONTENT is not "
        f"checked; see EXEMPT_CITING_FILES."
    )
    if exempt_notices:
        print("")
        print(
            f"{len(exempt_notices)} exempt-ledger citation(s) do not resolve "
            f"against HEAD. Advisory, not a failure - a ledger may legitimately "
            f"name a deleted file or a line that has since moved past EOF. "
            f"Review each: a missing path prefix is a real defect, deleted "
            f"history is not."
        )
        for notice in exempt_notices:
            print(f"  NOTE {notice}")

    if findings:
        print("")
        for finding in findings:
            emit(finding)
        print("")
        print(f"{len(findings)} line-citation problem(s).")
        print(
            "Line numbers rot whenever anything upstream of them changes. The "
            "durable fix is to cite the SYMBOL (a function, constant or unique "
            "assignment) instead of the line, which does not move."
        )
        return 1

    print("All pinned line citations still hold their recorded content.")
    return 0


def _first_difference(recorded: list[str], current: list[str]) -> str:
    """Human-readable account of the first divergence, for the error message."""
    if len(recorded) != len(current):
        return f"Pinned {len(recorded)} line(s), citation now spans {len(current)}."
    for i, (a, b) in enumerate(zip(recorded, current)):
        if a != b:
            return f"Pinned: {a!r}; now: {b!r}."
    return "Content differs."


if __name__ == "__main__":
    sys.exit(main())
