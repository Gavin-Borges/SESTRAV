#!/usr/bin/env python3
"""Verify that every commit SHA cited in tracked docs still resolves.

Why this exists
---------------
Sign-offs were once backfilled onto already-published commits with
``git rebase --signoff``. Every SHA in the rebased range changed, and the
commit hashes cited across tracked documentation silently stopped resolving:
20 dead citations across 12 files, undetected until 2026-07-24. A reader
auditing ``docs/paper.md`` got a hash that pointed at nothing.

Nothing in CI could see that. This script closes the gap.

What counts as a failure
------------------------
DEAD      the token looks like a cited commit but no such object exists.
ORPHANED  the object exists but is not reachable from the target ref, so a
          fresh clone cannot resolve it.

Both mean a reader cannot verify the citation, which is the property that
matters.

Reachability is measured from HEAD, not from the base branch. In a pull
request HEAD is the merge commit, so it models the post-merge state: a doc
may legitimately cite an earlier commit of the same PR, while a genuinely
orphaned commit (reachable from no ref at all) is still caught.

Deliberate exclusions
---------------------
``.github/`` is skipped wholesale: workflow files pin third-party actions to
*upstream* SHAs that intentionally do not exist in this repository. Lock files
are skipped for the same reason. 64-character tokens are treated as SHA-256
digests, not commit hashes.

A bare hex token that does not resolve is only an error when it appears in an
explicit commit-citation context (near the word "commit", or under a
``git_sha``-style JSON key). Without that guard, ordinary hex strings such as
colour codes or dataset identifiers would produce false failures.

Usage
-----
    python scripts/check_doc_commit_refs.py [--reachable-from HEAD]

Exit codes: 0 clean, 1 findings, 2 invocation/environment error.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SCAN_SUFFIXES = {".md", ".json", ".toml", ".cff", ".txt", ".yaml", ".yml"}

EXCLUDED_PREFIXES = (
    ".github/",  # third-party action pins are upstream SHAs by design
)

EXCLUDED_NAMES = {
    "requirements.lock",
    "requirements-ci.txt",
    "requirements-lock.in",
    "requirements-ci.in",
}

# 7-40 hex characters, not embedded in a longer alphanumeric run, and not the
# fractional part of a decimal number.
#
# The `(?<!\d\.)` guard exists because "." is not alphanumeric, so the leading
# lookbehind alone lets the tail of a decimal match: in `0.8275628` the token
# `8275628` is seven characters, every one of them a valid hex digit, preceded
# by "." and followed by a boundary. `docs/claims_register.md` D16 cites exactly
# that pair - the 31-feature v3 weighted CV mean 0.8275628 against the
# 30-feature pooled field metric 0.8277666, whose 0.0002 separation is the
# documented root cause of the Tier A mislabel - and both were reported as dead
# commits, failing this gate on a repository that had no dead citations.
#
# Deliberately NOT fixed by rejecting all-decimal tokens outright: a real
# abbreviated SHA is all digits with probability (10/16)^7, about 3.7%, so that
# rule would silently stop catching roughly one in twenty-seven dead citations -
# a false negative in the one gate whose job is catching them. Scoping the guard
# to "preceded by digit-then-dot" rejects decimal tails only; a SHA that follows
# a sentence period ("... the fix. abc1234 reverts it") keeps matching, because
# the intervening space breaks the `\d\.` sequence.
SHA_RE = re.compile(r"(?<![0-9a-zA-Z])(?<!\d\.)([0-9a-f]{7,40})(?![0-9a-zA-Z])")

# Signals that a hex token is being presented as a commit reference.
COMMIT_CONTEXT_RE = re.compile(
    r"(commit|git[_-]?sha|revision|\bsha\b|checkout|cherry[- ]?pick)",
    re.IGNORECASE,
)

# Signals that the token belongs to a THIRD-PARTY repository rather than this
# one. Docs legitimately record the pinned SHA of an upstream GitHub Action
# (for example "Upgraded ossf/scorecard-action ... SHA pinned: <sha>"); those
# objects do not exist here and must not be reported.
#
# Every alternative here suppresses an ENTIRE LINE before any SHA is examined,
# so an over-broad alternative is a silent false negative in the one gate whose
# job is catching dead citations. Each must therefore be anchored or qualified:
#
#   "action"    was unanchored, so it matched "interaction", "extraction",
#               "fraction" and "reaction" - words that saturate an immunology
#               repo. It suppressed 63 tracked doc lines; "\bactions?\b"
#               suppresses 23 and still matches "actions/checkout" and
#               "ossf/scorecard-action" ("-" is a word boundary).
#   "pinned"    was word-anchored but is ordinary English here ("version-pinned
#               dependencies", "MHCflurry 2.2.1 (pinned)"), suppressing 37
#               lines. Requiring a sha/digest/commit/version/tag qualifier - or
#               "pinned to/at/by" - drops that to 4 while still excluding
#               docs/SCORECARD_REMEDIATION.md's "SHA pinned: <sha>" line, the
#               only tracked line in the repo that depends on this rule.
#   "upstream"  is not a substring of any other English word; the leading \b is
#               belt-and-braces and changes nothing (15 lines either way).
EXTERNAL_CONTEXT_RE = re.compile(
    r"""(
      uses:                                          # workflow step calling an action
    | \bactions?\b                                   # actions/checkout, scorecard-action
    | \bupstream                                     # upstream / upstreams / upstreamed
    | third[- ]party
    | \b(?:sha|digest|commit|version|tag)[- ]?pinn?ed\b
    | \bpinn?ed\s+(?:to|at|by)\b
    | [A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@               # org/repo@<sha>
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Explicit per-line opt-out for anything the heuristics cannot classify.
SUPPRESS_MARKER = "sha-check:ignore"


def run_git(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout.strip()


def tracked_files() -> list[str]:
    code, out = run_git(["ls-files"])
    if code != 0:
        print("error: git ls-files failed; not a git repository?", file=sys.stderr)
        sys.exit(2)
    return [line for line in out.splitlines() if line]


def should_scan(path: str) -> bool:
    if path.startswith(EXCLUDED_PREFIXES):
        return False
    if Path(path).name in EXCLUDED_NAMES:
        return False
    return Path(path).suffix.lower() in SCAN_SUFFIXES


def resolve_commit(token: str) -> str | None:
    """Return the full SHA if the token names a commit object, else None."""
    code, out = run_git(["rev-parse", "--verify", "--quiet", f"{token}^{{commit}}"])
    return out if code == 0 and out else None


def is_ancestor(sha: str, base_ref: str) -> bool:
    code, _ = run_git(["merge-base", "--is-ancestor", sha, base_ref])
    return code == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reachable-from",
        default="HEAD",
        help=(
            "Ref that cited commits must be reachable from (default: HEAD). "
            "HEAD is correct for pull requests: it models the post-merge state, "
            "so a doc may cite an earlier commit of the same PR without failing, "
            "while genuinely orphaned commits are still caught."
        ),
    )
    args = parser.parse_args()

    code, _ = run_git(["rev-parse", "--verify", "--quiet", args.reachable_from])
    if code != 0:
        print(
            f"error: ref '{args.reachable_from}' does not resolve.", file=sys.stderr
        )
        print("hint: fetch full history (fetch-depth: 0) first.", file=sys.stderr)
        return 2

    findings: list[str] = []
    checked = 0
    resolved_cache: dict[str, str | None] = {}

    for path in tracked_files():
        if not should_scan(path):
            continue
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            if SUPPRESS_MARKER in line:
                continue
            # A pinned upstream action SHA is not a citation of our history.
            if EXTERNAL_CONTEXT_RE.search(line):
                continue
            for match in SHA_RE.finditer(line):
                token = match.group(1)
                if len(token) == 64:
                    continue  # sha256 digest, not a commit hash

                has_context = bool(COMMIT_CONTEXT_RE.search(line))
                if token not in resolved_cache:
                    resolved_cache[token] = resolve_commit(token)
                full = resolved_cache[token]

                if full is None:
                    # Only an error when explicitly presented as a commit.
                    if has_context:
                        checked += 1
                        findings.append(
                            f"{path}:{lineno}: DEAD - cited commit '{token}' "
                            f"does not exist in this repository"
                        )
                    continue

                checked += 1
                if not is_ancestor(full, args.reachable_from):
                    findings.append(
                        f"{path}:{lineno}: ORPHANED - commit '{token}' exists but is "
                        f"not reachable from {args.reachable_from} (a fresh clone cannot "
                        f"resolve it)"
                    )

    print(f"Checked {checked} commit citation(s) across tracked docs.")

    if findings:
        print("")
        for finding in findings:
            print(f"::error::{finding}" if _in_github_actions() else f"ERROR {finding}")
        print("")
        print(f"{len(findings)} unresolvable commit citation(s).")
        print(
            "Repoint each citation at the surviving commit, or cite the change "
            "without a SHA. Do not backfill sign-offs onto published commits - "
            "that is what orphaned these in the first place."
        )
        return 1

    print("All cited commits resolve and are reachable.")
    return 0


def _in_github_actions() -> bool:
    import os

    return os.environ.get("GITHUB_ACTIONS") == "true"


if __name__ == "__main__":
    sys.exit(main())
