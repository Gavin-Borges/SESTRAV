#!/usr/bin/env python
"""Fail if any requirement in a hash-pinned manifest is missing its --hash.

`pip install --require-hashes` rejects a file where *some* requirement lacks a
hash, but CI only finds out at install time and only for the files it happens
to install. This is the cheap, dependency-free gate: it parses the manifests
that are contractually hash-pinned and reports every requirement line that has
no `--hash=` attached.

Usage:
    python tools/check_hash_pins.py               # check the default manifests
    python tools/check_hash_pins.py path/to.txt   # check specific files
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# WIDENED 2026-08-22. The previous scope was ("requirements.txt",
# "environments/requirements-ci-*.txt"), which reached 7 manifests and left SIX
# unwatched: requirements-pip-audit.txt, requirements-sbom.txt,
# requirements-security.txt, requirements-semgrep.txt, requirements.lock - and
# requirements-ci.txt, which slipped through for a subtle reason worth recording:
# the old glob required a HYPHEN after "ci", so "requirements-ci.txt" did not
# match while its six "requirements-ci-*.txt" siblings did. A gate that silently
# covers less than its name implies is the failure mode this file exists to
# prevent, so the patterns below are deliberately broad rather than enumerated.
#
# Verified before widening: all six newly covered manifests are ALREADY fully
# hash-pinned (321 requirements, 0 unhashed), so this changes what is WATCHED,
# not what passes - the gate exits 0 at both 7 and 13 manifests.
DEFAULT_TARGETS: tuple[str, ...] = (
    "requirements.txt",
    "environments/requirements*.txt",
    "environments/requirements.lock",
)


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    text: str


def _strip_comment(line: str) -> str:
    """Drop a trailing comment. Requirement lines here never contain a URL fragment."""
    index = line.find("#")
    return line if index < 0 else line[:index]


def iter_requirements(text: str) -> list[tuple[int, str]]:
    """Yield (1-based start line, joined requirement) for real requirement lines.

    Skips blanks, comments, `-r`/`-c` includes and any other option-only line
    (`--index-url`, `--extra-index-url`, `--find-links`, ...). Backslash
    continuations are joined into a single logical requirement so that the
    `--hash=` entries pip-compile puts on following lines still count.
    """
    requirements: list[tuple[int, str]] = []
    buffer = ""
    start = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = _strip_comment(raw).strip()
        if not buffer:
            if not stripped:
                continue
            start = number
        elif not stripped:
            # A blank/comment-only line terminates an unterminated continuation.
            requirements.append((start, buffer.strip()))
            buffer = ""
            continue
        continued = stripped.endswith("\\")
        if continued:
            stripped = stripped[:-1].rstrip()
        buffer = f"{buffer} {stripped}".strip() if buffer else stripped
        if not continued:
            requirements.append((start, buffer))
            buffer = ""
    if buffer:
        requirements.append((start, buffer.strip()))
    return [(number, req) for number, req in requirements if not req.startswith("-")]


def check_file(path: pathlib.Path) -> list[Violation]:
    """Return every requirement in `path` that carries no --hash."""
    text = path.read_text(encoding="utf-8")
    relative = path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)
    return [
        Violation(relative, number, requirement)
        for number, requirement in iter_requirements(text)
        if "--hash=" not in requirement
    ]


def resolve_targets(patterns: list[str]) -> list[pathlib.Path]:
    """Expand glob patterns (relative to the repo root) into sorted file paths."""
    paths: set[pathlib.Path] = set()
    for pattern in patterns:
        if any(character in pattern for character in "*?["):
            paths.update(REPO_ROOT.glob(pattern))
        else:
            paths.add(REPO_ROOT / pattern)
    return sorted(paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        default=list(DEFAULT_TARGETS),
        help="manifest paths or globs relative to the repo root (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    paths = resolve_targets(args.targets or list(DEFAULT_TARGETS))
    missing = [path for path in paths if not path.is_file()]
    if missing or not paths:
        for path in missing:
            print(f"ERROR: no such manifest: {path}", file=sys.stderr)
        if not paths:
            print("ERROR: no manifests matched", file=sys.stderr)
        return 1

    violations: list[Violation] = []
    for path in paths:
        violations.extend(check_file(path))

    if violations:
        print("ERROR: un-hashed requirements found (breaks pip --require-hashes):", file=sys.stderr)
        for violation in violations:
            print(f"  {violation.path}:{violation.line}: {violation.text}", file=sys.stderr)
        return 1

    print(f"All requirements hash-pinned across {len(paths)} manifest(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
