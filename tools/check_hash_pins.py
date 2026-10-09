#!/usr/bin/env python
"""Fail if any requirement in a hash-pinned manifest is missing its --hash.

`pip install --require-hashes` rejects a file where *some* requirement lacks a
hash, but CI only finds out at install time and only for the files it happens
to install. This is the cheap, dependency-free gate: it parses the manifests
that are contractually hash-pinned and reports every requirement line that has
no `--hash=` attached.

It also rejects requirement-file options that add or swap a package source, or
exempt a host from TLS verification: `--extra-index-url`, `--trusted-host`,
`--find-links` / `-f`, and `--index-url` / `-i` naming anything but PyPI's default
index. Hashes would still refuse a substituted artifact, but these manifests take
their index from the install command, never from the file: the CPU torch lock is
installed with `--index-url https://download.pytorch.org/whl/cpu` on the command
line. Options are read the way pip reads them (its line joining, comment rule,
option/argument split and optparse table, abbreviations included), so a spelling
pip honours cannot slip past. `-r` / `-c` includes are not followed; no
hash-pinned manifest uses one. Until 2026-10-09 every option line was skipped
silently.

Usage:
    python tools/check_hash_pins.py               # check the default manifests
    python tools/check_hash_pins.py path/to.txt   # check specific files
"""

from __future__ import annotations

import argparse
import optparse
import pathlib
import re
import shlex
import sys
from dataclasses import dataclass
from typing import NoReturn

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


# `--index-url` / `-i` pointing at PyPI's own default index is allowed, so an
# explicitly PyPI-pinned compile step cannot fail this gate. Compared verbatim, as
# pip does not normalise it either.
PYPI_DEFAULT_INDEX = frozenset({"https://pypi.org/simple", "https://pypi.org/simple/"})

# pip's requirement-file comment rule: `#` opens a comment only at the start of a
# line or after whitespace, so `https://host/simple#frag` keeps its fragment.
_PIP_COMMENT_RE = re.compile(r"(^|\s+)#.*$")


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    text: str


def _strip_comment(line: str) -> str:
    """Drop a trailing comment. Requirement lines here never contain a URL fragment."""
    index = line.find("#")
    return line if index < 0 else line[:index]


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """(1-based start line, joined text) for every non-blank, non-comment logical line.

    Backslash continuations are joined into one logical line, so the `--hash=`
    entries the compiler puts on following lines stay with their requirement.
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
    return requirements


def iter_requirements(text: str) -> list[tuple[int, str]]:
    """Yield (1-based start line, joined requirement) for real requirement lines.

    Skips blanks, comments, `-r`/`-c` includes and any other option-only line
    (`--index-url`, `--extra-index-url`, `--find-links`, ...); the options that
    change the package source are checked separately by `iter_index_options`.
    """
    return [(number, req) for number, req in _logical_lines(text) if not req.startswith("-")]


class _OptionError(Exception):
    pass


class _RequirementFileOptionParser(optparse.OptionParser):
    def error(self, msg: str) -> NoReturn:  # optparse's default calls sys.exit
        raise _OptionError(msg)


def _option_parser() -> optparse.OptionParser:
    """Every option pip accepts in a requirement file, with pip's flags and arity.

    Transcribed from pip 26.2.1's `req_file.SUPPORTED_OPTIONS` plus
    `SUPPORTED_OPTIONS_REQ`. optparse resolves an unambiguous abbreviation such as
    `--extra=URL` exactly as pip's parser does. No list default is shared: an
    `append` option starts at None on every parse. tests/test_dependency_tooling.py
    compares this table's verdicts with pip's own parser.
    """
    parser = _RequirementFileOptionParser(add_help_option=False, usage=optparse.SUPPRESS_USAGE)
    parser.add_option("-i", "--index-url", "--pypi-url", dest="index_url")
    parser.add_option("--extra-index-url", dest="extra_index_urls", action="append")
    parser.add_option("--no-index", action="store_true")
    parser.add_option("-c", "--constraint", action="append")
    parser.add_option("-r", "--requirement", action="append")
    parser.add_option("-e", "--editable", action="append")
    parser.add_option("-f", "--find-links", dest="find_links", action="append")
    parser.add_option("--no-binary", action="append")
    parser.add_option("--only-binary", action="append")
    parser.add_option("--prefer-binary", action="store_true")
    parser.add_option("--require-hashes", action="store_true")
    parser.add_option("--no-require-hashes", action="store_true")
    parser.add_option("--pre", action="store_true")
    parser.add_option("--all-releases", action="append")
    parser.add_option("--only-final", action="append")
    parser.add_option("--trusted-host", dest="trusted_hosts", action="append")
    parser.add_option("--use-feature", action="append")
    parser.add_option("--hash", action="append")
    parser.add_option("-C", "--config-settings", action="append")
    return parser


def _pip_logical_lines(text: str) -> list[tuple[int, str]]:
    """pip's `join_lines` then `ignore_comments`, so a line reads as pip reads it."""
    joined: list[tuple[int, str]] = []
    primary = 0
    pending: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.endswith("\\") or _PIP_COMMENT_RE.match(line):
            if _PIP_COMMENT_RE.match(line):
                line = " " + line
            if pending:
                pending.append(line)
                joined.append((primary, "".join(pending)))
                pending = []
            else:
                joined.append((number, line))
        else:
            if not pending:
                primary = number
            pending.append(line.strip("\\"))
    if pending:
        joined.append((primary, "".join(pending)))
    stripped = ((number, _PIP_COMMENT_RE.sub("", line).strip()) for number, line in joined)
    return [(number, line) for number, line in stripped if line]


def _options_part(line: str) -> str:
    """pip's `break_args_options`: the options start at the first token led by `-`."""
    tokens = line.split(" ")
    for index, token in enumerate(tokens):
        if token.startswith("-"):
            return " ".join(tokens[index:])
    return ""


def iter_index_options(text: str) -> list[tuple[int, str]]:
    """Yield (1-based start line, line) for every option that changes the package source.

    A line whose options pip itself could not parse is reported too, marked as such:
    failing closed is the safe direction for a supply-chain gate.
    """
    parser = _option_parser()
    found: list[tuple[int, str]] = []
    for number, line in _pip_logical_lines(text):
        options = _options_part(line)
        if not options:
            continue
        shown = " ".join(line.split())  # pip joins continuations without a separator
        try:
            values, _args = parser.parse_args(shlex.split(options))
        except (_OptionError, ValueError) as error:
            found.append((number, f"{shown}  [unparseable options: {error}]"))
            continue
        index_url = values.index_url
        if (
            (index_url is not None and index_url not in PYPI_DEFAULT_INDEX)
            or values.extra_index_urls
            or values.find_links
            or values.trusted_hosts
        ):
            found.append((number, shown))
    return found


def _relative(path: pathlib.Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)


def check_file(path: pathlib.Path) -> list[Violation]:
    """Return every requirement in `path` that carries no --hash."""
    text = path.read_text(encoding="utf-8")
    relative = _relative(path)
    return [
        Violation(relative, number, requirement)
        for number, requirement in iter_requirements(text)
        if "--hash=" not in requirement
    ]


def check_index_options(path: pathlib.Path) -> list[Violation]:
    """Return every option line in `path` that adds or swaps a package source."""
    relative = _relative(path)
    return [
        Violation(relative, number, line)
        for number, line in iter_index_options(path.read_text(encoding="utf-8"))
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
    redirects: list[Violation] = []
    for path in paths:
        violations.extend(check_file(path))
        redirects.extend(check_index_options(path))

    if violations:
        print("ERROR: un-hashed requirements found (breaks pip --require-hashes):", file=sys.stderr)
        for violation in violations:
            print(f"  {violation.path}:{violation.line}: {violation.text}", file=sys.stderr)
    if redirects:
        print(
            "ERROR: index-redirecting options found (a hash-pinned manifest never chooses its own"
            " index; pass one on the install command, as the CPU torch lock's installs do):",
            file=sys.stderr,
        )
        for violation in redirects:
            print(f"  {violation.path}:{violation.line}: {violation.text}", file=sys.stderr)
    if violations or redirects:
        return 1

    print(
        f"All requirements hash-pinned across {len(paths)} manifest(s), "
        "with no index-redirecting options."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
