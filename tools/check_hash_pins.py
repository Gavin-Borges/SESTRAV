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
line. Options are read the way pip reads them: its byte-order-mark handling,
line joining, comment rule, option/argument split and optparse table,
abbreviations included. Two pip steps the gate cannot reproduce fail closed
instead: a PEP 263 coding declaration (pip would re-decode the file) and a
`${VAR}` reference (pip expands it from the installing environment). Until
2026-10-09 every option line was skipped silently.

Three more line shapes fail closed, because each hands pip something this gate
never inspects: an `-r` / `-c` include (the gate does not follow it, so a
redirect or an unhashed requirement inside it would pass), an `-e` editable
(pip cannot hash-check one), and any requirement that is not a plain
`name[extras] specifier ; marker`, such as a `name @ URL` direct reference, a
bare URL, a VCS reference, a path, or text ending in an archive extension, which
pip checks on the whole string (`numpy==2.4.6.tar.gz` is a local file). Each takes
its artifact from somewhere other than the index the install command names. The
default manifests carry none of them (measured 2026-10-10); until then all of
them exited 0.

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

# pip 26.2.1's req_file.BOMS, PEP263_ENCODING_RE and ENV_VAR_RE.
_PIP_BOMS = (
    (b"\xef\xbb\xbf", "utf-8"),
    (b"\xff\xfe\x00\x00", "utf-32"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16-be"),
)
_PIP_CODING_RE = re.compile(rb"coding[:=]\s*([-\w.]+)")
_PIP_ENV_VAR_RE = re.compile(r"\$\{[A-Z0-9_]+\}")

# The only requirement shape a hash-pinned manifest may use: a PEP 508 project
# name, optional extras, optional version specifiers (bare or parenthesised).
# Matched against the text before any `;` marker. An allowlist rather than a list
# of URL and path forms, so a spelling nobody thought of fails closed. The only
# whitespace it admits is the ASCII space, the only one a compiled lock contains:
# pip strips every Unicode space before an [extras] suffix, so a tab or U+3000
# there would otherwise hide an archive suffix from the check below.
_PLAIN_REQUIREMENT_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    r" *(?:\[[A-Za-z0-9._, -]*\])?"
    r" *(?:\( *[<>=!~][<>=!~A-Za-z0-9.*+,_ -]*\)|[<>=!~][<>=!~A-Za-z0-9.*+,_ -]*)?"
)

# pip 26.2.1's ARCHIVE_EXTENSIONS (utils/filetypes.py), same members. pip's
# parse_req_from_line tests the WHOLE requirement for them, not the project name:
# it takes os.path.abspath of the marker-free string (which on Windows drops
# trailing dots and spaces), strips a trailing [extras] and rstrip()s what is left,
# and treats a result ending in one of these as a local file, resolved from the
# directory pip runs in. So `numpy==2.4.6.tar.gz` and `pkg-1.0-py3-none-any.whl`,
# both accepted by the pattern above, are local files. (pip folds a preceding .tar
# into the extension, so in a few harmless cases this check is stricter than pip.)
_PIP_ARCHIVE_EXTENSIONS = (
    ".zip",
    ".whl",
    ".tar.bz2",
    ".tbz",
    ".tar.xz",
    ".txz",
    ".tlz",
    ".tar.lz",
    ".tar.lzma",
    ".tar.gz",
    ".tgz",
    ".tar",
)


def _is_plain_requirement(requirement: str) -> bool:
    """True for `name[extras] specifier`; False for a URL, path, VCS ref or archive."""
    if not _PLAIN_REQUIREMENT_RE.fullmatch(requirement):
        return False
    tail = re.sub(r"\[[^\]]*\]$", "", requirement.rstrip(". ")).rstrip(". ")
    return not tail.lower().endswith(_PIP_ARCHIVE_EXTENSIONS)


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


def _break_args_options(line: str) -> tuple[str, str]:
    """pip's `break_args_options`: the options start at the first token led by `-`."""
    tokens = line.split(" ")
    for index, token in enumerate(tokens):
        if token.startswith("-"):
            return " ".join(tokens[:index]), " ".join(tokens[index:])
    return line, ""


def _options_part(line: str) -> str:
    return _break_args_options(line)[1]


def iter_index_options(text: str) -> list[tuple[int, str]]:
    """Yield (1-based start line, line) for every line that changes the package source.

    That is an option adding or swapping an index, an include the gate does not
    follow, an editable, or a requirement that is not a plain project name. A line
    whose options this table cannot parse, or that holds a `${VAR}` reference pip
    would expand at install time, is reported too, marked as such: failing closed is
    the safe direction for a supply-chain gate.
    """
    parser = _option_parser()
    found: list[tuple[int, str]] = []
    for number, line in _pip_logical_lines(text):
        shown = " ".join(line.split())  # pip joins continuations without a separator
        if _PIP_ENV_VAR_RE.search(line):
            found.append((number, f"{shown}  [expands an environment variable]"))
            continue
        args, options = _break_args_options(line)
        requirement = args.split(";", 1)[0].strip()
        # Tested whenever the line has any argument at all, so a bare `;marker` with
        # an empty requirement in front of it fails closed too.
        if args.strip() and not _is_plain_requirement(requirement):
            found.append((number, f"{shown}  [not a plain project-name requirement]"))
            continue
        if not options:
            continue
        try:
            values, _args = parser.parse_args(shlex.split(options))
        except (_OptionError, ValueError) as error:
            found.append((number, f"{shown}  [unparseable options: {error}]"))
            continue
        if values.requirement or values.constraint:
            found.append((number, f"{shown}  [includes another requirement file, not followed]"))
            continue
        if values.editable:
            found.append((number, f"{shown}  [editable, which pip cannot hash-check]"))
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
    text, _problem = _decode_like_pip(path.read_bytes())  # reported by check_index_options
    relative = _relative(path)
    return [
        Violation(relative, number, requirement)
        for number, requirement in iter_requirements(text)
        if "--hash=" not in requirement
    ]


def _decode_like_pip(data: bytes) -> tuple[str, str | None]:
    """pip's `_decode_req_file`, except that a coding declaration fails closed.

    Returns (text, problem). A byte-order mark is stripped and its encoding used,
    as pip does. A PEP 263 declaration in the first two lines would make pip
    re-decode the whole file (`# coding: utf-7` turns `+AC0ALQ-` into `--`), so it
    is reported rather than trusted; no compiled lock carries one. Bytes that do not
    decode are reported too, rather than raised as a traceback.
    """
    for bom, encoding in _PIP_BOMS:
        if data.startswith(bom):
            body = data[len(bom) :]
            try:
                return body.decode(encoding), None
            except UnicodeDecodeError:
                return body.decode(encoding, errors="replace"), f"is not valid {encoding}"
    for line in data.split(b"\n")[:2]:
        if line[0:1] == b"#" and _PIP_CODING_RE.search(line):
            return data.decode("utf-8", errors="replace"), "declares its own encoding"
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace"), "is not valid UTF-8"


def check_index_options(path: pathlib.Path) -> list[Violation]:
    """Return every line in `path` that changes the package source, plus any decode problem.

    That is an index-redirecting option, an include, an editable, or a requirement that
    is not a plain project name (see `iter_index_options`); a byte sequence the gate
    cannot decode the way pip would is reported on line 1.
    """
    relative = _relative(path)
    text, problem = _decode_like_pip(path.read_bytes())
    violations = [Violation(relative, 1, f"[{problem}]")] if problem else []
    violations.extend(Violation(relative, number, line) for number, line in iter_index_options(text))
    return violations


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
            "ERROR: index-redirecting options, includes, editables or non-index requirements"
            " found (a hash-pinned manifest never chooses its own index or artifact source;"
            " pass an index on the install command, as the CPU torch lock's installs do):",
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
