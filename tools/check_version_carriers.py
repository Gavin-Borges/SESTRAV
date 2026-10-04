#!/usr/bin/env python3
"""Fail when a tracked current-version carrier disagrees with pyproject.toml.

Carrier discovery is format-based, not a fixed path allowlist. This matters because a
new badge, command-output example, API fallback, citation file, or model card should
join the census automatically wherever it is placed in the tracked tree.

The check fails closed rather than passing on a shrunken census:

* every carrier KIND in ``REQUIRED_KINDS`` must be found at least once. A carrier
  whose file is missing from the working tree, is not UTF-8, or was reformatted so
  its pattern no longer matches would otherwise drop out silently, and the carriers
  that remain would still "agree".
* the value compared is the whole version token, not its X.Y.Z prefix, so
  ``2.0.3rc1``, ``2.0.3.1`` and ``2.0.3-dev`` disagree with ``2.0.3`` instead of
  being read as it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO

VERSION = r"[0-9]+\.[0-9]+\.[0-9]+"
# A carrier's value is captured to the end of its version token, so a suffix such as
# rc1, .1 or -dev is part of what is compared. A trailing "." stays outside the token.
VERSION_TOKEN = rf"{VERSION}(?:[0-9A-Za-z.+-]*[0-9A-Za-z])?"


@dataclass(frozen=True)
class Carrier:
    path: str
    line: int
    kind: str
    version: str


@dataclass(frozen=True)
class Pattern:
    kind: str
    regex: re.Pattern[str]


PATTERNS = (
    Pattern("version badge", re.compile(rf"badge/version-(?P<version>{VERSION_TOKEN})-")),
    Pattern(
        "BibTeX version",
        re.compile(rf"^\s*version\s*=\s*\{{(?P<version>{VERSION_TOKEN})\}}\s*,?\s*$", re.M),
    ),
    Pattern(
        "CLI version output",
        re.compile(rf"sestrav version\s*:\s*(?P<version>{VERSION_TOKEN})"),
    ),
    Pattern(
        "source-run API fallback",
        re.compile(rf'^\s*_APP_VERSION\s*=\s*["\'](?P<version>{VERSION_TOKEN})["\']\s*$', re.M),
    ),
    Pattern(
        "model-card version field",
        re.compile(rf"^-\s*\*\*Version:\*\*\s*SESTRAV\s+v(?P<version>{VERSION_TOKEN})", re.M),
    ),
)

# The release workflow reads CITATION.cff's top-level version with optional quotes, so
# a quoted value is a valid release state and must be read here too.
CFF_VERSION = re.compile(
    rf"^version:[ \t]*(?P<quote>[\"']?)(?P<version>{VERSION_TOKEN})(?P=quote)[ \t]*\r?$", re.M
)

REQUIRED_KINDS: tuple[str, ...] = (
    "canonical project version",
    "citation metadata version",
    *(pattern.kind for pattern in PATTERNS),
)


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [part.decode("utf-8") for part in result.stdout.split(b"\0") if part]


def _read_text(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\0" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def canonical_version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as stream:
        value = tomllib.load(stream)["project"]["version"]
    if not isinstance(value, str) or re.fullmatch(VERSION, value) is None:
        raise ValueError(f"pyproject.toml [project].version is not X.Y.Z: {value!r}")
    return value


def enumerate_carriers(root: Path, tracked_files: Iterable[str] | None = None) -> list[Carrier]:
    files = list(tracked_files) if tracked_files is not None else _tracked_files(root)
    carriers: list[Carrier] = []

    project_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    project_match = re.search(rf'^version\s*=\s*"(?P<version>{VERSION})"\s*$', project_text, re.M)
    if project_match is None:
        raise ValueError("pyproject.toml has no literal [project] version carrier")
    carriers.append(
        Carrier(
            "pyproject.toml",
            _line_number(project_text, project_match.start("version")),
            "canonical project version",
            project_match.group("version"),
        )
    )

    for relative in files:
        if relative == "pyproject.toml":
            continue
        text = _read_text(root / relative)
        if text is None:
            continue

        if relative.endswith(".cff"):
            match = CFF_VERSION.search(text)
            if match:
                carriers.append(
                    Carrier(
                        relative,
                        _line_number(text, match.start("version")),
                        "citation metadata version",
                        match.group("version"),
                    )
                )

        for pattern in PATTERNS:
            for match in pattern.regex.finditer(text):
                carriers.append(
                    Carrier(
                        relative,
                        _line_number(text, match.start("version")),
                        pattern.kind,
                        match.group("version"),
                    )
                )

    return sorted(carriers, key=lambda item: (item.path, item.line, item.kind))


def run(
    root: Path,
    *,
    tracked_files: Iterable[str] | None = None,
    stream: TextIO = sys.stdout,
    required_kinds: Iterable[str] = REQUIRED_KINDS,
) -> int:
    expected = canonical_version(root)
    carriers = enumerate_carriers(root, tracked_files)
    for carrier in carriers:
        print(
            f"{carrier.path}:{carrier.line}: {carrier.kind}: {carrier.version}",
            file=stream,
        )

    failed = False
    found_kinds = {carrier.kind for carrier in carriers}
    for kind in required_kinds:
        if kind not in found_kinds:
            failed = True
            print(
                f"FAIL: no {kind} carrier was found; a carrier that is missing, "
                "unreadable, or no longer matches its pattern cannot be checked",
                file=stream,
            )

    if len(carriers) < 2:
        failed = True
        print("FAIL: fewer than two version carriers were discovered", file=stream)

    mismatches = [carrier for carrier in carriers if carrier.version != expected]
    if mismatches:
        failed = True
        for carrier in mismatches:
            print(
                f"FAIL: {carrier.path}:{carrier.line} carries {carrier.version}; "
                f"pyproject.toml carries {expected}",
                file=stream,
            )
        print(f"FAIL: {len(mismatches)} of {len(carriers)} version carriers disagree", file=stream)

    if failed:
        return 1
    print(f"OK: {len(carriers)} version carriers agree on {expected}", file=stream)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    return run(args.root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
