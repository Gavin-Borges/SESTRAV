#!/usr/bin/env python
"""Fail-closed check that hash-pinned lockfiles are fresh and fully hashed.

Read-only, stdlib-only. Compares each ``requirements*.in`` source file
against its compiled ``requirements*.txt`` / ``.lock`` output and fails if:

1. A package the ``.in`` file pins with ``==`` is missing from the compiled
   output, or is present at a different version (the classic staleness bug:
   someone bumped a version in the ``.in`` file and forgot to rerun
   pip-compile). Pins guarded by an environment marker, e.g.
   ``nvidia-nccl-cu12==X ; platform_system == "Linux"``, are only checked
   when present, since their resolution is legitimately platform-dependent.
2. A package the ``.in`` file floors with ``>=`` (the CVE-override pattern
   used in environments/requirements-lock.in) is present in the compiled
   output below that floor.
3. Any package entry in a compiled output has no ``--hash=sha256:`` pin
   (the ``--require-hashes`` install path this repo uses everywhere treats
   an unhashed entry as a supply-chain gap).
4. A ``requirements*.in`` file exists in the repo with no mapping in
   LOCKFILE_PAIRS below (fail closed rather than silently skip a new file).

This performs no network access and does not invoke pip-compile: it is a
static diff between declared intent (``.in``) and committed output
(``.txt`` / ``.lock``), which is exactly the drift a "bump the .in file,
forget to regenerate" PR produces. It intentionally does not second-guess
*what* pip-compile would resolve transitive versions to; it only holds the
direct pins in the ``.in`` files to their word.

Usage:
    python tools/check_lockfile_freshness.py          # human-readable report
    python tools/check_lockfile_freshness.py --check  # same, exit 1 on drift

CI usage (see .github/workflows/dependency-lockfile-check.yml):
    python tools/check_lockfile_freshness.py --check
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Explicit .in -> compiled-output mapping. Kept explicit (not glob-inferred)
# because environments/requirements-lock.in compiles to
# environments/requirements.lock, which breaks the simple ".in -> .txt"
# substring convention every other pair in this repo follows. If you add a
# new requirements*.in file, add its mapping here - discover_unmapped_in_files()
# fails the build if you forget, rather than silently skipping it.
LOCKFILE_PAIRS: tuple[tuple[str, str], ...] = (
    ("requirements.in", "requirements.txt"),
    ("environments/requirements-lock.in", "environments/requirements.lock"),
    ("environments/requirements-ci.in", "environments/requirements-ci.txt"),
    ("environments/requirements-ci-build.in", "environments/requirements-ci-build.txt"),
    ("environments/requirements-ci-mypy.in", "environments/requirements-ci-mypy.txt"),
    (
        "environments/requirements-ci-pytest-cov.in",
        "environments/requirements-ci-pytest-cov.txt",
    ),
    ("environments/requirements-ci-ruff.in", "environments/requirements-ci-ruff.txt"),
    ("environments/requirements-pip-audit.in", "environments/requirements-pip-audit.txt"),
    ("environments/requirements-security.in", "environments/requirements-security.txt"),
    ("environments/requirements-semgrep.in", "environments/requirements-semgrep.txt"),
)

_NAME_NORMALIZE_RE = re.compile(r"[-_.]+")
_PIN_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(==|>=)\s*([0-9][A-Za-z0-9.\-+]*)"
)
_INCLUDE_RE = re.compile(r"^-r\s+(\S+)")
_COMPILED_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\]+)")
_HASH_RE = re.compile(r"--hash=sha256:[0-9a-fA-F]{64}")


def normalize(name: str) -> str:
    """PEP 503 style normalization so pip-compile's casing never matters."""
    return _NAME_NORMALIZE_RE.sub("-", name).lower()


def _version_key(version: str) -> tuple[int, ...]:
    """Best-effort numeric ordering for simple X.Y.Z-style pins.

    Good enough for the ``>=`` CVE-floor overrides in this repo, all of which
    are plain dotted-numeric releases. Not a full PEP 440 comparator.
    """
    parts = re.findall(r"\d+", version)
    return tuple(int(p) for p in parts) if parts else (0,)


@dataclass
class Pin:
    name: str
    operator: str  # "==" or ">="
    version: str
    marker: bool  # True if the line carried an environment marker
    source: str  # relative path, for error messages


@dataclass
class CompiledEntry:
    version: str
    has_hash: bool


def parse_in_file(path: Path, _visited: set[Path] | None = None) -> dict[str, Pin]:
    """Parse a requirements*.in file, following -r includes recursively."""
    if _visited is None:
        _visited = set()
    resolved = path.resolve()
    if resolved in _visited:
        return {}
    _visited.add(resolved)

    pins: dict[str, Pin] = {}
    text = path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        include_match = _INCLUDE_RE.match(line)
        if include_match:
            included = (path.parent / include_match.group(1)).resolve()
            if included.exists():
                pins.update(parse_in_file(included, _visited))
            continue

        # Strip a trailing inline comment before pin/marker parsing.
        code = line.split(" #", 1)[0].strip()
        if not code:
            continue

        pin_match = _PIN_RE.match(code)
        if not pin_match:
            continue
        name, operator, version = pin_match.groups()
        marker = ";" in code
        try:
            rel_source = path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel_source = str(path)
        pins[normalize(name)] = Pin(
            name=name, operator=operator, version=version, marker=marker, source=rel_source
        )
    return pins


def parse_compiled_file(path: Path) -> dict[str, CompiledEntry]:
    """Parse a compiled requirements*.txt/.lock output for name/version/hash."""
    lines = path.read_text(encoding="utf-8").splitlines()
    entries: dict[str, CompiledEntry] = {}
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line and not line[0].isspace() and not line.startswith("#"):
            match = _COMPILED_NAME_RE.match(line)
            if match:
                name, version = match.groups()
                has_hash = bool(_HASH_RE.search(line))
                i += 1
                while i < n and lines[i][:1] in (" ", "\t"):
                    if _HASH_RE.search(lines[i]):
                        has_hash = True
                    i += 1
                entries[normalize(name)] = CompiledEntry(version=version, has_hash=has_hash)
                continue
        i += 1
    return entries


def check_pair(in_path: Path, out_path: Path) -> list[str]:
    """Return a list of human-readable problems for one .in/compiled pair."""
    problems: list[str] = []
    try:
        rel_in = in_path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel_in = str(in_path)
    try:
        rel_out = out_path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel_out = str(out_path)

    if not in_path.exists():
        problems.append(f"{rel_in}: source .in file is missing")
        return problems
    if not out_path.exists():
        problems.append(
            f"{rel_out}: compiled output is missing (expected, sourced from {rel_in})"
        )
        return problems

    pins = parse_in_file(in_path)
    compiled = parse_compiled_file(out_path)

    for key, pin in sorted(pins.items()):
        entry = compiled.get(key)
        if pin.operator == "==":
            if entry is None:
                if pin.marker:
                    # Platform-conditional pin legitimately absent on this
                    # resolve platform; nothing to compare.
                    continue
                problems.append(
                    f"{rel_out}: {pin.name}=={pin.version} is pinned in {pin.source} "
                    f"but missing from the compiled output (stale - regenerate)"
                )
                continue
            if entry.version != pin.version:
                problems.append(
                    f"{rel_out}: {pin.name}=={pin.version} in {pin.source} but "
                    f"{pin.name}=={entry.version} in the compiled output "
                    f"(stale - regenerate)"
                )
        elif pin.operator == ">=":
            if entry is None:
                continue
            if _version_key(entry.version) < _version_key(pin.version):
                problems.append(
                    f"{rel_out}: {pin.name}>={pin.version} floor declared in "
                    f"{pin.source} but compiled output pins {pin.name}=={entry.version} "
                    f"(below floor - stale or the CVE override was dropped)"
                )

    for key, entry in sorted(compiled.items()):
        if not entry.has_hash:
            problems.append(
                f"{rel_out}: {key}=={entry.version} has no --hash pins (unhashed - "
                f"regenerate with --generate-hashes)"
            )

    return problems


def discover_unmapped_in_files(mapped: set[str]) -> list[str]:
    """Fail closed: any requirements*.in file not in LOCKFILE_PAIRS is a gap."""
    unmapped: list[str] = []
    skip_dirs = {".git", ".venv", ".ci_test_venv", "node_modules", "_local", "results"}
    for path in REPO_ROOT.rglob("requirements*.in"):
        if any(part in skip_dirs for part in path.parts):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel not in mapped:
            unmapped.append(rel)
    return sorted(unmapped)


def _annotate(message: str) -> str:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return f"::error::{message}"
    return f"ERROR: {message}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="exit 1 on any staleness/hash gap (CI mode)"
    )
    args = parser.parse_args()

    mapped_in_files = {pair[0] for pair in LOCKFILE_PAIRS}
    problems: list[str] = []

    unmapped = discover_unmapped_in_files(mapped_in_files)
    for rel in unmapped:
        problems.append(
            f"{rel}: a requirements*.in file exists with no LOCKFILE_PAIRS mapping in "
            f"tools/check_lockfile_freshness.py - add it so it is covered by this gate"
        )

    checked = 0
    for in_rel, out_rel in LOCKFILE_PAIRS:
        in_path = REPO_ROOT / in_rel
        out_path = REPO_ROOT / out_rel
        problems.extend(check_pair(in_path, out_path))
        checked += 1

    if problems:
        print(f"Checked {checked} lockfile pair(s); found {len(problems)} problem(s):\n")
        for problem in problems:
            print(_annotate(problem))
        print(
            "\nRegenerate the affected lockfile(s) locally (pip-compile --allow-unsafe "
            "--generate-hashes ...; see the header comment of each .txt/.lock file for "
            "its exact command) and push the result yourself. This check never writes "
            "or pushes anything."
        )
        return 1

    print(f"All {checked} lockfile pair(s) are fresh and fully hashed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
