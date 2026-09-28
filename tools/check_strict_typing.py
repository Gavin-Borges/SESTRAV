#!/usr/bin/env python
"""Strict-typing ratchet: files that are clean under ``mypy --strict`` stay clean.

``[tool.mypy]`` in pyproject.toml is deliberately non-strict while legacy code is
annotated incrementally, and CI's lint job runs ``mypy src/`` under that config.
Nothing stopped a module that is ALREADY strict-clean from regressing, because a
non-strict run cannot see the regression at all. This gate closes that gap for an
explicit list of files, ``docs/strict_typing_ratchet.json``:

* every listed path must exist (a stale entry fails, as it does for
  ``.coveragerc.library``), be a relative ``.py`` path, and appear once, in
  sorted order;
* ``mypy --strict`` over the listed files must report zero errors located in
  them. pyproject's ``[tool.mypy]`` still applies, so ``ignore_missing_imports``
  and ``warn_unused_ignores`` keep their repository values.

Imports are followed SILENTLY (``--follow-imports=silent``): an unlisted module's
own errors never fail this gate, but its types still inform the listed files.

The check fails closed rather than passing vacuously: an empty list, a mypy exit
status other than 0 or 1, a failing status with no parsable error line, or a
mypy summary that names a different number of checked files than were listed
all exit 1.

The ratchet turns one way. Add a file once it is strict-clean (``--list-clean``
prints the candidates); removing one is a deliberate, reviewable edit.

Third-party packages: mypy resolves them from the interpreter that runs it, or
from ``--python-executable``. CI's lint job installs only mypy, so there every
third-party import is ``Any``; a development environment sees real types. A
listed file should be clean under both, which is why the list is measured both
ways before a file is added.

Usage:
    python tools/check_strict_typing.py --check
    python tools/check_strict_typing.py --list-clean
    python tools/check_strict_typing.py --check --python-executable /path/to/python

Exit codes: 0 clean; 1 regression, stale or malformed entry, or a vacuous run;
2 mypy is not importable by this interpreter.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import subprocess  # nosec B404 - fixed argv to this interpreter's mypy, never a shell
import sys
from collections import Counter
from dataclasses import dataclass, field

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RATCHET = REPO_ROOT / "docs" / "strict_typing_ratchet.json"
SOURCE_ROOTS = ("src", "functions")

MYPY_BASE_ARGS = (
    "--strict",
    "--follow-imports=silent",
    "--no-pretty",
    "--no-color-output",
    "--show-error-codes",
)

# One mypy diagnostic: a path, a line, an optional column, then the severity.
DIAGNOSTIC_RE = re.compile(
    r"^(?P<path>[^\s:][^:]*?):(?P<line>\d+)(?::\d+)?: (?P<severity>error|note): (?P<message>.*)$"
)
# "Success: no issues found in N source files" or "... (checked N source files)".
CHECKED_RE = re.compile(r"(?:found in|checked) (?P<n>\d+) source files?")


class RatchetError(Exception):
    """The ratchet file is unreadable or malformed."""


@dataclass
class MypyResult:
    returncode: int
    output: str


@dataclass
class Verdict:
    ok: bool
    messages: list[str] = field(default_factory=list)
    errors_by_file: Counter[str] = field(default_factory=Counter)


def load_ratchet(path: pathlib.Path) -> list[str]:
    """Return the listed paths, validating shape, order and uniqueness."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RatchetError(f"cannot read {path.name}: {exc}") from exc
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
        raise RatchetError(f'{path.name} must be an object whose "files" is a list of strings')
    problems: list[str] = []
    for entry in files:
        if not entry.endswith(".py"):
            problems.append(f"not a .py path: {entry}")
        if entry.startswith(("/", "\\")) or ".." in entry.split("/") or "\\" in entry:
            problems.append(f"not a relative posix path: {entry}")
    duplicates = sorted(f for f, n in Counter(files).items() if n > 1)
    if duplicates:
        problems.append(f"listed more than once: {', '.join(duplicates)}")
    if files != sorted(files):
        problems.append("entries are not in sorted order")
    if problems:
        raise RatchetError("; ".join(problems))
    return files


def normalise(path: str) -> str:
    """Map a mypy-reported path to the posix form used in the ratchet file."""
    posix = path.replace("\\", "/")
    return posix[2:] if posix.startswith("./") else posix


def parse_output(output: str) -> tuple[Counter[str], int | None]:
    """Count error diagnostics per file and read the checked-file total, if printed."""
    errors: Counter[str] = Counter()
    checked: int | None = None
    for line in output.splitlines():
        match = DIAGNOSTIC_RE.match(line.strip())
        if match and match.group("severity") == "error":
            errors[normalise(match.group("path"))] += 1
            continue
        summary = CHECKED_RE.search(line)
        if summary and (line.startswith("Success:") or line.startswith("Found ")):
            checked = int(summary.group("n"))
    return errors, checked


def mypy_available() -> bool:
    return importlib.util.find_spec("mypy") is not None


def run_mypy(
    files: list[str], repo_root: pathlib.Path, python_executable: str | None = None
) -> MypyResult:
    """Run this interpreter's mypy over ``files`` from ``repo_root``."""
    argv = [sys.executable, "-m", "mypy", *MYPY_BASE_ARGS]
    if python_executable:
        argv.append(f"--python-executable={python_executable}")
    argv.extend(files)
    proc = subprocess.run(  # nosec B603 - fixed argv, shell=False
        argv, cwd=repo_root, capture_output=True, text=True, check=False
    )
    return MypyResult(proc.returncode, proc.stdout + proc.stderr)


def evaluate(listed: list[str], result: MypyResult) -> Verdict:
    """Decide the verdict from one mypy run over exactly the listed files."""
    errors, checked = parse_output(result.output)
    listed_set = set(listed)
    regressions: Counter[str] = Counter({f: n for f, n in errors.items() if f in listed_set})
    verdict = Verdict(ok=True, errors_by_file=regressions)
    if result.returncode not in (0, 1):
        verdict.ok = False
        verdict.messages.append(f"mypy exited {result.returncode}; treating the run as failed")
    if result.returncode == 1 and not errors:
        verdict.ok = False
        verdict.messages.append("mypy reported failure but no error line could be parsed")
    if result.returncode == 0 and errors:
        verdict.ok = False
        verdict.messages.append("mypy exited 0 yet printed error lines; output is inconsistent")
    if checked is None:
        verdict.ok = False
        verdict.messages.append(
            "mypy printed no checked-file summary; cannot rule out a vacuous run"
        )
    elif checked != len(listed):
        verdict.ok = False
        verdict.messages.append(f"mypy checked {checked} source files but {len(listed)} are listed")
    if regressions:
        verdict.ok = False
        for path in sorted(regressions):
            verdict.messages.append(f"REGRESSED: {path} has {regressions[path]} strict error(s)")
    return verdict


def source_files(repo_root: pathlib.Path) -> list[str]:
    found: list[str] = []
    for root in SOURCE_ROOTS:
        base = repo_root / root
        if base.is_dir():
            for path in base.rglob("*.py"):
                if "__pycache__" not in path.parts:
                    found.append(path.relative_to(repo_root).as_posix())
    return sorted(found)


def check(ratchet: pathlib.Path, repo_root: pathlib.Path, python_executable: str | None) -> int:
    try:
        listed = load_ratchet(ratchet)
    except RatchetError as exc:
        print(f"FAIL: {exc}")
        return 1
    if not listed:
        print(f"FAIL: {ratchet.name} lists no files; an empty ratchet certifies nothing")
        return 1
    missing = [f for f in listed if not (repo_root / f).is_file()]
    if missing:
        for path in missing:
            print(f"FAIL: listed file does not exist (stale entry): {path}")
        return 1
    result = run_mypy(listed, repo_root, python_executable)
    verdict = evaluate(listed, result)
    if not verdict.ok:
        print(result.output.rstrip())
        for message in verdict.messages:
            print(f"FAIL: {message}")
        print(f"strict-typing ratchet: FAILED over {len(listed)} listed file(s)")
        return 1
    print(f"strict-typing ratchet: OK, {len(listed)} listed file(s) are strict-clean")
    return 0


def list_clean(
    ratchet: pathlib.Path, repo_root: pathlib.Path, python_executable: str | None
) -> int:
    try:
        listed = set(load_ratchet(ratchet))
    except RatchetError as exc:
        print(f"FAIL: {exc}")
        return 1
    files = source_files(repo_root)
    result = run_mypy(files, repo_root, python_executable)
    errors, checked = parse_output(result.output)
    if result.returncode not in (0, 1) or checked != len(files):
        print(result.output.rstrip())
        print(f"FAIL: mypy exited {result.returncode}, checked {checked} of {len(files)} files")
        return 1
    clean = [f for f in files if errors[f] == 0]
    print(f"{sum(errors.values())} strict error(s) in {len(errors)} of {len(files)} files")
    for path in clean:
        print(f"{'listed  ' if path in listed else 'UNLISTED'} {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict-typing ratchet gate.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail on any regression (default)")
    mode.add_argument("--list-clean", action="store_true", help="print strict-clean files")
    parser.add_argument("--ratchet", type=pathlib.Path, default=RATCHET)
    parser.add_argument("--repo-root", type=pathlib.Path, default=REPO_ROOT)
    parser.add_argument("--python-executable", default=None)
    args = parser.parse_args(argv)
    if not mypy_available():
        print("error: mypy is not importable by this interpreter; install it to run this gate")
        return 2
    if args.list_clean:
        return list_clean(args.ratchet, args.repo_root, args.python_executable)
    return check(args.ratchet, args.repo_root, args.python_executable)


if __name__ == "__main__":
    sys.exit(main())
