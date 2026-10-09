"""No tracked file may tell a reader to run a bare MHCflurry model fetch.

`mhcflurry-downloads fetch models_class1_presentation` on its own downloads and
unpacks about 135 MB with no integrity check, and a tampered archive can write
outside the target directory. The verified route is two steps:
`scripts/fetch_verified_mhcflurry.py` checks the pinned sha256, then
`mhcflurry-downloads fetch ... --already-downloaded-dir DIR` unpacks that file.

The model name makes a mention a runnable instruction rather than a warning
("do not run a bare `mhcflurry-downloads fetch`"), so the rule is: every command
naming `models_class1_presentation` must carry `--already-downloaded-dir` in the
same command, with backslash continuations joined. This is a ratchet over the
spelling below, not a proof: a command built from variables, or split across
separate string literals, is not seen. tests/ (fixtures and assertions) and
CHANGELOG.md (history) are out of scope.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FETCH = re.compile(r"mhcflurry-downloads\s+fetch\s+models_class1_presentation")
EXEMPT_PREFIXES = ("tests/",)
EXEMPT_FILES = {"CHANGELOG.md"}


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Join backslash-continued lines; return (first line number, joined text)."""
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    start = 0
    for number, line in enumerate(text.splitlines(), start=1):
        if not buf:
            start = number
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            buf.append(stripped[:-1])
            continue
        buf.append(line)
        out.append((start, " ".join(buf)))
        buf = []
    if buf:
        out.append((start, " ".join(buf)))
    return out


def bare_fetches(text: str) -> list[int]:
    """Line numbers of commands that fetch the model without --already-downloaded-dir."""
    return [
        number
        for number, line in _logical_lines(text)
        if FETCH.search(line) and "--already-downloaded-dir" not in line
    ]


def _tracked_text_files() -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True
    ).stdout.decode("utf-8").split("\0")
    return [
        path
        for path in listed
        if path
        and path not in EXEMPT_FILES
        and not path.startswith(EXEMPT_PREFIXES)
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mhcflurry-downloads fetch models_class1_presentation\n", [1]),
        ('print("Run: mhcflurry-downloads fetch models_class1_presentation")\n', [1]),
        ("# !mhcflurry-downloads fetch models_class1_presentation\n", [1]),
        ("mhcflurry-downloads fetch models_class1_presentation --already-downloaded-dir d\n", []),
        ("mhcflurry-downloads fetch models_class1_presentation \\\n    --already-downloaded-dir d\n", []),
        ("Do not run a bare `mhcflurry-downloads fetch`: it is unverified.\n", []),
        ("x\nmhcflurry-downloads  fetch   models_class1_presentation\n", [2]),
    ],
)
def test_the_detector_sees_each_spelling(text: str, expected: list[int]) -> None:
    assert bare_fetches(text) == expected


def test_no_tracked_file_advises_a_bare_model_fetch() -> None:
    offenders: list[str] = []
    verified_routes = 0
    for path in _tracked_text_files():
        try:
            text = (REPO_ROOT / path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        offenders.extend(f"{path}:{number}" for number in bare_fetches(text))
        verified_routes += sum(
            1
            for _, line in _logical_lines(text)
            if FETCH.search(line) and "--already-downloaded-dir" in line
        )
    # Not vacuous: the walk must reach real content, so it must find the verified route that
    # README, USAGE and singularity.def spell out. An empty file list would pass silently.
    assert verified_routes >= 3, f"the scan found only {verified_routes} verified-route commands"
    assert not offenders, (
        "these lines tell a reader to run an unverified MHCflurry fetch; give the "
        "verified two-step route instead (README, 'MHCflurry model data'):\n"
        + "\n".join(offenders)
    )
