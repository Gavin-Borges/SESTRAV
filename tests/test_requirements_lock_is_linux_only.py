"""requirements.txt is a Linux x86-64 lock, and README must say so.

The compiled lock pins CUDA packages with NO environment markers, so pip attempts
every pin on every platform. Measured on Windows: the install command README
documents exits 1 with "No matching distribution found for nvidia-nccl-cu12",
while a cross-platform pin from the same file resolves normally.

These tests bind the README's disclosure to the artifact it describes. If the
compile step is ever fixed to emit markers, the first test fails and forces the
README note to be revisited rather than left standing as a stale warning.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "requirements.txt"
README = REPO_ROOT / "README.md"

# A pinned requirement line, e.g. "nvidia-nccl-cu12==2.30.7 \"
PIN = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==", re.MULTILINE)
# An environment marker on a requirement line, e.g. "; platform_system == \"Linux\""
MARKER = re.compile(
    r"^[^#\s].*;\s*.*\b(platform_system|sys_platform|platform_machine|python_version)\b",
    re.MULTILINE,
)
# Packages that ship Linux-only CUDA wheels.
CUDA_PREFIXES = ("nvidia-", "triton")


def _requirements_text() -> str:
    return REQUIREMENTS.read_text(encoding="utf-8")


def _cuda_pins(text: str) -> list[str]:
    return sorted(
        m.group("name")
        for m in PIN.finditer(text)
        if m.group("name").lower().startswith(CUDA_PREFIXES)
    )


def test_the_lock_pins_cuda_packages_with_no_environment_markers():
    """The measured state this README note exists to describe.

    FAILS IF: the compile step starts emitting markers, or the CUDA pins are
    dropped. Either is a real change and both mean the README note must be
    re-read rather than silently kept.
    """
    text = _requirements_text()
    cuda = _cuda_pins(text)
    markers = MARKER.findall(text)

    assert cuda, "expected CUDA pins in requirements.txt; found none"
    assert "triton" in cuda
    assert sum(1 for name in cuda if name.startswith("nvidia-")) >= 10
    assert markers == [], (
        "requirements.txt now carries environment markers "
        f"({len(markers)} found). The README note calling it a Linux-only lock "
        "may no longer be accurate - re-measure before changing it."
    )


def test_readme_discloses_the_platform_constraint():
    """The disclosure must exist whenever the unmarked CUDA pins do.

    FAILS IF: the note is deleted or reworded past recognition while the lock
    still carries unmarked CUDA pins. That is the exact regression this guards:
    a documented command that cannot run on the reader's machine, with nothing
    saying so.
    """
    text = _requirements_text()
    if not _cuda_pins(text) or MARKER.findall(text):
        return  # premise gone; the other test reports it

    readme = README.read_text(encoding="utf-8")
    assert "requirements.txt" in readme
    assert "Linux x86-64 lock" in readme, (
        "README no longer discloses that requirements.txt is a Linux-only lock, "
        "but the lock still pins CUDA packages with no environment markers."
    )
    assert "nvidia-nccl-cu12" in readme, (
        "README no longer names the observed failure, so a reader hitting it "
        "cannot match the error text to the explanation."
    )


def test_readme_offers_a_path_for_other_platforms():
    """A constraint without an alternative is a dead end.

    FAILS IF: the note says the lock is Linux-only but stops there.
    """
    readme = README.read_text(encoding="utf-8")
    assert "Linux x86-64 lock" in readme
    tail = readme.split("Linux x86-64 lock", 1)[1][:1200]
    assert "conda" in tail and "editable" in tail, (
        "README states the Linux-only constraint but does not name the conda or "
        "editable-install path that works on macOS and Windows."
    )
