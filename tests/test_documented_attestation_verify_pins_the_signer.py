"""Every documented `gh attestation verify` command pins the signer and the tag.

`--repo` alone accepts an attestation signed by ANY workflow in the repository,
at any ref. The release attestation is signed by .github/workflows/release.yml on
a version tag, and `gh attestation verify` can enforce both: `--signer-workflow`
checks the workflow path and `--source-ref` the git ref. Measured 2026-10-19 on
the published v2.0.2 and v2.0.3 wheels and the v2.0.3 sdist: the pinned command
exits 0 for each, while the same command naming ci.yml as the signer, or v2.0.2
as the ref for the v2.0.3 wheel, exits 1. So the extra flags cost a consumer
nothing on a genuine release and refuse an attestation from any other workflow
or ref.

FAILS IF: a tracked Markdown file shows a `gh attestation verify <artifact>`
command line without both flags, or names a signer other than release.yml.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SIGNER = "--signer-workflow Gavin-Borges/SESTRAV/.github/workflows/release.yml"


def _command_lines() -> list[tuple[str, int, str]]:
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")
    found = []
    for raw in tracked:
        if not raw:
            continue
        rel = raw.decode("utf-8")
        # The changelog quotes old commands as history; it is not an instruction.
        if rel == "CHANGELOG.md":
            continue
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            # A command line, not prose that names the command in backticks.
            if line.strip().startswith("gh attestation verify "):
                found.append((rel, number, line.strip()))
    return found


def test_the_documented_commands_exist() -> None:
    # Without this, deleting every command would pass the test below vacuously.
    assert len(_command_lines()) >= 2


def test_every_documented_verify_command_pins_signer_and_ref() -> None:
    unpinned = [
        f"{rel} line {number}: {line}"
        for rel, number, line in _command_lines()
        if SIGNER not in line or "--source-ref refs/tags/v" not in line
    ]
    assert unpinned == [], (
        "Each `gh attestation verify` instruction must carry "
        f"`{SIGNER}` and `--source-ref refs/tags/v<version>`; "
        "`--repo` alone accepts any workflow in the repository:\n" + "\n".join(unpinned)
    )
