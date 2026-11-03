"""No tracked file tells a reader to install PANDORA by the PyPI name `pandora`.

scripts/run_pandora_structures.py's setup block said to run a pip install of
`pandora`. On PyPI that name belongs to CNES's stereo-matching framework
(pypi.org/project/pandora, read 2026-10-19: "Pandora is a stereo matching framework
that helps emulate state of the art algorithms"), which has nothing to do with
peptide-MHC modelling. Following the instruction installs another vendor's code
and no PANDORA at all. Upstream PANDORA (github.com/X-lab-3D/PANDORA, README read
2026-10-19) installs from conda as `csb-pandora` on the `csb-nijmegen` channel, or
from a clone with `pip install -e .`.

FAILS IF: a tracked file other than CHANGELOG.md, which records the old instruction
as history, tells a reader to install the bare PyPI name with pip.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_pandora_structures.py"

# Assembled so this module does not match itself. A bare `pandora` token: not
# `csb-pandora`, not `pandora-fetch`, not part of a longer word.
_NAME = "pan" + "dora"
_BARE_INSTALL = re.compile(r"pip3?\s+install\b[^\n#]*(?<![\w.-])" + _NAME + r"(?![\w.-])", re.I)


def _tracked_text_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=str(REPO_ROOT), capture_output=True, check=True
    ).stdout.split(b"\0")
    return [p.decode("utf-8") for p in out if p and p.decode("utf-8") != "CHANGELOG.md"]


def test_no_tracked_file_installs_pandora_by_its_pypi_name() -> None:
    offenders = []
    for rel in _tracked_text_files():
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if _BARE_INSTALL.search(line):
                offenders.append(f"{rel} line {number}")
    assert offenders == [], (
        "PyPI's `pandora` is CNES's stereo-matching framework, not PANDORA. Install "
        "upstream PANDORA as `csb-pandora` from the csb-nijmegen conda channel, or from a "
        "clone with `pip install -e .`:\n" + "\n".join(offenders)
    )


def test_the_script_names_the_upstream_install_routes() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "conda install -c csb-nijmegen csb-pandora" in text
    assert "https://github.com/X-lab-3D/PANDORA.git" in text
    assert "pandora-fetch" in text


def test_the_pattern_catches_the_old_instruction_and_spares_the_upstream_ones() -> None:
    # The old line, and the shapes it could take again.
    assert _BARE_INSTALL.search("       pip install " + _NAME + " pdb2sql mhctools")
    assert _BARE_INSTALL.search("pip3 install --upgrade " + _NAME)
    # Upstream's routes and look-alikes must not trip it.
    assert not _BARE_INSTALL.search("conda install -c csb-nijmegen csb-" + _NAME)
    assert not _BARE_INSTALL.search("pip install csb-" + _NAME)
    assert not _BARE_INSTALL.search("pip install -e .  # inside the " + _NAME.upper() + " clone")
    assert not _BARE_INSTALL.search("pip install " + _NAME + "s-box")
