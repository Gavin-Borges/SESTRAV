"""Every doc that PRESENTS the GNN Gate 1 figure must say it is the best of eight.

`docs/gnn_gate_retry_preregistration.md` section 1.1 records that the quoted
0.6458 and +0.0402 are the maximum of an n=8 series, that Gate 1 passes 0 of 8,
and that the 0.0042 deficit is the series smallest against a mean of 0.0122.

That qualifier changes what a reader concludes. Without it, "FAIL by 0.0042"
reads as one tweak away from passing, when no run in the series passed and the
mean deficit is nearly three times larger.

This guard exists because the correction had to touch SEVEN documents, and a
review then found three more places: the D3 row of docs/claims_register.md, a
notebook, and the src/verify/promote_gnn.py docstring. This file now guards the
notebook. It cannot guard the D3 row, which carries the deficit and the delta but
not 0.6458, in a file that carries the qualifier on another row; and it does not
read Python, so the docstring is left to the change that qualifies it. The repo
has already shipped a correction that handled three of five carriers and left two
standing (incident #9 in .claude/rules/third-party-claims-cases.md). Enumerating
the carriers this file can see is what stops the next edit from re-opening them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FIGURE = "0.6458"

# Files that PRESENT the figure as a current result and must carry the qualifier.
PRESENTING = (
    "README.md",
    "ARCHITECTURE.md",
    "ROADMAP.md",
    "docs/paper.md",
    "docs/architecture/gnn_models.md",
    "docs/Wet_Lab_Protocol_v1.md",
    "docs/claims_register.md",
)

# Deliberately exempt, each for a stated reason rather than by convenience.
EXEMPT = {
    # A dated entry recording what was reported on that date. Amending it would
    # rewrite the record rather than correct a live claim.
    "CHANGELOG.md",
    # This file IS the disclosure; requiring it to cite itself is circular.
    "docs/gnn_gate_retry_preregistration.md",
}

DISCLOSURE_TOKENS = ("best of eight", "best of EIGHT", "most favourable of eight")


def _says_best_of_eight(text: str) -> bool:
    return any(token in text for token in DISCLOSURE_TOKENS)


@pytest.mark.parametrize("relative", PRESENTING)
def test_presenting_carrier_discloses_the_series(relative: str):
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    assert FIGURE in text, (
        f"{relative} no longer contains {FIGURE}. If the figure was removed, drop this file "
        "from PRESENTING; do not leave the guard asserting against a file that cannot fail."
    )
    assert _says_best_of_eight(text), (
        f"{relative} presents Gate 1 = {FIGURE} without disclosing it is the best of eight "
        "runs. Across the series Gate 1 passes 0 of 8 and the mean deficit is 0.0122, so the "
        "unqualified figure makes the bar look nearer than the evidence supports. See "
        "docs/gnn_gate_retry_preregistration.md section 1.1."
    )


def _tracked_documents() -> list[str]:
    """Tracked Markdown and notebook paths, from git rather than a filesystem walk.

    A walk would also read untracked and gitignored files, such as a maintainer's
    private notes, and fail the pre-push test run on content that is never published.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", "*.md", "*.ipynb"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git cannot list tracked files here")
    return [p for p in result.stdout.decode("utf-8").split("\0") if p]


def test_no_undisclosed_carrier_appears_outside_the_known_set():
    """A new tracked doc or notebook quoting the figure must disclose, or be EXEMPT."""
    known = set(PRESENTING) | EXEMPT
    offenders = []
    for rel in _tracked_documents():
        if rel in known:
            continue
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - listed but unreadable (e.g. sparse checkout)
            continue
        if FIGURE in text and not _says_best_of_eight(text):
            offenders.append(rel)
    assert not offenders, (
        "these documents quote the Gate 1 figure without the best-of-eight qualifier:\n  "
        + "\n  ".join(offenders)
        + "\n\nAdd the disclosure, or add the file to EXEMPT with the reason it does not "
        "present the figure as a current result."
    )
