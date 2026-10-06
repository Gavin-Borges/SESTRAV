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

The Gate 2 half, at the foot of this file, checks the Gate 2 statements that
print no figure, which a qualifier binding keyed on a value cannot see.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from scripts import check_qualifier_bindings as gate

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


# Gate 2 half. Section 1.1 also records that the quoted Gate 2 figure, a
# cross-fold std of 0.0234, is the WORST of the same eight runs, and that Gate 2
# passes 4 of 8 at ddof=0 (3 of 8 at ddof=1). Where a document prints 0.0234,
# the gnn-gate2-worst-of-eight binding in docs/qualifier_bindings.json checks
# the disclosure. The statements pinned below report the Gate 2 FAIL without
# printing the figure, so a binding keyed on a value cannot see them. Each is
# checked with that binding's own qualifier patterns and window, so the two
# guards share one definition of the disclosure. Only these statements are
# pinned: a Gate 2 statement elsewhere that prints no figure is not checked here.
GATE2_BINDING = "gnn-gate2-worst-of-eight"

# (file, a regex that selects exactly one line of it: the line stating the FAIL)
GATE2_STATEMENTS_WITHOUT_THE_FIGURE = (
    ("README.md", r"Gate 2 \(cross-fold std\) FAIL"),
    ("README.md", r"Gate 2 FAIL"),
    ("ARCHITECTURE.md", r"Gates 1 and 2 FAIL on measured values"),
    ("docs/claims_register.md", r"^\| D3 \|"),
    # Pinned by ID and claim text together. The ID cell was added when Sections 2 to 4 of the
    # register gained one; pinning the claim text alone anchored at the line start, which that
    # column silently broke, and only the full suite caught it. Matching the ID loosely keeps a
    # renumbering from breaking the pin, while the claim text still fails on a rewording.
    ("docs/claims_register.md", r'^\| HD[0-9]+ \| "GNN structural scorer" \|'),
)


@pytest.mark.parametrize(
    ("relative", "statement"),
    GATE2_STATEMENTS_WITHOUT_THE_FIGURE,
    ids=(
        "README-pipeline-stage",
        "README-gnn-paragraph",
        "ARCHITECTURE-overview-table",
        "claims_register-D3",
        "claims_register-gnn-scorer",
    ),
)
def test_gate2_statement_without_the_figure_discloses_the_series(relative: str, statement: str):
    config = gate._load_config(ROOT / "docs/qualifier_bindings.json")
    binding = next(item for item in config["bindings"] if item["id"] == GATE2_BINDING)
    lines = (ROOT / relative).read_text(encoding="utf-8").splitlines()
    hits = [number for number, line in enumerate(lines, start=1) if re.search(statement, line)]
    assert len(hits) == 1, (
        f"{statement!r} selects {len(hits)} lines of {relative}, not 1. If the Gate 2 "
        "statement was reworded or moved, re-pin it here; do not drop it unless the "
        "document no longer reports the Gate 2 FAIL."
    )
    window = gate._window(lines, hits[0], binding["window_lines"])
    assert gate._qualified(window, binding["qualifier_patterns"]), (
        f"{relative} line {hits[0]} reports the Gate 2 FAIL without the series disclosure. "
        'Within the window, write "worst of eight" followed by either "passes 4 of 8 at '
        'ddof=0, 3 of 8 at ddof=1" or "passes 4 of 8 at ddof=0 (3 of 8 at ddof=1)", the '
        "forms the gnn-gate2-worst-of-eight binding accepts. See "
        "docs/gnn_gate_retry_preregistration.md section 1.1."
    )
