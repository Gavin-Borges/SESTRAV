"""Catch divergence between the shipped v5 corpus and the code that builds it.

Why this file exists: the tracked corpus was built at 2026-07-05T04:31:30Z, which is
the only timestamp its provenance sidecar records, from `scripts/build_dataset_v5.py`
at commit be3e260. Thirty five hours and forty three minutes later, commit e8d1010
added one line to `VIRUS_FAMILY_MAP`:

    "IBV": "Orthomyxoviridae",

That commit is an ancestor of `main`, so the code carries the fix and the artifact
does not. The consequence is not cosmetic. `apply_quarantine` quarantines any virus
whose `virus_family` is null, so all 52 IBV rows are quarantined in the shipped
corpus. Rebuilding with today's code resolves their family and releases them into
the active pool: **35,597 -> 35,649 active rows, +52 (6 positive, 46 negative)**,
measured 2026-09-17 by rebuilding into a scratch path and diffing column by column.
Exactly three columns differed, and only `reference_pmid` (18,671 rows, every one a
pure `N.0 -> N` reformat) was expected.

This matters because 35,597 is a published figure, appearing in `ARCHITECTURE.md`
five times and in `CHANGELOG.md` repeatedly, and because it means the pending
re-emission of the PMID-bearing artifacts is NOT the cosmetic string reformat it was
planned as. Nothing tested the relationship: `tests/test_published_metrics_lockstep.py`
explicitly excludes the corpus counts from its lockstep set.

These tests do not rebuild anything. They compare tracked code against tracked data,
so they run in a clone and in CI, where `data/immunogenicity_dataset_v4.csv` (the
rebuild's required input) is absent because it is gitignored.

DATE THIS ARTIFACT FROM THE SIDECAR, NOT FROM `git log`. An earlier draft of this
docstring put the build at "2026-07-05 23:43" and the fix "twelve and a half hours
later". Both numbers are real and both are measured against the wrong clock: 23:43
local is be3e260's COMMITTER date, and twelve and a half hours is the true gap from
that date to e8d1010. be3e260 was authored 23 minutes BEFORE this corpus was built
and committed 23 hours and 11 minutes AFTER it, so its committer date post-dates the
artifact it produced. A commit's dates cannot date an artifact that commit wrote.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPO_ROOT / "scripts" / "build_dataset_v5.py"
CORPUS_PATH = REPO_ROOT / "data" / "immunogenicity_dataset_v5.csv"
ARCHITECTURE_PATH = REPO_ROOT / "ARCHITECTURE.md"

# Viruses the map resolves that the shipped corpus leaves null. This is a RATCHET:
# it records a known, dated divergence so that a NEW one cannot appear unnoticed.
# Clear an entry when the corpus is regenerated, not before, and update the active
# row count in ARCHITECTURE.md and CHANGELOG.md in the same change.
KNOWN_FAMILY_DIVERGENCE = {"IBV"}


def _virus_family_map() -> dict[str, str]:
    """Read VIRUS_FAMILY_MAP without importing the builder's dependency stack."""
    tree = ast.parse(BUILDER_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
            if isinstance(target, ast.Name) and target.id == "VIRUS_FAMILY_MAP":
                assert node.value is not None
                return dict(ast.literal_eval(node.value))
    raise AssertionError("VIRUS_FAMILY_MAP not found as a literal in build_dataset_v5.py")


def _corpus() -> pd.DataFrame:
    """Read the corpus as text so no value is normalised away by dtype inference."""
    assert CORPUS_PATH.is_file(), f"tracked corpus missing: {CORPUS_PATH}"
    return pd.read_csv(CORPUS_PATH, dtype=str, keep_default_na=False, low_memory=False)


def _diverged_viruses(df: pd.DataFrame, mapping: dict[str, str]) -> dict[str, int]:
    """Viruses the map resolves but whose corpus rows carry no family."""
    out: dict[str, int] = {}
    for virus in mapping:
        rows = df[df["virus"] == virus]
        if rows.empty:
            continue
        blank = int((rows["virus_family"] == "").sum())
        if blank:
            out[virus] = blank
    return out


def test_no_new_virus_family_divergence() -> None:
    """A map entry added after the corpus was built must not go unrecorded.

    This is the general form of the IBV defect. Any virus added to
    VIRUS_FAMILY_MAP resolves a family on the next build and, through
    apply_quarantine, can move rows into the active pool. Catching it here is
    cheap; catching it after a retrain is not.
    """
    df = _corpus()
    mapping = _virus_family_map()
    diverged = _diverged_viruses(df, mapping)
    unexpected = {v: n for v, n in diverged.items() if v not in KNOWN_FAMILY_DIVERGENCE}
    assert not unexpected, (
        f"VIRUS_FAMILY_MAP resolves {sorted(unexpected)} but the shipped corpus leaves "
        f"their virus_family null ({unexpected}). The corpus predates that map entry, so "
        f"a rebuild would change which rows are quarantined. Either regenerate the corpus "
        f"and update the published active row count, or add the virus to "
        f"KNOWN_FAMILY_DIVERGENCE with the commit that introduced it."
    )


def test_corpus_family_never_contradicts_the_map() -> None:
    """A non-null family in the corpus must agree with what the map would assign."""
    df = _corpus()
    mapping = _virus_family_map()
    contradictions: dict[str, dict[str, int]] = {}
    for virus, family in mapping.items():
        rows = df[df["virus"] == virus]
        if rows.empty:
            continue
        wrong = {f: n for f, n in Counter(rows["virus_family"]).items() if f and f != family}
        if wrong:
            contradictions[virus] = wrong
    assert not contradictions, (
        f"the corpus assigns a different virus_family than VIRUS_FAMILY_MAP would: "
        f"{contradictions}"
    )


def test_the_known_divergence_is_still_exactly_what_was_recorded() -> None:
    """Fails once the corpus is regenerated, which is the point.

    A stale allowlist entry is the failure mode the repo records for every other
    named-path config, so this pins the entry to its measured extent rather than
    letting it outlive the divergence it describes.
    """
    df = _corpus()
    diverged = _diverged_viruses(df, _virus_family_map())
    assert diverged == {"IBV": 52}, (
        f"the recorded divergence has changed: expected {{'IBV': 52}}, measured {diverged}. "
        f"If the corpus was regenerated, clear KNOWN_FAMILY_DIVERGENCE and update the "
        f"published active row count in ARCHITECTURE.md and CHANGELOG.md."
    )


def test_published_active_row_count_matches_the_corpus() -> None:
    """Bind the certified 35,597 to the artifact it describes.

    Nothing did: test_published_metrics_lockstep.py names the corpus counts only to
    say they are outside its lockstep set. Read from ARCHITECTURE.md rather than
    hardcoded, so the assertion fails whichever side drifts.
    """
    df = _corpus()
    measured = int((df["is_quarantined"].str.lower() == "false").sum())

    text = ARCHITECTURE_PATH.read_text(encoding="utf-8")
    published = {
        int(m.replace(",", ""))
        for m in re.findall(r"([\d,]{4,})[ -]active[ -]row", text)
    }
    assert published, "ARCHITECTURE.md no longer states an active row count in a parsed form"
    assert published == {measured}, (
        f"ARCHITECTURE.md publishes active row count(s) {sorted(published)} but "
        f"data/immunogenicity_dataset_v5.csv carries {measured}."
    )


def test_releasing_the_known_divergence_would_change_the_published_count() -> None:
    """Documents the consequence so it cannot be rediscovered the expensive way."""
    df = _corpus()
    active = int((df["is_quarantined"].str.lower() == "false").sum())
    held = _diverged_viruses(df, _virus_family_map())
    assert active == 35597
    assert active + sum(held.values()) == 35649, (
        "the arithmetic behind the recorded 35,597 -> 35,649 consequence no longer holds; "
        "re-measure before citing either number."
    )


@pytest.mark.parametrize("virus", sorted(KNOWN_FAMILY_DIVERGENCE))
def test_every_allowlisted_virus_is_actually_in_the_map(virus: str) -> None:
    """An allowlist entry for a virus the map cannot resolve would be meaningless."""
    assert virus in _virus_family_map(), (
        f"{virus} is allowlisted as a known divergence but VIRUS_FAMILY_MAP has no entry "
        f"for it, so the allowlist is describing something that cannot happen."
    )
