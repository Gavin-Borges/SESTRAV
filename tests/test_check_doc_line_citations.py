"""Adversarial coverage for scripts/check_doc_line_citations.py.

The gate exists to catch line-citation rot. A gate that cannot be shown to FAIL
on the defect it targets is decoration, so the load-bearing test here is
``test_catches_the_real_historical_rot``: it reproduces the actual mechanism of
42de845, which made a NET +3 change (four lines added, one removed) upstream of
two cited sites and moved them from 675 and 941/946 to 678 and 944/949.

Every test that exercises the CLI builds a throwaway git repository, because the
checker resolves the repo root and the tracked-file set through git rather than
the filesystem. The one exception is test_normalize_is_idempotent, which imports
the module and calls a pure function directly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_doc_line_citations.py"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial")
    return repo


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def _pin(repo: Path) -> subprocess.CompletedProcess:
    return _run(repo, "--update")


# A target whose cited line sits far enough down that an upstream insertion
# shifts it, mirroring the real defect.
TARGET = "\n".join(
    [
        "import os",
        "",
        "def train_models(df):",
        "    prepared = df.copy()",
        "    gs_mask = prepared['peptide'].isin(GOLD_STANDARD_EPITOPES)",
        "    return prepared[~gs_mask]",
        "",
    ]
)

DOC = "The exclusion is applied at `src/train_classifier.py:5`.\n"


def test_clean_repo_passes(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path, {"src/train_classifier.py": TARGET, "docs/policy.md": DOC}
    )
    assert _pin(repo).returncode == 0
    result = _run(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All pinned line citations still hold" in result.stdout


def test_catches_the_real_historical_rot(tmp_path: Path) -> None:
    """The defect this gate exists for: an upstream insertion shifts a citation.

    42de845 added a net three lines above two cited sites in
    src/train_classifier.py. Both citations still named a real file and a line
    that existed, so nothing in CI complained. This reproduces that exactly.
    """
    repo = _make_repo(
        tmp_path, {"src/train_classifier.py": TARGET, "docs/policy.md": DOC}
    )
    assert _pin(repo).returncode == 0
    assert _run(repo).returncode == 0

    # Insert three lines ABOVE the cited line, as 42de845 did.
    target = repo / "src/train_classifier.py"
    lines = target.read_text(encoding="utf-8").splitlines()
    shifted = lines[:3] + ["    # added upstream"] * 3 + lines[3:]
    target.write_text("\n".join(shifted) + "\n", encoding="utf-8")

    result = _run(repo)
    assert result.returncode == 1, "gate did not fire on a real line shift"
    assert "DRIFTED" in result.stdout
    # The citation still points at a line that EXISTS - which is precisely why
    # an exists-plus-EOF checker finds nothing here.
    assert len(shifted) >= 5


def test_exists_plus_eof_would_not_catch_it(tmp_path: Path) -> None:
    """Guards the PREMISE of the design, not the gate itself.

    If a shifted citation ever pointed past end-of-file, a far simpler checker
    would do. It does not: the line remains in range, holding wrong content.

    It calls _pin, but asserts only on the fixture and never on the gate's
    output or exit status, so it passes even with the script removed (verified).
    That is intentional and it is why it is named for the premise -
    test_catches_the_real_historical_rot is the one that holds the gate to
    account.
    """
    repo = _make_repo(
        tmp_path, {"src/train_classifier.py": TARGET, "docs/policy.md": DOC}
    )
    _pin(repo)
    target = repo / "src/train_classifier.py"
    lines = target.read_text(encoding="utf-8").splitlines()
    shifted = lines[:3] + ["    # added upstream"] * 3 + lines[3:]
    target.write_text("\n".join(shifted) + "\n", encoding="utf-8")

    assert len(shifted) > 5  # cited line 5 is still well within the file
    assert "gs_mask" not in shifted[4]  # but no longer holds the cited content


def test_unpinned_citation_fails(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path, {"src/train_classifier.py": TARGET, "docs/policy.md": DOC}
    )
    result = _run(repo)  # never pinned
    assert result.returncode == 1
    assert "UNPINNED" in result.stdout


def test_out_of_range_citation_fails(tmp_path: Path) -> None:
    """Must reach the OUT-OF-RANGE branch, not be absorbed by UNPINNED.

    An earlier version cited a line that never existed, so --update refused to
    pin it and the plain run reported UNPINNED. A mutant that deleted the
    OUT-OF-RANGE branch entirely survived that test. The citation is therefore
    pinned while valid, and the target is truncated afterwards.
    """
    repo = _make_repo(
        tmp_path, {"src/train_classifier.py": TARGET, "docs/policy.md": DOC}
    )
    assert _pin(repo).returncode == 0
    target = repo / "src/train_classifier.py"
    target.write_text("only one line\n", encoding="utf-8")  # cited line 5 is gone

    result = _run(repo)
    assert result.returncode == 1
    assert "OUT-OF-RANGE" in result.stdout, result.stdout
    assert "UNPINNED" not in result.stdout


def test_missing_target_fails(tmp_path: Path) -> None:
    """Must reach the MISSING branch. Same absorption bug as above."""
    repo = _make_repo(
        tmp_path, {"src/train_classifier.py": TARGET, "docs/policy.md": DOC}
    )
    assert _pin(repo).returncode == 0
    _git(repo, "rm", "-q", "src/train_classifier.py")

    result = _run(repo)
    assert result.returncode == 1
    assert "MISSING" in result.stdout, result.stdout
    assert "UNPINNED" not in result.stdout


def test_stale_pin_is_reported(tmp_path: Path) -> None:
    """A baseline entry whose citation is gone must not linger unnoticed."""
    repo = _make_repo(
        tmp_path, {"src/train_classifier.py": TARGET, "docs/policy.md": DOC}
    )
    _pin(repo)
    (repo / "docs/policy.md").write_text("No citation here.\n", encoding="utf-8")
    result = _run(repo)
    assert result.returncode == 1
    assert "STALE-PIN" in result.stdout


def test_changelog_is_exempt(tmp_path: Path) -> None:
    """Historical ledgers quote dead line numbers on purpose."""
    repo = _make_repo(
        tmp_path,
        {
            "src/train_classifier.py": TARGET,
            "CHANGELOG.md": "Once cited `src/train_classifier.py:999`.\n",
        },
    )
    _pin(repo)
    result = _run(repo)
    assert result.returncode == 0, result.stdout


def test_exemption_is_scoped_to_the_ledger(tmp_path: Path) -> None:
    """The exemption must not rescue the same citation in a live doc.

    This mirrors the proof the integrity harness requires of its own citation
    exemptions: an exemption that leaks beyond its file is gate relaxation.
    """
    repo = _make_repo(
        tmp_path,
        {
            "src/train_classifier.py": TARGET,
            "CHANGELOG.md": "Once cited `src/train_classifier.py:999`.\n",
            "docs/policy.md": "Live cite `src/train_classifier.py:999`.\n",
        },
    )
    _pin(repo)
    result = _run(repo)
    assert result.returncode == 1, "live doc must still fail"
    assert "docs/policy.md" in result.stdout

    # The ledger must not be reported as a FAILURE. It may appear in the
    # advisory block added 2026-08-14, which lists exempt-ledger citations that
    # no longer resolve so the blind spot is visible rather than merely counted.
    # Asserting the bare string was absent conflated "not failed" with "not
    # mentioned", and would have blocked surfacing the blind spot at all.
    error_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("ERROR") or line.startswith("::error::")
    ]
    assert not any("CHANGELOG.md" in line for line in error_lines), (
        f"exemption leaked into a failure: {error_lines}"
    )


def test_ratchet_fails_when_a_ledger_gains_a_citation(tmp_path: Path) -> None:
    """The exempt-ledger blind spot may shrink, never grow.

    Drift inside the ledgers is unverifiable without a live/historical judgement
    no line-scoped gate can make, so the count is what gets enforced instead.
    """
    repo = _make_repo(
        tmp_path,
        {
            "src/train_classifier.py": TARGET,
            "CHANGELOG.md": "Once cited `src/train_classifier.py:1`.\n",
        },
    )
    _pin(repo)
    assert _run(repo).returncode == 0, "baseline must be clean before the probe"

    ledger = repo / "CHANGELOG.md"
    ledger.write_text(
        ledger.read_text(encoding="utf-8")
        + "And also `src/train_classifier.py:2`.\n",
        encoding="utf-8",
    )
    result = _run(repo)
    assert result.returncode == 1, "a new ledger citation must fail the ratchet"
    assert "RATCHET" in result.stdout


def test_ratchet_allows_the_blind_spot_to_shrink(tmp_path: Path) -> None:
    """Removing a ledger citation is always allowed and needs no re-baseline."""
    repo = _make_repo(
        tmp_path,
        {
            "src/train_classifier.py": TARGET,
            "CHANGELOG.md": (
                "Cited `src/train_classifier.py:1`.\n"
                "And `src/train_classifier.py:2`.\n"
            ),
        },
    )
    _pin(repo)
    ledger = repo / "CHANGELOG.md"
    ledger.write_text("Cited `src/train_classifier.py:1`.\n", encoding="utf-8")
    result = _run(repo)
    assert result.returncode == 0, result.stdout
    assert "RATCHET" not in result.stdout


def test_fenced_code_block_is_not_a_citation(tmp_path: Path) -> None:
    doc = "```\nsrc/train_classifier.py:999\n```\n"
    repo = _make_repo(
        tmp_path, {"src/train_classifier.py": TARGET, "docs/policy.md": doc}
    )
    _pin(repo)
    result = _run(repo)
    assert result.returncode == 0, result.stdout


def test_suppress_marker_opts_a_line_out(tmp_path: Path) -> None:
    doc = "Illustrative `src/train_classifier.py:999` line-cite:ignore\n"
    repo = _make_repo(
        tmp_path, {"src/train_classifier.py": TARGET, "docs/policy.md": doc}
    )
    _pin(repo)
    result = _run(repo)
    assert result.returncode == 0, result.stdout


def test_self_annotation_redirects_to_the_current_line(tmp_path: Path) -> None:
    """'now line N' pins N, not the dated forensic number.

    docs/security_compliance.md preserves scan-time line numbers as evidence
    while naming where the code lives today. The dated number must survive
    untouched; the current one is what a reader follows, so it is what is
    checked.
    """
    doc = "`src/train_classifier.py:2` (2026-06-18 scan; now line 5)\n"
    repo = _make_repo(
        tmp_path, {"src/train_classifier.py": TARGET, "docs/policy.md": doc}
    )
    assert _pin(repo).returncode == 0
    baseline = json.loads(
        (repo / "docs/line_citations.json").read_text(encoding="utf-8")
    )
    entry = baseline["citations"][0]
    assert entry["lines"] == "5", "should pin the current line, not the dated one"
    assert "gs_mask" in entry["pinned"][0]

    # And it must still fire when THAT line moves.
    target = repo / "src/train_classifier.py"
    lines = target.read_text(encoding="utf-8").splitlines()
    target.write_text(
        "\n".join(lines[:3] + ["    # added"] * 3 + lines[3:]) + "\n",
        encoding="utf-8",
    )
    assert _run(repo).returncode == 1


def test_sibling_relative_path_resolves(tmp_path: Path) -> None:
    """docs/a.md citing 'b.md:1' means docs/b.md, as a reader would read it."""
    repo = _make_repo(
        tmp_path,
        {
            "docs/b.md": "first line\nsecond line\n",
            "docs/a.md": "See `b.md:1`.\n",
        },
    )
    assert _pin(repo).returncode == 0
    result = _run(repo)
    assert result.returncode == 0, result.stdout


def test_dotfile_citation_is_seen(tmp_path: Path) -> None:
    """.gitignore:NNN citations were invisible to an earlier pattern."""
    repo = _make_repo(
        tmp_path,
        {
            ".gitignore": "*.pyc\n!results/keep.md\n",
            "docs/policy.md": "Un-ignored at `.gitignore:2`.\n",
        },
    )
    assert _pin(repo).returncode == 0
    baseline = json.loads(
        (repo / "docs/line_citations.json").read_text(encoding="utf-8")
    )
    assert any(e["target"] == ".gitignore" for e in baseline["citations"])


@pytest.mark.parametrize(
    "text",
    [
        "x" * 300,
        "a b   c    d " * 40,
        "   leading and trailing   ",
    ],
)
def test_normalize_is_idempotent(text: str) -> None:
    """Regression: a non-idempotent normalize reported long lines as drifted.

    Pins are normalized when written and again when read back. Truncation that
    landed on a space left a trailing space the second pass stripped, so a
    pinned line failed against its own pin. The trigger needs BOTH conditions -
    longer than the pin width AND the cut landing on whitespace - so only the
    middle parameter below is load-bearing; the other two would pass even with
    the .strip() removed and are kept as boundary cases.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import check_doc_line_citations as mod
    finally:
        sys.path.pop(0)
    once = mod.normalize(text)
    assert mod.normalize(once) == once


def test_reindentation_alone_is_not_drift(tmp_path: Path) -> None:
    """Whitespace-only change must not fire; it is not a moved citation."""
    repo = _make_repo(
        tmp_path, {"src/train_classifier.py": TARGET, "docs/policy.md": DOC}
    )
    _pin(repo)
    target = repo / "src/train_classifier.py"
    lines = target.read_text(encoding="utf-8").splitlines()
    lines[4] = "        " + lines[4].strip()  # deeper indent, same content
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = _run(repo)
    assert result.returncode == 0, result.stdout
