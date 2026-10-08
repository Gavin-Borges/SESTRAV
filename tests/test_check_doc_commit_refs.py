"""Regression tests for the dead-commit-citation gate's SHA detector.

`scripts/check_doc_commit_refs.py` is the CI gate (`.github/workflows/
doc_commit_refs.yml`) that fails a build when a tracked doc cites a commit SHA
that no longer resolves. Its value depends entirely on `SHA_RE` classifying
correctly in BOTH directions, and it had no test coverage at all before this
file.

The motivating defect: "." is not alphanumeric, so the original lookbehind
`(?<![0-9a-zA-Z])` allowed the fractional part of a decimal to match. In
`0.8275628` the token `8275628` is seven characters, every one a valid hex
digit, preceded by "." and followed by a boundary. `docs/claims_register.md`
D16 cites exactly that value - and 0.8277666 alongside it, the pair whose
0.0002 separation is the documented root cause of the Tier A mislabel - so the
gate reported two dead commits on a repository that had none.

The fix must not overcorrect. Rejecting all-decimal tokens outright would be
simpler, but a genuine abbreviated SHA is all digits with probability
(10/16)^7 ~ 3.7%, so that rule would silently stop catching roughly one in
twenty-seven dead citations. `test_all_digit_sha_is_still_detected` pins that
behaviour so the cheap-but-wrong fix cannot be reintroduced.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_doc_commit_refs.py"
_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "doc_commit_refs.yml"


def _load_module():
    """Import the checker by path - `scripts/` is not an installed package."""
    spec = importlib.util.spec_from_file_location("check_doc_commit_refs", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dedicated_gate_runs_on_push_to_main():
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"(?m)^  push:\n    branches:\n      - main$", workflow)


@pytest.fixture(scope="module")
def sha_re():
    return _load_module().SHA_RE


@pytest.fixture(scope="module")
def has_commit_context():
    return _load_module().has_commit_context


# --- The regression: decimals must not be read as commit SHAs ---------------


@pytest.mark.parametrize(
    "text",
    [
        "0.8275628",
        "0.8277666",
        "AUC-PR 0.8275628 against 0.8277666",
        "auc_pr 0.889738352647154 in training_results.csv",
        "0.9493670886075949",
    ],
)
def test_decimal_fraction_is_not_a_sha(sha_re, text):
    assert sha_re.findall(text) == [], (
        f"decimal tail in {text!r} was misread as a commit SHA; this is the "
        "defect that failed the doc_commit_refs gate on PR #229"
    )


# --- The other direction: real citations must still be caught ---------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("commit dd5a356 landed", ["dd5a356"]),
        ("see 8f48866ec20f353ea70a2159fa82ece53afb1a51", ["8f48866ec20f353ea70a2159fa82ece53afb1a51"]),
        ("recoverable at 69e0e5c on main", ["69e0e5c"]),
        # A SHA following a sentence period: the space breaks the `\d\.`
        # sequence, so the decimal guard must not suppress it.
        ("... corrected in the fix. abc1234 reverts it", ["abc1234"]),
    ],
)
def test_real_sha_is_detected(sha_re, text, expected):
    assert sha_re.findall(text) == expected


def test_all_digit_sha_is_still_detected(sha_re):
    """Guard against the overcorrection of rejecting all-decimal tokens.

    ~3.7% of genuine 7-character SHAs are all digits. Suppressing them wholesale
    would blind this gate to about one dead citation in twenty-seven.
    """
    assert sha_re.findall("commit 1234567 is cited here") == ["1234567"]


@pytest.mark.parametrize(
    "text",
    [
        "version 2.2.1 pinned",  # too short to be a SHA
        "x0.1234567y",  # embedded in an alphanumeric run
        "peptide SIINFEKL scored",  # not hex
    ],
)
def test_non_sha_tokens_are_ignored(sha_re, text):
    assert sha_re.findall(text) == []


# --- A distant context word must not tag an unrelated token -----------------


def test_distant_context_word_does_not_tag_unrelated_token(sha_re, has_commit_context):
    """docs/claims_register.md D30 (2026-08-19): a single-line table row used

    "commit" once, ~700 characters from an unrelated filename fragment
    `20260704`. Whole-line context search made every hex-shaped token on that
    row eligible for DEAD reporting and flagged the date fragment as a dead
    commit citation with no real commit citation error present. Reproduced
    here at a smaller scale: a context word far outside the window must not
    tag a token, while one just inside it must.
    """
    padding = "the quick brown fox jumps " * 30  # ~780 chars, no hex/alnum-adjacency risk
    line = f"see 20260704 {padding}in its own commit, not folded in"
    match = next(iter(sha_re.finditer(line)))
    assert not has_commit_context(line, match.start(), match.end())

    close_line = "commit dd5a356 landed"
    close_match = next(iter(sha_re.finditer(close_line)))
    assert has_commit_context(close_line, close_match.start(), close_match.end())


def _is_shallow_clone() -> bool:
    """True when the working clone lacks full history.

    `git rev-parse --is-shallow-repository` prints "true"/"false" (git >= 2.15).
    Any failure is treated as shallow, so the end-to-end check below skips
    rather than reporting a spurious failure in an environment it cannot judge.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=_SCRIPT.parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return out.stdout.strip() != "false"


@pytest.mark.skipif(
    _is_shallow_clone(),
    reason=(
        "shallow clone: abbreviated SHAs cited in docs cannot resolve without "
        "full history. CI's `test` job checks out at the default depth of 1, "
        "while the dedicated doc_commit_refs workflow sets fetch-depth: 0 - "
        "so that workflow, not this test, is the authoritative gate there."
    ),
)
def test_gate_passes_on_the_live_tree(monkeypatch):
    """End-to-end: the checker must exit clean on the repository as it stands.

    This mirrors the assertion CI's `doc_commit_refs` workflow makes, so a dead
    citation - or a fresh false positive - is caught by the local fast gate
    rather than only after a push.

    Requires full git history and is skipped without it (see the skipif above);
    the unit cases in this module are environment-independent and always run.

    `main()` builds its own argparse parser and reads `sys.argv`, which under
    pytest holds pytest's arguments, so argv is replaced with a bare program
    name for the duration of the call.
    """
    module = _load_module()
    monkeypatch.setattr(sys, "argv", ["check_doc_commit_refs.py"])
    assert module.main() == 0, (
        "scripts/check_doc_commit_refs.py reported unresolvable commit "
        "citations against the current tree"
    )


# --- The gate must READ Python sources, not only prose ----------------------
#
# Until 2026-09-13 SCAN_SUFFIXES omitted ".py", so a false commit citation in a
# docstring was invisible to the REQUIRED "Cited commits resolve" check, and one
# shipped before a hand audit caught it. The two sibling gates over the same
# tracked tree, scripts/check_doc_line_citations.py and
# scripts/check_affiliation_claims.py, both already scanned ".py"; this one was
# the sole holdout, with no rationale recorded anywhere in its docstring.


def test_python_sources_are_scanned():
    """The widening itself. Reverting it silently reopens the hole."""
    module = _load_module()
    assert ".py" in module.SCAN_SUFFIXES
    assert module.should_scan("src/features.py") is True


def test_the_gate_and_its_own_tests_are_exempt():
    """Both files carry example and fixture SHAs that exist to exercise the gate.

    Without the exemption, scanning ".py" makes the gate report its own
    docstring examples and this module's fixtures as dead citations - five
    findings, every one of them the gate flagging itself.
    """
    module = _load_module()
    assert module.should_scan("scripts/check_doc_commit_refs.py") is False
    assert module.should_scan("tests/test_check_doc_commit_refs.py") is False
    # The exemption is by BASENAME, so it must not leak to look-alike paths.
    assert module.should_scan("scripts/check_doc_line_citations.py") is True


def test_hugging_face_revision_pin_is_treated_as_third_party():
    """src/features.py pins the ESM-2 weights by upstream repository revision.

    That is a third-party SHA, but spelled as a keyword argument rather than the
    org/repo@sha form, so it reached the gate once ".py" was scanned. The
    suppression is anchored to the ASSIGNMENT: "revision" is itself a
    COMMIT_CONTEXT_RE trigger, so suppressing the bare word would blind the gate
    to any real citation written "revision abc1234".
    """
    module = _load_module()
    pinned = '_esm_model = EsmModel.from_pretrained(name, revision="8c576d2aba1b27317e9321c8491d72f00d1b110a")'
    assert module.EXTERNAL_CONTEXT_RE.search(pinned)
    # A prose citation using the same word must still be examined.
    assert not module.EXTERNAL_CONTEXT_RE.search("see revision abc1234 for the fix")


# --- End to end: the verdict, the exit code and the count -------------------
#
# Everything above exercises the detector; nothing ran the gate's verdict path
# on a repository that HAS a bad citation. The CX-F2 mutation campaign measured
# the cost at 5e1fbe79: making the findings branch unreachable, returning 2
# instead of 1, starting the citation count at 1, or counting a dead or a
# resolved citation twice each left this whole file green.

_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}


def _git(repo: Path, *argv: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@example.com", "-c", "user.name=T",
         "-c", "commit.gpgsign=false", *argv],
        capture_output=True, text=True, check=True, env=_GIT_ENV,
    ).stdout.strip()


def _repo_citing(where: Path, *, dead: bool, orphaned: bool) -> Path:
    """A repo whose tracked notes.md cites its own base commit, plus the bad kinds asked for.

    The orphan is a real commit object that no ref reaches (`git commit-tree`), so it
    resolves but is not reachable from HEAD. The dead token is assembled at runtime and
    names no object at all.
    """
    where.mkdir()
    _git(where, "init", "-q")
    (where / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(where, "add", "seed.txt")
    _git(where, "commit", "-q", "-m", "seed")
    lines = [f"Fixed in commit {_git(where, 'rev-parse', 'HEAD')[:12]}."]
    if orphaned:
        orphan = _git(where, "commit-tree", "HEAD^{tree}", "-p", "HEAD", "-m", "never referenced")
        lines.append(f"Reverted by commit {orphan[:12]}.")
    if dead:
        lines.append("See commit " + "feed" + "face" + "cafe" + ".")
    (where / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _git(where, "add", "notes.md")
    _git(where, "commit", "-q", "-m", "cite")
    return where


def _run_gate(repo: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in _GIT_ENV.items() if k != "GITHUB_ACTIONS"}
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--reachable-from", "HEAD"],
        cwd=repo, capture_output=True, text=True, check=False, env=env,
    )


def test_a_dead_and_an_orphaned_citation_fail_with_exit_one_and_an_exact_count(tmp_path):
    """Three citations: one resolving, one orphaned, one dead. Exit 1, "Checked 3".

    The exit code is asserted EXACTLY, so returning 2 fails; the count is asserted
    exactly, so starting it at 1 (4) or counting a dead (4) or a resolved (5) citation
    twice fails; and both findings must be printed, so a findings branch that can never
    run (exit 0, nothing reported) fails.
    """
    result = _run_gate(_repo_citing(tmp_path / "r", dead=True, orphaned=True))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Checked 3 commit citation(s)" in result.stdout, result.stdout
    assert "DEAD" in result.stdout and "ORPHANED" in result.stdout, result.stdout
    assert "2 unresolvable commit citation(s)." in result.stdout, result.stdout


def test_a_single_resolving_citation_passes_with_a_count_of_one(tmp_path):
    """The control: one reachable citation, so exit 0 and a count of exactly 1."""
    result = _run_gate(_repo_citing(tmp_path / "r", dead=False, orphaned=False))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Checked 1 commit citation(s)" in result.stdout, result.stdout
