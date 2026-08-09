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
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_doc_commit_refs.py"


def _load_module():
    """Import the checker by path - `scripts/` is not an installed package."""
    spec = importlib.util.spec_from_file_location("check_doc_commit_refs", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sha_re():
    return _load_module().SHA_RE


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


def test_gate_passes_on_the_live_tree(monkeypatch):
    """End-to-end: the checker must exit clean on the repository as it stands.

    This is the assertion CI actually makes. Keeping it here means a dead
    citation - or a fresh false positive - is caught by the local fast gate
    rather than only after a push.

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
