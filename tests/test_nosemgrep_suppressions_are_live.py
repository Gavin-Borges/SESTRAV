"""Every semgrep suppression in the tree is one measured to hide a live finding.

A bare `# nosemgrep` silences EVERY semgrep rule on its line, including rules
added to the registry after it was written, so one that suppresses nothing does
no work today and hides findings tomorrow. Measured 2026-10-08 with the pinned
semgrep 1.178.0 (environments/requirements-semgrep.txt, Python 3.11, Linux) and
CI's own configs, `--config p/python --config semgrep-rules/`, over every file
that carried one: once as shipped and once with `--disable-nosem`, beside a
planted control that both configs flagged.

- Three suppress a live finding,
  `python.lang.security.audit.dangerous-subprocess-use-tainted-env-args`, on the
  `subprocess.run(cmd, check=True)` call in each wrapper script listed in
  LIVE_SUPPRESSIONS. They stay bare on purpose: scripts/run_predig_wrapper.py
  records why the rule-qualified form never matched.
- Five suppressed nothing (in src/ann_benchmark.py, src/baseline_comparison.py,
  src/model.py, functions/stage4_immunogenicity_scoring.py and
  scripts/install_prime_wsl.py) and were removed.

Re-measured 2026-10-09 when the pin moved to semgrep 1.179.0, with the same two
configs (Python 3.11, Windows): the same three findings, on the same three calls,
appear only under `--disable-nosem`, and a planted control for each config is
flagged by it.

This test pins that population; it cannot run semgrep, so "live" means the
measurements above. Before adding a suppression, measure it the same way, with
and without `--disable-nosem`, and add the file here with the rule it silences.

Lines are matched the way semgrep 1.178.0 matches them, measured against the
semgrep-core that runs, not read from pysemgrep's semgrep/constants.py, which
differs for the own-line form (re-checked on 1.179.0: eight probe lines and an
unsuppressed control all agree; no probe covered a directive after non-ASCII prose
or a finding spanning several lines). It searches raw source lines,
case-insensitively, for `nosem`, so `nosemgrep`, `NOSEMGREP` and `nosemantics` all
qualify:
- on the line where a finding starts, after a space, anywhere on the line, a
  string literal included, it silences that finding; `#nosemgrep` with no space
  does not;
- with no ASCII letter or digit before it, space or none, it silences the line
  BELOW, so an own-line `#nosemgrep` above a call does silence the call.
A comment-only line whose directive follows ASCII prose silences nothing, which
is why the wrappers' explanatory comments do not count. The detector counts a
directive on any code line, a slight over-count in the safe direction.
"""

from __future__ import annotations

import io
import re
import subprocess
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# This file spells the directive in its docstring and cases, so it is the one exclusion.
SELF = "tests/test_nosemgrep_suppressions_are_live.py"

# file -> the rule its one suppression silences (measured 2026-10-08, see above).
LIVE_SUPPRESSIONS = {
    "scripts/run_predig_batched.py": "dangerous-subprocess-use-tainted-env-args",
    "scripts/run_predig_wrapper.py": "dangerous-subprocess-use-tainted-env-args",
    "scripts/run_prime_wrapper.py": "dangerous-subprocess-use-tainted-env-args",
}

# A rule-id suffix narrows a directive to the rules it names; it is counted anyway, the safe way.
# Only the word is case-insensitive: a global re.IGNORECASE would also fold U+0130 and U+0131
# into [a-zA-Z], and semgrep treats both as non-letters.
INLINE = re.compile(r" (?i:nosem)")
PREVIOUS_LINE = re.compile(r"^[^a-zA-Z0-9]*(?i:nosem)")

_NOT_CODE = {
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
}


def _tracked_python_files() -> list[str]:
    if not (ROOT / ".git").exists():
        pytest.skip("not a git work tree, so the tracked file set cannot be read")
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "--", "*.py"],
        capture_output=True,
        check=True,
    ).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def _code_lines(source: str) -> set[int]:
    """Line numbers that hold code, every line of a multi-line string included."""
    lines: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type not in _NOT_CODE:
            lines.update(range(tok.start[0], tok.end[0] + 1))
    return lines


def _directive_lines(source: str) -> list[int]:
    """Lines on which semgrep would honour a suppression."""
    code = _code_lines(source)
    # split("\n"), not splitlines(): splitlines() also breaks on form feeds and Unicode line
    # separators, which tokenize and semgrep do not, so every later line number would drift.
    return [
        number
        for number, line in enumerate(source.split("\n"), start=1)
        if PREVIOUS_LINE.search(line) or (number in code and INLINE.search(line))
    ]


def test_every_nosemgrep_is_a_measured_live_suppression() -> None:
    files = _tracked_python_files()
    assert files, "git ls-files returned no Python files, so this test would pass vacuously"
    found = {}
    for path in files:
        if path == SELF:
            continue
        # Raw bytes: read_text() would turn a lone CR into a line break, which semgrep does not.
        lines = _directive_lines((ROOT / path).read_bytes().decode("utf-8"))
        if lines:
            found[path] = lines
    unmeasured = {path: lines for path, lines in found.items() if path not in LIVE_SUPPRESSIONS}
    assert unmeasured == {}, (
        "semgrep would honour a suppression on these lines, and they are not on the measured "
        "list; run semgrep with and without --disable-nosem and either delete the directive (it "
        "suppresses nothing) or add the file to LIVE_SUPPRESSIONS with the rule it silences: "
        f"{unmeasured}"
    )
    for path in LIVE_SUPPRESSIONS:
        assert len(found.get(path, [])) == 1, (
            f"{path} should carry exactly one suppression, found lines {found.get(path, [])}; "
            "update LIVE_SUPPRESSIONS if the suppression was moved or retired"
        )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("run(cmd)  # nosemgrep\n", [1]),
        ("run(cmd)  # nosec B603  # nosemgrep\n", [1]),
        ("load(p)  # nosec B614 nosemgrep\n", [1]),
        ("run(cmd)  # nosemgrep: some.rule.id\n", [1]),
        ("run(cmd)  # nosem\n", [1]),
        ("run(cmd)  # NOSEMGREP\n", [1]),
        ("run(cmd)  # NoSemGrep: some.rule.id\n", [1]),
        ("run(cmd)  # nosemantics\n", [1]),
        ("text = ' nosemgrep'\n", [1]),
        ('x = """\n nosem\n"""\n', [2]),
        ("if x:\n    # nosem\n    run(cmd)\n", [2]),
        ("# nosemgrep is deliberate below\nrun(cmd)\n", [1]),
        ("run(cmd)  #nosemgrep\n", []),
        ("# The bare inline `# nosemgrep` below is deliberate.\n", []),
        ("run(cmd)  # see nosemgrep_docs\n", [1]),
        ("#nosemgrep\nrun(cmd)\n", [1]),
        ("if x:\n    #\tNOSEM\n    run(cmd)\n", [2]),
        ("x = 1\n\x0c\nrun(cmd)  # nosem\n", [3]),
        ("# " + chr(0x0130) + " nosem\nrun(cmd)\n", [1]),
        ("run(cmd)\r# see nosem\n", [1]),
    ],
    ids=[
        "bare",
        "after-a-nosec",
        "word-without-its-own-hash",
        "qualified",
        "short-form",
        "upper-case",
        "mixed-case-qualified",
        "prefix-only",
        "string-literal-on-a-code-line",
        "inside-a-multi-line-string",
        "own-line-silences-the-next",
        "prose-that-starts-with-the-directive",
        "inline-without-a-space-is-not-honoured",
        "prose-after-words-silences-nothing",
        "mention-on-a-code-line-still-silences-it",
        "own-line-without-a-space-silences-the-next",
        "own-line-after-a-tab-silences-the-next",
        "a-form-feed-does-not-shift-line-numbers",
        "dotted-capital-i-is-not-a-letter-to-semgrep",
        "a-lone-cr-does-not-split-the-line",
    ],
)
def test_the_detector_matches_what_semgrep_honours(source: str, expected: list[int]) -> None:
    assert _directive_lines(source) == expected
