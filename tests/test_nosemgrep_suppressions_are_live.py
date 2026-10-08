"""Every `# nosemgrep` in the tree suppresses a finding semgrep actually raises.

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

This test pins that population. Before adding a suppression, measure it the same
way, with and without `--disable-nosem`, and add the file here with the rule it
silences. Comments are read with `tokenize`, so the word inside a string literal
does not count, and a comment that quotes the directive in backticks is prose.
"""

from __future__ import annotations

import io
import re
import subprocess
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# file -> the rule its one suppression silences (measured 2026-10-08, see above).
LIVE_SUPPRESSIONS = {
    "scripts/run_predig_batched.py": "dangerous-subprocess-use-tainted-env-args",
    "scripts/run_predig_wrapper.py": "dangerous-subprocess-use-tainted-env-args",
    "scripts/run_prime_wrapper.py": "dangerous-subprocess-use-tainted-env-args",
}

# The word anywhere in a comment, as in `# nosec B614 nosemgrep`, once any
# backtick-quoted prose has been dropped from the comment.
DIRECTIVE = re.compile(r"\bnosemgrep\b")
QUOTED = re.compile(r"`[^`]*`")


def _tracked_python_files() -> list[str]:
    if not (ROOT / ".git").exists():
        pytest.skip("not a git work tree, so the tracked file set cannot be read")
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "--", "*.py"],
        capture_output=True,
        check=True,
    ).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def _directive_lines(source: str) -> list[int]:
    return [
        tok.start[0]
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type == tokenize.COMMENT and DIRECTIVE.search(QUOTED.sub("", tok.string))
    ]


def test_every_nosemgrep_is_a_measured_live_suppression() -> None:
    files = _tracked_python_files()
    assert files, "git ls-files returned no Python files, so this test would pass vacuously"
    found = {}
    for path in files:
        lines = _directive_lines((ROOT / path).read_text(encoding="utf-8"))
        if lines:
            found[path] = lines
    unmeasured = {path: lines for path, lines in found.items() if path not in LIVE_SUPPRESSIONS}
    assert unmeasured == {}, (
        "these # nosemgrep comments are not on the measured list; run semgrep with and without "
        "--disable-nosem and either delete the comment (it suppresses nothing) or add the file "
        f"to LIVE_SUPPRESSIONS with the rule it silences: {unmeasured}"
    )
    for path in LIVE_SUPPRESSIONS:
        assert len(found.get(path, [])) == 1, (
            f"{path} should carry exactly one # nosemgrep, found lines {found.get(path, [])}; "
            "update LIVE_SUPPRESSIONS if the suppression was moved or retired"
        )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("run(cmd)  # nosemgrep\n", 1),
        ("run(cmd)  # nosec B603  # nosemgrep\n", 1),
        ("load(p)  # nosec B614 nosemgrep\n", 1),
        ("run(cmd)  # nosemgrep: some.rule.id\n", 1),
        ("# The bare inline `# nosemgrep` below is deliberate.\n", 0),
        ("text = '# nosemgrep'\n", 0),
    ],
    ids=[
        "bare",
        "after-a-nosec",
        "word-without-its-own-hash",
        "qualified",
        "prose-in-backticks",
        "string-literal",
    ],
)
def test_the_detector_reads_directives_not_prose(source: str, expected: int) -> None:
    assert len(_directive_lines(source)) == expected
