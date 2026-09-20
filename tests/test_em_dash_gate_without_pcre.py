"""The em-dash gates must still block when grep has no PCRE support.

Why this test exists
--------------------
`scripts/hooks/commit-msg` gate 3 and `scripts/hooks/pre-commit` gate 3 both detected
U+2014 with `grep -qP "\\xe2\\x80\\x94"` and discarded stderr.

`-P` is a GNU extension. BSD grep, which is `/usr/bin/grep` on macOS, does not support
it and exits **2**. Because stderr was suppressed, the shell's `if` read that 2 as
"no match found" and the hook fell through to `exit 0`, ACCEPTING the commit. The gate
did not mis-report the result; it stopped gating entirely.

Measured 2026-09-20 against the hook at 21bbcac, under a grep that exits 2 on `-P`:
an em-dash commit message was ACCEPTED (exit 0). The fixed hook blocks it (exit 1).

This is the identical failure mode `scripts/hooks/pre-commit` already documents for its
credential gate -- "'--' is REQUIRED: the PEM pattern begins with '-----', which grep
would otherwise parse as an option bundle, exit 2, and (with stderr suppressed) silently
never match - letting private keys through this gate entirely." That one was fixed. The
em-dash gate three lines away was not, because `.claude/rules/encoding.md` rule 4
explicitly blessed the `-qP` form as correct.

The fix builds the pattern with printf OCTAL escapes and matches it as a literal byte
string, which requires no PCRE and is POSIX-guaranteed.

Anti-vacuity
------------
`test_shim_grep_really_refuses_pcre` is the load-bearing anchor. If the shim did not
actually suppress `-P` -- for instance because the test put a Windows-style path on
PATH, which Git Bash does not search -- then every other test here would run against
the REAL grep and would pass against the BROKEN hook. That exact mistake was made once
while developing this test and silently reported the unfixed hook as safe.

The companions guard the other directions: a clean message must still be accepted
(proving the fix is not the degenerate "reject every commit"), and the gate must still
block under a normal PCRE-capable grep (proving the fix did not trade one hole for
another).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMIT_MSG_HOOK = REPO_ROOT / "scripts" / "hooks" / "commit-msg"

# Never write the banned glyph literally: a detector that embeds the character it hunts
# becomes a source of it (.claude/rules/encoding.md rule 4).
#
# Built with chr() rather than written as "\\u2014", deliberately. The escape keeps the
# SOURCE ASCII, but tests/test_encoding_ascii_output.py walks ast.Constant nodes and
# judges the literal's VALUE, which is the banned character either way - so the escape
# form fails that gate and would have to be bought with a NON_ASCII_LITERAL_ALLOWLIST
# entry. chr() is not a string constant at all, so it needs no allowlist entry, and an
# enforced config list that names paths is one this repo would rather not grow.
EM_DASH = chr(0x2014)

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("grep") is None,
    reason="needs a POSIX bash and grep to exercise the shell hooks",
)


@pytest.fixture()
def shim_dir(tmp_path: Path) -> Path:
    """A directory holding a `grep` that refuses -P, exactly as BSD grep does."""
    real_grep = shutil.which("grep")
    assert real_grep is not None
    shim = tmp_path / "shimbin"
    shim.mkdir()
    # Delegate to the real grep for everything except a -P bundle, which exits 2 with a
    # message on stderr - the BSD behaviour that the old hook silently swallowed.
    (shim / "grep").write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do\n'
        "  case \"$a\" in -*P*) echo 'grep: invalid option -- P' >&2; exit 2;; esac\n"
        "done\n"
        f'exec "{real_grep}" "$@"\n',
        encoding="ascii",
        newline="\n",
    )
    (shim / "grep").chmod(0o755)
    return shim


def _posix(path: Path) -> str:
    """PATH entries must be POSIX-shaped; Git Bash does not search `C:/...` entries."""
    text = path.as_posix()
    if len(text) > 1 and text[1] == ":":
        text = "/" + text[0].lower() + text[2:]
    return text


def _run_commit_msg(message: str, tmp_path: Path, shim_dir: Path | None) -> int:
    msg_file = tmp_path / "COMMIT_EDITMSG"
    msg_file.write_text(message, encoding="utf-8", newline="\n")
    env = dict(os.environ)
    if shim_dir is not None:
        env["PATH"] = _posix(shim_dir) + os.pathsep + env.get("PATH", "")
    return subprocess.run(
        ["bash", str(COMMIT_MSG_HOOK), str(msg_file)],
        capture_output=True,
        env=env,
    ).returncode


def test_shim_grep_really_refuses_pcre(shim_dir: Path) -> None:
    """Anti-vacuity anchor: prove the shim is on PATH and actually suppresses -P.

    Without this, a mis-shaped PATH silently falls back to the real grep and every
    other test in this module would pass against the unfixed, fail-open hook.
    """
    env = dict(os.environ)
    env["PATH"] = _posix(shim_dir) + os.pathsep + env.get("PATH", "")

    resolved = subprocess.run(
        ["bash", "-c", "command -v grep"],
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()
    assert resolved.endswith("shimbin/grep"), f"shim not on PATH, grep resolved to {resolved}"

    probe = subprocess.run(
        ["bash", "-c", "grep -qP x /dev/null"],
        capture_output=True,
        env=env,
    )
    assert probe.returncode == 2, "shim grep must exit 2 on -P, as BSD grep does"


def test_em_dash_is_blocked_when_grep_has_no_pcre(tmp_path: Path, shim_dir: Path) -> None:
    """The regression. The pre-fix hook returned 0 here, accepting the commit."""
    rc = _run_commit_msg(f"fix: a message with an {EM_DASH} em-dash\n", tmp_path, shim_dir)
    assert rc == 1, "em-dash must be rejected even when grep lacks PCRE support"


def test_clean_message_is_accepted_when_grep_has_no_pcre(
    tmp_path: Path, shim_dir: Path
) -> None:
    """Not the degenerate fix: a clean message must still pass under the same grep."""
    rc = _run_commit_msg("fix: a clean message - spaced hyphen only\n", tmp_path, shim_dir)
    assert rc == 0, "a hyphen-only message must be accepted"


def test_em_dash_is_blocked_with_a_normal_grep(tmp_path: Path) -> None:
    """The fix must not trade the PCRE hole for a regression on a GNU grep."""
    rc = _run_commit_msg(f"fix: a message with an {EM_DASH} em-dash\n", tmp_path, None)
    assert rc == 1, "em-dash must still be rejected under a PCRE-capable grep"


def test_clean_message_is_accepted_with_a_normal_grep(tmp_path: Path) -> None:
    rc = _run_commit_msg("fix: a clean message - spaced hyphen only\n", tmp_path, None)
    assert rc == 0, "a hyphen-only message must be accepted"


def test_hooks_do_not_execute_a_pcre_grep() -> None:
    """No executed line in either hook may depend on `grep -P`.

    Mentions inside comments are allowed and expected - both hooks explain the defect
    they are guarding against - so comment lines are stripped before the check.
    """
    for hook in ("commit-msg", "pre-commit"):
        text = (REPO_ROOT / "scripts" / "hooks" / hook).read_text(encoding="utf-8")
        code = [
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        ]
        offenders = [line for line in code if "-qP" in line or "-oP" in line]
        assert not offenders, f"{hook} still runs a PCRE grep: {offenders}"
