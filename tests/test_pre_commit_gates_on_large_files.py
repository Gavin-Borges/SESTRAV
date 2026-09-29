"""The pre-commit gates must still fire when the staged file is large.

Why this test exists
--------------------
`scripts/hooks/pre-commit` sets `set -euo pipefail` and its gates tested their
patterns with `printf '%s' "$content" | grep -q ...`. `grep -q` exits 0 at the FIRST
match and closes the pipe; `printf` is then killed by SIGPIPE and exits 141; and
`pipefail` makes the PIPELINE report 141. A non-zero status sends the `if` down the
FALSE branch, so a SUCCESSFUL match was read as "no match" and the hook fell through
to `exit 0`, accepting the commit.

The failure is size-dependent, which is why it survived: the producer must still have
bytes to write when `grep -q` exits, i.e. roughly one pipe buffer (~64 KiB) of content
after the match point. Every existing hook fixture is small - the rename suite's
`BASELINE_BODY` is about 1.4 KB - so the suite was structurally unable to see it.

Measured 2026-09-20 against the real hooks, pre-fix vs fixed:

    AWS key id in a 180 KB staged file      exit 0 ACCEPTED  ->  exit 1 blocked
    PEM header in a 180 KB staged file      exit 0 ACCEPTED  ->  exit 1 blocked
    60-byte em-dash doc + 267 KB clean file exit 0 ACCEPTED  ->  exit 1 blocked
    same AWS key id in a 6 KB file          exit 1 blocked   ->  exit 1 blocked

The em-dash case matters most in practice: gate 3's outer detector reads the WHOLE
staged diff, so the file carrying the banned character does not need to be large. One
regenerated lockfile staged in the same commit is enough.

Two earlier fail-opens in this same hook reached gate 3 by different routes: the
`--diff-filter=ACM` rename hole, which dropped renamed files from the list every gate
iterates, and the `grep -P` exit-2 on a non-GNU grep. No ordinal is claimed for how
many there have been in total; what matters is that all three share one shape, a
non-zero or absent result being read as "clean".

Anti-vacuity
------------
`test_the_vulnerable_idiom_still_fails_open_on_this_payload` is the load-bearing
anchor, and it is the pattern session 101 established for the PCRE shim: assert that
the condition the test depends on actually holds before believing any result. It runs
the OLD idiom against this module's own payload and requires it to fail open. If a
larger pipe buffer, a different shell, or a shrunken fixture ever made the payload too
small, every other test here would pass against a reintroduced bug; that anchor fails
loudly instead.

`test_small_credential_is_still_blocked` guards the other direction: it proves the
fixture can trigger the gate at all, so a blocked result is about size and not about a
pattern that never matches.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_SOURCE = REPO_ROOT / "scripts" / "hooks" / "pre-commit"

# Built with chr() rather than as a literal, for the reason
# tests/test_pre_commit_hook_sees_renames.py records: a bare U+2014 would be caught by
# the gate under test, and an escape does not help because the ASCII allowlist gate
# inspects the PARSED value of ast.Constant nodes.
EM_DASH = chr(0x2014)

# Synthetic fixtures matching gate 2's AWS access-key-id and PEM patterns. Neither is
# a real credential; the AWS value is AWS's own published documentation example.
#
# CONCATENATED AT RUNTIME, never written as a contiguous literal, for the same reason
# the rename suite builds its em-dash with chr(): gate 2 scans the staged content of
# EVERY file, including this one, so a literal key id or PEM header here would block
# the very commit that adds this test. Splitting breaks the match in the SOURCE while
# leaving the runtime value intact, because gate 2's AWS rule requires 16 [0-9A-Z]
# characters immediately after the prefix and its PEM rule requires [A-Z ]* between
# "BEGIN" and "PRIVATE KEY"; the quote and plus interrupt both.
AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
PEM_HEADER = "-----BEGIN " + "RSA PRIVATE KEY-----"

# Comfortably more than one pipe buffer (64 KiB on Linux and on this workstation's
# Git Bash) AFTER the match, which is what the SIGPIPE race needs. Deliberately not
# tuned close to the threshold: the buffer size is a kernel detail, it is SMALLER on
# macOS, and a fixture sized just over the edge would decay into a vacuous pass on
# some other machine without anyone noticing.
PADDING_BYTES = 200_000
PADDING = "y" * PADDING_BYTES

# git reads the developer's GLOBAL and SYSTEM config even inside a throwaway repo, so
# without this the result depends on the machine rather than on the hook. Same
# rationale, and the same measured failure modes, as the rename suite.
GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="pre-commit is a bash script; without bash it cannot run at all",
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True, env=GIT_ENV
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A throwaway repo with the hook under test copied in and one baseline commit."""
    repo = tmp_path / "sandbox"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    # Invoked directly rather than through core.hooksPath so the exit code and stderr
    # are observable instead of being wrapped by `git commit`.
    (repo / "hook").write_text(
        HOOK_SOURCE.read_text(encoding="utf-8"), encoding="utf-8", newline=""
    )
    (repo / "baseline.md").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "baseline.md")
    _git(repo, "commit", "--quiet", "-m", "baseline")
    return repo


def _run_hook(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "hook"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env=GIT_ENV,
    )


def _stage(repo: Path, name: str, body: str) -> None:
    (repo / name).write_text(body, encoding="utf-8")
    _git(repo, "add", name)


def test_the_vulnerable_idiom_still_fails_open_on_this_payload(tmp_path: Path) -> None:
    """Anti-vacuity anchor: the payload must be big enough to trigger SIGPIPE.

    Runs the PRE-FIX idiom, not the hook. If this stops failing open, the fixtures
    below have gone vacuous and would pass against a reintroduced bug.

    The payload reaches bash through a FILE, not through argv. Passing 200 KB as a
    command-line argument raises WinError 206 ("filename or extension is too long")
    on Windows, and reading it into a shell variable with $(cat ...) is also what the
    hook itself does with $(git show ...), so this matches the real shape.
    """
    payload = tmp_path / "payload.txt"
    payload.write_text(f"{AWS_KEY}\n{PADDING}", encoding="utf-8")
    script = (
        "set -euo pipefail\n"
        'c=$(cat "$1")\n'
        'if printf "%s" "$c" | grep -qE -- "AKIA[0-9A-Z]{16}"; then\n'
        "  echo MATCHED\n"
        "else\n"
        "  echo MISSED\n"
        "fi\n"
    )
    result = subprocess.run(
        ["bash", "-c", script, "bash", str(payload)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.strip() == "MISSED", (
        "The old 'printf | grep -q' idiom did NOT fail open on this payload, so the "
        "large-file tests below are no longer testing the bug they were written for. "
        f"Increase PADDING_BYTES (currently {PADDING_BYTES}). stdout={result.stdout!r}"
    )


def test_small_credential_is_still_blocked(repo: Path) -> None:
    """Direction guard: the fixture can trigger gate 2 at all, independent of size."""
    _stage(repo, "small_secret.txt", f"{AWS_KEY}\n" + "y" * 500)
    result = _run_hook(repo)
    assert result.returncode != 0, (
        "gate 2 did not block a credential in a SMALL file, so the pattern itself is "
        f"not matching and the large-file tests prove nothing. stderr={result.stderr}"
    )


def test_large_file_credential_is_blocked(repo: Path) -> None:
    """Gate 2: a credential must not slip through because the file is large."""
    _stage(repo, "big_secret.txt", f"{AWS_KEY}\n{PADDING}")
    result = _run_hook(repo)
    assert result.returncode != 0, (
        "gate 2 accepted a commit staging an AWS access key id in a "
        f"{PADDING_BYTES}-byte file. stderr={result.stderr}"
    )


def test_large_file_private_key_is_blocked(repo: Path) -> None:
    """Gate 2, PEM arm: this is the pattern with no other gate behind it.

    scripts/check_secrets.py is assignment-shaped plus entropy and does not match a
    bare PEM header. Measured 2026-09-20 by calling scan_file directly: a fixture
    holding only the PEM_HEADER value above returns [], while an
    assignment-shaped control returns [1]. PR #506 wired that scanner into pre-push,
    and the CI secret-pattern job is not one of the required contexts, so for a PEM
    block specifically gate 2 remains the only detector that blocks anywhere.
    """
    _stage(repo, "big_key.txt", f"{PEM_HEADER}\n{PADDING}")
    result = _run_hook(repo)
    assert result.returncode != 0, (
        "gate 2 accepted a commit staging a PEM private key header in a "
        f"{PADDING_BYTES}-byte file. stderr={result.stderr}"
    )


def test_small_em_dash_file_is_blocked_when_staged_with_a_large_clean_file(
    repo: Path,
) -> None:
    """Gate 3 reads the WHOLE staged diff, so the offending file need not be large."""
    _stage(repo, "a_doc.md", f"title {EM_DASH} subtitle\n")
    _stage(repo, "z_big_clean.txt", PADDING)
    result = _run_hook(repo)
    assert result.returncode != 0, (
        "gate 3 accepted a commit whose small doc carries U+2014 because a large "
        f"clean file was staged alongside it. stderr={result.stderr}"
    )


def test_large_clean_file_is_still_allowed(repo: Path) -> None:
    """False-positive guard: the fix must not reject every large file."""
    _stage(repo, "big_clean.txt", PADDING)
    result = _run_hook(repo)
    assert result.returncode == 0, (
        "the hook rejected a large file containing nothing it gates on. "
        f"stdout={result.stdout} stderr={result.stderr}"
    )


def test_clean_file_with_no_linux_home_path_is_allowed(repo: Path) -> None:
    """Regression guard for gate 4's two-step rewrite.

    The home-path check now captures matches and then filters them. A here-string fed
    an EMPTY variable still presents one empty line, which does not match the
    allowlist, so without the `-n` guard `grep -qv` would succeed and flag every file.
    """
    _stage(repo, "ordinary.txt", "nothing interesting here\n")
    result = _run_hook(repo)
    assert result.returncode == 0, (
        "gate 4 flagged a file containing no /home/ path at all, which is the empty "
        f"here-string false positive. stdout={result.stdout} stderr={result.stderr}"
    )
