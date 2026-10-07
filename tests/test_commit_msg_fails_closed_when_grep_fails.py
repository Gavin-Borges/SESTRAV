"""commit-msg must reject, not accept, a message it could not check.

Each of the hook's four checks is an ``if grep ...``, and an ``if`` reads grep's
error status (2) and a missing grep (127) as "no match", so the hook used to fall
through to ``exit 0``. Measured 2026-10-06 with a grep shim on PATH: an
assistant marker, an attribution phrase, a co-author trailer and an em-dash were
all ACCEPTED, as was every clean message. Now any status above 1 rejects the
message and names grep and the status.

The assistant names below are assembled from fragments so that no tracked file
spells an attribution phrase out whole.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_HOOK = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "commit-msg"
_EM_DASH = chr(0x2014)

_MESSAGES = {
    "marker": "docs: tidy\n\nGenerated " + "with a tool\n",
    "attribution": "fix: tidy\n\nwritten by " + "clau" + "de\n",
    "co-author": "fix: tidy\n\nCo-" + "authored-by: Someone <someone@example.invalid>\n",
    "em-dash": "fix: one " + _EM_DASH + " two\n",
}
_CLEAN = "fix: a plain message\n"


def _run(
    tmp_path: Path, message: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    msg_file = tmp_path / "COMMIT_EDITMSG"
    msg_file.write_text(message, encoding="utf-8", newline="\n")
    return subprocess.run(
        ["bash", str(_HOOK), str(msg_file)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _with_grep_shim(tmp_path: Path, body: str) -> dict[str, str]:
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "grep"
    shim.write_bytes(("#!/bin/sh\n" + body).encode("ascii"))
    shim.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = str(shim_dir) + os.pathsep + env["PATH"]
    return env


@pytest.mark.parametrize("kind", sorted(_MESSAGES))
def test_each_check_rejects_its_violation_with_a_working_grep(tmp_path: Path, kind: str) -> None:
    result = _run(tmp_path, _MESSAGES[kind])
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Commit rejected" in result.stderr


def test_a_clean_message_is_accepted_with_a_working_grep(tmp_path: Path) -> None:
    result = _run(tmp_path, _CLEAN)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("status", [2, 127])
@pytest.mark.parametrize("kind", [*sorted(_MESSAGES), "clean"])
def test_a_grep_that_cannot_run_rejects_every_message(
    tmp_path: Path, kind: str, status: int
) -> None:
    env = _with_grep_shim(tmp_path, f"echo 'grep: simulated failure' >&2\nexit {status}\n")
    result = _run(tmp_path, _MESSAGES.get(kind, _CLEAN), env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"'grep' exited {status}" in result.stderr


# One argument per check, each used by that check alone: the first bare marker,
# the -qiE of the attribution pattern, the co-author text, and the lone -q of the
# em-dash test. A grep that fails EVERY call trips the first check and masks the
# rest, so each is tested on its own, against a clean message.
_TARGETS = {
    "bare-marker-loop": "ChatGPT",
    "attribution-pattern": "-qiE",
    "co-author-trailer": "co-authored-by",
    "em-dash": "-q",
}


@pytest.mark.parametrize("check", sorted(_TARGETS))
def test_a_grep_failing_in_one_check_still_rejects(tmp_path: Path, check: str) -> None:
    real_grep = subprocess.run(
        ["bash", "-c", "command -v grep"], capture_output=True, text=True, check=True
    ).stdout.strip()
    env = _with_grep_shim(
        tmp_path,
        f"for a in \"$@\"; do [ \"$a\" = '{_TARGETS[check]}' ] && {{ echo 'grep: simulated failure' >&2; exit 2; }}; done\n"
        f'exec "{real_grep}" "$@"\n',
    )
    result = _run(tmp_path, _CLEAN, env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "'grep' exited 2" in result.stderr
