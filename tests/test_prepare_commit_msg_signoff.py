"""The DCO sign-off must survive `git commit -v`.

Why this test exists
--------------------
`scripts/hooks/prepare-commit-msg` added the `Signed-off-by:` trailer by appending text
to the END of the message file. Under `git commit -v` (or `commit.verbose=true`) that
file ends with a scissors line followed by the staged diff, and git discards everything
below the scissors line when it finalizes the message. The appended trailer was
therefore silently dropped, the commit landed unsigned, and `.github/workflows/dco.yml`
failed the pull request that carried it.

Measured 2026-09-23 on git 2.55.0.windows.3 against the unfixed hook, counting
`Signed-off-by:` lines in `git log -1 --format=%B`:

    git commit -m                              1
    git commit -e -m                           1
    git commit -e -v -m                        0
    git -c commit.verbose=true commit -e -m    0
    git commit -s -m                           1

The fix inserts the trailer with `git interpret-trailers`, which places it after the
last line of the real message: above the trailing comment block and the scissors line.

Anti-vacuity
------------
`test_text_appended_below_the_scissors_line_is_discarded` is the load-bearing anchor.
The regression cases only mean something if `-e -v` really hands the hook a file with
a scissors line in it AND git really discards what is appended after it. If a git
release, a config leak or an editor setting stopped producing the verbose section, the
old append-to-end hook would score 1 everywhere and every case below would pass
against it. The anchor asserts that premise directly with a probe hook.

The companions guard the other directions: every non-verbose form must still produce
exactly one sign-off (the fix must not duplicate `git commit -s` or an existing
trailer), and an otherwise empty message must still abort the commit.

Two further groups pin the hook's own safety options, which the harness would never
exercise on its own because it nulls the global and system config: developer
`trailer.*` config must not drop or move the sign-off (the --where / --if-exists /
--if-missing pins), and a failing `git interpret-trailers` must warn and exit 0 with the
message untouched (the never-block guard).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_SOURCE = REPO_ROOT / "scripts" / "hooks" / "prepare-commit-msg"

NAME = "Test Person"
EMAIL = "test@example.invalid"
SIGNOFF = f"Signed-off-by: {NAME} <{EMAIL}>"
OTHER_SIGNOFF = "Signed-off-by: Other Person <other@example.invalid>"
SCISSORS = "------------------------ >8 ------------------------"
PROBE_MARKER = "PROBE-APPENDED-AT-END-OF-FILE"

# git reads the developer's GLOBAL and SYSTEM config even inside a throwaway repo, so
# without this the result depends on the machine rather than on the hook (for example
# commit.gpgsign=true with no key, or a global core.hooksPath). GIT_EDITOR=true makes
# `-e` a no-op editor so the verbose message file is produced without a terminal.
GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_EDITOR": "true",
}
# `git commit -s` signs with the COMMITTER identity, which these variables override,
# while the hook signs with user.name/user.email. A runner that exported them would
# make the -s case report two sign-offs for a reason unrelated to the hook.
for _var in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
    GIT_ENV.pop(_var, None)

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("git") is None,
    reason="prepare-commit-msg is a bash script run by git; both are required",
)


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=check,
        env=GIT_ENV,
    )


def _make_repo(tmp_path: Path, hook_text: str) -> Path:
    """A throwaway repo whose only hook is prepare-commit-msg, holding `hook_text`."""
    repo = tmp_path / "sandbox"
    repo.mkdir()
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    hook = hooks / "prepare-commit-msg"
    hook.write_text(hook_text, encoding="utf-8", newline="\n")
    hook.chmod(0o755)
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", NAME)
    _git(repo, "config", "user.email", EMAIL)
    # Installed through git, not invoked directly: the defect lives in how `git commit`
    # finalizes the file the hook edited, so only a real commit can observe it.
    _git(repo, "config", "core.hooksPath", hooks.as_posix())
    return repo


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    return _make_repo(tmp_path, HOOK_SOURCE.read_text(encoding="utf-8"))


def _posix(path: Path) -> str:
    """PATH entries must be POSIX-shaped; Git Bash does not search `C:/...` entries."""
    text = path.as_posix()
    if len(text) > 1 and text[1] == ":":
        text = "/" + text[0].lower() + text[2:]
    return text


def _stage_change(repo: Path, content: str) -> None:
    (repo / "f.txt").write_text(content + "\n", encoding="utf-8")
    _git(repo, "add", "f.txt")


def _last_message(repo: Path) -> str:
    return _git(repo, "log", "-1", "--format=%B").stdout


def _assert_signed_once_at_end(message: str) -> None:
    lines = [line for line in message.splitlines() if line.strip()]
    signoffs = [line for line in lines if line.startswith("Signed-off-by:")]
    assert signoffs == [SIGNOFF], (
        f"expected exactly one sign-off {SIGNOFF!r}; got {signoffs!r} in:\n{message}"
    )
    assert lines[-1] == SIGNOFF, (
        f"the sign-off must close the message as a trailer; message was:\n{message}"
    )


COMMIT_FORMS = [
    pytest.param([], ["-m", "subject"], id="commit -m"),
    pytest.param([], ["-e", "-m", "subject"], id="commit -e -m"),
    pytest.param([], ["-e", "-v", "-m", "subject"], id="commit -e -v -m"),
    pytest.param(
        ["-c", "commit.verbose=true"], ["-e", "-m", "subject"], id="commit.verbose=true -e -m"
    ),
    pytest.param([], ["-s", "-m", "subject"], id="commit -s -m"),
    pytest.param([], ["-e", "-v", "-s", "-m", "subject"], id="commit -e -v -s -m"),
    pytest.param([], ["-m", f"subject\n\n{SIGNOFF}"], id="message already signed"),
    pytest.param(
        [], ["-e", "-v", "-m", f"subject\n\n{SIGNOFF}"], id="message already signed, -e -v"
    ),
    # A "---" line in a commit body is prose, not a patch divider. Without --no-divider
    # interpret-trailers would insert the sign-off ABOVE it, mid-message, which is not
    # where `git commit -s` puts it.
    pytest.param([], ["-m", "subject\n\nintro\n\n---\nafter a rule"], id="body has --- line"),
    pytest.param(
        [], ["-e", "-v", "-m", "subject\n\nintro\n\n---\nafter a rule"], id="body has ---, -e -v"
    ),
]


@pytest.mark.parametrize(("git_opts", "commit_args"), COMMIT_FORMS)
def test_every_commit_form_carries_exactly_one_signoff(
    repo: Path, git_opts: list[str], commit_args: list[str]
) -> None:
    """The regression: the -v forms scored 0 before the fix; every form must score 1."""
    _stage_change(repo, "one")
    _git(repo, *git_opts, "commit", "--quiet", *commit_args)
    _assert_signed_once_at_end(_last_message(repo))


# Each row is a developer trailer.* setting that, reaching the hook's own
# `git interpret-trailers` call, would defeat one of its pinned options: the first
# drops the sign-off (--if-missing add), the second places it above an existing
# trailer (--where end), and the third drops it when someone else already signed
# (--if-exists addIfDifferent).
TRAILER_CONFIGS = [
    pytest.param("trailer.ifmissing=doNothing", "subject", id="trailer.ifmissing=doNothing"),
    pytest.param("trailer.where=start", "subject\n\nRefs: 12", id="trailer.where=start"),
    pytest.param(
        "trailer.ifexists=doNothing",
        f"subject\n\n{OTHER_SIGNOFF}",
        id="trailer.ifexists=doNothing",
    ),
]


@pytest.mark.parametrize(("config", "message"), TRAILER_CONFIGS)
def test_developer_trailer_config_cannot_drop_or_move_the_signoff(
    repo: Path, config: str, message: str
) -> None:
    """The option pins: sign-off present once, last, and nothing already there lost."""
    _stage_change(repo, "one")
    _git(repo, "-c", config, "commit", "--quiet", "-m", message)
    final = _last_message(repo)
    lines = [line for line in final.splitlines() if line.strip()]
    assert lines.count(SIGNOFF) == 1, f"expected exactly one {SIGNOFF!r} in:\n{final}"
    assert lines[-1] == SIGNOFF, f"the sign-off must close the message; got:\n{final}"
    for original in message.splitlines():
        if original.strip():
            assert original in lines, f"{original!r} was lost; message was:\n{final}"


def test_failing_interpret_trailers_warns_and_leaves_the_message_alone(
    repo: Path, tmp_path: Path
) -> None:
    """The never-block guard: a refused interpret-trailers must not fail the commit.

    A `git` shim that refuses only `interpret-trailers` is put first on PATH and the hook
    is run directly. It must exit 0, say why on stderr, and leave the file byte-identical.
    The control run without the shim proves the direct invocation signs normally.
    """
    real_git = shutil.which("git")
    assert real_git is not None
    shim = tmp_path / "shimbin"
    shim.mkdir()
    (shim / "git").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = interpret-trailers ]; then\n'
        "  echo 'shim: interpret-trailers refused' >&2; exit 1\n"
        "fi\n"
        f'exec "{real_git}" "$@"\n',
        encoding="ascii",
        newline="\n",
    )
    (shim / "git").chmod(0o755)
    msg_file = tmp_path / "MSG"
    msg_file.write_bytes(b"subject\n")
    original = msg_file.read_bytes()
    # The LF copy the fixture installed, not HOOK_SOURCE: a CRLF checkout must not decide
    # the outcome.
    hook = tmp_path / "hooks" / "prepare-commit-msg"

    shimmed_env = {**GIT_ENV, "PATH": _posix(shim) + os.pathsep + GIT_ENV.get("PATH", "")}
    refused = subprocess.run(
        ["bash", str(hook), str(msg_file)],
        cwd=repo,
        capture_output=True,
        text=True,
        env=shimmed_env,
    )
    assert "shim: interpret-trailers refused" in refused.stderr, (
        f"the shim was never reached, so this proves nothing:\n{refused.stderr}"
    )
    assert refused.returncode == 0, f"the hook blocked the commit:\n{refused.stderr}"
    assert "interpret-trailers failed" in refused.stderr
    assert msg_file.read_bytes() == original, "a failed call must leave the message alone"

    control = subprocess.run(
        ["bash", str(hook), str(msg_file)],
        cwd=repo,
        capture_output=True,
        text=True,
        env=GIT_ENV,
    )
    assert control.returncode == 0, control.stderr
    _assert_signed_once_at_end(msg_file.read_text(encoding="utf-8"))


def test_amend_under_verbose_signs_and_does_not_duplicate(repo: Path) -> None:
    """--amend -e -v must sign an unsigned commit once, and not re-sign a signed one."""
    no_hooks = repo.parent / "no_hooks"
    no_hooks.mkdir()
    _stage_change(repo, "one")
    _git(
        repo,
        "-c",
        f"core.hooksPath={no_hooks.as_posix()}",
        "commit",
        "--quiet",
        "-m",
        "unsigned subject",
    )
    assert "Signed-off-by:" not in _last_message(repo), "fixture commit must start unsigned"

    _stage_change(repo, "two")
    _git(repo, "commit", "--quiet", "--amend", "-e", "-v")
    _assert_signed_once_at_end(_last_message(repo))

    _stage_change(repo, "three")
    _git(repo, "commit", "--quiet", "--amend", "-e", "-v")
    _assert_signed_once_at_end(_last_message(repo))


def test_empty_message_still_aborts_the_commit(repo: Path) -> None:
    """Not the degenerate fix: a sign-off alone must not make an empty message valid."""
    _stage_change(repo, "one")
    _git(repo, "commit", "--quiet", "-m", "baseline")
    _stage_change(repo, "two")
    result = _git(repo, "commit", "--quiet", "-v", check=False)
    assert result.returncode != 0, (
        "an editor session that leaves no message must abort even with the hook's "
        f"sign-off present.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "1"


def test_text_appended_below_the_scissors_line_is_discarded(tmp_path: Path) -> None:
    """Anti-vacuity anchor: the premise of the -v regression cases must actually hold.

    A probe hook records the file it was handed and then appends a marker to its end,
    exactly as the unfixed hook appended the sign-off. Under -e -v the file must carry
    the scissors line and the marker must be discarded; without -v the marker must
    survive, proving the probe itself works.
    """
    probe = (
        "#!/bin/sh\n"
        'cp "$1" "$1.seen"\n'
        f"printf '\\n%s\\n' '{PROBE_MARKER}' >> \"$1\"\n"
    )
    repo = _make_repo(tmp_path, probe)
    seen = repo / ".git" / "COMMIT_EDITMSG.seen"

    _stage_change(repo, "one")
    _git(repo, "commit", "--quiet", "-e", "-m", "plain")
    assert SCISSORS not in seen.read_text(encoding="utf-8")
    assert PROBE_MARKER in _last_message(repo), "control: without -v the append survives"

    _stage_change(repo, "two")
    _git(repo, "commit", "--quiet", "-e", "-v", "-m", "verbose")
    handed = seen.read_text(encoding="utf-8")
    assert SCISSORS in handed, f"-e -v did not produce a scissors line:\n{handed}"
    assert "diff --git" in handed.split(SCISSORS, 1)[1], "no diff below the scissors line"
    assert PROBE_MARKER not in _last_message(repo), (
        "git kept text appended below the scissors line, so the -v cases above "
        "no longer exercise the defect they are named for"
    )
