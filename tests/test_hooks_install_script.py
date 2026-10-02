"""`scripts/hooks/install.sh` must activate the tracked hooks, and fail loudly when it cannot.

Why this test exists
--------------------
`install.sh` is the activation route CONTRIBUTING.md documents, in both its
"Development Environment Setup" steps and its "Pull Request Checklist". It runs
`git config core.hooksPath scripts/hooks` and then `chmod +x` on four named hooks,
under `set -euo pipefail`. No test ran it: the only test naming it checks that it is
LF, and the tests that run a hook either invoke it with bash directly or point
`core.hooksPath` at a copy themselves. A broken installer would leave every one of them
green, and a clone whose installer silently did nothing would have no local gates at
all.

The chmod is not cosmetic on POSIX. `git ls-tree` at 153d2b06 shows `pre-commit`,
`commit-msg` and `pre-push` tracked as mode 100644 and only `prepare-commit-msg` as
100755, so a Linux or macOS checkout gets three of the four non-executable, and git's
own githooks documentation says hooks without the executable bit set are ignored.

What is asserted, and how each is made observable
-------------------------------------------------
The installer is always run for real, by bash, exactly as documented
(`bash scripts/hooks/install.sh` from the clone root), but inside a throwaway clone
under `tmp_path` that holds a byte copy of this checkout's `scripts/hooks/`. It finds
its clone from its working directory (`git rev-parse --show-toplevel`, then a plain
`git config`), so once git's repository variables are cleared (see Hermeticity) the
working directory decides which clone it configures.

- `core.hooksPath` is set to the RELATIVE value `scripts/hooks`, read back with the
  exact verify command the script prints.
- `chmod +x` is called on each of the four named hooks, in the script's loop order.
  This is observed through a logging `chmod` shim first on PATH, because the execute
  bit itself cannot be observed on Windows (see the skip reason on the POSIX test).
- On POSIX, the four hooks, copied in as 0644 to match a 100644 checkout, end up
  executable.
- The header comment's promise that "the setting applies to every worktree of this
  clone": after one run, from either worktree, git resolves the hooks directory of
  EACH worktree to that worktree's own `scripts/hooks`. An absolute value would send
  one of the two to the other's copies.
- The counterfactual: with one named hook missing, `set -e` stops the script at the
  failing chmod. It exits non-zero, names the missing path on stderr, never prints its
  success banner, and never reaches the hook after the missing one.

Hermeticity
-----------
Every git and bash call, except the one that produces the list, gets an environment
with git's own list of repository-local variables removed (`git rev-parse
--local-env-vars`) and the global and system config pointed at the null device.
`git -c core.hooksPath=...` exports GIT_CONFIG_PARAMETERS to the processes it starts,
and a fresh repo then answers `git config --get core.hooksPath` with the injected value
although nothing was written (measured on git 2.55.0.windows.3 and 2.53.0). If that
value reached the read-back, the assertion would pass even if the installer never set
the value.

The root conftest drops GIT_CONFIG_PARAMETERS, GIT_CONFIG_COUNT and each
GIT_CONFIG_KEY_<n> / GIT_CONFIG_VALUE_<n> it finds, beside GIT_DIR, GIT_WORK_TREE,
GIT_INDEX_FILE and GIT_PREFIX, before any test runs. While that conftest is loaded,
this file's own removal of GIT_CONFIG_PARAMETERS is redundant. It is kept as a second
guard for a run in which the conftest's drop is missing; the GIT_CONFIG_PARAMETERS
entry under Anti-vacuity measures both halves. The conftest does not drop the other
names on git's list and does not null the global or system config; this file does both
itself.

`test_the_sandbox_starts_with_no_hooks_path` is the anchor that proves the read-back
starts empty. An autouse fixture reads the REAL clone's `core.hooksPath` before and
after each test and reports an error at teardown if it changed.

Anti-vacuity
------------
Measured 2026-10-02 against scratch copies of the installer, each with one behaviour
removed, under Git Bash on Windows and on WSL2 Ubuntu. Both failed the same tests,
apart from the POSIX-only one, which runs on Linux alone:

- the `git config` line deleted, or its value made absolute
  (`${REPO_ROOT}/scripts/hooks`): the read-back test, both worktree cases and the
  counterfactual fail.
- the `chmod` call removed: the read-back test (the shim sees no call) and the
  counterfactual fail, and on Linux the POSIX test.
- `pre-push` dropped from the loop: the read-back test fails, and on Linux the POSIX
  test.
- the banner or the verify line deleted: the read-back test fails.
- `|| true` appended to the chmod, or `-e` removed from `set -euo pipefail`: only the
  counterfactual fails.
- GIT_CONFIG_PARAMETERS injected with the `git config` line deleted: the same four
  still fail, and they still fail with this file's scrub also removed, because the
  root conftest drops the variable first. The anchor passes in both runs. With the
  conftest's entry for GIT_CONFIG_PARAMETERS also removed in the copy, the four still
  fail while this file's scrub is kept; with the scrub removed as well, those four
  pass and only the anchor fails.
- a copy that also runs `git config` in the repository holding this file: the autouse
  check reports an error at teardown.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOKS_SOURCE = REPO_ROOT / "scripts" / "hooks"

# Relative on purpose: CONTRIBUTING.md documents `bash scripts/hooks/install.sh`, run from
# the clone root, and that is the form run here.
INSTALLER = "scripts/hooks/install.sh"

# The hooks install.sh's loop names, in its loop order. CONTRIBUTING.md's checklist names
# the same four. The counterfactual depends on the order: it removes the third.
NAMED_HOOKS = ("pre-commit", "prepare-commit-msg", "commit-msg", "pre-push")

BANNER = "SESTRAV hooks activated via core.hooksPath -> scripts/hooks/"
VERIFY_LINE = "Verify with: git config --get core.hooksPath"

_GIT = shutil.which("git")
_BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(
    _GIT is None or _BASH is None,
    reason="install.sh is a bash script that runs git; without both it cannot run at all",
)


@pytest.fixture(scope="module")
def git_env() -> dict[str, str]:
    """The environment the git and bash calls in this file run under.

    The one exception is the `--local-env-vars` call below, which runs with the
    inherited environment because it produces the list of names to remove.

    Built in a fixture, not at import: a failing git call at import time is a
    collection error, and a collection error means no test in the run executed.
    """
    assert _GIT is not None
    names = subprocess.run(
        [_GIT, "rev-parse", "--local-env-vars"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert "GIT_CONFIG_PARAMETERS" in names, f"unexpected --local-env-vars output: {names}"
    env = {key: value for key, value in os.environ.items() if key not in names}
    # The developer's own config must not decide the outcome (a global core.hooksPath,
    # commit.gpgsign=true with no key, core.autocrlf rewriting the copied hooks).
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    return env


@pytest.fixture(scope="module")
def bash(git_env: dict[str, str]) -> str:
    """The bash to run the installer with, refused when it is not a usable one.

    On Windows, `bash` on PATH can be the WSL launcher rather than Git Bash. That bash
    runs Linux git, and inside it GIT_CONFIG_GLOBAL and a probe variable set by the
    caller were measured unset, so the isolation in `git_env` would not reach the
    installer. Measured on a Windows 11 workstation whose login PATH resolves `bash` to
    the launcher: `uname -s` printed Linux and `command -v git` printed /usr/bin/git.
    The check below skips unless `uname -s` from the bash on PATH starts with MINGW,
    MSYS or CYGWIN, so it admits more than Git Bash alone.
    """
    assert _BASH is not None
    if os.name == "nt":
        kernel = subprocess.run(
            [_BASH, "-c", "uname -s"], capture_output=True, text=True, env=git_env
        ).stdout.strip()
        if not kernel.startswith(("MINGW", "MSYS", "CYGWIN")):
            pytest.skip(
                f"bash on PATH ({_BASH}) is not Git Bash: `uname -s` printed {kernel!r}. "
                "A Linux kernel name here means the WSL launcher, which runs Linux git "
                "and inside which GIT_CONFIG_GLOBAL and a probe variable set by the "
                "caller were measured unset, so the config isolation this file depends "
                "on would not reach the installer."
            )
    return _BASH


def _real_clone_hooks_path(env: dict[str, str]) -> tuple[int, str]:
    assert _GIT is not None
    probe = subprocess.run(
        [_GIT, "-C", str(REPO_ROOT), "config", "--get", "core.hooksPath"],
        capture_output=True,
        text=True,
        env=env,
    )
    return probe.returncode, probe.stdout


@pytest.fixture(autouse=True)
def _real_clone_is_untouched(git_env: dict[str, str]):
    """The installer must only ever configure the throwaway clone.

    Read with the same isolated environment, so this compares the clone's own local
    config, which is the only file a leaked `git config` could write.
    """
    before = _real_clone_hooks_path(git_env)
    yield
    after = _real_clone_hooks_path(git_env)
    assert after == before, (
        f"core.hooksPath of the real clone at {REPO_ROOT} changed during this test: "
        f"(rc, value) before {before!r}, after {after!r}"
    )


def _git(
    repo: Path, env: dict[str, str], *args: str, check: bool = True
) -> subprocess.CompletedProcess:
    assert _GIT is not None
    return subprocess.run(
        [_GIT, *args], cwd=repo, capture_output=True, text=True, check=check, env=env
    )


def _posix(path: Path) -> str:
    """PATH entries must be POSIX-shaped; Git Bash does not search `C:/...` entries."""
    text = path.as_posix()
    if len(text) > 1 and text[1] == ":":
        text = "/" + text[0].lower() + text[2:]
    return text


@pytest.fixture()
def sandbox(tmp_path: Path, git_env: dict[str, str]) -> Path:
    """A throwaway clone holding a byte copy of this checkout's scripts/hooks/."""
    clone = tmp_path / "clone"
    clone.mkdir()
    _git(clone, git_env, "init", "--quiet")
    target = clone / "scripts" / "hooks"
    target.mkdir(parents=True)
    for source in sorted(HOOKS_SOURCE.iterdir()):
        if source.is_file():
            copy = target / source.name
            # Bytes, not text: line endings must be what a contributor's checkout holds.
            copy.write_bytes(source.read_bytes())
            # What a POSIX checkout gives a 100644 entry, so the chmod is observable.
            copy.chmod(0o644)
    return clone


@pytest.fixture()
def chmod_shim(
    tmp_path: Path, git_env: dict[str, str], bash: str
) -> tuple[dict[str, str], Path]:
    """An environment whose first `chmod` on PATH logs its arguments, then runs the real one.

    The real chmod is resolved by bash itself, so the shim delegates to the same binary
    the installer would have run without it.
    """
    real_chmod = subprocess.run(
        [bash, "-c", "command -v chmod"],
        capture_output=True,
        text=True,
        check=True,
        env=git_env,
    ).stdout.strip()
    assert real_chmod, "bash could not resolve chmod"
    shim_dir = tmp_path / "shimbin"
    shim_dir.mkdir()
    shim = shim_dir / "chmod"
    shim.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$CHMOD_SHIM_LOG\"\n"
        f'exec "{real_chmod}" "$@"\n',
        encoding="ascii",
        newline="\n",
    )
    shim.chmod(0o755)
    log = tmp_path / "chmod_calls.log"
    env = {
        **git_env,
        "PATH": _posix(shim_dir) + os.pathsep + git_env.get("PATH", ""),
        "CHMOD_SHIM_LOG": log.as_posix(),
    }
    return env, log


def _run_installer(
    workdir: Path, bash: str, env: dict[str, str]
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [bash, INSTALLER], cwd=workdir, capture_output=True, text=True, env=env, check=False
    )


def _chmod_calls(log: Path) -> list[tuple[str, str]]:
    """(mode argument, target path) per logged chmod call; empty if none was made."""
    if not log.exists():
        return []
    calls = []
    for line in log.read_text(encoding="utf-8").splitlines():
        mode, _, target = line.partition(" ")
        calls.append((mode, target))
    return calls


def _describe(result: subprocess.CompletedProcess) -> str:
    return f"rc={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_the_sandbox_starts_with_no_hooks_path(
    sandbox: Path, git_env: dict[str, str]
) -> None:
    """Anti-vacuity anchor: before the installer runs, nothing supplies core.hooksPath.

    If an inherited GIT_CONFIG_PARAMETERS, or a global or system config, still reached
    the sandbox, the read-back assertions below would pass without the installer
    writing anything.
    """
    probe = _git(sandbox, git_env, "config", "--get", "core.hooksPath", check=False)
    assert (probe.returncode, probe.stdout) == (1, ""), (
        f"core.hooksPath is already visible in a fresh repo: {_describe(probe)}"
    )


def test_installer_sets_hooks_path_and_chmods_the_named_hooks(
    sandbox: Path, bash: str, chmod_shim: tuple[dict[str, str], Path], git_env: dict[str, str]
) -> None:
    env, log = chmod_shim
    result = _run_installer(sandbox, bash, env)

    assert result.returncode == 0, f"the installer failed:\n{_describe(result)}"
    assert BANNER in result.stdout, f"no success banner:\n{_describe(result)}"
    assert VERIFY_LINE in result.stdout, f"no verify instruction:\n{_describe(result)}"

    # The script's own verify command, run as it prints it, from the clone root.
    verify = _git(sandbox, git_env, "config", "--get", "core.hooksPath", check=False)
    assert (verify.returncode, verify.stdout.strip()) == (0, "scripts/hooks"), (
        f"expected the relative value scripts/hooks:\n{_describe(verify)}"
    )

    calls = _chmod_calls(log)
    assert [Path(target).name for _, target in calls] == list(NAMED_HOOKS), (
        f"chmod calls seen by the shim: {calls!r}"
    )
    hooks_dir = sandbox / "scripts" / "hooks"
    for mode, target in calls:
        assert mode == "+x", f"unexpected chmod mode in {calls!r}"
        assert os.path.samefile(Path(target).parent, hooks_dir), (
            f"chmod reached {target!r}, outside the sandbox clone's scripts/hooks"
        )


@pytest.mark.skipif(
    os.name == "nt",
    reason=(
        "NTFS has no POSIX execute bit to observe. Measured on Windows 11 with Git Bash: "
        "chmod +x exits 0 and changes nothing, os.stat reports 0o100666 and "
        "os.access(X_OK) is True before and after, and Git Bash's test -x follows the "
        "shebang. The chmod calls themselves are still asserted here, through the shim "
        "in test_installer_sets_hooks_path_and_chmods_the_named_hooks."
    ),
)
def test_named_hooks_end_up_executable_on_posix(
    sandbox: Path, bash: str, git_env: dict[str, str]
) -> None:
    hooks_dir = sandbox / "scripts" / "hooks"
    for name in NAMED_HOOKS:
        assert not (hooks_dir / name).stat().st_mode & stat.S_IXUSR, (
            f"premise failed: {name} is executable before the installer ran"
        )

    result = _run_installer(sandbox, bash, git_env)
    assert result.returncode == 0, f"the installer failed:\n{_describe(result)}"

    not_executable = [
        name for name in NAMED_HOOKS if not (hooks_dir / name).stat().st_mode & stat.S_IXUSR
    ]
    assert not_executable == [], f"still not executable after the installer: {not_executable}"


@pytest.mark.parametrize("run_from", ["main checkout", "linked worktree"])
def test_one_run_reaches_every_worktree_of_the_clone(
    sandbox: Path, tmp_path: Path, bash: str, git_env: dict[str, str], run_from: str
) -> None:
    """The header comment's worktree promise, from either side.

    git resolves a relative core.hooksPath against each worktree's own root, so every
    worktree runs the hooks it has checked out. Asserted through git's own resolution
    (`rev-parse --git-path hooks`), not by re-reading the stored string.
    """
    _git(sandbox, git_env, "add", "scripts/hooks")
    _git(
        sandbox,
        git_env,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "--no-verify",
        "-m",
        "hooks",
    )
    linked = tmp_path / "linked"
    _git(sandbox, git_env, "worktree", "add", "--quiet", "--detach", str(linked))

    workdir = sandbox if run_from == "main checkout" else linked
    result = _run_installer(workdir, bash, git_env)
    assert result.returncode == 0, f"the installer failed:\n{_describe(result)}"

    for tree in (sandbox, linked):
        resolved = _git(
            tree, git_env, "rev-parse", "--path-format=absolute", "--git-path", "hooks"
        ).stdout.strip()
        assert os.path.samefile(resolved, tree / "scripts" / "hooks"), (
            f"after a run from the {run_from}, git resolves the hooks of {tree} to "
            f"{resolved!r}, not to that worktree's own scripts/hooks"
        )


def test_a_missing_named_hook_fails_loudly_and_stops_the_run(
    sandbox: Path, bash: str, chmod_shim: tuple[dict[str, str], Path], git_env: dict[str, str]
) -> None:
    """The counterfactual: a named hook that is not there must not be skipped quietly.

    Reachable whenever the tracked hook set and the script's list drift apart, for
    example a hook deleted or renamed without updating the loop. `commit-msg` is the
    third of the four, so the shim log shows exactly where `set -e` stopped.

    The last assertion pins the state a failed run leaves today: the script writes
    core.hooksPath BEFORE its chmod loop, so the value is already set when the loop
    fails. That is current ordering, not a requirement. If the script is changed to
    configure only after every chmod succeeds, change that assertion with it.
    """
    missing = "commit-msg"
    (sandbox / "scripts" / "hooks" / missing).unlink()
    env, log = chmod_shim

    result = _run_installer(sandbox, bash, env)

    assert result.returncode != 0, f"a missing hook was not an error:\n{_describe(result)}"
    assert f"scripts/hooks/{missing}" in result.stderr, (
        f"the failure does not name the missing hook:\n{_describe(result)}"
    )
    assert BANNER not in result.stdout, (
        f"the success banner was printed after a failure:\n{_describe(result)}"
    )
    assert [Path(target).name for _, target in _chmod_calls(log)] == [
        "pre-commit",
        "prepare-commit-msg",
        missing,
    ], f"set -e did not stop the loop at the failure; calls: {_chmod_calls(log)!r}"
    leftover = _git(sandbox, git_env, "config", "--get", "core.hooksPath", check=False)
    assert (leftover.returncode, leftover.stdout.strip()) == (0, "scripts/hooks"), (
        f"core.hooksPath after the failed run:\n{_describe(leftover)}"
    )
