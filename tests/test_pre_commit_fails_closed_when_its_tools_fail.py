"""pre-commit must block, not pass, when git or grep cannot do the scan it relies on.

Measured 2026-10-06 against the hook as it stood, with the tool replaced by a shim
on PATH, one throwaway repository per case:

- git exiting 128 (as on a repository it refuses, "dubious ownership") made the
  staged-file list empty, and every gate iterates that list, so a planted
  violation passed all four gates;
- grep exiting 2 (an error) or 127 (not found) was read by every ``if grep ...``
  as "no match", so the same four violations passed again;
- Gate 3 reads the staged diff, and three ordinary pieces of user configuration
  changed that diff's text enough to hide an em-dash: ``color.ui=always`` (added
  lines start with an escape, so ``^+`` never matches), an external diff command
  (it replaces git's output; measured, git then exits 0 with no ``+`` line), and a
  textconv driver named by ``.gitattributes``.

Every clean twin passed before and must still pass, except under a broken grep,
where nothing can be cleared and a clean commit blocks too.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_HOOK = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "pre-commit"
_EM_DASH = chr(0x2014)

# One violation per gate, each with a clean twin at the same path.
_PLANTS = {
    "gate1-path": ("notes/" + "clau" + "de.md", "configuration\n"),
    "gate2-credential": ("candidate.txt", "value=" + "AK" + "IA" + "A1B2C3D4E5F6G7H8" + "\n"),
    "gate3-em-dash": ("doc.md", "one " + _EM_DASH + " two\n"),
    "gate4-workstation-path": (
        "candidate.txt",
        "see " + "c:" + "/" + "users" + "/developer/private\n",
    ),
}
_CLEAN = {
    "gate1-path": ("notes/readme.md", "configuration\n"),
    "gate2-credential": ("candidate.txt", "value=placeholder\n"),
    "gate3-em-dash": ("doc.md", "one - two\n"),
    "gate4-workstation-path": ("candidate.txt", "see docs/readme\n"),
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repo(tmp_path: Path, rel: str, content: str, attributes: str | None = None) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "fail-closed-scratch")
    _git(repo, "config", "user.email", "fail-closed-scratch@invalid")
    if attributes is not None:
        (repo / ".gitattributes").write_text(attributes, encoding="utf-8")
        _git(repo, "add", ".gitattributes")
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _git(repo, "add", "-f", rel)
    return repo


def _isolated_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    for name in ("GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT", "GIT_EXTERNAL_DIFF"):
        env.pop(name, None)
    empty = tmp_path / "empty.gitconfig"
    empty.write_text("", encoding="utf-8")
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = str(empty)
    return env


def _shim(tmp_path: Path, env: dict[str, str], name: str, status: int, message: str) -> None:
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir(exist_ok=True)
    shim = shim_dir / name
    shim.write_bytes(f'#!/bin/sh\necho "{message}" >&2\nexit {status}\n'.encode("ascii"))
    shim.chmod(0o755)
    env["PATH"] = str(shim_dir) + os.pathsep + env["PATH"]


def _run(repo: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_HOOK)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


@pytest.mark.parametrize("gate", sorted(_PLANTS))
def test_a_git_that_cannot_list_the_staged_files_blocks(tmp_path: Path, gate: str) -> None:
    rel, content = _PLANTS[gate]
    repo = _repo(tmp_path, rel, content)
    env = _isolated_env(tmp_path)
    _shim(tmp_path, env, "git", 128, "fatal: detected dubious ownership in repository")
    result = _run(repo, env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "[BLOCKED]" in result.stderr
    assert "git could not list the staged files" in result.stderr


@pytest.mark.parametrize("status", [2, 127])
@pytest.mark.parametrize("gate", sorted(_PLANTS))
@pytest.mark.parametrize("kind", ["plant", "clean"])
def test_a_grep_that_cannot_scan_blocks_even_a_clean_commit(
    tmp_path: Path, gate: str, status: int, kind: str
) -> None:
    rel, content = (_PLANTS if kind == "plant" else _CLEAN)[gate]
    repo = _repo(tmp_path, rel, content)
    env = _isolated_env(tmp_path)
    _shim(tmp_path, env, "grep", status, "grep: simulated failure")
    result = _run(repo, env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "[BLOCKED]" in result.stderr
    assert f"'grep' exited {status}" in result.stderr


# A grep that fails for ONE argument and runs the real grep otherwise. That is the
# realistic shape of a dialect error: one pattern a given grep rejects, as the PEM
# pattern once did without '--'. The shim above fails EVERY call, so the first gate
# to call grep blocks and every later site is masked; each site is therefore tested
# here on its own. The targets: one pattern used only by Gate 1, Gate 2 and Gate 4
# each; '^+', Gate 3's diff prefilter; a lone '-q', only ever Gate 3's em-dash test;
# '-oE', Gate 4's home-path capture; and '-qvE', Gate 4's allowlist check, which
# runs only when a home path is present. Every other case stages a file the hook
# clears, so the failing call is the only thing that can block it.
_TARGETED = {
    "gate1-one-pattern": ("^\\.codex/", "notes/readme.md", "configuration\n", "'grep' exited 2"),
    "gate2-pem-pattern": (
        "-----BEGIN [A-Z ]*PRIVATE KEY-----",
        "candidate.txt",
        "value=placeholder\n",
        "'grep' exited 2",
    ),
    "gate3-diff-prefilter": ("^+", "doc.md", "one - two\n", "reading the staged diff"),
    "gate3-em-dash-test": ("-q", "doc.md", "one - two\n", "'grep' exited 2"),
    "gate4-one-pattern": (
        "/mnt/[a-z]/[U]sers/[A-Za-z0-9_.-]+",
        "candidate.txt",
        "see docs/readme\n",
        "'grep' exited 2",
    ),
    "gate4-home-path-capture": ("-oE", "candidate.txt", "see docs/readme\n", "for home paths"),
    "gate4-allowlist-check": (
        "-qvE",
        "candidate.txt",
        "see /" + "home/alice/notes\n",
        "'grep' exited 2",
    ),
}


@pytest.mark.parametrize("site", sorted(_TARGETED))
def test_a_grep_failing_deeper_in_the_hook_still_blocks(tmp_path: Path, site: str) -> None:
    target, rel, content, message = _TARGETED[site]
    real_grep = subprocess.run(
        ["bash", "-c", "command -v grep"], capture_output=True, text=True, check=True
    ).stdout.strip()
    repo = _repo(tmp_path, rel, content)
    env = _isolated_env(tmp_path)
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "grep"
    shim.write_bytes(
        (
            "#!/bin/sh\n"
            f"for a in \"$@\"; do [ \"$a\" = '{target}' ] && {{ echo 'grep: simulated failure' >&2; exit 2; }}; done\n"
            f'exec "{real_grep}" "$@"\n'
        ).encode("ascii")
    )
    shim.chmod(0o755)
    env["PATH"] = str(shim_dir) + os.pathsep + env["PATH"]
    result = _run(repo, env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "[BLOCKED]" in result.stderr
    assert message in result.stderr


def test_a_git_that_cannot_produce_the_staged_diff_blocks(tmp_path: Path) -> None:
    """Gate 3's own diff call, reached because the staged-file list still works.

    The shim fails only a ``diff --cached`` WITHOUT ``--name-only``: that is Gate 3's
    call, and the staged-file list (which has ``--name-only``) and every
    ``git show`` pass through to the real git.
    """
    real_git = subprocess.run(
        ["bash", "-c", "command -v git"], capture_output=True, text=True, check=True
    ).stdout.strip()
    rel, content = _CLEAN["gate3-em-dash"]
    repo = _repo(tmp_path, rel, content)
    env = _isolated_env(tmp_path)
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "git"
    shim.write_bytes(
        (
            "#!/bin/sh\n"
            "cached=0; names=0\n"
            'for a in "$@"; do\n'
            '  [ "$a" = "--cached" ] && cached=1\n'
            '  [ "$a" = "--name-only" ] && names=1\n'
            "done\n"
            'if [ "$cached" = 1 ] && [ "$names" = 0 ]; then echo "fatal: simulated diff failure" >&2; exit 128; fi\n'
            f'exec "{real_git}" "$@"\n'
        ).encode("ascii")
    )
    shim.chmod(0o755)
    env["PATH"] = str(shim_dir) + os.pathsep + env["PATH"]
    result = _run(repo, env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "git could not produce the staged diff" in result.stderr


def _colour(tmp_path: Path, env: dict[str, str]) -> None:
    config = tmp_path / "colour.gitconfig"
    config.write_text("[color]\n\tui = always\n", encoding="ascii")
    env["GIT_CONFIG_GLOBAL"] = str(config)


def _external_diff(tmp_path: Path, env: dict[str, str]) -> None:
    # `true` stands for any external differ: it runs, prints nothing git can
    # parse as a unified diff, and exits 0.
    env["GIT_EXTERNAL_DIFF"] = "true"


def _textconv(tmp_path: Path, env: dict[str, str]) -> None:
    # A WORKING driver: git appends the file name, and od rewrites every byte as
    # octal text, so the em-dash never appears on a '+' line of the diff.
    config = tmp_path / "textconv.gitconfig"
    config.write_text('[diff "hide"]\n\ttextconv = od -c\n', encoding="ascii")
    env["GIT_CONFIG_GLOBAL"] = str(config)


_DIFF_CONFIG = {"color-ui-always": _colour, "external-diff": _external_diff, "textconv": _textconv}


@pytest.mark.parametrize("config", sorted(_DIFF_CONFIG))
def test_gate3_reads_the_index_whatever_the_diff_configuration(tmp_path: Path, config: str) -> None:
    rel, content = _PLANTS["gate3-em-dash"]
    attributes = "*.md diff=hide\n" if config == "textconv" else None
    repo = _repo(tmp_path, rel, content, attributes)
    env = _isolated_env(tmp_path)
    _DIFF_CONFIG[config](tmp_path, env)
    result = _run(repo, env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Em-dash (U+2014) detected" in result.stderr


@pytest.mark.parametrize("config", sorted(_DIFF_CONFIG))
def test_gate3_still_clears_a_clean_file_under_that_configuration(
    tmp_path: Path, config: str
) -> None:
    rel, content = _CLEAN["gate3-em-dash"]
    attributes = "*.md diff=hide\n" if config == "textconv" else None
    repo = _repo(tmp_path, rel, content, attributes)
    env = _isolated_env(tmp_path)
    _DIFF_CONFIG[config](tmp_path, env)
    result = _run(repo, env)
    assert result.returncode == 0, result.stdout + result.stderr
