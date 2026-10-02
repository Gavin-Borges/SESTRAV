"""pre-push: every check that runs python verifies the interpreter itself.

Why this test exists
--------------------
Bare `python` in scripts/hooks/pre-push is whatever PATH resolves. In Git Bash
on a Windows workstation, where conda is not on PATH, that can be an
interpreter outside the project environment, with no pytest. The hook's
ensure_python function activates the environment where it can and blocks the
push unless `python` imports pytest.

That verification used to happen once, inside Check 2, so a later check was
covered only while Check 2 ran before it. Two properties are pinned here:

- Structure: every `# ---- Check` section that runs bare `python` calls
  ensure_python, on its own line, before its first python command, and the
  function is defined above the first check.
- Behaviour: with a check moved ahead of Check 2, or Check 2 removed, the hook
  blocks with the interpreter diagnosis before any check runs python. A
  `python` shim placed first on PATH records every call, so the test sees the
  ORDER in which the hook used the interpreter, not only the exit status.

Every variant is built in tmp_path from the hook's text; the tracked hook is
never edited. CONDA_DEFAULT_ENV is set so that the hook's conda activation is
skipped and the shim stays the interpreter on any machine.

Anti-vacuity
------------
`test_the_intact_hook_runs_every_python_command_after_one_probe` runs the
unmodified hook with a shim that reports pytest as importable. It asserts that
the push passes, that the probe ran exactly once, and that every python
command line the structure test found was executed, so the guard cannot pass
by blocking everything or by probing only once and skipping the rest.
`test_removing_check_2_leaves_the_later_checks_runnable` does the same for the
hook with Check 2 removed.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_SOURCE = REPO_ROOT / "scripts" / "hooks" / "pre-push"

BANNER = re.compile(r"^# ---- Check (\S+?):")
PUSH_ALLOWED_LINE = "printf '[pre-push] All checks passed -- push allowed.\\n'"
QUOTED = re.compile(r"\"(?:\\.|[^\"\\])*\"|'[^']*'")
PYTHON_WORD = re.compile(r"(?<![\w./-])python3?(?![\w.-])")
HEREDOC = re.compile(r"(?<!<)<<-?\s*(['\"]?)(\w+)\1")
GUARD = "ensure_python"

DIAGNOSIS = "pytest is not importable"
PROBE = "-c import pytest"

# The shim logs its arguments, one call per line, then answers the import
# probe with PREPUSH_SHIM_PROBE_RC and every other call with 0.
SHIM = (
    "#!/bin/sh\n"
    'printf \'%s\\n\' "$*" >> "$PREPUSH_SHIM_LOG"\n'
    'if [ "$1" = "-c" ]; then\n'
    '    exit "$PREPUSH_SHIM_PROBE_RC"\n'
    "fi\n"
    "exit 0\n"
)

# git reads the developer's GLOBAL and SYSTEM config even for a throwaway
# directory. Pinned so the result depends on the hook. The repository-discovery
# variables are dropped so every git call resolves to the fixture repository.
BASE_ENV = {
    key: value
    for key, value in os.environ.items()
    if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX", "REPO_ROOT"}
}
BASE_ENV.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull})

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="pre-push is a bash script; without bash it cannot run at all",
)


def _code_lines(lines: list[str]) -> list[tuple[int, str]]:
    """Shell code lines with their index: comments and heredoc bodies dropped."""
    code: list[tuple[int, str]] = []
    terminator: str | None = None
    for index, line in enumerate(lines):
        if terminator is not None:
            if line.strip() == terminator:
                terminator = None
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        code.append((index, line))
        match = HEREDOC.search(line)
        if match:
            terminator = match.group(2)
    return code


def _runs_python(line: str) -> bool:
    """True when a code line invokes bare python outside any quoted string."""
    return PYTHON_WORD.search(QUOTED.sub(" ", line)) is not None


def _split(text: str) -> tuple[list[str], list[tuple[str, list[str]]], list[str]]:
    """Preamble, the banner-delimited checks in file order, and the final pass lines."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if BANNER.match(line)]
    assert starts, "no '# ---- Check' banner found"
    finals = [i for i, line in enumerate(lines) if line.strip() == PUSH_ALLOWED_LINE]
    assert len(finals) == 1 and finals[0] > starts[-1], "final pass line not found once"
    bounds = [*starts, finals[0]]
    sections = []
    for begin, end in zip(bounds, bounds[1:]):
        match = BANNER.match(lines[begin])
        assert match is not None
        sections.append((match.group(1), lines[begin:end]))
    return lines[: starts[0]], sections, lines[finals[0] :]


def _python_lines(body: list[str]) -> list[int]:
    return [index for index, line in _code_lines(body) if _runs_python(line)]


def _command_count(body: list[str]) -> int:
    """Python commands in a check other than an inline import probe."""
    return sum(1 for index in _python_lines(body) if "import pytest" not in body[index])


HOOK_TEXT = HOOK_SOURCE.read_text(encoding="utf-8")
PREAMBLE, SECTIONS, TAIL = _split(HOOK_TEXT)
NAMES = [name for name, _ in SECTIONS]
PYTHON_SECTIONS = [name for name, body in SECTIONS if _python_lines(body)]
PYTHON_COMMANDS = sum(_command_count(body) for _, body in SECTIONS)
# Check 1 reads the pushed refs from stdin, so every variant keeps it first.
MOVABLE = [name for name in PYTHON_SECTIONS if name not in (NAMES[0], PYTHON_SECTIONS[0])]


def _variant(order: list[str]) -> str:
    bodies = dict(SECTIONS)
    lines = [*PREAMBLE, *(line for name in order for line in bodies[name]), *TAIL]
    return "\n".join(lines) + "\n"


def _moved_first(name: str) -> str:
    return _variant([NAMES[0], name, *(n for n in NAMES[1:] if n != name)])


def _fixture(tmp_path: Path, hook_text: str) -> Path:
    """A git repository holding the hook, a harness stub and the python shim.

    The harness stub and the _local/ directory make Checks 4 and 5 take their
    python branch, so every python command in the hook is reachable.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "hook").write_text(hook_text, encoding="utf-8", newline="\n")
    harness = root / "_local" / "integrity" / "integrity_check.py"
    harness.parent.mkdir(parents=True)
    harness.write_text("", encoding="utf-8")
    shim_dir = root / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "python"
    shim.write_text(SHIM, encoding="utf-8", newline="\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, env=BASE_ENV, timeout=60)
    return root


def _run(
    root: Path, *, probe_rc: int, extra_env: dict[str, str] | None = None
) -> tuple[subprocess.CompletedProcess, list[str]]:
    """Run the hook for a branch push; return the result and the shim's call log."""
    log = root / "shim.log"
    env = dict(BASE_ENV)
    env.update(
        {
            "PREPUSH_SHIM_LOG": log.as_posix(),
            "PREPUSH_SHIM_PROBE_RC": str(probe_rc),
            "CONDA_DEFAULT_ENV": "sestrav",
        }
    )
    env.update(extra_env or {})
    env["PATH"] = str(root / "shim") + os.pathsep + env.get("PATH", "")
    stdin = f"refs/heads/feature {'1' * 40} refs/heads/feature {'0' * 40}\n"
    result = subprocess.run(
        ["bash", "hook", "origin", "https://example.invalid/fixture.git"],
        cwd=root,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=120,
    )
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result, calls


def test_the_parser_finds_the_checks_that_run_python() -> None:
    """Premise: the section split and the python detector see the real hook."""
    assert NAMES[0] == "1", NAMES
    assert PYTHON_SECTIONS[0] == "2", PYTHON_SECTIONS
    assert MOVABLE, PYTHON_SECTIONS
    pytest_lines = [
        line
        for _, body in SECTIONS
        for _index, line in _code_lines(body)
        if _runs_python(line) and "-m pytest" in line
    ]
    assert len(pytest_lines) == 1, pytest_lines


def test_every_check_that_runs_python_calls_ensure_python_first() -> None:
    assert any(line.startswith(f"{GUARD}() {{") for line in PREAMBLE), (
        f"{GUARD} must be defined above the first check"
    )
    for name, body in SECTIONS:
        commands = _python_lines(body)
        if not commands:
            continue
        guards = [index for index, line in _code_lines(body) if line.strip() == GUARD]
        assert guards and guards[0] < commands[0], (
            f"Check {name} runs python before calling {GUARD}: {body[commands[0]].strip()}"
        )


@pytest.mark.parametrize("moved", MOVABLE)
def test_a_check_moved_ahead_of_check_2_probes_before_running_python(
    tmp_path: Path, moved: str
) -> None:
    result, calls = _run(_fixture(tmp_path, _moved_first(moved)), probe_rc=1)
    assert result.returncode != 0
    assert DIAGNOSIS in result.stderr, result.stderr
    assert calls == [PROBE], calls


@pytest.mark.parametrize("moved", MOVABLE)
def test_an_exported_repo_root_does_not_let_a_moved_check_skip_the_probe(
    tmp_path: Path, moved: str
) -> None:
    """A caller's shell may export REPO_ROOT, so the hook must not rely on it being unset."""
    root = _fixture(tmp_path, _moved_first(moved))
    result, calls = _run(root, probe_rc=1, extra_env={"REPO_ROOT": root.as_posix()})
    assert result.returncode != 0
    assert calls == [PROBE], calls


def test_removing_check_2_blocks_before_any_later_check_runs_python(tmp_path: Path) -> None:
    order = [name for name in NAMES if name != PYTHON_SECTIONS[0]]
    result, calls = _run(_fixture(tmp_path, _variant(order)), probe_rc=1)
    assert result.returncode != 0
    assert DIAGNOSIS in result.stderr, result.stderr
    assert calls == [PROBE], calls


def test_removing_check_2_leaves_the_later_checks_runnable(tmp_path: Path) -> None:
    first = PYTHON_SECTIONS[0]
    order = [name for name in NAMES if name != first]
    removed = _command_count(dict(SECTIONS)[first])
    result, calls = _run(_fixture(tmp_path, _variant(order)), probe_rc=0)
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls[0] == PROBE, calls
    assert calls.count(PROBE) == 1, calls
    assert len(calls) - 1 == PYTHON_COMMANDS - removed, calls


def test_the_intact_hook_runs_every_python_command_after_one_probe(tmp_path: Path) -> None:
    result, calls = _run(_fixture(tmp_path, HOOK_TEXT), probe_rc=0)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All checks passed -- push allowed." in result.stdout
    assert calls[0] == PROBE, calls
    assert calls.count(PROBE) == 1, calls
    assert len(calls) - 1 == PYTHON_COMMANDS, calls
