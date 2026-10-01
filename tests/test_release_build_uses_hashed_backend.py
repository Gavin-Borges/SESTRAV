"""The release build uses a hash-pinned build backend, never an isolated download.

release.yml's "Build sdist and wheel" step installed only the `build` frontend
from a hashed lock and then ran `python -m build`, whose default is to create a
throwaway environment and pip-install pyproject.toml's [build-system].requires
into it from PyPI with no hashes, once for the sdist and once for the wheel. The
published artifacts were therefore built by whatever setuptools and wheel PyPI
served on release day. The step now installs the backend from
environments/requirements-ci-build.txt and builds with --no-isolation, which
uses that environment and still refuses to run when a [build-system]
requirement is missing from it or below its specifier (build's dependency
check, which -x / --skip-dependency-check would switch off).

These tests pin that down:
- every distribution build in .github/workflows is `-m build` / `pyproject-build`
  run with --no-isolation and without the dependency-check skip, after a real
  --require-hashes install of the build lock earlier in the same step; any
  other builder (`uv build`, `pipx run build`, `pip wheel`) is refused;
- the build lock pins every [build-system].requires package within its
  specifier, with hashes, and its setuptools is requirements.in's pin;
- the build lock's spec keeps its unsafe packages, without which a recompile
  drops setuptools and wheel from it.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.version import Version

from tools.update_dependencies import LOCK_SPECS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = PROJECT_ROOT / ".github" / "workflows"
BUILD_LOCK = "environments/requirements-ci-build.txt"

# The frontend this repo allows, in any interpreter spelling: `python -m build`,
# `python3.13 -m build`, `"$PY" -I -m build`, `py -mbuild`, `pyproject-build`.
_FRONTEND = re.compile(r"(?:(?:^|\s)-m\s*build\b|\bpyproject-build\b)")
# Other ways to build a distribution, none of which this repo uses or allows.
# Matched at the start of a command (or after `-m`), so a package list such as
# `--ignore-packages pip wheel` is not mistaken for a build.
_OTHER_BUILDERS = re.compile(
    r"(?:^(?:\S*/)?(?:uv|hatch|poetry|flit)\s+build\b|^(?:\S*/)?pipx\s+run\s+build\b"
    r"|^(?:\S*/)?(?:env\s+(?:\S+\s+)*?)?pip3?(?:\.\d+)?\s+wheel\b"
    r"|(?:^|\s)-m\s+pip\s+wheel\b|(?:^|\s)\S*setup\.py\s+(?:sdist|bdist\w*)\b)"
)
_PIP_INSTALL = re.compile(r"(?<!\buv\s)\bpip3?(?:\.\d+)?(?:\s+-\S+(?:\s+[^\s-]\S*)?)*?\s+install\b")
_ENTRY = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==([^\s\\;]+)")
_NOT_AN_INSTALL = {"--dry-run", "--target", "-t", "--prefix", "--root"}


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _run_blocks() -> list[tuple[str, str]]:
    blocks = []
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for job_name, job in (data.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                if isinstance(step.get("run"), str):
                    blocks.append((f"{path.name}:{job_name}:{step.get('name', '?')}", step["run"]))
    return blocks


def _command_parts(run: str) -> list[tuple[str, bool]]:
    """(command, failure_tolerated) for each shell command in a run block.

    Continuations are joined and comments dropped. A command whose failure the
    script tolerates - the left side of `||`, or anything after `set +e` - is
    marked, because such an install may not have happened.
    """
    joined = re.sub(r"\\\s*\n", " ", run)
    parts, tolerated_from_here = [], False
    for line in joined.splitlines():
        line = re.sub(r"(?:^|\s)#.*$", "", line)
        for piece in re.split(r"&&|;", line):
            alternatives = [a.strip() for a in piece.split("||")]
            for position, command in enumerate(alternatives):
                if not command:
                    continue
                if re.fullmatch(r"set\s+\+e\b.*", command):
                    tolerated_from_here = True
                    continue
                tolerated = tolerated_from_here or position < len(alternatives) - 1
                parts.append((command, tolerated))
    return parts


def _commands(run: str) -> list[str]:
    return [command for command, _ in _command_parts(run)]


def _frontend_flags(command: str) -> set[str]:
    """Long flags (argparse prefix abbreviations resolved), short flags expanded (`-nx`)."""
    flags = set()
    tokens = command[_FRONTEND.search(command).end() :].split()
    for token in tokens:
        if token.startswith("--"):
            name = token.split("=", 1)[0]
            for full in ("--skip-dependency-check", "--no-isolation"):
                if len(name) >= 4 and full.startswith(name):
                    name = full
            flags.add(name)
        elif token.startswith("-") and len(token) > 1:
            for char in token[1:]:
                flags.add(f"-{char}")
                if char in "oC":  # these take a value, possibly glued on
                    break
    return flags


def _is_build_lock_install(command: str) -> bool:
    match = _PIP_INSTALL.search(command)
    if not match:
        return False
    args = command[match.end() :].split()
    files = [args[i + 1] for i, a in enumerate(args[:-1]) if a in ("-r", "--requirement")]
    files += [a.split("=", 1)[1] for a in args if a.startswith("--requirement=")]
    files += [a[2:] for a in args if a.startswith("-r") and not a.startswith("--") and len(a) > 2]
    files = [f[2:] if f.startswith("./") else f for f in files]
    glued_target = any(a.startswith("-t") and not a.startswith("--") and len(a) > 2 for a in args)
    return (
        BUILD_LOCK in files
        and "--require-hashes" in args
        and not glued_target
        and not _NOT_AN_INSTALL & {a.split("=", 1)[0] for a in args}
    )


def build_problems(run: str) -> list[str]:
    """Why a run block that builds a distribution does not use the hashed backend.

    It reads the build lock install textually. It does not follow which
    interpreter a command runs under; build's own dependency check is the
    backstop for an install into the wrong environment.
    """
    parts = _command_parts(run)
    commands = [command for command, _ in parts]
    installed = [not tolerated and _is_build_lock_install(command) for command, tolerated in parts]
    problems = []
    for index, command in enumerate(commands):
        if _OTHER_BUILDERS.search(command):
            problems.append(
                f"`{command}` builds with something other than `python -m build --no-isolation`"
            )
            continue
        if not _FRONTEND.search(command):
            continue
        flags = _frontend_flags(command)
        if not {"--no-isolation", "-n"} & flags:
            problems.append(f"`{command}` builds in an isolated environment")
        if {"--skip-dependency-check", "-x"} & flags:
            problems.append(f"`{command}` skips build's check of [build-system].requires")
        if not any(installed[:index]):
            problems.append(
                f"`{command}` is not preceded by a --require-hashes install of {BUILD_LOCK}"
            )
    return problems


def _lock_pins(relative: str) -> dict[str, tuple[str, int]]:
    """canonical name -> (version, number of hashes) for every pin in a compiled lock."""
    pins: dict[str, tuple[str, int]] = {}
    current = None
    for line in (PROJECT_ROOT / relative).read_text(encoding="utf-8").splitlines():
        match = _ENTRY.match(line)
        if match:
            current = _canonical(match.group(1))
            pins[current] = (match.group(2), 0)
        elif current and line.startswith(" ") and "--hash=sha256:" in line:
            version, count = pins[current]
            pins[current] = (version, count + line.count("--hash=sha256:"))
        elif line and not line.startswith(" "):
            current = None
    return pins


def test_every_distribution_build_in_ci_uses_the_hashed_backend() -> None:
    blocks = [
        (w, r)
        for w, r in _run_blocks()
        if any(_FRONTEND.search(c) or _OTHER_BUILDERS.search(c) for c in _commands(r))
    ]
    assert blocks, "found no distribution build in .github/workflows; this test has gone vacuous"
    problems = [f"{where}: {p}" for where, run in blocks for p in build_problems(run)]
    assert not problems, problems


def test_the_build_lock_pins_every_build_system_requirement_with_hashes() -> None:
    requires = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins = _lock_pins(BUILD_LOCK)
    problems = []
    for spec in requires["build-system"]["requires"]:
        requirement = Requirement(spec)
        name = _canonical(requirement.name)
        if name not in pins:
            problems.append(f"{name} is not pinned in {BUILD_LOCK}")
            continue
        version, hashes = pins[name]
        if Version(version) not in requirement.specifier:
            problems.append(
                f"{name}=={version} is outside [build-system]'s {requirement.specifier}"
            )
        if hashes == 0:
            problems.append(f"{name}=={version} carries no hash")
    assert not problems, problems


def test_the_build_lock_setuptools_is_requirements_in_pin() -> None:
    # The release builds with the same setuptools the runtime closure pins, so a
    # bump of one without the other fails here instead of drifting silently.
    pinned = [
        line.split("==", 1)[1].split()[0]
        for line in (PROJECT_ROOT / "requirements.in").read_text(encoding="utf-8").splitlines()
        if line.startswith("setuptools==")
    ]
    assert len(pinned) == 1, pinned
    assert _lock_pins(BUILD_LOCK)["setuptools"][0] == pinned[0]


def test_the_build_lock_spec_keeps_its_unsafe_packages() -> None:
    [spec] = [s for s in LOCK_SPECS if s.output == BUILD_LOCK]
    assert spec.allow_unsafe, (
        "a recompile without allow_unsafe drops setuptools and wheel from the build lock"
    )


INSTALL = f"pip install --require-hashes -r {BUILD_LOCK}"


@pytest.mark.parametrize(
    ("run", "problem_count"),
    [
        (f"{INSTALL}\npython -m build --no-isolation --sdist --wheel", 0),
        (f"{INSTALL} && python -m build -n", 0),
        (f"{INSTALL}\npython3.13 -I -m build --sdist -n --wheel", 0),
        (f"pip install --require-hashes --requirement=./{BUILD_LOCK}\npyproject-build -nw", 0),
        (f"{INSTALL}\npython -m build --sdist --wheel --outdir dist/", 1),
        (f"{INSTALL}\npython3.13 -m build --sdist --wheel", 1),
        (f'{INSTALL}\n"$PY" -mbuild', 1),
        (f"{INSTALL}\npython -m build --sdist --wheel  # --no-isolation", 1),
        (f"{INSTALL}\npython -m build --no-isolation -x", 1),
        (f"{INSTALL}\npython -m build -nx", 1),
        (f"{INSTALL}\npython -m build --no-isolation --skip-dependency-check", 1),
        ("python -m build --no-isolation", 1),
        (f"pip install -r {BUILD_LOCK}\npython -m build --no-isolation", 1),
        (f"pip install --dry-run --require-hashes -r {BUILD_LOCK}\npython -m build -n", 1),
        (f"uv pip install --require-hashes -r {BUILD_LOCK}\npython -m build -n", 1),
        ("python -m build", 2),
        (f"{INSTALL}\nuv build", 1),
        (f"{INSTALL}\npipx run build --sdist", 1),
        (f"{INSTALL}\npip wheel --no-deps -w dist .", 1),
        ("python -m pip install build", 0),
        (
            "pip-licenses --format=json --ignore-packages pip-licenses prettytable wcwidth pip wheel",
            0,
        ),
        (f"{INSTALL}\npython -m pip wheel --no-deps -w dist .", 1),
        ("# python -m build", 0),
        (f"{INSTALL}\npython -m build -n --skip-dep", 1),
        (f"{INSTALL}\npython -m build --no-isol", 0),
        (f"{INSTALL}\npython setup.py sdist bdist_wheel", 1),
        (f"{INSTALL}\nhatch build", 1),
        (f"{INSTALL}\npoetry build", 1),
        (f"{INSTALL}\nflit build", 1),
        (f"{INSTALL}\n/usr/bin/env pip wheel .", 1),
        (f"pip install --require-hashes -t/tmp/x -r {BUILD_LOCK}\npython -m build -n", 1),
        (f"pip install --require-hashes -r{BUILD_LOCK}\npython -m build -n", 0),
        (f"{INSTALL} || true\npython -m build -n", 1),
        (f"set +e\n{INSTALL}\npython -m build -n", 1),
    ],
)
def test_build_problems_classifies_each_shape(run: str, problem_count: int) -> None:
    assert len(build_problems(run)) == problem_count, build_problems(run)
