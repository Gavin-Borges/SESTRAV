"""CI's lock installs never build a package against a setuptools downloaded unhashed.

environments/requirements-ci.txt and environments/requirements.lock pin
connection-pool 0.0.3 (via snakemake) as an sdist only, so every CI job that
installs them builds it. pip's default is to build in a throwaway environment
that downloads the backend, here `setuptools>=40.8.0`, from PyPI with no hash.
iedb_benchmark.yml's editable install of the package itself was the same.

Measured when this was written: a census of every pin's hashed files against
PyPI, across all 16 hash-pinned lockfiles, found connection-pool the only pin
whose locked hashes include no wheel installable on Linux x86_64 for CPython
3.11, 3.12 or 3.13. So the locks below are the ones CI builds from; the tool
locks install from wheels. This test cannot see wheel availability offline,
so an sdist that enters a TOOL lock later is not caught here. release.yml's
two installs that resolve from PyPI on purpose, to behave like a user's
(`pip install dist/*.whl` and `pip install sestrav==...`), are outside it.

The rule, per job, in step order:
- every install of one of BUILD_LOCKS, and every install of the local package
  (`.`, `./`, `.[extra]`, editable or not), carries --no-build-isolation, and
  no `pip wheel` / `pip download` of those locks builds them instead;
- and comes after `pip install --require-hashes --no-deps -c <lock> setuptools`
  from a lock that pins setuptools. That install has to come first: pip builds
  every sdist before it installs anything, and on a runner whose Python ships
  its own setuptools (3.11's is 79.0.1) a failed or missing backend install
  would leave the build using that one instead;
- and that backend install comes after the hash-pinned pip bootstrap
  (environments/requirements-pip-bootstrap.txt). Taking hashes from a
  constraints file needs a recent pip: measured, pip 24.0 and 25.0.1 refuse
  it and 26.2.1 accepts and enforces it. The runners ship 26.2.1 today, but
  nothing pins that; the bootstrap does.

- and, in a job whose actions/setup-python step restores pip's cache, comes
  after that restore and a `pip cache remove` of each wheel pip builds locally
  from those locks (SDIST_WHEELS). setup-python falls back to the newest older
  cache for the same Python under a prefix key, and pip reuses a cached wheel
  under --require-hashes, so without the removal a wheel an earlier run built
  in isolation is installed as it is. A cache restored any other way
  (actions/cache, a composite action) is not detected.

The bootstrap, backend and cache-removal commands only count when they cannot
fail silently: not `--dry-run`, not in a `continue-on-error` step, a step with
an `if:`, or one with a custom `shell:` (which may drop `-e`), and not on the
left of `||` or after `set +e`. Commands are read as text: which interpreter an
install targets is not followed.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = PROJECT_ROOT / ".github" / "workflows"

BUILD_LOCKS = frozenset(
    {"requirements.txt", "environments/requirements-ci.txt", "environments/requirements.lock"}
)
# Locks that pin setuptools, and so can serve as the backend's constraints file.
BACKEND_SOURCES = frozenset({"requirements.txt", "environments/requirements.lock"})
PIP_BOOTSTRAP = "environments/requirements-pip-bootstrap.txt"
# The wheels pip builds locally from those locks: the census's one sdist-only pin,
# named as `pip cache remove` takes it. Measured on pip 26.2.1, removing it drops
# only that locally built wheel; `pip cache remove '*'` would also empty the
# download cache.
SDIST_WHEELS = ("connection_pool",)

_PIP = r"(?<!\buv\s)\bpip3?(?:\.\d+)?(?:\s+-\S+(?:\s+[^\s-]\S*)?)*?\s+"
_PIP_INSTALL = re.compile(_PIP + r"install\b")
_PIP_BUILDS_WITHOUT_INSTALLING = re.compile(_PIP + r"(?:wheel|download)\b")


def _command_parts(run: str) -> list[tuple[str, bool]]:
    """(command, failure_tolerated) per shell command; continuations joined, comments dropped."""
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
                parts.append((command, tolerated_from_here or position < len(alternatives) - 1))
    return parts


def _args(command: str, pattern: re.Pattern[str]) -> list[str] | None:
    match = pattern.search(command)
    if not match:
        return None
    rest = command[match.end() :]
    try:
        return shlex.split(rest)
    except ValueError:
        return rest.split()


def _normalize(path: str) -> str:
    return path[2:] if path.startswith("./") else path


def _option_values(args: list[str], short: str, long: str) -> list[str]:
    values = []
    for i, arg in enumerate(args):
        if arg in (short, long) and i + 1 < len(args):
            values.append(args[i + 1])
        elif arg.startswith(long + "="):
            values.append(arg.split("=", 1)[1])
        elif arg.startswith(short) and not arg.startswith("--") and len(arg) > len(short):
            values.append(arg[len(short) :])
    return [_normalize(v) for v in values]


def _positionals(args: list[str]) -> list[str]:
    takes_value = {
        "-r",
        "--requirement",
        "-c",
        "--constraint",
        "-t",
        "--target",
        "-i",
        "--index-url",
    }
    out, skip = [], False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg in takes_value:
            skip = True
            continue
        if not arg.startswith("-"):
            out.append(arg)
    return out


def is_bootstrap_install(args: list[str]) -> bool:
    return (
        _option_values(args, "-r", "--requirement") == [PIP_BOOTSTRAP]
        and "--require-hashes" in args
        and "--dry-run" not in args
    )


def is_backend_install(args: list[str]) -> bool:
    constraints = _option_values(args, "-c", "--constraint")
    return (
        "--require-hashes" in args
        and "--no-deps" in args
        and "--dry-run" not in args
        and len(constraints) == 1
        and constraints[0] in BACKEND_SOURCES
        and _positionals(args) == ["setuptools"]
    )


def builds_from_source(args: list[str]) -> str | None:
    """What this install may build, if it is one this rule governs."""
    locks = [r for r in _option_values(args, "-r", "--requirement") if r in BUILD_LOCKS]
    if locks:
        return ", ".join(locks)
    local = [p for p in _positionals(args) if re.fullmatch(r"\.(?:/)?(?:\[[^\]]*\])?", p)]
    editable = _option_values(args, "-e", "--editable")
    if local or any(re.fullmatch(r"\.(?:/)?(?:\[[^\]]*\])?", e) or e == "" for e in editable):
        return "the local package"
    return None


def _restores_pip_cache(step: dict) -> bool:
    return (
        str(step.get("uses", "")).startswith("actions/setup-python@")
        and str((step.get("with") or {}).get("cache", "")).strip("'\"") == "pip"
    )


def job_problems(steps: list[dict]) -> tuple[list[str], int]:
    """Problems in one job's steps, and how many governed installs were checked."""
    problems, checked, bootstrapped, backend = [], 0, False, False
    # setup-python restores the newest older pip cache for the same Python when the
    # exact key misses (its restore key is a prefix), and pip reuses a cached wheel
    # under --require-hashes, so a job that restores the cache must drop the
    # locally built SDIST_WHEELS after the restore and before installing, or it may
    # install one an earlier run built in isolation instead of building it here.
    uncleared: set[str] = set()
    for step in steps:
        if _restores_pip_cache(step):
            uncleared = set(SDIST_WHEELS)
        run = step.get("run")
        if not isinstance(run, str):
            continue
        step_tolerant = bool(step.get("continue-on-error")) or "if" in step or "shell" in step
        for command, tolerated in _command_parts(run):
            silent_failure = step_tolerant or tolerated
            removal = re.fullmatch(r"(?:\S*/)?pip3?(?:\.\d+)?\s+cache\s+remove\s+(\S+)", command)
            if removal:
                if not silent_failure:
                    uncleared.discard(removal.group(1).strip("'\""))
                continue
            wheel_args = _args(command, _PIP_BUILDS_WITHOUT_INSTALLING)
            if wheel_args is not None:
                locks = [
                    r for r in _option_values(wheel_args, "-r", "--requirement") if r in BUILD_LOCKS
                ]
                if locks:
                    problems.append(f"`{command}` builds {', '.join(locks)} outside an install")
                continue
            args = _args(command, _PIP_INSTALL)
            if args is None:
                continue
            if is_bootstrap_install(args):
                bootstrapped = bootstrapped or not silent_failure
                continue
            if is_backend_install(args):
                if silent_failure:
                    problems.append(
                        f"`{command}` can fail silently (continue-on-error, if:, shell:, || or set +e)"
                    )
                elif not bootstrapped:
                    problems.append(f"`{command}` runs before the hash-pinned pip bootstrap")
                else:
                    backend = True
                continue
            target = builds_from_source(args)
            if target is None:
                continue
            checked += 1
            if "--no-build-isolation" not in args:
                problems.append(f"`{command}` builds {target} in an isolated environment")
            if not backend:
                problems.append(f"`{command}` runs before a hash-checked setuptools install")
            if uncleared:
                problems.append(
                    f"`{command}` runs in a job that restores pip's cache without first running "
                    f"`pip cache remove` for {sorted(uncleared)}"
                )
    return problems, checked


def _jobs() -> list[tuple[str, list[dict]]]:
    jobs = []
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name, job in (data.get("jobs") or {}).items():
            jobs.append((f"{path.name}:{name}", job.get("steps") or []))
    return jobs


def test_every_ci_build_uses_a_hash_checked_setuptools() -> None:
    problems, checked = [], 0
    for where, steps in _jobs():
        found, count = job_problems(steps)
        problems += [f"{where}: {p}" for p in found]
        checked += count
    assert not problems, problems
    # 4 jobs x 2 installs (ci compat, ci test, fuzz, verify-benchmark = 8),
    # 2 security.yml lock installs, iedb's lock and editable installs.
    assert checked >= 12, (
        f"only {checked} governed installs found; this test has gone partly vacuous"
    )


def test_every_build_lock_is_installed_by_some_ci_job() -> None:
    installed = set()
    for _, steps in _jobs():
        for step in steps:
            if isinstance(step.get("run"), str):
                for command, _ in _command_parts(step["run"]):
                    args = _args(command, _PIP_INSTALL)
                    if args is not None:
                        installed |= set(_option_values(args, "-r", "--requirement"))
    assert BUILD_LOCKS <= installed, sorted(BUILD_LOCKS - installed)


def test_backend_sources_pin_setuptools_with_hashes() -> None:
    for lock in BACKEND_SOURCES:
        text = (PROJECT_ROOT / lock).read_text(encoding="utf-8")
        block = text.split("\nsetuptools==", 1)
        assert len(block) == 2, f"{lock} does not pin setuptools"
        entry = block[1].split("\n\n", 1)[0].split("\n# ", 1)[0]
        assert "--hash=sha256:" in entry, f"{lock}'s setuptools pin carries no hash"


BOOT = f"pip install --no-deps --require-hashes -r {PIP_BOOTSTRAP}"
BARE_BACKEND = "pip install --no-deps --require-hashes -c requirements.txt setuptools"
BACKEND = f"{BOOT}\n{BARE_BACKEND}"
LOCK_INSTALL = "pip install --no-deps --require-hashes --no-build-isolation -r requirements.txt"


@pytest.mark.parametrize(
    ("steps", "problem_count"),
    [
        ([{"run": f"{BACKEND}\n{LOCK_INSTALL}"}], 0),
        (
            [{"run": "pip install --no-deps --require-hashes -r environments/requirements-ci.txt"}],
            2,
        ),
        (
            [
                {
                    "run": f"{BACKEND}\npip install --no-deps --require-hashes -r environments/requirements-ci.txt"
                }
            ],
            1,
        ),
        ([{"run": LOCK_INSTALL}], 1),
        ([{"run": f"{LOCK_INSTALL}\n{BACKEND}"}], 1),
        ([{"run": BACKEND, "continue-on-error": True}, {"run": LOCK_INSTALL}], 2),
        (
            [
                {
                    "run": "pip install --require-hashes --no-deps -c environments/requirements-ci.txt setuptools\n"
                    "pip install --require-hashes --no-build-isolation -r environments/requirements-ci.txt"
                }
            ],
            1,
        ),
        ([{"run": f"{BACKEND}\npip install --no-deps -e ."}], 1),
        ([{"run": f"{BACKEND}\npip install --no-deps --no-build-isolation -e ."}], 0),
        ([{"run": "pip install --require-hashes -r environments/requirements-ci-ruff.txt"}], 0),
        ([{"run": "pip install --quiet dist/*.whl"}], 0),
        (
            [
                {
                    "run": f"{BACKEND} && pip --no-input install --no-deps --require-hashes -r requirements.txt"
                }
            ],
            1,
        ),
        ([{"run": f"{BARE_BACKEND}\n{LOCK_INSTALL}"}], 2),
        ([{"run": BOOT, "continue-on-error": True}, {"run": f"{BARE_BACKEND}\n{LOCK_INSTALL}"}], 2),
        ([{"run": BOOT}, {"run": BARE_BACKEND}, {"run": LOCK_INSTALL}], 0),
        # shapes the first version of this test accepted
        (
            [
                {
                    "run": f"{BACKEND}\npip install --no-deps --require-hashes -r requirements.txt  # --no-build-isolation"
                }
            ],
            1,
        ),
        ([{"run": f"{BACKEND}\npip install --no-deps --require-hashes -r ./requirements.txt"}], 1),
        ([{"run": f"{BACKEND}\npip install --no-deps --require-hashes -rrequirements.txt"}], 1),
        ([{"run": f"{BACKEND}\npip install --no-deps '.[gnn]'"}], 1),
        ([{"run": f"{BACKEND}\npip install --no-deps -e ./"}], 1),
        (
            [{"run": f"{BACKEND}\npip3.11 install --no-deps --require-hashes -r requirements.txt"}],
            1,
        ),
        (
            [
                {
                    "run": f"{BACKEND}\npip wheel --no-deps --require-hashes -r environments/requirements-ci.txt"
                }
            ],
            1,
        ),
        ([{"run": f"{BOOT}\n{BARE_BACKEND} || true\n{LOCK_INSTALL}"}], 2),
        ([{"run": f"set +e\n{BACKEND}\n{LOCK_INSTALL}"}], 2),
        ([{"run": BACKEND, "shell": "bash {0}"}, {"run": LOCK_INSTALL}], 2),
        ([{"run": BACKEND, "if": "false"}, {"run": LOCK_INSTALL}], 2),
        ([{"run": f"{BOOT}\n{BARE_BACKEND} --dry-run\n{LOCK_INSTALL}"}], 1),
        # spellings that must still be accepted
        (
            [
                {
                    "run": f"pip install --no-deps --require-hashes -r ./{PIP_BOOTSTRAP}\n"
                    "pip install --no-deps --require-hashes --constraint=requirements.txt setuptools\n"
                    "pip install --no-deps --require-hashes --no-build-isolation --requirement=./requirements.txt"
                }
            ],
            0,
        ),
        # a job that restores pip's cache must drop the locally built wheel first
        (
            [
                {"uses": "actions/setup-python@x", "with": {"cache": "pip"}},
                {"run": f"{BACKEND}\n{LOCK_INSTALL}"},
            ],
            1,
        ),
        (
            [
                {"uses": "actions/setup-python@x", "with": {"cache": "pip"}},
                {"run": f"{BACKEND}\npip cache remove connection_pool\n{LOCK_INSTALL}"},
            ],
            0,
        ),
        (
            [
                {"uses": "actions/setup-python@x", "with": {"cache": "pip"}},
                {"run": f"{BACKEND}\npip cache remove connection_pool || true\n{LOCK_INSTALL}"},
            ],
            1,
        ),
        ([{"uses": "actions/setup-python@x"}, {"run": f"{BACKEND}\n{LOCK_INSTALL}"}], 0),
        (
            [
                {"run": "pip cache remove connection_pool"},
                {"uses": "actions/setup-python@x", "with": {"cache": "pip"}},
                {"run": f"{BACKEND}\n{LOCK_INSTALL}"},
            ],
            1,
        ),
    ],
)
def test_job_problems_classifies_each_shape(steps: list[dict], problem_count: int) -> None:
    problems, _ = job_problems(steps)
    assert len(problems) == problem_count, problems
