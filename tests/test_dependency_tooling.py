"""Unit tests for the dependency-management tooling in tools/.

Covers tools/update_dependencies.py argument construction and its
uv-not-installed path (uv is deliberately not a repo dependency, so every uv
invocation here is mocked), plus the false-positive surface of
tools/check_hash_pins.py.
"""

import itertools
import pathlib
import re
import subprocess

import pytest

from tools import check_hash_pins, update_dependencies
from tools.update_dependencies import LOCK_SPECS, LockSpec, build_command

# ---------------------------------------------------------------------------
# update_dependencies - command construction
# ---------------------------------------------------------------------------

RUNTIME = next(spec for spec in LOCK_SPECS if spec.name == "runtime")
CI_MYPY = next(spec for spec in LOCK_SPECS if spec.name == "ci-mypy")


def test_lock_specs_point_at_real_files():
    root = pathlib.Path(update_dependencies.REPO_ROOT)
    for spec in LOCK_SPECS:
        assert (root / spec.source).is_file(), spec.source
        assert (root / spec.output).is_file(), spec.output


def test_lock_spec_names_are_unique():
    names = [spec.name for spec in LOCK_SPECS]
    assert len(names) == len(set(names))


def test_build_command_is_uv_pip_compile():
    command = build_command(RUNTIME)
    assert command[:3] == ["uv", "pip", "compile"]
    assert command[3] == "requirements.in"


def test_build_command_always_generates_hashes():
    for spec in LOCK_SPECS:
        assert "--generate-hashes" in build_command(spec)


def test_build_command_suppresses_index_url():
    assert "--no-emit-index-url" in build_command(RUNTIME)


def test_build_command_defaults_to_linux_platform():
    command = build_command(RUNTIME)
    assert command[command.index("--python-platform") + 1] == "linux"


def test_build_command_honours_platform_override():
    command = build_command(RUNTIME, python_platform="windows")
    assert command[command.index("--python-platform") + 1] == "windows"


def test_build_command_pins_the_interpreter_version():
    command = build_command(CI_MYPY)
    assert command[command.index("--python-version") + 1] == CI_MYPY.python_version


def test_build_command_writes_to_the_declared_output():
    command = build_command(CI_MYPY)
    assert command[command.index("--output-file") + 1] == CI_MYPY.output


def test_target_upgrade_is_single_package():
    command = build_command(RUNTIME, upgrade_package="pillow")
    assert command[command.index("--upgrade-package") + 1] == "pillow"
    assert "--upgrade" not in command


def test_full_relock_uses_bare_upgrade():
    command = build_command(RUNTIME, upgrade_all=True)
    assert "--upgrade" in command
    assert "--upgrade-package" not in command


def test_full_relock_wins_over_target():
    command = build_command(RUNTIME, upgrade_package="pillow", upgrade_all=True)
    assert "--upgrade-package" not in command


def test_no_upgrade_flag_when_neither_requested():
    command = build_command(RUNTIME)
    assert "--upgrade" not in command
    assert "--upgrade-package" not in command


def test_allow_unsafe_specs_do_not_exclude_setuptools():
    assert RUNTIME.allow_unsafe
    assert "--unsafe-package" not in build_command(RUNTIME)


def test_non_allow_unsafe_specs_exclude_the_unsafe_set():
    command = build_command(CI_MYPY)
    excluded = [command[i + 1] for i, arg in enumerate(command) if arg == "--unsafe-package"]
    assert excluded == list(update_dependencies.UNSAFE_PACKAGES)


def test_no_shell_metacharacter_joining():
    for spec in LOCK_SPECS:
        assert all(isinstance(part, str) for part in build_command(spec))


# ---------------------------------------------------------------------------
# update_dependencies - selection and CLI
# ---------------------------------------------------------------------------


def test_ci_env_choices_cover_every_tool_environment():
    # All 8 CI tool environments must be individually selectable. 4 of them
    # (ci, pip-audit, security, semgrep) are not `ci-` prefixed and were once
    # reachable only via --all. The four application lockfiles (runtime, lock,
    # and the api and demo image locks) are excluded.
    assert set(update_dependencies.ci_env_choices()) == {
        "build",
        "mypy",
        "pytest-cov",
        "ruff",
        "ci",
        "pip-audit",
        "security",
        "semgrep",
    }


@pytest.mark.parametrize("choice", update_dependencies.ci_env_choices())
def test_every_ci_env_choice_selects_exactly_one_spec(choice):
    selected = update_dependencies.select_specs(ci_env=choice)
    assert len(selected) == 1, f"{choice} selected {[s.name for s in selected]}"


def test_ci_env_cannot_select_the_application_lockfiles():
    for name in update_dependencies.RUNTIME_SPEC_NAMES:
        assert update_dependencies.select_specs(ci_env=name) == []


# The one spec permitted a uv override file, and the file it must name. Every
# other spec must compile without one.
OVERRIDDEN_SPECS = {"semgrep": "environments/semgrep-overrides.txt"}

# The exact set of packages the semgrep override may lift, and nothing else:
# mcp (CVE-2026-59951) and the five OpenTelemetry packages semgrep itself
# declares (opentelemetry-api carries CVE-2026-114603; semgrep pins the other
# four to the same release family, so they have to move with it).
SEMGREP_OVERRIDDEN_PACKAGES = {
    "mcp",
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp-proto-http",
    "opentelemetry-instrumentation-requests",
    "opentelemetry-instrumentation-threading",
}


def test_only_the_semgrep_spec_compiles_with_a_uv_override_file():
    # History: requirements.in / requirements-lock.in floor setuptools>=83.0.0
    # for GHSA-h35f-9h28-mq5c, which collided with torch 2.12.0's declared
    # `setuptools<82` build-metadata cap and made both specs unsatisfiable for
    # any resolver. Both therefore compiled with `--overrides overrides.txt`.
    # torch 2.13.0 raised the cap to `setuptools>=77.0.3`, so the override was
    # retired. This asserts the workaround does not creep back in: a
    # reintroduced override would silently mask a genuine resolution conflict.
    #
    # The semgrep spec compiled with environments/semgrep-overrides.txt to lift
    # pyjwt while semgrep declared pyjwt[crypto]~=2.13.0, until semgrep 1.179.0
    # declared pyjwt[crypto]>=2.15.0,<3 and that override was retired. It was
    # reinstated for mcp and OpenTelemetry: every semgrep release from 1.179.0
    # through 1.182.1 declares mcp==1.29.0 exactly (CVE-2026-59951, fixed in
    # 1.30.0) and opentelemetry-api~=1.37.0 (CVE-2026-114603, fixed in 1.45.1).
    # That spec is named here so that any other is not.
    root = pathlib.Path(update_dependencies.REPO_ROOT)
    for spec in LOCK_SPECS:
        command = build_command(spec)
        expected = OVERRIDDEN_SPECS.get(spec.name)
        assert spec.overrides == expected, spec.name
        if expected is None:
            assert "--overrides" not in command, spec.name
        else:
            assert command.count("--overrides") == 1, spec.name
            assert command[command.index("--overrides") + 1] == expected, spec.name
            assert (root / expected).is_file(), expected


def _requirement_lines(path: pathlib.Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _lock_pins(lock: pathlib.Path) -> dict[str, list[str]]:
    """Every `name==version` pin in a uv-compiled lock, by normalised name."""
    from packaging.requirements import Requirement

    pinned: dict[str, list[str]] = {}
    for pin_line in lock.read_text(encoding="utf-8").splitlines():
        if pin_line.startswith((" ", "#")) or "==" not in pin_line:
            continue
        name, rest = pin_line.split("==", 1)
        pinned.setdefault(Requirement(name).name.lower(), []).append(rest.split()[0])
    return pinned


def test_the_semgrep_override_lifts_exactly_its_named_set_and_states_its_exit_conditions():
    from packaging.requirements import Requirement

    path = pathlib.Path(update_dependencies.REPO_ROOT) / OVERRIDDEN_SPECS["semgrep"]
    requirements = _requirement_lines(path)
    names = [Requirement(line).name.lower() for line in requirements]
    assert len(names) == len(set(names)), f"a package is overridden twice: {names}"
    assert set(names) == SEMGREP_OVERRIDDEN_PACKAGES, names
    text = path.read_text(encoding="utf-8")
    # One exit condition per package group: the two groups retire independently.
    assert "EXIT CONDITION (mcp)" in text
    assert "EXIT CONDITION (OpenTelemetry)" in text


def test_every_overridden_lock_pins_a_version_its_override_admits():
    # A lock and its override can drift apart: a hand edit of either, or a
    # recompile through another tool, changes one without the other.
    from packaging.requirements import Requirement
    from packaging.version import Version

    root = pathlib.Path(update_dependencies.REPO_ROOT)
    overridden = [(spec, spec.overrides) for spec in LOCK_SPECS if spec.overrides]
    assert overridden, "no LockSpec sets overrides; this guard checked nothing"
    for spec, overrides in overridden:
        pinned = _lock_pins(root / spec.output)
        for line in _requirement_lines(root / overrides):
            requirement = Requirement(line)
            pins = pinned.get(requirement.name.lower(), [])
            assert len(pins) == 1, f"{spec.output}: {requirement.name} pinned {pins}"
            assert Version(pins[0]) in requirement.specifier, (
                f"{spec.output} pins {requirement.name}=={pins[0]}, "
                f"outside its override {requirement.specifier}"
            )


# The two floors below are stated independently of the override file, so each
# fails a lock compiled without the override (uv then resolves semgrep's own
# requirements without an error, which Dependabot's grouped /environments
# updates are expected to do) even if the override file itself is edited.
#
# pip-audit reports CVE-2026-59951 / GHSA-84m7-p3x7-pcfv against mcp 1.29.0;
# 1.30.0 is its first fixed 1.x release.
MCP_ADVISORY_FLOOR = "1.30.0"
# pip-audit reports CVE-2026-114603 / GHSA-xcq7-p3pf-p88h against
# opentelemetry-api 1.37.0; the advisory covers <= 1.45.0 and is fixed in 1.45.1.
OPENTELEMETRY_API_ADVISORY_FLOOR = "1.45.1"


def _single_semgrep_lock_pin(package: str) -> str:
    semgrep = next(spec for spec in LOCK_SPECS if spec.name == "semgrep")
    pins = _lock_pins(pathlib.Path(update_dependencies.REPO_ROOT) / semgrep.output)
    found = pins.get(package, [])
    assert len(found) == 1, f"{semgrep.output}: {package} pinned {found}"
    return found[0]


def test_the_semgrep_lock_pins_mcp_at_or_above_its_advisory_fix():
    from packaging.version import Version

    pin = _single_semgrep_lock_pin("mcp")
    assert Version(pin) >= Version(MCP_ADVISORY_FLOOR), (
        f"the semgrep lock pins mcp=={pin}, below {MCP_ADVISORY_FLOOR}"
    )


def test_the_semgrep_lock_pins_opentelemetry_api_at_or_above_its_advisory_fix():
    from packaging.version import Version

    pin = _single_semgrep_lock_pin("opentelemetry-api")
    assert Version(pin) >= Version(OPENTELEMETRY_API_ADVISORY_FLOOR), (
        f"the semgrep lock pins opentelemetry-api=={pin}, below {OPENTELEMETRY_API_ADVISORY_FLOOR}"
    )


# pip-audit reports advisories against pyjwt 2.13.0 and 2.14.0 and none against
# 2.15.0. semgrep 1.179.0 itself requires pyjwt[crypto]>=2.15.0,<3, so this holds
# without help today; it fails a lock that walks pyjwt back by any route, a hand
# edit or an older semgrep among them.
PYJWT_ADVISORY_FLOOR = "2.15.0"


def test_the_semgrep_lock_pins_pyjwt_at_or_above_its_advisory_floor():
    from packaging.requirements import Requirement
    from packaging.version import Version

    semgrep = next(spec for spec in LOCK_SPECS if spec.name == "semgrep")
    lock = pathlib.Path(update_dependencies.REPO_ROOT) / semgrep.output
    pins = []
    for pin_line in lock.read_text(encoding="utf-8").splitlines():
        if pin_line.startswith((" ", "#")) or "==" not in pin_line:
            continue
        name, rest = pin_line.split("==", 1)
        if Requirement(name).name.lower() == "pyjwt":
            pins.append(rest.split()[0])
    assert len(pins) == 1, f"{semgrep.output}: pyjwt pinned {pins}"
    assert Version(pins[0]) >= Version(PYJWT_ADVISORY_FLOOR), (
        f"{semgrep.output} pins pyjwt=={pins[0]}, below {PYJWT_ADVISORY_FLOOR}"
    )


_PIP_INSTALL = re.compile(r"\bpip(?:3(?:\.\d+)?)?\s+install\b")
_NO_DEPS = re.compile(r"(?:^|\s)--no-deps(?:\s|$)")
_COMMAND_SEPARATOR = re.compile(r"&&|\|\||;|\|")


def _install_commands(text: str) -> list[str]:
    """Every `pip install` command in `text`, continuations joined, comments dropped.

    Continuations are joined first so an install whose `-r` path sits on the next
    line (the shape Dockerfile.api uses) is one command; comments are dropped so a
    `--no-deps` that appears only in a comment does not count; and a line is split
    on shell separators so a `--no-deps` on one install cannot vouch for another.
    """
    commands = []
    for line in re.sub(r"\\\r?\n", " ", text).splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        code = re.split(r"\s#", stripped, maxsplit=1)[0]
        for part in _COMMAND_SEPARATOR.split(code):
            if _PIP_INSTALL.search(part):
                commands.append(part.strip())
    return commands


def _unguarded_installs(text: str, lock_name: str) -> list[str]:
    return [c for c in _install_commands(text) if lock_name in c and not _NO_DEPS.search(c)]


def _install_surfaces(root: pathlib.Path) -> list[pathlib.Path]:
    """Files that can install a lock: workflows and actions, images, make, shell."""
    paths = [p for pattern in ("*.yml", "*.yaml") for p in (root / ".github").rglob(pattern)]
    paths += [p for p in root.glob("Dockerfile*") if p.is_file()]
    paths += [p for p in (root / "Makefile",) if p.is_file()]
    for directory in ("scripts", "tools"):
        paths += list((root / directory).rglob("*.sh"))
    return sorted(set(paths))


def test_every_install_of_an_overridden_lock_skips_resolution():
    # An override makes the lock disagree with the overridden package's own
    # metadata, so a resolving install dies with ResolutionImpossible. Measured
    # while the semgrep spec overrode pyjwt:
    # `pip install --require-hashes -r environments/requirements-semgrep.txt`
    # exited 1 against the overridden lock and 0 with --no-deps added. It was
    # dormant while no spec set `overrides`, and is active again since the
    # semgrep spec's mcp and OpenTelemetry override: security.yml's two semgrep
    # installs are the installs it checks.
    overridden = [spec for spec in LOCK_SPECS if spec.overrides]
    if not overridden:
        # Report the dormancy instead of passing on an empty loop, so a run that
        # checked nothing reads as SKIPPED rather than as a green guard.
        pytest.skip("no LockSpec sets overrides; this guard applies again when one does")
    root = pathlib.Path(update_dependencies.REPO_ROOT)
    surfaces = _install_surfaces(root)
    for spec in overridden:
        lock_name = pathlib.PurePosixPath(spec.output).name
        texts = {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in surfaces}
        installs = [c for t in texts.values() for c in _install_commands(t) if lock_name in c]
        assert installs, f"nothing installs {spec.output}; this guard checked nothing"
        for where, text in texts.items():
            assert not _unguarded_installs(text, lock_name), where


_SEMGREP_LOCK = "environments/requirements-semgrep.txt"


@pytest.mark.parametrize(
    ("text", "unguarded"),
    [
        (f"pip install --require-hashes -r {_SEMGREP_LOCK}\n", 1),
        (f"RUN pip install --user \\\n    -r {_SEMGREP_LOCK}\n", 1),
        (f"pip3 install -r {_SEMGREP_LOCK}\n", 1),
        (f"pip3.11 install -r {_SEMGREP_LOCK}\n", 1),
        (f"python -m pip install -r {_SEMGREP_LOCK}  # --no-deps\n", 1),
        ("cd environments && pip install -r requirements-semgrep.txt\n", 1),
        (f"pip install --no-deps -r a.txt && pip install -r {_SEMGREP_LOCK}\n", 1),
        (f"# pip install -r {_SEMGREP_LOCK}\n", 0),
        (f"pip install --require-hashes --no-deps -r {_SEMGREP_LOCK}\n", 0),
        ("uv pip compile environments/requirements-semgrep.in\n", 0),
    ],
)
def test_the_no_deps_guard_sees_every_install_shape(text, unguarded):
    assert len(_unguarded_installs(text, "requirements-semgrep.txt")) == unguarded


def test_the_no_deps_guard_scans_every_install_surface(tmp_path):
    expected = [
        ".github/workflows/a.yml",
        ".github/workflows/b.yaml",
        ".github/actions/setup/action.yml",
        "Dockerfile",
        "Dockerfile.api",
        "Makefile",
        "scripts/nested/install.sh",
        "tools/install.sh",
    ]
    for relative in expected:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    found = {p.relative_to(tmp_path).as_posix() for p in _install_surfaces(tmp_path)}
    assert found == set(expected)


def test_the_only_override_files_are_the_permitted_ones():
    # Guards the other half of each retirement: the root overrides.txt must not
    # return, and no override file may exist beyond the ones OVERRIDDEN_SPECS
    # names. environments/semgrep-overrides.txt was retired once (pyjwt) and
    # reinstated for mcp and OpenTelemetry; when both of its exit conditions
    # are met, remove its OVERRIDDEN_SPECS entry and this test requires the
    # file to go too.
    root = pathlib.Path(update_dependencies.REPO_ROOT)
    assert not (root / "overrides.txt").exists()
    present = {
        p.relative_to(root).as_posix()
        for pattern in ("*overrides*.txt", "environments/*overrides*.txt")
        for p in root.glob(pattern)
    }
    assert present == set(OVERRIDDEN_SPECS.values())


# The release that first closed GHSA-h35f-9h28-mq5c. Any setuptools at or above
# this satisfies the advisory; anything below reopens it.
SETUPTOOLS_SECURITY_FLOOR = (83, 0, 0)


def _setuptools_version(text: str, operator: str) -> tuple[int, ...]:
    """Version attached to the single `setuptools<operator><version>` requirement line."""
    prefix = f"setuptools{operator}"
    matches = [line.strip() for line in text.splitlines() if line.strip().startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one '{prefix}' line, found {matches}"
    spec = matches[0][len(prefix) :]
    for terminator in ("#", ";", ","):
        spec = spec.split(terminator)[0]
    return tuple(int(part) for part in spec.strip().split("."))


def test_setuptools_floor_survived_the_override_retirement():
    # The floor is the security constraint (GHSA-h35f-9h28-mq5c); the override
    # was only the workaround. Retiring the workaround must not drop the floor.
    #
    # Asserted as ">= the advisory floor" rather than against a literal version:
    # the subject of this test is the advisory, not today's pin, and a hardcoded
    # "setuptools==83.0.0" made every routine setuptools bump fail a security
    # test that the bump did not actually violate.
    root = pathlib.Path(update_dependencies.REPO_ROOT)
    runtime = (root / "requirements.in").read_text(encoding="utf-8")
    lock_spec = (root / "environments" / "requirements-lock.in").read_text(encoding="utf-8")
    assert _setuptools_version(runtime, "==") >= SETUPTOOLS_SECURITY_FLOOR
    assert _setuptools_version(lock_spec, ">=") >= SETUPTOOLS_SECURITY_FLOOR


def test_select_specs_defaults_to_everything():
    assert update_dependencies.select_specs() == list(LOCK_SPECS)


def test_select_specs_narrows_to_one_ci_env():
    specs = update_dependencies.select_specs("ruff")
    assert [spec.name for spec in specs] == ["ci-ruff"]


def test_ci_env_source_naming_convention():
    # Tool environments come in two naming shapes: the `ci-` prefixed ones
    # (`--ci-env mypy` -> requirements-ci-mypy.in) and the standalone ones
    # (`--ci-env semgrep` -> requirements-semgrep.in). Both must map to a real
    # .in/.txt pair under environments/ with matching stems.
    for name in update_dependencies.ci_env_choices():
        spec = update_dependencies.select_specs(name)[0]
        assert spec.source in (
            f"environments/requirements-ci-{name}.in",
            f"environments/requirements-{name}.in",
        ), f"{name} -> {spec.source}"
        assert spec.output == spec.source.removesuffix(".in") + ".txt"


def test_dry_run_prints_commands_without_invoking_uv(capsys, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("subprocess must not run during --dry-run")

    monkeypatch.setattr(update_dependencies.subprocess, "run", explode)
    assert update_dependencies.main(["--ci-env", "ruff", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("uv pip compile environments/requirements-ci-ruff.in")


def test_requires_a_selection():
    with pytest.raises(SystemExit) as excinfo:
        update_dependencies.main([])
    assert excinfo.value.code == 2


def test_all_conflicts_with_target():
    with pytest.raises(SystemExit) as excinfo:
        update_dependencies.main(["--all", "--target", "pillow"])
    assert excinfo.value.code == 2


def test_unknown_ci_env_rejected():
    with pytest.raises(SystemExit) as excinfo:
        update_dependencies.main(["--ci-env", "nope"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# update_dependencies - uv detection
# ---------------------------------------------------------------------------


def _fake_run(returncode=0, stdout="uv 0.9.7"):
    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")

    return runner


def test_uv_version_returns_none_when_uv_absent(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("uv")

    monkeypatch.setattr(update_dependencies.subprocess, "run", missing)
    assert update_dependencies.uv_version() is None


def test_uv_version_returns_none_on_failure(monkeypatch):
    monkeypatch.setattr(update_dependencies.subprocess, "run", _fake_run(returncode=1, stdout=""))
    assert update_dependencies.uv_version() is None


def test_uv_version_reports_the_version(monkeypatch):
    monkeypatch.setattr(update_dependencies.subprocess, "run", _fake_run())
    assert update_dependencies.uv_version() == "uv 0.9.7"


def test_missing_uv_exits_nonzero_with_install_hint(monkeypatch, capsys):
    monkeypatch.setattr(update_dependencies, "uv_version", lambda: None)
    assert update_dependencies.main(["--ci-env", "ruff"]) == 1
    err = capsys.readouterr().err
    assert "pip install uv" in err


def test_missing_uv_does_not_attempt_installation(monkeypatch):
    calls = []

    def record(command, **kwargs):
        calls.append(command)
        raise FileNotFoundError("uv")

    monkeypatch.setattr(update_dependencies.subprocess, "run", record)
    assert update_dependencies.main(["--ci-env", "ruff"]) == 1
    assert calls == [["uv", "--version"]]


def test_compile_failure_propagates_return_code(monkeypatch):
    monkeypatch.setattr(update_dependencies, "uv_version", lambda: "uv 0.9.7")
    monkeypatch.setattr(update_dependencies.subprocess, "run", _fake_run(returncode=3))
    assert update_dependencies.main(["--ci-env", "ruff"]) == 3


def test_successful_compile_runs_one_command_per_spec(monkeypatch):
    monkeypatch.setattr(update_dependencies, "uv_version", lambda: "uv 0.9.7")
    seen = []

    def runner(command, **kwargs):
        seen.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(update_dependencies.subprocess, "run", runner)
    assert update_dependencies.main(["--target", "pillow"]) == 0
    assert len(seen) == len(LOCK_SPECS)
    for spec, command in zip(LOCK_SPECS, seen):
        # Seeded image specs apply the bump through their preference list,
        # because uv ignores preferences for a package it is told to upgrade.
        seeded = spec.name in update_dependencies.SEEDED_SPECS
        assert ("--upgrade-package" in command) is not seeded, spec.name


def test_subprocess_is_never_invoked_with_a_shell(monkeypatch):
    monkeypatch.setattr(update_dependencies, "uv_version", lambda: "uv 0.9.7")
    kwargs_seen = []

    def runner(command, **kwargs):
        kwargs_seen.append(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(update_dependencies.subprocess, "run", runner)
    # Anchor the selection the way test_successful_compile_runs_one_command_per_spec
    # does: `--ci-env ruff` resolves to exactly the ci-ruff spec. A refactor that
    # renamed it would leave kwargs_seen empty and make a shell-safety assertion
    # pass by running nothing at all - main() has no empty-selection guard and
    # still returns 0.
    assert update_dependencies.main(["--ci-env", "ruff"]) == 0
    assert len(kwargs_seen) == 1
    assert all(not kwargs.get("shell", False) for kwargs in kwargs_seen)


def test_build_command_accepts_unknown_spec_shape():
    spec = LockSpec("scratch", "a.in", "a.txt", "3.13", False)
    command = build_command(spec, python_platform="macos", upgrade_package="numpy")
    assert command[3] == "a.in"
    assert command[command.index("--python-platform") + 1] == "macos"


# ---------------------------------------------------------------------------
# check_hash_pins - parsing
# ---------------------------------------------------------------------------


def _requirements(text):
    # pip joins a continuation without a separator, so compare on single spaces.
    return [
        (number, " ".join(req.split())) for number, req in check_hash_pins.iter_requirements(text)
    ]


def test_hashed_requirement_accepted():
    text = "absl-py==2.4.0 \\\n    --hash=sha256:aaa \\\n    --hash=sha256:bbb\n    # via keras\n"
    assert [req for _, req in _requirements(text)] == [
        "absl-py==2.4.0 --hash=sha256:aaa --hash=sha256:bbb"
    ]


def test_unhashed_requirement_detected(tmp_path, monkeypatch):
    path = tmp_path / "requirements.txt"
    path.write_text("numpy==2.4.6\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    violations = check_hash_pins.check_file(path)
    assert len(violations) == 1
    assert violations[0].line == 1
    assert violations[0].text == "numpy==2.4.6"


def test_blank_and_comment_lines_ignored():
    text = "#\n# autogenerated by pip-compile\n#\n\n   \n"
    assert _requirements(text) == []


def test_include_directives_ignored():
    text = "-r ../requirements.in\n-c constraints.txt\n--requirement other.in\n"
    assert _requirements(text) == []


def test_option_lines_ignored():
    text = (
        "--index-url https://pypi.org/simple\n"
        "--extra-index-url https://download.pytorch.org/whl/cpu\n"
        "--find-links ./wheels\n"
        "--only-binary :all:\n"
        "--pre\n"
    )
    assert _requirements(text) == []


def test_editable_install_ignored():
    assert _requirements("-e .\n") == []


@pytest.mark.parametrize(
    "line",
    [
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        "--extra-index-url=https://download.pytorch.org/whl/cpu",
        "--index-url https://mirror.example/simple",
        "--index-url=https://mirror.example/simple",
        "-i https://mirror.example/simple",
        "--trusted-host mirror.example",
        "--find-links ./wheels",
        "--find-links=https://mirror.example/wheels",
        "-f ./wheels",
    ],
)
def test_index_redirecting_option_is_a_violation(tmp_path, monkeypatch, line):
    # A hash-pinned lock resolves from PyPI. An option that adds or swaps a package
    # source, or exempts a host from TLS verification, is a violation even though
    # every requirement in the file still carries a hash.
    path = tmp_path / "requirements.txt"
    path.write_text(f"{line}\nnumpy==2.4.6 \\\n    --hash=sha256:aaa\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    violations = check_hash_pins.check_index_options(path)
    assert [(v.line, v.text) for v in violations] == [(1, line)]
    assert check_hash_pins.check_file(path) == []


@pytest.mark.parametrize(
    "line",
    [
        "--index-url https://pypi.org/simple",
        "--index-url https://pypi.org/simple/",
        "--index-url=https://pypi.org/simple",
        "-i https://pypi.org/simple",
        "--only-binary :all:",
        "--pre",
    ],
)
def test_pypi_default_and_non_index_options_are_allowed(tmp_path, monkeypatch, line):
    path = tmp_path / "requirements.txt"
    path.write_text(f"{line}\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert check_hash_pins.check_index_options(path) == []


# Lines that hand pip a file or an artifact the gate never sees. Until 2026-10-10
# every one of these exited 0: an include was parsed and dropped, an editable was
# skipped as an option line, and a direct reference passed because it carried a hash.
UNVERIFIABLE_LINES = [
    "-r extra.txt",
    "-rextra.txt",
    "--requirement extra.txt",
    "--requirement=extra.txt",
    "-c constraints.txt",
    "--constraint=constraints.txt",
    "-e .",
    "--editable git+https://mirror.example/x.git#egg=x",
    "evilpkg @ https://mirror.example/evilpkg-1.0.tar.gz --hash=sha256:aaa",
    "evilpkg@https://mirror.example/evilpkg-1.0.tar.gz --hash=sha256:aaa",
    "evilpkg[extra] @ file:///opt/evilpkg-1.0.tar.gz --hash=sha256:aaa",
    "https://mirror.example/evilpkg-1.0-py3-none-any.whl --hash=sha256:aaa",
    "file:///opt/evilpkg-1.0-py3-none-any.whl --hash=sha256:aaa",
    "git+https://mirror.example/x.git#egg=x",
    "./vendor/evilpkg-1.0-py3-none-any.whl --hash=sha256:aaa",
    "vendor/evilpkg-1.0.tar.gz --hash=sha256:aaa",
    "evilpkg-1.0-py3-none-any.whl --hash=sha256:aaa",
    # pip tests the WHOLE string for an archive extension, not the project name, so
    # an innocent-looking pin with a suffix is a local file, resolved from the
    # directory pip runs in.
    "numpy==2.4.6.tar.gz --hash=sha256:aaa",
    "foo>=1.0.tar --hash=sha256:aaa",
    "foo==1.0.TAR.GZ --hash=sha256:aaa",
    "foo[bar]==1.0.zip --hash=sha256:aaa",
    "foo===1.0+cpu.tgz --hash=sha256:aaa",
    # pip strips a trailing [extras] and then rstrip()s every kind of whitespace, so
    # neither the extras nor a tab before them hides the suffix.
    "evilpkg-1.0-py3-none-any.whl [x] --hash=sha256:aaa",
    "evilpkg-1.0-py3-none-any.whl\t[x] --hash=sha256:aaa",
    "numpy-2.4.6.tar.gz\t[x] --hash=sha256:aaa",
]

# pip 26.2.1's ARCHIVE_EXTENSIONS, written out so that dropping one from the gate fails.
ARCHIVE_EXTENSIONS = [
    ".zip", ".whl", ".tar.bz2", ".tbz", ".tar.xz", ".txz",
    ".tlz", ".tar.lz", ".tar.lzma", ".tar.gz", ".tgz", ".tar",
]  # fmt: skip


@pytest.mark.parametrize("extension", ARCHIVE_EXTENSIONS)
def test_every_pip_archive_extension_fails_closed(extension):
    assert check_hash_pins.iter_index_options(f"foo==1.0{extension} --hash=sha256:aaa")


def test_the_archive_extension_table_is_pips():
    filetypes = pytest.importorskip("pip._internal.utils.filetypes")
    assert set(ARCHIVE_EXTENSIONS) == set(filetypes.ARCHIVE_EXTENSIONS)
    assert set(check_hash_pins._PIP_ARCHIVE_EXTENSIONS) == set(filetypes.ARCHIVE_EXTENSIONS)


# Whitespace pip strips before an [extras] suffix that is not an ASCII space. Built
# with chr() because the repo's AST gate rejects a non-ASCII string literal, escaped
# or not (tests/test_encoding_ascii_output.py).
WIDE_SPACES = [chr(0x09), chr(0xA0), chr(0x2009), chr(0x3000)]


@pytest.mark.parametrize("space", WIDE_SPACES, ids=["tab", "nbsp", "thin", "ideographic"])
def test_whitespace_other_than_a_space_fails_closed(space):
    line = f"evilpkg-1.0-py3-none-any.whl{space}[x] --hash=sha256:aaa"
    assert check_hash_pins.iter_index_options(line)


@pytest.mark.parametrize("line", UNVERIFIABLE_LINES)
def test_includes_editables_and_direct_references_fail_closed(tmp_path, monkeypatch, line):
    path = tmp_path / "requirements.txt"
    path.write_text(f"{line}\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert [v.line for v in check_hash_pins.check_index_options(path)] == [1]


PLAIN_REQUIREMENT_LINES = [
    "numpy==2.4.6 --hash=sha256:aaa",
    "torch==2.13.0+cpu --hash=sha256:aaa",
    "Foo.Bar_baz-qux==1.0 --hash=sha256:aaa",
    "uvicorn[standard]==0.54.0 --hash=sha256:aaa",
    "pkg[a, b]==1.0 --hash=sha256:aaa",
    "pkg>=1.0,<2 --hash=sha256:aaa",
    "pkg~=1.4.2 --hash=sha256:aaa",
    "pkg (==1.0) --hash=sha256:aaa",
    'colorama==0.4.6 ; sys_platform == "win32" --hash=sha256:aaa',
    'nvidia-nccl-cu12==2.28.3 ; platform_machine == "x86_64" and sys_platform == "linux" --hash=sha256:aaa',
]


@pytest.mark.parametrize("line", PLAIN_REQUIREMENT_LINES)
def test_plain_name_requirements_are_allowed(tmp_path, monkeypatch, line):
    path = tmp_path / "requirements.txt"
    path.write_text(f"{line}\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert check_hash_pins.check_index_options(path) == []


def test_requirement_verdicts_match_pips_own_parser():
    # The plain-name test is a transcription of pip's rule for when a requirement is
    # fetched from somewhere other than the index (a link, or a PEP 508 URL). Held to
    # pip itself wherever its internals are importable, as the option table is above.
    constructors = pytest.importorskip("pip._internal.req.constructors")
    for line in PLAIN_REQUIREMENT_LINES + UNVERIFIABLE_LINES:
        requirement = line.split(" --", 1)[0]
        if requirement.startswith("-"):
            continue  # an option line: its verdicts are pinned against pip's option parser
        parts = constructors.parse_req_from_line(requirement, None)
        pip_direct = parts.link is not None or (
            parts.requirement is not None and parts.requirement.url is not None
        )
        assert bool(check_hash_pins.iter_index_options(requirement)) is pip_direct, line


def test_generated_requirement_verdicts_match_pips_own_parser():
    # A fixed list could not see the archive-suffix bypass; a cross product of names,
    # extras, specifiers and suffixes can. A shape pip itself refuses to parse is
    # skipped: pip would install nothing from it, so neither verdict is unsafe.
    constructors = pytest.importorskip("pip._internal.req.constructors")
    names = ["numpy", "Foo.Bar_baz-qux", "pkg-1.0-py3-none-any"]
    extras = ["", "[x]"]
    specifiers = ["", "==1.0", " >=1,<2", "===1.0+cpu", "~=1.4"]
    suffixes = ["", ".tar.gz", ".zip", ".whl", ".TGZ", ".tar", ".tar.xz"]
    compared = 0
    for name in names:
        for extra in extras:
            for specifier in specifiers:
                for suffix in suffixes:
                    requirement = f"{name}{extra}{specifier}{suffix}"
                    try:
                        parts = constructors.parse_req_from_line(requirement, None)
                    except Exception:  # noqa: S112 - a shape pip refuses installs nothing
                        continue
                    pip_direct = parts.link is not None or (
                        parts.requirement is not None and parts.requirement.url is not None
                    )
                    gate_rejects = bool(check_hash_pins.iter_index_options(requirement))
                    assert gate_rejects is pip_direct, requirement
                    compared += 1
    assert compared >= 185  # 190 of the 210 parse under pip 26.2.1
    # The order the v2 gate missed: an archive filename, a separator, then extras.
    # Where pip says "local file" the gate must reject; the gate may also reject a
    # shape pip reads as an index requirement, since only an ASCII space is allowed.
    linked = 0
    for name in ["evilpkg-1.0-py3-none-any.whl", "numpy-2.4.6.tar.gz", "pkg-1.0.zip"]:
        for separator in ["", " ", "  ", *WIDE_SPACES]:
            for extra in ["[x]", "[x,y]"]:
                requirement = f"{name}{separator}{extra}"
                try:
                    parts = constructors.parse_req_from_line(requirement, None)
                except Exception:  # noqa: S112 - a shape pip refuses installs nothing
                    continue
                if parts.link is not None:
                    linked += 1
                    assert check_hash_pins.iter_index_options(requirement), requirement
    assert linked >= 40  # 3 names x 7 separators x 2 extras = 42 shapes; all 42 link


@pytest.mark.parametrize(
    "line", [';python_version>"0" --hash=sha256:aaa', "; python_version>'0' --hash=sha256:aaa"]
)
def test_an_empty_requirement_before_a_marker_fails_closed(line):
    # pip refuses it in most directories, but in one named like an archive (evil.tar.gz)
    # it reads the line as that directory. Not a plain requirement, so it fails here.
    assert check_hash_pins.iter_index_options(line)


@pytest.mark.parametrize("line", ["foo==1.0.tar.gz.", "foo==1.0.tar.gz. .", "foo==1.0.zip.."])
def test_a_trailing_dot_after_an_archive_suffix_fails_closed(tmp_path, monkeypatch, line):
    # On Windows os.path.abspath drops trailing dots and spaces, so pip there sees
    # the archive suffix. Rejected everywhere, which is stricter than pip on Linux.
    path = tmp_path / "requirements.txt"
    path.write_text(f"{line} --hash=sha256:aaa\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert [v.line for v in check_hash_pins.check_index_options(path)] == [1]


def test_a_truncated_wide_bom_manifest_fails_cleanly(tmp_path, monkeypatch, capsys):
    body = "numpy==2.4.6 --hash=sha256:aaa\n".encode("utf-16-le")
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe" + body + b"\x00")  # odd byte count
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert check_hash_pins.main(["bad.txt"]) == 1
    err = capsys.readouterr().err
    assert "is not valid utf-16" in err
    assert "Traceback" not in err


def test_an_include_that_redirects_the_index_fails_the_cli(tmp_path, monkeypatch, capsys):
    # The security review's reproduction: the manifest itself is clean, and the
    # redirect sits in a file it includes, which the gate does not follow.
    (tmp_path / "extra.txt").write_text(
        "--extra-index-url https://mirror.example/simple\nevilpkg==1.0 \\\n    --hash=sha256:aaa\n",
        encoding="utf-8",
    )
    (tmp_path / "bad.txt").write_text(
        "-r extra.txt\nnumpy==2.4.6 \\\n    --hash=sha256:aaa\n", encoding="utf-8"
    )
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert check_hash_pins.main(["bad.txt"]) == 1
    assert "includes another requirement file" in capsys.readouterr().err


@pytest.mark.parametrize(
    "bom, codec",
    [(b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be"), (b"\xff\xfe\x00\x00", "utf-32-le")],
)
def test_a_wide_bom_manifest_is_read_as_pip_reads_it(tmp_path, monkeypatch, bom, codec):
    # pip strips a UTF-16 or UTF-32 byte-order mark and decodes the rest. The hash
    # check used to read every file as UTF-8 and died in a traceback instead.
    good = tmp_path / "good.txt"
    good.write_bytes(bom + "numpy==2.4.6 --hash=sha256:aaa\n".encode(codec))
    bad = tmp_path / "bad.txt"
    bad.write_bytes(bom + "numpy==2.4.6\n".encode(codec))
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert check_hash_pins.check_file(good) == []
    assert check_hash_pins.check_index_options(good) == []
    assert [v.text for v in check_hash_pins.check_file(bad)] == ["numpy==2.4.6"]


def test_an_undecodable_manifest_fails_cleanly(tmp_path, monkeypatch, capsys):
    (tmp_path / "bad.txt").write_bytes(b"numpy==2.4.6 --hash=sha256:aaa\n\xff\xfe\xfd\n")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert check_hash_pins.main(["bad.txt"]) == 1
    err = capsys.readouterr().err
    assert "not valid UTF-8" in err
    assert "Traceback" not in err


# Spellings pip 26.2.1's own requirement-file parser honours. The first version of
# this gate split each line on its first space and missed every one of these.
PIP_HONOURED_REDIRECTS = [
    "--index-url\thttps://mirror.example/simple",
    "-ihttps://mirror.example/simple",
    "-f./wheels",
    "-fhttps://mirror.example/wheels",
    "--extra-index=https://mirror.example/simple",
    "--extra https://mirror.example/simple",
    "--index=https://mirror.example/simple",
    "--find=./wheels",
    "--trusted mirror.example",
    "--pre --extra-index-url https://mirror.example/simple",
    "--index-url=https://pypi.org/simple --extra-index-url https://mirror.example/simple",
    "--index-url=https://pypi.org/simple --trusted-host mirror.example",
    "--only-binary :all: --find-links ./wheels",
    "--no-index --find-links ./wheels",
    "--index-url https://pypi.org/simple#fragment",
    "--pypi-url https://mirror.example/simple",
]

PIP_HONOURED_ALLOWED = [
    "--index-url https://pypi.org/simple  # PyPI",
    "--index-url=https://pypi.org/simple/",
    "--pre --only-binary :all:",
    "numpy==2.4.6 --hash=sha256:aaa --hash=sha256:bbb",
    'colorama==0.4.6 ; sys_platform == "win32" --hash=sha256:aaa',
    "--require-hashes",
]


@pytest.mark.parametrize("line", PIP_HONOURED_REDIRECTS)
def test_every_spelling_pip_honours_is_caught(tmp_path, monkeypatch, line):
    path = tmp_path / "requirements.txt"
    path.write_text(line + "\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert [v.line for v in check_hash_pins.check_index_options(path)] == [1]


@pytest.mark.parametrize("line", PIP_HONOURED_ALLOWED)
def test_pip_honoured_allowed_spellings_pass(tmp_path, monkeypatch, line):
    path = tmp_path / "requirements.txt"
    path.write_text(line + "\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert check_hash_pins.check_index_options(path) == []


@pytest.mark.parametrize(
    "data, expected",
    [
        # A UTF-8 byte-order mark: pip strips it, so the gate must too.
        (b"\xef\xbb\xbf--extra-index-url https://mirror.example/simple --hash=sha256:a\n", [1]),
        # A coding declaration: pip re-decodes the file, which turns +AC0ALQ- into --.
        # Line 2 also fails on its own: read as UTF-8 it is not a plain requirement.
        (
            b"# coding: utf-7\n+AC0ALQ-extra-index-url https://mirror.example/simple --hash=sha256:a\n",
            [1, 2],
        ),
        # An environment variable: pip expands it from the installing environment.
        (b"${EVIL} --hash=sha256:a\n", [1]),
    ],
)
def test_pip_decoding_and_expansion_steps_fail_closed(tmp_path, monkeypatch, data, expected):
    path = tmp_path / "requirements.txt"
    path.write_bytes(data)
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert [v.line for v in check_hash_pins.check_index_options(path)] == expected


def test_a_line_pip_cannot_parse_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "requirements.txt"
    path.write_text("--no-such-option x\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    [violation] = check_hash_pins.check_index_options(path)
    assert "unparseable options" in violation.text


def test_verdicts_match_pips_own_parser():
    # The gate stays dependency-free, so it transcribes pip's parsing rather than
    # importing it. This holds the transcription to the real thing wherever pip's
    # internals are importable.
    req_file = pytest.importorskip("pip._internal.req.req_file")
    parse_line = req_file.get_line_parser(None)
    for line in PIP_HONOURED_REDIRECTS + PIP_HONOURED_ALLOWED:
        _args, values = parse_line(req_file.COMMENT_RE.sub("", line).strip())
        pip_redirects = bool(
            (values.index_url and values.index_url not in check_hash_pins.PYPI_DEFAULT_INDEX)
            or values.extra_index_urls
            or values.find_links
            or values.trusted_hosts
        )
        assert bool(check_hash_pins.iter_index_options(line)) is pip_redirects, line


def test_index_option_split_by_a_continuation_is_still_caught(tmp_path, monkeypatch):
    path = tmp_path / "requirements.txt"
    path.write_text("--extra-index-url \\\n    https://mirror.example/simple\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    violations = check_hash_pins.check_index_options(path)
    assert [(v.line, v.text) for v in violations] == [
        (1, "--extra-index-url https://mirror.example/simple")
    ]


def test_environment_marker_survives_joining():
    text = 'colorama==0.4.6 ; sys_platform == "win32" \\\n    --hash=sha256:aaa\n'
    assert [req for _, req in _requirements(text)] == [
        'colorama==0.4.6 ; sys_platform == "win32" --hash=sha256:aaa'
    ]


def test_trailing_inline_comment_stripped():
    text = "numpy==2.4.6 \\\n    --hash=sha256:aaa  # pinned\n"
    assert [req for _, req in _requirements(text)] == ["numpy==2.4.6 --hash=sha256:aaa"]


def test_line_number_points_at_the_requirement_start():
    text = "# header\n\nnumpy==2.4.6 \\\n    --hash=sha256:aaa\nscipy==1.17.1\n"
    assert [number for number, _ in _requirements(text)] == [3, 5]


def test_continuation_terminated_by_comment_line_is_still_checked():
    text = "numpy==2.4.6 \\\n# stray comment\nscipy==1.17.1 \\\n    --hash=sha256:aaa\n"
    numbers = [number for number, _ in _requirements(text)]
    unhashed = [req for _, req in _requirements(text) if "--hash=" not in req]
    assert numbers == [1, 3]
    assert unhashed == ["numpy==2.4.6"]


def test_unterminated_continuation_at_eof_is_still_checked():
    assert [req for _, req in _requirements("numpy==2.4.6 \\\n")] == ["numpy==2.4.6"]


# Lines that end in a backslash only once a comment is cut at ANY `#`. pip cuts one
# only after whitespace, so to pip neither continues: the first is `--pre` and an
# escaped `#`, the second `--pre` and an escaped backslash before a comment. Until
# 2026-10-11 the hash check used its own line reader, joined the next requirement
# into the option line and never checked it.
LINES_THAT_ONLY_LOOK_CONTINUED = ["--pre \\#", "--pre \\\\ # comment"]


@pytest.mark.parametrize("line", LINES_THAT_ONLY_LOOK_CONTINUED)
def test_a_line_pip_does_not_continue_hides_no_requirement(tmp_path, monkeypatch, line):
    path = tmp_path / "requirements.txt"
    path.write_text(f"{line}\nevilpkg\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert [(v.line, v.text) for v in check_hash_pins.check_file(path)] == [(2, "evilpkg")]
    assert check_hash_pins.check_index_options(path) == []


@pytest.mark.parametrize(
    "line",
    ["evilpkg -C--hash=x", "evilpkg --config-settings=--hash=x", "evilpkg -C --hash=sha256:aaa"],
)
def test_hash_text_in_another_options_value_is_not_a_hash(tmp_path, monkeypatch, line):
    # optparse hands -C the next token whatever it starts with, so pip attaches no
    # hash to any of these and installs evilpkg unchecked. The check used to accept
    # any `--hash=` text.
    path = tmp_path / "requirements.txt"
    path.write_text(f"{line}\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert [v.line for v in check_hash_pins.check_file(path)] == [1]


def test_a_requirement_whose_options_cannot_be_parsed_counts_as_unhashed(tmp_path, monkeypatch):
    path = tmp_path / "requirements.txt"
    path.write_text("numpy==2.4.6 --hash=sha256:aaa --no-such-option\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert [v.line for v in check_hash_pins.check_file(path)] == [1]
    [violation] = check_hash_pins.check_index_options(path)
    assert "unparseable options" in violation.text


@pytest.mark.parametrize(
    "text",
    [
        # `#` after a quote is not a comment to pip, so the backslash still continues
        # onto the hash. Linux platform_version strings start with `#`.
        'numpy==2.4.6 ; platform_version != "#1 SMP" \\\n    --hash=sha256:aaa\n',
        # optparse takes an option's value from the next token too.
        "numpy==2.4.6 --hash sha256:aaa\n",
    ],
    ids=["hash-inside-a-marker", "space-separated-hash"],
)
def test_a_hash_pip_attaches_is_accepted(tmp_path, monkeypatch, text):
    path = tmp_path / "requirements.txt"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert check_hash_pins.check_file(path) == []
    assert check_hash_pins.check_index_options(path) == []


def test_unhashed_requirement_lines_match_pips_own_reading():
    # Every three-line manifest built from lines that probe pip's line joining,
    # comment rule and option split, held to pip's own reading wherever its internals
    # are importable. A manifest pip refuses to parse installs nothing, so it is skipped.
    req_file = pytest.importorskip("pip._internal.req.req_file")
    parse_line = req_file.get_line_parser(None)
    pool = [
        "",
        "evilpkg",
        "numpy==2.4.6 --hash=sha256:aaa",
        "numpy==2.4.6 \\",
        "    --hash=sha256:aaa",
        "# --hash=sha256:aaa",
        "evilpkg #--hash=sha256:aaa",
        "evilpkg#--hash=sha256:aaa",
        "evilpkg -C--hash=x",
        "numpy==2.4.6 --hash sha256:aaa",
        'numpy==2.4.6 ; platform_version != "#1 SMP" \\',
        "--pre \\",
        *LINES_THAT_ONLY_LOOK_CONTINUED,
    ]
    compared = 0
    for lines in itertools.product(pool, repeat=3):
        text = "\n".join(lines) + "\n"
        try:
            parsed = [(number, *parse_line(line)) for number, line in req_file.preprocess(text)]
        except req_file.OptionParsingError:
            continue
        pip_unhashed = [number for number, args, values in parsed if args and not values.hashes]
        gate_unhashed = [number for number, _ in check_hash_pins.iter_unhashed_requirements(text)]
        assert gate_unhashed == pip_unhashed, text
        compared += 1
    assert compared >= 2700  # all 2,744 parse under pip 26.2.1


# ---------------------------------------------------------------------------
# check_hash_pins - CLI
# ---------------------------------------------------------------------------


def test_repo_manifests_are_all_hash_pinned():
    assert check_hash_pins.main([]) == 0


def test_default_targets_cover_runtime_and_ci_manifests():
    paths = check_hash_pins.resolve_targets(list(check_hash_pins.DEFAULT_TARGETS))
    names = {path.name for path in paths}
    assert "requirements.txt" in names
    assert any(name.startswith("requirements-ci-") for name in names)


def test_cli_fails_on_an_unhashed_manifest(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "bad.txt"
    manifest.write_text("numpy==2.4.6\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert check_hash_pins.main(["bad.txt"]) == 1
    assert "un-hashed requirements" in capsys.readouterr().err


def test_cli_fails_on_an_index_redirecting_option(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "bad.txt"
    manifest.write_text(
        "--extra-index-url https://mirror.example/simple\nnumpy==2.4.6 \\\n    --hash=sha256:aaa\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert check_hash_pins.main(["bad.txt"]) == 1
    err = capsys.readouterr().err
    assert "index-redirecting options" in err
    # Assembled from two pieces: the contiguous file-colon-line form would be read by
    # scripts/check_doc_line_citations.py as an unpinned citation.
    assert "bad.txt" + ":1: --extra-index-url https://mirror.example/simple" in err
    assert "un-hashed requirements" not in err


def test_the_review_reproduction_fails_the_cli(tmp_path, monkeypatch, capsys):
    # An independent review's reproduction, which exited 0. pip 26.2.1 installs
    # evilpkg from it unpinned and unhashed wherever the install command omits
    # --require-hashes: line 1 turns off the hash checking goodpkg's hash implies,
    # and line 2 is, to pip, `--pre` and an escaped `#`, so it does not continue.
    manifest = tmp_path / "bad.txt"
    manifest.write_text(
        "--no-require-hashes\n--pre \\#\nevilpkg\ngoodpkg==1.0 --hash=sha256:" + "0" * 64 + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert check_hash_pins.main(["bad.txt"]) == 1
    err = capsys.readouterr().err
    # Assembled from two pieces, for the reason given in the test above.
    assert "bad.txt" + ":3: evilpkg\n" in err
    assert "bad.txt" + ":1: --no-require-hashes" in err


NO_REQUIRE_HASHES_LINES = [
    "--no-require-hashes",
    "--no-req",
    "--pre --no-require-hashes",
    "--only-binary :all: --no-require-hashes",
]


@pytest.mark.parametrize("line", NO_REQUIRE_HASHES_LINES)
def test_turning_off_pips_implied_hash_checking_fails_closed(tmp_path, monkeypatch, line):
    path = tmp_path / "requirements.txt"
    path.write_text(f"{line}\nnumpy==2.4.6 --hash=sha256:aaa\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert [v.line for v in check_hash_pins.check_index_options(path)] == [1]
    assert check_hash_pins.check_file(path) == []


def test_no_require_hashes_spellings_are_pips():
    req_file = pytest.importorskip("pip._internal.req.req_file")
    parse_line = req_file.get_line_parser(None)
    for line in NO_REQUIRE_HASHES_LINES:
        _args, values = parse_line(line)
        assert values.no_require_hashes is True, line


def test_cli_passes_on_a_fully_hashed_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "good.txt"
    manifest.write_text("numpy==2.4.6 \\\n    --hash=sha256:aaa\n    # via -r x.in\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert check_hash_pins.main(["good.txt"]) == 0


def test_cli_fails_when_a_manifest_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert check_hash_pins.main(["absent.txt"]) == 1
    assert "no such manifest" in capsys.readouterr().err


def test_cli_fails_when_a_glob_matches_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert check_hash_pins.main(["nothing-*.txt"]) == 1
    assert "no manifests matched" in capsys.readouterr().err


def test_glob_targets_expand(tmp_path, monkeypatch):
    (tmp_path / "requirements-ci-a.txt").write_text("a==1 \\\n --hash=sha256:x\n", encoding="utf-8")
    (tmp_path / "requirements-ci-b.txt").write_text("b==1\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    paths = check_hash_pins.resolve_targets(["requirements-ci-*.txt"])
    assert [path.name for path in paths] == ["requirements-ci-a.txt", "requirements-ci-b.txt"]
    assert check_hash_pins.main(["requirements-ci-*.txt"]) == 1
