"""Unit tests for the dependency-management tooling in tools/.

Covers tools/update_dependencies.py argument construction and its
uv-not-installed path (uv is deliberately not a repo dependency, so every uv
invocation here is mocked), plus the false-positive surface of
tools/check_hash_pins.py.
"""

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


def test_no_spec_compiles_with_a_uv_override_file():
    # History: requirements.in / requirements-lock.in floor setuptools>=83.0.0
    # for GHSA-h35f-9h28-mq5c, which collided with torch 2.12.0's declared
    # `setuptools<82` build-metadata cap and made both specs unsatisfiable for
    # any resolver. Both therefore compiled with `--overrides overrides.txt`.
    # torch 2.13.0 raised the cap to `setuptools>=77.0.3`, so the override was
    # retired. This asserts the workaround does not creep back in: a
    # reintroduced override would silently mask a genuine resolution conflict.
    #
    # The semgrep spec was a deliberate exception while semgrep declared
    # pyjwt[crypto]~=2.13.0: it compiled with environments/semgrep-overrides.txt
    # to lift pyjwt. semgrep 1.179.0 declares pyjwt[crypto]>=2.15.0,<3, so that
    # override was retired too.
    for spec in LOCK_SPECS:
        assert spec.overrides is None, spec.name
        assert "--overrides" not in build_command(spec), spec.name


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
    # while the semgrep spec had one:
    # `pip install --require-hashes -r environments/requirements-semgrep.txt`
    # exited 1 against the overridden lock and 0 with --no-deps added. Dormant
    # while no spec sets `overrides` (see the test above); it applies again to
    # any spec that does.
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


def test_the_retired_override_files_are_gone():
    # Guards the other half of each retirement: the files themselves must not return.
    root = pathlib.Path(update_dependencies.REPO_ROOT)
    assert not (root / "overrides.txt").exists()
    assert not (root / "environments" / "semgrep-overrides.txt").exists()


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
    return check_hash_pins.iter_requirements(text)


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
        "-r ../requirements.in",
        "-c constraints.txt",
        "--only-binary :all:",
        "--pre",
        "-e .",
    ],
)
def test_pypi_default_and_non_index_options_are_allowed(tmp_path, monkeypatch, line):
    path = tmp_path / "requirements.txt"
    path.write_text(f"{line}\n", encoding="utf-8")
    monkeypatch.setattr(check_hash_pins, "REPO_ROOT", tmp_path)
    assert check_hash_pins.check_index_options(path) == []


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
