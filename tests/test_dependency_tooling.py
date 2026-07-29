"""Unit tests for the dependency-management tooling in tools/.

Covers tools/update_dependencies.py argument construction and its
uv-not-installed path (uv is deliberately not a repo dependency, so every uv
invocation here is mocked), plus the false-positive surface of
tools/check_hash_pins.py.
"""

import pathlib
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
    # reachable only via --all. The two application lockfiles are excluded.
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


def test_runtime_lockfiles_compile_with_the_uv_override():
    # requirements.in / requirements-lock.in floor setuptools against torch's
    # `setuptools<82` cap; without --overrides uv returns ResolutionImpossible.
    for name in update_dependencies.RUNTIME_SPEC_NAMES:
        spec = next(s for s in LOCK_SPECS if s.name == name)
        command = build_command(spec)
        assert "--overrides" in command
        assert command[command.index("--overrides") + 1] == update_dependencies.OVERRIDES_FILE


def test_override_file_exists_and_is_only_used_where_needed():
    root = pathlib.Path(update_dependencies.REPO_ROOT)
    assert (root / update_dependencies.OVERRIDES_FILE).is_file()
    for spec in LOCK_SPECS:
        if spec.name not in update_dependencies.RUNTIME_SPEC_NAMES:
            assert "--overrides" not in build_command(spec), spec.name


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
    assert all("--upgrade-package" in command for command in seen)


def test_subprocess_is_never_invoked_with_a_shell(monkeypatch):
    monkeypatch.setattr(update_dependencies, "uv_version", lambda: "uv 0.9.7")
    kwargs_seen = []

    def runner(command, **kwargs):
        kwargs_seen.append(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(update_dependencies.subprocess, "run", runner)
    update_dependencies.main(["--ci-env", "ruff"])
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
