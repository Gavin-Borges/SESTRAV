"""Release jobs with OIDC authority must not resolve Python dependencies, and the
job that builds the release artifacts must not run unhashed third-party code."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "release.yml"


def _jobs() -> dict:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return document["jobs"]


def _has_oidc_write(job: dict) -> bool:
    return job.get("permissions", {}).get("id-token") == "write"


def _run_script(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in job.get("steps", []))


def test_token_holding_jobs_do_not_run_pip_install() -> None:
    token_jobs = {name: job for name, job in _jobs().items() if _has_oidc_write(job)}

    assert set(token_jobs) == {"release", "publish"}
    for name, job in token_jobs.items():
        assert "pip install" not in _run_script(job), name


def _has_checkout(job: dict) -> bool:
    return any(
        str(step.get("uses", "")).startswith("actions/checkout@") for step in job.get("steps", [])
    )


def test_gh_steps_in_jobs_without_a_checkout_name_the_repository() -> None:
    """gh resolves the repository from the checkout's git remote, or from GH_REPO.

    It does not read GITHUB_REPOSITORY, so in a job with no checkout a bare
    `gh release create` fails with "not a git repository". Splitting release.yml
    into build, release and publish jobs created exactly such a job.
    """
    checked = 0
    for name, job in _jobs().items():
        if _has_checkout(job):
            continue
        for step in job.get("steps", []):
            script = str(step.get("run", ""))
            if not any(line.lstrip().startswith("gh ") for line in script.splitlines()):
                continue
            checked += 1
            env = step.get("env", {}) or {}
            assert "GH_REPO" in env or "--repo" in script, (
                f"job {name!r}, step {step.get('name')!r} runs gh with no checkout "
                "and neither sets GH_REPO nor passes --repo"
            )
    assert checked >= 1, "no gh step in a checkout-less job was found; the guard is vacuous"


def _needs(job: dict) -> set[str]:
    needs = job.get("needs", [])
    return {needs} if isinstance(needs, str) else set(needs)


# pip's global options, each possibly followed by a value (`pip --log x install`).
_PIP_OPTIONS = r"(?:\s+-\S+(?:\s+[^\s-]\S*)?)*?"
_PIP_INSTALL = re.compile(
    rf"\bpip3?(?:\.\d+)?{_PIP_OPTIONS}\s+install\b|-m\s+pip{_PIP_OPTIONS}\s+install\b"
)
# Anything else that fetches or builds third-party code: pip wheel and pip download
# run an sdist's build backend too.
_OTHER_INSTALLERS = re.compile(
    rf"(?:^|\s)(?:uvx|pipx|uv\s+(?:pip|tool|run|add|sync)|conda\s+install|easy_install)\b"
    rf"|\bpip3?(?:\.\d+)?{_PIP_OPTIONS}\s+(?:wheel|download)\b|-m\s+pip{_PIP_OPTIONS}\s+(?:wheel|download)\b"
)


def _can_be_skipped_or_ignored(node: dict) -> bool:
    """A job or step that might not run, or whose failure would not count."""
    return "if" in node or bool(node.get("continue-on-error"))


def _commands(job: dict) -> list[str]:
    """Every shell command in a job's run steps: continuations joined, comments dropped,
    split on &&, ||, | and ;, whitespace collapsed."""
    commands = []
    for step in job.get("steps", []):
        run = step.get("run")
        if not isinstance(run, str):
            continue
        for line in re.sub(r"\\\s*\n", " ", run).splitlines():
            line = re.sub(r"(?:^|\s)#.*$", "", line)
            for command in re.split(r"&&|\|\||;|\|", line):
                command = " ".join(command.split())
                if command:
                    commands.append(command)
    return commands


def _pip_installs(job: dict) -> list[str]:
    return [c for c in _commands(job) if _PIP_INSTALL.search(c)]


def _step_index(job: dict, predicate) -> int:
    return next((i for i, step in enumerate(job.get("steps", [])) if predicate(step)), -1)


def test_dependency_checks_run_before_token_holding_jobs() -> None:
    jobs = _jobs()

    assert not _has_oidc_write(jobs["build"])
    assert not _has_oidc_write(jobs["verify"])
    assert "pip install" in _run_script(jobs["build"])
    assert "pip install" in _run_script(jobs["verify"])
    assert _needs(jobs["verify"]) == {"build"}
    assert _needs(jobs["release"]) == {"build", "verify"}
    assert _needs(jobs["publish"]) == {"release"}
    assert _needs(jobs["smoke"]) == {"publish"}
    assert not _has_oidc_write(jobs["smoke"])
    assert "pip install" in _run_script(jobs["smoke"])
    # An `if:` such as always() would run the release even when verify failed.
    assert "if" not in jobs["release"], jobs["release"].get("if")


def test_the_job_that_builds_the_artifacts_runs_no_unhashed_install() -> None:
    """The pre-publish gate installs the built wheel with its dependencies resolved
    from PyPI, unhashed by design, then imports the package and runs its CLI. In
    the build job that third-party code ran beside dist/ before the upload, where a
    compromised dependency could alter what the release job attests and publishes.
    The build job may only run hash-pinned installs; the gate runs in `verify`,
    against the artifact `build` already uploaded.
    """
    jobs = _jobs()
    build = jobs["build"]
    installs = _pip_installs(build)
    assert installs, "no pip install found in the build job; this guard has gone vacuous"
    unhashed = [c for c in installs if "--require-hashes" not in c.split()]
    assert not unhashed, unhashed
    others = [c for c in _commands(build) if _OTHER_INSTALLERS.search(c)]
    assert not others, others
    third_party = [
        step["uses"]
        for step in build.get("steps", [])
        if "uses" in step and not str(step["uses"]).startswith("actions/")
    ]
    assert not third_party, third_party

    verify = jobs["verify"]
    assert verify.get("permissions") == {"contents": "read"}, verify.get("permissions")
    download = _step_index(
        verify, lambda s: str(s.get("uses", "")).startswith("actions/download-artifact@")
    )
    gate = _step_index(verify, lambda s: "dist/*.whl" in str(s.get("run", "")))
    assert gate >= 0, "the gate is not in the verify job"
    assert 0 <= download < gate, (download, gate)
    assert (
        verify["steps"][download].get("with", {}).get("name")
        == "release-input-${{ github.ref_name }}"
    )
    # A gate that could be skipped or whose failure is ignored would let verify pass.
    assert not _can_be_skipped_or_ignored(verify)
    assert not _can_be_skipped_or_ignored(verify["steps"][gate])
    assert not _can_be_skipped_or_ignored(verify["steps"][download])


def test_the_release_job_checks_the_artifact_against_the_build_digests() -> None:
    """An artifact is downloaded by name, and a later job in the run can replace one of
    the same name, so the release job checks every file against digests the build job
    recorded as a job output, which nothing after that job can change."""
    jobs = _jobs()
    build, release = jobs["build"], jobs["release"]
    assert build.get("outputs", {}).get("artifact-sha256") == "${{ steps.digests.outputs.sha256 }}"

    record = _step_index(build, lambda s: s.get("id") == "digests")
    upload = _step_index(
        build, lambda s: str(s.get("uses", "")).startswith("actions/upload-artifact@")
    )
    assert 0 <= record < upload, (record, upload)
    recorded = build["steps"][record]["run"]
    assert "$GITHUB_OUTPUT" in recorded and "sha256sum" in recorded
    uploaded_globs = build["steps"][upload]["with"]["path"].split()
    hashed_globs = recorded.split("sha256sum", 1)[1].splitlines()[0].split()
    assert sorted(hashed_globs) == sorted(uploaded_globs), (hashed_globs, uploaded_globs)

    download = _step_index(
        release, lambda s: str(s.get("uses", "")).startswith("actions/download-artifact@")
    )
    check = _step_index(
        release,
        lambda s: s.get("env", {}).get("EXPECTED") == "${{ needs.build.outputs.artifact-sha256 }}",
    )
    attest = _step_index(
        release, lambda s: str(s.get("uses", "")).startswith("actions/attest-build-provenance@")
    )
    assert download == 0 and check == 1 and attest > check, (download, check, attest)
    script = release["steps"][check]["run"]
    assert 'test -n "$EXPECTED"' in script
    assert "sha256sum --check --strict" in script
    assert "diff " in script
    # The check must be able to fail the job.
    assert not _can_be_skipped_or_ignored(release["steps"][check])
    assert "||" not in script and "set +e" not in script and "shell" not in release["steps"][check]


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ("pip install --require-hashes -r environments/requirements-ci-build.txt", []),
        (
            "python -m build --no-isolation\npip install --quiet dist/*.whl",
            ["pip install --quiet dist/*.whl"],
        ),
        ("a && pip install x  # comment", ["pip install x"]),
        ("pip install \\\n  --require-hashes -r x.txt", []),
        ("pip install \\\n  -r x.txt", ["pip install -r x.txt"]),
        ("pip3 install foo", ["pip3 install foo"]),
        ("python -m pip --quiet install foo", ["python -m pip --quiet install foo"]),
        ("pip  install foo", ["pip install foo"]),
        ("pip install --require-hashes -r x.txt || pip install build", ["pip install build"]),
        ("echo x | pip install foo", ["pip install foo"]),
        ("pip --log /tmp/pip.log install foo", ["pip --log /tmp/pip.log install foo"]),
        (
            "python -m pip --cache-dir c --quiet install foo",
            ["python -m pip --cache-dir c --quiet install foo"],
        ),
        ("pip --version", []),
    ],
)
def test_pip_installs_reads_commands(script: str, expected: list[str]) -> None:
    found = _pip_installs({"steps": [{"run": script}]})
    unhashed = [c for c in found if "--require-hashes" not in c.split()]
    assert unhashed == expected


@pytest.mark.parametrize(
    ("command", "matches"),
    [
        ("pip wheel --no-deps -w w foo", True),
        ("python -m pip --quiet download foo", True),
        ("pip --log x wheel foo", True),
        ("conda install -y foo", True),
        ("easy_install foo", True),
        ("uvx build", True),
        ("uv pip install foo", True),
        ("pipx run build", True),
        ("pip install --require-hashes -r x.txt", False),
        ("python -m build --no-isolation --sdist --wheel --outdir dist/", False),
        ("sha256sum dist/*.whl", False),
    ],
)
def test_other_installers_are_recognised(command: str, matches: bool) -> None:
    assert bool(_OTHER_INSTALLERS.search(command)) is matches


@pytest.mark.parametrize(
    ("node", "flagged"),
    [
        ({}, False),
        ({"continue-on-error": False}, False),
        ({"continue-on-error": True}, True),
        ({"continue-on-error": "${{ always() }}"}, True),
        ({"if": "success()"}, True),
    ],
)
def test_skippable_or_ignored_is_recognised(node: dict, flagged: bool) -> None:
    assert _can_be_skipped_or_ignored(node) is flagged


def _gate_step() -> tuple[str, dict]:
    """The pre-publish gate step, found by the command list it builds."""
    for name, job in _jobs().items():
        for step in job.get("steps", []):
            if "--list-commands" in str(step.get("run", "")):
                return name, step
    raise AssertionError("no step in release.yml runs --list-commands")


def _subcommand_help_block() -> str:
    """The gate's per-subcommand --help block, taken verbatim from release.yml.

    Continuations are joined first, so the block reads the same whether the
    command list is produced on one line or several.
    """
    _name, step = _gate_step()
    lines = re.sub(r"\\\s*\n", " ", str(step["run"])).splitlines()
    start = next(i for i, line in enumerate(lines) if "--list-commands" in line)
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "done")
    return "\n".join(lines[start : end + 1])


# The two executables in the gate's throwaway venv, matched as a whole absolute
# path so the surrounding `$(` or `"` is never swallowed with it.
_VENV_EXECUTABLE = re.compile(r"(?:/[\w.-]+)*/relcheck/bin/(python|sestrav)\b")
_STUBS = {"python": "stub_list", "sestrav": "stub_sestrav"}


def _run_help_block(tmp_path: Path, stdout: str, status: int) -> tuple[int, list[str]]:
    """Run that block under GitHub's default shell with the list producer planted.

    The two venv executables are the only text replaced; the assignment, the
    emptiness check and the loop itself are whatever release.yml says they are,
    which is what makes this a control on the workflow rather than on a copy of
    it.
    """
    block = _VENV_EXECUTABLE.sub(lambda match: _STUBS[match.group(1)], _subcommand_help_block())
    assert "stub_list" in block and "stub_sestrav" in block, block
    # The planted stdout goes through a file so the test never has to quote a
    # multi-line string into shell source.
    # Written as bytes, not text: on Windows a text-mode write would turn every
    # newline into CRLF, and the stray CR then rides into the loop variable.
    listing = tmp_path / "listing.txt"
    listing.write_bytes(stdout.encode("utf-8"))
    ran = tmp_path / "ran.txt"
    ran.unlink(missing_ok=True)
    script = tmp_path / "help_block.sh"
    script.write_bytes(
        (
            'stub_list() { cat "$LISTING"; return ' + str(status) + "; }\n"
            'stub_sestrav() { printf "%s\\n" "$*" >> "$RAN"; }\n'
            f"{block}\n"
        ).encode("utf-8")
    )
    # GitHub's default shell for a `run:` step with no `shell:` key is
    # `bash -e {0}`: errexit on, pipefail off. The step sets no `shell:`, so
    # relying on either -o pipefail or -u here would test something else.
    assert "shell" not in _gate_step()[1]
    environment = dict(os.environ)
    environment.update(GITHUB_WORKSPACE=str(tmp_path), LISTING=str(listing), RAN=str(ran))
    completed = subprocess.run(
        ["bash", "-e", str(script)],
        capture_output=True,
        text=True,
        env=environment,
    )
    exercised = ran.read_text(encoding="utf-8").splitlines() if ran.exists() else []
    return completed.returncode, exercised


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs a POSIX bash")
def test_the_subcommand_help_loop_fails_on_an_empty_or_failed_list(tmp_path: Path) -> None:
    """A planted empty or failed command list must fail the step.

    `for command in $(...)` throws the producer's exit status away and runs zero
    iterations on empty output, so before this guard all three planted failures
    below left the gate green having run no subcommand at all. The third is the
    one the step actually meets: `check_consumer_install.py --list-commands`
    prints the names and returns 1 when a subparser has no cmd_ function, so the
    discarded status was the only signal that anything was wrong.
    """
    for stdout, status, label in [
        ("", 0, "empty output, producer succeeded"),
        ("", 1, "empty output, producer failed"),
        ("predict\nvalidate\n", 1, "names printed, producer failed"),
    ]:
        returncode, exercised = _run_help_block(tmp_path, stdout, status)
        assert returncode != 0, f"{label}: step passed (rc 0) on a list it must reject"
        assert exercised == [], (label, exercised)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs a POSIX bash")
def test_the_subcommand_help_loop_still_runs_every_command_it_is_given(tmp_path: Path) -> None:
    """Negative control: the harness above can pass, so its failures are real."""
    returncode, exercised = _run_help_block(tmp_path, "predict\nvalidate\n", 0)
    assert returncode == 0, returncode
    assert exercised == ["predict --help", "validate --help"], exercised
