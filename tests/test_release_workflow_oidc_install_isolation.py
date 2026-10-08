"""Release jobs with OIDC authority must not resolve Python dependencies, the
job that builds the release artifacts must not run unhashed third-party code,
and every job that uses a downloaded artifact first checks it against the build
job's digests, with a check these tests fail if it is disarmed."""

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "release.yml"


def _document() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _jobs() -> dict:
    return _document()["jobs"]


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
    assert _needs(jobs["publish"]) == {"build", "release"}
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
    # The check must be able to fail the job, and nothing may disarm it.
    assert not digest_check_problems(_document(), "release")


def test_the_publish_job_checks_the_distributions_against_the_build_digests() -> None:
    """publish downloads a second artifact by name, uploaded by the release job, so
    it repeats the release job's check, on the sdist and wheel, before PyPI."""
    publish = _jobs()["publish"]
    download = _step_index(
        publish, lambda s: str(s.get("uses", "")).startswith("actions/download-artifact@")
    )
    check = _step_index(
        publish,
        lambda s: s.get("env", {}).get("EXPECTED") == "${{ needs.build.outputs.artifact-sha256 }}",
    )
    upload = _step_index(
        publish, lambda s: str(s.get("uses", "")).startswith("pypa/gh-action-pypi-publish@")
    )
    assert download == 0 and check == 1 and upload == 2, (download, check, upload)
    script = publish["steps"][check]["run"]
    assert "grep -E '^[0-9a-f]{64}  dist/[^/]+$'" in script
    assert 'test -s "$RUNNER_TEMP/dist.sha256"' in script
    assert 'sha256sum --check --strict "$RUNNER_TEMP/dist.sha256"' in script
    assert "find dist -type f" in script and "diff " in script
    assert not digest_check_problems(_document(), "publish")


# Every job that checks a downloaded artifact against the digests the build job
# recorded, and the `uses:` prefix of the step that check exists to guard.
DIGEST_CHECKS = {
    "release": "actions/attest-build-provenance@",
    "publish": "pypa/gh-action-pypi-publish@",
}
_EXPECTED = "${{ needs.build.outputs.artifact-sha256 }}"
# `printf '%s\n' "$EXPECTED"` as shlex splits it: the one source of expected digests.
_FROM_EXPECTED = ["printf", "%s\\n", "$EXPECTED"]
# The commands a digest check may run, each line one plain pipeline of them. Under
# the default `bash -e` the step then fails as soon as a line's last command
# fails; anything else (exit, set, trap, true, if, a subshell, a list) could end
# the step early or swallow a failure.
_CHECK_COMMANDS = frozenset({"awk", "diff", "find", "grep", "printf", "sha256sum", "sort", "test"})
_OPERATOR = re.compile(r"[();<>|&]+")
_CHECK_ERREXIT_OFF = re.compile(r"\bset\s+(?:-\S*\s+)*\+[A-Za-z]*e|\bset\s+\+o\s+errexit\b")
_SHELL_HAS_ERREXIT = re.compile(r"(?:^|\s)-[A-Za-z]*e")


def _pipelines(script: str) -> list[list[list[str]]]:
    """Each line of a run script as its pipeline stages, each a list of shell words.

    Continuations are joined and comments dropped, quotes are removed as the
    shell removes them, and every operator (`||`, `;`, `>`) is a word of its own.
    """
    pipelines = []
    for line in re.sub(r"\\\s*\n", " ", script).splitlines():
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        stages: list[list[str]] = [[]]
        for word in lexer:
            if word == "|":
                stages.append([])
            else:
                stages[-1].append(word)
        if stages != [[]]:
            pipelines.append(stages)
    return pipelines


def _temp_files(words: list[str]) -> set[str]:
    return {word for word in words if word.startswith("$RUNNER_TEMP/")}


def _check_index(job: dict) -> int:
    return _step_index(job, lambda s: "sha256sum --check" in str(s.get("run", "")))


def digest_check_problems(document: dict, job_name: str) -> list[str]:
    """Why a job's digest check could let the step it guards run on unchecked files.

    The check has to take its expected digests from the build job's output and
    never from the download itself, fail the step on any mismatch or added file,
    and sit immediately before the step it guards, so nothing changes the files
    in between. Anything that skips the step, ignores its failure, runs it
    without -e, turns errexit off or exits early disarms it.
    """
    job = document["jobs"][job_name]
    steps = job.get("steps", [])
    check = _check_index(job)
    guarded = _step_index(job, lambda s: str(s.get("uses", "")).startswith(DIGEST_CHECKS[job_name]))
    if check < 0 or guarded < 0:
        return [f"no digest check ({check}) or no guarded step ({guarded}) in {job_name}"]
    step, script = steps[check], str(steps[check]["run"])
    problems = []
    if guarded != check + 1:
        problems.append(f"the check is step {check}, not the one just before step {guarded}")
    if "build" not in _needs(job) or (step.get("env") or {}).get("EXPECTED") != _EXPECTED:
        problems.append("EXPECTED is not the build job's artifact-sha256 output")
    if _can_be_skipped_or_ignored(step):
        problems.append("the check can be skipped or its failure ignored")
    problems += [f"the check sets {key}" for key in ("shell", "working-directory") if key in step]
    for where, node in (("the workflow", document), (f"job {job_name}", job)):
        run = (node.get("defaults") or {}).get("run") or {}
        if "shell" in run and not _SHELL_HAS_ERREXIT.search(str(run["shell"])):
            problems.append(f"{where} defaults.run.shell {run['shell']!r} carries no -e")
        if "working-directory" in run:
            problems.append(f"{where} defaults.run.working-directory moves the check")
    if _CHECK_ERREXIT_OFF.search(script):
        problems.append("the check turns errexit off")
    if re.search(r"\bexit\b", script):
        problems.append("the check can exit before it fails")

    from_expected: set[str] = set()  # files holding only what EXPECTED said
    listed: set[str] = set()  # files listing what was downloaded
    checked = compared = False
    for stages in _pipelines(script):
        output = None
        if len(stages[-1]) >= 2 and stages[-1][-2] == ">":
            output, stages[-1] = stages[-1][-1], stages[-1][:-2]
        if any(
            not stage or stage[0] not in _CHECK_COMMANDS or any(map(_OPERATOR.fullmatch, stage))
            for stage in stages
        ):
            problems.append(f"not a plain pipeline of {sorted(_CHECK_COMMANDS)}: {stages}")
            continue
        first = stages[0]
        reads_expected = first == _FROM_EXPECTED or (
            bool(_temp_files(first)) and _temp_files(first) <= from_expected
        )
        for position, stage in enumerate(stages):
            if stage[0] != "sha256sum":
                continue
            operands = [word for word in stage[1:] if word == "-" or not word.startswith("-")]
            if "--check" not in stage or "--strict" not in stage or "--ignore-missing" in stage:
                problems.append(f"sha256sum computes digests or checks loosely: {stage}")
            elif operands == ["-"] and position > 0 and reads_expected:
                checked = True
            elif len(operands) == 1 and position == 0 and operands[0] in from_expected:
                checked = True
            else:
                problems.append(f"sha256sum checks digests that are not EXPECTED's: {stage}")
        if first[0] == "diff":
            operands = [word for word in first[1:] if not word.startswith("-")]
            compared = compared or (
                len(operands) == 2
                and any(a in from_expected and b in listed for a, b in (operands, operands[::-1]))
            )
        if output is not None:
            (from_expected.add if reads_expected else from_expected.discard)(output)
            (listed.add if first[0] == "find" else listed.discard)(output)
    if not checked:
        problems.append("no sha256sum --check reads the digests EXPECTED holds")
    if not compared:
        problems.append("no diff compares EXPECTED's file names with what find lists")
    return problems


@pytest.mark.parametrize("job_name", sorted(DIGEST_CHECKS))
def test_every_digest_check_is_armed(job_name: str) -> None:
    assert not digest_check_problems(_document(), job_name)


def test_every_token_holding_job_that_downloads_has_a_digest_check() -> None:
    """A job that holds a token and downloads an artifact must be one the armed-check
    test above reads, so a new such job cannot slip past it."""
    for name, job in _jobs().items():
        downloads = any(
            str(step.get("uses", "")).startswith("actions/download-artifact@")
            for step in job.get("steps", [])
        )
        if downloads and _has_oidc_write(job):
            assert name in DIGEST_CHECKS, f"{name} downloads an artifact with no digest check"


def _check_step(document: dict, job_name: str) -> dict:
    job = document["jobs"][job_name]
    return job["steps"][_check_index(job)]


def _edit_check(transform):
    def mutate(document: dict, job_name: str) -> None:
        step = _check_step(document, job_name)
        before = step["run"]
        step["run"] = transform(before)
        assert step["run"] != before, "the mutation changed nothing"

    return mutate


def _edit_check_line(match: str, transform):
    def edit(script: str) -> str:
        return "\n".join(transform(line) if match in line else line for line in script.splitlines())

    return _edit_check(edit)


def _set_on_check(key: str, value):
    return lambda document, job_name: _check_step(document, job_name).__setitem__(key, value)


def _set_defaults(on_job: bool, run: dict):
    def mutate(document: dict, job_name: str) -> None:
        (document["jobs"][job_name] if on_job else document)["defaults"] = {"run": run}

    return mutate


def _insert_after_check(document: dict, job_name: str) -> None:
    job = document["jobs"][job_name]
    job["steps"].insert(_check_index(job) + 1, {"name": "late", "run": "touch dist/late.whl"})


def _move_check_after_guarded(document: dict, job_name: str) -> None:
    steps = document["jobs"][job_name]["steps"]
    check = _check_index(document["jobs"][job_name])
    steps.insert(check + 1, steps.pop(check))


# Each mutant disarms the check, or lets files change after it, without deleting it.
_DISARMING_MUTANTS = {
    "exit 0 before the first command": _edit_check(lambda s: "exit 0\n" + s),
    "set +o errexit before the first command": _edit_check(lambda s: "set +o errexit\n" + s),
    "set +e before the first command": _edit_check(lambda s: "set +e\n" + s),
    "a trap that exits 0 on error": _edit_check(lambda s: "trap 'exit 0' ERR\n" + s),
    "the digest check's failure ignored": _edit_check_line(
        "sha256sum --check", lambda line: line + " || true"
    ),
    "digests taken from the download": _edit_check(
        lambda s: s.replace("printf '%s\\n' \"$EXPECTED\"", "sha256sum dist/*")
    ),
    "--ignore-missing": _edit_check(
        lambda s: s.replace("--check --strict", "--check --strict --ignore-missing")
    ),
    "diff of the expected list with itself": _edit_check_line(
        "diff ", lambda line: line.replace("downloaded-files", "expected-files")
    ),
    "if: always()": _set_on_check("if", "${{ always() }}"),
    "continue-on-error": _set_on_check("continue-on-error", True),
    "a step shell without -e": _set_on_check("shell", "bash {0}"),
    "a step working-directory": _set_on_check("working-directory", "elsewhere"),
    "a job defaults.run.shell without -e": _set_defaults(True, {"shell": "bash {0}"}),
    "a workflow defaults.run.shell without -e": _set_defaults(False, {"shell": "bash {0}"}),
    "a job defaults.run.working-directory": _set_defaults(True, {"working-directory": "x"}),
    "EXPECTED from another output": _set_on_check(
        "env", {"EXPECTED": "${{ needs.verify.outputs.artifact-sha256 }}"}
    ),
    "a step between the check and the guarded step": _insert_after_check,
    "the check after the guarded step": _move_check_after_guarded,
}


@pytest.mark.parametrize("job_name", sorted(DIGEST_CHECKS))
@pytest.mark.parametrize("mutant", sorted(_DISARMING_MUTANTS))
def test_digest_check_problems_catches_each_disarming_mutant(job_name: str, mutant: str) -> None:
    document = _document()
    _DISARMING_MUTANTS[mutant](document, job_name)
    assert digest_check_problems(document, job_name), f"{mutant} in {job_name} went unnoticed"


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
