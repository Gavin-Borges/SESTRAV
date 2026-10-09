"""Release jobs with OIDC authority must not resolve Python dependencies, the
job that builds the release artifacts must not run unhashed third-party code,
and every job that uses a downloaded artifact first checks it against the build
job's digests, with a check these tests fail if it carries one of the disarmings
they list. The digest-check tests are a regression ratchet: they refuse the
listed spellings, the list is not exhaustive, and nothing here runs the
workflow."""

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
    # Read as parsed pipelines, so a decoy in another command cannot satisfy a pin.
    pipelines = _pipelines(release["steps"][check]["run"])
    assert [["test", "-n", "$EXPECTED"]] in pipelines
    assert [_FROM_EXPECTED, ["sha256sum", "--check", "--strict", "-"]] in pipelines
    assert [["find", *_FIND_ARGS["release"]], ["sort", ">", _DOWNLOADED]] in pipelines
    assert pipelines[-1] == [["diff", _EXPECTED_FILES, _DOWNLOADED]]
    # The check must be able to fail the job, in the shape digest_check_problems allows.
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
    # Read as parsed pipelines, so a decoy in another command cannot satisfy a pin.
    pipelines = _pipelines(publish["steps"][check]["run"])
    digests = "$RUNNER_TEMP/dist.sha256"
    assert [_FROM_EXPECTED, ["grep", "-E", _DIST_PATTERN, ">", digests]] in pipelines
    assert [["test", "-s", digests]] in pipelines
    assert [["sha256sum", "--check", "--strict", digests]] in pipelines
    assert [["find", *_FIND_ARGS["publish"]], ["sort", ">", _DOWNLOADED]] in pipelines
    assert pipelines[-1] == [["diff", _EXPECTED_FILES, _DOWNLOADED]]
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
# awk's one allowed program: the file name of a `<digest>  <file>` line.
_PRINT_NAME = "{print $2}"
# What turns the lines sha256sum checked into the list of their file names.
_NAMES_OF = [["awk", _PRINT_NAME], ["sort"]]
# The one place a check may write: a plain file name directly in $RUNNER_TEMP, so
# no `..`, `/`, `$` or glob can carry the write elsewhere, into dist/ for one. A
# symlink planted in $RUNNER_TEMP is out of scope: no allowed command makes one.
_TEMP_FILE = re.compile(r"\$RUNNER_TEMP/[A-Za-z0-9][A-Za-z0-9._-]*")
# The only expansions a check may hold outside single quotes. Any other `$` or a
# backtick could run a command (`test -z "$(cp x dist/y)"`) or change a variable.
_ALLOWED_EXPANSION = re.compile(r"\$(?:EXPECTED|RUNNER_TEMP)(?![A-Za-z0-9_])")
_CHECK_ERREXIT_OFF = re.compile(r"\bset\s+(?:-\S*\s+)*\+[A-Za-z]*e|\bset\s+\+o\s+errexit\b")
# A `shell:` key, wherever it sits (workflow or job `defaults.run`, or the check
# step), must be one of GitHub's two documented spellings of errexit on: the
# default for a `run:` step with no `shell:`, and what `shell: bash` expands to.
# release.yml sets none today. Anything else (`bash -e +e {0}`, `env
# BASH_ENV=... bash -e {0}`) could turn errexit off or run a planted file first.
_ALLOWED_SHELLS = frozenset({"bash -e {0}", "bash --noprofile --norc -eo pipefail {0}"})
# Keys that, on the release or publish job, could make a failed check not fail
# the job or run it somewhere else. release.yml sets none of them on either job.
_FORBIDDEN_JOB_KEYS = ("continue-on-error", "container", "services", "strategy")
# What `find` lists in each job: exactly the directories the job downloads into,
# then `-type f`; no glob, no further path, no other test.
_FIND_ARGS = {
    "release": ["dist", "dist_release_bundle", "-type", "f"],
    "publish": ["dist", "-type", "f"],
}
# publish's one `grep -E` pattern: the `dist/` lines of the build job's digests.
_DIST_PATTERN = "^[0-9a-f]{64}  dist/[^/]+$"
_EXPECTED_FILES = "$RUNNER_TEMP/expected-files.txt"
_DOWNLOADED = "$RUNNER_TEMP/downloaded-files.txt"
# The `with:` of the step each check guards, exactly as release.yml has it: the
# attestation's subjects and, for the PyPI upload, no input at all.
GUARDED_WITH = {
    "release": {"subject-path": "dist/*.tar.gz,dist/*.whl,dist_release_bundle/*.zip"},
    "publish": {},
}


def _live_expansions(script: str) -> list[str]:
    """What bash would expand in `script` other than $EXPECTED and $RUNNER_TEMP:
    every `$` or backtick outside single quotes and not escaped by a backslash."""
    found, quote, i = [], "", 0
    while i < len(script):
        char = script[i]
        if quote == "'":
            quote = "" if char == "'" else quote
        elif char == "\\":
            i += 1
        elif char == '"':
            quote = "" if quote == '"' else '"'
        elif char == "'" and not quote:
            quote = "'"
        elif char == "`" or (char == "$" and not _ALLOWED_EXPANSION.match(script, i)):
            found.append(script[i : i + 16])
        i += 1
    return found


def _join_continuations(script: str) -> str:
    """Join a line to the next only where a backslash is the LAST character before
    the newline (an odd run of them), removing both, as sh does. A backslash
    followed by spaces, or by a blank line, joins nothing: the next line is a
    command of its own."""
    return re.sub(r"(?<!\\)((?:\\\\)*)\\\n", r"\1", script)


def _pipelines(script: str) -> list[list[list[str]]]:
    """Each line of a run script as its pipeline stages, each a list of shell words.

    Continuations are joined as sh joins them (_join_continuations), quotes are
    removed as the shell removes them, and
    every operator (`||`, `;`, `>`) is a word of its own. `#` is read as an
    ordinary character: shlex would start a comment at any `#`, even mid-word
    where bash does not (`b"#x || true` keeps its `|| true` in bash), so a check
    holding a comment is refused rather than half-read.
    """
    pipelines = []
    for line in _join_continuations(script).splitlines():
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.commenters = ""
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


def _shape_problem(stage: list[str], position: int, length: int, job_name: str) -> str:
    """Why one stage of a check's pipeline is outside its command's allowed shape, or "".

    No allowed shape writes a file, runs another command or drops what it reads
    without a later step noticing: printf prints EXPECTED and nothing else; grep
    filters its input with the one -E pattern release.yml uses (_DIST_PATTERN);
    awk runs only `{print $2}`, on its input or on one $RUNNER_TEMP file (no
    system(), print > or getline); sort takes no argument (no -o or --output);
    find takes exactly the directories its job downloads into and then `-type f`
    (_FIND_ARGS: no glob, no further path, no -exec, -fprint, -delete or test that
    skips a file); diff takes its two files and no option (no -I or
    --ignore-matching-lines); test stands alone and is one of the two forms
    release.yml has, `test -n "$EXPECTED"` or `test -s <$RUNNER_TEMP file>`.
    sha256sum is read by the caller.
    """
    name, args = stage[0], stage[1:]
    allowed = {
        "printf": stage == _FROM_EXPECTED and position == 0,
        "grep": position > 0 and args == ["-E", _DIST_PATTERN],
        "awk": args == [_PRINT_NAME]
        if position
        else len(args) == 2 and args[0] == _PRINT_NAME and args[1].startswith("$RUNNER_TEMP/"),
        "sort": position > 0 and not args,
        "find": position == 0 and args == _FIND_ARGS[job_name],
        "diff": length == 1 and len(args) == 2 and not any(arg.startswith("-") for arg in args),
        "test": length == 1
        and (
            stage == ["test", "-n", "$EXPECTED"]
            or (len(args) == 2 and args[0] == "-s" and bool(_TEMP_FILE.fullmatch(args[1])))
        ),
        "sha256sum": True,
    }
    return "" if allowed[name] else f"{name} outside its allowed shape: {stage}"


def digest_check_problems(document: dict, job_name: str) -> list[str]:
    """Why a job's digest check could let the step it guards run on unchecked files.

    The check has to take its expected digests from the build job's output and
    never from the download itself, fail the step on any mismatch or added file,
    and sit immediately before the step it guards, so nothing changes the files
    in between. Anything that skips the step, ignores its failure, runs it
    without -e, turns errexit off or exits early disarms it, and so does an `if`
    on the guarded step (`always()` runs it after a failed check). The check's
    env holds EXPECTED alone and neither its job nor the workflow sets env, so
    no BASH_ENV, PATH or RUNNER_TEMP reaches it from there. A `shell:` key on
    the check or in a job's or the workflow's defaults.run must be exactly
    `bash -e {0}` or `bash --noprofile --norc -eo pipefail {0}`; the job sets
    none of continue-on-error, container, services or strategy; and the guarded
    step's `with:` is exactly release.yml's (GUARDED_WITH).

    Its script expands nothing but $EXPECTED and $RUNNER_TEMP outside single
    quotes (no other `$`, no backtick), and may hold only these lines, each a
    plain pipeline of the allowed shapes (_shape_problem), `#` read as an
    ordinary character, writing, if at all, one `> "$RUNNER_TEMP/<name>"`
    with a plain name (no `/`, so no `..`):
    `test -n "$EXPECTED"` or `test -s <file>`; `sha256sum --check --strict -`
    fed by exactly `printf '%s\\n' "$EXPECTED"`, with no stage between (the
    release job), or `sha256sum --check --strict <file>` alone, on a file made
    from EXPECTED's lines and not rewritten since (the publish job); a list of
    the names of every digest that check read, made by `awk '{print $2}' |
    sort` from the same lines with nothing filtered out; `find` of exactly the
    job's _FIND_ARGS, `| sort`; and, as the last line, `diff` of those two
    lists.

    This is a regression ratchet: it refuses the spellings in
    _DISARMING_MUTANTS and _JOB_DISARMING_MUTANTS, each of which a test shows
    it refusing. The list is not exhaustive, it is not shown to refuse any
    other disarming, and nothing here runs the workflow.
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
    if _can_be_skipped_or_ignored(steps[guarded]):
        problems.append("the guarded step has an if or continue-on-error")
    if "working-directory" in step:
        problems.append("the check sets working-directory")
    if "shell" in step and str(step["shell"]) not in _ALLOWED_SHELLS:
        problems.append(f"the check's shell {step['shell']!r} is not one of {sorted(_ALLOWED_SHELLS)}")
    problems += [f"job {job_name} sets {key}" for key in _FORBIDDEN_JOB_KEYS if key in job]
    if (steps[guarded].get("with") or {}) != GUARDED_WITH[job_name]:
        problems.append(f"the guarded step's with: is not {GUARDED_WITH[job_name]}")
    if set(step.get("env") or {}) != {"EXPECTED"}:
        problems.append("the check's env sets more than EXPECTED (BASH_ENV, PATH, RUNNER_TEMP)")
    if _live_expansions(script):
        problems.append(f"the check expands more than $EXPECTED and $RUNNER_TEMP: {script!r}")
    for where, node in (("the workflow", document), (f"job {job_name}", job)):
        if "env" in node:
            problems.append(f"{where} sets env, which reaches the check")
        run = (node.get("defaults") or {}).get("run") or {}
        if "shell" in run and str(run["shell"]) not in _ALLOWED_SHELLS:
            problems.append(f"{where} defaults.run.shell {run['shell']!r} is not an allowed shell")
        if "working-directory" in run:
            problems.append(f"{where} defaults.run.working-directory moves the check")
    if _CHECK_ERREXIT_OFF.search(script):
        problems.append("the check turns errexit off")
    if re.search(r"\bexit\b", script):
        problems.append("the check can exit before it fails")

    from_expected: set[str] = set()  # files holding only lines of EXPECTED
    verified: set[str] = set()  # what a sha256sum --check --strict read in full
    names: set[str] = set()  # files listing the names of every digest verified
    listed: set[str] = set()  # files listing every file find saw
    checked = compared = False
    pipelines = _pipelines(script)
    for stages in pipelines:
        output = None
        if len(stages[-1]) >= 2 and stages[-1][-2] == ">":
            output, stages[-1] = stages[-1][-1], stages[-1][:-2]
        if any(
            not stage or stage[0] not in _CHECK_COMMANDS or any(map(_OPERATOR.fullmatch, stage))
            for stage in stages
        ):
            problems.append(f"not a plain pipeline of {sorted(_CHECK_COMMANDS)}: {stages}")
            continue
        if output is not None and not _TEMP_FILE.fullmatch(output):
            problems.append(f"the check writes other than a plain file in $RUNNER_TEMP: {output}")
        shapes = (_shape_problem(stage, i, len(stages), job_name) for i, stage in enumerate(stages))
        problems += [shape for shape in shapes if shape]
        first = stages[0]
        reads_expected = first == _FROM_EXPECTED or (
            bool(_temp_files(first)) and _temp_files(first) <= from_expected
        )
        for stage in stages:
            if stage[0] != "sha256sum":
                continue
            if stage[1:-1] != ["--check", "--strict"]:
                problems.append(f"sha256sum computes digests or checks loosely: {stage}")
            elif stage[-1] == "-" and stages == [_FROM_EXPECTED, stage]:
                checked = True
                verified.add("$EXPECTED")
            elif stages == [stage] and stage[-1] in from_expected:
                checked = True
                verified.add(stage[-1])
            else:
                problems.append(f"sha256sum checks digests that are not all EXPECTED's: {stage}")
        if first[0] == "diff" and len(first) == 3:
            a, b = first[1:]
            compared = compared or (a in names and b in listed) or (b in names and a in listed)
        if output is not None:
            names_of_verified = (
                stages == [_FROM_EXPECTED, *_NAMES_OF] and "$EXPECTED" in verified
            ) or (
                len(first) == 3 and [first[:2], *stages[1:]] == _NAMES_OF and first[2] in verified
            )
            for group, member in (
                (from_expected, reads_expected),
                (names, names_of_verified),
                (listed, first[0] == "find" and stages[1:] == [["sort"]]),
            ):
                (group.add if member else group.discard)(output)
            verified.discard(output)
    if not checked:
        problems.append("no sha256sum --check reads the digests EXPECTED holds")
    if not compared:
        problems.append("no diff compares the names of every checked digest with what find lists")
    if not pipelines or pipelines[-1][0][0] != "diff":
        problems.append("the check's last command is not its diff")
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


def _append_to_check(line: str):
    return _edit_check(lambda script: script.rstrip("\n") + "\n" + line)


def _set_on_guarded(key: str, value):
    def mutate(document: dict, job_name: str) -> None:
        job = document["jobs"][job_name]
        prefix = DIGEST_CHECKS[job_name]
        guarded = _step_index(job, lambda s: str(s.get("uses", "")).startswith(prefix))
        job["steps"][guarded][key] = value

    return mutate


def _set_on_job(key: str, value):
    return lambda document, job_name: document["jobs"][job_name].__setitem__(key, value)


def _set_shell(where: str, shell: str):
    """Put `shell:` on the check step, a job's defaults.run or the workflow's."""
    if where == "step":
        return _set_on_check("shell", shell)
    return _set_defaults(where == "job", {"shell": shell})


def _set_env(on_job: bool, env: dict):
    def mutate(document: dict, job_name: str) -> None:
        (document["jobs"][job_name] if on_job else document)["env"] = env

    return mutate


def _before_the_diff(line: str):
    """Insert a line after find has listed the download and before diff compares it."""

    def edit(script: str) -> str:
        lines = script.rstrip("\n").split("\n")
        assert lines[-1].startswith("diff "), lines[-1]
        return "\n".join([*lines[:-1], line, lines[-1]])

    return _edit_check(edit)


_FROM_EXPECTED_TEXT = "printf '%s\\n' \"$EXPECTED\""
_DIST_LINES = "grep -E '^[0-9a-f]{64}  dist/[^/]+$'"
_SDIST_LINES = "grep -E '^[0-9a-f]{64}  dist/[^/]+[.]tar[.]gz$'"
_PUBLISH_DIGESTS = '"$RUNNER_TEMP/dist.sha256"'
_LATE = 'sort -o dist/late.whl "$RUNNER_TEMP/expected-files.txt"'
# From $RUNNER_TEMP (/home/runner/work/_temp) up into the workspace's dist/.
_THROUGH_TEMP = '"$RUNNER_TEMP/../SESTRAV/SESTRAV/dist/'
_NAMES_INTO = _FROM_EXPECTED_TEXT + " | awk '{print $2}' | sort > " + _THROUGH_TEMP
# A file in the download that bash sources first and that stubs the checks out.
_BASH_ENV = {"BASH_ENV": "dist/x.sh"}


def _publish_names_from_expected(script: str) -> str:
    """Check the sdist's digest only, but list every dist/ name straight from EXPECTED."""
    narrowed = script.replace(_DIST_LINES + " >", _SDIST_LINES + " >")
    return narrowed.replace(
        "awk '{print $2}' " + _PUBLISH_DIGESTS + " | sort",
        _FROM_EXPECTED_TEXT + " | " + _DIST_LINES + " | awk '{print $2}' | sort",
    )


def _publish_digests_rewritten(script: str) -> str:
    """Check the sdist's digest only, then rewrite the checked file with every dist/ line."""
    narrowed = script.replace(_DIST_LINES + " >", _SDIST_LINES + " >")
    check = "sha256sum --check --strict " + _PUBLISH_DIGESTS
    rewrite = _FROM_EXPECTED_TEXT + " | " + _DIST_LINES + " > " + _PUBLISH_DIGESTS
    return narrowed.replace(check, check + "\n" + rewrite)


# Each mutant is a spelling the helper must refuse, applied without deleting the
# check. Most disarm it or let a file change after it; a few (an extra `find`
# directory, a widened `grep`, an unrelated `with:`) only widen what it allows.
# The table is a regression ratchet, not a list of every way to disarm a check.
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
    # Added after review: each passed the earlier helper, and each, run under bash
    # -e (GNU coreutils 9.7 or uutils 0.8.0 alike), lets a tampered or extra file
    # through the check or writes a file into dist/ after find has listed it.
    "diff -I.": _edit_check_line("diff ", lambda line: line.replace("diff ", "diff -I. ")),
    "diff --ignore-matching-lines=.": _edit_check_line(
        "diff ", lambda line: line.replace("diff ", "diff --ignore-matching-lines=. ")
    ),
    "sort -o into dist/ after the diff": _append_to_check(_LATE),
    "awk system() copying into dist/ after the diff": _append_to_check(
        "awk 'BEGIN{system(\"cp /etc/hostname dist/late.whl\")}'"
    ),
    "sort -o into dist/ between find and diff": _before_the_diff(_LATE),
    "a redirection into dist/ between find and diff": _before_the_diff(
        _FROM_EXPECTED_TEXT + " | awk '{print $2}' | sort > dist/late.whl"
    ),
    "find skipping a file with -not -name": _edit_check_line(
        "find ", lambda line: line.replace("-type f", "-type f -not -name 'late*'")
    ),
    "grep -v dropping a file from find's list": _edit_check_line(
        "find ", lambda line: line.replace("-type f |", "-type f | grep -v late |")
    ),
    # Added after a second review: each passed the helper before it. The first
    # five are the review's own; the other five are the same kinds.
    "if: always() on the guarded step": _set_on_guarded("if", "${{ always() }}"),
    "a # hiding || true after sha256sum": _edit_check_line(
        "sha256sum --check", lambda line: line + "# || true"
    ),
    "a # hiding || true after the diff": _edit_check_line(
        "diff ", lambda line: line + "#x || true"
    ),
    "a write through $RUNNER_TEMP/.. creating dist/late.whl": _before_the_diff(
        _NAMES_INTO + 'late.whl"'
    ),
    "a write through $RUNNER_TEMP/.. over the checked sdist": _before_the_diff(
        _NAMES_INTO + 'sestrav-1.0.tar.gz"'
    ),
    "test running $(cp ...) into dist/": _before_the_diff(
        'test -z "$(cp /etc/hostname dist/late.whl)"'
    ),
    "test running a backtick cp into dist/": _before_the_diff(
        'test -z "`cp /etc/hostname dist/late.whl`"'
    ),
    "BASH_ENV in the check's env": _set_on_check("env", {"EXPECTED": _EXPECTED, **_BASH_ENV}),
    "BASH_ENV in the job's env": _set_env(True, _BASH_ENV),
    "BASH_ENV in the workflow's env": _set_env(False, _BASH_ENV),
    # Added after a third review: each passed the helper before it.
    # A backslash joins the next line only when it ends the line (sh), so a blank
    # line or trailing spaces after one do not hide the command that follows.
    "a backslash and a blank line before e''xit 0": _edit_check(
        lambda s: 'test -n "$EXPECTED" \\\n\ne\'\'xit 0\n' + s
    ),
    "a backslash and a trailing space before e''xit 0": _edit_check(
        lambda s: 'test -n "$EXPECTED" \\ \ne\'\'xit 0\n' + s
    ),
    # A `test` takes one of the two forms release.yml has, and `find` lists exactly
    # its job's directories, so a decoy `test -n "find dist -type f"` pins nothing.
    "test with further arguments": _before_the_diff('test -n "$EXPECTED" -o -z "$EXPECTED"'),
    "find narrowed to globs, with a decoy test": _edit_check(
        lambda s: s.replace("find dist", "find dist/*.tar.gz dist/*.whl").replace(
            "diff ", 'test -n "find dist -type f"\ndiff ', 1
        )
    ),
    "find with an extra directory": _edit_check_line(
        "find ", lambda line: line.replace("-type f", "/srv -type f")
    ),
    # A shell: key, wherever it sits, is one of GitHub's two spellings or nothing.
    **{
        f"shell {shell!r} at the {where}": _set_shell(where, shell)
        for where in ("workflow", "job", "step")
        for shell in ("bash -e +e {0}", "bash -e +o errexit {0}", "env BASH_ENV=dist/x.sh bash -e {0}")
    },
    # Job keys that could stop a failed check failing the job.
    "job continue-on-error": _set_on_job("continue-on-error", True),
    "job container": _set_on_job("container", "python:3"),
    "job services": _set_on_job("services", {"x": {"image": "y"}}),
    "job strategy": _set_on_job("strategy", {"matrix": {"a": [1]}}),
    "an unrelated with: on the guarded step": _set_on_guarded("with", {"x": "y"}),
}

# Disarmings that only one job's check can carry.
_JOB_DISARMING_MUTANTS = {
    "release": {
        "grep -v whl between printf and sha256sum": _edit_check(
            lambda s: s.replace(
                _FROM_EXPECTED_TEXT + " | sha256sum",
                _FROM_EXPECTED_TEXT + " | grep -v whl | sha256sum",
            )
        ),
        "grep -E picking the non-wheel lines between printf and sha256sum": _edit_check(
            lambda s: s.replace(
                _FROM_EXPECTED_TEXT + " | sha256sum",
                _FROM_EXPECTED_TEXT + " | grep -E 'gz$|zip$|json$' | sha256sum",
            )
        ),
        "the attestation's subject-path pointed elsewhere": _set_on_guarded(
            "with", {"subject-path": "other/*.whl"}
        ),
        "the attestation's subject-path narrowed to the wheels": _set_on_guarded(
            "with", {"subject-path": "dist/*.whl"}
        ),
    },
    "publish": {
        "the sdist's digest checked, every name listed from EXPECTED": _edit_check(
            _publish_names_from_expected
        ),
        "the checked digests rewritten after the check": _edit_check(_publish_digests_rewritten),
        "grep -E widened to the dist/ prefix": _edit_check(
            lambda s: s.replace("dist/[^/]+$'", "dist/'")
        ),
        "the upload's packages-dir pointed elsewhere": _set_on_guarded(
            "with", {"packages-dir": "other/"}
        ),
        "the upload told to skip existing files": _set_on_guarded("with", {"skip-existing": True}),
    },
}


@pytest.mark.parametrize("job_name", sorted(DIGEST_CHECKS))
@pytest.mark.parametrize("mutant", sorted(_DISARMING_MUTANTS))
def test_digest_check_problems_catches_each_disarming_mutant(job_name: str, mutant: str) -> None:
    document = _document()
    _DISARMING_MUTANTS[mutant](document, job_name)
    assert digest_check_problems(document, job_name), f"{mutant} in {job_name} went unnoticed"


@pytest.mark.parametrize(
    ("job_name", "mutant"),
    [(job, mutant) for job, mutants in _JOB_DISARMING_MUTANTS.items() for mutant in mutants],
)
def test_digest_check_problems_catches_each_job_disarming_mutant(
    job_name: str, mutant: str
) -> None:
    document = _document()
    _JOB_DISARMING_MUTANTS[job_name][mutant](document, job_name)
    assert digest_check_problems(document, job_name), f"{mutant} in {job_name} went unnoticed"


@pytest.mark.parametrize("job_name", sorted(DIGEST_CHECKS))
@pytest.mark.parametrize("shell", sorted(_ALLOWED_SHELLS))
@pytest.mark.parametrize("where", ["workflow", "job", "step"])
def test_the_two_documented_errexit_shells_are_allowed(job_name: str, shell: str, where: str) -> None:
    """Negative control for the shell mutants: the same keys with an allowed value pass."""
    document = _document()
    _set_shell(where, shell)(document, job_name)
    assert not digest_check_problems(document, job_name)


@pytest.mark.parametrize(
    ("script", "joined"),
    [
        ("a \\\nb", "a b"),
        ("a\\\nb", "ab"),
        ("a \\\n\nb", "a \nb"),
        ("a \\ \nb", "a \\ \nb"),
        ("a \\  \n\nb", "a \\  \n\nb"),
        ("a \\\\\nb", "a \\\\\nb"),
        ("a \\\\\\\nb", "a \\\\b"),
    ],
)
def test_continuations_join_only_where_a_backslash_ends_the_line(script: str, joined: str) -> None:
    assert _join_continuations(script) == joined


def test_a_blank_line_after_a_backslash_starts_a_new_command() -> None:
    script = 'test -n "$EXPECTED" \\\n\ne\'\'xit 0\n'
    assert _pipelines(script) == [[["test", "-n", "$EXPECTED"]], [["exit", "0"]]]


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
