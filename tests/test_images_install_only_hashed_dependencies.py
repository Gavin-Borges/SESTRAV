"""Every container image installs only hash-pinned dependencies, and only its own.

Dockerfile.api and Dockerfile.demo used to finish with `pip install ".[api]"` and
`pip install ".[demo]"`, which resolved the package's whole dependency tree from
PyPI at build time with no hashes. The production Dockerfile had already moved
to a hashed lock; the two smaller images had not, and nothing checked. These
tests make every image's install sequence a gate:

- each `pip install` in a RUN is either a hash-pinned `-r <lock>` install with
  `--require-hashes --no-deps`, or the package itself (`.`, no extras) with
  `--no-deps --no-build-isolation`, so nothing is resolved or built unpinned;
- every file a Dockerfile COPYs exists and survives .dockerignore (a lock the
  build context never receives fails `docker build`, which no CI job runs);
- each image's locks pin every [project].dependencies requirement and the
  requirements of the extra it serves, within their specifiers;
- the API and demo images' specs name exactly those requirements (plus
  setuptools, their --no-build-isolation build backend), each at the version
  requirements.txt pins, and their compiled locks share every package with
  requirements.txt at the same version, so every package an image shares with
  requirements.txt runs at the version CI tests, and the specs cannot drift
  from requirements.txt unnoticed;
- each of those images installs its own lock and no other.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from tools import update_dependencies
from tools.update_dependencies import LOCK_SPECS, build_command, preference_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The extra each image serves. A new Dockerfile fails
# test_every_dockerfile_is_classified until it is added here.
IMAGE_EXTRAS: dict[str, tuple[str, ...]] = {
    "Dockerfile": (),
    "Dockerfile.api": ("api",),
    "Dockerfile.demo": ("demo",),
}
# The images whose lock is compiled from a dedicated spec pinned to requirements.txt.
IMAGE_SPECS = {
    "Dockerfile.api": "environments/requirements-api.in",
    "Dockerfile.demo": "environments/requirements-demo.in",
}
BUILD_BACKEND = "setuptools"
PIP_BOOTSTRAP = "environments/requirements-pip-bootstrap.txt"
RUNTIME_LOCK = "requirements.txt"
PRODUCTION_LOCK = "environments/requirements.lock"

_PIP_INSTALL = re.compile(r"\bpip(?:3(?:\.\d+)?)?\s+install\b")
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==([^\s\\;]+)")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _read(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def instructions(text: str) -> list[str]:
    """Dockerfile instructions with continuations joined and comment lines dropped."""
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    joined = re.sub(r"\\\s*\n", " ", "\n".join(lines))
    return [i.strip() for i in joined.splitlines() if i.strip()]


def pip_installs(text: str) -> list[list[str]]:
    """The argument list of every `pip install` in the Dockerfile's shell-form RUNs."""
    installs = []
    for instruction in instructions(text):
        if not instruction.startswith("RUN "):
            continue
        for command in re.split(r"&&|\|\||;", instruction[len("RUN ") :]):
            match = _PIP_INSTALL.search(command)
            if match:
                installs.append(command[match.end() :].split())
    return installs


def unparsed_pip_runs(text: str) -> list[str]:
    """RUNs pip_installs cannot read: exec form mentioning pip, and every heredoc.

    A heredoc's body sits on the lines after `RUN <<EOF`, so whether it runs pip
    cannot be read off the instruction; none is used here, so all are refused.
    """
    return [
        i
        for i in instructions(text)
        if i.startswith("RUN ") and ("<<" in i or ("pip" in i and i[4:].lstrip().startswith("[")))
    ]


def _requirement_files(args: list[str]) -> list[str]:
    return [args[i + 1] for i, arg in enumerate(args[:-1]) if arg in ("-r", "--requirement")]


def install_problems(args: list[str]) -> list[str]:
    """Why a `pip install` argument list is not a hashed-lock or no-deps local install."""
    files = _requirement_files(args)
    if files:
        missing = [flag for flag in ("--require-hashes", "--no-deps") if flag not in args]
        return [f"`-r {' '.join(files)}` lacks {flag}" for flag in missing]
    targets = [a for a in args if not a.startswith("-")]
    if targets != ["."]:
        return [f"installs {targets} rather than a hashed lock or the local package"]
    return [f"`.` lacks {flag}" for flag in ("--no-deps", "--no-build-isolation") if flag not in args]


def copy_sources(text: str) -> list[str]:
    """Every source path of every COPY/ADD instruction (the last token is the dest)."""
    sources = []
    for instruction in instructions(text):
        keyword, _, rest = instruction.partition(" ")
        if keyword not in ("COPY", "ADD"):
            continue
        tokens = [t for t in rest.split() if not t.startswith("--")]
        sources += [t.rstrip("/") for t in tokens[:-1]]
    return sources


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    """Docker's glob: `*` and `?` stop at `/`; `**/` spans zero or more directories."""
    out = ""
    for part in re.split(r"(\*\*/|\*\*|\*|\?)", pattern):
        out += {"**/": "(?:.*/)?", "**": ".*", "*": "[^/]*", "?": "[^/]"}.get(part, re.escape(part))
    return re.compile(out + r"\Z")


def dockerignore_excludes(path: str, rules: list[str]) -> bool:
    """Docker's rule: the last pattern matching the path or one of its parents wins."""
    parts = path.strip("/").split("/")
    candidates = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    excluded = False
    for rule in rules:
        rule = rule.strip()
        if not rule or rule.startswith("#"):
            continue
        negated = rule.startswith("!")
        regex = _pattern_regex(rule.lstrip("!").strip("/"))
        if any(regex.match(candidate) for candidate in candidates):
            excluded = not negated
    return excluded


def _pins(relative: str) -> dict[str, str]:
    pins = {}
    for line in _read(relative).splitlines():
        match = _PIN.match(line)
        if match:
            pins[_canonical(match.group(1))] = match.group(2)
    return pins


def _declared(extras: tuple[str, ...]) -> list[Requirement]:
    project = tomllib.loads(_read("pyproject.toml"))["project"]
    declared = [Requirement(r) for r in project["dependencies"]]
    for extra in extras:
        declared += [Requirement(r) for r in project["optional-dependencies"][extra]]
    return declared


def _image_locks(dockerfile: str) -> list[str]:
    return [
        lock
        for args in pip_installs(_read(dockerfile))
        for lock in _requirement_files(args)
        if lock != PIP_BOOTSTRAP
    ]


def test_every_dockerfile_is_classified() -> None:
    found = {p.name for p in PROJECT_ROOT.glob("Dockerfile*") if p.is_file()}
    assert found == set(IMAGE_EXTRAS), found ^ set(IMAGE_EXTRAS)


@pytest.mark.parametrize("dockerfile", sorted(IMAGE_EXTRAS))
def test_every_pip_install_is_hash_pinned_or_the_local_package(dockerfile: str) -> None:
    text = _read(dockerfile)
    installs = pip_installs(text)
    problems = [p for args in installs for p in install_problems(args)]
    problems += [f"pip in an exec-form or heredoc RUN: {r}" for r in unparsed_pip_runs(text)]
    assert not problems, f"{dockerfile}: {problems}"
    assert len(installs) >= 3, f"{dockerfile}: expected bootstrap, lock and package installs"


@pytest.mark.parametrize("dockerfile", sorted(IMAGE_EXTRAS))
def test_every_copied_path_exists_and_reaches_the_build_context(dockerfile: str) -> None:
    rules = _read(".dockerignore").splitlines()
    sources = copy_sources(_read(dockerfile))
    missing = [s for s in sources if not (PROJECT_ROOT / s).exists()]
    ignored = [s for s in sources if dockerignore_excludes(s, rules)]
    assert not missing and not ignored, f"{dockerfile}: missing={missing} ignored={ignored}"


@pytest.mark.parametrize("dockerfile", sorted(IMAGE_EXTRAS))
def test_every_installed_lock_is_copied_into_the_image(dockerfile: str) -> None:
    sources = set(copy_sources(_read(dockerfile)))
    for args in pip_installs(_read(dockerfile)):
        for lock in _requirement_files(args):
            assert lock in sources, f"{dockerfile} installs {lock} without copying it in"


@pytest.mark.parametrize(("dockerfile", "extras"), sorted(IMAGE_EXTRAS.items()))
def test_the_image_locks_pin_every_declared_requirement(
    dockerfile: str, extras: tuple[str, ...]
) -> None:
    locks = _image_locks(dockerfile)
    assert locks, f"{dockerfile} installs no dependency lock"
    pins: dict[str, str] = {}
    for lock in locks:
        pins.update(_pins(lock))
    problems = []
    for requirement in _declared(extras):
        name = _canonical(requirement.name)
        if name not in pins:
            problems.append(f"{name} is not pinned")
        elif Version(pins[name]) not in requirement.specifier:
            problems.append(f"{name}=={pins[name]} is outside {requirement.specifier}")
    assert not problems, f"{dockerfile} ({', '.join(locks)}): {problems}"


@pytest.mark.parametrize(("dockerfile", "spec"), sorted(IMAGE_SPECS.items()))
def test_image_spec_names_exactly_the_declared_requirements_at_runtime_versions(
    dockerfile: str, spec: str
) -> None:
    wanted = {_canonical(r.name) for r in _declared(IMAGE_EXTRAS[dockerfile])} | {BUILD_BACKEND}
    spec_pins = _pins(spec)
    assert set(spec_pins) == wanted, {"extra": set(spec_pins) - wanted, "missing": wanted - set(spec_pins)}
    runtime, production = _pins(RUNTIME_LOCK), _pins(PRODUCTION_LOCK)
    drift = {}
    for name, version in spec_pins.items():
        reference = runtime.get(name) or production.get(name)
        if reference is not None and reference != version:
            drift[name] = (version, reference)
    assert not drift, f"{spec} pins differ from requirements.txt (or the lock): {drift}"


@pytest.mark.parametrize("dockerfile", sorted(IMAGE_SPECS))
def test_image_lock_shares_every_runtime_package_at_the_same_version(dockerfile: str) -> None:
    runtime = _pins(RUNTIME_LOCK)
    for lock in _image_locks(dockerfile):
        text = _read(lock)
        assert "--hash=" in text, f"{lock} is not a hashed lock"
        pins = _pins(lock)
        drift = {n: (v, runtime[n]) for n, v in pins.items() if n in runtime and runtime[n] != v}
        assert not drift, f"{lock} runs versions requirements.txt does not: {drift}"
        assert len(set(pins) & set(runtime)) >= 50, f"{lock} shares implausibly few packages"


def test_image_specs_are_managed_and_seeded_from_the_runtime_lock() -> None:
    by_source = {spec.source: spec for spec in LOCK_SPECS}
    for spec_path in IMAGE_SPECS.values():
        spec = by_source[spec_path]
        assert spec.python_version == "3.13", spec.name
        assert spec.name in update_dependencies.RUNTIME_SPEC_NAMES
        assert update_dependencies.SEEDED_SPECS[spec.name] == RUNTIME_LOCK


@pytest.mark.parametrize(("dockerfile", "spec_path"), sorted(IMAGE_SPECS.items()))
def test_each_image_installs_its_own_lock_and_no_other(dockerfile: str, spec_path: str) -> None:
    # Without this, Dockerfile.api could install the demo lock (streamlit, shap,
    # pyarrow) and every other check would still pass.
    spec = {s.source: s for s in LOCK_SPECS}[spec_path]
    assert _image_locks(dockerfile) == [spec.output], (dockerfile, _image_locks(dockerfile))


def test_seeded_specs_are_self_contained() -> None:
    for spec in LOCK_SPECS:
        if spec.name in update_dependencies.SEEDED_SPECS:
            lines = [line.strip() for line in _read(spec.source).splitlines()]
            includes = [ln for ln in lines if ln.startswith(("-r", "-c", "--requirement", "--constraint"))]
            assert not includes, (spec.source, includes)


@pytest.mark.parametrize(
    ("line", "problem_count"),
    [
        ("RUN pip install --user --require-hashes --no-deps -r requirements.txt", 0),
        ("RUN pip install --user --no-deps --no-build-isolation .", 0),
        ('RUN pip install --user ".[api]"', 1),
        ("RUN pip install --user --no-deps .", 1),
        ("RUN pip install --user -r requirements.txt", 2),
        ("RUN pip install fastapi", 1),
        ("RUN pip3 install --require-hashes -r x.txt", 1),
        ("RUN python -m pip install --no-deps --no-build-isolation .", 0),
    ],
)
def test_install_problems_classifies_each_shape(line: str, problem_count: int) -> None:
    [args] = pip_installs(line)
    assert len(install_problems(args)) == problem_count, install_problems(args)


def test_continuations_chains_and_unparseable_runs() -> None:
    text = (
        "# RUN pip install ignored-because-commented\n"
        "RUN pip install --user --require-hashes --no-deps \\\n"
        "        -r a.txt && \\\n"
        '    pip install --user ".[api]"\n'
        'RUN ["pip", "install", "x"]\n'
        "RUN <<EOF\npip install y\nEOF\n"
    )
    installs = pip_installs(text)
    assert len(installs) == 2
    assert install_problems(installs[0]) == []
    assert install_problems(installs[1])
    assert len(unparsed_pip_runs(text)) == 2


def test_copy_sources_and_dockerignore_follow_docker_rules() -> None:
    text = "COPY --chown=u:u pyproject.toml README.md ./\nCOPY src/ ./src/\n"
    assert copy_sources(text) == ["pyproject.toml", "README.md", "src"]
    rules = ["docs/", "*.md", "!README.md", "environments/*", "!environments/keep.txt"]
    assert dockerignore_excludes("docs/a.txt", rules)
    assert dockerignore_excludes("NOTES.md", rules)
    assert not dockerignore_excludes("README.md", rules)
    assert dockerignore_excludes("environments/other.txt", rules)
    assert not dockerignore_excludes("environments/keep.txt", rules)
    assert not dockerignore_excludes("src/x.py", rules)
    assert dockerignore_excludes("x.pyc", ["**/*.pyc"])
    assert dockerignore_excludes("a/b/x.pyc", ["**/*.pyc"])


def test_preference_text_prefers_the_seed_and_keeps_the_rest(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("numpy==2.4.6 \\\n    --hash=sha256:a\nfoo==1.0\n")
    lock = tmp_path / "environments" / "requirements-demo.txt"
    lock.parent.mkdir()
    lock.write_text("numpy==2.0.0 \\\n    --hash=sha256:b\nstreamlit==1.64.0\n")
    spec = next(s for s in LOCK_SPECS if s.name == "demo")
    assert preference_text(spec, root=tmp_path) == "foo==1.0\nnumpy==2.4.6\nstreamlit==1.64.0\n"
    assert "numpy==2.0.0" in lock.read_text(), "preference_text must not write"


def _seeded_tree(tmp_path: Path):
    spec = next(s for s in LOCK_SPECS if s.name == "api")
    (tmp_path / "requirements.txt").write_text("numpy==2.4.6\n")
    (tmp_path / spec.source).parent.mkdir(parents=True)
    (tmp_path / spec.source).write_text("numpy==2.4.6\n")
    (tmp_path / spec.output).write_text("numpy==2.4.6 \\\n    --hash=sha256:old\n")
    return spec


@pytest.mark.parametrize("returncode", [0, 1])
def test_a_compile_that_writes_nothing_leaves_the_real_lock_untouched(
    tmp_path: Path, monkeypatch, returncode: int
) -> None:
    # A stubbed or failed compile must not leave the preference list behind: an
    # earlier draft seeded the real output in place, and a test that stubbed
    # subprocess.run replaced both image locks with unhashed preference lists.
    spec = _seeded_tree(tmp_path)
    monkeypatch.setattr(
        update_dependencies.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, returncode),
    )
    before = (tmp_path / spec.output).read_text()
    assert update_dependencies._compile_seeded(spec, build_command(spec), root=tmp_path) == returncode
    assert (tmp_path / spec.output).read_text() == before


def test_a_real_compile_is_installed_with_the_same_relative_argv(tmp_path: Path, monkeypatch) -> None:
    spec = _seeded_tree(tmp_path)
    seen = {}

    def fake_uv(command, cwd, **kwargs):
        seen["preferences"] = (Path(cwd) / spec.output).read_text()
        (Path(cwd) / spec.output).write_text("numpy==2.4.6 \\\n    --hash=sha256:new\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(update_dependencies.subprocess, "run", fake_uv)
    command = build_command(spec)
    assert update_dependencies._compile_seeded(spec, command, root=tmp_path) == 0
    assert seen["preferences"] == "numpy==2.4.6\n"
    assert "--hash=sha256:new" in (tmp_path / spec.output).read_text()
    assert command[command.index("--output-file") + 1] == spec.output


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], "foo==9.0\nnumpy==2.4.6\n"),
        (["--upgrade"], "numpy==2.4.6\n"),
        (["--upgrade-package", "foo"], "numpy==2.4.6\n"),
        (["--upgrade-package", "numpy"], "foo==9.0\nnumpy==2.4.6\n"),
    ],
)
def test_upgrade_flags_become_preference_rules_and_never_reach_uv(
    tmp_path: Path, monkeypatch, flags: list[str], expected: str
) -> None:
    # uv ignores preferences under --upgrade (and a package's own under
    # --upgrade-package), which let a re-lock float the image locks off the seed.
    spec = _seeded_tree(tmp_path)
    (tmp_path / spec.output).write_text(
        "foo==9.0 \\\n    --hash=sha256:x\nnumpy==1.0.0 \\\n    --hash=sha256:y\n"
    )
    seen = {}

    def fake_uv(command, cwd, **kwargs):
        seen["command"] = command
        seen["preferences"] = (Path(cwd) / spec.output).read_text()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(update_dependencies.subprocess, "run", fake_uv)
    command = build_command(spec) + flags
    update_dependencies._compile_seeded(spec, command, root=tmp_path)
    assert "--upgrade" not in seen["command"] and "--upgrade-package" not in seen["command"]
    assert seen["preferences"] == expected


@pytest.mark.parametrize(
    "written",
    [b"numpy==2.4.6 \\\n    --hash=sha256:lf\n", b"numpy==2.4.6 \\\r\n    --hash=sha256:crlf\r\n"],
    ids=["lf", "crlf"],
)
def test_the_installed_lock_is_the_bytes_uv_wrote(tmp_path: Path, monkeypatch, written: bytes) -> None:
    # A text-mode copy changes one of these on every platform: on Windows it
    # writes uv's LF back as CRLF, and on Linux it reads CRLF back as LF. Each
    # case alone passes a text-mode copy on one of the two.
    spec = _seeded_tree(tmp_path)

    def fake_uv(command, cwd, **kwargs):
        (Path(cwd) / spec.output).write_bytes(written)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(update_dependencies.subprocess, "run", fake_uv)
    update_dependencies._compile_seeded(spec, build_command(spec), root=tmp_path)
    assert (tmp_path / spec.output).read_bytes() == written
