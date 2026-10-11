"""Every container image installs only hash-pinned dependencies, and only its own.

Dockerfile.api and Dockerfile.demo used to finish with `pip install ".[api]"` and
`pip install ".[demo]"`, which resolved the package's whole dependency tree from
PyPI at build time with no hashes. The production Dockerfile had already moved
to a hashed lock; the two smaller images had not, and nothing checked. These
tests make every image's install sequence a gate:

- each `pip install` in a RUN is a hash-pinned `-r <lock>` install with
  `--require-hashes --no-deps` (and `--no-build-isolation`, except the pip
  bootstrap), the build backend alone with `--require-hashes --no-deps` and the
  image's lock as a `-c` constraints file (which supplies its version and
  hashes), or the package itself (`.`, no extras) with `--no-deps
  --no-build-isolation`, so nothing is resolved or built unpinned;
- that backend install follows the pip bootstrap and precedes the lock install
  and the package install, so every build, including a lock's sdist-only pin
  (connection-pool in requirements.lock), runs against a hash-checked setuptools
  rather than one an isolated build environment downloads unhashed, and the
  setuptools each image's lock pins satisfies pyproject.toml's
  [build-system].requires, which pip does not check without build isolation;
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
- each of those images installs its own lock and no other;
- singularity.def's %post runs the Dockerfile's first three installs (bootstrap,
  backend, lock), in that order, from files its %files section copies in, and
  opens with a `set -e` it sets itself; no section runs pip install outside
  %post, and no uncommented line runs another installer;
- %post is checked against a fixed list of spellings that let a build carry on
  past a failing command or end at status 0, measured under dash 0.5.12 (and,
  for `source`, which dash lacks, under bash; `false; true` stops, and `;` is
  refused only so that each command stands on its own line). This is a
  regression ratchet that refuses the listed spellings: the list is not
  exhaustive, it is not a proof that no other disarming exists, and nothing
  here executes %post or builds the image. The list: a `set` that turns errexit
  off (`+o errexit`, or a `+` flag group holding e, such as `+e`, `+eu` or
  `+ex`, read with the line's quotes and backslashes removed), or a `set`
  followed by a `$` expansion; a backtick or a `<<`; and, outside quotes and
  `$(...)` substitutions, an AND-OR list, a pipeline or a `;` list, a command
  negated with `!`, run in the background (`&`, or `&>`, which dash reads as
  one), run as an `if`, `elif`, `while` or `until` condition, a word `exit`,
  `return`, `exec`, `trap` or `eval`, a `.` or `source` command, a command word
  that is a `$` expansion, and a `#` comment that starts inside a command;
- lines are joined the way sh joins them, only where a backslash is the last
  character of the line, and command words are read with their quotes and
  backslashes removed, so an escaped or quoted spelling of `exit` or `.` counts
  as the word itself;
- text inside quotes or `$(...)` is not read for most of those, so a failure
  hidden there (`sh -c "pip install x || true"`, `export X="$(false)"`) is not
  caught.
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

# pip's global options may sit between `pip` and `install` (`pip --cache-dir /x
# install ...`), each optionally followed by a value; without them in the pattern
# such an install would be invisible to every check below. `pip` also follows
# `-m` with no space (`python -mpip install x`), which has no word boundary.
_PIP_WORD = r"(?:\bpip|-m\s*pip)[0-9.]*"
_PIP_INSTALL = re.compile(_PIP_WORD + r"(?:\s+-\S+(?:\s+[^\s-]\S*)?)*?\s+install\b")
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


def _constraint_files(args: list[str]) -> list[str]:
    return [args[i + 1] for i, arg in enumerate(args[:-1]) if arg in ("-c", "--constraint")]


def install_problems(args: list[str]) -> list[str]:
    """Why a `pip install` argument list is not one of the three permitted shapes.

    A hashed lock install, the build backend installed from a lock used as a
    constraints file, or the local package with no dependencies.
    """
    files = _requirement_files(args)
    constraints = _constraint_files(args)
    if files:
        # The bootstrap is exempt from --no-build-isolation: it runs before the
        # backend exists, and its one pin, pip, installs from a wheel.
        required = ("--require-hashes", "--no-deps")
        if files != [PIP_BOOTSTRAP]:
            required += ("--no-build-isolation",)
        return [f"`-r {' '.join(files)}` lacks {flag}" for flag in required if flag not in args]
    targets = [a for a in args if not a.startswith("-") and a not in constraints]
    if constraints:
        problems = [
            f"`-c {' '.join(constraints)}` lacks {flag}"
            for flag in ("--require-hashes", "--no-deps")
            if flag not in args
        ]
        if targets != [BUILD_BACKEND]:
            problems.append(f"`-c` installs {targets} rather than {BUILD_BACKEND} alone")
        return problems
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
    assert len(installs) >= 4, f"{dockerfile}: expected bootstrap, backend, lock and package installs"


@pytest.mark.parametrize("dockerfile", sorted(IMAGE_EXTRAS))
def test_the_build_backend_comes_hashed_from_the_image_lock_before_anything_is_built(
    dockerfile: str,
) -> None:
    # pip builds every sdist before it installs anything, so a setuptools pinned
    # inside the lock install would not exist yet when that install builds the
    # lock's sdists; without isolation they would then fail, and with isolation
    # they fetch setuptools unhashed. The backend therefore needs its own install,
    # from the same lock, ahead of every install that can build.
    installs = pip_installs(_read(dockerfile))
    locks = _image_locks(dockerfile)
    backend = [i for i, args in enumerate(installs) if _constraint_files(args)]
    assert len(backend) == 1, f"{dockerfile}: expected one `-c <lock> {BUILD_BACKEND}` install"
    [at] = backend
    assert _constraint_files(installs[at]) == locks, (dockerfile, _constraint_files(installs[at]), locks)
    # Hash enforcement from a constraints file was measured on the bootstrap's
    # pip, so the backend install must run after the bootstrap, not under
    # whatever pip the base image ships.
    bootstrap = [i for i, args in enumerate(installs) if _requirement_files(args) == [PIP_BOOTSTRAP]]
    assert bootstrap and bootstrap[0] < at, (dockerfile, bootstrap, at)
    for lock in locks:
        assert BUILD_BACKEND in _pins(lock), f"{lock} does not pin {BUILD_BACKEND}"
    builders = [
        i
        for i, args in enumerate(installs)
        if set(_requirement_files(args)) & set(locks) or "." in args
    ]
    assert builders and all(i > at for i in builders), (dockerfile, at, builders)


@pytest.mark.parametrize("dockerfile", sorted(IMAGE_EXTRAS))
def test_the_backend_each_image_builds_with_meets_the_build_system_floor(dockerfile: str) -> None:
    # With --no-build-isolation pip neither installs nor checks
    # [build-system].requires (measured on pip 26.2.1: a package requiring
    # setuptools>=99 installs under 84.0.0 unless --check-build-dependencies is
    # given), so pyproject.toml's setuptools floor reaches an image build only
    # through the version that image's lock pins. No other test ties the two:
    # the floor tests compare requirements.in and requirements-lock.in against
    # their own floors, not against this one.
    requires = [Requirement(r) for r in tomllib.loads(_read("pyproject.toml"))["build-system"]["requires"]]
    [backend] = [r for r in requires if _canonical(r.name) == BUILD_BACKEND]
    for lock in _image_locks(dockerfile):
        version = _pins(lock)[BUILD_BACKEND]
        assert Version(version) in backend.specifier, (
            f"{lock} builds with {BUILD_BACKEND}=={version}, outside [build-system]'s {backend.specifier}"
        )


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
        for lock in _requirement_files(args) + _constraint_files(args):
            assert lock in sources, f"{dockerfile} installs {lock} without copying it in"


# singularity.def builds the HPC image from the same production lock, in %post.
SINGULARITY = "singularity.def"
_SIF_APP = "/app/"
_ERREXIT_ON = re.compile(r"set -[A-Za-z]*e[A-Za-z]*")
# Every pattern from here to _ENDS_OR_HIDES matches a %post command that, under
# dash (Debian's sh) with -e, carries on past a failing command or ends the
# script at status 0 before a later `false` (each measured, dash 0.5.12), with
# one exception, `;`, noted here.
# An AND-OR list (sh -e does not apply to a command that fails before the last
# one of it), a pipeline (its status is the last command's), or a `;` list.
# `false; true` does stop under -e: `;` is refused so that every command stands
# on its own line, where the checks below read it from its first word.
_COMPOUND = re.compile(r"&&|\|\||[;|]")
# `set` turning errexit off: `+o errexit`, or a `+` flag group holding e (`+e`,
# `+eu`, `+ue`, `+ex`), wherever it sits among the other options, read with the
# line's quotes and backslashes removed (`set '+e'`, `set +o "errexit"` and
# `set +\e` all turn it off). Not `-e`, `-eu`, `+u`, `+o pipefail` or
# `set -- +e`, which sets $1.
_ERREXIT_OFF = re.compile(
    r"\bset(?:\s+(?:[-+]o\s+\w+|[-+][A-Za-z]*))*?\s+(?:\+[A-Za-z]*e[A-Za-z]*|\+o\s+errexit)(?!\S)"
)
_QUOTING = re.compile(r"[\"'\\]")
# A `set` whose options come from an expansion (`X=+e` then `set $X`, or
# `set${IFS}+e`), which the pattern above cannot read.
_SET_EXPANDS = re.compile(r"\bset(?:\s.*)?\$")
# Shell text these checks cannot read, refused anywhere in a command: a backtick
# substitution (`echo \`false\`` carries on) and a here-document (`sh <<EOF` runs
# its lines without -e).
_UNREAD = re.compile(r"`|<<")
# A file run in this shell with `.` or `source`, whose lines these checks never
# see (a `set +e` in it turns errexit off here).
_SOURCED = re.compile(r"(?:^|[({]|\b(?:then|do|else)\b)\s*(?:\.|source)(?=\s)")
# `!` in command position: sh -e ignores a negated command's status, so `! pip
# install x` carries on whether pip fails or not. As an argument (`[ ! -e x ]`)
# it negates nothing the shell checks.
_NEGATED = re.compile(r"(?:^|[({]|\b(?:then|do|else)\b)\s*!(?!\S)")
# A `#` that starts a word inside a command. Lines are joined where a backslash
# ends them, so a comment line after one (or `cmd # note \`) is read here; a
# comment on its own line is dropped before this runs.
_COMMENT = re.compile(r"(?:^|\s)#")
# A lone `&`: the command runs in the background, where nothing reads its status.
# Under dash `cmd &>log` is `cmd &` then `>log`, so that counts. `&&`, `>&2`,
# `2>&1` and `<&0` do not.
_BACKGROUND = re.compile(r"(?<![<>&])&(?!&)")
# A condition, whose status sh -e ignores (a multi-line `if pip install x` carries
# on when pip fails), and words that end %post early or at status 0 (`exit 0`,
# `return 0`, `exec true`, `trap 'exit 0' EXIT`) or run text these checks never
# read (`eval`), matched as whole words anywhere: an innocent `echo exit` fails
# too, which is the safe direction.
_CONDITION = re.compile(r"(?<![\w./-])(?:if|elif|while|until)(?![\w./-])")
_ENDS_OR_HIDES = re.compile(r"(?<![\w./-])(?:exit|return|exec|trap|eval)(?![\w./-])")
# Every other way to fetch or build third-party code, which the `pip install`
# checks below cannot see: pip wheel and pip download run an sdist's build
# backend too. The same list as the release workflow's test.
_PIP_OPTIONS = r"(?:\s+-\S+(?:\s+[^\s-]\S*)?)*?"
_OTHER_INSTALLERS = re.compile(
    r"(?:^|\s)(?:uvx|pipx|uv\s+(?:pip|tool|run|add|sync)|conda\s+install|easy_install)\b"
    rf"|{_PIP_WORD}{_PIP_OPTIONS}\s+(?:wheel|download)\b"
)


def singularity_section(text: str, name: str) -> list[str]:
    """The lines of one %section of a Singularity definition file."""
    lines, inside = [], False
    for line in text.splitlines():
        if line.startswith("%"):
            inside = line.split()[0] == f"%{name}"
            continue
        if inside:
            lines.append(line)
    return lines


def singularity_post_commands(text: str) -> list[str]:
    """%post's commands, as sh splits them into lines.

    A line that begins with `#` outside a continuation is a comment and is
    dropped. A backslash that is the LAST character of a line (an odd run of
    them) joins the next line to it, removing both, whatever that line holds,
    even a blank line or a comment: a backslash followed by spaces joins
    nothing. Whitespace is then collapsed.
    """
    commands: list[str] = []
    pending: str | None = None
    for line in singularity_section(text, "post"):
        if pending is None and line.lstrip().startswith("#"):
            continue
        current = line if pending is None else pending + line
        trailing = len(current) - len(current.rstrip("\\"))
        if trailing % 2 == 1:
            pending = current[:-1]
            continue
        pending = None
        if current.strip():
            commands.append(" ".join(current.split()))
    if pending is not None and pending.strip():
        commands.append(" ".join(pending.split()))
    return commands


def singularity_pip_installs(text: str) -> list[list[str]]:
    """The argument list of every `pip install` in %post, exactly as written."""
    installs = []
    for command in singularity_post_commands(text):
        match = _PIP_INSTALL.search(command)
        if match:
            installs.append(command[match.end() :].split())
    return installs


def _from_app(args: list[str]) -> list[str]:
    """%files copies each tracked path to /app/<path>; map the arguments back."""
    return [arg[len(_SIF_APP) :] if arg.startswith(_SIF_APP) else arg for arg in args]


def _top_level(command: str) -> str:
    """`command` with its quoted text and `$(...)` substitutions removed.

    An operator inside quotes, or inside a command substitution such as
    `"$(python -c "a; b")"`, belongs to a string or to another shell, so only
    what is left here can join two commands of this one into a list.
    """
    kept: list[str] = []
    stack: list[str] = []
    i = 0
    while i < len(command):
        char, top = command[i], (stack[-1] if stack else "")
        if top == "'":
            if char == "'":
                stack.pop()
        elif char == "\\":
            i += 1
        elif top == '"' and char == '"':
            stack.pop()
        elif command.startswith("$(", i):
            stack.append("(")
            i += 1
        elif top == "(" and char == "(":
            stack.append("(")
        elif top == "(" and char == ")":
            stack.pop()
        elif char == '"' or (char == "'" and top != '"'):
            # Inside double quotes a single quote is an ordinary character.
            stack.append(char)
        elif not stack:
            kept.append(char)
        i += 1
    return "".join(kept)


def _command_words(command: str) -> list[str]:
    """`command`'s top-level words as the shell reads them after quote removal.

    A backslash yields the character after it, matched quotes are dropped and
    their text kept (`e''xit`, `"."` and `e\\xit` read as `exit`, `.` and
    `exit`), and a `$(...)` substitution stays as the text `$()`. A bare `(` or
    `)` is a word of its own, so a subshell's first word is found.
    """
    words: list[str] = []
    word: list[str] = []
    started = False
    stack: list[str] = []
    i = 0

    def end_word() -> None:
        nonlocal word, started
        if started:
            words.append("".join(word))
        word, started = [], False

    while i < len(command):
        char, top = command[i], (stack[-1] if stack else "")
        inside = "(" in stack
        if top == "'":
            if char == "'":
                stack.pop()
            elif not inside:
                word.append(char)
        elif char == "\\":
            if not inside and i + 1 < len(command):
                word.append(command[i + 1])
                started = True
            i += 1
        elif top == '"' and char == '"':
            stack.pop()
        elif command.startswith("$(", i):
            if not inside:
                word.append("$()")
                started = True
            stack.append("(")
            i += 1
        elif top == "(" and char == "(":
            stack.append("(")
        elif top == "(" and char == ")":
            stack.pop()
        elif char == '"' or (char == "'" and top != '"'):
            stack.append(char)
            started = started or not inside
        elif inside:
            pass
        elif not stack and char.isspace():
            end_word()
        elif not stack and char in "()":
            end_word()
            words.append(char)
        else:
            word.append(char)
            started = True
        i += 1
    end_word()
    return words


# A word after which the next word is a command: `( cmd )`, `{ cmd; }`, `! cmd`,
# and the bodies of `then`, `do` and `else`.
_COMMAND_STARTERS = frozenset({"(", "{", "!", "then", "do", "else"})
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\+?=")
# Words that, wherever they stand as a whole word, can end %post early or at
# status 0 or run text these checks never read.
_STOP_WORDS = frozenset({"exit", "return", "exec", "trap", "eval"})


def _command_position_words(words: list[str]) -> list[str]:
    """The words that name the command being run: the first non-assignment word
    of the line and of each `(`, `{`, `!`, `then`, `do` or `else` body."""
    found: list[str] = []
    at_command = True
    for word in words:
        if at_command and _ASSIGNMENT.match(word):
            continue
        if at_command:
            found.append(word)
        at_command = word in _COMMAND_STARTERS
    return found


def post_stop_problems(text: str) -> list[str]:
    """Which of the refused spellings %post contains, each one a way %post could
    carry on past a failed command or end early at status 0 (a regression
    ratchet over a fixed list, not a proof that no other way exists).

    Apptainer's user guide says the build halts if any command fails, and that
    %post runs under sh or bash, but names no shell flag, so the file sets -e
    itself, before anything else runs. And under sh -e a command that fails
    before the last one of an `a && b` list does not stop the script, so every
    command has to stand on its own line, not only the installs. The `set`,
    backtick and here-document checks read the whole line, quotes included;
    every other check reads only what `_top_level` leaves, so a failure hidden
    in a string another program runs (`sh -c "pip install x || true"`) or in a
    `$(...)` substitution (`export X="$(false)"`, which carries on under dash)
    is not found.
    """
    commands = singularity_post_commands(text)
    problems = []
    if not commands or not _ERREXIT_ON.fullmatch(commands[0]):
        problems.append(f"%post does not start with `set -e`: {commands[:1]}")
    checks = (
        (_COMPOUND, "compound command"),
        (_NEGATED, "negated with `!`"),
        (_BACKGROUND, "runs in the background"),
        (_CONDITION, "runs a condition, whose failure -e ignores"),
        (_ENDS_OR_HIDES, "can end %post early or at status 0, or hide a command"),
        (_SOURCED, "runs a file these checks do not read"),
        (_COMMENT, "has a comment inside a command"),
    )
    for command in commands:
        if _ERREXIT_OFF.search(_QUOTING.sub("", command)) or _SET_EXPANDS.search(command):
            problems.append(f"turns -e off, or may: {command}")
        if _UNREAD.search(command):
            problems.append(f"backtick or here-document: {command}")
        top = _top_level(command)
        problems += [f"{why}: {command}" for pattern, why in checks if pattern.search(top)]
        # The same words again with their quotes and backslashes removed, since
        # the patterns above read the text as written (`e\\xit`, `"."`).
        words = _command_words(command)
        at_command = _command_position_words(words)
        if _STOP_WORDS.intersection(words):
            problems.append(f"can end %post early or at status 0, or hide a command: {command}")
        if {".", "source"}.intersection(at_command):
            problems.append(f"runs a file these checks do not read: {command}")
        if any("$" in word for word in at_command):
            problems.append(f"command word is an expansion: {command}")
    return problems


def test_singularity_post_has_none_of_the_refused_spellings() -> None:
    problems = post_stop_problems(_read(SINGULARITY))
    assert not problems, problems


@pytest.mark.parametrize(
    ("post", "stops"),
    [
        ("%post\n    set -e\n    pip install --require-hashes -r /app/x.txt\n", True),
        ("%post\n    set -eu\n    pip install x\n", True),
        ("%post\n    pip install --require-hashes -r /app/x.txt\n", False),
        ("%post\n    pip install x\n    set -e\n", False),
        ("%post\n    # comment\n    set -e\n    pip install x\n", True),
        ("%post\n    set -e\n    true && pip install x\n", False),
        ("%post\n    set -e\n    pip install x && true\n", False),
        ("%post\n    set -e\n    pip install x | tee log\n", False),
        ("%post\n    set -e\n    pip install x; true\n", False),
        ("%post\n    set -e\n    set +e\n    pip install x\n", False),
        ("%post\n    set -e\n    set +o errexit\n    pip install x\n", False),
        ("%post\n    set -e\n    pip --log l install x || true\n", False),
        (
            "%post\n    set -e\n    pip install \\\n        x\n%test\n    false && pip install y\n",
            True,
        ),
        ("%post\n    set -e\n    pip install \\\n        x || true\n", False),
        # Not only the installs: an apt line re-joined with && escapes -e the same way.
        ("%post\n    set -e\n    apt-get update && apt-get install -y gcc\n", False),
        ("%post\n    set -e\n    apt-get update\n    apt-get install -y gcc\n", True),
        ("%post\n    set -e\n    cd /app; make\n", False),
        # An operator inside quotes or a command substitution joins nothing here.
        ('%post\n    set -e\n    X="$(python -c "import a; print(a.b)")"\n', True),
        ("%post\n    set -e\n    echo 'a && b | c'\n", True),
        ('%post\n    set -e\n    echo "it\'s; fine"\n', True),
        ('%post\n    set -e\n    X="$(false)" && true\n', False),
        ('%post\n    set -e\n    echo "a" | tee log\n', False),
        # Errexit off in any spelling, and what must not count as that.
        ("%post\n    set -e\n    set +eu\n    pip install x\n", False),
        ("%post\n    set -e\n    set +ex\n    pip install x\n", False),
        ("%post\n    set -e\n    set +ue\n    pip install x\n", False),
        ("%post\n    set -e\n    set -u +e\n    pip install x\n", False),
        ("%post\n    set -e\n    set -o pipefail +o errexit\n    pip install x\n", False),
        ("%post\n    set -e\n    set -eu\n    set +u\n    set +x\n    pip install x\n", True),
        ("%post\n    set -e\n    set +o pipefail\n    set -- +e\n    pip install x\n", True),
        # `!` in command position, alone or in a subshell; not as an argument or quoted.
        ("%post\n    set -e\n    ! pip install x\n", False),
        ("%post\n    set -e\n    ! apt-get update\n", False),
        ("%post\n    set -e\n    ( ! pip install x )\n", False),
        ("%post\n    set -e\n    [ ! -e /app/x ]\n    test ! -e /app/x\n", True),
        ("%post\n    set -e\n    echo \"done!\"\n    echo '! x'\n", True),
        # A background command; not `&&`, a redirection to a descriptor, or a quoted `&`.
        ("%post\n    set -e\n    apt-get update &\n", False),
        ("%post\n    set -e\n    pip install x &>/dev/null\n", False),
        ("%post\n    set -e\n    pip install x >/dev/null 2>&1\n    echo x >&2\n", True),
        ("%post\n    set -e\n    echo \"a & b\"\n    echo 'a & b'\n    echo a \\& b\n", True),
        # A condition, or anything that ends %post early, wherever it sits.
        ("%post\n    set -e\n    if pip install x\n    then\n        true\n    fi\n", False),
        ("%post\n    set -e\n    while apt-get update\n    do\n        break\n    done\n", False),
        ("%post\n    set -e\n    pip install x\n    exit 0\n", False),
        ("%post\n    set -e\n    exit\n    pip install x\n", False),
        ("%post\n    set -e\n    return 0\n", False),
        ("%post\n    set -e\n    exec true\n", False),
        ("%post\n    set -e\n    trap 'exit 0' EXIT\n", False),
        ('%post\n    set -e\n    eval "pip install x || true"\n', False),
        # Errexit off behind quotes, a backslash or an expansion.
        ('%post\n    set -e\n    set +o "errexit"\n    pip install x\n', False),
        ("%post\n    set -e\n    set '+e'\n    pip install x\n", False),
        ("%post\n    set -e\n    set +\\e\n    pip install x\n", False),
        ("%post\n    set -e\n    X=+e\n    set $X\n    pip install x\n", False),
        ('%post\n    set -e\n    echo "set -e"\n    set -u\n    pip install x\n', True),
        # Text these checks cannot read: a backtick, a here-document, a sourced file.
        ("%post\n    set -e\n    echo `apt-get update`\n", False),
        ("%post\n    set -e\n    sh <<'EOF'\n    apt-get update\n    true\n    EOF\n", False),
        ("%post\n    set -e\n    . /app/setup.sh\n", False),
        ("%post\n    set -e\n    source /app/setup.sh\n", False),
        ("%post\n    set -e\n    pip install .\n    sh /app/setup.sh\n    ./setup.sh\n", True),
        # Escaped or quoted command words, and a `set` with an expansion glued to it.
        ("%post\n    set -e\n    e\\xit 0\n", False),
        ("%post\n    set -e\n    tr\\ap 'exit 0' EXIT\n", False),
        ("%post\n    set -e\n    ex\\ec true\n", False),
        ("%post\n    set -e\n    \\. ./off.sh\n", False),
        ('%post\n    set -e\n    "." ./off.sh\n', False),
        ("%post\n    set -e\n    sour\\ce ./off.sh\n", False),
        ("%post\n    set -e\n    set${IFS}+e\n", False),
        ("%post\n    set -e\n    set +o${IFS}errexit\n", False),
        # A continuation joins only a backslash that ends the line.
        ("%post\n    set -e\n    apt-get clean \\\n\n    ! apt-get update\n", False),
        ("%post\n    set -e\n    apt-get clean \\ \n    . ./off.sh\n", False),
        ("%post\n    set -e\n    apt-get clean \\\n    apt-get update\n", True),
        ("%post\n    set -e\n    # note \\\n    apt-get update\n", True),
        ("%post\n    set -e\n    apt-get clean \\\n    # note\n    ! apt-get update\n", False),
        # A command word that is an expansion; an assignment, or `.` as an argument, is not one.
        ("%post\n    set -e\n    P=pip\n    $P install x\n", False),
        ('%post\n    set -e\n    X="$(echo a)"\n    cp -r a .\n    echo "$X"\n', True),
        (
            '%post\n    set -e\n    python -c "import sys; sys.exit(0)"\n    rm -f /tmp/exit.log\n',
            True,
        ),
    ],
)
def test_post_stop_problems_reads_post(post: str, stops: bool) -> None:
    assert (not post_stop_problems(post)) is stops, post_stop_problems(post)


def test_singularity_runs_no_other_installer() -> None:
    """The `pip install` checks cannot see any other installer, so none may appear."""
    text = _read(SINGULARITY)
    live = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    others = [line for line in live if _OTHER_INSTALLERS.search(line)]
    assert not others, others


@pytest.mark.parametrize(
    ("command", "matches"),
    [
        ("pipx install somepkg", True),
        ("easy_install somepkg", True),
        ("uv pip install foo", True),
        ("uvx build", True),
        ("conda install -y foo", True),
        ("pip wheel --no-deps -w w foo", True),
        ("python -m pip --quiet download foo", True),
        ("python -mpip download foo", True),
        ("pip install --require-hashes --no-deps -r /app/x.txt", False),
        ("apt-get install -y --no-install-recommends build-essential", False),
        ("mhcflurry-downloads fetch models_class1_presentation", False),
    ],
)
def test_other_installers_are_recognised(command: str, matches: bool) -> None:
    assert bool(_OTHER_INSTALLERS.search(command)) is matches


def singularity_install_problems(text: str) -> list[str]:
    """Why singularity.def's %post pip installs are not the bootstrap, backend, lock sequence."""
    installs = [_from_app(args) for args in singularity_pip_installs(text)]
    problems = [p for args in installs for p in install_problems(args)]
    bootstrap = [i for i, args in enumerate(installs) if _requirement_files(args) == [PIP_BOOTSTRAP]]
    backend = [i for i, args in enumerate(installs) if _constraint_files(args)]
    builders = [
        i
        for i, args in enumerate(installs)
        if PRODUCTION_LOCK in _requirement_files(args) or "." in args
    ]
    if not (len(bootstrap) == 1 and len(backend) == 1 and builders):
        problems.append(f"expected one bootstrap, one backend and a lock: {bootstrap, backend, builders}")
        return problems
    if _constraint_files(installs[backend[0]]) != [PRODUCTION_LOCK]:
        problems.append("the backend install does not take the production lock as constraints")
    if not bootstrap[0] < backend[0] < min(builders):
        problems.append(f"installs out of order: {bootstrap, backend, builders}")
    return problems


def test_singularity_builds_the_lock_against_the_hashed_backend() -> None:
    problems = singularity_install_problems(_read(SINGULARITY))
    assert not problems, problems


# Each case puts the text after `apt-get clean` in the shipped %post and must be
# refused by the %post checks or the install checks. A regression ratchet: it
# lists spellings that once passed, it is not a proof that none other does, and
# nothing here runs %post. BS is one backslash, so each entry reads as the lines
# it puts in the file.
BS = "\\"
_AFTER_CLEAN = "    apt-get clean\n"
_SHIPPED_MUTANTS = {
    # A continuation joins only a backslash that ends the line (sh), so a blank
    # line, a trailing space or a comment after it does not hide the next command.
    "negation after a backslash and a blank line": f"    apt-get clean {BS}\n\n    ! apt-get update\n",
    "source after a backslash and a blank line": f"    apt-get clean {BS}\n\n    . ./off.sh\n",
    "source after a backslash and a trailing space": f"    apt-get clean {BS} \n    . ./off.sh\n",
    "negation after a backslash and trailing spaces": f"    apt-get clean {BS}   \n    ! apt-get update\n",
    "negation after a backslash and a comment line": f"    apt-get clean {BS}\n    # note\n    ! apt-get update\n",
    "comment ending in a backslash, then exit": f"    apt-get clean # note {BS}\n    exit 0\n",
    # The command word, read with its quotes and backslashes removed.
    "exit with a backslash": f"    e{BS}xit 0\n",
    "exit with empty quotes": "    e''xit 0\n",
    "exit split by a continuation": f"    e{BS}\nxit 0\n",
    "trap with a backslash": f"    tr{BS}ap 'exit 0' EXIT\n",
    "exec with a backslash": f"    ex{BS}ec true\n",
    "dot with a backslash": f"    {BS}. ./off.sh\n",
    "dot in quotes": '    "." ./off.sh\n',
    "source with a backslash": f"    sour{BS}ce ./off.sh\n",
    "source in single quotes": "    's'ource ./off.sh\n",
    # `set` with no space before an expansion.
    "set ${IFS} +e": "    set${IFS}+e\n",
    "set +o ${IFS} errexit": "    set +o${IFS}errexit\n",
    # Installs the pip pattern must see, and a command word that is a variable.
    "python -mpip install": "    python -mpip install requests\n",
    "python -m pip install": "    python -m pip install requests\n",
    "pip3.11 install": "    pip3.11 install requests\n",
    "pip through a variable": "    P=pip\n    $P install requests\n",
    "pip through a quoted variable": '    P=pip\n    "${P}" install requests\n',
    "pip through a substitution": "    $(command -v pip) install requests\n",
}


def test_shipped_mutants_are_refused() -> None:
    shipped = _read(SINGULARITY)
    assert shipped.count(_AFTER_CLEAN) == 1, "the anchor line moved; update _AFTER_CLEAN"
    assert not post_stop_problems(shipped) and not singularity_install_problems(shipped)
    survivors = []
    for name, text in _SHIPPED_MUTANTS.items():
        mutant = shipped.replace(_AFTER_CLEAN, _AFTER_CLEAN + text)
        assert mutant != shipped, name
        if not (post_stop_problems(mutant) or singularity_install_problems(mutant)):
            survivors.append(name)
    assert not survivors, survivors


def test_singularity_runs_pip_install_only_in_post() -> None:
    text = _read(SINGULARITY)
    live = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    everywhere = [line for line in live if _PIP_INSTALL.search(line)]
    in_post = [line for line in singularity_section(text, "post") if line in everywhere]
    assert everywhere and everywhere == in_post, everywhere


def test_singularity_copies_in_every_file_it_installs_from() -> None:
    text = _read(SINGULARITY)
    copies = {}
    for line in singularity_section(text, "files"):
        parts = line.split()
        if len(parts) == 2 and not line.lstrip().startswith("#"):
            copies[parts[0]] = parts[1]
    checked = 0
    for args in singularity_pip_installs(text):
        for path in _requirement_files(args) + _constraint_files(args):
            checked += 1
            assert path.startswith(_SIF_APP), f"{path} is not under {_SIF_APP}"
            [relative] = _from_app([path])
            covered = [
                source
                for source, destination in copies.items()
                if destination == _SIF_APP + source
                and (relative == source or relative.startswith(source + "/"))
            ]
            assert covered and (PROJECT_ROOT / relative).is_file(), (path, covered)
    assert checked >= 3, checked


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
        ("RUN pip install --user --require-hashes --no-deps --no-build-isolation -r requirements.txt", 0),
        ("RUN pip install --user --require-hashes --no-deps -r requirements.txt", 1),
        (f"RUN pip install --user --require-hashes --no-deps -r {PIP_BOOTSTRAP}", 0),
        ("RUN pip install --user --no-deps --no-build-isolation .", 0),
        ('RUN pip install --user ".[api]"', 1),
        ("RUN pip install --user --no-deps .", 1),
        ("RUN pip install --user -r requirements.txt", 3),
        ("RUN pip install fastapi", 1),
        ("RUN python -mpip install fastapi", 1),
        ("RUN pip3 install --require-hashes -r x.txt", 2),
        ("RUN python -m pip install --no-deps --no-build-isolation .", 0),
        ("RUN pip install --user --require-hashes --no-deps -c lock.txt setuptools", 0),
        ("RUN pip install --user --require-hashes --no-deps --constraint lock.txt setuptools", 0),
        ("RUN pip install --user --no-deps -c lock.txt setuptools", 1),
        ("RUN pip install --user --require-hashes --no-deps -c lock.txt setuptools wheel", 1),
        ("RUN pip install --user --require-hashes --no-deps -c lock.txt", 1),
        ("RUN pip --disable-pip-version-check install --user fastapi", 1),
        ("RUN pip --cache-dir /tmp/pip install --user fastapi", 1),
        ("RUN python -m pip --no-input install --no-deps --no-build-isolation .", 0),
    ],
)
def test_install_problems_classifies_each_shape(line: str, problem_count: int) -> None:
    [args] = pip_installs(line)
    assert len(install_problems(args)) == problem_count, install_problems(args)


def test_continuations_chains_and_unparseable_runs() -> None:
    text = (
        "# RUN pip install ignored-because-commented\n"
        "RUN pip install --user --require-hashes --no-deps --no-build-isolation \\\n"
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


def test_dockerignore_keeps_secrets_and_assistant_config_out_of_the_context() -> None:
    # No Dockerfile COPYs these today; the build context is still uploaded whole to
    # the builder, so a stray key or an assistant's config must not travel with it.
    rules = _read(".dockerignore").splitlines()
    kept_out = [
        ".env",
        ".env.local",
        "deploy/.env",
        "deploy/.env.production",
        "server.pem",
        "certs/client.pem",
        ".agents/rules/x.md",
        ".cursor/rules/x.mdc",
        ".codex/config.toml",
        ".claude/settings.json",
        ".mcp.json",
    ]
    leaked = [path for path in kept_out if not dockerignore_excludes(path, rules)]
    assert not leaked, f".dockerignore lets these into the build context: {leaked}"


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
