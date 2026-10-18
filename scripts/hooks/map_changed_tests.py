#!/usr/bin/env python3
"""Map the files a push changes to the test files that exercise them.

Two entry points, one mapping.

As a pytest plugin, which is how scripts/hooks/pre-push uses it in its default
(fast) mode, from the repository root:

    python -m pytest tests/ -p scripts.hooks.map_changed_tests \
        --sestrav-changed-list=FILE --maxfail=1 [--ignore=...]

Pass the option as ONE token. pytest chooses its rootdir from the common
ancestor of every argument naming an existing path, so a separate FILE token
in the temp directory moves rootdir above the repository and pytest.ini is not
read; the plugin refuses to run when that has happened.

FILE lists the changed repository-relative paths, one per line. At configure
time the plugin byte-compiles every changed .py file that exists (a failure
stops the run before any test does) and maps the changes to test files; during
collection it then ignores every other test file, so unselected files are never
imported. pytest's own --ignore values are excluded from the mapping. When the
mapping says __FULL__ the plugin filters nothing and the whole suite runs. The
hook therefore makes ONE python call for this check, and its full mode is the
same command line without the plugin options.

As a command, for a human or a script:

    python scripts/hooks/map_changed_tests.py [--repo-root DIR] [--exclude PATH]...
        [--fallback PATH]... [--max-files N] [--explain] [PATH... | -]

It prints one test file per line, relative to the repository root, sorted and
de-duplicated, and nothing else on stdout; or the single line __FULL__ when a
changed path affects the whole suite. PATH arguments are repository-relative
paths as `git diff --name-only` prints them; `-` reads them from stdin, one per
line. Deleted paths are welcome: a test that still imports a deleted module is
exactly the test that should run.

Rules, applied to each changed path:
  1. Any conftest.py, or a path in FULL_SUITE_TRIGGERS, maps to __FULL__.
  2. A test file (tests/**/test_*.py or tests/**/*_test.py) maps to itself
     while it exists.
  3. A .py module maps to tests whose filename matches test_*<stem>*.py, and
     to tests whose text names it: its dotted module name (src.foo.bar), a
     `from src.foo import bar` import, its repository path, or its basename
     when the basename is distinctive.
  4. A documentation file (.md, .rst, .txt, .cff, or anything under docs/)
     maps ONLY to tests whose filename contains its stem (test_*readme*.py for
     README.md). Text mentions are not followed: nearly every pull request
     edits CHANGELOG.md, and the tests that merely read README.md or
     CHANGELOG.md turned a four-file change into 39 test files and four
     minutes (measured 2026-10-14 on PR #786's diff). The doc gates that
     block merges still run on every push as pre-push Checks 7 and 8, and CI
     runs the doc-reading tests.
  5. Any other file maps to tests whose text names its repository path, or its
     basename when distinctive. A file under tests/fixtures/<dir>/ also maps to
     tests that name <dir>.
  6. When nothing maps, the --fallback files are printed (default
     tests/test_cli_smoke.py).

More than --max-files selected test files (default 60) prints __FULL__: at that
size the fast path is no longer fast, and the full suite is the honest answer.
--exclude removes a test file from the selection. The plugin fills it from
pytest's --ignore list, so a file the hook ignores on Windows is never the only
thing selected (that would collect nothing and fail the gate).

Direct references only. A test that reaches a changed module through another
module is not selected; CI's full suite is the authoritative check for that.
Standard library only, so it runs under whatever interpreter the hook found.
"""

from __future__ import annotations

import argparse
import os
import py_compile
import re
import sys
from pathlib import Path, PurePosixPath

FULL = "__FULL__"
FULL_SUITE_TRIGGERS = {"pytest.ini", "pyproject.toml", "setup.cfg", "setup.py", "tox.ini"}
DEFAULT_FALLBACK = ("tests/test_cli_smoke.py",)
DEFAULT_MAX_FILES = 60
MIN_STEM = 3
# Basenames too common to identify one file: matching them would select tests
# that name some OTHER file of the same name.
GENERIC_BASENAMES = {
    "__init__.py", "__main__.py", "main.py", "utils.py", "util.py", "common.py",
    "base.py", "config.py", "constants.py", "README.md", "index.md", "config.yaml",
    "__init__", "Makefile",
}
GENERIC_STEMS = {"__init__", "__main__", "main", "utils", "util", "common", "base", "conftest"}
DOC_SUFFIXES = {".md", ".rst", ".txt", ".cff"}


def _is_doc(rel: str) -> bool:
    pure = PurePosixPath(rel)
    return pure.suffix.lower() in DOC_SUFFIXES or (bool(pure.parts) and pure.parts[0] == "docs")


def _norm(path: str) -> str:
    text = path.strip().strip("\"'").replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _is_test_file(rel: str) -> bool:
    parts = PurePosixPath(rel).parts
    if not parts or parts[0] != "tests" or "__pycache__" in parts:
        return False
    name = parts[-1]
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def _discover_tests(repo: Path) -> dict[str, str]:
    """Every test file under tests/, mapped to its text."""
    found: dict[str, str] = {}
    tests_dir = repo / "tests"
    if not tests_dir.is_dir():
        return found
    for path in tests_dir.rglob("*.py"):
        rel = path.relative_to(repo).as_posix()
        if _is_test_file(rel):
            try:
                found[rel] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                found[rel] = ""
    return found


def _word(token: str) -> str:
    # A token bounded so that it cannot be the middle of a longer name.
    return r"(?<![\w.-])" + re.escape(token) + r"(?![\w-])"


def _patterns_for(rel: str) -> list[tuple[str, re.Pattern[str]]]:
    """(label, compiled pattern) pairs whose hit in a test's text selects it."""
    pure = PurePosixPath(rel)
    patterns: list[tuple[str, str]] = [("path", _word(rel))]
    if pure.name not in GENERIC_BASENAMES and pure.name != rel:
        patterns.append(("basename", _word(pure.name)))
    if pure.suffix == ".py":
        parts = list(pure.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            dotted = ".".join(parts)
            patterns.append(("module", r"(?<![\w.])" + re.escape(dotted) + r"(?!\w)"))
            if len(parts) > 1:
                parent, name = ".".join(parts[:-1]), re.escape(parts[-1])
                patterns.append((
                    "from-import",
                    r"from\s+" + re.escape(parent) + r"\s+import\s+"
                    r"(?:\([^)]*?\b" + name + r"\b|[^\n]*?\b" + name + r"\b)",
                ))
    if len(pure.parts) >= 3 and pure.parts[0] == "tests" and pure.parts[1] == "fixtures":
        patterns.append(("fixture-dir", _word(pure.parts[2])))
    return [(label, re.compile(pattern, re.S)) for label, pattern in patterns]


def _stem_matches(rel: str, tests: dict[str, str]) -> list[str]:
    """Tests named test_*<stem>*.py, for a .py module or a documentation file."""
    pure = PurePosixPath(rel)
    doc = _is_doc(rel)
    stem = pure.stem.lower() if doc else pure.stem
    if (pure.suffix != ".py" and not doc) or len(stem) < MIN_STEM or stem in GENERIC_STEMS:
        return []
    hits = []
    for test in tests:
        name = PurePosixPath(test).name
        middle = name[len("test_"):-len(".py")]
        if name.startswith("test_") and stem in (middle.lower() if doc else middle):
            hits.append(test)
    return hits


def map_changed(
    changed: list[str],
    repo: Path,
    *,
    exclude: set[str] | None = None,
    fallback: tuple[str, ...] = DEFAULT_FALLBACK,
    max_files: int = DEFAULT_MAX_FILES,
    explain: list[str] | None = None,
) -> list[str]:
    """Return the selected test files, or [FULL]."""
    exclude = {_norm(e) for e in (exclude or set())}
    note = explain.append if explain is not None else (lambda _msg: None)
    paths = sorted({_norm(p) for p in changed if _norm(p)})
    for rel in paths:
        name = PurePosixPath(rel).name
        if name == "conftest.py" or rel in FULL_SUITE_TRIGGERS:
            note(f"{rel}: affects the whole suite")
            return [FULL]

    tests = _discover_tests(repo)
    selected: dict[str, str] = {}
    for rel in paths:
        if _is_test_file(rel):
            if rel in tests:
                selected.setdefault(rel, f"changed test file {rel}")
            continue
        for test in _stem_matches(rel, tests):
            selected.setdefault(test, f"filename matches stem of {rel}")
        if _is_doc(rel):
            continue
        patterns = _patterns_for(rel)
        for test, text in tests.items():
            if test in selected:
                continue
            for label, pattern in patterns:
                if pattern.search(text):
                    selected[test] = f"names {rel} ({label})"
                    break

    for test in list(selected):
        if test in exclude:
            note(f"{test}: excluded by the caller")
            del selected[test]

    if len(selected) > max_files:
        note(f"{len(selected)} test files selected, over --max-files {max_files}")
        return [FULL]
    if not selected:
        present = tuple(f for f in fallback if (repo / f).is_file() and f not in exclude)
        if not present:
            note("nothing mapped and no fallback file exists")
            return [FULL]
        for f in present:
            note(f"{f}: fallback smoke file")
        return sorted(present)
    for test, why in sorted(selected.items()):
        note(f"{test}: {why}")
    return sorted(selected)


def read_changed_list(path: str | os.PathLike[str]) -> list[str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return [line for line in text.replace("\r", "").splitlines() if line.strip()]


def compile_errors(changed: list[str], repo: Path) -> list[str]:
    """py_compile every changed .py file that exists; return one line per failure."""
    errors = []
    for rel in sorted({_norm(p) for p in changed}):
        path = repo / rel
        if rel.endswith(".py") and path.is_file():
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                errors.append(f"{rel}: {exc.msg.strip()}")
    return errors


# ---- pytest plugin --------------------------------------------------------
# Active only when --sestrav-changed-list is given; otherwise every hook below
# returns without changing anything.

_SELECTED_KEY = "_sestrav_fast_selected"


def pytest_addoption(parser):
    group = parser.getgroup("sestrav-fast", "SESTRAV fast pre-push selection")
    group.addoption(
        "--sestrav-changed-list",
        default=None,
        metavar="FILE",
        help="run only the tests mapped from the changed paths listed in FILE",
    )


def pytest_configure(config):
    changed_file = config.getoption("sestrav_changed_list", default=None)
    if not changed_file:
        return
    import pytest

    repo = Path(str(config.rootpath)).resolve()
    if not (repo / "tests").is_dir() or config.inipath is None:
        # A wrong rootdir means pytest.ini was not read either, so neither the
        # selection nor the suite's own configuration can be trusted.
        raise pytest.UsageError(
            f"pytest chose rootdir {repo} (config file: {config.inipath}), which is "
            "not the repository root; pass --sestrav-changed-list=FILE as one token"
        )
    changed = read_changed_list(changed_file)
    errors = compile_errors(changed, repo)
    if errors:
        raise pytest.UsageError(
            "a changed .py file does not compile:\n  " + "\n  ".join(errors)
        )
    exclude = set()
    for ignored in config.getoption("ignore") or []:
        ignored_path = Path(ignored)
        if not ignored_path.is_absolute():
            ignored_path = Path(str(config.invocation_params.dir)) / ignored_path
        try:
            exclude.add(ignored_path.resolve().relative_to(repo).as_posix())
        except ValueError:
            continue
    selected = map_changed(changed, repo, exclude=exclude)
    if selected == [FULL]:
        sys.stderr.write("[pre-push] Fast mode: the changes affect the whole suite; running all of it.\n")
        setattr(config, _SELECTED_KEY, None)
        return
    sys.stderr.write(
        f"[pre-push] Fast mode: {len(changed)} changed path(s) -> {len(selected)} test file(s): "
        + " ".join(selected) + "\n"
    )
    setattr(config, _SELECTED_KEY, {(repo / rel).resolve() for rel in selected})


def pytest_ignore_collect(collection_path, config):
    selected = getattr(config, _SELECTED_KEY, None)
    if selected is None:
        return None
    path = Path(str(collection_path))
    name = path.name
    if not (name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))):
        return None
    return True if path.resolve() not in selected else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="changed paths, or - to read stdin")
    parser.add_argument("--repo-root", default=".", help="repository root (default: cwd)")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATH")
    parser.add_argument("--fallback", action="append", default=None, metavar="PATH")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--explain", action="store_true", help="say why, on stderr")
    args = parser.parse_args(argv)

    changed: list[str] = []
    for item in args.paths:
        if item == "-":
            changed.extend(sys.stdin.read().replace("\r", "").splitlines())
        else:
            changed.append(item)

    repo = Path(args.repo_root).resolve()
    if not repo.is_dir():
        print(f"map_changed_tests: --repo-root {repo} is not a directory", file=sys.stderr)
        return 2
    reasons: list[str] = []
    result = map_changed(
        changed,
        repo,
        exclude=set(args.exclude),
        fallback=tuple(args.fallback) if args.fallback else DEFAULT_FALLBACK,
        max_files=args.max_files,
        explain=reasons,
    )
    if args.explain:
        for line in reasons:
            print(f"  {line}", file=sys.stderr)
    # Bytes, so Windows text mode cannot turn LF into CRLF: a shell reading this
    # with $(...) would otherwise get "tests/test_x.py\r" and find no such file.
    sys.stdout.flush()
    sys.stdout.buffer.write("".join(f"{line}\n" for line in result).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
