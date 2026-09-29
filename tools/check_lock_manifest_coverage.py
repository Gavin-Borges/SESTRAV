#!/usr/bin/env python
"""Fail if environments/requirements.lock pins a distribution no GitHub-parsed manifest declares.

WHY THIS GATE EXISTS
--------------------
``environments/requirements.lock`` is what the Docker image and several CI jobs
actually install from, but GitHub's dependency graph does not parse a bare
``.lock`` filename for the pip ecosystem. Measured on 2026-09-19: that file has
raised 0 of this repository's 108 Dependabot alerts, while its compile input
``environments/requirements-lock.in`` is parsed and has raised 2.

Today every distribution pinned in the lock also appears in at least one parsed
manifest, so nothing is unmonitored. That is a property of today's content, not
an enforced invariant. A future lock-only package would be invisible to
Dependabot forever with nothing to notice. This gate converts the coincidence
into an invariant.

OFFLINE BY DESIGN
-----------------
The parsed-manifest set is derived from repo content on disk, not from the
GitHub API. A gate that needs network and a token is a gate that silently skips
in half the places it should run. The filename rule below was calibrated ONCE
against the live dependency graph and is hardcoded; see PARSED_MANIFEST_RULE.

EXIT CODES
----------
0  every locked distribution is covered by a parsed manifest
1  one or more locked distributions are covered by nothing (findings)
2  could not run: lock missing, empty, or a line that cannot be classified
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_COULD_NOT_RUN = 2

DEFAULT_LOCK = "environments/requirements.lock"

# --------------------------------------------------------------------------
# The parsed-manifest rule, calibrated against the live dependency graph.
# --------------------------------------------------------------------------
#
# MEASURED 2026-09-19 against Gavin-Borges/SESTRAV at 21bbcac with:
#
#     gh api graphql -f query='query { repository(owner:"Gavin-Borges",
#       name:"SESTRAV") { dependencyGraphManifests(first:100) {
#       totalCount nodes { filename parseable exceedsMaxSize } } } }'
#
# 31 manifests came back. 16 are .github/workflows/*.yml, which belong to the
# GitHub Actions ecosystem and declare no Python distributions, so they are
# outside this gate's scope. The remaining 15 are the pip-ecosystem set:
#
#     pyproject.toml
#     requirements.txt
#     environments/requirements-ci-build.txt
#     environments/requirements-ci-mypy.txt
#     environments/requirements-ci-pytest-cov.txt
#     environments/requirements-ci-render.txt
#     environments/requirements-ci-ruff.txt
#     environments/requirements-ci-torch-cpu.txt
#     environments/requirements-ci.txt
#     environments/requirements-pip-audit.txt
#     environments/requirements-pip-bootstrap.txt
#     environments/requirements-sbom.txt
#     environments/requirements-security.txt
#     environments/requirements-semgrep.txt
#     environments/requirements-lock.in
#
# Three observations fix the rule:
#
#   1. ALL 13 tracked requirements*.txt files are parsed. None is missing.
#   2. pyproject.toml is parsed.
#   3. Exactly 1 of the 10 tracked requirements*.in files is parsed, and it is
#      the only one with no compiled <stem>.txt sibling beside it.
#      requirements-lock.in compiles to requirements.lock, not to
#      requirements-lock.txt. The other nine each sit next to their own
#      compiled .txt and are absent from the graph, consistent with GitHub
#      deduplicating a pip-compile input against its emitted output.
#      That is a 10-of-10 fit, with no exceptions in either direction.
#   4. environments/requirements.lock is NOT parsed, which is the whole reason
#      for this gate. Neither are environment.yml, environments/predig.yaml or
#      environments/prime.yaml (conda, not pip).
#
# The rule encoded below reproduces all 15 measured pip manifests exactly, with
# no false positives and no false negatives. It is a FILENAME PATTERN rule, not
# a hardcoded file list, so a newly added requirements file is picked up
# automatically.
#
# The .in half of the rule is an empirical fit to observed behaviour rather
# than to published documentation, and GitHub can change it without notice.
# It is the PERMISSIVE half (it grows the covered set), so if GitHub stops
# parsing that file this gate would pass where it should fail. Re-run the
# calibration query above when the dependency graph is next inspected.
PARSED_MANIFEST_RULE = (
    "pyproject.toml at the repository root; any requirements*.txt at any "
    "depth; any requirements*.in at any depth that has no compiled "
    "<stem>.txt sibling in the same directory"
)
PARSED_MANIFEST_RULE_MEASURED = "2026-09-19"

# Directories never walked in the no-git fallback. .claude/worktrees holds FULL
# DUPLICATE CHECKOUTS parked on other branches, so a requirements file found
# there would credit coverage from code that is not on this branch.
SKIP_DIRS = frozenset(
    {
        ".git",
        ".claude",
        ".agents",
        ".cursor",
        "_local",
        ".venv",
        "venv",
        ".ci_test_venv",
        "node_modules",
        "build",
        "dist",
        "site-packages",
        "__pycache__",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".eggs",
    }
)

# Global pip options that carry no distribution name. Recognised so they are
# not mistaken for a requirement and not silently dropped either.
NO_DISTRIBUTION_OPTIONS = (
    "--index-url",
    "--extra-index-url",
    "--no-index",
    "--find-links",
    "--trusted-host",
    "--no-binary",
    "--only-binary",
    "--prefer-binary",
    "--require-hashes",
    "--pre",
    "--use-feature",
    "--use-deprecated",
    "--no-deps",
    "--python-version",
    "--platform",
    "--implementation",
    "--abi",
    "--target",
    "-i",
    "-f",
)

INCLUDE_OPTIONS = ("-r", "--requirement", "-c", "--constraint")
EDITABLE_OPTIONS = ("-e", "--editable")

# pip's own comment regex: a '#' at line start or after whitespace.
_COMMENT_RE = re.compile(r"(^|\s+)#.*$")
_HASH_RE = re.compile(r"\s--hash[= ]\S+")
# PEP 508 name, then an optional extras group, then whatever follows.
_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$", re.DOTALL)
# One version-specifier clause, e.g. '== 2.5.0' or '>=1.24.0'. Deliberately
# permissive about the version itself and strict about there being NOTHING
# after it: trailing text is how two requirements on one physical line would
# slip through as one, silently dropping the second.
_SPEC_CLAUSE_RE = re.compile(r"^(===|==|>=|<=|~=|!=|<|>)\s*[A-Za-z0-9*][A-Za-z0-9*.+!_-]*$")


def _tail_is_classifiable(tail: str) -> bool:
    """Is the text after a distribution name something this tool understands?

    Accepts a bare name, a PEP 508 direct reference, or a comma-separated list
    of version specifiers. Everything else is refused so it surfaces as a parse
    error instead of a silently half-read line.
    """
    tail = tail.strip()
    if not tail:
        return True
    if tail.startswith("@"):
        return True
    return all(_SPEC_CLAUSE_RE.match(clause.strip()) for clause in tail.split(",") if clause.strip())


class ManifestParseError(Exception):
    """A line that could not be classified.

    Raised rather than skipped on purpose. A line this tool skips is a package
    it does not monitor, which is the exact defect the gate exists to catch.
    """

    def __init__(self, path: str, lineno: int, line: str, reason: str) -> None:
        self.path = path
        self.lineno = lineno
        self.line = line
        self.reason = reason
        super().__init__(
            "{} line {}: {}\n    offending text: {!r}".format(path, lineno, reason, line[:200])
        )


@dataclass
class Requirement:
    """One declared distribution and where it was declared."""

    name: str
    normalized: str
    path: str
    lineno: int
    raw: str


@dataclass
class ParseResult:
    requirements: list[Requirement] = field(default_factory=list)
    includes: list[tuple[str, int, str]] = field(default_factory=list)
    editables: list[tuple[str, int, str]] = field(default_factory=list)


def normalize(name: str) -> str:
    """PEP 503 normalization: lowercase, runs of -_. collapsed to a single -.

    Flask_Login, flask-login and flask.login are one project. Getting this
    wrong manufactures false failures.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


# --------------------------------------------------------------------------
# Requirements-file parsing
# --------------------------------------------------------------------------


def logical_lines(text: str) -> list[tuple[int, str]]:
    """Join backslash continuations, then strip comments, in pip's order.

    A comment line never continues, even when it ends in a backslash. The CI
    render manifest's header contains exactly that shape, so getting this
    wrong turns a documentation comment into a phantom requirement.
    """
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    start = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        is_comment = bool(_COMMENT_RE.match(raw))
        continues = stripped.endswith("\\") and not is_comment
        if continues:
            if not buf:
                start = lineno
            buf.append(stripped[:-1])
            continue
        if buf:
            buf.append(stripped)
            out.append((start, " ".join(part.strip() for part in buf).strip()))
            buf = []
            continue
        out.append((lineno, stripped))
    if buf:
        out.append((start, " ".join(part.strip() for part in buf).strip()))

    cleaned: list[tuple[int, str]] = []
    for lineno, line in out:
        line = _COMMENT_RE.sub("", line).strip()
        if line:
            cleaned.append((lineno, line))
    return cleaned


def _split_marker(text: str) -> str:
    """Drop a PEP 508 environment marker. A marker never changes the project."""
    return text.split(";", 1)[0].strip()


def parse_requirement_line(path: str, lineno: int, line: str) -> Requirement | None:
    """Return the Requirement a line declares, or None for a recognised non-requirement.

    Raises ManifestParseError for anything that cannot be classified.
    """
    body = _HASH_RE.sub("", " " + line).strip()
    if not body:
        raise ManifestParseError(path, lineno, line, "line is only --hash tokens, with no requirement")

    head = body.split(None, 1)[0]
    if head in INCLUDE_OPTIONS or head.split("=", 1)[0] in INCLUDE_OPTIONS:
        return None
    if head in EDITABLE_OPTIONS:
        return None
    if head in NO_DISTRIBUTION_OPTIONS or head.split("=", 1)[0] in NO_DISTRIBUTION_OPTIONS:
        return None
    if head.startswith("-"):
        raise ManifestParseError(
            path, lineno, line, "unrecognised pip option {!r}; classify it before trusting this gate".format(head)
        )

    spec = _split_marker(body)
    if not spec:
        raise ManifestParseError(path, lineno, line, "environment marker with no requirement before it")

    match = _NAME_RE.match(spec)
    if not match:
        raise ManifestParseError(
            path, lineno, line, "does not begin with a PEP 508 distribution name (URL or path requirement?)"
        )
    name, _extras, tail = match.group(1), match.group(2), match.group(3).strip()
    if not _tail_is_classifiable(tail):
        raise ManifestParseError(
            path,
            lineno,
            line,
            "text after the distribution name {!r} is not a version specifier or URL: {!r}".format(name, tail[:80]),
        )
    return Requirement(name=name, normalized=normalize(name), path=path, lineno=lineno, raw=line)


def parse_requirements_file(root: Path, rel: str) -> ParseResult:
    result = ParseResult()
    text = (root / rel).read_text(encoding="utf-8")
    for lineno, line in logical_lines(text):
        body = _HASH_RE.sub("", " " + line).strip()
        head = body.split(None, 1)[0] if body else ""
        if head in INCLUDE_OPTIONS or head.split("=", 1)[0] in INCLUDE_OPTIONS:
            target = body.split(None, 1)[1].strip() if len(body.split(None, 1)) > 1 else body.split("=", 1)[-1].strip()
            if not target:
                raise ManifestParseError(rel, lineno, line, "include option with no target path")
            result.includes.append((rel, lineno, target))
            continue
        if head in EDITABLE_OPTIONS:
            result.editables.append((rel, lineno, body))
            continue
        req = parse_requirement_line(rel, lineno, line)
        if req is not None:
            result.requirements.append(req)
    return result


def parse_pyproject(root: Path, rel: str) -> ParseResult:
    """Read declared distributions out of a pyproject.toml.

    Covers [project].dependencies, every [project].optional-dependencies group,
    [build-system].requires and PEP 735 [dependency-groups].
    """
    result = ParseResult()
    path = root / rel
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ManifestParseError(rel, 0, "", "not valid TOML: {}".format(exc)) from exc

    buckets: list[tuple[str, list]] = []
    project = data.get("project", {})
    if isinstance(project.get("dependencies"), list):
        buckets.append(("project.dependencies", project["dependencies"]))
    optional = project.get("optional-dependencies", {})
    if isinstance(optional, dict):
        for group, items in sorted(optional.items()):
            if isinstance(items, list):
                buckets.append(("project.optional-dependencies." + group, items))
    build = data.get("build-system", {})
    if isinstance(build.get("requires"), list):
        buckets.append(("build-system.requires", build["requires"]))
    groups = data.get("dependency-groups", {})
    if isinstance(groups, dict):
        for group, items in sorted(groups.items()):
            if isinstance(items, list):
                buckets.append(("dependency-groups." + group, items))

    for label, items in buckets:
        for item in items:
            if not isinstance(item, str):
                # PEP 735 include-group tables and the like carry no name.
                continue
            req = parse_requirement_line("{} [{}]".format(rel, label), 0, item.strip())
            if req is not None:
                result.requirements.append(req)
    return result


# --------------------------------------------------------------------------
# Manifest discovery
# --------------------------------------------------------------------------


def _git_toplevel(root: Path) -> Path | None:
    """Resolve git's own idea of the work tree containing ``root``, or None."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=False,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, ValueError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.decode("utf-8", "replace").strip()
    if not out:
        return None
    return Path(out).resolve()


def _candidate_files(root: Path) -> list[str]:
    """Every file the repository tracks, as forward-slash relative paths.

    Prefers ``git ls-files`` because GitHub's dependency graph only ever sees
    TRACKED content: an untracked scratch requirements file must not be allowed
    to credit coverage. Falls back to a pruned walk outside a work tree.

    git is trusted ONLY when its own toplevel resolves to ``root``. Run against
    a directory nested inside some other repository, git walks upward and
    happily enumerates that repository instead, which would enumerate a file
    set from a tree nobody asked about.
    """
    if _git_toplevel(root) == root.resolve():
        try:
            proc = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-z"],
                capture_output=True,
                check=False,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            )
        except (OSError, ValueError):
            proc = None
        if proc is not None and proc.returncode == 0 and proc.stdout.strip():
            return [entry for entry in proc.stdout.decode("utf-8", "replace").split("\0") if entry]

    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            rel = Path(dirpath, filename).relative_to(root).as_posix()
            found.append(rel)
    return found


def is_parsed_manifest(rel: str, all_files: frozenset[str]) -> bool:
    """Apply PARSED_MANIFEST_RULE to one repo-relative path."""
    posix = rel.replace("\\", "/")
    if any(part in SKIP_DIRS for part in posix.split("/")[:-1]):
        return False
    name = posix.rsplit("/", 1)[-1]
    if posix == "pyproject.toml":
        return True
    if name.startswith("requirements") and name.endswith(".txt"):
        return True
    if name.startswith("requirements") and name.endswith(".in"):
        sibling = posix[: -len(".in")] + ".txt"
        return sibling not in all_files
    return False


def discover_parsed_manifests(root: Path) -> list[str]:
    files = _candidate_files(root)
    as_set = frozenset(f.replace("\\", "/") for f in files)
    return sorted(f for f in as_set if is_parsed_manifest(f, as_set))


# --------------------------------------------------------------------------
# The check
# --------------------------------------------------------------------------


def build_coverage(root: Path, manifests: list[str], follow_includes: bool) -> tuple[dict[str, list[Requirement]], list[str]]:
    """Map normalized distribution name -> the parsed-manifest declarations of it."""
    coverage: dict[str, list[Requirement]] = {}
    visited: set[str] = set()
    notes: list[str] = []
    queue = list(manifests)
    while queue:
        rel = queue.pop(0)
        if rel in visited:
            continue
        visited.add(rel)
        if not (root / rel).is_file():
            notes.append("manifest listed but not present on disk: {}".format(rel))
            continue
        if rel.endswith(".toml"):
            result = parse_pyproject(root, rel)
        else:
            result = parse_requirements_file(root, rel)
        for req in result.requirements:
            coverage.setdefault(req.normalized, []).append(req)
        for src, lineno, target in result.includes:
            resolved = (root / src).parent.joinpath(target).resolve()
            try:
                rel_target = resolved.relative_to(root.resolve()).as_posix()
            except ValueError:
                notes.append("include target outside the repository, not followed: {} from {}".format(target, src))
                continue
            if follow_includes:
                queue.append(rel_target)
            else:
                notes.append(
                    "include not followed (conservative default): {} declares -r/-c {} "
                    "-- run with --follow-includes to credit it".format(src, rel_target)
                )
        for src, lineno, body in result.editables:
            notes.append("editable install carries no resolvable distribution name: {} line {}".format(src, lineno))
    return coverage, notes


def run(root: Path, lock_rel: str, follow_includes: bool) -> tuple[int, dict]:
    report: dict = {
        "repo_root": str(root),
        "lock": lock_rel,
        "parsed_manifest_rule": PARSED_MANIFEST_RULE,
        "parsed_manifest_rule_measured": PARSED_MANIFEST_RULE_MEASURED,
        "follow_includes": follow_includes,
        "parsed_manifests": [],
        "locked_distribution_count": 0,
        "covered_count": 0,
        "uncovered": [],
        "notes": [],
        "status": "ok",
    }

    lock_path = root / lock_rel
    if not lock_path.is_file():
        report["status"] = "could_not_run"
        report["error"] = "lock file not found: {}".format(lock_rel)
        return EXIT_COULD_NOT_RUN, report

    try:
        lock_result = parse_requirements_file(root, lock_rel)
    except ManifestParseError as exc:
        report["status"] = "could_not_run"
        report["error"] = "unparseable line in the lock: {}".format(exc)
        return EXIT_COULD_NOT_RUN, report

    if not lock_result.requirements:
        report["status"] = "could_not_run"
        report["error"] = (
            "lock file {} declares no distributions. An empty lock cannot be "
            "certified as covered; it is a broken input, not a clean result.".format(lock_rel)
        )
        return EXIT_COULD_NOT_RUN, report

    manifests = discover_parsed_manifests(root)
    manifests = [m for m in manifests if m.replace("\\", "/") != lock_rel.replace("\\", "/")]
    report["parsed_manifests"] = manifests
    if not manifests:
        report["status"] = "could_not_run"
        report["error"] = (
            "no GitHub-parsed manifest found under {}. With nothing to compare "
            "against, every package would report uncovered; that is a broken "
            "input, not a finding.".format(root)
        )
        return EXIT_COULD_NOT_RUN, report

    try:
        coverage, notes = build_coverage(root, manifests, follow_includes)
    except ManifestParseError as exc:
        report["status"] = "could_not_run"
        report["error"] = "unparseable line in a parsed manifest: {}".format(exc)
        return EXIT_COULD_NOT_RUN, report

    report["notes"] = notes

    seen: set[str] = set()
    uncovered: list[dict] = []
    covered = 0
    for req in lock_result.requirements:
        if req.normalized in seen:
            continue
        seen.add(req.normalized)
        if req.normalized in coverage:
            covered += 1
        else:
            uncovered.append(
                {
                    "name": req.name,
                    "normalized": req.normalized,
                    "lock_line": req.lineno,
                    "lock_text": req.raw,
                }
            )

    report["locked_distribution_count"] = len(seen)
    report["covered_count"] = covered
    report["uncovered"] = uncovered
    report["status"] = "findings" if uncovered else "ok"
    return (EXIT_FINDINGS if uncovered else EXIT_OK), report


def render(report: dict, exit_code: int) -> str:
    lines: list[str] = []
    if exit_code == EXIT_COULD_NOT_RUN:
        lines.append("COULD NOT RUN -- this gate certified NOTHING.")
        lines.append("  repo root: {}".format(report["repo_root"]))
        lines.append("  reason:    {}".format(report.get("error", "unknown")))
        lines.append("")
        lines.append("Exit code 2 is not a pass. Fix the input and re-run.")
        return "\n".join(lines)

    lines.append("Lock manifest coverage")
    lines.append("  repo root:        {}".format(report["repo_root"]))
    lines.append("  lock:             {}".format(report["lock"]))
    lines.append("  parsed manifests: {} (rule calibrated {})".format(len(report["parsed_manifests"]), report["parsed_manifest_rule_measured"]))
    for manifest in report["parsed_manifests"]:
        lines.append("      {}".format(manifest))
    lines.append("  locked distributions: {}".format(report["locked_distribution_count"]))
    lines.append("  covered:              {}".format(report["covered_count"]))
    lines.append("  uncovered:            {}".format(len(report["uncovered"])))

    if report["uncovered"]:
        lines.append("")
        lines.append("UNCOVERED -- Dependabot cannot see these distributions.")
        lines.append("They are pinned in the lock, which GitHub's dependency graph does not")
        lines.append("parse, and they are declared in no manifest that it does parse. No")
        lines.append("advisory against any of them will ever raise an alert on this repo.")
        lines.append("")
        for item in report["uncovered"]:
            lines.append("  {}".format(item["name"]))
            lines.append("      normalized:  {}".format(item["normalized"]))
            lines.append("      declared at: {} line {}".format(report["lock"], item["lock_line"]))
            lines.append("      lock text:   {}".format(item["lock_text"][:160]))
        lines.append("")
        lines.append("Fix by declaring each one in a parsed manifest, most naturally by adding")
        lines.append("it to the lock's compile input, which IS parsed, and recompiling.")
    else:
        lines.append("")
        lines.append("OK: every distribution pinned in the lock is declared in at least one")
        lines.append("manifest that GitHub's dependency graph parses.")

    if report["notes"]:
        lines.append("")
        lines.append("Notes ({}):".format(len(report["notes"])))
        for note in report["notes"]:
            lines.append("  - {}".format(note))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if the pip lockfile pins a distribution that no GitHub-parsed "
            "manifest declares, which would make it permanently invisible to Dependabot."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="repository root to check; defaults to the checkout containing this script",
    )
    parser.add_argument("--lock", default=DEFAULT_LOCK, help="lock file, relative to the repo root")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--follow-includes",
        action="store_true",
        help=(
            "credit coverage through -r/-c includes reached from a parsed manifest. "
            "Off by default: whether GitHub resolves includes is unmeasured here, and "
            "crediting them unmeasured would risk a false pass."
        ),
    )
    args = parser.parse_args(argv)

    # Resolved from this script's own location, not the working directory, so a
    # run from a worktree certifies the tree the script was checked out into.
    # The resolved root is printed on every run so the reader can see WHICH tree
    # was certified rather than assuming.
    root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]

    exit_code, report = run(root, args.lock.replace("\\", "/"), args.follow_includes)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report, exit_code))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
