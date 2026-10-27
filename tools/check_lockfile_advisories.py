#!/usr/bin/env python
"""Fail-closed check that every pip-audit finding on the production lockfile is accepted.

Two structural gaps let CVE-2025-3000 sit patched-but-unnoticed in this repo for four
weeks after torch 2.13.0 shipped, and they compound:

1. **Dependabot never sees a vulnerable pin that exists only in the production lock.**
   GitHub's dependency graph parses the `requirements*.txt` locks and
   `environments/requirements-lock.in`, but not `environments/requirements.lock`,
   the file this gate audits. Of the 130 Dependabot alerts that can still be listed
   (measured 2026-09-30; 9 more are withdrawn and unreadable), zero carry a
   `manifest_path` ending in `.lock`.
2. **CI's pip-audit job could not have failed on it even if Dependabot had.** Every
   `pip-audit` invocation in `.github/workflows/security.yml` was `continue-on-error`,
   AND every one of them carried permanent `--ignore-vuln` flags for this exact
   advisory with no scheduled re-review. A CLI suppression flag is invisible in any
   report pip-audit itself produces - it is applied before the JSON is even written, so
   nothing that reads pip-audit's own output can ever notice the finding, whether it is
   still vulnerable or has since been fixed.

This script closes the second gap. It reads a pip-audit JSON report that MUST be
generated without any `--ignore-vuln` flags - suppression logic moves entirely into
`environments/accepted_advisories.toml`, a single auditable file, checked against
`SECURITY.md`'s Risk-Acceptance Register rather than scattered across CLI flags in a
workflow file that no test exercises.

Policy: every IN-SCOPE finding must be either fixed or explicitly accepted, scoped to
the specific package it was found on (an advisory ID accepted for one package does not
cover the same ID surfacing through a different one - that combination has not been
assessed). There is no severity filter: pip-audit's JSON does not reliably carry a
severity field, resolving one would add a network call and a token requirement to a
gate that should stay hermetic, and failing closed on everything unaccepted is both
simpler and stricter.

Acceptances are also scoped to ONE LOCK (added 2026-10-17, with
`tools/audit_all_locks.py`, which runs this policy over every tracked hash-pinned lock).
An entry's optional `lock` field names the repo-relative lock it covers, and an entry
with no `lock` field covers `environments/requirements.lock` only - which is what every
entry meant before the field existed, since that was the only lock this gate read. So
accepting an advisory for the production lock never silently accepts the same (id,
package) pinned in a CI tool lock: that pin installs somewhere else, for another reason,
and its exposure has not been assessed. The lock a report is judged against is
`--lock-id` when given, else `--lockfile`'s path relative to the repository, else (for a
file outside the repository) that file's own resolved path, which no acceptance names -
an out-of-tree copy of a lock is judged with nothing accepted rather than borrowing the
production lock's list. The Risk-Acceptance Register check below still covers every
entry in the file, whichever lock it names.

Scope: only packages this repo actually pins in the lock being judged
(`environments/requirements.lock` unless `--lockfile` names another).
Confirmed empirically (a real pip-audit run against a live dev environment, not just
read from docs) that the installed set an audited venv sees is wider than that: `pip`
itself showed a live PYSEC finding despite never appearing as a pin in the lockfile -
it is whatever the CI runner's Python bootstrap ships, entirely outside this repo's
control, with no pin here to bump and no clean remediation path. Gating on it would
block every PR on a finding this repo cannot act on. Findings on out-of-scope packages
are reported for visibility but never required in the acceptance file and never fail
the build - this mirrors Dependabot's own scope, which likewise never alerts on a
package absent from a tracked manifest.

Usage:
    python tools/check_lockfile_advisories.py pip-audit-report.json
    python tools/check_lockfile_advisories.py report.json --accept path/to/accepted.toml \
        --lockfile path/to/requirements.lock
    python tools/check_lockfile_advisories.py report.json \
        --lockfile environments/requirements-semgrep.txt   # judged with that lock's entries

CI usage (see .github/workflows/security.yml, job `pip-audit`):
    pip-audit --format json --output pip-audit-report.json   # no --ignore-vuln
    python tools/check_lockfile_advisories.py pip-audit-report.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_ACCEPT_FILE = "environments/accepted_advisories.toml"
DEFAULT_REGISTER_FILE = "SECURITY.md"
# The lock an acceptance entry with no `lock` field covers. Before per-lock scoping
# this was the only lock the gate read, so absence keeps every older entry's meaning.
DEFAULT_LOCKFILE = "environments/requirements.lock"


@dataclass(frozen=True)
class Finding:
    package: str
    version: str
    advisory_id: str
    fix_versions: tuple[str, ...]


@dataclass(frozen=True)
class Acceptance:
    advisory_id: str
    package: str
    reason: str
    register_entry: str
    lock: str = DEFAULT_LOCKFILE


class ReportError(RuntimeError):
    """The report could not be parsed, or does not look like a real audit."""


def _normalize(name: str) -> str:
    """PEP 503 form, the one pip-audit's JSON reports: lowercase, ``-_.`` runs to ``-``.

    Until 2026-10-17 only ``_`` was folded, so a lock pin spelled ``zope.interface``
    and pip-audit's ``zope-interface`` never matched, and a finding on it would have
    been classed out of scope and never gated. No tracked lock pins a dotted name
    today (measured 2026-10-17), so this changed no verdict.
    """
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def parse_report(text: str) -> list[Finding]:
    """Extract findings from a `pip-audit --format json` report.

    Raises ReportError on anything that is not a populated audit - an empty or
    malformed report must never read as a clean pass, since that silently disables
    this entire gate.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReportError(f"report is not valid JSON: {exc}") from exc

    dependencies = data.get("dependencies")
    if not isinstance(dependencies, list):
        raise ReportError("report has no 'dependencies' list")
    if not dependencies:
        raise ReportError("report audited zero packages - tooling failure, not a clean scan")

    findings: list[Finding] = []
    for dep in dependencies:
        name = _normalize(str(dep.get("name", "")))
        version = str(dep.get("version", "")).strip()
        for vuln in dep.get("vulns", []) or []:
            findings.append(
                Finding(
                    package=name,
                    version=version,
                    advisory_id=str(vuln.get("id", "")).strip(),
                    fix_versions=tuple(str(v) for v in (vuln.get("fix_versions") or [])),
                )
            )
    return findings


def normalize_lock(value: str) -> str:
    """Spell a repo-relative lock path one way: forward slashes, no leading ``./``."""
    lock = value.strip().replace("\\", "/")
    while lock.startswith("./"):
        lock = lock[2:]
    return lock


def lock_identity(lockfile: str | pathlib.Path) -> str:
    """The name acceptances use for ``lockfile``: its repo-relative POSIX path.

    A file outside the repository has no repo-relative name, so its own resolved
    path is used, which no acceptance entry names. That is deliberate: an
    out-of-tree copy of a lock is judged with nothing accepted, never with the
    production lock's acceptances borrowed for it.
    """
    resolved = pathlib.Path(lockfile).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_all_acceptances(path: pathlib.Path) -> dict[tuple[str, str, str], Acceptance]:
    """Load every acceptance in the file, keyed by (advisory_id, package, lock).

    The optional ``lock`` field names the repo-relative lock an entry covers; an
    entry without one covers ``DEFAULT_LOCKFILE``, which preserves the meaning of
    every entry written before the field existed. A ``lock`` that is present but
    empty or not a string raises ValueError rather than quietly covering nothing.

    A missing file means nothing is accepted - every finding must fail closed rather
    than silently pass because the seed file was never created.
    """
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str, str], Acceptance] = {}
    for entry in data.get("accepted", []):
        advisory_id = str(entry["id"]).strip()
        package = _normalize(str(entry["package"]))
        raw_lock = entry.get("lock", DEFAULT_LOCKFILE)
        if not isinstance(raw_lock, str) or not normalize_lock(raw_lock):
            raise ValueError(
                f"acceptance {advisory_id} for {package} has an invalid lock {raw_lock!r}: "
                f"name a repo-relative lock path, or omit the field to mean {DEFAULT_LOCKFILE}"
            )
        lock = normalize_lock(raw_lock)
        out[(advisory_id, package, lock)] = Acceptance(
            advisory_id=advisory_id,
            package=package,
            reason=str(entry["reason"]).strip(),
            register_entry=str(entry["register_entry"]).strip(),
            lock=lock,
        )
    return out


def acceptances_for_lock(
    acceptances: Mapping[tuple[str, str, str], Acceptance], lock: str
) -> dict[tuple[str, str], Acceptance]:
    """The entries covering ``lock``, keyed by (advisory_id, package) for evaluate()."""
    wanted = normalize_lock(lock)
    return {
        (acceptance.advisory_id, acceptance.package): acceptance
        for acceptance in acceptances.values()
        if acceptance.lock == wanted
    }


def load_acceptances(
    path: pathlib.Path, lock: str = DEFAULT_LOCKFILE
) -> dict[tuple[str, str], Acceptance]:
    """Load the acceptances covering one lock, keyed by (advisory_id, package).

    ``lock`` defaults to ``DEFAULT_LOCKFILE``, so a caller written before per-lock
    scoping still gets exactly the entries it used to: all of them, since none
    named a lock. Entries for other locks are left out, never merged in.
    """
    return acceptances_for_lock(load_all_acceptances(path), lock)


def parse_lockfile_packages(path: pathlib.Path) -> set[str]:
    """Return the normalized set of package names pinned by a compiled lockfile.

    Only exact `name==version` lines count as a pin this repo controls. This is
    deliberately narrower than "everything pip-audit finds installed": an audited
    venv also contains build-time tooling (pip itself, whatever the CI runner's
    Python bootstrap ships) that never appears in the lockfile at all, has no pin
    here to bump, and is not this repo's dependency choice to answer for.
    """
    packages: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip().rstrip("\\").strip()
        if "==" not in line:
            continue
        # `pkg[extra]==1.0` pins the distribution `pkg`, which is the name pip-audit
        # reports; keeping the brackets would class its findings out of scope.
        name = line.split("==", 1)[0].split("[", 1)[0].strip()
        if name:
            packages.add(_normalize(name))
    return packages


def filter_in_scope(
    findings: list[Finding], in_scope_packages: set[str]
) -> tuple[list[Finding], list[Finding]]:
    """Split findings into (in-scope, out-of-scope) by lockfile package membership."""
    in_scope = [f for f in findings if f.package in in_scope_packages]
    out_of_scope = [f for f in findings if f.package not in in_scope_packages]
    return in_scope, out_of_scope


def unregistered_acceptances(
    acceptances: (
        Mapping[tuple[str, str], Acceptance] | Mapping[tuple[str, str, str], Acceptance]
    ),
    register_text: str,
) -> list[Acceptance]:
    """Return acceptances whose ``register_entry`` is absent from the register text.

    ``environments/accepted_advisories.toml`` states that every entry MUST have a
    matching Risk-Acceptance Register entry in SECURITY.md carrying a re-review
    trigger. Until now nothing checked that: ``register_entry`` was parsed into the
    ``Acceptance`` dataclass and then never read again, and the only other mentions
    of SECURITY.md in this file are static strings in messages. An unenforced MUST on
    the sole suppression mechanism for a security gate is the worst place to have one.

    The match is a plain substring test, deliberately. No format for a register entry
    is specified anywhere, so anything cleverer would be guessing at one. An empty
    ``register_entry`` never matches, which is correct: the TOML requires a value.

    Takes either keying - one lock's (id, package) view or the whole file's
    (id, package, lock) view - since only the values are read.
    """
    missing = [
        acceptance
        for acceptance in acceptances.values()
        if not acceptance.register_entry or acceptance.register_entry not in register_text
    ]
    return sorted(missing, key=lambda a: (a.advisory_id, a.package, a.lock))


def evaluate(
    findings: list[Finding], acceptances: dict[tuple[str, str], Acceptance]
) -> tuple[list[Finding], list[tuple[str, str]]]:
    """Return (unaccepted findings, stale acceptance keys)."""
    unaccepted = [f for f in findings if (f.advisory_id, f.package) not in acceptances]
    seen = {(f.advisory_id, f.package) for f in findings}
    stale = sorted(key for key in acceptances if key not in seen)
    return unaccepted, stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="path to a pip-audit --format json report")
    parser.add_argument(
        "--accept",
        default=str(REPO_ROOT / DEFAULT_ACCEPT_FILE),
        help="TOML file listing explicitly accepted (advisory id, package, lock) entries",
    )
    parser.add_argument(
        "--lockfile",
        default=str(REPO_ROOT / DEFAULT_LOCKFILE),
        help="compiled lockfile defining which packages are in scope for this gate",
    )
    parser.add_argument(
        "--lock-id",
        default=None,
        help=(
            "repo-relative lock whose acceptances apply (default: --lockfile's path "
            "relative to the repository; a file outside it gets no acceptances)"
        ),
    )
    parser.add_argument(
        "--register",
        default=str(REPO_ROOT / DEFAULT_REGISTER_FILE),
        help="file that must carry a Risk-Acceptance Register entry per acceptance",
    )
    args = parser.parse_args(argv)

    try:
        findings = parse_report(pathlib.Path(args.report).read_text(encoding="utf-8"))
    except (OSError, ReportError) as exc:
        print(f"::error::cannot evaluate lockfile advisories: {exc}")
        return 2

    try:
        in_scope_packages = parse_lockfile_packages(pathlib.Path(args.lockfile))
    except OSError as exc:
        print(f"::error::cannot read lockfile for scoping: {exc}")
        return 2

    in_scope, out_of_scope = filter_in_scope(findings, in_scope_packages)
    for finding in out_of_scope:
        print(
            f"::notice::{finding.advisory_id} affects {finding.package}=="
            f"{finding.version}, which is not a pin in {args.lockfile} (likely CI/build "
            f"tooling outside this repo's control) - reported for visibility, not gated"
        )

    accept_path = pathlib.Path(args.accept)
    lock_id = (
        normalize_lock(args.lock_id) if args.lock_id is not None else lock_identity(args.lockfile)
    )
    all_acceptances = load_all_acceptances(accept_path)
    acceptances = acceptances_for_lock(all_acceptances, lock_id)

    # The register is only consulted when something is actually accepted, so a repo
    # with an empty acceptance list does not depend on SECURITY.md being present. It
    # covers every entry in the file, not only this lock's: the requirement is on the
    # file, and an unregistered entry for another lock is no less unjustified.
    if all_acceptances:
        register_path = pathlib.Path(args.register)
        try:
            register_text = register_path.read_text(encoding="utf-8")
        except OSError as exc:
            # Cannot verify is not the same as verified, and an unreadable register
            # must not let an unjustified suppression through.
            print(
                f"::error::cannot read {register_path} to verify acceptance register "
                f"entries: {exc}"
            )
            return 2
        unregistered = unregistered_acceptances(all_acceptances, register_text)
        if unregistered:
            for acceptance in unregistered:
                shown = acceptance.register_entry or "(empty)"
                print(
                    f"::error::{acceptance.advisory_id} for {acceptance.package} "
                    f"({acceptance.lock}) is accepted in {accept_path} with register_entry "
                    f"{shown!r}, which does not appear in {register_path}"
                )
            print(
                f"\n{len(unregistered)} acceptance(s) with no Risk-Acceptance Register "
                f"entry. {accept_path} requires one per entry, carrying a re-review "
                f"trigger."
            )
            return 1

    unaccepted, stale = evaluate(in_scope, acceptances)

    print(
        f"audited findings: {len(findings)} ({len(out_of_scope)} out of scope); "
        f"in-scope: {len(in_scope)}; accepted: {len(in_scope) - len(unaccepted)}"
    )

    for advisory_id, package in stale:
        # Not a failure: a stale acceptance means the advisory no longer appears in the
        # audit, most likely because it was fixed. This is exactly the signal that went
        # unnoticed for four weeks with CVE-2025-3000, so it is surfaced, not silenced.
        print(
            f"::notice::accepted advisory {advisory_id} for {package} ({lock_id}) no longer "
            f"appears in the audit - it may be resolved; consider retiring its {accept_path} "
            f"entry and the matching SECURITY.md register entry"
        )

    if not unaccepted:
        print("No unaccepted advisories in the audited dependency set.")
        return 0

    for finding in unaccepted:
        fix = ", ".join(finding.fix_versions) or "none published"
        print(
            f"::error::{finding.advisory_id} affects {finding.package}=="
            f"{finding.version} (fix: {fix}) and is not accepted for {lock_id} in "
            f"{accept_path}"
        )
    print(
        f"\n{len(unaccepted)} unaccepted advisory finding(s). Either upgrade the pin, or "
        f"add an entry to {accept_path} (with lock = \"{lock_id}\" unless it is "
        f"{DEFAULT_LOCKFILE}) together with a matching Risk-Acceptance Register entry in "
        f"SECURITY.md that carries a re-review trigger."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
