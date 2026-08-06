#!/usr/bin/env python
"""Fail-closed check that every pip-audit finding on the production lockfile is accepted.

Two structural gaps let CVE-2025-3000 sit patched-but-unnoticed in this repo for four
weeks after torch 2.13.0 shipped, and they compound:

1. **Dependabot never sees a vulnerable pin that exists only in a compiled lockfile.**
   It parses `.in` sources, not `.lock`/`.txt` artifacts. Across every alert this
   repository has ever had, zero carry a `manifest_path` ending in `.lock`.
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

Scope: only packages this repo actually pins in `environments/requirements.lock`.
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

CI usage (see .github/workflows/security.yml, job `pip-audit`):
    pip-audit --format json --output pip-audit-report.json   # no --ignore-vuln
    python tools/check_lockfile_advisories.py pip-audit-report.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tomllib
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_ACCEPT_FILE = "environments/accepted_advisories.toml"


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


class ReportError(RuntimeError):
    """The report could not be parsed, or does not look like a real audit."""


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-")


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


def load_acceptances(path: pathlib.Path) -> dict[tuple[str, str], Acceptance]:
    """Load the acceptance list, keyed by (advisory_id, package).

    A missing file means nothing is accepted - every finding must fail closed rather
    than silently pass because the seed file was never created.
    """
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], Acceptance] = {}
    for entry in data.get("accepted", []):
        advisory_id = str(entry["id"]).strip()
        package = _normalize(str(entry["package"]))
        out[(advisory_id, package)] = Acceptance(
            advisory_id=advisory_id,
            package=package,
            reason=str(entry["reason"]).strip(),
            register_entry=str(entry["register_entry"]).strip(),
        )
    return out


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
        name = line.split("==", 1)[0].strip()
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
        help="TOML file listing explicitly accepted (advisory id, package) pairs",
    )
    parser.add_argument(
        "--lockfile",
        default=str(REPO_ROOT / "environments" / "requirements.lock"),
        help="compiled lockfile defining which packages are in scope for this gate",
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
    acceptances = load_acceptances(accept_path)
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
            f"::notice::accepted advisory {advisory_id} for {package} no longer appears "
            f"in the audit - it may be resolved; consider retiring its {accept_path} entry "
            f"and the matching SECURITY.md register entry"
        )

    if not unaccepted:
        print("No unaccepted advisories in the audited dependency set.")
        return 0

    for finding in unaccepted:
        fix = ", ".join(finding.fix_versions) or "none published"
        print(
            f"::error::{finding.advisory_id} affects {finding.package}=="
            f"{finding.version} (fix: {fix}) and is not accepted in {accept_path}"
        )
    print(
        f"\n{len(unaccepted)} unaccepted advisory finding(s). Either upgrade the pin, or "
        f"add an entry to {accept_path} together with a matching Risk-Acceptance Register "
        f"entry in SECURITY.md that carries a re-review trigger."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
