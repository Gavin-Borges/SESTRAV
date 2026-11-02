#!/usr/bin/env python
"""Audit EVERY tracked hash-pinned lock for published advisories, not only the production one.

WHY THIS EXISTS
---------------
An advisory published after a pin lands reached no CI job that can fail, for any lock
but one. Measured 2026-10-17: CVE-2026-59951 was published on 2026-10-15 against
mcp 1.29.0, a pin already on main in environments/requirements-semgrep.txt, and
nothing went red. Three gaps compound:

1. tools/check_lockfile_advisories.py, the only fail-closed advisory gate, judges a
   report of environments/requirements.lock and nothing else.
2. .github/workflows/dependency-review.yml judges only the versions a pull request
   CHANGES, so an advisory published after the pin merged never passes through it.
3. The bare `pip-audit --desc` in security.yml audits what that job's runner
   installed: the production lock plus pip-audit's own tool lock, not the rest.

`pip-audit -r <lock> --no-deps --disable-pip` over every tracked lock found exactly
that one finding. This script runs that audit for each lock and applies the existing
gate's policy to each, so the next one is seen the day it is published.

WHICH FILES ARE LOCKS
---------------------
The audit targets are the manifests tools/check_hash_pins.py treats as contractually
hash-pinned - its DEFAULT_TARGETS globs - imported here rather than listed again, so
one source answers both "must carry hashes" and "is audited", and a lock added under
those globs is audited the moment it exists. On 2026-10-17 the globs reached 16 files:
the 12 compiled outputs of check_lockfile_freshness.LOCKFILE_PAIRS (which equal
update_dependencies.LOCK_SPECS' outputs) plus 4 hand-curated hashed files with no
`.in` source (requirements-ci-render.txt, requirements-ci-torch-cpu.txt,
requirements-pip-bootstrap.txt, requirements-sbom.txt). `.in` sources are not targets:
they declare intent, not the artifacts anything installs. A lock outside those globs
would be missed by both gates, so tests/test_audit_all_locks.py derives the population
a second, independent way - tracked files whose own lines pin `name==version` with a
`--hash=sha256:` - and fails if any of them is not audited here.

HOW EACH LOCK IS AUDITED
------------------------
    python -m pip_audit -r <lock> --no-deps --disable-pip -s osv --format json --output <file>

`--no-deps --disable-pip` means no resolver, no throwaway venv and no install: the pins
exactly as written, which is also the form that runs on a Windows workstation where a
Linux-compiled lock's wheels do not exist. No `--ignore-vuln`, ever: a CLI suppression
is applied before the report is written, so nothing reading the report could notice it
(see check_lockfile_advisories.py for the four weeks that cost). Each report is then
judged with that gate's own functions - parse_report, filter_in_scope against the
lock's own pins, the acceptances whose `lock` names this lock, evaluate - and the
Risk-Acceptance Register check covers every acceptance in the file.

pip-audit's exit status is not the verdict. It exits 1 both when it finds an advisory
and when it fails, so the JSON report is the evidence, and a run that wrote none is a
tooling failure. Any report left in the output directory by an earlier run is deleted
first, so a failed run can never be judged on a stale report.

WHICH ADVISORY FEED
-------------------
`-s osv`, not pip-audit's default service, which is `pypi` (`pip-audit --help`, 2.10.1:
"(default: pypi)"). The PyPI feed lacks advisories that OSV carries. Measured 2026-10-23
with pip-audit 2.10.1 on a scratch lock of 15 older pins: OSV reported 131 distinct
(package, version, id) findings and PyPI 127. None was PyPI-only, and the 4 that only
OSV reported are all against pillow 12.3.0, CVE-2026-112403 (GHSA-8g5g-8vmr-9782, a heap
buffer overflow in ImagingJpegDecode) among them, with no alias of any of the 4 in PyPI's
report. Over the 16 tracked locks that day both services reported the same two findings,
so the switch changed no verdict then.

The switch has one cost, also measured that day. Only pip-audit's PyPI service marks a
pinned name it cannot find with `skip_reason`; with `-s osv` such a name is reported as
audited with no advisories. So the skip check under COMPLETENESS below cannot fire on an
OSV report, while the check for pinned names absent from the report still does.

DUPLICATE ROWS
--------------
pip-audit's JSON can list one advisory more than once for the same package, under either
service. On that scratch lock 75 of OSV's 206 rows, and 72 of PyPI's 199, repeated an id
already listed for the same package, differing only in the description, the aliases or a
fix version's spelling (26.2 and 26.2.0). judge_report counts each (package, version, id)
once and keeps every fix version, so the counts and the error lines are per advisory. The
verdict never depended on it. Two DIFFERENT ids naming one advisory (two PYSEC records for
one CVE, which pip-audit's own alias merge keeps apart) stay two findings, and each needs
its own acceptance: that direction fails closed.

COMPLETENESS
------------
A report that audited fewer packages than the lock pins reads exactly like a clean one.
So every name the lock pins must appear in its report, and a dependency pip-audit lists
as skipped (`skip_reason`, for example not found on PyPI) counts as NOT audited. Either
is a tooling failure for that lock, never a pass. Measured 2026-10-17 on Windows with
pip-audit 2.10.1 and its default service: every lock's report named exactly the packages
it pins, and none was skipped; again on 2026-10-23 under each service. Findings on a lock
with a tooling failure are still reported.

EXIT CODES
----------
0  every lock audited completely, and every in-scope finding accepted for its lock
1  an unaccepted finding on some lock, or an acceptance with no register entry
2  could not certify: no lock found, the acceptance file or register unreadable, or
   some lock not audited completely (no report, malformed report, pinned packages
   missing from it or skipped)

ADVISORY FIRST
--------------
Owner ruling 2026-10-17: this runs in security.yml's `pip-audit` job as a
`continue-on-error` step, to be made blocking after one green week. The script always
exits with its real verdict; whether that verdict blocks lives in the workflow only.

Usage:
    python tools/audit_all_locks.py
    python tools/audit_all_locks.py --report-dir pip-audit-all-locks
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

if __package__:  # imported as tools.audit_all_locks (the tests, or `python -m`)
    from tools import check_hash_pins as hash_pins
    from tools import check_lockfile_advisories as advisories
else:  # `python tools/audit_all_locks.py`: sys.path[0] is this file's own directory,
    # so these are the siblings beside it, never another checkout's copies.
    import check_hash_pins as hash_pins  # type: ignore[no-redef]
    import check_lockfile_advisories as advisories  # type: ignore[no-redef]

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_TIMEOUT_SECONDS = 180


@dataclass
class LockResult:
    """What one lock's audit showed. ``error`` set means it was NOT fully audited."""

    lock: str
    audited: int = 0
    skipped: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    in_scope: int = 0
    out_of_scope: list[advisories.Finding] = field(default_factory=list)
    unaccepted: list[advisories.Finding] = field(default_factory=list)
    stale: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None

    @property
    def accepted(self) -> int:
        return self.in_scope - len(self.unaccepted)

    @property
    def status(self) -> str:
        if self.error is not None:
            return "ERROR"
        return "FAIL" if self.unaccepted else "ok"

    def summary_line(self) -> str:
        counts = (
            f"{self.audited} audited, {len(self.skipped)} skipped, "
            f"{len(self.missing)} missing; {self.in_scope} finding(s), "
            f"{self.accepted} accepted, {len(self.unaccepted)} unaccepted"
        )
        tail = f" - {self.error}" if self.error is not None else ""
        return f"[{self.status:5}] {self.lock}: {counts}{tail}"

    def annotations(self, accept_path: str) -> list[str]:
        lines: list[str] = []
        if self.error is not None:
            lines.append(f"::error::{self.lock} was not fully audited: {self.error}")
        for finding in self.out_of_scope:
            lines.append(
                f"::notice::{finding.advisory_id} affects {finding.package}=="
                f"{finding.version}, which is not a pin in {self.lock} - reported for "
                f"visibility, not gated"
            )
        for advisory_id, package in self.stale:
            lines.append(
                f"::notice::accepted advisory {advisory_id} for {package} ({self.lock}) no "
                f"longer appears in the audit - it may be resolved; consider retiring its "
                f"{accept_path} entry and the matching SECURITY.md register entry"
            )
        for finding in self.unaccepted:
            fix = ", ".join(finding.fix_versions) or "none published"
            lines.append(
                f"::error::{finding.advisory_id} affects {finding.package}=="
                f"{finding.version} in {self.lock} (fix: {fix}) and is not accepted for "
                f"that lock in {accept_path}"
            )
        return lines


def discover_locks() -> list[str]:
    """Repo-relative POSIX paths of every lock to audit, from check_hash_pins' targets.

    A target named without a glob is kept even when the file is gone, so a deleted
    production lock is reported as unauditable rather than dropped from the run.
    """
    paths = hash_pins.resolve_targets(list(hash_pins.DEFAULT_TARGETS))
    return sorted({path.resolve().relative_to(REPO_ROOT).as_posix() for path in paths})


def report_name(lock: str) -> str:
    """A per-lock report filename that cannot collide across directories."""
    return lock.replace("/", "__") + ".json"


def pip_audit_command(lock_path: pathlib.Path, output_path: pathlib.Path) -> list[str]:
    """The audit for one lock: the pins as written, no resolution, no suppression.

    Against the OSV feed, not the default PyPI one; see WHICH ADVISORY FEED above.
    """
    return [
        sys.executable,
        "-m",
        "pip_audit",
        "-r",
        str(lock_path),
        "--no-deps",
        "--disable-pip",
        "-s",
        "osv",
        "--progress-spinner",
        "off",
        "--format",
        "json",
        "--output",
        str(output_path),
    ]


def run_pip_audit(command: Sequence[str], timeout: int) -> tuple[int, str]:
    """Run one audit; return (exit status, last stderr line). Never raises."""
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return -1, f"timed out after {timeout}s"
    except OSError as exc:
        return -1, f"could not start: {exc}"
    lines = [line for line in (result.stderr or "").splitlines() if line.strip()]
    return result.returncode, lines[-1].strip() if lines else ""


def distinct_findings(findings: Sequence[advisories.Finding]) -> list[advisories.Finding]:
    """One finding per (package, version, advisory id), first-seen order.

    A repeated row adds only the fix versions not already listed. See DUPLICATE ROWS
    above for why a report repeats rows at all.
    """
    merged: dict[tuple[str, str, str], advisories.Finding] = {}
    for finding in findings:
        key = (finding.package, finding.version, finding.advisory_id)
        seen = merged.get(key)
        if seen is None:
            merged[key] = finding
            continue
        extra = tuple(v for v in finding.fix_versions if v not in seen.fix_versions)
        if extra:
            merged[key] = replace(seen, fix_versions=seen.fix_versions + extra)
    return list(merged.values())


def judge_report(
    lock: str,
    report_text: str,
    pinned: set[str],
    acceptances: Mapping[tuple[str, str], advisories.Acceptance],
) -> LockResult:
    """Judge one lock's pip-audit JSON against its own pins and acceptances.

    Names are compared in PEP 503 form on both sides, the form pip-audit reports, so
    a lock spelling `typing_extensions` or `zope.interface` still matches. A row the
    report repeats is judged once.
    """
    try:
        findings = distinct_findings(advisories.parse_report(report_text))
    except advisories.ReportError as exc:
        return LockResult(lock=lock, error=str(exc))
    pinned = {advisories._normalize(name) for name in pinned}

    reported: set[str] = set()
    skipped: list[str] = []
    skipped_names: set[str] = set()
    for dep in json.loads(report_text)["dependencies"]:
        name = advisories._normalize(str(dep.get("name", "")))
        if "skip_reason" in dep:
            skipped.append(f"{name} ({dep.get('skip_reason')})")
            skipped_names.add(name)
        else:
            reported.add(name)
    missing = sorted(pinned - reported - skipped_names)

    in_scope, out_of_scope = advisories.filter_in_scope(findings, pinned)
    unaccepted, stale = advisories.evaluate(in_scope, dict(acceptances))

    problems = []
    if skipped:
        problems.append(f"pip-audit skipped {len(skipped)}: {', '.join(skipped)}")
    if missing:
        problems.append(f"{len(missing)} pinned package(s) absent from the report: {', '.join(missing)}")
    return LockResult(
        lock=lock,
        audited=len(reported),
        skipped=tuple(skipped),
        missing=tuple(missing),
        in_scope=len(in_scope),
        out_of_scope=out_of_scope,
        unaccepted=unaccepted,
        stale=stale,
        error="; ".join(problems) if problems else None,
    )


def audit_lock(
    lock: str,
    report_path: pathlib.Path,
    acceptances: Mapping[tuple[str, str], advisories.Acceptance],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> LockResult:
    """Run pip-audit on one lock and judge what it wrote."""
    lock_path = REPO_ROOT / lock
    try:
        pinned = advisories.parse_lockfile_packages(lock_path)
    except OSError as exc:
        return LockResult(lock=lock, error=f"cannot read the lock: {exc}")

    # Never judge a report an earlier run left behind: a failed run must read as failed.
    report_path.unlink(missing_ok=True)
    status, detail = run_pip_audit(pip_audit_command(lock_path, report_path), timeout)
    try:
        report_text = report_path.read_text(encoding="utf-8")
    except OSError:
        said = f": {detail}" if detail else ""
        return LockResult(lock=lock, error=f"pip-audit exited {status} and wrote no report{said}")
    return judge_report(lock, report_text, pinned, acceptances)


def exit_code(results: Sequence[LockResult], register_failed: bool = False) -> int:
    if any(result.error is not None for result in results):
        return 2
    if register_failed or any(result.unaccepted for result in results):
        return 1
    return 0


def total_line(results: Sequence[LockResult]) -> str:
    unaccepted = sum(len(result.unaccepted) for result in results)
    failing = sum(1 for result in results if result.unaccepted)
    errors = sum(1 for result in results if result.error is not None)
    return (
        f"TOTAL: {len(results)} lock(s); {sum(r.audited for r in results)} package pin(s) "
        f"audited; {sum(r.in_scope for r in results)} finding(s), "
        f"{sum(r.accepted for r in results)} accepted, {unaccepted} unaccepted in "
        f"{failing} lock(s); {errors} lock(s) not fully audited"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="directory for the per-lock pip-audit JSON reports (default: a new temp dir)",
    )
    parser.add_argument(
        "--accept",
        default=str(REPO_ROOT / advisories.DEFAULT_ACCEPT_FILE),
        help="TOML file of accepted (advisory id, package, lock) entries",
    )
    parser.add_argument(
        "--register",
        default=str(REPO_ROOT / advisories.DEFAULT_REGISTER_FILE),
        help="file that must carry a Risk-Acceptance Register entry per acceptance",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="seconds allowed for each lock's pip-audit run (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    locks = discover_locks()
    if not locks:
        print("::error::no hash-pinned lock found to audit - tooling failure, not a clean scan")
        return 2

    accept_path = pathlib.Path(args.accept)
    try:
        all_acceptances = advisories.load_all_acceptances(accept_path)
    except (OSError, tomllib.TOMLDecodeError, KeyError, ValueError) as exc:
        print(f"::error::cannot read {accept_path}: {exc!r}")
        return 2

    register_failed = False
    if all_acceptances:
        register_path = pathlib.Path(args.register)
        try:
            register_text = register_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"::error::cannot read {register_path} to verify acceptances: {exc}")
            return 2
        for acceptance in advisories.unregistered_acceptances(all_acceptances, register_text):
            register_failed = True
            print(
                f"::error::{acceptance.advisory_id} for {acceptance.package} "
                f"({acceptance.lock}) is accepted in {accept_path} with register_entry "
                f"{acceptance.register_entry or '(empty)'!r}, which does not appear in "
                f"{register_path}"
            )
        for acceptance in all_acceptances.values():
            if acceptance.lock not in locks:
                print(
                    f"::warning::{acceptance.advisory_id} for {acceptance.package} is "
                    f"accepted for {acceptance.lock}, which is not an audited lock, so it "
                    f"can never apply"
                )

    report_dir = (
        pathlib.Path(args.report_dir)
        if args.report_dir
        else pathlib.Path(tempfile.mkdtemp(prefix="pip-audit-all-locks-"))
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"Auditing {len(locks)} hash-pinned lock(s); reports in {report_dir}")

    results: list[LockResult] = []
    for lock in locks:
        result = audit_lock(
            lock,
            report_dir / report_name(lock),
            advisories.acceptances_for_lock(all_acceptances, lock),
            args.timeout,
        )
        results.append(result)
        print(result.summary_line())
        for line in result.annotations(str(accept_path)):
            print(line)

    print(total_line(results))
    code = exit_code(results, register_failed)
    if code == 1:
        print(
            f"Either upgrade each unaccepted pin, or add an entry to {accept_path} with "
            f"lock = \"<that lock>\" together with a matching Risk-Acceptance Register "
            f"entry in SECURITY.md that carries a re-review trigger."
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
