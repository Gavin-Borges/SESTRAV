#!/usr/bin/env python
"""Fail if a dismissed code-scanning alert carries no written justification.

This repository currently has ZERO open code-scanning alerts and twenty-three
dismissed ones. With nothing open, the security posture IS the dismissal
quality: every remaining risk decision lives in a `dismissed_reason` plus a
free-text `dismissed_comment`, and nothing in the repo reads either. Nine of
the twenty-three carry `dismissed_comment: null` - a suppression on a security
surface with no recorded reasoning at all.

Two finding classes, reported separately:

1. MISSING JUSTIFICATION (failure). A dismissed alert whose `dismissed_comment`
   is absent, null, or whitespace-only. The remediation is unambiguous: write
   the justification, or reopen the alert.

2. STANDING SUPPRESSION (warning by default; `--fail-on-repo-level` promotes
   it). A dismissed REPO-LEVEL alert - one whose instance location carries the
   sentinel path `no file associated with this alert` instead of a real file.
   OSSF Scorecard's repo-level checks (BranchProtectionID, CIIBestPracticesID,
   CodeReviewID, FuzzingID, ...) emit exactly ONE alert per check, forever, and
   rewrite the message text in place as the underlying condition changes.
   Dismissing one is therefore not a judgement about a finding; it is a
   permanent suppression of that whole check, which keeps applying to
   conditions nobody has read. This repo has already been bitten by exactly
   that shape once, when a dismissal written for a torch CVE silently went on
   covering two unrelated anyio advisories for three months. A comment does not
   fix it - a dated re-review does - which is why it is a separate class rather
   than folded into class 1.

Why class 2 only WARNs by default: the remediation for class 1 is mechanical
and has one correct answer, so blocking on it is safe. The remediation for
class 2 is a judgement call (re-review, re-scope, or reopen) with no single
right answer, and a gate that is red on day one for a condition the owner may
legitimately accept is a gate that gets muted. It is always printed and always
present in `--json`; promoting it to a blocking failure is an owner decision,
exposed as a flag rather than baked in.

SCOPE - measured, not assumed. The census is taken with `?state=dismissed`,
never `?state=all`. Measured against this repo: `?state=all` returned 78 alerts
while alert numbering runs to 87, with nine numbers absent. Each of those nine
was fetched individually and every one is an alert whose
`most_recent_instance.ref` is a pull-request merge ref of the form
`refs/pull/N/merge`; all nine report a top-level `state` of null and are
therefore excluded from any `state=<value>` listing. They are transient
per-PR findings, not standing risk decisions, so they must not enter the
census. Findings are additionally filtered to the default branch's ref
(`refs/heads/<branch>`) so a dismissal recorded against a PR merge ref can
never distort the count either.

AUTHENTICATION. Prefers the `gh` CLI when it is on PATH and authenticated,
falling back to a direct API call carrying `GITHUB_TOKEN` or `GH_TOKEN` from
the environment. No token, path, or address is embedded here. If neither route
works the gate exits 2, NOT 0: an empty alert list produced by a failed fetch
must never read as a clean census. GitHub answers an unauthenticated or
malformed request with a JSON object carrying a `message` key rather than a
list, so the page parser rejects any payload that is not a list outright.

Exit codes:
    0  clean - census taken, no findings
    1  findings - at least one blocking finding
    2  could not run - the census was never taken; verdict UNKNOWN, not clean

Usage:
    python tools/check_dismissal_justifications.py
    python tools/check_dismissal_justifications.py --json
    python tools/check_dismissal_justifications.py --fail-on-repo-level
    python tools/check_dismissal_justifications.py --repo OWNER/NAME --branch main
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess  # nosec B404 - fixed argv to the gh CLI, never a shell
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field

API_ROOT = "https://api.github.com"
DEFAULT_REPO = "Gavin-Borges/SESTRAV"
DEFAULT_BRANCH = "main"
PER_PAGE = 100
MAX_PAGES = 50
HTTP_TIMEOUT_SECONDS = 30
GH_TIMEOUT_SECONDS = 120

# The exact sentinel GitHub stores in place of a file path for an alert that
# describes the repository rather than a line of code. Scorecard's repo-level
# checks are the only producers of it here. Matched literally: inferring
# "repo-level" from a heuristic (missing line number, unusual rule id prefix)
# would silently reclassify real file findings.
REPO_LEVEL_PATH = "no file associated with this alert"

EXIT_CLEAN = 0
EXIT_FINDINGS = 1
EXIT_COULD_NOT_RUN = 2


class CouldNotRun(RuntimeError):
    """The census could not be taken.

    Raised for every failure mode that would otherwise surface as an empty
    alert list: no auth, a rejected request, a non-list payload, a `gh`
    invocation that failed. Callers must translate this into exit code 2 and
    must never treat it as a clean result.
    """


@dataclass(frozen=True)
class Finding:
    """One offending dismissed alert."""

    number: int
    rule_id: str
    severity: str
    location: str
    reason: str
    repo_level: bool
    has_comment: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "number": self.number,
            "rule_id": self.rule_id,
            "severity": self.severity,
            "location": self.location,
            "dismissed_reason": self.reason,
            "repo_level": self.repo_level,
            "has_comment": self.has_comment,
        }

    def describe(self) -> str:
        scope = "repo-level" if self.repo_level else self.location
        return (
            f"  alert #{self.number}  [{self.severity}]  {self.rule_id}\n"
            f"      location: {scope}\n"
            f"      reason:   {self.reason}"
        )


@dataclass
class Report:
    """The judgement rendered over one census."""

    repo: str
    branch: str
    dismissed_in_scope: int = 0
    out_of_scope: int = 0
    not_dismissed: int = 0
    missing_justification: list[Finding] = field(default_factory=list)
    standing_suppressions: list[Finding] = field(default_factory=list)

    def exit_code(self, fail_on_repo_level: bool = False) -> int:
        if self.missing_justification:
            return EXIT_FINDINGS
        if fail_on_repo_level and self.standing_suppressions:
            return EXIT_FINDINGS
        return EXIT_CLEAN

    def as_dict(self, fail_on_repo_level: bool = False) -> dict[str, object]:
        return {
            "status": "findings" if self.exit_code(fail_on_repo_level) else "clean",
            "repo": self.repo,
            "branch": self.branch,
            "dismissed_in_scope": self.dismissed_in_scope,
            "excluded_other_ref": self.out_of_scope,
            "excluded_not_dismissed": self.not_dismissed,
            "missing_justification_count": len(self.missing_justification),
            "standing_suppression_count": len(self.standing_suppressions),
            "missing_justification": [f.as_dict() for f in self.missing_justification],
            "standing_suppressions": [f.as_dict() for f in self.standing_suppressions],
            "fail_on_repo_level": fail_on_repo_level,
        }


# --------------------------------------------------------------------------
# Judgement - pure, no network. Everything below this line is fed a list of
# already-fetched alert dicts, which is what makes the gate testable offline.
# --------------------------------------------------------------------------


def _instance(alert: dict) -> dict:
    inst = alert.get("most_recent_instance")
    return inst if isinstance(inst, dict) else {}


def alert_ref(alert: dict) -> str:
    """The git ref the most recent instance of this alert was seen on."""
    ref = _instance(alert).get("ref")
    return ref if isinstance(ref, str) else ""


def alert_location(alert: dict) -> str:
    """A human-readable location.

    Deliberately NOT rendered as `path` + colon + line: this repo runs a
    required merge check that pins every such citation to a real line, and a
    generated string would be pinned to whatever it happened to say.
    """
    loc = _instance(alert).get("location")
    if not isinstance(loc, dict):
        return "unknown"
    path = loc.get("path")
    if not isinstance(path, str) or not path.strip():
        return "unknown"
    start = loc.get("start_line")
    if isinstance(start, int):
        return f"{path} (line {start})"
    return path


def is_repo_level(alert: dict) -> bool:
    """True when the alert describes the repository, not a line of code."""
    loc = _instance(alert).get("location")
    if not isinstance(loc, dict):
        return False
    path = loc.get("path")
    if not isinstance(path, str):
        return False
    return path.strip() == REPO_LEVEL_PATH


def is_dismissed(alert: dict) -> bool:
    return alert.get("state") == "dismissed"


def justification_is_missing(alert: dict) -> bool:
    """True when the dismissal carries no usable written reasoning.

    Absent key, null, empty and whitespace-only are all the same defect: a
    reader learns nothing about why the risk was accepted.
    """
    comment = alert.get("dismissed_comment")
    if comment is None:
        return True
    if not isinstance(comment, str):
        return True
    return not comment.strip()


def _severity(alert: dict) -> str:
    rule = alert.get("rule")
    if not isinstance(rule, dict):
        return "unknown"
    for key in ("security_severity_level", "severity"):
        value = rule.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "unknown"


def _rule_id(alert: dict) -> str:
    rule = alert.get("rule")
    if isinstance(rule, dict):
        value = rule.get("id")
        if isinstance(value, str) and value.strip():
            return value
    return "unknown"


def _to_finding(alert: dict) -> Finding:
    reason = alert.get("dismissed_reason")
    return Finding(
        number=int(alert.get("number", -1)),
        rule_id=_rule_id(alert),
        severity=_severity(alert),
        location=alert_location(alert),
        reason=reason if isinstance(reason, str) and reason.strip() else "unstated",
        repo_level=is_repo_level(alert),
        has_comment=not justification_is_missing(alert),
    )


def evaluate(
    alerts: list[dict],
    repo: str = DEFAULT_REPO,
    branch: str = DEFAULT_BRANCH,
) -> Report:
    """Classify an already-fetched list of alerts. No network, no I/O."""
    report = Report(repo=repo, branch=branch)
    wanted_ref = f"refs/heads/{branch}"
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        if not is_dismissed(alert):
            # An open or fixed alert is not a risk-acceptance decision, so it
            # has nothing to justify. Counted, never judged.
            report.not_dismissed += 1
            continue
        if alert_ref(alert) != wanted_ref:
            # A dismissal recorded against a PR merge ref is not a standing
            # decision about the default branch.
            report.out_of_scope += 1
            continue
        report.dismissed_in_scope += 1
        finding = _to_finding(alert)
        if justification_is_missing(alert):
            report.missing_justification.append(finding)
        if finding.repo_level:
            report.standing_suppressions.append(finding)
    report.missing_justification.sort(key=lambda f: f.number)
    report.standing_suppressions.sort(key=lambda f: f.number)
    return report


# --------------------------------------------------------------------------
# Fetch - the only part that touches the network.
# --------------------------------------------------------------------------


def parse_page_payload(raw: str) -> list[dict]:
    """Turn one API page body into a list of alerts, or raise CouldNotRun.

    This is the guard that stops an auth failure reading as an empty census.
    GitHub answers a rejected request with a JSON OBJECT carrying a `message`
    key, never a list, so anything that is not a list is a failed fetch.
    """
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise CouldNotRun(f"API response was not JSON: {exc}") from exc
    if isinstance(payload, dict):
        message = payload.get("message")
        detail = message if isinstance(message, str) else "no message field"
        raise CouldNotRun(f"API returned an error object instead of a list: {detail}")
    if not isinstance(payload, list):
        raise CouldNotRun(f"API returned {type(payload).__name__}, expected a list")
    return [item for item in payload if isinstance(item, dict)]


def _alerts_path(repo: str, state: str, page: int) -> str:
    return (
        f"repos/{repo}/code-scanning/alerts"
        f"?state={state}&per_page={PER_PAGE}&page={page}"
    )


def _fetch_page_gh(repo: str, state: str, page: int) -> list[dict]:
    gh = shutil.which("gh")
    if gh is None:
        raise CouldNotRun("gh CLI not on PATH")
    command = [
        gh,
        "api",
        "-H",
        "Accept: application/vnd.github+json",
        _alerts_path(repo, state, page),
    ]
    try:
        # Captured as BYTES and decoded explicitly. `text=True` would decode
        # with the platform's preferred encoding, which on Windows is cp1252
        # and raises UnicodeDecodeError on the UTF-8 alert bodies GitHub
        # returns - a decode failure inside the fetch, not a real API problem.
        proc = subprocess.run(  # nosec B603 - fixed argv, shell=False, no user input
            command,
            capture_output=True,
            timeout=GH_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CouldNotRun(f"gh invocation failed: {exc}") from exc
    stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
    if proc.returncode != 0:
        detail = (stderr or stdout).strip().splitlines()
        raise CouldNotRun(
            f"gh api exited {proc.returncode}: {detail[0] if detail else 'no output'}"
        )
    return parse_page_payload(stdout)


def _fetch_page_http(repo: str, state: str, page: int, token: str) -> list[dict]:
    url = f"{API_ROOT}/{_alerts_path(repo, state, page)}"
    # Constrain the scheme and host BEFORE opening, rather than only
    # asserting in a comment that they are safe. API_ROOT is a module
    # constant, so this can only fail if someone edits it to a non-https
    # or off-host value - and then it fails loudly here instead of letting
    # urlopen honour file:// or a custom scheme. This guard is what makes
    # the suppression below honest.
    if not url.startswith(f"{API_ROOT}/"):
        raise CouldNotRun(f"refusing to fetch a URL outside {API_ROOT}: {url}")
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "sestrav-dismissal-justification-gate",
        },
    )
    try:
        # The B310 suppression on the next line rests on the scheme and host
        # guard above, not on an assertion that urlopen is safe in general.
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:  # nosec B310
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise CouldNotRun(f"API request rejected with HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise CouldNotRun(f"API request failed: {exc}") from exc
    return parse_page_payload(raw)


def _environment_token() -> str | None:
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def fetch_dismissed_alerts(repo: str = DEFAULT_REPO) -> list[dict]:
    """Page through every dismissed alert, or raise CouldNotRun.

    `?state=dismissed` is used explicitly. `?state=all` is not reliable here -
    see this module's docstring for the measurement.
    """
    errors: list[str] = []
    routes: list[tuple[str, object]] = []
    if shutil.which("gh") is not None:
        routes.append(("gh CLI", None))
    token = _environment_token()
    if token is not None:
        routes.append(("GITHUB_TOKEN/GH_TOKEN", token))
    if not routes:
        raise CouldNotRun(
            "no authentication route available: gh CLI is not on PATH and "
            "neither GITHUB_TOKEN nor GH_TOKEN is set"
        )

    for label, route_token in routes:
        collected: list[dict] = []
        try:
            for page in range(1, MAX_PAGES + 1):
                if route_token is None:
                    batch = _fetch_page_gh(repo, "dismissed", page)
                else:
                    batch = _fetch_page_http(repo, "dismissed", page, str(route_token))
                collected.extend(batch)
                if len(batch) < PER_PAGE:
                    break
            else:
                raise CouldNotRun(f"pagination did not terminate within {MAX_PAGES} pages")
        except CouldNotRun as exc:
            errors.append(f"{label}: {exc}")
            continue
        return collected
    raise CouldNotRun("; ".join(errors))


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def render(report: Report, fail_on_repo_level: bool = False) -> str:
    lines: list[str] = []
    lines.append(f"Dismissal justification gate: {report.repo} @ refs/heads/{report.branch}")
    lines.append(
        f"  dismissed alerts in scope: {report.dismissed_in_scope}"
        f"  (excluded: {report.out_of_scope} on another ref,"
        f" {report.not_dismissed} not dismissed)"
    )
    lines.append("")

    if report.missing_justification:
        lines.append(
            f"FAIL - {len(report.missing_justification)} dismissed alert(s) carry no "
            "written justification:"
        )
        for finding in report.missing_justification:
            lines.append(finding.describe())
        lines.append("")
        lines.append(
            "  Remediation: add a dismissal comment naming the condition that makes "
            "the finding acceptable, or reopen the alert."
        )
    else:
        lines.append("OK - every dismissed alert in scope carries a written justification.")
    lines.append("")

    label = "FAIL" if fail_on_repo_level else "WARN"
    if report.standing_suppressions:
        lines.append(
            f"{label} - {len(report.standing_suppressions)} dismissed alert(s) are "
            "REPO-LEVEL, i.e. a standing suppression of an entire check:"
        )
        for finding in report.standing_suppressions:
            lines.append(finding.describe())
        lines.append("")
        lines.append(
            "  A repo-level check emits one alert for its whole lifetime and rewrites "
            "its message in place, so a dismissal keeps applying to conditions nobody "
            "has read. These need a dated re-review, not only a comment."
        )
    else:
        lines.append("OK - no repo-level check is under a standing dismissal.")
    return "\n".join(lines)


def run(
    repo: str = DEFAULT_REPO,
    branch: str = DEFAULT_BRANCH,
    as_json: bool = False,
    fail_on_repo_level: bool = False,
    fetcher=fetch_dismissed_alerts,
    stream=None,
) -> int:
    """Take the census, judge it, print it, and return the exit code.

    `fetcher` is injectable so the judgement can be exercised without network.
    """
    out = stream if stream is not None else sys.stdout
    try:
        alerts = fetcher(repo)
    except CouldNotRun as exc:
        message = str(exc)
        if as_json:
            print(
                json.dumps(
                    {"status": "could-not-run", "repo": repo, "branch": branch, "error": message},
                    indent=2,
                ),
                file=out,
            )
        else:
            print("COULD NOT RUN - the dismissal census was never taken.", file=out)
            print(f"  reason: {message}", file=out)
            print(
                "  This is NOT a pass. Provide gh CLI auth or set GITHUB_TOKEN with "
                "the security-events: read permission, then re-run.",
                file=out,
            )
        return EXIT_COULD_NOT_RUN

    report = evaluate(alerts, repo=repo, branch=branch)
    if as_json:
        print(json.dumps(report.as_dict(fail_on_repo_level), indent=2), file=out)
    else:
        print(render(report, fail_on_repo_level), file=out)
    return report.exit_code(fail_on_repo_level)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if a dismissed code-scanning alert carries no written justification, "
            "and flag dismissals that silence an entire repo-level check."
        )
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPO,
        help="OWNER/NAME to audit (default: GITHUB_REPOSITORY, else this repository).",
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help="Default branch whose refs/heads ref scopes the census.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report.")
    parser.add_argument(
        "--fail-on-repo-level",
        action="store_true",
        help="Promote standing repo-level suppressions from WARN to a blocking failure.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(
        repo=args.repo,
        branch=args.branch,
        as_json=args.json,
        fail_on_repo_level=args.fail_on_repo_level,
    )


if __name__ == "__main__":
    sys.exit(main())
