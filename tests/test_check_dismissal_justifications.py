"""Tests for the dismissed-alert justification gate.

NO NETWORK. Every test here feeds `evaluate` / `run` a hand-built fixture list
shaped like the GitHub code-scanning alert payload, which is why the module
keeps fetching and judging in separate functions.

The load-bearing test is `test_auth_failure_is_not_a_clean_census`: a gate that
silently passes when it could not run is the exact failure this repository
gates against, so an unreachable API must surface as exit code 2 and never as
exit code 0.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "check_dismissal_justifications.py"


def _load_module():
    """Import the checker by path because ``tools/`` is not a package.

    Matches the convention the other tools tests use. A ``sys.path`` insert
    would work too, but it is a global side effect that outlives this module
    when the whole suite runs.

    The module is registered in ``sys.modules`` BEFORE execution: the checker
    uses dataclasses under ``from __future__ import annotations``, and the
    dataclass machinery resolves those string annotations by looking its own
    module up by name. Without the registration, collection dies with
    ``AttributeError: 'NoneType' object has no attribute '__dict__'``.
    """
    spec = importlib.util.spec_from_file_location("check_dismissal_justifications", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_module()

MAIN_REF = "refs/heads/main"


def make_alert(
    number: int,
    comment: str | None = "justified because ...",
    path: str = "src/example.py",
    state: str = "dismissed",
    ref: str = MAIN_REF,
    rule_id: str = "py/example-rule",
    severity: str = "medium",
    reason: str = "false positive",
    start_line: int | None = 12,
) -> dict:
    """Build one alert dict in the shape the API returns."""
    location: dict = {"path": path}
    if start_line is not None:
        location["start_line"] = start_line
    return {
        "number": number,
        "state": state,
        "dismissed_reason": reason,
        "dismissed_comment": comment,
        "rule": {"id": rule_id, "security_severity_level": severity, "severity": "error"},
        "most_recent_instance": {"ref": ref, "location": location},
    }


def make_repo_level_alert(number: int, comment: str | None = None, **kwargs) -> dict:
    """A Scorecard repo-level alert: the sentinel path, no line number."""
    kwargs.setdefault("rule_id", "BranchProtectionID")
    kwargs.setdefault("severity", "high")
    return make_alert(
        number,
        comment=comment,
        path=gate.REPO_LEVEL_PATH,
        start_line=None,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Class 1: missing justification
# ---------------------------------------------------------------------------


def test_all_commented_dismissals_pass():
    alerts = [make_alert(1), make_alert(2), make_alert(3)]
    report = gate.evaluate(alerts)
    assert report.dismissed_in_scope == 3
    assert report.missing_justification == []
    assert report.standing_suppressions == []
    assert report.exit_code() == gate.EXIT_CLEAN


def test_null_comment_fails():
    report = gate.evaluate([make_alert(1), make_alert(2, comment=None)])
    assert [f.number for f in report.missing_justification] == [2]
    assert report.exit_code() == gate.EXIT_FINDINGS


def test_absent_comment_key_fails():
    alert = make_alert(7)
    del alert["dismissed_comment"]
    report = gate.evaluate([alert])
    assert [f.number for f in report.missing_justification] == [7]
    assert report.exit_code() == gate.EXIT_FINDINGS


@pytest.mark.parametrize("blank", ["", " ", "\t", "\n", "  \n\t  ", "\r\n"])
def test_whitespace_only_comment_fails(blank):
    report = gate.evaluate([make_alert(5, comment=blank)])
    assert [f.number for f in report.missing_justification] == [5]
    assert report.exit_code() == gate.EXIT_FINDINGS


def test_non_string_comment_fails():
    report = gate.evaluate([make_alert(9, comment=None), make_alert(10, comment=12345)])  # type: ignore[arg-type]
    assert [f.number for f in report.missing_justification] == [9, 10]


def test_finding_carries_number_rule_severity_and_location():
    report = gate.evaluate(
        [make_alert(42, comment=None, rule_id="py/tainted-path", severity="critical")]
    )
    finding = report.missing_justification[0]
    assert finding.number == 42
    assert finding.rule_id == "py/tainted-path"
    assert finding.severity == "critical"
    assert "src/example.py" in finding.location
    assert finding.reason == "false positive"
    assert finding.has_comment is False


def test_severity_falls_back_when_security_severity_absent():
    alert = make_alert(3, comment=None)
    alert["rule"] = {"id": "SomeID", "severity": "error"}
    finding = gate.evaluate([alert]).missing_justification[0]
    assert finding.severity == "error"


def test_location_is_not_rendered_as_a_line_citation():
    # The repo runs a required merge check pinning every path + colon + line
    # citation, so the gate must never manufacture one.
    finding = gate.evaluate([make_alert(1, comment=None)]).missing_justification[0]
    assert ":12" not in finding.location
    assert finding.location == "src/example.py (line 12)"


# ---------------------------------------------------------------------------
# Class 2: repo-level standing suppression
# ---------------------------------------------------------------------------


def test_repo_level_dismissal_is_a_standing_suppression():
    report = gate.evaluate([make_repo_level_alert(11, comment="accepted for now")])
    assert [f.number for f in report.standing_suppressions] == [11]
    assert report.standing_suppressions[0].repo_level is True
    # It has a comment, so it is NOT a class-1 finding.
    assert report.missing_justification == []
    # WARN by default: advisory only.
    assert report.exit_code() == gate.EXIT_CLEAN
    # ...and blocking when the owner opts in.
    assert report.exit_code(fail_on_repo_level=True) == gate.EXIT_FINDINGS


def test_repo_level_with_null_comment_appears_in_both_classes():
    report = gate.evaluate([make_repo_level_alert(13, comment=None)])
    assert [f.number for f in report.missing_justification] == [13]
    assert [f.number for f in report.standing_suppressions] == [13]
    assert report.exit_code() == gate.EXIT_FINDINGS


def test_file_level_alert_is_not_repo_level():
    assert gate.is_repo_level(make_alert(1)) is False
    assert gate.is_repo_level(make_repo_level_alert(1)) is True


def test_repo_level_sentinel_is_matched_exactly_not_by_heuristic():
    # A real file finding with no line number must not be mistaken for one.
    alert = make_alert(4, path="Dockerfile", start_line=None)
    assert gate.is_repo_level(alert) is False
    # A path that merely contains the sentinel text is not the sentinel.
    near = make_alert(5, path="docs/no file associated with this alert.md")
    assert gate.is_repo_level(near) is False


def test_missing_location_is_not_repo_level():
    alert = make_alert(6, comment=None)
    alert["most_recent_instance"] = {"ref": MAIN_REF}
    report = gate.evaluate([alert])
    assert report.standing_suppressions == []
    assert report.missing_justification[0].location == "unknown"


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def test_open_alert_is_ignored():
    report = gate.evaluate([make_alert(1, comment=None, state="open")])
    assert report.dismissed_in_scope == 0
    assert report.not_dismissed == 1
    assert report.missing_justification == []
    assert report.exit_code() == gate.EXIT_CLEAN


def test_fixed_and_null_state_alerts_are_ignored():
    alerts = [
        make_alert(1, comment=None, state="fixed"),
        make_alert(2, comment=None, state=None),  # type: ignore[arg-type]
    ]
    report = gate.evaluate(alerts)
    assert report.not_dismissed == 2
    assert report.missing_justification == []


def test_pull_request_merge_ref_dismissal_is_out_of_scope():
    # Measured against the live repo: alerts whose most_recent_instance.ref is
    # refs/pull/N/merge report a top-level state of null and are absent from
    # state listings. Filtering on the default-branch ref keeps such an alert
    # out of the census even if one were ever dismissed.
    alerts = [
        make_alert(1, comment=None),
        make_alert(19, comment=None, ref="refs/pull/43/merge"),
    ]
    report = gate.evaluate(alerts)
    assert report.dismissed_in_scope == 1
    assert report.out_of_scope == 1
    assert [f.number for f in report.missing_justification] == [1]


def test_branch_scope_is_configurable():
    alerts = [make_alert(1, comment=None, ref="refs/heads/develop")]
    assert gate.evaluate(alerts, branch="main").out_of_scope == 1
    assert gate.evaluate(alerts, branch="develop").dismissed_in_scope == 1


def test_non_dict_entries_are_skipped():
    report = gate.evaluate([make_alert(1, comment=None), "garbage", None, 7])  # type: ignore[list-item]
    assert [f.number for f in report.missing_justification] == [1]


def test_findings_are_sorted_by_alert_number():
    alerts = [make_alert(n, comment=None) for n in (76, 1, 18, 11)]
    report = gate.evaluate(alerts)
    assert [f.number for f in report.missing_justification] == [1, 11, 18, 76]


# ---------------------------------------------------------------------------
# Could-not-run: the load-bearing half
# ---------------------------------------------------------------------------


def test_auth_failure_is_not_a_clean_census():
    """An empty result caused by a failed fetch must NOT pass silently."""

    def failing_fetcher(repo):
        raise gate.CouldNotRun("no authentication route available")

    buffer = io.StringIO()
    code = gate.run(fetcher=failing_fetcher, stream=buffer)
    assert code == gate.EXIT_COULD_NOT_RUN
    assert code != gate.EXIT_CLEAN
    text = buffer.getvalue()
    assert "COULD NOT RUN" in text
    assert "NOT a pass" in text


def test_could_not_run_is_visibly_different_from_clean_in_json():
    def failing_fetcher(repo):
        raise gate.CouldNotRun("API request rejected with HTTP 401")

    buffer = io.StringIO()
    code = gate.run(fetcher=failing_fetcher, as_json=True, stream=buffer)
    payload = json.loads(buffer.getvalue())
    assert code == gate.EXIT_COULD_NOT_RUN
    assert payload["status"] == "could-not-run"
    assert "401" in payload["error"]


def test_error_object_payload_raises_instead_of_yielding_an_empty_list():
    # GitHub answers a rejected request with an object carrying `message`.
    # Parsing it as "no alerts" is precisely the silent pass being prevented.
    with pytest.raises(gate.CouldNotRun) as excinfo:
        gate.parse_page_payload('{"message": "Bad credentials", "status": "401"}')
    assert "Bad credentials" in str(excinfo.value)


def test_non_json_payload_raises():
    with pytest.raises(gate.CouldNotRun):
        gate.parse_page_payload("<html>404 not found</html>")


def test_empty_object_payload_still_raises():
    with pytest.raises(gate.CouldNotRun):
        gate.parse_page_payload("{}")


def test_well_formed_empty_list_parses_to_zero_alerts():
    # A genuinely empty page is legitimate - that is why the object guard
    # above, not the emptiness, is what distinguishes failure from cleanliness.
    assert gate.parse_page_payload("[]") == []


def test_fetch_raises_when_no_auth_route_exists(monkeypatch):
    monkeypatch.setattr(gate.shutil, "which", lambda _name: None)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with pytest.raises(gate.CouldNotRun) as excinfo:
        gate.fetch_dismissed_alerts("owner/name")
    assert "no authentication route" in str(excinfo.value)


# ---------------------------------------------------------------------------
# End-to-end through run()
# ---------------------------------------------------------------------------


def test_run_returns_one_on_findings_and_names_the_alert():
    alerts = [make_alert(1), make_repo_level_alert(11, comment=None)]
    buffer = io.StringIO()
    code = gate.run(fetcher=lambda repo: alerts, stream=buffer)
    text = buffer.getvalue()
    assert code == gate.EXIT_FINDINGS
    assert "alert #11" in text
    assert "BranchProtectionID" in text
    assert "repo-level" in text


def test_run_returns_zero_on_a_clean_census():
    buffer = io.StringIO()
    code = gate.run(fetcher=lambda repo: [make_alert(1), make_alert(2)], stream=buffer)
    assert code == gate.EXIT_CLEAN
    assert "OK - every dismissed alert" in buffer.getvalue()


def test_json_report_shape():
    alerts = [
        make_alert(1),
        make_alert(16, comment=None, path=".github/workflows/ci.yml"),
        make_repo_level_alert(12, comment=None, rule_id="CodeReviewID"),
    ]
    buffer = io.StringIO()
    code = gate.run(fetcher=lambda repo: alerts, as_json=True, stream=buffer)
    payload = json.loads(buffer.getvalue())
    assert code == gate.EXIT_FINDINGS
    assert payload["status"] == "findings"
    assert payload["dismissed_in_scope"] == 3
    assert payload["missing_justification_count"] == 2
    assert payload["standing_suppression_count"] == 1
    assert sorted(f["number"] for f in payload["missing_justification"]) == [12, 16]
    assert payload["standing_suppressions"][0]["rule_id"] == "CodeReviewID"


def test_fail_on_repo_level_flag_changes_only_the_verdict_not_the_report():
    alerts = [make_repo_level_alert(13, comment="reviewed 2026-01-01")]
    report = gate.evaluate(alerts)
    assert len(report.standing_suppressions) == 1
    assert report.exit_code(fail_on_repo_level=False) == gate.EXIT_CLEAN
    assert report.exit_code(fail_on_repo_level=True) == gate.EXIT_FINDINGS
    lenient = json.loads(json.dumps(report.as_dict(False)))
    strict = json.loads(json.dumps(report.as_dict(True)))
    assert lenient["standing_suppressions"] == strict["standing_suppressions"]
    assert lenient["status"] == "clean"
    assert strict["status"] == "findings"


def test_parser_defaults():
    args = gate.build_parser().parse_args([])
    assert args.branch == "main"
    assert args.json is False
    assert args.fail_on_repo_level is False
    assert "/" in args.repo


# ---------------------------------------------------------------------------
# Regression fixture mirroring the live census measured 2026-09-19
# ---------------------------------------------------------------------------


def test_live_shaped_fixture_reproduces_nine_findings_four_repo_level():
    """The shape the gate must report against this repo today."""
    repo_level = [
        ("BranchProtectionID", "won't fix"),
        ("CIIBestPracticesID", "false positive"),
        ("CodeReviewID", "false positive"),
        ("FuzzingID", "false positive"),
    ]
    alerts: list[dict] = []
    for number, (rule_id, reason) in zip((1, 11, 12, 13), repo_level):
        alerts.append(make_repo_level_alert(number, comment=None, rule_id=rule_id, reason=reason))
    for number, path in (
        (16, ".github/workflows/ci.yml"),
        (17, ".github/workflows/fuzzing.yml"),
        (18, ".github/workflows/security.yml"),
        (75, ".github/workflows/release.yml"),
        (76, ".github/workflows/release.yml"),
    ):
        alerts.append(
            make_alert(number, comment=None, path=path, rule_id="PinnedDependenciesID")
        )
    for number in (33, 34, 35, 36, 37, 38, 40, 41, 42, 50, 52, 82, 83, 84):
        alerts.append(make_alert(number, comment="justified in the dismissal note"))

    report = gate.evaluate(alerts)
    assert report.dismissed_in_scope == 23
    assert len(report.missing_justification) == 9
    assert [f.number for f in report.missing_justification] == [1, 11, 12, 13, 16, 17, 18, 75, 76]
    assert len(report.standing_suppressions) == 4
    assert [f.number for f in report.standing_suppressions] == [1, 11, 12, 13]
    assert report.exit_code() == gate.EXIT_FINDINGS
