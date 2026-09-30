"""Regression tests for tools/check_lockfile_advisories.py.

The gate this script backs closes a structural blind spot: a vulnerable pin that
exists only in a compiled lockfile is invisible to Dependabot (it parses `.in`
sources, never `.lock`/`.txt` artifacts) and was previously invisible to CI too, since
every pip-audit invocation in .github/workflows/security.yml carried permanent
`--ignore-vuln` flags applied before any report was written. These tests exercise the
parsing/evaluation logic directly against fixture reports and acceptance files, since
pip-audit itself cannot run against environments/requirements.lock on Windows (it is a
Linux-compiled lockfile; nvidia-cufile has no Windows wheel).
"""

from __future__ import annotations

import json
import pathlib
import textwrap

import pytest

from tools.check_lockfile_advisories import (
    Acceptance,
    Finding,
    ReportError,
    evaluate,
    filter_in_scope,
    load_acceptances,
    main,
    unregistered_acceptances,
    parse_lockfile_packages,
    parse_report,
)


def _report(dependencies: list[dict]) -> str:
    return json.dumps({"dependencies": dependencies})


def _write_toml(path, content: str):
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


# --- parse_report -----------------------------------------------------------------


def test_parse_report_extracts_findings_per_vuln():
    text = _report(
        [
            {
                "name": "torch",
                "version": "2.12.0",
                "vulns": [
                    {"id": "PYSEC-2025-194", "fix_versions": ["2.13.0"]},
                ],
            },
            {"name": "numpy", "version": "2.4.6", "vulns": []},
        ]
    )
    findings = parse_report(text)
    assert findings == [
        Finding(package="torch", version="2.12.0", advisory_id="PYSEC-2025-194", fix_versions=("2.13.0",))
    ]


def test_parse_report_normalizes_package_name():
    text = _report(
        [{"name": "Torch_Geometric", "version": "2.7.0", "vulns": [{"id": "X-1", "fix_versions": []}]}]
    )
    findings = parse_report(text)
    assert findings[0].package == "torch-geometric"


def test_parse_report_handles_multiple_vulns_on_one_package():
    text = _report(
        [
            {
                "name": "aiohttp",
                "version": "3.13.5",
                "vulns": [
                    {"id": "GHSA-1", "fix_versions": ["3.14.1"]},
                    {"id": "GHSA-2", "fix_versions": ["3.14.1"]},
                ],
            }
        ]
    )
    findings = parse_report(text)
    assert len(findings) == 2
    assert {f.advisory_id for f in findings} == {"GHSA-1", "GHSA-2"}


def test_parse_report_rejects_invalid_json():
    with pytest.raises(ReportError, match="not valid JSON"):
        parse_report("not json at all")


def test_parse_report_rejects_missing_dependencies_key():
    with pytest.raises(ReportError, match="no 'dependencies' list"):
        parse_report(json.dumps({"something_else": []}))


def test_parse_report_rejects_empty_dependencies_as_tooling_failure():
    # An empty list must never be treated as "clean scan" - it means the lockfile
    # never installed (e.g. a Linux-only wheel failing on a different runner), which
    # is exactly the failure mode the sibling tooling-assert step in CI guards against.
    with pytest.raises(ReportError, match="zero packages"):
        parse_report(_report([]))


def test_parse_report_missing_vuln_id_defaults_to_empty_string():
    text = _report([{"name": "foo", "version": "1.0", "vulns": [{"fix_versions": ["1.1"]}]}])
    findings = parse_report(text)
    assert findings[0].advisory_id == ""


# --- load_acceptances --------------------------------------------------------------


def test_load_acceptances_missing_file_returns_empty(tmp_path):
    assert load_acceptances(tmp_path / "nonexistent.toml") == {}


def test_load_acceptances_parses_entries_keyed_by_id_and_package(tmp_path):
    path = _write_toml(
        tmp_path / "accepted.toml",
        """\
        [[accepted]]
        id = "PYSEC-2025-194"
        package = "torch"
        reason = "temporary, see PR #205"
        register_entry = "SECURITY.md#cve-2025-3000"
        """,
    )
    acceptances = load_acceptances(path)
    assert acceptances == {
        ("PYSEC-2025-194", "torch"): Acceptance(
            advisory_id="PYSEC-2025-194",
            package="torch",
            reason="temporary, see PR #205",
            register_entry="SECURITY.md#cve-2025-3000",
        )
    }


def test_load_acceptances_normalizes_package_name(tmp_path):
    path = _write_toml(
        tmp_path / "accepted.toml",
        """\
        [[accepted]]
        id = "X-1"
        package = "Torch_Geometric"
        reason = "r"
        register_entry = "SECURITY.md"
        """,
    )
    acceptances = load_acceptances(path)
    assert ("X-1", "torch-geometric") in acceptances


def test_load_acceptances_missing_required_field_raises(tmp_path):
    path = _write_toml(
        tmp_path / "accepted.toml",
        """\
        [[accepted]]
        id = "X-1"
        package = "torch"
        """,
    )
    with pytest.raises(KeyError):
        load_acceptances(path)


# --- evaluate ------------------------------------------------------------------


def test_evaluate_accepted_finding_is_not_unaccepted():
    findings = [Finding("torch", "2.12.0", "PYSEC-2025-194", ("2.13.0",))]
    acceptances = {
        ("PYSEC-2025-194", "torch"): Acceptance("PYSEC-2025-194", "torch", "r", "SECURITY.md")
    }
    unaccepted, stale = evaluate(findings, acceptances)
    assert unaccepted == []
    assert stale == []


def test_evaluate_unaccepted_finding_fails():
    findings = [Finding("torch", "2.12.0", "PYSEC-2025-194", ("2.13.0",))]
    unaccepted, stale = evaluate(findings, {})
    assert unaccepted == findings
    assert stale == []


def test_evaluate_acceptance_scoped_to_wrong_package_does_not_cover_finding():
    # The same advisory id surfacing through a different package has not been
    # assessed and must still fail - an accepted (id, package) pair is not a
    # blanket acceptance of the id alone.
    findings = [Finding("other-pkg", "1.0", "PYSEC-2025-194", ())]
    acceptances = {
        ("PYSEC-2025-194", "torch"): Acceptance("PYSEC-2025-194", "torch", "r", "SECURITY.md")
    }
    unaccepted, stale = evaluate(findings, acceptances)
    assert unaccepted == findings


def test_evaluate_stale_acceptance_detected_when_finding_disappears():
    # No findings at all (e.g. torch was upgraded past the vulnerable range) but the
    # acceptance entry was left in place - this must be reported as stale, not fail
    # the build. Catching this automatically is the whole point of the design: a
    # fixed advisory should not require anyone to remember to clean up its exemption.
    acceptances = {
        ("PYSEC-2025-194", "torch"): Acceptance("PYSEC-2025-194", "torch", "r", "SECURITY.md")
    }
    unaccepted, stale = evaluate([], acceptances)
    assert unaccepted == []
    assert stale == [("PYSEC-2025-194", "torch")]


def test_evaluate_mixed_accepted_and_unaccepted():
    findings = [
        Finding("torch", "2.12.0", "PYSEC-2025-194", ("2.13.0",)),
        Finding("some-lib", "1.0.0", "GHSA-new", ("1.0.1",)),
    ]
    acceptances = {
        ("PYSEC-2025-194", "torch"): Acceptance("PYSEC-2025-194", "torch", "r", "SECURITY.md")
    }
    unaccepted, stale = evaluate(findings, acceptances)
    assert unaccepted == [findings[1]]
    assert stale == []


# --- parse_lockfile_packages / filter_in_scope -------------------------------------


def test_parse_lockfile_packages_reads_exact_pins(tmp_path):
    lock = _write_toml(
        tmp_path / "requirements.lock",
        """\
        torch==2.13.0 \\
            --hash=sha256:deadbeef
        cryptography==50.0.0 \\
            --hash=sha256:deadbeef
        """,
    )
    assert parse_lockfile_packages(lock) == {"torch", "cryptography"}


def test_parse_lockfile_packages_normalizes_and_ignores_non_pin_lines(tmp_path):
    lock = _write_toml(
        tmp_path / "requirements.lock",
        """\
        # via -r requirements.in
        Torch_Geometric==2.7.0 \\
            --hash=sha256:deadbeef
        setuptools>=83.0.0
        """,
    )
    # torch-geometric: exact pin, normalized. setuptools: a floor (>=), not an exact
    # `==` pin, so it is not treated as an in-scope package by this parser - matching
    # check_lockfile_freshness.py's own distinction between floors and exact pins.
    assert parse_lockfile_packages(lock) == {"torch-geometric"}


def test_filter_in_scope_splits_by_lockfile_membership():
    findings = [
        Finding("torch", "2.12.0", "PYSEC-2025-194", ("2.13.0",)),
        Finding("pip", "24.0", "PYSEC-2026-196", ("26.1.2",)),
    ]
    in_scope, out_of_scope = filter_in_scope(findings, {"torch"})
    assert in_scope == [findings[0]]
    assert out_of_scope == [findings[1]]


# --- main (end-to-end via CLI) --------------------------------------------------


def _write_register(tmp_path, *entries: str):
    """A stand-in Risk-Acceptance Register.

    Tests that accept an advisory must supply one. Before the register check
    existed these tests passed no --register and so silently consulted the REAL
    SECURITY.md, which never contained the fixture's placeholder value: the string
    "SECURITY.md" appears 0 times inside SECURITY.md. They were hermetic only by
    accident of nothing reading it.
    """
    path = tmp_path / "REGISTER.md"
    path.write_text(
        "\n".join(("# Risk-Acceptance Register", *entries)) + "\n", encoding="utf-8"
    )
    return path


def _write_lockfile(tmp_path, *package_names: str):
    lines = "".join(f"{name}==1.0.0 \\\n    --hash=sha256:deadbeef\n" for name in package_names)
    lock = tmp_path / "requirements.lock"
    lock.write_text(lines, encoding="utf-8")
    return lock


def test_main_clean_report_no_acceptances_needed_returns_zero(tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text(_report([{"name": "numpy", "version": "2.4.6", "vulns": []}]), encoding="utf-8")
    accept = tmp_path / "accepted.toml"
    accept.write_text("", encoding="utf-8")
    lockfile = _write_lockfile(tmp_path, "numpy")
    assert main([str(report), "--accept", str(accept), "--lockfile", str(lockfile)]) == 0
    assert "No unaccepted advisories" in capsys.readouterr().out


def test_main_unaccepted_finding_returns_one(tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text(
        _report([{"name": "torch", "version": "2.12.0", "vulns": [{"id": "PYSEC-2025-194", "fix_versions": ["2.13.0"]}]}]),
        encoding="utf-8",
    )
    accept = tmp_path / "accepted.toml"
    accept.write_text("", encoding="utf-8")
    lockfile = _write_lockfile(tmp_path, "torch")
    exit_code = main([str(report), "--accept", str(accept), "--lockfile", str(lockfile)])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "::error::PYSEC-2025-194 affects torch==2.12.0" in out
    assert "fix: 2.13.0" in out


def test_main_accepted_finding_returns_zero(tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text(
        _report([{"name": "torch", "version": "2.12.0", "vulns": [{"id": "PYSEC-2025-194", "fix_versions": ["2.13.0"]}]}]),
        encoding="utf-8",
    )
    accept = _write_toml(
        tmp_path / "accepted.toml",
        """\
        [[accepted]]
        id = "PYSEC-2025-194"
        package = "torch"
        reason = "temporary"
        register_entry = "SECURITY.md"
        """,
    )
    lockfile = _write_lockfile(tmp_path, "torch")
    register = _write_register(tmp_path, "SECURITY.md")
    exit_code = main(
        [str(report), "--accept", str(accept), "--lockfile", str(lockfile),
         "--register", str(register)]
    )
    assert exit_code == 0
    assert "accepted: 1" in capsys.readouterr().out


def test_main_out_of_scope_finding_does_not_block_and_needs_no_acceptance(tmp_path, capsys):
    # The pip/setuptools case, reproduced from a real pip-audit run against a live
    # dev environment: a package with a live finding that is not a lockfile pin
    # (build-time tooling the CI runner provides, not this repo's dependency choice)
    # must never require an allowlist entry and must never fail the build.
    report = tmp_path / "report.json"
    report.write_text(
        _report([{"name": "pip", "version": "24.0", "vulns": [{"id": "PYSEC-2026-196", "fix_versions": ["26.1.2"]}]}]),
        encoding="utf-8",
    )
    accept = tmp_path / "accepted.toml"
    accept.write_text("", encoding="utf-8")
    lockfile = _write_lockfile(tmp_path, "torch")  # pip is NOT a pin here
    exit_code = main([str(report), "--accept", str(accept), "--lockfile", str(lockfile)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "::notice::PYSEC-2026-196 affects pip==24.0" in out
    assert "not a pin" in out


def test_main_malformed_report_returns_two_not_zero(tmp_path):
    # Exit code 2 (tooling error) must be distinguishable from 0 (clean) and 1
    # (unaccepted findings) - a CI step checking `if: failure()` should not have to
    # guess which failure mode occurred.
    report = tmp_path / "report.json"
    report.write_text("{not json", encoding="utf-8")
    lockfile = _write_lockfile(tmp_path, "torch")
    assert main([str(report), "--lockfile", str(lockfile)]) == 2


def test_main_missing_report_file_returns_two(tmp_path):
    lockfile = _write_lockfile(tmp_path, "torch")
    assert main([str(tmp_path / "does_not_exist.json"), "--lockfile", str(lockfile)]) == 2


def test_main_missing_lockfile_returns_two(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(_report([{"name": "numpy", "version": "2.4.6", "vulns": []}]), encoding="utf-8")
    exit_code = main([str(report), "--lockfile", str(tmp_path / "does_not_exist.lock")])
    assert exit_code == 2


def test_main_stale_acceptance_still_returns_zero_but_notices(tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text(_report([{"name": "numpy", "version": "2.4.6", "vulns": []}]), encoding="utf-8")
    accept = _write_toml(
        tmp_path / "accepted.toml",
        """\
        [[accepted]]
        id = "PYSEC-2025-194"
        package = "torch"
        reason = "temporary"
        register_entry = "SECURITY.md"
        """,
    )
    lockfile = _write_lockfile(tmp_path, "numpy", "torch")
    register = _write_register(tmp_path, "SECURITY.md")
    exit_code = main(
        [str(report), "--accept", str(accept), "--lockfile", str(lockfile),
         "--register", str(register)]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "::notice::accepted advisory PYSEC-2025-194" in out


def test_main_missing_accept_file_treated_as_empty_acceptances(tmp_path):
    # A missing acceptance file must fail closed on any real finding, not silently
    # pass everything - this is the same "no file means nothing is accepted" contract
    # as load_acceptances, exercised through the CLI entry point.
    report = tmp_path / "report.json"
    report.write_text(
        _report([{"name": "torch", "version": "2.12.0", "vulns": [{"id": "PYSEC-2025-194", "fix_versions": []}]}]),
        encoding="utf-8",
    )
    lockfile = _write_lockfile(tmp_path, "torch")
    exit_code = main(
        [str(report), "--accept", str(tmp_path / "missing.toml"), "--lockfile", str(lockfile)]
    )
    assert exit_code == 1


# ---------------------------------------------------------------------------
# The Risk-Acceptance Register requirement
# ---------------------------------------------------------------------------
#
# environments/accepted_advisories.toml states that every entry MUST have a
# matching Risk-Acceptance Register entry in SECURITY.md carrying a re-review
# trigger. Nothing checked it: register_entry was parsed into Acceptance and
# never read again, and the file's other SECURITY.md mentions are static strings
# in error messages. An unenforced MUST on the sole suppression mechanism for a
# security gate is the worst possible place for one.


def test_unregistered_acceptances_flags_an_entry_absent_from_the_register():
    acceptances = {
        ("PYSEC-1", "torch"): Acceptance("PYSEC-1", "torch", "r", "listed entry"),
        ("PYSEC-2", "numpy"): Acceptance("PYSEC-2", "numpy", "r", "absent entry"),
    }
    missing = unregistered_acceptances(acceptances, "register holds: listed entry")
    assert [a.advisory_id for a in missing] == ["PYSEC-2"]


def test_unregistered_acceptances_treats_an_empty_register_entry_as_missing():
    acceptances = {("PYSEC-1", "torch"): Acceptance("PYSEC-1", "torch", "r", "")}
    assert unregistered_acceptances(acceptances, "anything at all") != []


def test_main_acceptance_without_a_register_entry_fails(tmp_path, capsys):
    """The positive control. Without it the check could not be shown to fire."""
    report = tmp_path / "report.json"
    report.write_text(
        _report([{"name": "torch", "version": "2.12.0",
                  "vulns": [{"id": "PYSEC-2025-194", "fix_versions": ["2.13.0"]}]}]),
        encoding="utf-8",
    )
    accept = _write_toml(
        tmp_path / "accepted.toml",
        """        [[accepted]]
        id = "PYSEC-2025-194"
        package = "torch"
        reason = "temporary"
        register_entry = "PYSEC-2025-194 torch"
        """,
    )
    lockfile = _write_lockfile(tmp_path, "torch")
    register = _write_register(tmp_path, "some unrelated register entry")

    exit_code = main(
        [str(report), "--accept", str(accept), "--lockfile", str(lockfile),
         "--register", str(register)]
    )

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "does not appear in" in out
    assert "PYSEC-2025-194" in out


def test_main_unreadable_register_with_acceptances_fails_closed(tmp_path, capsys):
    """Cannot verify is not the same as verified."""
    report = tmp_path / "report.json"
    report.write_text(
        _report([{"name": "torch", "version": "2.12.0",
                  "vulns": [{"id": "PYSEC-2025-194", "fix_versions": ["2.13.0"]}]}]),
        encoding="utf-8",
    )
    accept = _write_toml(
        tmp_path / "accepted.toml",
        """        [[accepted]]
        id = "PYSEC-2025-194"
        package = "torch"
        reason = "temporary"
        register_entry = "PYSEC-2025-194 torch"
        """,
    )
    lockfile = _write_lockfile(tmp_path, "torch")

    exit_code = main(
        [str(report), "--accept", str(accept), "--lockfile", str(lockfile),
         "--register", str(tmp_path / "no_such_register.md")]
    )

    assert exit_code == 2
    assert "cannot read" in capsys.readouterr().out


def test_main_with_no_acceptances_does_not_require_a_register(tmp_path, capsys):
    """An empty acceptance list must not make the gate depend on SECURITY.md."""
    report = tmp_path / "report.json"
    report.write_text(_report([{"name": "torch", "version": "2.12.0", "vulns": []}]),
                      encoding="utf-8")
    accept = _write_toml(tmp_path / "accepted.toml", "")
    lockfile = _write_lockfile(tmp_path, "torch")

    exit_code = main(
        [str(report), "--accept", str(accept), "--lockfile", str(lockfile),
         "--register", str(tmp_path / "no_such_register.md")]
    )

    assert exit_code == 0


def test_the_shipped_acceptance_list_is_fully_registered():
    """Live tree, and VACUOUS TODAY: the shipped list parses to 0 entries.

    Stated rather than hidden. A textual grep for "[[accepted]]" returns 1, from a
    commented example, so the count must be parsed and never grepped. This test
    becomes load-bearing the moment a real acceptance is added, which is exactly
    when the requirement matters.
    """
    import tomllib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    accept_path = repo_root / "environments" / "accepted_advisories.toml"
    entries = tomllib.loads(accept_path.read_text(encoding="utf-8")).get("accepted", [])

    register_text = (repo_root / "SECURITY.md").read_text(encoding="utf-8")
    missing = [e for e in entries if str(e["register_entry"]).strip() not in register_text]
    assert missing == [], f"acceptances with no SECURITY.md register entry: {missing}"
