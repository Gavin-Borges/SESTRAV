"""Tests for tools/audit_all_locks.py, the advisory audit over every tracked lock.

The gap it closes, measured 2026-10-17: CVE-2026-59951 was published against mcp 1.29.0
two days after nothing could have caught it, because the only fail-closed advisory gate
reads environments/requirements.lock alone and the pin sat in
environments/requirements-semgrep.txt. Two properties matter more than any single
verdict, and both are tested here without the network:

1. SET COVERAGE. Every tracked hash-pinned lock is audited. The population is derived
   a second, independent way - from git's own view of tracked file CONTENT - and
   compared with what the script would audit, so adding a lock the script never sees
   fails here rather than going quiet.
2. NO FALSE CLEAN. A report that audited fewer packages than the lock pins, skipped a
   package, was malformed, or was never written must not read as a pass.

pip-audit itself is never run: a fake writes canned JSON shaped like pip-audit 2.10.1's
`--format json` output, including its `skip_reason` form for a skipped dependency.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import textwrap

import pytest

from tools import audit_all_locks
from tools import check_lockfile_advisories as advisories
from tools.check_lockfile_freshness import LOCKFILE_PAIRS
from tools.update_dependencies import LOCK_SPECS

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SEMGREP_LOCK = "environments/requirements-semgrep.txt"


# --- set coverage: every tracked lock is audited ------------------------------------


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    # Inherited GIT_DIR / GIT_WORK_TREE (pre-push exports them) would outrank -C.
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")
    }
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=60,
    )


def _git_grep_files(pattern: str) -> set[str]:
    result = _git("grep", "-l", "-E", "-e", pattern, "--", ".", ":(exclude)tests/")
    # 1 is git grep's "no match", which is an answer; anything else is a failure.
    assert result.returncode in (0, 1), f"git grep failed: {result.stderr}"
    return set(result.stdout.splitlines())


def _tracked_hash_pinned_locks() -> set[str]:
    """The population, derived WITHOUT check_hash_pins' globs.

    A tracked file is a lock when its own lines pin a requirement with `==` at the
    start of a line AND carry a `--hash=sha256:` digest. tests/ is left out: its
    fixtures are inputs to tests, not something any job installs. Measured
    2026-10-17, this returned 16 paths over the whole tracked tree, and the same 16
    without the tests/ exclusion; nothing under tests/ qualifies today.
    """
    pinned = _git_grep_files(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[^]]*\])?==[^[:space:]]+")
    hashed = _git_grep_files(r"--hash=sha256:[0-9a-f]{64}")
    return pinned & hashed


@pytest.fixture(scope="module")
def tracked_files() -> set[str]:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("needs a git checkout: the population is read from git")
    result = _git("ls-files")
    assert result.returncode == 0, result.stderr
    return set(result.stdout.splitlines())


def test_every_tracked_hash_pinned_lock_is_audited(tracked_files):
    population = _tracked_hash_pinned_locks()
    # Anti-vacuity: an empty or broken git grep must not read as "nothing missing".
    assert {"environments/requirements.lock", "requirements.txt", SEMGREP_LOCK} <= population

    audited = set(audit_all_locks.discover_locks())
    unaudited = sorted(population - audited)
    assert not unaudited, (
        f"tracked hash-pinned lock(s) tools/audit_all_locks.py never audits: {unaudited}. "
        f"Widen tools/check_hash_pins.py's DEFAULT_TARGETS, which both gates read."
    )
    # And the other direction, over tracked paths only (an untracked local file can
    # match the globs): nothing audited is a file that is not a lock.
    not_locks = sorted((audited & tracked_files) - population)
    assert not not_locks, f"audited tracked file(s) that pin nothing with hashes: {not_locks}"


def test_every_compiled_lock_is_audited():
    # A third source: the .in -> lock pairs and the recompile specs. Both list only
    # the compiled locks, never the hand-curated ones, so this is a subset check.
    audited = set(audit_all_locks.discover_locks())
    compiled = {output for _source, output in LOCKFILE_PAIRS} | {s.output for s in LOCK_SPECS}
    assert compiled <= audited, sorted(compiled - audited)


def test_a_named_lock_that_vanished_is_still_audited_and_fails():
    # requirements.txt is a non-glob target: if it were deleted it must surface as
    # an unauditable lock, not silently drop out of the run.
    assert "requirements.txt" in audit_all_locks.discover_locks()
    result = audit_all_locks.audit_lock(
        "no/such/requirements.lock", REPO_ROOT / "never-written.json", {}, timeout=1
    )
    assert result.status == "ERROR"
    assert "cannot read the lock" in (result.error or "")


# --- the pip-audit invocation -------------------------------------------------------


def test_pip_audit_command_audits_the_pins_as_written_and_suppresses_nothing(tmp_path):
    command = audit_all_locks.pip_audit_command(tmp_path / "x.txt", tmp_path / "x.json")
    assert command[1:3] == ["-m", "pip_audit"]
    for flag in ("--no-deps", "--disable-pip"):
        assert flag in command
    assert command[command.index("--format") + 1] == "json"
    assert command[command.index("-r") + 1] == str(tmp_path / "x.txt")
    assert not any(part.startswith("--ignore-vuln") for part in command)


def test_report_names_cannot_collide_across_directories():
    names = {audit_all_locks.report_name(lock) for lock in audit_all_locks.discover_locks()}
    assert len(names) == len(audit_all_locks.discover_locks())


# --- judging one report (canned pip-audit JSON) --------------------------------------


def _report(*deps: dict) -> str:
    return json.dumps({"dependencies": list(deps), "fixes": []})


def _dep(name: str, *vuln_ids: str, version: str = "1.0") -> dict:
    return {
        "name": name,
        "version": version,
        "vulns": [{"id": v, "fix_versions": ["9.9"], "aliases": []} for v in vuln_ids],
    }


def _acceptance(advisory_id: str, package: str, lock: str) -> advisories.Acceptance:
    return advisories.Acceptance(advisory_id, package, "r", "REGISTER", lock=lock)


def test_judge_clean_report_is_ok():
    result = audit_all_locks.judge_report(
        "a.txt", _report(_dep("numpy"), _dep("pandas")), {"numpy", "pandas"}, {}
    )
    assert result.status == "ok"
    assert (result.audited, result.in_scope, result.accepted) == (2, 0, 0)
    assert "[ok   ] a.txt: 2 audited, 0 skipped, 0 missing" in result.summary_line()


def test_judge_unaccepted_finding_fails_and_names_the_lock():
    result = audit_all_locks.judge_report(
        SEMGREP_LOCK, _report(_dep("mcp", "CVE-2026-59951", version="1.29.0")), {"mcp"}, {}
    )
    assert result.status == "FAIL"
    assert audit_all_locks.exit_code([result]) == 1
    errors = [line for line in result.annotations("accept.toml") if line.startswith("::error::")]
    assert errors == [
        "::error::CVE-2026-59951 affects mcp==1.29.0 in environments/requirements-semgrep.txt "
        "(fix: 9.9) and is not accepted for that lock in accept.toml"
    ]


def test_judge_finding_accepted_for_its_own_lock_passes():
    acceptances = advisories.acceptances_for_lock(
        {("CVE-1", "mcp", SEMGREP_LOCK): _acceptance("CVE-1", "mcp", SEMGREP_LOCK)},
        SEMGREP_LOCK,
    )
    result = audit_all_locks.judge_report(
        SEMGREP_LOCK, _report(_dep("mcp", "CVE-1")), {"mcp"}, acceptances
    )
    assert result.status == "ok"
    assert (result.in_scope, result.accepted) == (1, 1)


def test_judge_acceptance_for_another_lock_does_not_cover_the_finding():
    # The scoping this unit adds: the production lock's acceptance of (CVE-1, mcp)
    # says nothing about the same pin in the semgrep tool lock.
    all_acceptances = {
        ("CVE-1", "mcp", advisories.DEFAULT_LOCKFILE): _acceptance(
            "CVE-1", "mcp", advisories.DEFAULT_LOCKFILE
        )
    }
    result = audit_all_locks.judge_report(
        SEMGREP_LOCK,
        _report(_dep("mcp", "CVE-1")),
        {"mcp"},
        advisories.acceptances_for_lock(all_acceptances, SEMGREP_LOCK),
    )
    assert result.status == "FAIL"
    assert [f.package for f in result.unaccepted] == ["mcp"]


def test_judge_skipped_dependency_is_not_audited():
    skipped = {"name": "torch", "skip_reason": "Dependency not found on PyPI"}
    result = audit_all_locks.judge_report(
        "a.txt", _report(_dep("numpy"), skipped), {"numpy", "torch"}, {}
    )
    assert result.status == "ERROR"
    assert result.skipped == ("torch (Dependency not found on PyPI)",)
    assert result.missing == ()
    assert audit_all_locks.exit_code([result]) == 2


def test_judge_pinned_package_absent_from_the_report_is_not_audited():
    result = audit_all_locks.judge_report(
        "a.txt", _report(_dep("numpy")), {"numpy", "zope.interface"}, {}
    )
    assert result.status == "ERROR"
    assert result.missing == ("zope-interface",)


def test_judge_matches_pins_and_report_names_in_pep503_form():
    # pip-audit reports canonical names; a lock may spell them with _ or . instead.
    result = audit_all_locks.judge_report(
        "a.txt",
        _report(_dep("typing-extensions"), _dep("zope-interface", "X-1")),
        {"typing_extensions", "zope.interface"},
        {},
    )
    assert result.missing == ()
    assert [f.package for f in result.unaccepted] == ["zope-interface"]


@pytest.mark.parametrize(
    "text, reason",
    [("{not json", "not valid JSON"), (_report(), "zero packages"), ("{}", "no 'dependencies'")],
)
def test_judge_malformed_or_empty_report_is_an_error_not_a_pass(text, reason):
    result = audit_all_locks.judge_report("a.txt", text, {"numpy"}, {})
    assert result.status == "ERROR"
    assert reason in (result.error or "")


def test_judge_still_reports_findings_on_an_incompletely_audited_lock():
    result = audit_all_locks.judge_report(
        "a.txt", _report(_dep("numpy", "X-1")), {"numpy", "pandas"}, {}
    )
    assert result.status == "ERROR"
    assert [f.advisory_id for f in result.unaccepted] == ["X-1"]


def test_exit_code_precedence():
    ok = audit_all_locks.LockResult(lock="a")
    failing = audit_all_locks.LockResult(
        lock="b", in_scope=1, unaccepted=[advisories.Finding("p", "1", "X", ())]
    )
    broken = audit_all_locks.LockResult(lock="c", error="no report")
    assert audit_all_locks.exit_code([ok]) == 0
    assert audit_all_locks.exit_code([ok], register_failed=True) == 1
    assert audit_all_locks.exit_code([ok, failing]) == 1
    assert audit_all_locks.exit_code([ok, failing, broken]) == 2


# --- end to end through main(), with pip-audit faked ---------------------------------


def _fake_pip_audit(vulns: dict[str, dict[str, list[str]]], *, write: bool = True):
    """Stand in for pip-audit: report every package the lock pins, with canned vulns.

    ``vulns`` maps a lock's repo-relative path to {package: [advisory ids]}.
    """
    calls: list[list[str]] = []

    def fake(command, timeout):
        calls.append(list(command))
        lock_path = pathlib.Path(command[command.index("-r") + 1])
        output = pathlib.Path(command[command.index("--output") + 1])
        lock = lock_path.resolve().relative_to(REPO_ROOT).as_posix()
        found = vulns.get(lock, {})
        deps = [
            _dep(name, *found.get(name, []))
            for name in sorted(advisories.parse_lockfile_packages(lock_path))
        ]
        if write:
            output.write_text(_report(*deps), encoding="utf-8")
        return (1 if found else 0), ""

    fake.calls = calls  # type: ignore[attr-defined]
    return fake


def _empty_accept(tmp_path) -> pathlib.Path:
    path = tmp_path / "accepted.toml"
    path.write_text("", encoding="utf-8")
    return path


def test_main_reports_one_line_per_lock_and_fails_on_the_one_finding(
    tmp_path, monkeypatch, capsys
):
    fake = _fake_pip_audit({SEMGREP_LOCK: {"mcp": ["CVE-2026-59951"]}})
    monkeypatch.setattr(audit_all_locks, "run_pip_audit", fake)
    code = audit_all_locks.main(
        ["--report-dir", str(tmp_path / "reports"), "--accept", str(_empty_accept(tmp_path))]
    )
    out = capsys.readouterr().out
    locks = audit_all_locks.discover_locks()

    assert code == 1
    assert len(fake.calls) == len(locks)
    for lock in locks:
        status = "FAIL " if lock == SEMGREP_LOCK else "ok   "
        assert f"[{status}] {lock}: " in out
        assert (tmp_path / "reports" / audit_all_locks.report_name(lock)).is_file()
    assert "::error::CVE-2026-59951 affects mcp==1.0 in environments/requirements-semgrep.txt" in out
    assert f"TOTAL: {len(locks)} lock(s);" in out
    assert "1 unaccepted in 1 lock(s); 0 lock(s) not fully audited" in out


def test_main_accepts_a_finding_only_for_the_lock_its_entry_names(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        audit_all_locks, "run_pip_audit", _fake_pip_audit({SEMGREP_LOCK: {"mcp": ["CVE-1"]}})
    )
    accept = tmp_path / "accepted.toml"
    register = tmp_path / "REGISTER.md"
    register.write_text("CVE-1 mcp semgrep\n", encoding="utf-8")
    common = ["--report-dir", str(tmp_path / "r"), "--register", str(register)]

    # No `lock` field: covers environments/requirements.lock only, so still unaccepted.
    accept.write_text(
        textwrap.dedent(
            """\
            [[accepted]]
            id = "CVE-1"
            package = "mcp"
            reason = "r"
            register_entry = "CVE-1 mcp semgrep"
            """
        ),
        encoding="utf-8",
    )
    assert audit_all_locks.main([*common, "--accept", str(accept)]) == 1
    capsys.readouterr()

    accept.write_text(
        accept.read_text(encoding="utf-8") + f'lock = "{SEMGREP_LOCK}"\n', encoding="utf-8"
    )
    assert audit_all_locks.main([*common, "--accept", str(accept)]) == 0
    assert "TOTAL:" in capsys.readouterr().out


def test_main_unregistered_acceptance_for_any_lock_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(audit_all_locks, "run_pip_audit", _fake_pip_audit({}))
    accept = tmp_path / "accepted.toml"
    accept.write_text(
        textwrap.dedent(
            f"""\
            [[accepted]]
            id = "CVE-1"
            package = "mcp"
            reason = "r"
            register_entry = "absent from the register"
            lock = "{SEMGREP_LOCK}"
            """
        ),
        encoding="utf-8",
    )
    register = tmp_path / "REGISTER.md"
    register.write_text("something else\n", encoding="utf-8")
    code = audit_all_locks.main(
        ["--report-dir", str(tmp_path / "r"), "--accept", str(accept), "--register", str(register)]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "does not appear in" in out
    # The stale notice proves the entry was scoped to the semgrep lock, not dropped.
    assert f"::notice::accepted advisory CVE-1 for mcp ({SEMGREP_LOCK})" in out


def test_main_warns_about_an_acceptance_for_a_lock_that_is_not_audited(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(audit_all_locks, "run_pip_audit", _fake_pip_audit({}))
    accept = tmp_path / "accepted.toml"
    accept.write_text(
        textwrap.dedent(
            """\
            [[accepted]]
            id = "CVE-1"
            package = "mcp"
            reason = "r"
            register_entry = "REG"
            lock = "environments/requirements-semgrp.txt"
            """
        ),
        encoding="utf-8",
    )
    register = tmp_path / "REGISTER.md"
    register.write_text("REG\n", encoding="utf-8")
    code = audit_all_locks.main(
        ["--report-dir", str(tmp_path / "r"), "--accept", str(accept), "--register", str(register)]
    )
    assert code == 0
    assert "which is not an audited lock, so it can never apply" in capsys.readouterr().out


def test_main_run_that_writes_no_report_is_a_tooling_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(audit_all_locks, "run_pip_audit", _fake_pip_audit({}, write=False))
    code = audit_all_locks.main(
        ["--report-dir", str(tmp_path / "r"), "--accept", str(_empty_accept(tmp_path))]
    )
    out = capsys.readouterr().out
    assert code == 2
    assert "pip-audit exited 0 and wrote no report" in out
    assert f"{len(audit_all_locks.discover_locks())} lock(s) not fully audited" in out


def test_main_never_judges_a_report_left_by_an_earlier_run(tmp_path, monkeypatch, capsys):
    reports = tmp_path / "r"
    accept = str(_empty_accept(tmp_path))
    monkeypatch.setattr(audit_all_locks, "run_pip_audit", _fake_pip_audit({}))
    assert audit_all_locks.main(["--report-dir", str(reports), "--accept", accept]) == 0
    # Every report from that clean run is still on disk; a run that writes nothing
    # must not be judged on them.
    monkeypatch.setattr(audit_all_locks, "run_pip_audit", _fake_pip_audit({}, write=False))
    assert audit_all_locks.main(["--report-dir", str(reports), "--accept", accept]) == 2
    capsys.readouterr()


def test_main_unreadable_acceptance_file_is_a_tooling_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(audit_all_locks, "run_pip_audit", _fake_pip_audit({}))
    accept = tmp_path / "accepted.toml"
    accept.write_text('[[accepted]]\nid = "X"\n', encoding="utf-8")  # no package field
    code = audit_all_locks.main(["--report-dir", str(tmp_path / "r"), "--accept", str(accept)])
    assert code == 2
    assert "::error::cannot read" in capsys.readouterr().out
