"""scripts/check_semgrep_rule_fixtures.py compares semgrep's findings with fixture annotations.

These tests never run semgrep: they feed the comparison synthetic semgrep --json payloads,
so they hold the gate's logic in place on any machine. The CI step in security.yml runs the
real semgrep against the real fixtures.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "scripts" / "check_semgrep_rule_fixtures.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_semgrep_rule_fixtures", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load()


def _tree(tmp_path: Path) -> Path:
    rules = tmp_path / "semgrep-rules"
    rules.mkdir()
    (rules / "demo.yml").write_text("rules:\n  - id: rule-a\n  - id: rule-b\n", encoding="utf-8")
    (rules / "demo.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# ruleid: rule-a\n"
        "bad a\n"
        "\n"
        "# ok: rule-a\n"
        "good a\n"
        "# ruleid: rule-b\n"
        "bad b\n",
        encoding="utf-8",
    )
    return rules


LOADED = ("rule-a", "rule-b")


def _payload(*hits: tuple[str, str, int], loaded: tuple[str, ...] = LOADED) -> dict:
    return {
        "errors": [],
        "skipped_rules": [],
        "time": {"rules": [f"semgrep-rules.{rule}" for rule in loaded]},
        "results": [
            {"check_id": f"semgrep-rules.{rule}", "path": path, "start": {"line": line}}
            for rule, path, line in hits
        ],
    }


def _verdict(tmp_path, payload):
    rules = _tree(tmp_path)
    expected, forbidden = gate.fixture_annotations(rules / "demo.sh", root=tmp_path)
    found = gate.findings_from_json(payload, root=tmp_path)
    return gate.compare(found, expected, forbidden, gate.loaded_rules(payload))


def test_annotations_target_the_next_code_line(tmp_path):
    rules = _tree(tmp_path)
    expected, forbidden = gate.fixture_annotations(rules / "demo.sh", root=tmp_path)
    assert expected == {
        gate.Hit("rule-a", "semgrep-rules/demo.sh", 3),
        gate.Hit("rule-b", "semgrep-rules/demo.sh", 8),
    }
    assert forbidden == {gate.Hit("rule-a", "semgrep-rules/demo.sh", 6)}


def test_findings_matching_the_annotations_pass(tmp_path):
    payload = _payload(
        ("rule-a", "semgrep-rules/demo.sh", 3), ("rule-b", "semgrep-rules/demo.sh", 8)
    )
    assert _verdict(tmp_path, payload) == []


def test_a_rule_that_stopped_firing_fails(tmp_path):
    payload = _payload(("rule-a", "semgrep-rules/demo.sh", 3))
    assert _verdict(tmp_path, payload) == ["MISSING rule-b at semgrep-rules/demo.sh line 8"]


def test_a_rule_firing_on_an_ok_line_fails(tmp_path):
    payload = _payload(
        ("rule-a", "semgrep-rules/demo.sh", 3),
        ("rule-a", "semgrep-rules/demo.sh", 6),
        ("rule-b", "semgrep-rules/demo.sh", 8),
    )
    assert _verdict(tmp_path, payload) == [
        "FIRED ON AN ok: LINE rule-a at semgrep-rules/demo.sh line 6"
    ]


def test_an_unannotated_finding_fails(tmp_path):
    payload = _payload(
        ("rule-a", "semgrep-rules/demo.sh", 3),
        ("rule-b", "semgrep-rules/demo.sh", 8),
        ("rule-b", "semgrep-rules/demo.sh", 1),
    )
    assert _verdict(tmp_path, payload) == ["UNANNOTATED rule-b at semgrep-rules/demo.sh line 1"]


def test_every_loaded_rule_needs_a_ruleid_fixture_whatever_file_defines_it(tmp_path):
    # rule-c could live in a rule file with no fixture beside it, written in any YAML
    # style: semgrep reports it as loaded, and that is what the coverage check reads.
    payload = _payload(
        ("rule-a", "semgrep-rules/demo.sh", 3),
        ("rule-b", "semgrep-rules/demo.sh", 8),
        loaded=("rule-a", "rule-b", "rule-c"),
    )
    assert _verdict(tmp_path, payload) == ["NO ruleid: FIXTURE for rule-c"]


def test_a_dotted_rule_id_is_kept_whole():
    assert gate.rule_name("semgrep-rules.sestrav-rule.v2") == "sestrav-rule.v2"
    assert gate.rule_name("other.rule") == "other.rule"


def test_skipped_rules_are_not_a_clean_result(tmp_path):
    payload = _payload(("rule-a", "semgrep-rules/demo.sh", 3))
    payload["skipped_rules"] = [{"rule_id": "semgrep-rules.rule-b"}]
    with pytest.raises(ValueError, match="skipped rules"):
        gate.findings_from_json(payload, root=tmp_path)


def test_a_payload_without_the_rule_list_cannot_run(tmp_path, monkeypatch):
    rules = _tree(tmp_path)
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gate, "RULES_DIR", rules)
    payload = _payload(
        ("rule-a", "semgrep-rules/demo.sh", 3), ("rule-b", "semgrep-rules/demo.sh", 8)
    )
    del payload["time"]
    findings = tmp_path / "no_time.json"
    findings.write_text(json.dumps(payload), encoding="utf-8")
    assert gate.main(["--findings", str(findings)]) == 2


def test_semgrep_errors_are_not_read_as_a_clean_result(tmp_path):
    with pytest.raises(ValueError, match="semgrep reported errors"):
        gate.findings_from_json(
            {"errors": [{"message": "rule failed to parse"}], "results": []}, root=tmp_path
        )


def test_cli_exit_codes_from_a_findings_file(tmp_path, monkeypatch, capsys):
    rules = _tree(tmp_path)
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gate, "RULES_DIR", rules)
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            _payload(("rule-a", "semgrep-rules/demo.sh", 3), ("rule-b", "semgrep-rules/demo.sh", 8))
        ),
        encoding="utf-8",
    )
    assert gate.main(["--findings", str(good)]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_payload(("rule-a", "semgrep-rules/demo.sh", 3))), encoding="utf-8")
    assert gate.main(["--findings", str(bad)]) == 1
    assert "MISSING rule-b" in capsys.readouterr().err
    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text("not json", encoding="utf-8")
    assert gate.main(["--findings", str(unreadable)]) == 2


def test_a_ruleid_line_may_name_several_rules(tmp_path):
    fixture = tmp_path / "semgrep-rules" / "multi.sh"
    fixture.parent.mkdir()
    fixture.write_text("# ruleid: rule-a, rule-b\nbad\n", encoding="utf-8")
    expected, _ = gate.fixture_annotations(fixture, root=tmp_path)
    assert expected == {
        gate.Hit("rule-a", "semgrep-rules/multi.sh", 2),
        gate.Hit("rule-b", "semgrep-rules/multi.sh", 2),
    }


def test_the_live_fixtures_annotate_every_rule_in_every_rule_file():
    # The real tree, without semgrep. semgrep's own rule list leaves out a rule whose
    # language no fixture uses, so this is the check for those: every rule id in every
    # rule file under semgrep-rules/, at any depth (semgrep loads the folder
    # recursively), read by a YAML parser, is unique and has a ruleid: line.
    yaml = pytest.importorskip("yaml")
    annotated = set()
    for fixture in gate.fixture_files():
        expected, _ = gate.fixture_annotations(fixture)
        annotated |= {hit.rule for hit in expected}
    rule_files = sorted(
        p for p in gate.RULES_DIR.rglob("*") if p.is_file() and p.suffix in gate.RULE_SUFFIXES
    )
    assert rule_files, "no rule files found; the gate would check nothing"
    seen: dict[str, str] = {}
    for rule_file in rule_files:
        name = rule_file.relative_to(gate.RULES_DIR).as_posix()
        ids = [
            rule["id"] for rule in yaml.safe_load(rule_file.read_text(encoding="utf-8"))["rules"]
        ]
        assert ids, f"{name} defines no rules"
        for rule_id in ids:
            assert rule_id not in seen, (
                f"rule id {rule_id} is defined in {seen[rule_id]} and {name}"
            )
            seen[rule_id] = name
        missing = sorted(set(ids) - annotated)
        assert not missing, f"{name}: no ruleid: fixture for {missing}"
