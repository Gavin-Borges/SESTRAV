"""Regression tests for tools/check_sbom_freshness.py.

The gate this script backs closes a drift that went unnoticed for six weeks:
`docs/sbom.json` and `docs/DEPENDENCY_LICENSES.md` are committed `pip-licenses` output
that nothing regenerated or compared against the lockfiles, so 46 of the 124 packages
they share with `environments/requirements.lock` had gone stale, four of them in the
security-relevant direction.

These exercise the parse/compare logic against fixtures rather than a live
`pip-licenses` run, because the production lockfile is Linux-compiled and cannot be
installed on a Windows dev box (`nvidia-cufile` ships no Windows wheel). The CI job is
where the real generation happens; this file pins the comparison semantics.
"""

from __future__ import annotations

import json

import pytest

from tools.check_sbom_freshness import (
    ArtifactError,
    compare,
    main,
    normalize,
    parse_artifact,
)


def _json_artifact(entries: list[tuple[str, str]]) -> str:
    return json.dumps([{"Name": n, "Version": v, "License": "MIT"} for n, v in entries])


def _md_artifact(entries: list[tuple[str, str]]) -> str:
    lines = ["| Name | Version | License |", "|---|---|---|"]
    lines += [f"| {n} | {v} | MIT |" for n, v in entries]
    return "\n".join(lines)


# --- normalize -------------------------------------------------------------------


def test_normalize_pep503():
    assert normalize("Foo_Bar") == "foo-bar"
    assert normalize("  GitPython ") == "gitpython"


# --- parse_artifact --------------------------------------------------------------


def test_parse_json_artifact():
    assert parse_artifact(_json_artifact([("torch", "2.13.0")]), "x.json") == {"torch": "2.13.0"}


def test_parse_markdown_artifact():
    got = parse_artifact(_md_artifact([("torch", "2.13.0"), ("numpy", "2.4.6")]), "x.md")
    assert got == {"torch": "2.13.0", "numpy": "2.4.6"}


def test_parse_normalizes_names_in_both_formats():
    assert "gitpython" in parse_artifact(_json_artifact([("GitPython", "3.1.57")]), "x.json")
    assert "gitpython" in parse_artifact(_md_artifact([("GitPython", "3.1.57")]), "x.md")


def test_parse_markdown_ignores_header_and_separator_rows():
    # The header row's "Version" is not a version, and the |---|---| separator must not
    # be mistaken for a package. Only real rows count.
    got = parse_artifact(_md_artifact([("torch", "2.13.0")]), "x.md")
    assert got == {"torch": "2.13.0"}


def test_parse_json_skips_entries_missing_name_or_version():
    payload = json.dumps(
        [
            {"Name": "torch", "Version": "2.13.0", "License": "BSD-3-Clause"},
            {"Name": "", "Version": "1.0", "License": "MIT"},
            {"Name": "broken", "License": "MIT"},
        ]
    )
    assert parse_artifact(payload, "x.json") == {"torch": "2.13.0"}


def test_parse_empty_artifact_raises():
    with pytest.raises(ArtifactError, match="is empty"):
        parse_artifact("   \n  ", "x.json")


def test_parse_invalid_json_raises():
    with pytest.raises(ArtifactError, match="not valid JSON"):
        parse_artifact("[{not json", "x.json")


def test_parse_json_non_list_raises():
    with pytest.raises(ArtifactError, match="not a JSON list"):
        parse_artifact('{"Name": "torch"}', "x.json")


def test_parse_artifact_with_zero_packages_raises():
    # A structurally valid but package-free artifact must never read as "nothing to
    # compare, therefore fresh" - that would silently disable the gate exactly when
    # SBOM generation had broken.
    with pytest.raises(ArtifactError, match="zero packages"):
        parse_artifact("[]", "x.json")
    with pytest.raises(ArtifactError, match="zero packages"):
        parse_artifact("| Name | Version | License |\n|---|---|---|", "x.md")


# --- compare ---------------------------------------------------------------------


def test_compare_detects_drift():
    drifted, undocumented, not_regenerated = compare({"torch": "2.13.0"}, {"torch": "2.12.0"})
    assert drifted == [("torch", "2.12.0", "2.13.0")]
    assert undocumented == []
    assert not_regenerated == []


def test_compare_clean_when_versions_match():
    drifted, undocumented, not_regenerated = compare({"torch": "2.13.0"}, {"torch": "2.13.0"})
    assert drifted == []
    assert undocumented == []
    assert not_regenerated == []


def test_compare_reports_not_regenerated_without_treating_it_as_drift():
    # A committed package absent from the fresh run is expected: the CI install step is
    # continue-on-error because platform wheels (CUDA etc.) legitimately fail on the
    # runner. It must be reported, never fatal.
    drifted, undocumented, not_regenerated = compare(
        {"torch": "2.13.0"}, {"torch": "2.13.0", "nvidia-cufile": "1.15.1.6"}
    )
    assert drifted == []
    assert undocumented == []
    assert not_regenerated == ["nvidia-cufile"]


def test_compare_flags_package_missing_from_committed_artifact():
    # Regression test for a real hole in the first revision of this tool: it compared
    # only the intersection, so a package deleted outright from the committed SBOM
    # passed as "in sync". Silent deletion is the strongest form of the
    # understates-its-own-posture failure this gate exists to prevent.
    drifted, undocumented, not_regenerated = compare(
        {"torch": "2.13.0", "cryptography": "50.0.0"}, {"torch": "2.13.0"}
    )
    assert drifted == []
    assert undocumented == ["cryptography"]
    assert not_regenerated == []


def test_compare_drift_is_sorted_and_reports_both_versions():
    drifted, _, _ = compare(
        {"aaa": "2.0", "zzz": "2.0", "mmm": "2.0"},
        {"zzz": "1.0", "aaa": "1.0", "mmm": "1.0"},
    )
    assert [d[0] for d in drifted] == ["aaa", "mmm", "zzz"]
    assert drifted[0] == ("aaa", "1.0", "2.0")


# --- main (end-to-end) -----------------------------------------------------------


def _write_pair(tmp_path, generated, committed, gen_name="gen.json", com_name="com.json"):
    g = tmp_path / gen_name
    c = tmp_path / com_name
    g.write_text(generated, encoding="utf-8")
    c.write_text(committed, encoding="utf-8")
    return g, c


def test_main_in_sync_returns_zero(tmp_path, capsys):
    g, c = _write_pair(
        tmp_path, _json_artifact([("torch", "2.13.0")]), _json_artifact([("torch", "2.13.0")])
    )
    assert main(["--generated", str(g), "--committed", str(c)]) == 0
    assert "in sync" in capsys.readouterr().out


def test_main_drift_returns_one_and_names_both_versions(tmp_path, capsys):
    g, c = _write_pair(
        tmp_path, _json_artifact([("torch", "2.13.0")]), _json_artifact([("torch", "2.12.0")])
    )
    exit_code = main(["--generated", str(g), "--committed", str(c)])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "::error::" in out
    assert "torch==2.12.0" in out
    assert "torch==2.13.0" in out
    # The fix path must not invite hand-editing individual rows.
    assert "Do not hand-edit" in out


def test_main_missing_package_is_a_notice_not_a_failure(tmp_path, capsys):
    g, c = _write_pair(
        tmp_path,
        _json_artifact([("torch", "2.13.0")]),
        _json_artifact([("torch", "2.13.0"), ("nvidia-cufile", "1.15.1.6")]),
    )
    exit_code = main(["--generated", str(g), "--committed", str(c)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "::notice::" in out
    assert "nvidia-cufile" in out


def test_main_deleted_row_fails_the_gate(tmp_path, capsys):
    # End-to-end guard for the deletion hole: a committed artifact that simply omits a
    # package the production install produced must FAIL, not pass as "in sync".
    g, c = _write_pair(
        tmp_path,
        _json_artifact([("torch", "2.13.0"), ("cryptography", "50.0.0")]),
        _json_artifact([("torch", "2.13.0")]),
    )
    exit_code = main(["--generated", str(g), "--committed", str(c)])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "::error::" in out
    assert "does not list cryptography==50.0.0" in out
    assert "under-reports" in out
    assert "in sync" not in out


def test_main_markdown_pair_works(tmp_path, capsys):
    g, c = _write_pair(
        tmp_path,
        _md_artifact([("torch", "2.13.0")]),
        _md_artifact([("torch", "2.12.0")]),
        "gen.md",
        "com.md",
    )
    assert main(["--generated", str(g), "--committed", str(c)]) == 1
    assert "torch==2.12.0" in capsys.readouterr().out


def test_main_cross_format_comparison_works(tmp_path):
    # sbom.json and DEPENDENCY_LICENSES.md are generated from the same run, so the
    # parser must give identical results regardless of format.
    g, c = _write_pair(
        tmp_path, _json_artifact([("torch", "2.13.0")]), _md_artifact([("torch", "2.13.0")]), "g.json", "c.md"
    )
    assert main(["--generated", str(g), "--committed", str(c)]) == 0


def test_main_unreadable_artifact_returns_two_not_one(tmp_path):
    # Exit 2 (tooling error) must be distinguishable from exit 1 (real drift).
    g = tmp_path / "gen.json"
    g.write_text(_json_artifact([("torch", "2.13.0")]), encoding="utf-8")
    assert main(["--generated", str(g), "--committed", str(tmp_path / "nope.json")]) == 2


def test_main_empty_generated_artifact_returns_two(tmp_path):
    # If SBOM generation silently produced nothing, that is a tooling failure, and it
    # must not pass as "no drift detected".
    g, c = _write_pair(tmp_path, "[]", _json_artifact([("torch", "2.13.0")]))
    assert main(["--generated", str(g), "--committed", str(c)]) == 2


def test_main_reports_comparison_count(tmp_path, capsys):
    g, c = _write_pair(
        tmp_path,
        _json_artifact([("torch", "2.13.0"), ("numpy", "2.4.6")]),
        _json_artifact([("torch", "2.13.0"), ("pandas", "3.0.3")]),
    )
    main(["--generated", str(g), "--committed", str(c)])
    assert "compared 1 package(s)" in capsys.readouterr().out
