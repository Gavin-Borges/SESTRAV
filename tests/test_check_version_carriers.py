"""Tests for tools/check_version_carriers.py, the current-version carrier gate.

The live-repository tests run the gate against this checkout, so CI's pytest job
executes the gate itself on every pull request. The fixture tests pin its behaviour:
a carrier that drops out of the census, or carries a suffixed version, must fail.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from tools import check_version_carriers as gate

ROOT = Path(__file__).resolve().parents[1]

# The current census, one carrier per kind. A carrier that vanishes or moves, or a new
# one that appears, changes this list on purpose: update it in the same change.
LIVE_CARRIERS = [
    ("CITATION.cff", "citation metadata version"),
    ("README.md", "BibTeX version"),
    ("README.md", "version badge"),
    ("USAGE.md", "CLI version output"),
    ("api/main.py", "source-run API fallback"),
    ("docs/model_cards/rf_31feature_integrated.md", "model-card version field"),
    ("pyproject.toml", "canonical project version"),
]

# Carrier-shaped fixture text holds a {v} placeholder and receives its version at run
# time. This file is tracked, so a literal carrier written here would be found by the
# gate it tests: the live census would gain this file and the live gate would fail.
PYPROJECT = '[project]\nname = "example"\nversion = "{v}"\n'
CFF = "cff-version: 1.2.0\nversion: {v}\n"
BADGE = "![Version](https://img.shields.io/badge/version-{v}-informational)\n"
BIBTEX = "@software{example,\n  version   = {{v}}\n}\n"
CLI = "  sestrav version : {v}\n"
API = 'try:\n    pass\nexcept ImportError:\n    _APP_VERSION = "{v}"\n'
CARD = "- **Version:** SESTRAV v{v} - current model.\n"


def _v(template: str, version: str) -> str:
    return template.replace("{v}", version)


# One carrier of every required kind, all agreeing on 3.4.5.
FULL_TREE = {
    "pyproject.toml": _v(PYPROJECT, "3.4.5"),
    "CITATION.cff": _v(CFF, "3.4.5"),
    "README.md": _v(BADGE + BIBTEX, "3.4.5"),
    "USAGE.md": _v(CLI, "3.4.5"),
    "api/main.py": _v(API, "3.4.5"),
    "docs/card.md": _v(CARD, "3.4.5"),
}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _full_tree(root: Path, overrides: dict[str, str | bytes] | None = None) -> list[str]:
    """Write FULL_TREE plus overrides byte for byte, and return the tracked list."""
    files: dict[str, str | bytes] = {**FULL_TREE, **(overrides or {})}
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return sorted(files)


def _run(root: Path, tracked: list[str]) -> tuple[int, str]:
    output = io.StringIO()
    code = gate.run(root, tracked_files=tracked, stream=output)
    return code, output.getvalue()


def _inside_git_work_tree() -> bool:
    """True only if `git ls-files` here can report what is tracked.

    In a `git archive` export or a tarball there is no index, so the gate cannot
    enumerate tracked files; the live tests say so instead of failing for a reason
    unrelated to the carriers. CI's test job runs in an actions/checkout work tree.
    """
    out = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return out.returncode == 0 and out.stdout.strip() == "true"


live = pytest.mark.skipif(
    not _inside_git_work_tree(),
    reason="not inside a git work tree, so `git ls-files` cannot report what is tracked",
)


def test_discovers_carriers_by_format_not_path(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", _v(PYPROJECT, "3.4.5"))
    _write(tmp_path / "moved" / "anything.md", _v(BADGE + CLI, "3.4.5"))
    _write(tmp_path / "metadata.cff", _v(CFF, "3.4.5"))

    carriers = gate.enumerate_carriers(
        tmp_path,
        ["pyproject.toml", "moved/anything.md", "metadata.cff"],
    )

    assert len(carriers) == 4
    assert {carrier.kind for carrier in carriers} == {
        "canonical project version",
        "version badge",
        "CLI version output",
        "citation metadata version",
    }


def test_historical_mentions_are_not_current_carriers(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nversion = "3.4.5"\n')
    _write(
        tmp_path / "history.md",
        "Version v3.4.5 was signed.\n## [3.4.5] - 2026-01-01\n",
    )

    carriers = gate.enumerate_carriers(tmp_path, ["pyproject.toml", "history.md"])

    assert [(carrier.kind, carrier.version) for carrier in carriers] == [
        ("canonical project version", "3.4.5")
    ]


def test_divergence_fails_and_names_the_file(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nversion = "3.4.5"\n')
    _write(tmp_path / "docs" / "readme.md", _v(BADGE, "9.9.9"))
    output = io.StringIO()

    code = gate.run(
        tmp_path,
        tracked_files=["pyproject.toml", "docs/readme.md"],
        stream=output,
        required_kinds=(),
    )

    assert code == 1
    # Two fragments on purpose: a contiguous path, colon and line number in this tracked
    # file is read as a real line citation by the doc-line-citation gate.
    assert "docs/readme.md" + ":1 carries 9.9.9" in output.getvalue()
    assert "1 of 2 version carriers disagree" in output.getvalue()


def test_a_complete_agreeing_tree_passes(tmp_path: Path) -> None:
    # The control for the fail-closed cases below: the same fixture, unbroken, passes.
    tracked = _full_tree(tmp_path)

    code, output = _run(tmp_path, tracked)

    assert code == 0, output
    assert "OK: 7 version carriers agree on 3.4.5" in output


@pytest.mark.parametrize(
    ("overrides", "missing_on_disk", "kind"),
    [
        # A citation file that is not UTF-8 is unreadable, so its 9.9.9 goes unseen.
        (
            {"CITATION.cff": _v(CFF, "9.9.9").encode("utf-8") + b"author: \xe9\n"},
            None,
            "citation metadata version",
        ),
        # A tracked carrier deleted from the working tree.
        ({"USAGE.md": _v(CLI, "9.9.9")}, "USAGE.md", "CLI version output"),
        # A carrier reformatted so that its pattern no longer matches it.
        (
            {"docs/card.md": _v(CARD, "9.9.9").replace("Version:**", "Version**:")},
            None,
            "model-card version field",
        ),
    ],
    ids=["non-utf8-file", "missing-from-disk", "pattern-matches-nothing"],
)
def test_a_carrier_that_drops_out_fails_closed(
    tmp_path: Path, overrides: dict[str, str | bytes], missing_on_disk: str | None, kind: str
) -> None:
    tracked = _full_tree(tmp_path, overrides)
    if missing_on_disk is not None:
        (tmp_path / missing_on_disk).unlink()

    code, output = _run(tmp_path, tracked)

    assert code == 1, output
    assert f"FAIL: no {kind} carrier was found" in output


@pytest.mark.parametrize(
    ("overrides", "value"),
    [
        ({"USAGE.md": _v(CLI, "3.4.5rc1")}, "3.4.5rc1"),
        ({"USAGE.md": _v(CLI, "3.4.5.1")}, "3.4.5.1"),
        ({"USAGE.md": _v(CLI, "3.4.50")}, "3.4.50"),
        ({"docs/card.md": _v(CARD, "3.4.5-dev")}, "3.4.5-dev"),
        ({"CITATION.cff": _v(CFF, "3.4.5rc1")}, "3.4.5rc1"),
    ],
    ids=["cli-rc", "cli-fourth-part", "cli-longer-patch", "model-card-dev", "citation-rc"],
)
def test_a_suffixed_version_is_not_read_as_its_prefix(
    tmp_path: Path, overrides: dict[str, str | bytes], value: str
) -> None:
    tracked = _full_tree(tmp_path, overrides)

    code, output = _run(tmp_path, tracked)

    assert code == 1, output
    assert f"carries {value}; pyproject.toml carries 3.4.5" in output


def test_a_quoted_citation_version_is_read(tmp_path: Path) -> None:
    # The release workflow accepts a quoted CITATION.cff version, so the gate must too.
    # CRLF on purpose: a Windows checkout with core.autocrlf writes the file that way.
    quoted = _v(CFF, '"3.4.5"').replace("\n", "\r\n")
    tracked = _full_tree(tmp_path, {"CITATION.cff": quoted})

    carriers = gate.enumerate_carriers(tmp_path, tracked)

    assert ("CITATION.cff", 2, "citation metadata version", "3.4.5") in [
        (c.path, c.line, c.kind, c.version) for c in carriers
    ]


def test_the_reported_line_is_the_line_that_carries_the_version(tmp_path: Path) -> None:
    # A leading \s* in a multiline pattern can start the match on a blank line above.
    shifted = "x = 1\n\n\n" + _v(API, "3.4.5").splitlines(keepends=True)[-1]
    tracked = _full_tree(tmp_path, {"api/main.py": shifted})

    carriers = gate.enumerate_carriers(tmp_path, tracked)

    assert [c.line for c in carriers if c.path == "api/main.py"] == [4]


def test_this_unit_carries_no_carrier_of_its_own() -> None:
    # Both files are tracked, so the live census scans them. Checked by explicit path
    # so that it holds before the files are committed: an earlier draft of this test
    # file passed while untracked and failed the live gate once committed.
    own = ["tests/test_check_version_carriers.py", "tools/check_version_carriers.py"]

    carriers = gate.enumerate_carriers(ROOT, own)

    assert [(c.path, c.kind) for c in carriers] == [("pyproject.toml", "canonical project version")]


@live
def test_the_live_repository_passes_the_gate() -> None:
    output = io.StringIO()

    code = gate.run(ROOT, stream=output)
    carriers = gate.enumerate_carriers(ROOT)

    assert code == 0, output.getvalue()
    assert {carrier.version for carrier in carriers} == {gate.canonical_version(ROOT)}
    assert sorted((carrier.path, carrier.kind) for carrier in carriers) == LIVE_CARRIERS


@live
def test_the_command_line_entry_point_passes_on_the_live_repository() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_version_carriers.py"), "--root", str(ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    expected = f"OK: {len(LIVE_CARRIERS)} version carriers agree on {gate.canonical_version(ROOT)}"
    assert expected in result.stdout
