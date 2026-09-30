"""Tests for the strict-typing ratchet gate (tools/check_strict_typing.py).

Most tests feed canned mypy output to the gate, so they run without mypy. The
tests marked by ``_needs_mypy`` run the real mypy and are skipped when it is not
importable, which is the case in CI's test job: that job installs
requirements.txt and four environments/requirements-ci*.txt files, none of which
pins mypy. The lint job is where the gate itself is meant to run.

Diagnostic lines are assembled from separate path and line arguments on
purpose: a literal path-colon-number string in this file would be read as a
real citation by scripts/check_doc_line_citations.py.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "tools" / "check_strict_typing.py"


def _load_module():
    """Import the checker by path because ``tools/`` is not a package."""
    spec = importlib.util.spec_from_file_location("check_strict_typing", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_module()

_needs_mypy = pytest.mark.skipif(
    importlib.util.find_spec("mypy") is None, reason="mypy is not importable here"
)


def _diag(path: str, line: int, severity: str = "error", message: str = "boom  [misc]") -> str:
    return f"{path}:{line}: {severity}: {message}"


def _summary_ok(n: int) -> str:
    return f"Success: no issues found in {n} source files"


def _summary_found(errors: int, files: int, checked: int) -> str:
    return f"Found {errors} errors in {files} files (checked {checked} source files)"


def _write_ratchet(root: Path, files: list[str]) -> Path:
    path = root / "ratchet.json"
    path.write_text(json.dumps({"files": files}), encoding="utf-8")
    return path


def _touch(root: Path, *relative: str) -> None:
    for rel in relative:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")


# --- ratchet file validation -------------------------------------------------


def test_load_ratchet_accepts_a_sorted_unique_list(tmp_path):
    path = _write_ratchet(tmp_path, ["pkg/a.py", "pkg/b.py"])
    assert gate.load_ratchet(path) == ["pkg/a.py", "pkg/b.py"]


@pytest.mark.parametrize(
    ("files", "needle"),
    [
        (["pkg/b.py", "pkg/a.py"], "sorted"),
        (["pkg/a.py", "pkg/a.py"], "more than once"),
        (["pkg/a.txt"], "not a .py path"),
        (["/abs/a.py"], "relative posix"),
        (["pkg/../a.py"], "relative posix"),
        (["pkg\\a.py"], "relative posix"),
    ],
)
def test_load_ratchet_rejects_malformed_lists(tmp_path, files, needle):
    path = _write_ratchet(tmp_path, files)
    with pytest.raises(gate.RatchetError, match=needle):
        gate.load_ratchet(path)


@pytest.mark.parametrize(
    "payload", ['["pkg/a.py"]', '{"files": "pkg/a.py"}', '{"files": [1]}', "{"]
)
def test_load_ratchet_rejects_the_wrong_shape(tmp_path, payload):
    path = tmp_path / "ratchet.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(gate.RatchetError):
        gate.load_ratchet(path)


# --- output parsing ----------------------------------------------------------


def test_parse_output_counts_errors_per_file_and_ignores_notes():
    output = "\n".join(
        [
            _diag("pkg/a.py", 3),
            _diag("pkg/a.py", 9, message="missing annotation  [no-untyped-def]"),
            _diag("pkg/a.py", 9, severity="note", message="use --help"),
            _diag("pkg\\sub\\b.py", 1),
            _summary_found(3, 2, 4),
        ]
    )
    errors, checked = gate.parse_output(output)
    assert errors == {"pkg/a.py": 2, "pkg/sub/b.py": 1}
    assert checked == 4


def test_parse_output_reads_the_success_summary():
    errors, checked = gate.parse_output(_summary_ok(7))
    assert not errors
    assert checked == 7


# --- the verdict -------------------------------------------------------------


def test_evaluate_passes_a_clean_run_over_every_listed_file():
    verdict = gate.evaluate(["pkg/a.py", "pkg/b.py"], gate.MypyResult(0, _summary_ok(2)))
    assert verdict.ok, verdict.messages


def test_evaluate_fails_on_an_error_in_a_listed_file():
    output = "\n".join([_diag("pkg/b.py", 12), _summary_found(1, 1, 2)])
    verdict = gate.evaluate(["pkg/a.py", "pkg/b.py"], gate.MypyResult(1, output))
    assert not verdict.ok
    assert verdict.errors_by_file == {"pkg/b.py": 1}
    assert any("REGRESSED: pkg/b.py" in m for m in verdict.messages)


def test_evaluate_fails_closed_when_the_checked_count_disagrees():
    verdict = gate.evaluate(["pkg/a.py", "pkg/b.py"], gate.MypyResult(0, _summary_ok(1)))
    assert not verdict.ok
    assert any("checked 1 source files but 2" in m for m in verdict.messages)


def test_evaluate_fails_closed_without_a_summary_line():
    verdict = gate.evaluate(["pkg/a.py"], gate.MypyResult(0, ""))
    assert not verdict.ok


def test_evaluate_fails_closed_on_an_unparsable_failure():
    verdict = gate.evaluate(
        ["pkg/a.py"], gate.MypyResult(1, "something went wrong\n" + _summary_ok(1))
    )
    assert not verdict.ok
    assert any("no error line" in m for m in verdict.messages)


@pytest.mark.parametrize("returncode", [2, 3, -11])
def test_evaluate_fails_closed_on_a_crash_status(returncode):
    verdict = gate.evaluate(["pkg/a.py"], gate.MypyResult(returncode, _summary_ok(1)))
    assert not verdict.ok


# --- check() end to end with a canned mypy -----------------------------------


def _canned(result):
    def fake(files, repo_root, python_executable=None):
        fake.calls.append(list(files))
        return result

    fake.calls = []
    return fake


def test_check_fails_on_a_stale_entry_without_running_mypy(tmp_path, monkeypatch, capsys):
    _touch(tmp_path, "pkg/a.py")
    ratchet = _write_ratchet(tmp_path, ["pkg/a.py", "pkg/gone.py"])
    fake = _canned(gate.MypyResult(0, _summary_ok(2)))
    monkeypatch.setattr(gate, "run_mypy", fake)
    assert gate.check(ratchet, tmp_path, None) == 1
    assert "stale entry): pkg/gone.py" in capsys.readouterr().out
    assert fake.calls == []


def test_check_fails_on_an_empty_ratchet(tmp_path, monkeypatch):
    ratchet = _write_ratchet(tmp_path, [])
    monkeypatch.setattr(gate, "run_mypy", _canned(gate.MypyResult(0, _summary_ok(0))))
    assert gate.check(ratchet, tmp_path, None) == 1


def test_check_passes_exactly_the_listed_files_to_mypy(tmp_path, monkeypatch):
    _touch(tmp_path, "pkg/a.py", "pkg/b.py", "pkg/unlisted.py")
    ratchet = _write_ratchet(tmp_path, ["pkg/a.py", "pkg/b.py"])
    fake = _canned(gate.MypyResult(0, _summary_ok(2)))
    monkeypatch.setattr(gate, "run_mypy", fake)
    assert gate.check(ratchet, tmp_path, None) == 0
    assert fake.calls == [["pkg/a.py", "pkg/b.py"]]


def test_check_reports_a_regression(tmp_path, monkeypatch, capsys):
    _touch(tmp_path, "pkg/a.py")
    ratchet = _write_ratchet(tmp_path, ["pkg/a.py"])
    output = "\n".join([_diag("pkg/a.py", 5), _summary_found(1, 1, 1)])
    monkeypatch.setattr(gate, "run_mypy", _canned(gate.MypyResult(1, output)))
    assert gate.check(ratchet, tmp_path, None) == 1
    assert "REGRESSED: pkg/a.py has 1 strict error(s)" in capsys.readouterr().out


def test_main_exits_2_when_mypy_is_not_importable(monkeypatch, capsys):
    monkeypatch.setattr(gate, "mypy_available", lambda: False)
    assert gate.main(["--check"]) == 2
    assert "mypy is not importable" in capsys.readouterr().out


def test_the_shipped_ratchet_file_is_well_formed_and_current():
    """Runs without mypy: the list parses, and every entry names a real file."""
    listed = gate.load_ratchet(gate.RATCHET)
    assert listed, "an empty ratchet certifies nothing"
    missing = [f for f in listed if not (_REPO_ROOT / f).is_file()]
    assert not missing, f"stale ratchet entries: {missing}"


# --- the real mypy -----------------------------------------------------------


_TYPED = "def add(a: int, b: int) -> int:\n    return a + b\n"
_UNTYPED = "def add(a: int, b) -> int:\n    return a + b\n"


@_needs_mypy
def test_real_mypy_fails_a_regression_and_passes_its_restore(tmp_path):
    """The discriminator in miniature: drop one annotation, then put it back."""
    _touch(tmp_path, "pkg/__init__.py")
    module = tmp_path / "pkg" / "calc.py"
    module.write_text(_TYPED, encoding="utf-8")
    ratchet = _write_ratchet(tmp_path, ["pkg/__init__.py", "pkg/calc.py"])

    assert gate.check(ratchet, tmp_path, None) == 0
    module.write_text(_UNTYPED, encoding="utf-8")
    assert gate.check(ratchet, tmp_path, None) == 1
    module.write_text(_TYPED, encoding="utf-8")
    assert gate.check(ratchet, tmp_path, None) == 0


@_needs_mypy
def test_real_mypy_ignores_errors_in_an_unlisted_import(tmp_path):
    """An unlisted module's own strict errors do not fail the gate."""
    _touch(tmp_path, "pkg/__init__.py")
    (tmp_path / "pkg" / "messy.py").write_text("def helper(x):\n    return x\n", encoding="utf-8")
    (tmp_path / "pkg" / "clean.py").write_text(
        "from pkg.messy import helper as _helper\n\n\ndef twice(n: int) -> int:\n    return n * 2\n",
        encoding="utf-8",
    )
    ratchet = _write_ratchet(tmp_path, ["pkg/__init__.py", "pkg/clean.py"])
    assert gate.check(ratchet, tmp_path, None) == 0


@_needs_mypy
def test_the_live_repository_passes_its_own_gate():
    """Live tree, not a fixture: the shipped list must be strict-clean right now."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--check"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
