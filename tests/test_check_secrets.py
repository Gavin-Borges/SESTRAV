"""Regression tests for scripts/check_secrets.py.

The CI secret-pattern job and pre-commit Gate 2 both delegate here. Entropy
used to run only on lines matching `keyword\\s*=\\s*[\"']`, so YAML/JSON
colon assignment and `AWS_SECRET_ACCESS_KEY = \"...\"` never reached it.
A walk from the wrong cwd could also print SUCCESS over zero files.

Payloads are assembled at runtime so this test module itself does not contain
a credential-keyword assignment that the repo-wide scan would flag.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_secrets.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_secrets", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _token() -> str:
    # Mixed alphabet, length > 8, entropy > 3.0. Not a real credential.
    return "a8f3k9d2m1q7x4z0b5"


def _write(path: Path, left: str, op: str, token: str) -> None:
    path.write_text(left + op + '"' + token + '"\n', encoding="utf-8")


def test_python_equals_assignment_still_flagged(tmp_path: Path) -> None:
    mod = _load()
    target = tmp_path / "case.py"
    _write(target, "api_" + "key", " = ", _token())
    assert mod.scan_file(str(target)) == [1]


def test_yaml_colon_assignment_is_flagged(tmp_path: Path) -> None:
    mod = _load()
    target = tmp_path / "case.yaml"
    _write(target, "api_" + "key", ": ", _token())
    assert mod.scan_file(str(target)) == [1]


def test_json_colon_assignment_is_flagged(tmp_path: Path) -> None:
    mod = _load()
    target = tmp_path / "case.json"
    _write(target, '"' + "sec" + "ret" + '"', ": ", _token())
    assert mod.scan_file(str(target)) == [1]


def test_keyword_with_intervening_identifier_is_flagged(tmp_path: Path) -> None:
    mod = _load()
    target = tmp_path / "case.py"
    _write(target, "AWS_" + "SECRET" + "_ACCESS_KEY", " = ", _token())
    assert mod.scan_file(str(target)) == [1]


def test_hash_pin_line_is_not_flagged(tmp_path: Path) -> None:
    mod = _load()
    target = tmp_path / "req.txt"
    target.write_text(
        "foo==1.0.0 --hash=sha256:" + _token() + "abcdef\n", encoding="utf-8"
    )
    assert mod.scan_file(str(target)) == []


def test_empty_tree_is_not_a_pass(tmp_path: Path) -> None:
    mod = _load()
    assert mod.scan_tree(str(tmp_path), min_files=10) == 1


def test_repo_root_clears_the_file_count_floor() -> None:
    mod = _load()
    paths = mod.iter_scanned_files(str(Path(__file__).resolve().parents[1]))
    assert len(paths) >= mod.MIN_SCANNED_FILES
