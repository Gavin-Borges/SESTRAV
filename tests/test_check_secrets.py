"""Regression tests for scripts/check_secrets.py.

The CI secret-pattern job (.github/workflows/security.yml) delegates here, and it
is the ONLY credential-content gate that runs in CI. pre-commit Gate 2 does NOT
delegate here: it carries its own CRED_PATTERNS array and names this file only in
a false-positive help string, so a false negative here is not covered by it, and
the shape patterns it holds do not run on a fresh clone at all.

Entropy used to run only on lines matching `keyword\\s*=\\s*[\"']`, so YAML/JSON
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


# --- False-negative regressions -------------------------------------------------
#
# Each of the four below was a measured BLOCK-to-allow regression in an earlier
# revision of this scanner, and the suite as it stood could not see any of them:
# three separate mutations of the scanner left all seven original tests green.
# A false negative here is the severe direction, because this is the only
# credential-content gate CI runs.


def test_second_assignment_on_a_line_is_not_shielded_by_the_first(
    tmp_path: Path,
) -> None:
    """A short decoy value must not hide a real secret later on the same line.

    Pins `finditer` over `search`. With `search` the scanner inspects only the
    FIRST match, so prefixing any line with `token = "abc";` disarmed it.
    """
    mod = _load()
    target = tmp_path / "case.py"
    decoy = "to" + "ken" + ' = "abc"; '
    target.write_text(
        decoy + "pass" + "word" + ' = "' + _token() + '"\n', encoding="utf-8"
    )
    assert mod.scan_file(str(target)) == [1]


def test_run_together_credential_name_is_flagged(tmp_path: Path) -> None:
    """camelCase and run-together names must still match.

    Pins the ABSENCE of a left anchor on the keyword. A `(?:^|[^a-z0-9])` prefix
    silently dropped accessToken, sessionToken, mytoken, authtoken, apitoken,
    userpassword, dbpassword and clientsecret, all of which the scanner caught
    before it was added.
    """
    mod = _load()
    for name in ("access" + "Token", "db" + "password", "client" + "secret"):
        target = tmp_path / (name + ".py")
        _write(target, name, " = ", _token())
        assert mod.scan_file(str(target)) == [1], name


def test_secret_beside_a_hash_marker_is_still_flagged(tmp_path: Path) -> None:
    """A digest elsewhere on the line must not suppress the whole line.

    Pins the absence of a whole-line skip. Skipping any line containing
    `--hash=` or `sha256:` made the gate bypassable with one appended comment,
    and it suppressed 5,625 lines across the repo while flagging none of them.
    """
    mod = _load()
    target = tmp_path / "case.py"
    target.write_text(
        "pass" + "word" + ' = "' + _token() + '"  # sha256:deadbeefcafe\n',
        encoding="utf-8",
    )
    assert mod.scan_file(str(target)) == [1]


def test_value_of_nine_characters_is_flagged(tmp_path: Path) -> None:
    """Pins both thresholds from the flagging side.

    Nine distinct characters give entropy log2(9) = 3.17, just over the 3.0 floor,
    and length 9, just over the 8 floor. Raising either threshold breaks this,
    which the single 18-character fixture used elsewhere in this module does not
    detect: it clears length by +10 and entropy by +1.17.
    """
    mod = _load()
    target = tmp_path / "case.py"
    _write(target, "api_" + "key", " = ", "a8f3k9d2m")
    assert mod.scan_file(str(target)) == [1]


# --- False-positive direction ---------------------------------------------------


def test_prose_value_with_spaces_is_not_flagged(tmp_path: Path) -> None:
    """Discriminate on the VALUE, not on the line.

    Colon-assignment matching made ordinary documentation sentences match. A
    credential value never contains whitespace, so the guard costs no true
    positive; it was measured to kill both new false positives and lose none.
    """
    mod = _load()
    target = tmp_path / "doc.md"
    target.write_text(
        "A pass" + "word: " + '"must be at least twelve characters long" per policy.\n',
        encoding="utf-8",
    )
    assert mod.scan_file(str(target)) == []


def test_long_low_entropy_value_is_not_flagged(tmp_path: Path) -> None:
    """Length alone must not flag; the entropy floor has to carry its weight."""
    mod = _load()
    target = tmp_path / "case.py"
    _write(target, "pass" + "word", " = ", "a" * 22)
    assert mod.scan_file(str(target)) == []


# ---------------------------------------------------------------------------
# EXCLUDE_DIRS prunes by directory NAME, which silently hid TRACKED files
# ---------------------------------------------------------------------------


def _git(repo: Path, *argv: str) -> None:
    import subprocess

    subprocess.run(["git", "-C", str(repo), *argv], check=True, capture_output=True)


def _repo_with(tmp_path: Path, relpath: str, *, track: bool) -> Path:
    """A throwaway git repo holding one credential-bearing file at relpath."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    target = repo / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    _write(target, "api_" + "key", " = ", _token())
    if track:
        _git(repo, "add", "-f", relpath)
    return repo


def _basenames(mod, repo: Path) -> set:
    import os as _os

    return {_os.path.basename(p) for p in mod.iter_scanned_files(str(repo))}


def test_tracked_file_under_an_excluded_dir_is_scanned(tmp_path: Path) -> None:
    """The defect: 'results' is in EXCLUDE_DIRS and pruning is by directory NAME,
    so os.walk never opened 26 tracked files. A tracked file is published
    content, which is precisely what this gate exists to stop being published.

    min_files=0 is deliberate. With the default floor this test would pass even
    with the fix reverted, because scan_tree also returns 1 when it refuses a
    vacuous pass over too few files - a mutation confirmed exactly that. The
    floor is set out of the way so the 1 can only mean "credential found", and
    the membership assertion pins that the file was OPENED rather than merely
    that something somewhere failed.
    """
    mod = _load()
    repo = _repo_with(tmp_path, "results/leak.md", track=True)
    assert "leak.md" in _basenames(mod, repo)
    assert mod.scan_tree(str(repo), min_files=0) == 1


def test_untracked_file_under_an_excluded_dir_is_still_skipped(tmp_path: Path) -> None:
    """The safety net must stay ADDITIVE. Untracked material under an excluded
    name - .venv, __pycache__, _local, the gitignored assistant trees - is still
    pruned, so neither the walk's cost nor its intent changes."""
    mod = _load()
    repo = _repo_with(tmp_path, "results/leak.md", track=False)
    assert "leak.md" not in _basenames(mod, repo)
    assert mod.scan_tree(str(repo), min_files=0) == 0


def test_tracked_file_outside_an_excluded_dir_is_unaffected(tmp_path: Path) -> None:
    mod = _load()
    repo = _repo_with(tmp_path, "docs/leak.md", track=True)
    assert "leak.md" in _basenames(mod, repo)
    assert mod.scan_tree(str(repo), min_files=0) == 1


def test_tracked_paths_returns_empty_outside_a_work_tree(tmp_path: Path) -> None:
    """Failure must degrade to the old behaviour, not to an exception: a
    non-git checkout scans exactly what the walk found."""
    mod = _load()
    plain = tmp_path / "plain"
    plain.mkdir()
    assert mod._tracked_paths(str(plain)) == []


def test_scanned_list_has_no_duplicates_at_the_repo_root() -> None:
    """A tracked file that the walk already found must not be scanned twice."""
    mod = _load()
    root = str(Path(__file__).resolve().parents[1])
    paths = mod.iter_scanned_files(root)
    import os as _os

    keys = [_os.path.normcase(_os.path.abspath(p)) for p in paths]
    assert len(keys) == len(set(keys))


def test_the_net_respects_the_same_suffix_filter_as_the_walk(tmp_path: Path) -> None:
    """The net must not widen WHAT is scanned, only WHERE it is looked for.

    Without this, dropping the filter would pull every tracked .csv and binary
    under an excluded name into the scan: more work, and entropy false positives
    on data files. .csv is deliberately absent from _SCAN_SUFFIXES.
    """
    mod = _load()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "results").mkdir()
    (repo / "results" / "data.csv").write_text("peptide,label\nAAAA,1\n", encoding="utf-8")
    (repo / "results" / "note.md").write_text("plain prose, no credential\n", encoding="utf-8")
    _git(repo, "add", "-f", "results/data.csv", "results/note.md")

    names = _basenames(mod, repo)
    assert "note.md" in names
    assert "data.csv" not in names
