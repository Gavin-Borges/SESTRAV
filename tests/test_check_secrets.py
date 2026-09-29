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


def _repo_with(tmp_path: Path, relpath: str, *, track: bool, ignore: bool = False) -> Path:
    """A throwaway git repo holding one credential-bearing file at relpath.

    `ignore=True` writes a .gitignore matching relpath. Combined with
    track=True that produces the force-added case (`git add -f`), which must
    still be scanned.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    if ignore:
        (repo / ".gitignore").write_text(relpath + "\n", encoding="utf-8")
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


def test_gitignored_untracked_file_at_the_repo_root_is_skipped(tmp_path: Path) -> None:
    """The defect: EXCLUDE_DIRS prunes the walk by directory NAME, so a
    gitignored file at the REPO ROOT has no directory to prune and was opened
    anyway. The live case was STATE.md, which is gitignored and absent from
    HEAD: it turned this gate red locally on prose describing the scanner's own
    patterns, while CI stayed green because CI never sees the file. A gate that
    is red for a reason CI cannot reproduce is one people learn to skip.

    min_files=0 for the reason the excluded-dir tests give: under the default
    floor a 1 could mean "too few files scanned" rather than "credential
    found", so the floor is moved out of the way and the membership assertion
    pins that the file was never opened.
    """
    mod = _load()
    repo = _repo_with(tmp_path, "leak.md", track=False, ignore=True)
    assert "leak.md" not in _basenames(mod, repo)
    assert mod.scan_tree(str(repo), min_files=0) == 0


def test_gitignored_but_force_added_file_is_still_scanned(tmp_path: Path) -> None:
    """`git add -f` must not become a way past this gate.

    Two independent layers hold this, measured rather than assumed. The one
    that actually operates is `git check-ignore`: without --no-index it does
    not report a TRACKED file as ignored (exit 1, empty output), so the filter
    never subtracts a force-added file. The tracked-file union in
    iter_scanned_files is a real but redundant second net, confirmed by forcing
    --no-index on and watching this test still pass; it fails only when both
    layers are removed.
    """
    mod = _load()
    repo = _repo_with(tmp_path, "leak.md", track=True, ignore=True)
    assert "leak.md" in _basenames(mod, repo)
    assert mod.scan_tree(str(repo), min_files=0) == 1


def test_ignored_paths_returns_empty_outside_a_work_tree(tmp_path: Path) -> None:
    """Failure must degrade to the old behaviour, not to an exception: with no
    git answer available the walk scans exactly what it found, as before."""
    mod = _load()
    plain = tmp_path / "plain"
    plain.mkdir()
    probe = plain / "note.md"
    probe.write_text("nothing here\n", encoding="utf-8")
    assert mod._ignored_paths(str(plain), [str(probe)]) == set()


def test_ignored_paths_is_empty_for_an_empty_candidate_list(tmp_path: Path) -> None:
    """No candidates means no subprocess and no subtraction."""
    mod = _load()
    assert mod._ignored_paths(str(tmp_path), []) == set()


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
# --- Unquoted values, scoped by file format -------------------------------------
#
# The quoted-only value pattern missed the everyday leak shape entirely:
# `AWS_SECRET_ACCESS_KEY=<value>` in a .env, and `export API_KEY=<value>` in a
# shell script, both went BLOCK-to-allow because neither value is quoted.
#
# The scoping is not a heuristic, it is a language fact, and these tests pin BOTH
# directions of it. In Python, JSON and TOML a string literal is always quoted, so
# an unquoted right-hand side there is an EXPRESSION and cannot be a hardcoded
# credential. In YAML, shell, .env, Dockerfile and prose an unquoted scalar IS the
# literal. Allowing bare values everywhere was measured to flag exactly one line,
# `token = match.group(1)` in check_doc_commit_refs.py's SHA_RE loop - a real
# CI failure on a real false positive, which is why the .py direction is tested.


def _bare(path: Path, text: str) -> None:
    path.write_text(text + "\n", encoding="utf-8")


def test_unquoted_value_in_shell_script_is_flagged(tmp_path: Path) -> None:
    """The `export API_KEY=<value>` shape, unquoted, in a shell script."""
    mod = _load()
    target = tmp_path / "deploy.sh"
    _bare(target, "export API_" + "KEY=" + _token())
    assert mod.scan_file(str(target)) == [1]


def test_unquoted_value_in_dotenv_is_flagged(tmp_path: Path) -> None:
    """The `.env` shape: a bare AWS_SECRET_ACCESS_KEY assignment."""
    mod = _load()
    target = tmp_path / ".env"
    _bare(target, "AWS_" + "SECRET" + "_ACCESS_KEY=" + _token())
    assert mod.scan_file(str(target)) == [1]


def test_unquoted_value_in_yaml_is_flagged(tmp_path: Path) -> None:
    """YAML plain scalars are literals, so a bare colon value must be scanned."""
    mod = _load()
    target = tmp_path / "workflow.yml"
    _bare(target, "  api_" + "key: " + _token())
    assert mod.scan_file(str(target)) == [1]


def test_unquoted_value_in_an_extensionless_hook_is_flagged(tmp_path: Path) -> None:
    """Files under scripts/hooks/ are shell with no suffix, so a bare value is literal.

    The selection layer opens the directory's extensionless hooks; without this they
    were scanned for quoted assignments only, and `API_TOKEN=<value>` passed.
    """
    mod = _load()
    hooks = tmp_path / "scripts" / "hooks"
    hooks.mkdir(parents=True)
    target = hooks / "pre-push"
    _bare(target, "API_" + "TOKEN=" + _token())
    assert mod.scan_file(str(target)) == [1]


def test_unquoted_value_in_a_makefile_is_flagged(tmp_path: Path) -> None:
    """Every Make assignment operator, including `:=`, which `[=:]` alone missed."""
    mod = _load()
    for operator in (" = ", " := ", " ?= ", " += "):
        target = tmp_path / "Makefile"
        _bare(target, "API_" + "TOKEN" + operator + _token())
        assert mod.scan_file(str(target)) == [1], operator


def test_python_helper_under_the_hook_dir_keeps_python_rules(tmp_path: Path) -> None:
    """Only EXTENSIONLESS hook files are shell; a .py there keeps the .py exemption."""
    mod = _load()
    hooks = tmp_path / "scripts" / "hooks"
    hooks.mkdir(parents=True)
    target = hooks / "helper.py"
    _bare(target, "to" + "ken = match.group(1)")
    assert mod.scan_file(str(target)) == []


def test_makefile_rule_line_is_not_read_as_an_assignment(tmp_path: Path) -> None:
    """In a Makefile, `target: prerequisites` is a rule; a lone `:` is not assignment."""
    mod = _load()
    target = tmp_path / "Makefile"
    _bare(target, "au" + "th-check: scripts/check_affiliation_claims.py")
    assert mod.scan_file(str(target)) == []


def test_extensionless_file_outside_the_hook_dir_stays_quoted_only(tmp_path: Path) -> None:
    """The widening is scoped: an extensionless file elsewhere keeps quoted-only mode."""
    mod = _load()
    target = tmp_path / "CODEOWNERS"
    _bare(target, "API_" + "TOKEN=" + _token())
    assert mod.scan_file(str(target)) == []


def test_unquoted_value_in_dockerfile_is_flagged(tmp_path: Path) -> None:
    """Dockerfile ENV/ARG values are unquoted literals; match on the basename."""
    mod = _load()
    target = tmp_path / "Dockerfile.api"
    _bare(target, "ENV API_" + "KEY=" + _token())
    assert mod.scan_file(str(target)) == [1]


def test_quoted_value_in_shell_script_is_still_flagged(tmp_path: Path) -> None:
    """Adding the bare branch must not cost the quoted case in the same format."""
    mod = _load()
    target = tmp_path / "deploy.sh"
    _write(target, "export API_" + "KEY", "=", _token())
    assert mod.scan_file(str(target)) == [1]


def test_python_expression_rhs_is_not_flagged_but_same_text_in_shell_is(
    tmp_path: Path,
) -> None:
    """The false-positive guard, and the proof that it is not vacuous.

    `token = match.group(1)` is real tracked code in
    check_doc_commit_refs.py's SHA_RE loop. Under a bare-value branch applied to
    every format it captures a 13-character value with entropy 3.7, which clears
    both the length and the entropy floor and turns the CI gate red.

    The second assertion is what keeps this test honest: the SAME text in a .sh
    file IS flagged, so the .py assertion can only pass because of the format
    scoping, never because the fixture failed to clear a threshold.
    """
    mod = _load()
    line = "to" + "ken = match.group(1)"
    py_case = tmp_path / "case.py"
    sh_case = tmp_path / "case.sh"
    _bare(py_case, line)
    _bare(sh_case, line)
    assert mod.scan_file(str(py_case)) == []
    assert mod.scan_file(str(sh_case)) == [1]


def test_bare_pattern_does_not_shadow_a_quoted_secret_on_the_same_line(
    tmp_path: Path,
) -> None:
    """The bare pattern must be ADDITIVE, never a replacement.

    A bare match stops at the first quote, so it can swallow the keyword that a
    later quoted match needed. Here the bare pattern alone consumes `auth=aatoken=`
    and captures an 8-character value that clears no floor, leaving the quoted
    secret behind the resume point of finditer - BLOCK-to-allow, the same shape as
    the search()-vs-finditer regression above. Running BOTH patterns on a
    bare-eligible format is what keeps this line caught.
    """
    mod = _load()
    target = tmp_path / "deploy.sh"
    _bare(target, "au" + "th=aa" + "token=" + '"s3cr3tv4lue1234"')
    assert mod.scan_file(str(target)) == [1]


def test_unquoted_rhs_in_json_and_toml_is_not_flagged(tmp_path: Path) -> None:
    """JSON and TOML string literals are always quoted, same as Python.

    Paired with a .yaml control so a threshold change cannot silently make this
    pass for the wrong reason.
    """
    mod = _load()
    rhs = "sec" + "ret: " + _token()
    for name in ("case.json", "case.toml"):
        target = tmp_path / name
        _bare(target, rhs)
        assert mod.scan_file(str(target)) == [], name
    control = tmp_path / "case.yaml"
    _bare(control, rhs)
    assert mod.scan_file(str(control)) == [1]


# --- a file the scanner cannot decode or open must not read as clean ------------
#
# scan_file opened with encoding="utf-8" and the default errors="strict", and
# caught UnicodeDecodeError by returning the lines found SO FAR, printing nothing.
# One byte that is not valid UTF-8 therefore truncated the scan of that file and
# the run still reported success.
#
# Measured 2026-09-20 on two fixtures identical except for a single byte:
#   valid UTF-8                        -> [2]   credential on line 2 FLAGGED
#   same content, one 0xff on line 1   -> []    nothing reported
#
# Anti-vacuity: test_valid_encoding_control_is_flagged is load-bearing. If the
# payload ever stopped clearing the length or entropy floor, the undecodable case
# would return [] for the innocent reason and would pass against the BROKEN
# scanner. An earlier version of this probe used a repeated three-character motif,
# whose Shannon entropy is about 1.58, and the control came back empty - proving
# nothing at all.


def _undecodable(path: Path, token: str) -> None:
    """Line 1 carries a byte that is not valid UTF-8; line 2 carries the payload."""
    payload = "api" + "_key = " + '"' + token + '"\n'
    path.write_bytes(b"# ordinary comment \xff\n" + payload.encode("utf-8"))


def _decodable(path: Path, token: str) -> None:
    """Byte-for-byte the same, minus the one bad byte."""
    payload = "api" + "_key = " + '"' + token + '"\n'
    path.write_bytes(b"# ordinary comment\n" + payload.encode("utf-8"))


def test_valid_encoding_control_is_flagged(tmp_path: Path) -> None:
    """Anti-vacuity anchor: the payload really does trip the scanner."""
    mod = _load()
    target = tmp_path / "control.py"
    _decodable(target, _token())
    assert mod.scan_file(str(target)) == [2]


def test_undecodable_byte_does_not_hide_a_later_secret(tmp_path: Path) -> None:
    """The regression. Before the fix this returned [] and printed nothing."""
    mod = _load()
    target = tmp_path / "dirty.py"
    _undecodable(target, _token())
    assert mod.scan_file(str(target)) == [2]


def test_unreadable_path_is_recorded_rather_than_passing_quietly(
    tmp_path: Path,
) -> None:
    """An OSError means the file was never examined, so it must not read as clean.

    A directory named like a scannable file is the portable way to force an
    OSError from open(): POSIX raises IsADirectoryError, Windows PermissionError,
    and both are OSError subclasses.
    """
    mod = _load()
    target = tmp_path / "looks_like_a_file.py"
    target.mkdir()
    assert mod.scan_file(str(target)) == []
    assert str(target) in mod.UNREADABLE_PATHS


def test_unreadable_paths_does_not_leak_between_runs(tmp_path: Path) -> None:
    """scan_tree clears the record, so one run cannot fail because of an earlier one."""
    mod = _load()
    mod.UNREADABLE_PATHS.append("stale/entry/from/a/previous/run.py")
    # Far below the floor, so this returns 1 for the vacuity reason, not the
    # unreadable one - the point is only that the stale entry is gone.
    mod.scan_tree(str(tmp_path), min_files=10)
    assert mod.UNREADABLE_PATHS == []


# ---------------------------------------------------------------------------
# EXCLUDE_PATHS is keyed to the repo-relative PATH, not to a basename
# ---------------------------------------------------------------------------
#
# The exclusion set used to be basenames tested with `name in EXCLUDE_FILES`,
# so a file called check_secrets.py ANYWHERE in the tree was skipped by a gate
# nobody had asked to skip it there. That is a widening of a security gate's
# blind spot, and it widens on its own as the tree grows.
#
# It is the file-name analogue of a defect this same function already had for
# DIRECTORY names, recorded in iter_scanned_files: EXCLUDE_DIRS prunes by
# directory name, which measurably hid 26 tracked files under results/.
#
# Measured before narrowing, because an exclusion that is load-bearing cannot
# simply be tightened: scan_file returns zero findings for all three excluded
# paths, and iter_scanned_files(".") returns the SAME 526 files before and
# after, with nothing gained and nothing lost. No collision exists today; these
# tests are what keep one from being introduced silently.


def test_intended_exclusions_are_still_excluded_at_their_real_paths() -> None:
    mod = _load()
    for rel in (
        "scripts/check_secrets.py",
        "tools/apply_protection.sh",
        "scripts/apply-branch-ruleset.ps1",
    ):
        assert mod._is_scannable(rel) is False, f"{rel} should remain excluded"


def test_a_colliding_basename_elsewhere_is_now_scanned() -> None:
    """The actual fix. Under the old basename set both of these were skipped."""
    mod = _load()
    assert mod._is_scannable("tests/fixtures/check_secrets.py") is True
    assert mod._is_scannable("vendor/tools/apply_protection.sh") is True


def test_exclusions_are_paths_not_basenames() -> None:
    """Anti-vacuity anchor: proves the two tests above differ for the right reason.

    If EXCLUDE_PATHS ever regressed to holding bare basenames, the collision
    test would fail; if it regressed to matching nothing, the exclusion test
    would fail. This asserts the stored shape directly so neither regression can
    be mistaken for the other.
    """
    mod = _load()
    assert all("/" in entry for entry in mod.EXCLUDE_PATHS), (
        f"EXCLUDE_PATHS must hold repo-relative paths, got {sorted(mod.EXCLUDE_PATHS)}"
    )


def test_windows_separators_match_the_same_exclusions() -> None:
    """os.walk yields backslashes on Windows; git ls-files yields forward slashes."""
    mod = _load()
    assert mod._is_scannable(r"scripts\check_secrets.py") is False
    assert mod._is_scannable("./scripts/check_secrets.py") is False


def test_ordinary_files_are_unaffected() -> None:
    mod = _load()
    assert mod._is_scannable("src/train_classifier.py") is True
    assert mod._is_scannable("README.md") is True
    assert mod._is_scannable("Dockerfile.api") is True
    assert mod._is_scannable("models/weights.bin") is False


# ---------------------------------------------------------------------------
# File-SELECTION coverage. Added 2026-09-23.
#
# The defect these cover was in _is_scannable, not in the patterns. Measured on
# b080b7ef before the fix: scan_file found the planted assignment in all ten
# formats below, while _is_scannable returned False for all ten, so scan_tree
# never opened one of them. A tree holding ten planted credentials and twelve
# clean .py files printed "[SUCCESS] No secrets detected." and exited 0.
#
# 145 of 678 tracked files were unscannable, 22 of them non-binary, including
# all four git hooks, the Makefile, both Snakemake files, pytest.ini, the
# rendered report source and the lockfile.
# ---------------------------------------------------------------------------

_NEWLY_COVERED = (
    "pipeline.smk",
    "docs/results_report.qmd",
    "notebooks/run.ipynb",
    "environments/requirements.lock",
    "setup.cfg",
    "pytest.ini",
    "deploy.env",
    "Makefile",
    "Snakefile",
    "LICENSE",
    ".github/CODEOWNERS",
    "scripts/hooks/pre-push",
    "scripts/hooks/pre-commit",
    "scripts/hooks/commit-msg",
    "scripts/hooks/prepare-commit-msg",
)


def test_formats_that_were_never_opened_are_now_scannable() -> None:
    mod = _load()
    missed = [rel for rel in _NEWLY_COVERED if not mod._is_scannable(rel)]
    assert missed == [], f"still unscannable: {missed}"


def test_a_new_hook_is_covered_without_editing_a_name_list() -> None:
    """scripts/hooks/ is selected as a DIRECTORY, so a fifth hook needs no edit."""
    mod = _load()
    assert mod._is_scannable("scripts/hooks/post-checkout") is True
    assert mod._is_scannable(r"scripts\hooks\post-checkout") is True


def test_selection_is_still_selective() -> None:
    """Anti-vacuity partner. If _is_scannable regressed to returning True for
    everything, the test above would pass for the wrong reason. These must stay
    OUT: binaries and bulk data are not credential surfaces and opening 84
    tracked .csv files would cost the walk for nothing.
    """
    mod = _load()
    for rel in (
        "models/weights.bin",
        "data/immunogenicity_dataset_v5.csv",
        "figures/roc.png",
        "data/proteome.fasta",
        "models/rf.joblib",
    ):
        assert mod._is_scannable(rel) is False, f"{rel} should not be scanned"


def test_every_bare_value_format_can_actually_be_opened() -> None:
    """The defect CLASS, not one instance.

    _BARE_VALUE_SUFFIXES decides how a value is parsed once a file is open. Three
    of its entries - .env, .cfg and .ini - were absent from the selection layer,
    so no file of those types was ever opened and that branch was unreachable.
    A capability declared in one layer and unreachable from another is exactly
    the shape that hid this bug; assert the two layers agree.
    """
    mod = _load()
    unreachable = [
        suffix
        for suffix in mod._BARE_VALUE_SUFFIXES
        if not mod._is_scannable("case" + suffix)
    ]
    assert unreachable == [], f"declared but never opened: {unreachable}"


def test_planted_credential_in_a_hook_turns_the_tree_red(tmp_path: Path) -> None:
    """End-to-end regression. Before the fix this exited 0 with [SUCCESS].

    The twelve clean .py files are load-bearing: without them scan_tree returns 1
    from the MIN_SCANNED_FILES floor, which is a different failure and would have
    made this test pass for the wrong reason.
    """
    mod = _load()
    hook = tmp_path / "scripts" / "hooks" / "pre-push"
    hook.parent.mkdir(parents=True)
    _write(hook, "api_" + "key", " = ", _token())
    for i in range(12):
        (tmp_path / f"mod{i}.py").write_text("x = 1\n", encoding="utf-8")
    assert mod.scan_tree(str(tmp_path), min_files=mod.MIN_SCANNED_FILES) == 1


def test_that_tree_is_clean_once_the_hook_is(tmp_path: Path) -> None:
    """Non-vacuity partner for the test above: same tree, credential removed."""
    mod = _load()
    hook = tmp_path / "scripts" / "hooks" / "pre-push"
    hook.parent.mkdir(parents=True)
    hook.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    for i in range(12):
        (tmp_path / f"mod{i}.py").write_text("x = 1\n", encoding="utf-8")
    assert mod.scan_tree(str(tmp_path), min_files=mod.MIN_SCANNED_FILES) == 0


# ---------------------------------------------------------------------------
# Vendor credential FORMATS. Added 2026-09-23.
#
# Every literal below is assembled at runtime. A real vendor-format token
# written into this file would be flagged by the very gates it tests: pre-commit
# Gate 2 reads staged content, and check_secrets scans tests/ as tracked source.
# The same reason the file writes "api_" + "key" elsewhere.
#
# Measured before the port: the assignment patterns need a credential-class NAME
# followed by = or :, so a self-identifying vendor token assigned to an innocuous
# name, embedded in a URL, or standing alone on a line was not flagged at all.
# ---------------------------------------------------------------------------


def _vendor_samples() -> dict:
    return {
        "github pat": "ghp_" + "016C4aBcDeFgHiJkLmNoPqRsTuVwXyZ1234",
        "github actions": "ghs_" + "016C4aBcDeFgHiJkLmNoPqRsTuVwXyZ1234",
        "github fine grained": "github_pat_" + "11ABCDEFG0aBcDeFgHiJkLmNoPqRsTuVwXyZ",
        "aws access key id": "AKIA" + "IOSFODNN7EXAMPLE",
        "aws temporary": "ASIA" + "IOSFODNN7EXAMPLE",
        "google api": "AIza" + "SyD1e2F3g4H5i6J7k8L9m0N1o2P3q4R5s6T",
        "slack": "xox" + "b-1234567890-abcdefGHIJ",
        "openai project": "sk-proj-" + "9fJkLmNoPqRsTuVwXyZaBcDeFgHi",
        "openai classic": "sk-" + "a" * 20 + "B3cD4eF5gH6iJ7kL8mN9oP0qR1sT2uV3",
        "anthropic": "sk-ant-" + "api03aBcDeFgHiJkLmNoPqRsTuVwXyZ",
        "pem header": "-----BEGIN RSA " + "PRIVATE KEY-----",
    }


def test_vendor_formats_are_flagged_without_a_credential_keyword(
    tmp_path: Path,
) -> None:
    """The gap: these carry no api_key/token/secret NAME, so the assignment
    patterns never saw them. They are assigned to an innocuous identifier here
    on purpose."""
    mod = _load()
    missed = []
    for label, sample in _vendor_samples().items():
        target = tmp_path / "case.py"
        target.write_text('default_value = "' + sample + '"\n', encoding="utf-8")
        if not mod.scan_file(str(target)):
            missed.append(label)
    assert missed == [], f"vendor formats not flagged: {missed}"


def test_vendor_format_inside_a_url_is_flagged(tmp_path: Path) -> None:
    """A token in a clone URL is assigned to nothing the keyword patterns match."""
    mod = _load()
    token = "ghp_" + "016C4aBcDeFgHiJkLmNoPqRsTuVwXyZ1234"
    target = tmp_path / "case.py"
    target.write_text(
        'remote = "https://' + token + '@github.com/o/r.git"\n', encoding="utf-8"
    )
    assert mod.scan_file(str(target)) == [1]


def test_pem_header_alone_on_a_line_is_flagged(tmp_path: Path) -> None:
    """No assignment at all, which is exactly how a pasted key block arrives."""
    mod = _load()
    target = tmp_path / "id_rsa"
    target.write_text("-----BEGIN RSA " + "PRIVATE KEY-----\n", encoding="utf-8")
    assert mod.scan_file(str(target)) == [1]


def test_vendor_pass_does_not_fire_on_lookalikes(tmp_path: Path) -> None:
    """Anti-vacuity partner. If the vendor pass regressed to matching broadly,
    the tests above would pass for the wrong reason. Two of these are real
    measured false-positive risks, not invented ones:

    - the amino-acid run is the FASTA collision the hook's own comment records
      for an unanchored AKIA/ASIA rule (DENV2_NGC_panel1.fasta:31);
    - the malformed sk-proj value has no hyphen after 'proj', and a first draft
      of this suite reported it as a coverage gap when the test value, not the
      pattern, was wrong.
    """
    mod = _load()
    for label, line in {
        "amino acid run": 'seq = "FTDPASIAARGYISTRVEMGEAAGIF"',
        "malformed openai prefix": 'c = "sk-' + 'proj9fJkLmNoPqRsTuVwXyZaBcDeFgHi"',
        "prose about a password": 'password: "must be at least twelve characters"',
        "short lookalike": 'x = "ghp_' + 'abc"',
        "the word private key in prose": "# rotate the private key every 90 days",
    }.items():
        target = tmp_path / "case.py"
        target.write_text(line + "\n", encoding="utf-8")
        assert mod.scan_file(str(target)) == [], f"false positive on {label}"


def test_ci_scanner_carries_every_pattern_the_local_hook_does() -> None:
    """Drift guard, and the reason this port exists.

    scripts/hooks/pre-commit is a LOCAL hook: it runs on a developer machine and
    on no CI runner. check_secrets.py is the only content scanner CI runs. When
    the hook knows a credential format and the scanner does not, a commit made
    without the hook installed - or created server side by a squash merge -
    reaches the public remote with CI green. Measured 2026-09-23: the scanner
    carried none of the hook's 12 formats.

    Asserting the sets are equal rather than a subset, so a pattern added to
    either side has to be added to both.
    """
    import re as _re

    mod = _load()
    hook_text = (
        Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "pre-commit"
    ).read_text(encoding="utf-8")
    block = _re.search(r"CRED_PATTERNS=\((.*?)\n\)", hook_text, _re.S)
    assert block is not None, "CRED_PATTERNS array not found in pre-commit"
    hook_patterns = {
        m.group(1) for m in _re.finditer(r"^\s*'([^']+)'", block.group(1), _re.M)
    }
    assert len(hook_patterns) >= 10, f"parsed only {len(hook_patterns)} patterns"
    ported = {p.pattern for p in mod.VENDOR_CREDENTIAL_FORMATS}
    assert hook_patterns == ported, (
        f"only in hook: {sorted(hook_patterns - ported)}; "
        f"only in scanner: {sorted(ported - hook_patterns)}"
    )


# ---------------------------------------------------------------------------
# Credential-class NAMES and URL userinfo. Added 2026-09-23.
#
# The third measured gap in the same finding. Unlike the vendor formats these
# ARE assignments; the alternation simply did not carry the name.
# ---------------------------------------------------------------------------


def test_pwd_and_credentials_names_are_flagged(tmp_path: Path) -> None:
    mod = _load()
    for left in ("DB_" + "PWD", "credential" + "s", "credential"):
        target = tmp_path / "case.py"
        _write(target, left, " = ", _token())
        assert mod.scan_file(str(target)) == [1], f"{left} not flagged"


def test_password_inside_a_url_is_flagged(tmp_path: Path) -> None:
    """Keyword-independent: `postgres://u:<value>@h/db` names nothing the
    assignment patterns recognise."""
    mod = _load()
    target = tmp_path / "case.py"
    target.write_text(
        'DATABASE_URL = "postgres://u:' + _token() + '@h:5432/db"\n', encoding="utf-8"
    )
    assert mod.scan_file(str(target)) == [1]


def test_url_placeholder_is_not_flagged(tmp_path: Path) -> None:
    """Anti-vacuity partner for the test above, and the reason the URL pattern
    is routed through the same length floor as the others: a documentation
    placeholder carries an 8-character value and does not clear `len > 8`."""
    mod = _load()
    target = tmp_path / "doc.py"
    target.write_text(
        'doc = "https://user:' + "pass" + 'word@example.com"\n', encoding="utf-8"
    )
    assert mod.scan_file(str(target)) == []


def test_suffix_key_names_stay_unflagged(tmp_path: Path) -> None:
    """A deliberate NON-widening, kept as a test so it is not quietly reversed.

    An audit recommended covering `*_KEY` names. Measured over every scanned
    file at b080b7ef, 547 of them, the obvious pattern `[a-z0-9]+[_-]key`
    produces six hits on tracked code and every one is a false positive: two
    module constants holding a ledger key and a baseline name, and four
    `score_key` lines - one assignment and three dict-rename arguments.
    Re-measured after PRs #540, #541 and #543 landed, 549 files, the same six
    at the same lines: the file count moves with the tree, the finding does not.
    Adding it would turn a blocking CI gate red on legitimate code.

    These three lines are copied from the real tracked hits. If a future change
    widens the alternation to reach them, this test fails and the measurement
    above has to be redone rather than rediscovered.
    """
    mod = _load()
    for line in (
        'RATCHET_' + 'KEY = "exempt_ledger_citation_ceiling"',
        'BASELINE_' + 'KEY = "iedb_ebv_hpv16_tcell"',
        'df.rename(columns={peptide_key: "peptide", score_' + 'key: "predig_max_score"})',
    ):
        target = tmp_path / "case.py"
        target.write_text(line + "\n", encoding="utf-8")
        assert mod.scan_file(str(target)) == [], f"false positive on: {line[:30]}"


# ---------------------------------------------------------------------------
# Tracked paths that git has to quote. Added 2026-09-23.
#
# _tracked_paths is the additive net that pulls TRACKED files back in after
# EXCLUDE_DIRS prunes the walk by directory name. It is the only thing that
# reaches a tracked file under results/, so a path it cannot represent is a
# file nothing scans. Anywhere else the walk finds the file regardless, which
# is why this is tested under an excluded directory specifically.
# ---------------------------------------------------------------------------

# Built with chr() so this SOURCE file stays pure ASCII. A literal here
# fails tests/test_encoding_ascii_output.py, which allowlists non-ASCII
# string literals per module - a real gate that caught exactly this.
_NON_ASCII_NAME = "caf" + chr(0xE9) + ".md"


def test_tracked_non_ascii_path_under_an_excluded_dir_is_scanned(
    tmp_path: Path,
) -> None:
    """Two defects had to be fixed together and either one alone still drops it.

    Without -z, git applies core.quotePath and the name arrives as
    "caf\303\251.md", which names no file. With -z but with text=True alone,
    the UTF-8 bytes are decoded using the process's preferred encoding - cp1252
    on the Windows workstation - and the name arrives double-encoded, which also
    names no file. Both end at the same os.path.isfile() guard.
    """
    mod = _load()
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)
    _git(repo, "init", "-q")
    for name in ("plain.md", _NON_ASCII_NAME):
        _write(repo / "results" / name, "api_" + "key", " = ", _token())
    _git(repo, "add", "-A")
    names = _basenames(mod, repo)
    assert "plain.md" in names, "the ASCII control was not selected either"
    assert _NON_ASCII_NAME in names, f"non-ASCII tracked path dropped; got {names}"


def test_untracked_non_ascii_path_under_an_excluded_dir_is_still_skipped(
    tmp_path: Path,
) -> None:
    """Anti-vacuity partner. Selection under an excluded directory must still be
    driven by TRACKEDNESS; if the fix had widened the walk instead, this would
    start being scanned and the test above would pass for the wrong reason."""
    mod = _load()
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)
    _git(repo, "init", "-q")
    _write(repo / "results" / _NON_ASCII_NAME, "api_" + "key", " = ", _token())
    assert _NON_ASCII_NAME not in _basenames(mod, repo)


def test_tracked_paths_parses_the_repo_without_quoting_artifacts() -> None:
    """Every entry must name a real file. A quoted or mis-decoded path is not an
    error anywhere - it simply fails os.path.isfile() and vanishes - so assert
    the property directly rather than waiting for a count to look wrong."""
    import os as _os

    mod = _load()
    root = str(Path(__file__).resolve().parents[1])
    tracked = mod._tracked_paths(root)
    assert len(tracked) > 100, f"only {len(tracked)} tracked paths parsed"
    quoted = [p for p in tracked if p.startswith('"') and p.endswith('"')]
    assert quoted == [], f"quoted paths returned: {quoted[:5]}"
    missing = [p for p in tracked if not _os.path.exists(_os.path.join(root, p))]
    assert missing == [], f"paths naming no file: {missing[:5]}"
