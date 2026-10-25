"""pre-push Check 6, pass (b): the pushed-range scan reads every blob a push publishes.

Why this test exists
--------------------
Pass (b) scans the blobs that the pushed commits introduce, so a value that a
later commit in the same push removes is still caught: pass (a) reads only the
checkout, where that value no longer exists. Three kinds of pushed blob were
never read, and a value added then removed inside one push reached the remote's
history with "Pushed commit ranges contain no credential assignments." printed
(security sweep finding SCR-02):

1. Paths under a directory name in scripts/check_secrets.py's EXCLUDE_DIRS
   (results/, build/, scratch/ and the rest). Pass (a) prunes those names from
   its directory WALK, where they hold untracked build output and virtualenvs,
   and then adds every tracked file back. Pass (b) inherited the prune, and every
   blob in a pushed commit is published.
2. Type changes. `git diff-tree --diff-filter=AM` drops status T, so a symlink
   replaced by a regular file holding a value was never read.
3. Content that only a merge commit introduced, such as a conflict resolution.
   Without -c, `git diff-tree` lists no paths for a merge commit.

How the hook is driven
----------------------
The harness is the one tests/test_pre_push_blob_scan_keeps_non_ascii_names.py
uses. The whole tracked hook is copied into a throwaway repository and run with
bash, as git runs it, with the pushed ref on stdin. A `python` shim placed first
on PATH runs pass (b)'s heredoc, the hook's only `python -` call, with the
interpreter running this test, and answers every other python call with 0, so
no other check can block or pass these pushes. scripts/check_secrets.py is
copied beside the hook because pass (b) imports its scanner from there.

Every push is a new branch whose boundary is refs/remotes/origin/main, set to
the commit the fixture treats as already published. That is the hook's
`--not --remotes=origin` enumeration, the first push of a pull-request branch.

Which tests need the hook change
--------------------------------
scripts/hooks/pre-push is an owner-only path, so the hook change lands as a
separate owner commit in the same pull request. Until it does, exactly these
six cases fail by design, because they assert the gaps are closed:
the three `test_add_then_delete_under_a_pruned_directory_is_blocked` cases,
`test_a_type_change_then_delete_is_blocked`,
`test_a_conflict_resolution_only_secret_is_blocked` and
`test_a_file_added_only_by_a_merge_is_blocked`. Every other test here passes
with or without it.

Anti-vacuity
------------
`test_add_then_delete_under_src_is_blocked` is the control: the same push
shape outside the pruned names blocks on the hook before and after the change,
so a passing positive case cannot mean the scan never ran. Each positive case
asserts the block is a [FLAGGED-RANGE] finding naming the path, never the
hook's fail-closed [UNREADABLE-RANGE] error path. The negatives guard the
other direction: a clean range that touches the same directories and carries
a merge passes; a deletion, in a plain commit or inside a merge, passes,
which fails if the status filter is dropped (a deleted path cannot be read and
fails closed); and a merge that brings in history the remote already holds is
not re-read, which fails if merges are read pairwise with -m instead of -c.
The premise tests assert the git behaviour each fixture relies on.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_SOURCE = REPO_ROOT / "scripts" / "hooks" / "pre-push"
SCANNER_SOURCE = REPO_ROOT / "scripts" / "check_secrets.py"

FLAGGED = "[FLAGGED-RANGE]"
UNREADABLE = "[UNREADABLE-RANGE]"
RANGE_CLEAN = "Pushed commit ranges contain no credential assignments."
RANGE_BLOCKED = "Pushed commit range credential scan failed."
PUSH_ALLOWED = "All checks passed -- push allowed."

ZERO_SHA = "0" * 40

# Runs pass (b)'s heredoc with a real interpreter; every other python call exits 0.
SHIM = (
    "#!/bin/sh\n"
    'if [ "$1" = "-" ]; then\n'
    '    exec "$PREPUSH_REAL_PYTHON" "$@"\n'
    "fi\n"
    "exit 0\n"
)

# git reads the developer's GLOBAL and SYSTEM config even for a throwaway directory.
# Pinned so the result depends on the hook. The repository-discovery variables are
# dropped so every git call resolves to the fixture repository.
BASE_ENV = {
    key: value
    for key, value in os.environ.items()
    if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX", "REPO_ROOT"}
}
BASE_ENV.update(
    {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }
)

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="pre-push is a bash script; without bash it cannot run at all",
)


def _payload() -> str:
    """One credential-keyword assignment that scan_file flags, assembled at runtime.

    The value is mixed-alphabet, longer than 8 and above the 3.0 entropy floor.
    It is not a real credential.
    """
    keyword = "api_" + "key"
    value = "a8f3k9d2" + "m1q7x4z0b5"
    return keyword + ' = "' + value + '"\n'


def _git(root: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", "-c", "core.autocrlf=false", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
        env=BASE_ENV,
        timeout=60,
    ).stdout


def _repo(tmp_path: Path) -> Path:
    """An empty repository on `main`; the hook, scanner and shim sit beside it, untracked."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "hook").write_text(
        HOOK_SOURCE.read_text(encoding="utf-8"), encoding="utf-8", newline=""
    )
    scanner = root / "scripts" / "check_secrets.py"
    scanner.parent.mkdir()
    shutil.copyfile(SCANNER_SOURCE, scanner)
    shim_dir = root / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "python"
    shim.write_text(SHIM, encoding="utf-8", newline="\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    _git(root, "init", "-q", "-b", "main")
    return root


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _commit(root: Path, message: str, *paths: str) -> str:
    # -f because a developer's global excludes are not read here, but a path such
    # as build/ must stay addable whatever a future fixture's .gitignore says.
    if paths:
        _git(root, "add", "-f", "--", *paths)
    _git(root, "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD").strip()


def _push(root: Path, local_sha: str, published: str) -> subprocess.CompletedProcess:
    """Push a new branch at `local_sha`, with `published` as what origin already holds."""
    _git(root, "update-ref", "refs/remotes/origin/main", published)
    env = dict(BASE_ENV)
    env.update(
        {
            # Skips the hook's conda activation, so the shim stays the interpreter.
            "CONDA_DEFAULT_ENV": "sestrav",
            "PREPUSH_REAL_PYTHON": Path(sys.executable).as_posix(),
        }
    )
    env["PATH"] = str(root / "shim") + os.pathsep + env.get("PATH", "")
    # Bytes, so the line reaches the hook with LF exactly as git writes it.
    stdin = f"refs/heads/feature {local_sha} refs/heads/feature {ZERO_SHA}\n".encode()
    result = subprocess.run(
        ["bash", "hook", "origin", "https://example.invalid/fixture.git"],
        cwd=root,
        input=stdin,
        capture_output=True,
        check=False,
        env=env,
        timeout=180,
    )
    return subprocess.CompletedProcess(
        result.args,
        result.returncode,
        result.stdout.decode("utf-8", "replace"),
        result.stderr.decode("utf-8", "replace"),
    )


def _assert_blocked_on(result: subprocess.CompletedProcess, rel_path: str) -> None:
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert f"{FLAGGED} {rel_path} at" in result.stderr, output
    assert UNREADABLE not in result.stderr, output
    assert RANGE_BLOCKED in result.stderr, output
    assert PUSH_ALLOWED not in result.stdout, output


def _assert_allowed(result: subprocess.CompletedProcess) -> None:
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert RANGE_CLEAN in result.stdout, output
    assert FLAGGED not in result.stderr, output
    assert UNREADABLE not in result.stderr, output
    assert PUSH_ALLOWED in result.stdout, output


def _load_scanner():
    spec = importlib.util.spec_from_file_location("check_secrets", SCANNER_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- fixtures: each returns (repo, pushed tip, published base) -------------


def _add_then_delete(tmp_path: Path, rel_path: str) -> tuple[Path, str, str]:
    root = _repo(tmp_path)
    _write(root, "README.md", "base\n")
    base = _commit(root, "base", "README.md")
    _write(root, rel_path, _payload())
    _commit(root, "add", rel_path)
    _git(root, "rm", "-q", "--", rel_path)
    tip = _commit(root, "remove")
    return root, tip, base


def _type_change_then_delete(tmp_path: Path) -> tuple[Path, str, str, str]:
    """link.py is published as a symlink, becomes a regular file holding the value, then goes."""
    root = _repo(tmp_path)
    _write(root, "link.py", "target.py")
    _git(root, "add", "--", "link.py")
    blob = _git(root, "rev-parse", ":link.py").strip()
    # Records link.py as a symlink without creating one on disk, which Windows
    # does not allow by default. update-index only rewrites the index entry.
    _git(root, "update-index", "--cacheinfo", f"120000,{blob},link.py")
    base = _commit(root, "base: link.py is a symlink")
    _git(root, "rm", "-q", "--cached", "--", "link.py")
    _write(root, "link.py", _payload())
    changed = _commit(root, "link.py becomes a regular file", "link.py")
    _git(root, "rm", "-q", "--", "link.py")
    tip = _commit(root, "remove")
    return root, tip, base, changed


def _conflict_resolution_then_drop(tmp_path: Path) -> tuple[Path, str, str, str]:
    root = _repo(tmp_path)
    _write(root, "app.py", "x = 1\n")
    base = _commit(root, "base", "app.py")
    _git(root, "checkout", "-q", "-b", "side")
    _write(root, "app.py", "x = 2\n")
    _commit(root, "side", "app.py")
    _git(root, "checkout", "-q", "main")
    _write(root, "app.py", "x = 3\n")
    _commit(root, "main", "app.py")
    conflicted = subprocess.run(
        ["git", "merge", "-q", "side"],
        cwd=root,
        capture_output=True,
        check=False,
        env=BASE_ENV,
        timeout=60,
    )
    assert conflicted.returncode != 0, "the fixture needs a conflict to resolve"
    _write(root, "app.py", "x = 4\n" + _payload())
    merge = _commit(root, "merge side", "app.py")
    _write(root, "app.py", "x = 4\n")
    tip = _commit(root, "drop", "app.py")
    return root, tip, base, merge


# ---- premises --------------------------------------------------------------


def test_the_payload_is_flagged_by_the_scanner(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(_payload(), encoding="utf-8", newline="\n")
    assert _load_scanner().scan_file(str(probe)) == [1], "payload is not flagged"


def test_premise_a_type_change_is_status_t_and_am_drops_it(tmp_path: Path) -> None:
    root, _tip, _base, changed = _type_change_then_delete(tmp_path)
    status = _git(root, "diff-tree", "-r", "--no-commit-id", "--name-status", changed)
    assert status.split() == ["T", "link.py"], status
    filtered = _git(
        root, "diff-tree", "-r", "--no-commit-id", "--name-only", "--diff-filter=AM", changed
    )
    assert filtered.strip() == "", filtered


def test_premise_a_merge_lists_paths_only_with_c(tmp_path: Path) -> None:
    root, _tip, _base, merge = _conflict_resolution_then_drop(tmp_path)
    plain = _git(root, "diff-tree", "-r", "--no-commit-id", "--name-only", merge)
    assert plain.strip() == "", plain
    combined = _git(root, "diff-tree", "-r", "-c", "--no-commit-id", "--name-only", merge)
    assert combined.split() == ["app.py"], combined


# ---- positives -------------------------------------------------------------


def test_add_then_delete_under_src_is_blocked(tmp_path: Path) -> None:
    root, tip, base = _add_then_delete(tmp_path, "src/cfg.py")
    _assert_blocked_on(_push(root, tip, base), "src/cfg.py")


@pytest.mark.parametrize("directory", ["results", "build", "scratch"])
def test_add_then_delete_under_a_pruned_directory_is_blocked(
    tmp_path: Path, directory: str
) -> None:
    rel_path = f"{directory}/cfg.py"
    root, tip, base = _add_then_delete(tmp_path, rel_path)
    _assert_blocked_on(_push(root, tip, base), rel_path)


def test_a_type_change_then_delete_is_blocked(tmp_path: Path) -> None:
    root, tip, base, _changed = _type_change_then_delete(tmp_path)
    _assert_blocked_on(_push(root, tip, base), "link.py")


def test_a_conflict_resolution_only_secret_is_blocked(tmp_path: Path) -> None:
    root, tip, base, _merge = _conflict_resolution_then_drop(tmp_path)
    _assert_blocked_on(_push(root, tip, base), "app.py")


def test_a_file_added_only_by_a_merge_is_blocked(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "a.py", "a = 1\n")
    base = _commit(root, "base", "a.py")
    _git(root, "checkout", "-q", "-b", "side")
    _write(root, "b.py", "b = 1\n")
    _commit(root, "side", "b.py")
    _git(root, "checkout", "-q", "main")
    _write(root, "c.py", "c = 1\n")
    _commit(root, "main", "c.py")
    _git(root, "merge", "-q", "--no-ff", "--no-commit", "side")
    _write(root, "merged.py", _payload())
    _commit(root, "merge side", "merged.py")
    _git(root, "rm", "-q", "--", "merged.py")
    tip = _commit(root, "drop")
    _assert_blocked_on(_push(root, tip, base), "merged.py")


# ---- negatives -------------------------------------------------------------


def test_a_clean_range_through_pruned_directories_and_a_merge_passes(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "README.md", "one\ntwo\nthree\nfour\nfive\nsix\n")
    base = _commit(root, "base", "README.md")
    _git(root, "checkout", "-q", "-b", "side")
    _write(root, "results/summary.md", "| metric | value |\n")
    _write(root, "README.md", "one\ntwo\nthree\nfour\nfive\nSIX\n")
    _commit(root, "side", "results/summary.md", "README.md")
    _git(root, "checkout", "-q", "main")
    _write(root, "build/notes.txt", "built\n")
    _write(root, "src/app.py", "print('hello')\n")
    _write(root, "README.md", "ONE\ntwo\nthree\nfour\nfive\nsix\n")
    _commit(root, "main", "build/notes.txt", "src/app.py", "README.md")
    # README.md auto-merges to content neither parent holds, so -c lists it.
    _git(root, "merge", "-q", "--no-ff", "-m", "merge side", "side")
    merge = _git(root, "rev-parse", "HEAD").strip()
    combined = _git(root, "diff-tree", "-r", "-c", "--no-commit-id", "--name-only", merge)
    assert combined.split() == ["README.md"], combined
    _git(root, "rm", "-q", "--", "build/notes.txt")
    tip = _commit(root, "drop notes")
    _assert_allowed(_push(root, tip, base))


def test_deleting_a_published_file_passes(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "src/old.py", _payload())
    base = _commit(root, "base holds it", "src/old.py")
    _git(root, "rm", "-q", "--", "src/old.py")
    tip = _commit(root, "delete")
    _assert_allowed(_push(root, tip, base))


def test_a_merge_that_deletes_a_file_passes(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "keep.py", "k = 1\n")
    _write(root, "gone.py", "g = 1\n")
    base = _commit(root, "base", "keep.py", "gone.py")
    _git(root, "checkout", "-q", "-b", "side")
    _write(root, "b.py", "b = 1\n")
    _commit(root, "side", "b.py")
    _git(root, "checkout", "-q", "main")
    _write(root, "c.py", "c = 1\n")
    _commit(root, "main", "c.py")
    _git(root, "merge", "-q", "--no-ff", "--no-commit", "side")
    _git(root, "rm", "-q", "--", "gone.py")
    merge = _commit(root, "merge side, dropping gone.py")
    combined = _git(root, "diff-tree", "-r", "-c", "--no-commit-id", "--name-status", merge)
    assert combined.split() == ["DD", "gone.py"], combined
    _assert_allowed(_push(root, merge, base))


def test_merging_history_the_remote_holds_is_not_read_again(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "README.md", "base\n")
    _commit(root, "base", "README.md")
    _git(root, "checkout", "-q", "-b", "feature")
    _write(root, "clean.py", "ok = 1\n")
    _commit(root, "feature", "clean.py")
    _git(root, "checkout", "-q", "main")
    # Stands for a blob the remote already published. Pass (a) answers for a
    # value present in the checkout; this asserts only that the merge commit's
    # scan covers the paths that differ from EVERY parent, not the merged branch.
    _write(root, "vendor.py", _payload())
    published = _commit(root, "published main", "vendor.py")
    _git(root, "checkout", "-q", "feature")
    _git(root, "merge", "-q", "--no-ff", "-m", "merge main", "main")
    tip = _git(root, "rev-parse", "HEAD").strip()
    pairwise = _git(root, "diff-tree", "-r", "-m", "--no-commit-id", "--name-only", tip)
    assert "vendor.py" in pairwise.split(), pairwise
    _assert_allowed(_push(root, tip, published))
