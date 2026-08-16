import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

# Drop inherited git repository-discovery variables before anything is collected.
#
# Git exports GIT_DIR / GIT_WORK_TREE / GIT_INDEX_FILE into hook subprocesses,
# and scripts/hooks/pre-push runs pytest, so every test in this suite inherits
# them on the pre-push path. They take PRECEDENCE over cwd-based discovery, so
# any test that builds a throwaway repo and runs git in it with cwd= alone
# targets the REAL repository instead. From the main worktree GIT_DIR is the
# relative ".git", which re-resolves against the temp dir and happens to work;
# from a linked worktree it is absolute, and the throwaway git commands hit the
# real repo. That is not hypothetical: it wrote bogus "initial" commits onto a
# real branch ref on 2026-08-16, and tests/test_check_doc_line_citations.py
# additionally runs `git rm src/train_classifier.py` and can rewrite
# docs/line_citations.json under the same leak.
#
# Stripped here, at module scope rather than in a fixture, because
# tests/test_check_doc_commit_refs.py shells out to git at COLLECTION time (in a
# skipif decorator), which is earlier than any session-scoped fixture runs.
# Individual helpers still pass an explicit scrubbed env as a second guard - see
# tests/test_check_lockfile_freshness.py, which established this idiom, and
# tools/check_lockfile_freshness.py, whose docstring documents the mechanism.
for _leaked_git_var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX"):
    os.environ.pop(_leaked_git_var, None)

# Choose the root pytest uses for tmp_path / tmpdir.
#
# Prefer the system temp dir. Pinning the root inside the repo breaks any deep
# checkout on Windows (for example a git worktree created under
# AppData/Local/Temp): the generated per-test directories then exceed the
# 260-character MAX_PATH limit and tests fail with spurious FileNotFoundError.
#
# The in-repo root is kept only as a fallback for the case this override was
# originally added for: a system temp root that is not writable. The fallback
# directory name must stay exactly ".pytest_tmp2" - that literal is hardcoded in
# .gitignore, .dockerignore and scripts/check_secrets.py, and the directory
# contains the local username, so a different name would silently un-ignore it.
#
# An explicit PYTEST_DEBUG_TEMPROOT in the environment always wins.
if "PYTEST_DEBUG_TEMPROOT" not in os.environ:
    try:
        _system_tmp = tempfile.gettempdir()
        with tempfile.TemporaryDirectory(dir=_system_tmp):
            pass
        _tmp_root = _system_tmp
    except OSError:
        _tmp_root = os.path.join(os.path.dirname(__file__), ".pytest_tmp2")
        os.makedirs(_tmp_root, exist_ok=True)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = _tmp_root
