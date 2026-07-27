import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

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
