import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Keep pytest temp directories inside the writable workspace when the system
# temp root is locked down on Windows. Key by PID to avoid collisions with
# locked directories from previous sessions.
if "PYTEST_DEBUG_TEMPROOT" not in os.environ:
    _tmp_root = os.path.join(os.path.dirname(__file__), ".pytest_tmp2")
    os.makedirs(_tmp_root, exist_ok=True)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = _tmp_root
