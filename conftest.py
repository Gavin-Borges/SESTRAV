import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Keep pytest temp directories inside the writable workspace when the system
# temp root is locked down on Windows.
if "PYTEST_DEBUG_TEMPROOT" not in os.environ:
    os.environ["PYTEST_DEBUG_TEMPROOT"] = os.path.join(os.path.dirname(__file__), ".pytest_tmp")
os.makedirs(os.environ["PYTEST_DEBUG_TEMPROOT"], exist_ok=True)
