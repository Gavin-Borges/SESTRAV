"""Semgrep positive and negative examples for sestrav-custom.yml.

Never imported or executed. The semgrep-custom CI job runs
scripts/check_semgrep_rule_fixtures.py over this file, which proves each rule
still reports every `ruleid:` line and none of the `ok:` lines. `.semgrepignore`
keeps it out of the blocking scan itself. Every input is a constant, so no
outside data reaches any call.
"""

import io
import pickle
import subprocess

import joblib

from src.artifact_integrity import load_verified_joblib, read_verified_bytes

stream = io.BytesIO(b"")

# ruleid: sestrav-forbid-pickle-non-literal
pickle.load(stream)  # noqa: S301  # nosec B301

with open("model.pkl", "rb") as fixed:
    # ok: sestrav-forbid-pickle-non-literal
    pickle.load(fixed)  # noqa: S301  # nosec B301

opened = open("model.pkl", "rb")
# ok: sestrav-forbid-pickle-non-literal
pickle.load(opened)  # noqa: S301  # nosec B301

# ruleid: sestrav-require-verified-joblib-load
joblib.load("model.joblib")

# ruleid: sestrav-require-verified-joblib-load
load_verified_joblib("model.joblib", required_checksum=False)

# ruleid: sestrav-require-verified-joblib-load
load_verified_joblib("model.joblib", False)

# ok: sestrav-require-verified-joblib-load
load_verified_joblib("model.joblib", required_checksum=True)

# read_verified_bytes is the single read load_verified_joblib performs, so its
# `required=False` is the same opt-out under another parameter name.
# ruleid: sestrav-require-verified-joblib-load
read_verified_bytes("model.joblib", required=False)

# ruleid: sestrav-require-verified-joblib-load
read_verified_bytes("model.joblib", None, False)

# ok: sestrav-require-verified-joblib-load
read_verified_bytes("model.joblib", required=True)


def _load_joblib_impl(path):
    """The rule exempts this body by name, so the opt-out inside it is allowed."""
    # ok: sestrav-require-verified-joblib-load
    return read_verified_bytes(path, required=False)


# ruleid: sestrav-forbid-subprocess-shell-true
subprocess.run("echo fixture", shell=True)  # noqa: S602  # nosec B602

# ok: sestrav-forbid-subprocess-shell-true
subprocess.run(["echo", "fixture"], check=True)
