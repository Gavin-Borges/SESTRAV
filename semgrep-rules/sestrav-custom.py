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
from typing import Any

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

# A flag that is not the literal True can still be False at run time, so every
# such spelling is the opt-out too: a name, an expression, the third positional
# argument, or a mapping unpacked into the call.
strict = bool(stream.getvalue())
options: dict[str, Any] = {"required_checksum": False}
read_options: dict[str, Any] = {"required": False}

# ruleid: sestrav-require-verified-joblib-load
load_verified_joblib("model.joblib", required_checksum=strict)

# ruleid: sestrav-require-verified-joblib-load
load_verified_joblib("model.joblib", required_checksum=not strict)

# ruleid: sestrav-require-verified-joblib-load
load_verified_joblib("model.joblib", None, strict)

# ruleid: sestrav-require-verified-joblib-load
load_verified_joblib("model.joblib", **options)

# ruleid: sestrav-require-verified-joblib-load
read_verified_bytes("model.joblib", required=strict)

# ruleid: sestrav-require-verified-joblib-load
read_verified_bytes("model.joblib", None, strict)

# ruleid: sestrav-require-verified-joblib-load
read_verified_bytes("model.joblib", **read_options)

# The default is True, so a call that passes no flag is the verified load.
# ok: sestrav-require-verified-joblib-load
load_verified_joblib("model.joblib")

# ok: sestrav-require-verified-joblib-load
load_verified_joblib("model.joblib", None, True)

# ok: sestrav-require-verified-joblib-load
read_verified_bytes("model.joblib", None, True)


def _load_joblib_impl(path):
    """The rule exempts this body by name, so the opt-out inside it is allowed."""
    # ok: sestrav-require-verified-joblib-load
    return read_verified_bytes(path, required=False)


# ruleid: sestrav-forbid-subprocess-shell-true
subprocess.run("echo fixture", shell=True)  # noqa: S602  # nosec B602

# ok: sestrav-forbid-subprocess-shell-true
subprocess.run(["echo", "fixture"], check=True)
