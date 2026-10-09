"""Semgrep positive and negative examples."""

import subprocess
import sys

# ruleid: sestrav-prefer-sys-executable
subprocess.run(["python", "task.py"], check=True)

# ok: sestrav-prefer-sys-executable
subprocess.run([sys.executable, "task.py"], check=True)
