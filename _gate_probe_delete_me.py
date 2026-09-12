"""THROWAWAY probe. Exists only to prove the blocking semgrep gate fails.

Deliberately violates semgrep-rules/sestrav-custom.yml rule
sestrav-require-verified-joblib-load. Never merge this file.
"""

import joblib


def load(path):
    return joblib.load(path)
