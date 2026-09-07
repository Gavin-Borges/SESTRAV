"""The HYPOTHESIS_MAX_EXAMPLES knob must actually control fuzz depth.

`.github/workflows/fuzzing.yml` exports HYPOTHESIS_MAX_EXAMPLES (200 on push and
pull_request, 1000 on the weekly schedule) and `docs/SCORECARD_REMEDIATION.md`
documents it as the depth control. For the life of that workflow it was inert.
Hypothesis has no built-in support for that variable name, and nothing in this
repository read it - there was no register_profile, load_profile or settings()
call anywhere. Measured on hypothesis 6.157.1: with HYPOTHESIS_MAX_EXAMPLES=7
exported, settings.default.max_examples still reported the library default 100.

The consequence was not a slow test suite but a false claim: the weekly
"extended" run was never deeper than a PR run, only differently seeded, while a
tracked document told readers otherwise.

Run in a subprocess because the root conftest.py reads the variable at import
time, so whatever value THIS process started with is already baked in and
cannot be varied from inside a test. The subprocess only imports conftest.py
and reads hypothesis.settings.default.max_examples straight back - it does not
run pytest or parse `--hypothesis-show-statistics` text. That text format is
not version-stable: this repo pins hypothesis==6.165.10 (requirements.txt), an
earlier version of this test was written and measured against 6.157.1, and its
statistics-scraping logic silently matched zero lines on the pinned version.
Reading settings.default directly has no such surface - register_profile,
load_profile and settings.default are the stable public API this fix itself
depends on.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Importing conftest.py re-runs its module-level HYPOTHESIS_MAX_EXAMPLES branch
# (see conftest.py), which registers and loads the "sestrav" profile only when
# the variable is set to a truthy value. settings.default then reports whatever
# profile is actually ACTIVE, which is a stronger check than settings.get_profile
# ("sestrav") would be: get_profile only proves a profile was registered, not
# that load_profile made it the one hypothesis will actually use.
_PROBE = "import conftest\nfrom hypothesis import settings\nprint(settings.default.max_examples)"


def _max_examples_with(value: str | None) -> int:
    env = dict(os.environ)
    env.pop("HYPOTHESIS_MAX_EXAMPLES", None)
    if value is not None:
        env["HYPOTHESIS_MAX_EXAMPLES"] = value
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return int(result.stdout.strip())


def test_env_var_sets_the_example_count():
    """The load-bearing half: a set variable must change the depth.

    Uses a value that is neither Hypothesis's default (100) nor either CI value
    (200, 1000), so a pass cannot come from the variable being ignored.
    """
    assert _max_examples_with("13") == 13


def test_unset_env_var_falls_back_to_the_hypothesis_default():
    """The narrowness half: the profile must not pin a value when unset.

    Without this, a conftest that hardcoded a depth would pass the test above
    while making the variable irrelevant again in the other direction.
    """
    assert _max_examples_with(None) == 100
