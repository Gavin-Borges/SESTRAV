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

Asserted in a subprocess because the root conftest.py reads the variable at
import time, so whatever value THIS process started with is already baked in and
cannot be varied from inside a test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_fuzz_with(max_examples: str | None) -> str:
    env = dict(os.environ)
    env.pop("HYPOTHESIS_MAX_EXAMPLES", None)
    if max_examples is not None:
        env["HYPOTHESIS_MAX_EXAMPLES"] = max_examples
    # argparse and pytest wrap output to terminal width; pin it so the
    # statistics lines this test parses are not split mid-number.
    env["COLUMNS"] = "200"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_fuzz.py",
            "-q",
            "--hypothesis-show-statistics",
            "-p",
            "no:cacheprovider",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _example_counts(output: str) -> list[int]:
    """Passing-example counts from the GENERATE phase only.

    max_examples caps generation, so that is the phase the profile controls.
    Hypothesis also emits a "reuse phase" block with its own "passing examples"
    line whenever its database holds a stored counterexample to replay, which
    happens after any fuzz test has ever failed on this machine. Counting every
    matching line would then mix a small replay count in with the generated one
    and fail these assertions for a reason unrelated to the profile.
    """
    counts = []
    in_generate_phase = False
    for line in output.splitlines():
        stripped = line.strip()
        if "phase (" in stripped:
            in_generate_phase = "generate phase" in stripped
        elif in_generate_phase and "passing examples" in stripped:
            counts.append(int(stripped.split()[1]))
    return counts


def test_env_var_sets_the_example_count():
    """The load-bearing half: a set variable must change the depth.

    Uses a value that is neither Hypothesis's default (100) nor either CI value
    (200, 1000), so a pass cannot come from the variable being ignored.
    """
    counts = _example_counts(_run_fuzz_with("13"))

    assert counts, "no Hypothesis statistics were reported"
    assert all(c == 13 for c in counts), f"expected every test at 13 examples, got {counts}"


def test_unset_env_var_falls_back_to_the_hypothesis_default():
    """The narrowness half: the profile must not pin a value when unset.

    Without this, a conftest that hardcoded a depth would pass the test above
    while making the variable irrelevant again in the other direction.
    """
    counts = _example_counts(_run_fuzz_with(None))

    assert counts, "no Hypothesis statistics were reported"
    assert all(c == 100 for c in counts), f"expected the default 100, got {counts}"
