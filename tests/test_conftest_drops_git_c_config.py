"""The root conftest.py must keep `git -c` settings away from the test suite.

`git -c name=value` passes its settings to the processes it starts through the
GIT_CONFIG_PARAMETERS environment variable, and git reads the same kind of setting
from GIT_CONFIG_COUNT with GIT_CONFIG_KEY_<n> / GIT_CONFIG_VALUE_<n>. Either form
outranks a repository's local config, so a pytest process that inherits them runs
its throwaway-repo tests under the outer command's settings. Measured 2026-10-01 on
git 2.55.0.windows.3, before this scrub: with
GIT_CONFIG_PARAMETERS="'core.hooksPath'='scripts/hooks'" exported, and again with
the COUNT form, tests/test_prepare_commit_msg_signoff.py failed 11 of its 17 tests.
In ten of the failures the commit carried no sign-off from the test identity; in
the eleventh, the probe hook's marker file was never written, so that hook never ran.
The first test below pins the mechanism: either form outranks the hooks path a
throwaway repo sets in its own config.

The root conftest.py therefore drops those variables at import, beside the git
repository-discovery variables it already dropped. Whether a hook started by
`git -c ... push` inherits them was not measured; either way, pytest imports the
root conftest.py before any test module, so the scrub runs first.

What each test pins:

- test_either_form_outranks_a_throwaway_repos_own_hooks_path: the premise. Both
  forms override a hooks path set in a fresh repo's local config. If git stopped
  honouring them the scrub would guard nothing, and the tests below would not say so.
- test_no_git_c_setting_reaches_a_test: no carrier is present inside a test. This
  holds in any run; under a `git -c` caller it holds only because of the scrub.
- test_a_pytest_run_started_with_them_set_does_not_pass_them_to_tests: the
  production path. A nested pytest run of the test above, started from the repo
  root with both forms set, must pass.
- test_importing_conftest_keeps_only_what_callers_set_on_purpose: the narrowness
  half. GIT_CONFIG_GLOBAL, GIT_CONFIG_SYSTEM and GIT_CONFIG_NOSYSTEM survive the
  import with their values; the carriers do not.

Anti-vacuity: with the new names removed from the scrub in a scratch copy of
conftest.py, the nested-run test and the narrowness test both fail.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# What `git -c core.hooksPath=scripts/hooks` exports, read back from a shell alias
# run under that command on git 2.55.0.windows.3.
PARAMETERS_FORM = {"GIT_CONFIG_PARAMETERS": "'core.hooksPath'='scripts/hooks'"}
COUNT_FORM = {
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "core.hooksPath",
    "GIT_CONFIG_VALUE_0": "scripts/hooks",
}
# Set on purpose by callers to keep a developer's own config files out of git.
KEPT = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}
CARRIER = re.compile(r"GIT_CONFIG_(PARAMETERS|COUNT|KEY_[0-9]+|VALUE_[0-9]+)")

# Importing conftest.py from the repo root re-runs its module-scope scrub, the idiom
# tests/test_hypothesis_profile.py uses for the same file.
_PROBE = (
    "import json, os, conftest\n"
    "print(json.dumps({k: v for k, v in os.environ.items() if k.startswith('GIT_CONFIG')}))"
)


def _without_git_config(environ) -> dict[str, str]:
    return {k: v for k, v in environ.items() if not k.startswith("GIT_CONFIG")}


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not on PATH")
@pytest.mark.parametrize("form", [PARAMETERS_FORM, COUNT_FORM], ids=["parameters", "count"])
def test_either_form_outranks_a_throwaway_repos_own_hooks_path(tmp_path, form):
    env = {
        **_without_git_config(os.environ),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }
    subprocess.run(["git", "init", "-q", str(tmp_path)], env=env, check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "core.hooksPath", "local-hooks"],
        env=env,
        check=True,
    )

    def hooks_path(extra: dict[str, str]) -> str:
        result = subprocess.run(
            ["git", "-C", str(tmp_path), "config", "--get", "core.hooksPath"],
            env={**env, **extra},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    assert hooks_path({}) == "local-hooks"
    assert hooks_path(form) == "scripts/hooks"


def test_no_git_c_setting_reaches_a_test():
    assert sorted(name for name in os.environ if CARRIER.fullmatch(name)) == []


def test_a_pytest_run_started_with_them_set_does_not_pass_them_to_tests():
    node = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    node += "::test_no_git_c_setting_reaches_a_test"
    env = {**os.environ, **PARAMETERS_FORM, **COUNT_FORM}
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", node, "-p", "no:cacheprovider", "-o", "addopts=", "-q"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout, result.stdout


def test_importing_conftest_keeps_only_what_callers_set_on_purpose():
    env = {**_without_git_config(os.environ), **PARAMETERS_FORM, **COUNT_FORM, **KEPT}
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == KEPT
