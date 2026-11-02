"""environment.yml's conda `pip` pin is the version the hash-locked bootstrap pins.

`conda env create -f environment.yml` installs the conda `pip` first and then runs
the `pip:` block with it, so that pin is the pip that installs requirements.txt.
It was `pip=24.0`, which pip-audit 2.10.1 reports six advisories against
(PYSEC-2026-196, -1795, -1796, -2875, -2876, -3721), fixed in releases up to
26.2. Nothing watched it: Dependabot has no conda ecosystem in
.github/dependabot.yml, and Scorecard's vulnerability check on main reported only
an unrelated advisory while this pin was live.

environments/requirements-pip-bootstrap.txt is the repo's single hash-locked `pip`
pin, the one the Docker images, singularity.def, release.yml and the CI jobs
install first, and it is a `requirements*.txt` file under a Dependabot `pip`
directory, which GitHub's dependency graph reads. Tying the conda pin to it means
the conda path moves when that file does, instead of staying on whatever release
it was first written with.

FAILS IF: environment.yml stops pinning `pip` to one exact version, or pins a
version different from the bootstrap lock's.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_YML = REPO_ROOT / "environment.yml"
BOOTSTRAP = REPO_ROOT / "environments" / "requirements-pip-bootstrap.txt"

# A conda match spec for pip: `pip`, `pip=X`, `pip==X`, `pip>=X`, `pip X` ...
_CONDA_PIP = re.compile(r"pip(?![\w.-])\s*(?P<spec>.*)")
_EXACT = re.compile(r"(?:==?)\s*(?P<version>\d+(?:\.\d+)*)")
_BOOTSTRAP_PIN = re.compile(r"pip==(?P<version>\S+)")


def _conda_pip_specs() -> list[str]:
    data = yaml.safe_load(ENV_YML.read_text(encoding="utf-8"))
    specs = []
    for dep in data["dependencies"]:
        # The `pip:` block is a mapping, not a conda spec, and is skipped.
        if isinstance(dep, str):
            match = _CONDA_PIP.fullmatch(dep.strip())
            if match:
                specs.append(match.group("spec").strip())
    return specs


def _bootstrap_pip_version() -> str:
    versions = []
    for line in BOOTSTRAP.read_text(encoding="utf-8").splitlines():
        match = _BOOTSTRAP_PIN.match(line.strip())
        if match:
            versions.append(match.group("version"))
    assert len(versions) == 1, f"{BOOTSTRAP.name} must pin pip exactly once, found {versions}"
    return versions[0]


def test_conda_env_pins_pip_to_one_exact_version() -> None:
    specs = _conda_pip_specs()
    assert len(specs) == 1, f"environment.yml must list conda `pip` once, found {specs}"
    assert _EXACT.fullmatch(specs[0]), (
        f"environment.yml's conda pip spec {specs[0]!r} is not one exact version. The "
        "conda pip is the pip that runs the hash-checked `pip:` block, so an open "
        "spec lets conda pick any release, including one with known advisories."
    )


def test_conda_env_pip_equals_the_hash_locked_bootstrap_pin() -> None:
    specs = _conda_pip_specs()
    assert len(specs) == 1
    match = _EXACT.fullmatch(specs[0])
    assert match is not None
    assert match.group("version") == _bootstrap_pip_version(), (
        f"environment.yml pins conda pip {match.group('version')}, but "
        f"environments/requirements-pip-bootstrap.txt pins pip=={_bootstrap_pip_version()}. "
        "Change environment.yml's `- pip=` line to the bootstrap lock's version: that "
        "lock is read by GitHub's dependency graph, and no configured scanner reads a "
        "conda pin."
    )
