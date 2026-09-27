from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MEMBERS = {
    "src/verify/mhc_pseudo_sequences.json",
    "src/verify/targets.json",
}
# What the [project] and [tool.setuptools] tables in pyproject.toml read.
BUILD_INPUT_FILES = ("pyproject.toml", "README.md", "LICENSE")
BUILD_INPUT_PACKAGES = ("sestrav", "src", "functions")


def _stage_build_inputs(dest: Path) -> None:
    """Copy the build inputs into ``dest`` so every build starts clean.

    An in-tree build reuses whatever an earlier in-tree build left behind, and
    either a stale ``build/lib`` or a stale ``sestrav.egg-info/SOURCES.txt``
    alone carries both JSON files into the wheel after the package-data line
    is deleted. Building in the checkout would therefore make this guard
    vacuous from its second run onward, and would also write those
    directories into the checkout while the rest of the suite runs.
    """
    for name in BUILD_INPUT_FILES:
        shutil.copy2(REPO_ROOT / name, dest / name)
    skip = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", "build")
    for package in BUILD_INPUT_PACKAGES:
        shutil.copytree(REPO_ROOT / package, dest / package, ignore=skip)


def test_built_wheel_carries_verify_json(tmp_path: Path) -> None:
    source = tmp_path / "source"
    wheel_dir = tmp_path / "wheel"
    source.mkdir()
    _stage_build_inputs(source)
    env = os.environ.copy()
    env.update({"PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            ".",
        ],
        cwd=source,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1, [wheel.name for wheel in wheels]
    with zipfile.ZipFile(wheels[0]) as archive:
        file_members = {item.filename for item in archive.infolist() if not item.is_dir()}
        missing = REQUIRED_MEMBERS - file_members
        assert not missing, f"built wheel is missing package data: {sorted(missing)}"
        for member in sorted(REQUIRED_MEMBERS):
            payload = json.loads(archive.read(member))
            assert isinstance(payload, dict) and payload, f"{member} must be a non-empty object"
