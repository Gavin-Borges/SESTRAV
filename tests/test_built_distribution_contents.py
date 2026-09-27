from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MEMBERS = {
    "src/verify/mhc_pseudo_sequences.json",
    "src/verify/targets.json",
}


def test_built_wheel_carries_verify_json(tmp_path: Path) -> None:
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
            str(tmp_path),
            ".",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, [wheel.name for wheel in wheels]
    with zipfile.ZipFile(wheels[0]) as archive:
        file_members = {item.filename for item in archive.infolist() if not item.is_dir()}
        missing = REQUIRED_MEMBERS - file_members
        assert not missing, f"built wheel is missing package data: {sorted(missing)}"
        for member in sorted(REQUIRED_MEMBERS):
            payload = json.loads(archive.read(member))
            assert isinstance(payload, dict) and payload, f"{member} must be a non-empty object"
