"""Behaviour of build_release_bundle itself.

tests/test_release_bundle_canonical_files.py checks that the canonical inputs
exist in the clone. Nothing exercised the bundler: what it records for each
input, and that it refuses to build when an input is missing. These tests do,
from inputs written to a temporary directory, so they do not depend on the
tracked results/ files.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from src.release_bundle import build_release_bundle


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _outputs(out_dir: Path) -> list[str]:
    if not out_dir.exists():
        return []
    return sorted(p.name for p in out_dir.iterdir() if p.suffix in {".json", ".zip"})


def test_build_refuses_missing_inputs_and_writes_no_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every missing input is named, and no manifest or archive is written.

    The refusal must come from the bundler's own check, before anything is
    written. Without that check the archive step still fails on the first
    missing file, but only after a manifest and a partial archive have been
    written, and with an error that names only that first missing input.
    """
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "present_a.txt", b"a\n")
    _write(tmp_path / "present_b.txt", b"b\n")
    out_dir = tmp_path / "bundle"

    with pytest.raises(
        FileNotFoundError,
        match="Missing required files for release bundle: missing_1.txt, missing_2.txt",
    ):
        build_release_bundle(
            output_dir=str(out_dir),
            files=["present_a.txt", "missing_1.txt", "present_b.txt", "missing_2.txt"],
        )

    assert _outputs(out_dir) == []


def test_build_records_the_size_and_sha256_of_each_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest and archive describe exactly the inputs given.

    Expected sizes and digests come from the input bytes themselves (len and
    hashlib.sha256), not from any output of the bundler. The output directory
    already exists, because a release run may reuse it.
    """
    monkeypatch.chdir(tmp_path)
    inputs = {
        "results/metrics.csv": b"metric,value\nauc_pr,0.5\n",
        "notes.md": b"# notes\n",
    }
    for rel, data in inputs.items():
        _write(tmp_path / rel, data)
    out_dir = tmp_path / "bundle"
    out_dir.mkdir()

    paths = build_release_bundle(output_dir=str(out_dir), bundle_name="t", files=list(inputs))

    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["bundle_name"] == "t"
    assert manifest["file_count"] == len(inputs)
    assert manifest["files"] == [
        {"path": rel, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for rel, data in inputs.items()
    ]

    with zipfile.ZipFile(paths["archive"]) as archive:
        assert sorted(archive.namelist()) == sorted([*inputs, Path(paths["manifest"]).name])
        for rel, data in inputs.items():
            assert archive.read(rel) == data
