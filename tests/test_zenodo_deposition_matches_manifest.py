"""The Zenodo deposition guide must quote the digests the DOI manifest pins.

docs/zenodo_deposition.md is the human copy of the deposit: a table of the
three files with their SHA-256, and a ``sha256sum -c`` block to run before
uploading. docs/zenodo_manifest_v5.json is the machine copy, and
scripts/check_digest_portability.py already recomputes the manifest's digests
from the tracked files. Nothing compared the guide with either, so when #540
changed the schema file and the manifest was corrected, the guide kept the old
schema digest and its own verification step failed on every clean clone.

These tests tie the guide to the manifest, path by path, in both places the
guide states a digest. With the manifest gated against the bytes, that closes
the loop without hashing the dataset a second time here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE = REPO_ROOT / "docs" / "zenodo_deposition.md"
MANIFEST = REPO_ROOT / "docs" / "zenodo_manifest_v5.json"

DIGEST = r"[0-9a-f]{64}"
TABLE_ROW = re.compile(r"^\| `(?P<path>[^`]+)` \|.*\| `(?P<digest>" + DIGEST + r")` \|$")
CHECK_LINE = re.compile(r"^(?P<digest>" + DIGEST + r")  (?P<path>\S+)$")


def manifest_digests() -> dict[str, str]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {entry["path"]: entry["sha256"] for entry in payload["files"]}


def guide_lines() -> list[str]:
    return GUIDE.read_text(encoding="utf-8").splitlines()


def table_digests() -> dict[str, str]:
    found = {}
    for line in guide_lines():
        match = TABLE_ROW.match(line)
        if match:
            found[match["path"]] = match["digest"]
    return found


def check_block_digests() -> dict[str, str]:
    lines = guide_lines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "sha256sum -c <<'EOF'"]
    assert len(starts) == 1, f"expected one sha256sum -c heredoc in the guide, found {len(starts)}"
    found = {}
    for line in lines[starts[0] + 1 :]:
        if line.strip() == "EOF":
            break
        match = CHECK_LINE.match(line)
        assert match, f"unparseable line inside the sha256sum -c block: {line!r}"
        found[match["path"]] = match["digest"]
    else:
        raise AssertionError("the sha256sum -c block in the guide has no closing EOF")
    return found


def test_the_manifest_names_the_three_deposit_files():
    assert set(manifest_digests()) == {
        "data/immunogenicity_dataset_v5.csv",
        "data/immunogenicity_dataset_v5_schema.json",
        "data/immunogenicity_dataset_v5_provenance.json",
    }


def test_the_guide_table_quotes_the_manifest_digests():
    assert table_digests() == manifest_digests()


def test_the_guide_verification_block_checks_the_manifest_digests():
    assert check_block_digests() == manifest_digests()
