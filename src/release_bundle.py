"""
SESTRAV release bundle utility.

Creates a checksum manifest and zip archive for canonical result artifacts so
third parties can verify exact run outputs against a published release asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


CANONICAL_RESULT_FILES = [
    "results/final_validation_report.md",
    "results/h2_tier_a_summary.csv",
    "results/h2_tier_a_summary.md",
    "results/h2_tier_a_fold_metrics.csv",
    "results/gold_standard_validation.csv",
    "results/baseline_comparison.csv",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_release_bundle(
    output_dir: str = "release_artifacts",
    bundle_name: str = "sestrav-results-bundle",
    files: List[str] | None = None,
) -> Dict[str, str]:
    files = files or CANONICAL_RESULT_FILES
    os.makedirs(output_dir, exist_ok=True)

    manifest_rows = []
    missing = []
    for rel in files:
        p = Path(rel)
        if not p.exists():
            missing.append(rel)
            continue
        manifest_rows.append(
            {
                "path": rel.replace("\\", "/"),
                "size_bytes": p.stat().st_size,
                "sha256": _sha256(p),
            }
        )

    if missing:
        raise FileNotFoundError(
            "Missing required files for release bundle: " + ", ".join(missing)
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = Path(output_dir) / f"{bundle_name}-{stamp}.manifest.json"
    zip_path = Path(output_dir) / f"{bundle_name}-{stamp}.zip"

    manifest_obj = {
        "bundle_name": bundle_name,
        "generated_utc": stamp,
        "file_count": len(manifest_rows),
        "files": manifest_rows,
    }
    manifest_path.write_text(json.dumps(manifest_obj, indent=2) + "\n", encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            zf.write(rel, arcname=rel.replace("\\", "/"))
        zf.write(manifest_path, arcname=manifest_path.name)

    return {
        "manifest": str(manifest_path),
        "archive": str(zip_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build canonical SESTRAV release artifact bundle"
    )
    parser.add_argument(
        "--output-dir",
        default="release_artifacts",
        help="Directory where manifest + zip are written",
    )
    parser.add_argument(
        "--bundle-name",
        default="sestrav-results-bundle",
        help="Filename prefix for generated artifacts",
    )
    args = parser.parse_args()

    out = build_release_bundle(
        output_dir=args.output_dir,
        bundle_name=args.bundle_name,
    )
    print(f"[release_bundle] Manifest: {out['manifest']}")
    print(f"[release_bundle] Archive:  {out['archive']}")
