#!/usr/bin/env python3
"""Report whether tracked SHA256 records reproduce from Git blobs."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import textwrap
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath

PATTERNS = ("*provenance*.json", "*.provenance.json", "*model_artifact_checksums.json")
DIGEST = re.compile(r"[0-9a-fA-F]{64}")
CATEGORIES = ("PORTABLE", "WINDOWS_ONLY", "MISMATCH", "MISSING", "UNRESOLVED")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=repo, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _paired_path(node: dict, key: str) -> str | None:
    stem = key.removesuffix("_sha256").removesuffix("_checksum")
    names = [stem, f"{stem}_path", f"{stem}_file"]
    if key == "sha256":
        names = ["artifact", "output_path", "output_file", "output"]
    return next((node[name] for name in names if isinstance(node.get(name), str)), None)


def extract_records(manifest: str, payload: object) -> list[dict[str, str | bool | None]]:
    """Extract every SHA256-looking value and its described path, if resolvable."""
    records: list[dict[str, str | bool | None]] = []

    def add(recorded: str, source: str, path: str | None, relative: bool = False) -> None:
        reason = None if path else f"no subject path paired with {source}"
        records.append(
            {
                "manifest": manifest,
                "path": path,
                "recorded": recorded.lower(),
                "source": source,
                "relative": relative,
                "reason": reason,
            }
        )

    def walk(node: object, location: str = "") -> None:
        if isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{location}[{index}]")
            return
        if not isinstance(node, dict):
            return
        artifacts = node.get("artifacts")
        if isinstance(artifacts, dict):
            for path, entry in artifacts.items():
                if isinstance(entry, dict) and isinstance(entry.get("sha256"), str):
                    if DIGEST.fullmatch(entry["sha256"]):
                        add(entry["sha256"], f"{location}.artifacts[{path}]", path, True)
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            for path, recorded in inputs.items():
                if isinstance(recorded, str) and DIGEST.fullmatch(recorded):
                    add(recorded, f"{location}.inputs[{path}]", path)
        for key, value in node.items():
            source = f"{location}.{key}".strip(".")
            if key not in {"artifacts", "inputs"} and isinstance(value, str):
                if DIGEST.fullmatch(value):
                    add(value, source, _paired_path(node, key))
            elif key not in {"artifacts", "inputs"}:
                walk(value, source)

    walk(payload)
    return records


def _resolve(record: dict, tracked: set[str]) -> tuple[str | None, str | None]:
    raw = record["path"]
    if not isinstance(raw, str):
        return None, str(record["reason"])
    normalized = raw.replace("\\", "/").removeprefix("./")
    if PurePosixPath(normalized).is_absolute() or PureWindowsPath(raw).is_absolute():
        return None, "absolute subject path is not repository-relative"
    if ".." in PurePosixPath(normalized).parts:
        return None, "subject path escapes the repository"
    parent = PurePosixPath(str(record["manifest"])).parent
    relative = (parent / normalized).as_posix()
    if record["relative"] or ("/" not in normalized and relative in tracked):
        return relative, None
    return normalized, None


def classify(recorded: str, blob: str | None, worktree: str | None) -> str:
    if blob is None:
        return "MISSING"
    if recorded == blob:
        return "PORTABLE"
    if worktree is not None and recorded == worktree:
        return "WINDOWS_ONLY"
    return "MISMATCH"


def scan_repository(repo: Path) -> list[dict[str, str | None]]:
    raw = _git(repo, "ls-files", "-z").stdout.decode("utf-8")
    tracked = {path for path in raw.split("\0") if path}
    manifests = sorted(
        path for path in tracked if any(fnmatch.fnmatch(path, pattern) for pattern in PATTERNS)
    )
    rows: list[dict[str, str | None]] = []
    for manifest in manifests:
        result = _git(repo, "cat-file", "blob", f"HEAD:{manifest}")
        if result.returncode:
            rows.append({"manifest": manifest, "path": None, "recorded": None, "blob": None,
                         "worktree": None, "eol": None, "verdict": "UNRESOLVED",
                         "reason": "manifest is not present in HEAD"})
            continue
        try:
            records = extract_records(manifest, json.loads(result.stdout))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            rows.append({"manifest": manifest, "path": None, "recorded": None, "blob": None,
                         "worktree": None, "eol": None, "verdict": "UNRESOLVED",
                         "reason": f"invalid JSON: {exc}"})
            continue
        for record in records:
            path, reason = _resolve(record, tracked)
            if path is None:
                rows.append({**record, "blob": None, "worktree": None, "eol": None,
                             "verdict": "UNRESOLVED", "reason": reason})
                continue
            blob_result = _git(repo, "cat-file", "blob", f"HEAD:{path}")
            blob = _sha256(blob_result.stdout) if blob_result.returncode == 0 else None
            target = repo / Path(path)
            worktree = _sha256(target.read_bytes()) if target.is_file() else None
            attr_result = _git(repo, "check-attr", "eol", "--", path)
            attr = attr_result.stdout.decode("utf-8", "replace").strip().rsplit(": ", 1)[-1]
            rows.append({**record, "path": path, "blob": blob, "worktree": worktree,
                         "eol": attr or None, "verdict": classify(record["recorded"], blob, worktree),
                         "reason": None})
    return rows


def print_human(rows: list[dict[str, str | None]]) -> None:
    counts = Counter(row["verdict"] for row in rows)
    print("SUMMARY " + " ".join(f"{name}={counts[name]}" for name in CATEGORIES))
    for row in rows:
        fields = [(f"{row['verdict']} ", f"path={row.get('path') or '<unresolved>'}")]
        fields.extend(
            ("  ", f"{key}={row[key]}")
            for key in ("manifest", "recorded", "blob", "worktree", "eol", "reason")
            if row.get(key) is not None
        )
        for prefix, value in fields:
            for line in textwrap.wrap(
                prefix + value, width=119, subsequent_indent="  ", break_long_words=True
            ):
                print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--strict", action="store_true", help="fail on non-portable digests")
    args = parser.parse_args(argv)
    root = _git(Path.cwd(), "rev-parse", "--show-toplevel")
    if root.returncode:
        parser.error("not inside a Git worktree")
    rows = scan_repository(Path(root.stdout.decode().strip()))
    if args.json:
        counts = Counter(row["verdict"] for row in rows)
        print(json.dumps({"counts": {name: counts[name] for name in CATEGORIES}, "rows": rows},
                         indent=2, sort_keys=True))
    else:
        print_human(rows)
    return int(args.strict and any(row["verdict"] in {"WINDOWS_ONLY", "MISMATCH"} for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
