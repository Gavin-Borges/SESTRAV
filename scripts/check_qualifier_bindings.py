#!/usr/bin/env python3
"""Check that published values retain the qualifiers that bound their meaning.

The tracked JSON config supplies values, exact carrier anchors, accepted qualifier
regexes, line windows, and per-entry justified exemptions. A violation ratchet lets
the gate land before every existing carrier is repaired; ``--strict`` ignores it.
``--update`` refreshes review metadata and SETS each ceiling to the measured count,
raising it as readily as lowering it; the declared ceilings are pinned in
tests/test_check_qualifier_bindings.py, so a raise fails the suite.

Only TRACKED files are read when the root is a work tree's top level (the default):
``git ls-files`` supplies the candidates for ``scan_globs``; a registered carrier that git
does not track counts as missing. A gitignored or untracked file in a working
checkout therefore cannot change the result. Otherwise (outside a work tree, as
in a ``git archive`` export or a test fixture, or at a root below the top level)
the filesystem is read.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("docs/qualifier_bindings.json")
DEFAULT_EXCLUDED_PREFIXES = (".git/", "_local/", ".claude/", ".agents/", ".cursor/")


class ConfigError(ValueError):
    """Raised when the qualifier binding config is malformed."""


@dataclass(frozen=True)
class Occurrence:
    line: int
    text: str


@dataclass
class BindingResult:
    binding_id: str
    carriers_checked: set[str]
    violations: list[str]
    observations: dict[tuple[str, str], dict[str, Any]]


@dataclass
class AuditResult:
    bindings_checked: int
    carriers_checked: int
    violations: list[str]
    over_ceiling: bool
    per_binding: dict[str, BindingResult]


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if raw.get("version") != 1 or not isinstance(raw.get("bindings"), list):
        raise ConfigError("config must have version 1 and a bindings array")
    return raw


def _validate_binding(binding: dict[str, Any]) -> None:
    required = {
        "id",
        "value",
        "decimals",
        "qualifier_patterns",
        "window_lines",
        "carriers",
        "exemptions",
        "scan_globs",
        "violation_ceiling",
    }
    missing = sorted(required - binding.keys())
    if missing:
        raise ConfigError(f"binding missing required keys: {', '.join(missing)}")

    binding_id = binding["id"]
    if not isinstance(binding_id, str) or not binding_id.strip():
        raise ConfigError("binding id must be a non-empty string")
    decimals = binding["decimals"]
    value = binding["value"]
    if not isinstance(decimals, int) or decimals < 0:
        raise ConfigError(f"{binding_id}: decimals must be a non-negative integer")
    if not isinstance(value, str) or not re.fullmatch(rf"[+-]?\d+\.\d{{{decimals}}}", value):
        raise ConfigError(f"{binding_id}: value does not match decimals={decimals}")
    if not isinstance(binding["window_lines"], int) or binding["window_lines"] < 0:
        raise ConfigError(f"{binding_id}: window_lines must be a non-negative integer")
    if not isinstance(binding["violation_ceiling"], int) or binding["violation_ceiling"] < 0:
        raise ConfigError(f"{binding_id}: violation_ceiling must be non-negative")
    if not binding["qualifier_patterns"]:
        raise ConfigError(f"{binding_id}: qualifier_patterns must not be empty")
    for pattern in binding["qualifier_patterns"]:
        try:
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
        except re.error as exc:
            raise ConfigError(f"{binding_id}: invalid qualifier regex {pattern!r}: {exc}") from exc

    carrier_paths: set[str] = set()
    for carrier in binding["carriers"]:
        path = carrier.get("path", "")
        anchor = carrier.get("anchor_pattern", "")
        if not isinstance(path, str) or not path or not isinstance(anchor, str) or not anchor:
            raise ConfigError(f"{binding_id}: every carrier needs path and anchor_pattern")
        if path in carrier_paths:
            raise ConfigError(f"{binding_id}: duplicate carrier path {path}")
        carrier_paths.add(path)
        try:
            re.compile(anchor, re.IGNORECASE)
        except re.error as exc:
            raise ConfigError(f"{binding_id}: invalid anchor regex {anchor!r}: {exc}") from exc

    exempt_paths: set[str] = set()
    for exemption in binding["exemptions"]:
        path = exemption.get("path", "")
        note = exemption.get("note", "")
        if not isinstance(path, str) or not path:
            raise ConfigError(f"{binding_id}: every exemption needs a path")
        if not isinstance(note, str) or not note.strip():
            raise ConfigError(f"{binding_id}: exemption {path} needs a non-empty note")
        if path in exempt_paths:
            raise ConfigError(f"{binding_id}: duplicate exemption path {path}")
        exempt_paths.add(path)
    overlap = sorted(carrier_paths & exempt_paths)
    if overlap:
        raise ConfigError(f"{binding_id}: carrier/exemption overlap: {', '.join(overlap)}")


def _inside_inline_code(line: str, start: int) -> bool:
    return line[:start].count("`") % 2 == 1


def _inside_path_token(line: str, start: int, end: int) -> bool:
    token_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./\\+-")
    left = start
    right = end
    while left > 0 and line[left - 1] in token_chars:
        left -= 1
    while right < len(line) and line[right] in token_chars:
        right += 1
    token = line[left:right]
    return "/" in token or "\\" in token


def _occurrences(lines: list[str], value: str) -> list[Occurrence]:
    # A trailing "." ends a match only when a digit follows it, so a value that
    # ends a sentence ("... was 0.6458.") is still read as a claim.
    pattern = re.compile(rf"(?<![\d.]){re.escape(value)}(?!\d)(?!\.\d)")
    found: list[Occurrence] = []
    for number, line in enumerate(lines, start=1):
        for match in pattern.finditer(line):
            if _inside_inline_code(line, match.start()):
                continue
            if _inside_path_token(line, match.start(), match.end()):
                continue
            found.append(Occurrence(number, line))
    return found


def _window(lines: list[str], line: int, radius: int) -> str:
    start = max(0, line - radius - 1)
    end = min(len(lines), line + radius)
    return "\n".join(lines[start:end])


def _qualified(window: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, window, re.IGNORECASE | re.DOTALL) for pattern in patterns)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes] | None:
    # git exports GIT_DIR / GIT_WORK_TREE / GIT_INDEX_FILE into hook subprocesses
    # such as pre-push, and they outrank -C; stripped so -C alone picks the repo,
    # as tools/check_lockfile_freshness.py does.
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")
    }
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            check=False,
            env=env,
        )
    except OSError:
        return None


def _tracked_files(root: Path) -> set[str] | None:
    """Return the paths git tracks under ``root``, or None outside a work tree.

    git is trusted only when its own top level resolves to ``root``: run from a
    directory nested inside some other repository, git walks upward and would
    list that repository instead. A root that carries its own ``.git`` entry but
    cannot be listed is an error, never a silent fall back to the filesystem.
    """
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is not None and top.returncode == 0:
        resolved = top.stdout.decode("utf-8", "replace").strip()
        if resolved and Path(resolved).resolve() == root.resolve():
            listed = _git(root, "ls-files", "-z")
            if listed is None or listed.returncode != 0:
                raise ConfigError(f"git ls-files failed in {root}")
            return {entry for entry in listed.stdout.decode("utf-8").split("\0") if entry}
    if (root / ".git").exists():
        raise ConfigError(f"{root} has a .git entry but git cannot list its tracked files")
    return None


def _glob_parts_match(parts: list[str], pattern: list[str]) -> bool:
    if not pattern:
        return not parts
    head, rest = pattern[0], pattern[1:]
    if head == "**":
        return any(_glob_parts_match(parts[index:], rest) for index in range(len(parts) + 1))
    return (
        bool(parts) and fnmatch.fnmatchcase(parts[0], head) and _glob_parts_match(parts[1:], rest)
    )


def _glob_matches(relative: str, pattern: str) -> bool:
    """Match a POSIX relative path the way ``Path.glob`` selects files.

    ``*`` and ``?`` stay inside one path segment and ``**`` spans zero or more
    whole directories. Case-sensitive, as on the Linux CI runner.
    """
    return _glob_parts_match(relative.split("/"), pattern.split("/"))


def _scanned_paths(root: Path, globs: list[str], tracked: set[str] | None) -> set[str]:
    paths: set[str] = set()
    if tracked is not None:
        for relative in tracked:
            if relative.startswith(DEFAULT_EXCLUDED_PREFIXES):
                continue
            if not any(_glob_matches(relative, pattern) for pattern in globs):
                continue
            if (root / relative).is_file():
                paths.add(relative)
        return paths
    for pattern in globs:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative.startswith(DEFAULT_EXCLUDED_PREFIXES):
                continue
            paths.add(relative)
    return paths


def _audit_binding(
    root: Path, binding: dict[str, Any], tracked: set[str] | None = None
) -> BindingResult:
    _validate_binding(binding)
    binding_id = binding["id"]
    value = binding["value"]
    radius = binding["window_lines"]
    patterns = binding["qualifier_patterns"]
    carriers_checked: set[str] = set()
    violations: list[str] = []
    observations: dict[tuple[str, str], dict[str, Any]] = {}

    configured = {entry["path"] for entry in binding["carriers"]}
    exempt = {entry["path"] for entry in binding["exemptions"]}

    for carrier in binding["carriers"]:
        relative = carrier["path"]
        path = root / relative
        carriers_checked.add(relative)
        if not path.is_file() or (tracked is not None and relative not in tracked):
            violations.append(f"{relative}:0: {binding_id}: carrier is missing")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        anchor = re.compile(carrier["anchor_pattern"], re.IGNORECASE)
        selected_by_line = {
            item.line: item for item in _occurrences(lines, value) if anchor.search(item.text)
        }
        selected = list(selected_by_line.values())
        if len(selected) != 1:
            violations.append(
                f"{relative}:0: {binding_id}: anchor matched {len(selected)} occurrences; expected 1"
            )
            continue
        occurrence = selected[0]
        text_window = _window(lines, occurrence.line, radius)
        observations[(binding_id, relative)] = {
            "line": occurrence.line,
            "window_sha256": hashlib.sha256(text_window.encode("utf-8")).hexdigest(),
        }
        if not _qualified(text_window, patterns):
            violations.append(
                f"{relative}:{occurrence.line}: {binding_id}: value {value} lacks a required "
                f"qualifier within {radius} line(s)"
            )

    for relative in sorted(_scanned_paths(root, binding["scan_globs"], tracked)):
        if relative in configured or relative in exempt:
            continue
        path = root / relative
        lines = path.read_text(encoding="utf-8").splitlines()
        found = _occurrences(lines, value)
        if not found:
            continue
        carriers_checked.add(relative)
        for occurrence in found:
            if not _qualified(_window(lines, occurrence.line, radius), patterns):
                violations.append(
                    f"{relative}:{occurrence.line}: {binding_id}: unregistered carrier has "
                    f"unqualified value {value}"
                )

    return BindingResult(binding_id, carriers_checked, violations, observations)


def audit(root: Path, config: dict[str, Any], *, strict: bool = False) -> AuditResult:
    identifiers: set[str] = set()
    per_binding: dict[str, BindingResult] = {}
    all_carriers: set[tuple[str, str]] = set()
    all_violations: list[str] = []
    over_ceiling = False
    tracked = _tracked_files(root)

    for binding in config["bindings"]:
        binding_id = binding.get("id", "")
        if binding_id in identifiers:
            raise ConfigError(f"duplicate binding id {binding_id}")
        identifiers.add(binding_id)
        result = _audit_binding(root, binding, tracked)
        per_binding[binding_id] = result
        all_violations.extend(result.violations)
        all_carriers.update((binding_id, path) for path in result.carriers_checked)
        ceiling = 0 if strict else binding["violation_ceiling"]
        if len(result.violations) > ceiling:
            over_ceiling = True

    return AuditResult(
        bindings_checked=len(config["bindings"]),
        carriers_checked=len(all_carriers),
        violations=all_violations,
        over_ceiling=over_ceiling,
        per_binding=per_binding,
    )


def _write_update(path: Path, config: dict[str, Any], result: AuditResult) -> None:
    for binding in config["bindings"]:
        measured = result.per_binding[binding["id"]]
        binding["violation_ceiling"] = len(measured.violations)
        for carrier in binding["carriers"]:
            observed = measured.observations.get((binding["id"], carrier["path"]))
            if observed is not None:
                carrier["observed"] = observed
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--strict", action="store_true", help="fail on any violation")
    parser.add_argument("--update", action="store_true", help="refresh observations and ceilings")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    try:
        config = _load_config(config_path)
        result = audit(root, config, strict=args.strict)
        if args.update:
            _write_update(config_path, config, result)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    for violation in result.violations:
        print(violation)
    print(
        f"Checked {result.bindings_checked} qualifier binding(s) across "
        f"{result.carriers_checked} carrier(s). Violations: {len(result.violations)}."
    )
    if args.update:
        print(f"Updated {config_path.relative_to(root)} from the current tree.")
        return 0
    return 1 if result.over_ceiling else 0


if __name__ == "__main__":
    raise SystemExit(main())
