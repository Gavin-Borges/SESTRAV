#!/usr/bin/env python
"""Fail unless every custom semgrep rule still fires exactly where its fixtures say.

The repo-root `.semgrepignore` keeps the fixture files in `semgrep-rules/` out of
the blocking `semgrep-custom` scan, so that scan passes whether or not a rule
still matches anything. A rule broken by an edit, or by a semgrep version that
reads its pattern differently, would leave the job green. This gate closes that
gap: it names the fixtures explicitly, which bypasses `.semgrepignore`, and
compares semgrep's findings with the fixtures' own annotations:

- `# ruleid: <id>` means rule <id> MUST report the next non-blank,
  non-comment line;
- `# ok: <id>` means rule <id> must NOT report it;
- every rule semgrep RAN against the fixtures, from any rule file, must carry
  at least one `ruleid:` annotation, so a new rule or rule file in a language
  the fixtures already cover fails rather than arriving untested. The list is
  semgrep's own (`--time`), not a parse of the YAML, so a rule's YAML style
  cannot hide it.

What this cannot see: semgrep's `--time` list leaves out a rule whose language
matches no fixture, since such a rule never runs. That case belongs to
tests/test_check_semgrep_rule_fixtures.py, which parses every rule file under
`semgrep-rules/`, recursively, and fails on a duplicate rule id or a rule with no
`ruleid:` line anywhere; that test runs in CI's required `test (3.13)` job.
`# ruleid: a, b` declares the same line for both rules.

Exit codes: 0 when findings and annotations agree, 1 when they disagree, 2 when
the check could not run (no fixtures, semgrep failed or skipped a rule,
unreadable output, no rule list).

Usage:
    python scripts/check_semgrep_rule_fixtures.py                 # runs semgrep
    python scripts/check_semgrep_rule_fixtures.py --findings f.json  # offline
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess  # nosec B404 - runs the pinned semgrep with a fixed argv, no shell
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RULES_DIR = REPO_ROOT / "semgrep-rules"
RULE_SUFFIXES = (".yml", ".yaml")
ANNOTATION = re.compile(r"#\s*(ruleid|ok):\s*([A-Za-z0-9_.-]+(?:\s*,\s*[A-Za-z0-9_.-]+)*)")
# semgrep prefixes every rule id with its config path, dotted: `--config semgrep-rules/`
# reports rule `x` as `semgrep-rules.x`. Stripped exactly, so a rule id that itself
# contains a dot survives intact.
CHECK_ID_PREFIX = "semgrep-rules."


@dataclass(frozen=True, order=True)
class Hit:
    rule: str
    path: str
    line: int


def fixture_files(rules_dir: Path | None = None) -> list[Path]:
    """Every file in the rules directory that is not itself a rule file."""
    rules_dir = RULES_DIR if rules_dir is None else rules_dir
    return sorted(p for p in rules_dir.iterdir() if p.is_file() and p.suffix not in RULE_SUFFIXES)


def fixture_annotations(path: Path, root: Path | None = None) -> tuple[set[Hit], set[Hit]]:
    """(expected, forbidden) hits declared by a fixture's ruleid / ok comments."""
    root = REPO_ROOT if root is None else root
    lines = path.read_text(encoding="utf-8").splitlines()
    relative = path.relative_to(root).as_posix()
    expected: set[Hit] = set()
    forbidden: set[Hit] = set()
    for index, text in enumerate(lines):
        match = ANNOTATION.search(text)
        if not match or not text.lstrip().startswith("#"):
            continue
        target = next(
            (
                i
                for i in range(index + 1, len(lines))
                if lines[i].strip() and not lines[i].lstrip().startswith("#")
            ),
            None,
        )
        if target is None:
            continue
        bucket = expected if match.group(1) == "ruleid" else forbidden
        for rule in re.split(r"\s*,\s*", match.group(2)):
            bucket.add(Hit(rule, relative, target + 1))
    return expected, forbidden


def rule_name(check_id: str) -> str:
    return check_id[len(CHECK_ID_PREFIX) :] if check_id.startswith(CHECK_ID_PREFIX) else check_id


def loaded_rules(payload: dict) -> set[str]:
    """Every rule id semgrep loaded, from `semgrep --json --time`'s `time.rules`."""
    entries = (payload.get("time") or {}).get("rules") or []
    names = {rule_name(e if isinstance(e, str) else e["id"]) for e in entries}
    if not names:
        raise ValueError("semgrep reported no loaded rules (time.rules is empty or absent)")
    return names


def findings_from_json(payload: dict, root: Path | None = None) -> set[Hit]:
    """Semgrep --json results as hits; a rule id is reported with a config-path prefix."""
    root = REPO_ROOT if root is None else root
    if payload.get("errors"):
        raise ValueError(f"semgrep reported errors: {payload['errors']}")
    if payload.get("skipped_rules"):
        raise ValueError(f"semgrep skipped rules: {payload['skipped_rules']}")
    hits = set()
    for result in payload.get("results", []):
        rule = rule_name(result["check_id"])
        path = Path(result["path"])
        relative = (
            (path if path.is_absolute() else (root / path)).resolve().relative_to(root).as_posix()
        )
        hits.add(Hit(rule, relative, int(result["start"]["line"])))
    return hits


def compare(
    found: set[Hit], expected: set[Hit], forbidden: set[Hit], loaded: set[str]
) -> list[str]:
    problems = [f"MISSING {h.rule} at {h.path} line {h.line}" for h in sorted(expected - found)]
    problems += [
        f"FIRED ON AN ok: LINE {h.rule} at {h.path} line {h.line}"
        for h in sorted(found & forbidden)
    ]
    problems += [
        f"UNANNOTATED {h.rule} at {h.path} line {h.line}"
        for h in sorted(found - expected - forbidden)
    ]
    annotated = {h.rule for h in expected}
    problems += [f"NO ruleid: FIXTURE for {rule}" for rule in sorted(loaded - annotated)]
    return problems


def run_semgrep(fixtures: list[Path]) -> dict:
    argv = [
        "semgrep",
        "scan",
        "--config",
        str(RULES_DIR),
        "--json",
        "--time",
        "--metrics=off",
        *[str(p) for p in fixtures],
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=REPO_ROOT, check=False)  # nosec B603
    if proc.returncode != 0:
        # semgrep writes the cause into the JSON's `errors`; stderr is mostly a summary box.
        try:
            detail = json.dumps(json.loads(proc.stdout).get("errors"))[:1500]
        except (ValueError, AttributeError):
            detail = proc.stderr.strip()[-500:]
        raise RuntimeError(f"semgrep exited {proc.returncode}: {detail}")
    return json.loads(proc.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--findings", type=Path, help="read semgrep --json output instead of running semgrep"
    )
    args = parser.parse_args(argv)
    fixtures = fixture_files()
    if not fixtures:
        print("ERROR: no fixture files in semgrep-rules/", file=sys.stderr)
        return 2
    expected: set[Hit] = set()
    forbidden: set[Hit] = set()
    for fixture in fixtures:
        e, f = fixture_annotations(fixture)
        expected |= e
        forbidden |= f
    if not expected:
        print("ERROR: the fixtures carry no ruleid: annotations", file=sys.stderr)
        return 2
    try:
        payload = (
            json.loads(args.findings.read_text(encoding="utf-8"))
            if args.findings
            else run_semgrep(fixtures)
        )
        found = findings_from_json(payload)
        loaded = loaded_rules(payload)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"ERROR: could not read semgrep findings: {error}", file=sys.stderr)
        return 2
    problems = compare(found, expected, forbidden, loaded)
    if problems:
        print("ERROR: custom semgrep rules disagree with their fixtures:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(
        f"{len(found)} findings across {len(fixtures)} fixture file(s) match their "
        f"{len(expected)} ruleid: and {len(forbidden)} ok: annotations; "
        f"all {len(loaded)} rules semgrep ran have a ruleid: fixture."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
