#!/usr/bin/env python
"""Keep .coveragerc.library's omit list in sync with the source tree.

The OpenSSF Silver ``test_statement_coverage80`` criterion is measured on
SESTRAV's importable *library* surface. Modules that expose a command-line
entry point (``if __name__ == "__main__"``) are executable research/pipeline
*scripts*; they are validated by integration tests and the CI data/benchmark
gates rather than by unit statement coverage, so they are omitted from the
library-coverage config.

This script discovers those scripts mechanically (presence of a ``__main__``
guard) so the omit list is objective and reproducible rather than hand-picked.

Usage:
    python tools/check_library_coverage.py            # list executable scripts
    python tools/check_library_coverage.py --check     # exit 1 if config drifts
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE_ROOTS = ("src", "functions")
CONFIG = REPO_ROOT / ".coveragerc.library"
MAIN_GUARD = re.compile(r"if __name__ == ['\"]__main__['\"]")


def discover_scripts() -> list[str]:
    """Return posix paths of source modules that have a __main__ entry point."""
    scripts: list[str] = []
    for root in SOURCE_ROOTS:
        for path in (REPO_ROOT / root).rglob("*.py"):
            if path.name == "__init__.py":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if MAIN_GUARD.search(text):
                scripts.append(path.relative_to(REPO_ROOT).as_posix())
    return sorted(scripts)


def config_omits() -> set[str]:
    """Parse the omit entries currently declared in .coveragerc.library."""
    omits: set[str] = set()
    in_omit = False
    for line in CONFIG.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("omit"):
            in_omit = True
            continue
        if in_omit:
            if stripped.startswith("[") or (line and not line[0].isspace()):
                break
            # Only compare concrete module paths; skip glob boilerplate such as
            # "*/tests/*" and "*/__init__.py".
            if stripped and stripped.endswith(".py") and "*" not in stripped:
                omits.add(stripped)
    return omits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the config is out of sync")
    args = parser.parse_args()

    scripts = discover_scripts()
    if not args.check:
        for s in scripts:
            print(s)
        return 0

    declared = config_omits()
    missing = set(scripts) - declared
    stale = declared - set(scripts)
    if missing or stale:
        if missing:
            print("Scripts missing from .coveragerc.library omit list:")
            for m in sorted(missing):
                print(f"  + {m}")
        if stale:
            print("Stale omit entries (no __main__ guard found):")
            for s in sorted(stale):
                print(f"  - {s}")
        print("\nRegenerate the omit list to restore objectivity.")
        return 1
    print(f".coveragerc.library omit list is in sync ({len(scripts)} executable scripts).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
