"""Every Python process the pipeline spawns must be the interpreter running Snakemake.

A bare ``python`` in a ``shell:`` string or a ``run:`` block is resolved through
PATH when the job runs. On a machine with no ``python`` on PATH (a stock Linux
box with no activated environment) the job dies with status 127; where some
other ``python`` comes first on PATH, the job silently runs under the wrong
interpreter. Both CI Snakemake legs are ``--dry-run`` and never execute a job
body, so neither can see this. This test reads ``pipeline.smk`` and every file
it includes, and fails on any string literal whose command word is a bare
``python`` or ``python3``.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY = REPO_ROOT / "pipeline.smk"

# A string literal, optionally r/f-prefixed, whose content starts with a bare
# ``python`` or ``python3`` command word followed by whitespace.
BARE_PYTHON = re.compile(r"""(?<![\w.])[rRfF]{0,2}["']\s*python3?(?=\s)""")
INCLUDE = re.compile(r"""^\s*include:\s*["']([^"']+)["']""", re.MULTILINE)


def _workflow_files(entry: Path) -> list[Path]:
    seen: list[Path] = []
    pending = [entry.resolve()]
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.append(path)
        for name in INCLUDE.findall(path.read_text(encoding="utf-8")):
            pending.append((path.parent / name).resolve())
    return seen


def _bare_spawns(path: Path) -> list[str]:
    hits = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        if BARE_PYTHON.search(line):
            hits.append(f"{path.name} line {number}: {line.strip()}")
    return hits


def test_the_pattern_matches_the_defect_and_not_the_fix():
    assert BARE_PYTHON.search('        "python scripts/stage.py > {log} 2>&1"')
    assert BARE_PYTHON.search('        cmd = f"python src/verify/tool.py {input.targets}"')
    assert BARE_PYTHON.search("        'python3 -m src.module'")
    assert not BARE_PYTHON.search('        "\\"{sys.executable}\\" scripts/stage.py"')
    assert not BARE_PYTHON.search("""        cmd = f'"{sys.executable}" src/verify/tool.py'""")
    assert not BARE_PYTHON.search('    language = "python"')


def test_the_include_chain_is_followed():
    names = {path.name for path in _workflow_files(ENTRY)}
    assert {"pipeline.smk", "standardize_outputs.smk"} <= names


def test_no_rule_spawns_a_bare_python():
    hits = [hit for path in _workflow_files(ENTRY) for hit in _bare_spawns(path)]
    assert hits == []
