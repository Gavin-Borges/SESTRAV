"""Every --flag a workflow rule passes to a repository script must be one that script registers.

A rule's ``shell:`` string is only parsed by the script's argparse when the job
runs. Both CI Snakemake legs are ``--dry-run`` and never execute a job body, so a
misspelled flag passes CI and the job dies with "unrecognized arguments" the first
time anyone runs it. Rule ``generate_hard_decoys`` shipped that way, passing
``--num_decoys`` to a script that registers ``--num-decoys``.

This test reads ``pipeline.smk`` and every file it includes, joins the string
literals of each ``shell:`` and ``run:`` block, finds the repository script the
block runs (a path under scripts/, src/, tools/ or functions/, or a ``-m`` module),
and requires every ``--flag`` in the block to be registered by an
``add_argument`` call in that script, spelled exactly. Exactly, not as an
abbreviation: argparse accepts a unique prefix, so ``--allele`` reached
``--alleles`` only until any other option starting ``--allele`` is added.

Not covered, by construction: a flag supplied through a ``{placeholder}``
(its value is decided at run time), and a script that registers its options
somewhere other than its own file.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY = REPO_ROOT / "pipeline.smk"

INCLUDE = re.compile(r"""^\s*include:\s*["']([^"']+)["']""", re.MULTILINE)
BLOCK_START = re.compile(r"^(\s*)(shell|run):\s*$")
STRING = re.compile(r"""[rRfF]{0,2}("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')""")
PLACEHOLDER = re.compile(r"\{[^{}]*\}")
TARGET = re.compile(
    r"(?:^|[\s\"'])((?:scripts|src|tools|functions)/[\w/]+\.py)"
    r"|-m\s+((?:scripts|src|tools|functions)(?:\.\w+)+)"
)
FLAG = re.compile(r"(?<![\w.-])--[A-Za-z][\w-]*")


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


def _block_texts(text: str) -> list[tuple[int, str]]:
    """Return (line number of the block's first body line, joined string contents) per shell:/run: block."""
    lines = text.splitlines()
    found = []
    i = 0
    while i < len(lines):
        match = BLOCK_START.match(lines[i])
        if not match:
            i += 1
            continue
        indent, first, body = len(match.group(1)), i + 2, []
        i += 1
        while i < len(lines) and (
            not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent
        ):
            if not lines[i].lstrip().startswith("#"):
                body.append(lines[i])
            i += 1
        joined = " ".join(m.group(1)[1:-1] for line in body for m in STRING.finditer(line))
        found.append((first, PLACEHOLDER.sub(" ", joined)))
    return found


def _target(block: str) -> str | None:
    match = TARGET.search(block)
    if not match:
        return None
    return match.group(1) or match.group(2).replace(".", "/") + ".py"


def _registered_flags(source: str) -> set[str]:
    flags = set()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.startswith("--"):
                        flags.add(arg.value)
    return flags


def _invocations() -> list[tuple[str, int, str, list[str]]]:
    out = []
    for path in _workflow_files(ENTRY):
        for line, block in _block_texts(path.read_text(encoding="utf-8")):
            target = _target(block)
            if target is not None:
                out.append((path.name, line, target, FLAG.findall(block)))
    return out


def test_the_block_reader_joins_adjacent_strings_and_drops_placeholders():
    text = (
        "rule r:\n"
        "    shell:\n"
        '        "\\"{sys.executable}\\" -m src.tool "\n'
        "        # a comment with --not-a-flag\n"
        '        "--alpha {input.x} --beta-two {params.flag} "\n'
        '        "> {log} 2>&1"\n'
        "\n"
        "rule s:\n"
        "    run:\n"
        "        cmd = f'\"{sys.executable}\" scripts/other.py --gamma'\n"
        "        shell(cmd)\n"
    )
    blocks = _block_texts(text)
    assert [_target(b) for _, b in blocks] == ["src/tool.py", "scripts/other.py"]
    assert FLAG.findall(blocks[0][1]) == ["--alpha", "--beta-two"]
    assert FLAG.findall(blocks[1][1]) == ["--gamma"]


def test_only_add_argument_options_count_as_registered():
    source = (
        "import argparse, subprocess\n"
        "p = argparse.ArgumentParser()\n"
        'p.add_argument("--num-decoys", "-n", type=int)\n'
        'p.add_argument("positional")\n'
        'subprocess.run(["docker", "run", "--rm"])\n'
    )
    assert _registered_flags(source) == {"--num-decoys"}


def test_every_flag_a_rule_passes_is_registered_by_its_script():
    invocations = _invocations()
    targets = {target for _, _, target, _ in invocations}
    # Anti-vacuity: the scan must reach the rule this test was written for, and pass flags somewhere.
    assert "scripts/generate_hard_decoys.py" in targets
    assert sum(len(flags) for *_, flags in invocations) > 0
    problems = []
    for workflow, line, target, flags in invocations:
        script = REPO_ROOT / target
        if not script.exists():
            problems.append(f"{workflow} block at line {line}: {target} does not exist")
            continue
        registered = _registered_flags(script.read_text(encoding="utf-8"))
        for flag in flags:
            if flag not in registered:
                problems.append(
                    f"{workflow} block at line {line}: {target} does not register {flag} "
                    f"(it registers {sorted(registered)})"
                )
    assert not problems, "\n".join(problems)
