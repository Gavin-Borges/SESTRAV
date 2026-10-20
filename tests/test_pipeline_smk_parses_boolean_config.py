"""pipeline.smk parses its boolean config keys, rather than testing them for truthiness.

Why this test exists
--------------------
Three rules in `pipeline.smk` build a command-line flag from a boolean config key:
`full_validation_report` (`--freeze-mode` from `freeze_mode`), `extract_verify_data`
(`--mock` from `mock_ingestion`) and `evaluate_verify_gnn` (`--mock` from
`mock_evaluation`). Each did it as `"--flag" if config.get(key, False) else ""`.

Snakemake does not deliver every spelling of a boolean as a bool. Measured with
snakemake 9.23.1: `--config freeze_mode=false` arrives as the str 'false', and so
do `FALSE`, `no` and `No`; only `False` and `True` become bools, and `0` and `1`
ints. A non-empty str is truthy, so a dry run of `full_validation_report` against
the smoke fixture printed `--freeze-mode` for `--config freeze_mode=false`, and the
two verify rules printed `--mock` for `--config mock_ingestion=false
mock_evaluation=false`. `False` and `0` did drop the flag, which is why the defect
was easy to miss. `scripts/stage4.py` had the same defect with bare bool() and now
parses with `_parse_flag`; `tests/test_stage4_parses_boolean_params.py` covers it.

What is asserted here
---------------------
- `_config_flag`, lifted out of the Snakefile and run directly, maps each accepted
  spelling to the bool it names and rejects anything else, naming the key and value;
- it accepts exactly what Stage 4's `_parse_flag` accepts, so the two halves of
  `--config freeze_mode=...` cannot disagree;
- each of the three rules reads its OWN key through it, and none still tests the
  raw config value for truthiness;
- a real `snakemake --dry-run --printshellcmds` renders each flag as the override
  says, with the keys set to opposite values so that none can be read for another,
  and a value that is not a boolean stops the workflow with the key named.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / "pipeline.smk"
STAGE4 = REPO_ROOT / "scripts" / "stage4.py"
SMOKE_CONFIG = "tests/fixtures/dag_smoke/config.smoke.yaml"
KEYS = ("freeze_mode", "mock_ingestion", "mock_evaluation")

ACCEPTED = [
    ("false", False),
    ("False", False),
    ("FALSE", False),
    ("0", False),
    ("no", False),
    ("No", False),
    ("true", True),
    ("True", True),
    ("TRUE", True),
    ("1", True),
    ("yes", True),
    ("YES", True),
    (False, False),
    (True, True),
    (0, False),
    (1, True),
]
REJECTED = ["maybe", "", "falsee", "2", "off", 2, -1, None, 0.0]


class _WorkflowError(Exception):
    """Stands in for snakemake.exceptions.WorkflowError, so these checks need no snakemake."""


def _top_level_block(source: str, header: str) -> str:
    """Return the unindented statement starting with `header`, with its indented body."""
    lines = source.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(header)]
    assert len(starts) == 1, f"expected one top-level {header!r} in pipeline.smk, found {len(starts)}"
    end = starts[0] + 1
    while end < len(lines) and (not lines[end].strip() or lines[end][:1] in (" ", "\t")):
        end += 1
    return "\n".join(lines[starts[0]:end])


def _config_flag(config: dict):
    """Exec pipeline.smk's own `_config_flag` (and what it calls) against `config`."""
    source = PIPELINE.read_text(encoding="utf-8")
    namespace = {"WorkflowError": _WorkflowError, "config": config}
    for header in ("def _reject(", "_TRUE_SPELLINGS =", "_FALSE_SPELLINGS =", "def _config_flag("):
        exec(_top_level_block(source, header), namespace)  # noqa: S102 - the tracked Snakefile
    return namespace["_config_flag"]


def _stage4_parse_flag():
    """Exec scripts/stage4.py's `_parse_flag` without running the script body."""
    tree = ast.parse(STAGE4.read_text(encoding="utf-8"))
    wanted = {"_TRUE_SPELLINGS", "_FALSE_SPELLINGS", "_parse_flag"}
    body: list[ast.stmt] = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in wanted for t in node.targets))
    ]
    assert len(body) == 3, "scripts/stage4.py no longer defines _parse_flag and its two spelling sets"
    namespace: dict = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(STAGE4), "exec"), namespace)  # noqa: S102
    return namespace["_parse_flag"]


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize(("raw", "expected"), ACCEPTED, ids=[repr(raw) for raw, _ in ACCEPTED])
def test_each_spelling_parses_to_the_bool_it_names(key, raw, expected):
    got = _config_flag({key: raw})(key)
    assert got is expected, f"config {key}={raw!r} parsed as {got!r}, not {expected!r}"


@pytest.mark.parametrize("key", KEYS)
def test_an_absent_key_is_off(key):
    assert _config_flag({})(key) is False


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("raw", REJECTED, ids=[repr(raw) for raw in REJECTED])
def test_anything_else_is_rejected_naming_the_key_and_value(key, raw):
    with pytest.raises(_WorkflowError) as excinfo:
        _config_flag({key: raw})(key)
    message = str(excinfo.value)
    assert repr(key) in message, f"the error does not name the key {key!r}: {message}"
    assert repr(raw) in message, f"the error does not show the value it refused: {message}"
    assert "must be a boolean" in message


@pytest.mark.parametrize(
    "raw",
    [raw for raw, _ in ACCEPTED] + REJECTED,
    ids=[repr(raw) for raw, _ in ACCEPTED] + [repr(raw) for raw in REJECTED],
)
def test_the_snakefile_accepts_exactly_what_stage4_accepts(raw):
    parse_flag = _stage4_parse_flag()

    def stage4(value):
        try:
            return parse_flag("freeze_mode", value)
        except ValueError:
            return "rejected"

    def snakefile(value):
        try:
            return _config_flag({"freeze_mode": value})("freeze_mode")
        except _WorkflowError:
            return "rejected"

    assert snakefile(raw) == stage4(raw), (
        f"freeze_mode={raw!r}: pipeline.smk gives {snakefile(raw)!r}, "
        f"scripts/stage4.py gives {stage4(raw)!r}"
    )


@pytest.mark.parametrize(
    ("param", "flag", "key"),
    [
        ("freeze_flag", "--freeze-mode", "freeze_mode"),
        ("mock_flag", "--mock", "mock_ingestion"),
        ("mock_flag", "--mock", "mock_evaluation"),
    ],
)
def test_each_rule_parses_its_own_key(param, flag, key):
    source = PIPELINE.read_text(encoding="utf-8")
    pinned = rf'{param}\s*=\s*"{re.escape(flag)}"\s+if\s+_config_flag\(\s*"{key}"\s*\)\s+else\s+""'
    assert len(re.findall(pinned, source)) == 1, f"{flag} is not built from _config_flag({key!r})"
    truthiness = rf"""\bif\s+config(?:\.get\(\s*|\[\s*)["']{key}["']"""
    assert not re.search(truthiness, source), (
        f"pipeline.smk still tests the raw config value of {key} for truthiness, "
        f"which makes --config {key}=false switch the flag ON"
    )


def _snakemake() -> str:
    """PATH first, then the launcher beside the running interpreter (see the quoting tests)."""
    found = shutil.which("snakemake")
    if found is None:
        launcher_dir = Path(sys.executable).parent
        for name in ("Scripts/snakemake.exe", "snakemake.exe", "bin/snakemake", "snakemake"):
            if (launcher_dir / name).exists():
                found = str(launcher_dir / name)
                break
    if found is None:
        pytest.skip("snakemake launcher not found on PATH or beside the running interpreter")
    return str(found)


# The printed command of each rule, found by its own script and its own log redirect so
# that the job summary lines (which name the log but not the script) are not counted.
COMMANDS = {
    "freeze_mode": ("src.final_validation_report", "logs/full_validation_report.log", "--freeze-mode"),
    "mock_ingestion": ("iedb_multi_virus_extractor.py", "logs/extract_verify_data.log", "--mock"),
    "mock_evaluation": ("sestrav_evaluator.py", "logs/evaluate_verify_gnn.log", "--mock"),
}


def _dry_run(overrides: dict) -> subprocess.CompletedProcess:
    """Dry-run the three rules, forced so their commands print, against the smoke fixture.

    A dry run executes no job and takes no lock, so it is run in the repository itself.
    """
    return subprocess.run(
        [
            _snakemake(),
            "results/final_validation_report.md",
            "results/verify/validation_report.json",
            "--forcerun",
            "full_validation_report",
            "extract_verify_data",
            "evaluate_verify_gnn",
            "--snakefile",
            "pipeline.smk",
            "--configfile",
            SMOKE_CONFIG,
            "--dry-run",
            "--printshellcmds",
            "--cores",
            "1",
            "--config",
            *(f"{key}={value}" for key, value in overrides.items()),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _flag_rendered(output: str, key: str) -> bool:
    script, log, flag = COMMANDS[key]
    lines = [line for line in output.splitlines() if script in line and log in line]
    assert len(lines) == 1, f"expected one printed {script} command, found {len(lines)}:\n{output}"
    return re.search(rf"\s{re.escape(flag)}(\s|$)", lines[0]) is not None


# Each pair is (a spelling of false, a spelling of true). The three keys get opposite
# values in each arrangement, so a rule reading another rule's key cannot pass.
PAIRS = [("false", "true"), ("No", "YES"), ("0", "1")]


@pytest.mark.parametrize("first_on", [False, True], ids=["freeze-off", "freeze-on"])
@pytest.mark.parametrize(("off", "on"), PAIRS, ids=[f"{off}-{on}" for off, on in PAIRS])
def test_the_rendered_commands_follow_the_override(off, on, first_on):
    wanted = {"freeze_mode": first_on, "mock_ingestion": not first_on, "mock_evaluation": first_on}
    result = _dry_run({key: (on if value else off) for key, value in wanted.items()})
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    for key, expected in wanted.items():
        spelled = on if expected else off
        assert _flag_rendered(output, key) is expected, (
            f"--config {key}={spelled} rendered the {COMMANDS[key][2]} flag "
            f"{'absent' if expected else 'present'}"
        )


@pytest.mark.parametrize("key", KEYS)
def test_a_value_that_is_not_a_boolean_stops_the_workflow(key):
    result = _dry_run({key: "maybe"})
    output = result.stdout + result.stderr
    assert result.returncode != 0, f"--config {key}=maybe was accepted:\n{output}"
    assert f"config[{key!r}]" in output and "'maybe'" in output, (
        f"the failure does not name the key and the value:\n{output}"
    )
    assert "must be a boolean" in output, output
