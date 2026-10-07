"""Every Snakemake `shell:` interpolation must carry Snakemake's `:q` quoting.

`pipeline.smk` and the `standardize_outputs.smk` it includes build their shell
commands by interpolating config-derived and wildcard-derived values. Before this
file, every one of those interpolations was bare: a value could carry shell
metacharacters and the shell would act on them. The shipped default decoy allele
is `HLA-A*02:01`, which is itself a glob, so even the untouched configuration put
an unquoted `*` on a command line.

Measured on the pre-fix tree: a `--config decoy_allele` value of `A& mkdir PWNED&`
caused a real `snakemake --cores 1` run to create a directory named PWNED, and the
same payload routed through the `proteome_id` wildcard reached `run_prime`'s shell
command verbatim. The defect is in released v2.0.3.

No scanner covers this. Bandit skips both files even when they are named on its
command line (it reports a syntax error parsing the AST, because `rule x:` blocks
are not Python), and nothing in `.github/`, `semgrep-rules/` or `pyproject.toml`
opts a `.smk` extension into semgrep or CodeQL. The only CI coverage of these
files is two Snakemake dry-run legs, which resolve file names and never inspect
quoting. A test is therefore the only available gate.

Three independent instruments are used here deliberately, because they have
different traversal policies and can disagree:

1. A static scan of the tracked workflow files, which parses each `shell:` block
   back into one string and asks Snakemake's own `string.Formatter` which fields
   carry a `q` format spec. This is the gate that holds the fix in place.
2. Snakemake's own `snakemake.utils.format`, run over the real shipped `shell:`
   strings with a hostile value, under both quote functions. This checks that
   `:q` actually neutralises the payload rather than merely being present.
3. A live `snakemake --dry-run --printshellcmds`, plus one real sandboxed run,
   which check the whole path end to end including Snakemake's own machinery.

On cross-platform correctness, measured rather than assumed: `:q` is not
POSIX-only. `snakemake.shell` sets no shell executable on Windows, so a job runs
through `subprocess.Popen(shell=True)`, which uses COMSPEC (cmd.exe here), and
the same module swaps the quote function from `shlex.quote` to its cmd.exe
quoting whenever no explicit shell executable is set. So `:q` adapts to the
platform. One asymmetry does follow from that and is accounted for below: the
`--printshellcmds` DISPLAY path formats without that substitution, so a dry run
shows POSIX-style quoting even on Windows, while the executed command uses
cmd.exe quoting. The assertions below test for "quoted somehow", not for a
specific quote character.
"""

from __future__ import annotations

import ast
import io
import re
import shutil
import string
import subprocess
import sys
import tarfile
import textwrap
import types
from pathlib import Path

import pytest

# snakemake lives in the [pipeline] extra, NOT [dev]. A bare module-scope import here
# aborts COLLECTION of the entire suite on a `pip install -e ".[dev]"` box, and by this
# repo's rule 3 a collection error means ZERO tests ran - so the failure mode is the
# whole suite, not this module. tests/test_dev_extra_runs_the_test_suite.py exists to
# catch exactly that and did catch it, on the composed tree rather than in this module's
# own green run.
#
# importorskip keeps collection safe and skips this module only where snakemake is
# absent. CI installs the pipeline extra in order to run the two Snakemake battery legs,
# so coverage there is unchanged. `cmd_exe_quote` cannot be imported lazily inside a
# test: it is consumed by a @pytest.mark.parametrize decorator, which is evaluated at
# module scope.
_snakemake_utils = pytest.importorskip(
    "snakemake.utils",
    reason=(
        "snakemake is in the [pipeline] extra, not [dev]; "
        'install ".[pipeline]" to exercise the shell-quoting tests'
    ),
)
cmd_exe_quote = _snakemake_utils.cmd_exe_quote
snakemake_format = _snakemake_utils.format

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY = REPO_ROOT / "pipeline.smk"
SMOKE_CONFIG = "tests/fixtures/dag_smoke/config.smoke.yaml"

INCLUDE = re.compile(r"""^\s*include:\s*["']([^"']+)["']""", re.MULTILINE)

# Dangerous on both shells this workflow can run under: `&` separates commands in
# bash and in cmd.exe alike, and `mkdir` needs no stdout, so the payload survives
# the trailing `> {log} 2>&1` that every rule appends. An earlier payload using
# `touch` and a redirect was absorbed by that redirect on cmd.exe (the injected
# command still RAN, its output simply landed in the log file), which is why the
# observable chosen here is a directory and not a file.
PAYLOAD = "A& mkdir PWNED&"


def _quoted_spans(text: str) -> list[tuple[int, int]]:
    """Return the (start, end) interiors of each quoted region in a command line.

    A deliberate hand-rolled scanner rather than `shlex`: cmd.exe quoting doubles
    backslashes, which POSIX `shlex` would consume as escapes, mangling the
    Windows interpreter path. This only needs to know which byte ranges a shell
    would treat as one word, and the payload contains no quote character.
    """
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char in "'\"":
            close = text.find(char, index + 1)
            if close == -1:
                break
            spans.append((index + 1, close))
            index = close + 1
        else:
            index += 1
    return spans


def payload_is_contained(text: str) -> bool:
    """True when every occurrence of PAYLOAD sits inside one quoted region.

    That is the property that matters: a shell splits on the metacharacters in
    the payload unless the whole of it lies inside a single quoted word.
    """
    spans = _quoted_spans(text)
    for match in re.finditer(re.escape(PAYLOAD), text):
        if not any(start <= match.start() and match.end() <= end for start, end in spans):
            return False
    return True


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _directive_blocks(text: str, keyword: str) -> list[tuple[int, str]]:
    """Return (1-based line number, raw body) for each `<keyword>:` directive."""
    lines = text.splitlines()
    blocks: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() != f"{keyword}:":
            index += 1
            continue
        base = _indent(lines[index])
        body: list[str] = []
        cursor = index + 1
        while cursor < len(lines):
            line = lines[cursor]
            if not line.strip():
                body.append("")
                cursor += 1
                continue
            if _indent(line) <= base:
                break
            body.append(line)
            cursor += 1
        blocks.append((index + 1, "\n".join(body)))
        index = cursor
    return blocks


def shell_commands(text: str) -> list[tuple[int, str]]:
    """Assemble each `shell:` block into the single string Snakemake receives.

    The body of a `shell:` directive is implicitly concatenated string literals,
    which Snakemake's parser wraps in a call. Wrapping it in parentheses and
    handing it to `ast` reproduces that concatenation and drops comments, which a
    regex over the raw lines could not do correctly.
    """
    commands = []
    for lineno, body in _directive_blocks(text, "shell"):
        node = ast.parse("(\n" + textwrap.dedent(body) + "\n)", mode="eval")
        commands.append((lineno, ast.literal_eval(node)))
    return commands


def interpolations(command: str) -> list[tuple[str, str]]:
    """Return (field_name, format_spec) for every placeholder in a command."""
    return [
        (field, spec or "")
        for _, field, spec, _ in string.Formatter().parse(command)
        if field is not None
    ]


def workflow_files() -> list[Path]:
    seen: list[Path] = []
    pending = [ENTRY.resolve()]
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.append(path)
        for name in INCLUDE.findall(path.read_text(encoding="utf-8")):
            pending.append((path.parent / name).resolve())
    return seen


# ---------------------------------------------------------------------------
# Instrument 1: static scan of the tracked workflow files
# ---------------------------------------------------------------------------


def test_the_parser_sees_quoting_and_the_absence_of_it():
    """Premise anchor for instrument 1.

    If this parser could not tell a quoted field from an unquoted one, the
    zero-unquoted assertion below would pass on any input at all.
    """
    quoted = shell_commands('shell:\n    "a {x:q} b"\n')
    assert quoted == [(1, "a {x:q} b")]
    assert interpolations(quoted[0][1]) == [("x", "q")]

    bare = shell_commands('shell:\n    "a {x} b"\n')
    assert interpolations(bare[0][1]) == [("x", "")]

    joined = shell_commands('shell:\n    # a comment\n    "a {x:q} "\n    "b {y} c"\n')
    assert joined[0][1] == "a {x:q} b {y} c"
    assert interpolations(joined[0][1]) == [("x", "q"), ("y", "")]

    assert shell_commands("rule r:\n    output: 'o'\n") == []


def test_the_include_chain_is_followed():
    names = {path.name for path in workflow_files()}
    assert {"pipeline.smk", "standardize_outputs.smk"} <= names


def test_every_shell_interpolation_is_quoted():
    unquoted = []
    total = 0
    for path in workflow_files():
        for lineno, command in shell_commands(path.read_text(encoding="utf-8")):
            for field, spec in interpolations(command):
                total += 1
                if not spec.endswith("q"):
                    unquoted.append(f"{path.name} shell: at line {lineno} -> {{{field}}}")
    assert total > 0, (
        "no shell: interpolations were found at all, so this test is vacuous; "
        "the workflow files or the parser have moved"
    )
    assert unquoted == [], (
        "these shell: interpolations are missing Snakemake's :q quoting, so a "
        "config or wildcard value reaches the shell unquoted:\n  "
        + "\n  ".join(unquoted)
    )


def test_no_rule_builds_its_command_as_a_string_and_calls_shell():
    """A `run:` block that f-strings a command defeats `:q` entirely.

    Python interpolates an f-string before Snakemake's formatter ever sees it, so
    there is no point at which a format spec could be applied. Any such block is
    unquotable by construction and must be a `shell:` directive or an argv list.
    """
    offenders = []
    for path in workflow_files():
        for lineno, body in _directive_blocks(path.read_text(encoding="utf-8"), "run"):
            if "shell(" in body:
                offenders.append(f"{path.name} run: at line {lineno}")
    assert offenders == [], (
        "these run: blocks hand a constructed string to shell(), which cannot be "
        "quoted with :q:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Instrument 2: Snakemake's own formatter over the real shipped strings
# ---------------------------------------------------------------------------


def _namespace_for(command: str) -> dict:
    """Build just enough of a job namespace to format `command`."""
    fields = {field.split(".")[0] for field, _ in interpolations(command)}
    attrs = {}
    for field, _ in interpolations(command):
        root, _, attr = field.partition(".")
        if attr:
            attrs.setdefault(root, {})[attr] = PAYLOAD
    namespace: dict = {}
    for root in fields:
        if root in attrs:
            namespace[root] = types.SimpleNamespace(**attrs[root])
        else:
            namespace[root] = PAYLOAD
    return namespace


@pytest.mark.parametrize("quote_func", [None, cmd_exe_quote], ids=["posix", "cmd_exe"])
def test_the_shipped_commands_neutralise_a_hostile_value(quote_func):
    """The payload must never appear in a form a shell would split.

    Run over the real `shell:` strings, not a synthetic one, under both of the
    quote functions Snakemake selects between.
    """
    checked = 0
    for path in workflow_files():
        for lineno, command in shell_commands(path.read_text(encoding="utf-8")):
            if not interpolations(command):
                continue
            kwargs = _namespace_for(command)
            if quote_func is not None:
                kwargs["quote_func"] = quote_func
            rendered = snakemake_format(command, **kwargs)
            checked += 1
            assert PAYLOAD in rendered, (
                f"{path.name} shell: at line {lineno} did not interpolate the "
                "payload at all, so this assertion is vacuous"
            )
            assert payload_is_contained(rendered), (
                f"{path.name} shell: at line {lineno} emitted the payload outside "
                f"any quoted word, so a shell would split on it:\n  {rendered}"
            )
    assert checked > 0, "no shipped shell: command was exercised; this test is vacuous"


def test_the_containment_check_rejects_an_unquoted_command():
    """Premise anchor for instruments 2 and 3.

    Without this, `payload_is_contained` returning True would be unfalsifiable and
    both the formatter test and the dry-run test would pass on the defect.
    """
    command = "python s.py --allele {params.allele} > {log} 2>&1"
    unquoted = snakemake_format(command, **_namespace_for(command))
    assert PAYLOAD in unquoted
    assert not payload_is_contained(unquoted), (
        "an unquoted interpolation read as contained, so the check cannot detect "
        "the defect this file exists for"
    )

    fixed = "python s.py --allele {params.allele:q} > {log:q} 2>&1"
    assert payload_is_contained(snakemake_format(fixed, **_namespace_for(fixed)))

    # A payload that merely sits NEAR a quoted region must not read as contained.
    assert not payload_is_contained(f"--alleles 'HLA-A*02:01' --out x/{PAYLOAD}.csv")
    # And one inside a longer quoted path must.
    assert payload_is_contained(f"--out 'results/{PAYLOAD}_binding.csv'")


# ---------------------------------------------------------------------------
# Instrument 3: live Snakemake, dry-run and one real run
# ---------------------------------------------------------------------------


def _snakemake() -> str:
    """Resolve the snakemake launcher, PATH first and the running env second.

    PATH alone is not enough here. On a workstation where the conda environment
    is active for the interpreter but not exported to the shell, `shutil.which`
    returns None while snakemake is installed and importable, and every test
    below would skip - certifying nothing while reading as green. The second
    lookup is the launcher directory beside the interpreter actually running
    these tests, which is the environment snakemake was installed into.
    """
    found = shutil.which("snakemake")
    if found is None:
        launcher_dir = Path(sys.executable).parent
        for name in ("Scripts/snakemake.exe", "snakemake.exe", "bin/snakemake", "snakemake"):
            candidate = launcher_dir / name
            if candidate.exists():
                found = str(candidate)
                break
    if found is None:
        pytest.skip("snakemake launcher not found on PATH or beside the running interpreter")
    return str(found)


def _sandbox(destination: Path) -> Path:
    """Export the tracked tree, then overlay the WORKING-TREE workflow files.

    The supporting tree (scripts, fixtures, config.yaml) comes from `git archive`
    so the sandbox holds committed content only. The two workflow files are then
    taken from the working tree on purpose, so an uncommitted edit to the files
    under test is what gets measured; otherwise this test could only ever report
    on the previous commit.
    """
    result = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("not a git work tree, so the tracked sandbox cannot be exported")
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
        archive.extractall(destination, filter="data")
    for path in workflow_files():
        shutil.copyfile(path, destination / path.name)
    return destination


def _run(tree: Path, args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_snakemake(), *args],
        cwd=tree,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_a_hostile_wildcard_is_printed_quoted(tmp_path):
    """End-to-end quoting, through the one vector config validation cannot see.

    Wildcards come from the requested target, not from the config, so the
    validation prelude has no say over them. This is therefore the honest
    end-to-end test of the quoting layer on its own.
    """
    if not hasattr(tarfile, "data_filter"):
        pytest.skip("tarfile extraction filters need Python 3.11.4 or later")
    tree = _sandbox(tmp_path / "tracked-tree")
    (tree / "data" / "proteomes").mkdir(parents=True, exist_ok=True)
    (tree / "data" / "proteomes" / f"{PAYLOAD}.fasta").write_text(
        ">stub\nMKTAYIAKQRQISFVKSHFSRQ\n", encoding="utf-8"
    )

    result = _run(
        tree,
        [
            f"results/{PAYLOAD}_prime_output.txt",
            "--snakefile",
            "pipeline.smk",
            "--configfile",
            SMOKE_CONFIG,
            "--dry-run",
            "--printshellcmds",
            "--cores",
            "1",
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr

    printed = [
        line for line in (result.stdout + result.stderr).splitlines()
        if "run_prime_wrapper.py" in line
    ]
    assert printed, (
        "no run_prime shell command was printed, so there is nothing to check:\n"
        + result.stdout
        + result.stderr
    )
    for line in printed:
        assert PAYLOAD in line, f"the payload did not reach the command at all:\n{line}"
        assert payload_is_contained(line), f"payload printed unquoted:\n{line}"


def test_a_real_run_with_a_hostile_wildcard_creates_no_pwned(tmp_path):
    """The one real execution. Pre-fix this created a directory named PWNED.

    The inputs `run_prime` needs are stubbed so exactly one shell rule runs. The
    rule's own exit status is not asserted: the injected command would have fired
    at shell-expansion time, before the wrapper did anything, so the observable
    is the PWNED directory and nothing else.
    """
    if not hasattr(tarfile, "data_filter"):
        pytest.skip("tarfile extraction filters need Python 3.11.4 or later")
    tree = _sandbox(tmp_path / "tracked-tree")
    (tree / "results").mkdir(exist_ok=True)
    (tree / "results" / f"{PAYLOAD}_binding.csv").write_text(
        "peptide,allele\nCLGGLLTMV,HLA-A*02:01\n", encoding="utf-8"
    )

    try:
        _run(
            tree,
            [
                f"results/{PAYLOAD}_prime_output.txt",
                "--snakefile",
                "pipeline.smk",
                "--configfile",
                SMOKE_CONFIG,
                "--cores",
                "1",
            ],
        )
    except subprocess.TimeoutExpired:
        pass  # the assertion below is still the one that matters

    assert not (tree / "PWNED").exists(), (
        "the injected command ran: a wildcard-borne shell payload created PWNED"
    )
    assert list(tree.glob("**/PWNED*")) == [], "an injected command created a PWNED path"


def test_a_hostile_config_value_is_rejected_before_any_rule_runs(tmp_path):
    """The second defence layer: the config validation prelude.

    The quoting above already neutralises this payload. The prelude refuses it
    outright, so it never reaches a command line at all, and the failure names
    itself rather than surfacing as an unexplained rule error.
    """
    if not hasattr(tarfile, "data_filter"):
        pytest.skip("tarfile extraction filters need Python 3.11.4 or later")
    tree = _sandbox(tmp_path / "tracked-tree")

    result = _run(
        tree,
        [
            "data/hard_decoys.csv",
            "--snakefile",
            "pipeline.smk",
            "--configfile",
            SMOKE_CONFIG,
            "--cores",
            "1",
            "--config",
            f"decoy_allele={PAYLOAD}",
        ],
    )
    assert result.returncode != 0, "a hostile decoy_allele was accepted:\n" + result.stdout
    combined = result.stdout + result.stderr
    assert "validation prelude" in combined, (
        "the run failed for some other reason than config validation:\n" + combined
    )
    assert not (tree / "PWNED").exists()


# One case per rule in the prelude, each pinned to the REASON that rule gives.
#
# Pinning the reason rather than just the exit status is deliberate, and it was
# earned: the prelude's three path rules overlap, so a mutation battery that
# deleted the absolute-path rule, the `..` rule or the path-segment rule in turn
# left every case still rejected by one of the other two. All three mutants
# SURVIVED a test that only asserted a non-zero exit. Asserting which rule fired
# makes each one independently load-bearing. The `data/my file.csv` case was added
# for the same reason: it is the only value here that is neither absolute nor
# contains `..`, so it is the only one the path-segment rule alone can catch.
REJECTED_CONFIGS = [
    ("decoy_allele", "not an allele", "must be an allele name"),
    ("alleles", '["bad; allele"]', "must be an allele name"),
    ("antigens", '["bad antigen"]', "must be a bare proteome id"),
    ("proteome_files", '{"bad id": "data/x.fasta"}', "must be a bare proteome id"),
    ("proteome_files", '{"SMOKE": "/etc/passwd"}', "not absolute"),
    ("dataset_mode", "mode; rm -rf /", "must be a bare identifier"),
    ("dataset_version", "$(whoami)", "must look like"),
    ("num_decoys", "10000; echo hi", "must be an integer"),
    ("training_dataset", "/etc/passwd", "not absolute"),
    ("gnn_checkpoint", "C:/Windows/System32/x.pth", "not absolute"),
    ("training_dataset", "../../outside.csv", "must not escape the repository root"),
    ("training_dataset", "data/my file.csv", "path segments are restricted"),
    ("training_dataset", "data\\x.csv", "must use forward slashes"),
]


@pytest.mark.parametrize(
    "key,value,reason",
    REJECTED_CONFIGS,
    ids=[f"{key}-{index}" for index, (key, _, _) in enumerate(REJECTED_CONFIGS)],
)
def test_the_prelude_rejects_each_shape_it_claims_to(tmp_path, key, value, reason):
    if not hasattr(tarfile, "data_filter"):
        pytest.skip("tarfile extraction filters need Python 3.11.4 or later")
    tree = _sandbox(tmp_path / "tracked-tree")
    result = _run(
        tree,
        [
            "--snakefile",
            "pipeline.smk",
            "--configfile",
            SMOKE_CONFIG,
            "--dry-run",
            "--cores",
            "1",
            "--config",
            f"{key}={value}",
        ],
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0, f"config {key}={value!r} was accepted:\n{combined}"
    assert "validation prelude" in combined, (
        f"config {key}={value!r} failed for some other reason than validation:\n{combined}"
    )
    assert reason in combined, (
        f"config {key}={value!r} was rejected, but not by the rule that claims it "
        f"(expected a reason containing {reason!r}):\n{combined}"
    )


def test_both_battery_legs_still_resolve(tmp_path):
    """The fixture leg and the bare leg, which test different things.

    The fixture leg overrides every `config.get()` default, so only the bare leg
    exercises the shipped `config.yaml`. Both are run here against a tracked-only
    export, which is what CI sees.
    """
    if not hasattr(tarfile, "data_filter"):
        pytest.skip("tarfile extraction filters need Python 3.11.4 or later")
    tree = _sandbox(tmp_path / "tracked-tree")

    fixture_leg = _run(
        tree,
        [
            "--snakefile",
            "pipeline.smk",
            "--configfile",
            SMOKE_CONFIG,
            "--dry-run",
            "--cores",
            "1",
        ],
    )
    assert fixture_leg.returncode == 0, fixture_leg.stdout + fixture_leg.stderr

    bare_leg = _run(
        tree,
        ["--snakefile", "pipeline.smk", "--dry-run", "--cores", "1"],
    )
    assert bare_leg.returncode == 0, bare_leg.stdout + bare_leg.stderr


def test_the_shipped_allele_glob_is_quoted_end_to_end(tmp_path):
    """The shipped default is itself a glob, and must be quoted in its own right.

    `HLA-A*02:01` passes validation, so this exercises the quoting layer with a
    value the prelude permits. Pre-fix the `*` reached the shell bare.
    """
    if not hasattr(tarfile, "data_filter"):
        pytest.skip("tarfile extraction filters need Python 3.11.4 or later")
    tree = _sandbox(tmp_path / "tracked-tree")
    result = _run(
        tree,
        [
            "data/hard_decoys.csv",
            "--snakefile",
            "pipeline.smk",
            "--configfile",
            SMOKE_CONFIG,
            "--dry-run",
            "--printshellcmds",
            "--cores",
            "1",
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    printed = [
        line for line in (result.stdout + result.stderr).splitlines()
        if "generate_hard_decoys.py" in line
    ]
    assert printed, "no generate_hard_decoys command was printed"
    for line in printed:
        assert "--alleles 'HLA-A*02:01'" in line or '--alleles "HLA-A*02:01"' in line, (
            f"the shipped allele glob was not quoted:\n{line}"
        )
