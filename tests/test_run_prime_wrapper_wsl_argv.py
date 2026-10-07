"""The PRIME wrapper's WSL invocation must exec, not go through a Linux shell.

`wsl.exe CMD ARG...` with no `-e`/`--exec` does not exec CMD: it joins the argv into
a command line and hands it to the default Linux shell, which re-parses it. A Python
list and `shell=False` therefore say nothing about the Linux side. Measured on the
workstation these guards were written on, with `subprocess.run` and a list argv:

    wsl    printf "[%s]" "a;b"        -> stdout '[a]',  rc 127,
                                         stderr "/bin/bash: line 1: b: command not found"
    wsl -e printf "[%s]" "a;b"        -> stdout '[a;b]', rc 0
    wsl    printf "[%s]" "$(id -u)"   -> stdout '[1000]'   (command substitution RAN)
    wsl -e printf "[%s]" "$(id -u)"   -> stdout '[$(id -u)]'

`--output` reaches that argv with no pattern check at all, which is what made the
missing `-e` the exposure rather than a style point.

PRIME itself is not installed on this workstation, so no live PRIME run is exercised
here and none is claimed: the end-to-end behaviour of the exec form against a real
PRIME is UNMEASURED. What is checked is the argv the wrapper builds and the allele
pattern, neither of which needs WSL.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import run_prime_wrapper  # noqa: E402


# A value that survives the wrapper's HLA-/*/: stripping unchanged, so it can be
# located in the built argv, and that the allele pattern accepts.
ALLELE_SENTINEL = "SENTINELALLELE"


@pytest.mark.parametrize(
    "value",
    [
        "A0201",
        "HLA-A*02:01",
        "HLA-A*02:01,HLA-B*07:02",
        "A_0201",
        "A-0201",
        # The whole shipped list, joined the way standardize_outputs.smk joins it.
        "HLA-A*02:01,HLA-A*01:01,HLA-A*03:01,HLA-B*07:02,HLA-B*44:02",
    ],
)
def test_the_allele_pattern_accepts_real_allele_lists(value: str) -> None:
    """Negative control: the tightened pattern must not reject what the pipeline sends."""
    assert run_prime_wrapper.ALLELES_RE.match(value), value


@pytest.mark.parametrize(
    "value",
    [
        "A0201\nid -u",  # embedded newline, the shell's own command separator
        "A0201\n",  # trailing newline: Python's `$` matches before one, `\Z` does not
        "A0201;id -u",
        "A0201;",
        "A0201$(id -u)",
        "$(id -u)",
        "A0201`id -u`",
        "A0201 B0702",  # a plain space split one argument into two
        "A0201\tB0702",
        "A0201\rB0702",
        "A0201|id",
        "A0201&id",
        "A0201>out",
        "A0201,,B0702",  # empty token
        ",A0201",
        "A0201,",
        "",
        "../../etc/passwd",
    ],
)
def test_the_allele_pattern_rejects_shell_metacharacters_and_whitespace(value: str) -> None:
    assert not run_prime_wrapper.ALLELES_RE.match(value), value


def test_the_allele_pattern_admits_no_whitespace_at_all() -> None:
    """Stated as a property, so a future widening of the class cannot quietly re-admit it."""
    for char in " \t\n\r\v\f":
        assert not run_prime_wrapper.ALLELES_RE.match(f"A0201{char}B0702"), repr(char)
        assert not run_prime_wrapper.ALLELES_RE.match(f"A0201{char}"), repr(char)


def test_the_probe_and_the_exec_name_the_same_binary() -> None:
    """Detection and execution must not be able to disagree about which file they mean.

    The old probe was `wsl which PRIME`, a PATH lookup, while the run used PATH too.
    Both are now built from one constant.
    """
    assert run_prime_wrapper.PRIME_WSL_BIN in run_prime_wrapper.PRIME_WSL_SCRIPT
    assert run_prime_wrapper.PRIME_WSL_BIN in run_prime_wrapper.PRIME_WSL_PROBE
    # The installer's own default root, so the two stay in step.
    assert "${SESTRAV_TOOL_ROOT:-$HOME/tools/sestrav_external}" in run_prime_wrapper.PRIME_WSL_BIN
    assert "/PRIME2.1/PRIME" in run_prime_wrapper.PRIME_WSL_BIN
    assert run_prime_wrapper.PRIME_WSL_SCRIPT.startswith("exec ")
    assert run_prime_wrapper.PRIME_WSL_SCRIPT.endswith('"$@"')


def _run_wsl_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alleles: str) -> list[str]:
    """Drive main() down the WSL branch and return the argv it hands subprocess.run.

    Nothing is mocked inside the wrapper's own argv construction: the list returned
    is the one the real code builds.
    """
    binding = tmp_path / "binding.csv"
    binding.write_text("peptide,presentation_score\nSIINFEKL,0.9\n", encoding="utf-8")
    output = tmp_path / "SENTINELOUT" / "prime_raw.txt"
    # The wrapper opens `<output>.pep_temp` before it makedirs the output directory,
    # so the directory has to exist already. Pre-existing ordering quirk, untouched
    # here; it is reported separately rather than fixed in a security change.
    output.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(run_prime_wrapper.sys, "platform", "win32")
    # No native PRIME anywhere: every candidate path must miss, but the binding CSV
    # and the temp peptides file must still be found.
    real_isfile = run_prime_wrapper.os.path.isfile
    monkeypatch.setattr(
        run_prime_wrapper.os.path,
        "isfile",
        lambda p: False if "PRIME" in str(p) else real_isfile(p),
    )

    recorded: list[list[str]] = []

    def fake_run(argv, *args, **kwargs):
        recorded.append(list(argv))
        if argv[0] in ("where", "which"):
            # No native PRIME on PATH; the real call passes check=True.
            raise subprocess.CalledProcessError(1, argv)
        if argv[:4] == ["wsl", "-e", "bash", "-c"] and argv[4].startswith("test -x"):
            return subprocess.CompletedProcess(argv, 0)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(run_prime_wrapper.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_prime_wrapper.sys,
        "argv",
        [
            "run_prime_wrapper.py",
            "--binding-csv",
            str(binding),
            "--output",
            str(output),
            "--alleles",
            alleles,
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        run_prime_wrapper.main()
    assert excinfo.value.code == 0, f"wrapper did not take the WSL success path: {recorded}"

    invocations = [argv for argv in recorded if argv[0] == "wsl" and "test -x" not in argv[-1]]
    assert len(invocations) == 1, recorded
    return invocations[0]


def test_the_wsl_invocation_uses_an_explicit_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = _run_wsl_branch(tmp_path, monkeypatch, ALLELE_SENTINEL)
    assert argv[0] == "wsl"
    assert argv[1] in ("-e", "--exec"), (
        f"argv[1] is {argv[1]!r}; without -e/--exec wsl.exe re-parses the whole "
        "command line in the default Linux shell"
    )
    assert argv[2:4] == ["bash", "-c"], argv


def test_the_wsl_invocation_keeps_every_input_out_of_the_shell_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `bash -c` script must be a constant, with inputs supplied positionally."""
    argv = _run_wsl_branch(tmp_path, monkeypatch, ALLELE_SENTINEL)
    script = argv[4]

    assert script == run_prime_wrapper.PRIME_WSL_SCRIPT, script
    # Nothing caller-controlled is interpolated into shell source.
    assert ALLELE_SENTINEL not in script
    assert "SENTINELOUT" not in script
    assert str(tmp_path) not in script

    # Both inputs are present, and only after the script, where "$@" reads them.
    positional = argv[5:]
    assert positional[0] == "prime", positional
    assert ALLELE_SENTINEL in positional, positional
    assert any("SENTINELOUT" in part for part in positional), positional
    assert "-a" in positional and "-i" in positional and "-o" in positional

    # Each value is its OWN argv element, not spliced into a larger one.
    assert positional[positional.index("-a") + 1] == ALLELE_SENTINEL


def test_the_wsl_probe_also_uses_an_explicit_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe gates the branch above, so it must not go through a shell either."""
    binding = tmp_path / "binding.csv"
    binding.write_text("peptide\nSIINFEKL\n", encoding="utf-8")

    monkeypatch.setattr(run_prime_wrapper.sys, "platform", "win32")
    real_isfile = run_prime_wrapper.os.path.isfile
    monkeypatch.setattr(
        run_prime_wrapper.os.path,
        "isfile",
        lambda p: False if "PRIME" in str(p) else real_isfile(p),
    )

    recorded: list[list[str]] = []

    def fake_run(argv, *args, **kwargs):
        recorded.append(list(argv))
        if argv[0] in ("where", "which"):
            raise subprocess.CalledProcessError(1, argv)
        # Report PRIME as absent, so the wrapper falls through to simulation and the
        # probe is the only wsl call made.
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(run_prime_wrapper.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_prime_wrapper.sys,
        "argv",
        [
            "run_prime_wrapper.py",
            "--binding-csv",
            str(binding),
            "--output",
            str(tmp_path / "prime_raw.txt"),
            "--alleles",
            "A0201",
        ],
    )

    run_prime_wrapper.main()

    probes = [argv for argv in recorded if argv[0] == "wsl"]
    assert len(probes) == 1, recorded
    probe = probes[0]
    assert probe[1] in ("-e", "--exec"), probe
    assert probe[2:4] == ["bash", "-c"], probe
    assert probe[4] == run_prime_wrapper.PRIME_WSL_PROBE, probe
    # A bare `which PRIME` cannot succeed for an installer-placed PRIME, because the
    # installer appends its PATH line after ~/.bashrc's non-interactive early return.
    assert "which" not in probe, probe


def test_an_allele_value_carrying_a_newline_is_refused_before_any_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: the pattern must stop the payload before anything is launched."""
    binding = tmp_path / "binding.csv"
    binding.write_text("peptide\nSIINFEKL\n", encoding="utf-8")

    recorded: list[list[str]] = []

    def fake_run(argv, *args, **kwargs):
        recorded.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(run_prime_wrapper.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_prime_wrapper.sys,
        "argv",
        [
            "run_prime_wrapper.py",
            "--binding-csv",
            str(binding),
            "--output",
            str(tmp_path / "prime_raw.txt"),
            "--alleles",
            "A0201\nid -u",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        run_prime_wrapper.main()
    assert excinfo.value.code != 0
    assert "invalid characters" in str(excinfo.value.code)
    assert recorded == [], recorded
