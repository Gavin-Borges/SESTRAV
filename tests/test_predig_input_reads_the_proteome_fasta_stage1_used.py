"""PredIG's recombinant input takes protein context from the FASTA the peptides came from.

Why this test exists
--------------------
`rule run_predig` hands stage 2's binding CSV to `build_recombinant_table`, which looks
each peptide up in a parent protein so PredIG (`--type recombinant`) can score it in
context. The lookup went through `PROTEOME_FASTA`, a hand-written table naming two
panels, `EBV_B95_8_panel8` and `HPV16_18_panel8`. `config.yaml` runs eight antigens, so
for the other six (`HBV_ayw_panel4`, `HCV_1a_panel4`, `HIV1_HXB2_panel4`,
`SARSCOV2_wuhan1_panel4`, `IAV_PR8_panel4`, `CMV_AD169_panel4`) every peptide fell
through to `synthetic_protein_context`: the peptide padded with glycine. Once the
recombinant input could be built at all, with Docker present those rows became REAL
PredIG output computed on that synthetic context, and nothing in the output said so.

The peptides were never without a parent. `rule generate_peptides` cut every one of them
out of the FASTA that `config.yaml`'s `proteome_files` names for that antigen. So the
rule now passes that same file, by reusing `generate_peptides`' own input function
(`rules.generate_peptides.input.fasta`) rather than restating the lookup, and the two
rules cannot resolve a proteome to different files.

What is asserted
----------------
- `build_recombinant_table(proteome_fasta=...)` takes context from that file, for a
  proteome the legacy table does not know, and searches it BEFORE the table;
- it applies only to the rows of the proteome it belongs to, is refused without that
  proteome's id, and a FASTA that does not exist is an error rather than a silent
  fall-through to synthetic context;
- a peptide the file does not hold still falls back, and is still counted;
- the wrapper forwards `--proteome-fasta`, and Docker is then handed real context;
- for EVERY antigen `config.yaml` runs, a Snakemake dry run hands `run_predig` exactly
  the FASTA `config.yaml` names, that file is tracked, and a peptide cut from it maps
  to a real protein, never to synthetic context.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.build_predig_recombinant_input import PROTEOME_FASTA, build_recombinant_table, parse_fasta

yaml = pytest.importorskip("yaml", reason="the shipped config.yaml is YAML")

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLELE = "HLA-A*02:01"
PEPTIDES = ["GILGFVFTL", "SLYNTVATL"]
PARENT_SEQ = "MKT" + PEPTIDES[0] + "AAAA" + PEPTIDES[1] + "WWW"
PARENT_NAME = "STAGE1_PARENT"
# A proteome the legacy PROTEOME_FASTA table does not name.
UNTABLED = "HBV_ayw_panel4"


def _pairs(tmp_path: Path, peptides=PEPTIDES) -> Path:
    path = tmp_path / "pairs.csv"
    pd.DataFrame({"peptide": peptides, "allele": [ALLELE] * len(peptides)}).to_csv(
        path, index=False
    )
    return path


def _stage2_binding(tmp_path: Path, peptides=PEPTIDES) -> Path:
    """The shape stage 2 writes: no proteome_id column and no virus column."""
    path = tmp_path / f"{UNTABLED}_binding.csv"
    pd.DataFrame(
        {
            "peptide": peptides,
            "allele": [ALLELE] * len(peptides),
            "presentation_score": [0.9] * len(peptides),
            "bind_A0201": [0.9] * len(peptides),
        }
    ).to_csv(path, index=False)
    return path


def _fasta(
    tmp_path: Path, name: str = PARENT_NAME, seq: str = PARENT_SEQ, file_name: str = "stage1.fasta"
) -> Path:
    path = tmp_path / file_name
    path.write_text(
        f">sp|Q99999|{name} the protein stage 1 cut the peptides from\n{seq}\n", encoding="utf-8"
    )
    return path


def _build(tmp_path: Path, **kwargs):
    """Call build_recombinant_table, writing a default input only where none was passed.

    The defaults are created lazily on purpose: written eagerly they would land on the
    same paths as a caller's own inputs and overwrite them.
    """
    kwargs.setdefault("proteome_dir", str(tmp_path / "no_panel_dir"))
    kwargs.setdefault("output_path", str(tmp_path / "recombinant.csv"))
    if "pairs_path" not in kwargs:
        kwargs["pairs_path"] = str(_pairs(tmp_path))
    if "validation_input_path" not in kwargs:
        kwargs["validation_input_path"] = str(_stage2_binding(tmp_path))
    return build_recombinant_table(**kwargs)


# ---------------------------------------------------------------------------
# build_recombinant_table
# ---------------------------------------------------------------------------


def test_the_premise_the_legacy_table_does_not_know_the_proteome():
    assert UNTABLED not in PROTEOME_FASTA


def test_without_the_fasta_an_untabled_proteome_gets_synthetic_context(tmp_path):
    """The control: this is the behaviour the pipeline produced for six of eight antigens."""
    out, synthetic = _build(tmp_path, proteome_id=UNTABLED)
    assert synthetic == len(PEPTIDES)
    assert out["protein_name"].str.startswith("SYNTH_").all()


def test_the_given_fasta_supplies_real_context_for_an_untabled_proteome(tmp_path):
    out, synthetic = _build(tmp_path, proteome_id=UNTABLED, proteome_fasta=str(_fasta(tmp_path)))
    assert synthetic == 0
    assert (out["protein_name"] == PARENT_NAME).all()
    assert (out["protein_seq"] == PARENT_SEQ).all()


def test_the_given_fasta_is_searched_before_the_legacy_table(tmp_path):
    """For a proteome the table DOES name, the file the peptides came from still wins."""
    tabled = "EBV_B95_8_panel8"
    assert tabled in PROTEOME_FASTA
    panel_dir = tmp_path / "panels"
    panel_dir.mkdir()
    (panel_dir / PROTEOME_FASTA[tabled]).write_text(
        f">sp|P00001|LEGACY_PANEL legacy\n{PARENT_SEQ}\n", encoding="utf-8"
    )
    out, synthetic = _build(
        tmp_path,
        proteome_dir=str(panel_dir),
        proteome_id=tabled,
        proteome_fasta=str(_fasta(tmp_path)),
    )
    assert synthetic == 0
    assert (out["protein_name"] == PARENT_NAME).all(), "the legacy table outranked the given FASTA"


def test_a_peptide_the_given_fasta_lacks_still_falls_back_and_is_counted(tmp_path):
    only_first = "MKT" + PEPTIDES[0] + "WWW"
    out, synthetic = _build(
        tmp_path, proteome_id=UNTABLED, proteome_fasta=str(_fasta(tmp_path, seq=only_first))
    )
    assert synthetic == 1
    by_peptide = dict(zip(out["epitope"], out["protein_name"]))
    assert by_peptide[PEPTIDES[0]] == PARENT_NAME
    assert by_peptide[PEPTIDES[1]].startswith("SYNTH_")


def test_the_given_fasta_applies_only_to_its_own_proteome_rows(tmp_path):
    """A metadata CSV that carries its own proteome ids keeps them; only matching rows
    read the file, even when the file happens to hold the other rows' peptides too."""
    meta = tmp_path / "external_validation_input.csv"
    pd.DataFrame(
        {"peptide": PEPTIDES, "proteome_id": ["SOME_OTHER_PANEL", UNTABLED], "virus": ["", ""]}
    ).to_csv(meta, index=False)
    out, synthetic = _build(
        tmp_path,
        validation_input_path=str(meta),
        proteome_id=UNTABLED,
        proteome_fasta=str(_fasta(tmp_path)),
    )
    by_peptide = dict(zip(out["epitope"], out["protein_name"]))
    assert by_peptide[PEPTIDES[1]] == PARENT_NAME
    assert by_peptide[PEPTIDES[0]].startswith("SYNTH_"), "another proteome's row read this FASTA"
    assert synthetic == 1


def test_a_named_fasta_that_does_not_exist_is_an_error_not_synthetic_context(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.fasta"):
        _build(tmp_path, proteome_id=UNTABLED, proteome_fasta=str(tmp_path / "missing.fasta"))


def test_a_fasta_without_the_proteome_id_it_belongs_to_is_refused(tmp_path):
    """Measured with a metadata CSV that CARRIES its own proteome_id column, which is the
    one case this check decides. With stage 2's CSV the older "no proteome_id column and
    no proteome_id was given" error fires first, so a test built on it passes whether or
    not this check exists; mutation testing showed exactly that. Here, without the check,
    the FASTA would be loaded and then silently match no row."""
    meta = tmp_path / "external_validation_input.csv"
    pd.DataFrame({"peptide": PEPTIDES, "proteome_id": [UNTABLED] * len(PEPTIDES)}).to_csv(
        meta, index=False
    )
    with pytest.raises(ValueError, match="names the FASTA of one proteome"):
        _build(tmp_path, validation_input_path=str(meta), proteome_fasta=str(_fasta(tmp_path)))


# ---------------------------------------------------------------------------
# scripts/run_predig_wrapper.py
# ---------------------------------------------------------------------------


def _load_wrapper():
    spec = importlib.util.spec_from_file_location(
        "run_predig_wrapper_fasta_under_test", REPO_ROOT / "scripts" / "run_predig_wrapper.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_wrapper(tmp_path, monkeypatch, extra_args):
    """Run the wrapper with Docker detected and `docker run` faked; return what Docker saw."""
    wrapper = _load_wrapper()
    binding = _stage2_binding(tmp_path)
    monkeypatch.chdir(tmp_path)  # the wrapper writes its temp files under ./results
    docker_runs: list[list[str]] = []
    seen: dict[str, pd.DataFrame] = {}

    def fake_run(cmd, *args, **kwargs):
        if list(cmd[:2]) == ["docker", "--version"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=b"Docker version test", stderr=b"")
        if list(cmd[:2]) == ["docker", "run"]:
            docker_runs.append(list(cmd))
            seen["input"] = pd.read_csv(tmp_path / "results" / "temp_predig_input.csv")
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"unexpected subprocess call: {cmd!r}")

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)
    out = tmp_path / "results" / f"{UNTABLED}_predig_output.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_predig_wrapper.py",
            "--binding-csv",
            str(binding),
            "--output",
            str(out),
            "--alleles",
            ALLELE,
            "--proteome-id",
            UNTABLED,
            *extra_args,
        ],
    )
    code = None
    try:
        wrapper.main()
    except SystemExit as exc:
        code = exc.code
    return docker_runs, seen.get("input"), code, out


def test_the_wrapper_forwards_the_fasta_and_docker_is_handed_real_context(tmp_path, monkeypatch):
    fasta = _fasta(tmp_path)
    docker_runs, built, code, _out = _run_wrapper(
        tmp_path, monkeypatch, ["--proteome-fasta", str(fasta)]
    )
    assert code == 0
    assert len(docker_runs) == 1, "the PredIG docker command was never invoked"
    assert built is not None
    assert (built["protein_name"] == PARENT_NAME).all(), "Docker was handed synthetic context"


def test_the_wrapper_reports_a_missing_fasta_and_simulates_rather_than_running(
    tmp_path, monkeypatch, capsys
):
    docker_runs, _built, code, out = _run_wrapper(
        tmp_path, monkeypatch, ["--proteome-fasta", str(tmp_path / "missing.fasta")]
    )
    assert docker_runs == [], "PredIG ran on an input built without the named FASTA"
    assert code is None  # the simulated branch returns rather than exiting
    assert "Failed to build recombinant input" in capsys.readouterr().out
    assert (pd.read_csv(out)["simulated"] == 1).all()


# ---------------------------------------------------------------------------
# standardize_outputs.smk, and every antigen config.yaml runs
# ---------------------------------------------------------------------------


def _rule_block(rule: str) -> str:
    text = (REPO_ROOT / "standardize_outputs.smk").read_text(encoding="utf-8")
    assert f"rule {rule}:" in text, f"rule {rule} is no longer in standardize_outputs.smk"
    return text.split(f"rule {rule}:", 1)[1].split("\nrule ", 1)[0]


def test_the_rule_reuses_generate_peptides_own_fasta_lookup():
    block = _rule_block("run_predig")
    assert re.search(r"\bfasta\s*=\s*rules\.generate_peptides\.input\.fasta\b", block), (
        "run_predig must take its FASTA from generate_peptides' input function, so the two "
        "rules cannot resolve a proteome to different files"
    )
    shell = ast.literal_eval("(" + block.split("shell:", 1)[1] + "\n)")
    assert "--proteome-fasta {input.fasta:q}" in shell


def _snakemake() -> str:
    """PATH first, then the launcher beside the running interpreter (as the quoting tests do)."""
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


def _shipped_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))


SHIPPED_ANTIGENS = _shipped_config()["antigens"]


@pytest.fixture(scope="module")
def dry_run_commands() -> dict[str, str]:
    """One forced dry run of run_predig for every shipped antigen: {proteome id: command}."""
    targets = [f"results/{antigen}_predig_output.csv" for antigen in SHIPPED_ANTIGENS]
    result = subprocess.run(
        [
            _snakemake(),
            *targets,
            "--snakefile",
            "pipeline.smk",
            "--forcerun",
            "run_predig",
            "--dry-run",
            "--printshellcmds",
            "--cores",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    commands: dict[str, str] = {}
    for line in output.splitlines():
        if "scripts/run_predig_wrapper.py" not in line:
            continue
        match = re.search(r"--proteome-id\s+['\"]?([A-Za-z0-9_.-]+)", line)
        if match:
            commands[match.group(1)] = line
    return commands


def _tracked(path: str) -> bool:
    listed = subprocess.run(
        ["git", "ls-files", "--", path], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return bool(listed.strip())


@pytest.mark.parametrize("antigen", SHIPPED_ANTIGENS)
def test_each_shipped_antigen_is_handed_the_fasta_config_names(antigen, dry_run_commands):
    assert antigen in dry_run_commands, f"no run_predig command was printed for {antigen}"
    match = re.search(r"--proteome-fasta\s+['\"]?([^'\"\s]+)", dry_run_commands[antigen])
    assert match, f"run_predig for {antigen} is not passed --proteome-fasta"
    handed = match.group(1).replace("\\", "/")
    expected = _shipped_config()["proteome_files"][antigen]
    assert handed == expected, f"{antigen}: run_predig got {handed}, config.yaml names {expected}"
    assert _tracked(expected), f"{antigen}'s FASTA {expected} is not a tracked file"


@pytest.mark.parametrize("antigen", SHIPPED_ANTIGENS)
def test_each_shipped_antigen_gets_real_protein_context(antigen, tmp_path):
    """A peptide cut from the antigen's own FASTA maps to its protein, never to padding."""
    fasta = REPO_ROOT / _shipped_config()["proteome_files"][antigen]
    proteins = parse_fasta(str(fasta))
    assert proteins, f"{fasta} holds no protein"
    name, (_header, sequence) = next(iter(proteins.items()))
    peptide = sequence[:9]
    assert len(peptide) == 9, f"the first protein in {fasta} is shorter than a 9-mer"
    out, synthetic = _build(
        tmp_path,
        pairs_path=str(_pairs(tmp_path, [peptide])),
        validation_input_path=str(_stage2_binding(tmp_path, [peptide])),
        proteome_id=antigen,
        proteome_fasta=str(fasta),
    )
    assert synthetic == 0, f"{antigen}: a peptide from its own FASTA got synthetic context"
    assert out["protein_name"].tolist() == [name]
