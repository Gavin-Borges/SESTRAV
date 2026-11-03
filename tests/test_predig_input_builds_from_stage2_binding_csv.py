"""PredIG's recombinant input builds from the CSV `rule run_predig` hands it (PC-PG1).

Why this test exists
--------------------
`rule run_predig` (standardize_outputs.smk) passes stage 2's
`results/{proteome_id}_binding.csv` to scripts/run_predig_wrapper.py, which hands it
to `build_recombinant_table` as the peptide metadata file. That function selected
`peptide`, `proteome_id` and `virus`, and functions/stage2_mhc_binding_prediction.py
writes neither of the last two. The selection raised KeyError, the wrapper caught it
as "failed to build recombinant input", and so even with Docker installed the default
pipeline always wrote SIMULATED PredIG scores.

Pinned here: the function takes the proteome id from the caller when the CSV has no
`proteome_id` column, and raises ValueError (never a bare KeyError) when neither
supplies one; a CSV that does carry `proteome_id` keeps precedence, so the
external-validation route through this module's own main() is unchanged; the wrapper
forwards `--proteome-id`; and the rule passes its wildcard, quoted. Docker and
subprocess are mocked throughout. Nothing here runs PredIG.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.build_predig_recombinant_input import build_recombinant_table

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLELE = "HLA-A*02:01"
PEPTIDES = ["GILGFVFTL", "SLYNTVATL"]
PARENT_SEQ = "MKT" + PEPTIDES[0] + "AAAA" + PEPTIDES[1] + "WWW"
PARENT_NAME = "EBNA_TEST"
OUTPUT_COLUMNS = ["epitope", "HLA_allele", "protein_seq", "protein_name"]


def _pairs(tmp_path: Path) -> Path:
    path = tmp_path / "pairs.csv"
    pd.DataFrame({"peptide": PEPTIDES, "allele": [ALLELE] * len(PEPTIDES)}).to_csv(
        path, index=False
    )
    return path


def _stage2_binding(tmp_path: Path) -> Path:
    """The shape stage 2 writes: best-allele columns, protein ids and wide bind_* columns.

    No proteome_id column and no virus column, which is the whole defect.
    """
    path = tmp_path / "EBV_B95_8_panel8_binding.csv"
    pd.DataFrame(
        {
            "peptide": PEPTIDES,
            "allele": [ALLELE] * len(PEPTIDES),
            "affinity": [12.0, 340.0],
            "presentation_score": [0.9, 0.4],
            "presentation_percentile": [0.1, 2.5],
            "protein_id": ["P00001", "P00001"],
            "bind_A0201": [0.9, 0.4],
        }
    ).to_csv(path, index=False)
    return path


def _validation_input(tmp_path: Path) -> Path:
    """The external-validation shape this module's own main() reads: it DOES carry both."""
    path = tmp_path / "external_validation_input.csv"
    pd.DataFrame(
        {
            "peptide": PEPTIDES,
            "proteome_id": ["EBV_B95_8_panel8"] * len(PEPTIDES),
            "virus": ["EBV"] * len(PEPTIDES),
            "label": [1, 0],
        }
    ).to_csv(path, index=False)
    return path


def _proteome_dir(tmp_path: Path) -> Path:
    """EBV's panel FASTA holds the parent; HPV's holds neither peptide."""
    directory = tmp_path / "proteomes"
    directory.mkdir()
    (directory / "EBV_B95_8_panel8.fasta").write_text(
        f">sp|P00001|{PARENT_NAME} test protein\n{PARENT_SEQ}\n", encoding="utf-8"
    )
    (directory / "HPV16_18_panel8.fasta").write_text(
        ">sp|P00002|HPV_TEST other protein\nMHQKRTAMFQDPQERPRKLPQLCTEL\n", encoding="utf-8"
    )
    return directory


# ---------------------------------------------------------------------------
# build_recombinant_table
# ---------------------------------------------------------------------------


def test_a_stage2_binding_csv_builds_with_the_proteome_id_keyword(tmp_path):
    output = tmp_path / "out" / "recombinant.csv"
    out, synthetic = build_recombinant_table(
        pairs_path=str(_pairs(tmp_path)),
        validation_input_path=str(_stage2_binding(tmp_path)),
        proteome_dir=str(_proteome_dir(tmp_path)),
        output_path=str(output),
        proteome_id="EBV_B95_8_panel8",
    )
    assert list(out.columns) == OUTPUT_COLUMNS
    assert out["epitope"].tolist() == PEPTIDES
    assert (out["HLA_allele"] == ALLELE).all()
    # The keyword reached the FASTA lookup: both peptides map to their real parent,
    # not to the synthetic padding used when nothing is found.
    assert synthetic == 0
    assert (out["protein_name"] == PARENT_NAME).all()
    assert (out["protein_seq"] == PARENT_SEQ).all()
    assert list(pd.read_csv(output).columns) == OUTPUT_COLUMNS


def test_a_proteome_with_no_panel_fasta_falls_back_to_synthetic_context(tmp_path):
    """Most pipeline proteome ids have no entry in PROTEOME_FASTA. They still build."""
    out, synthetic = build_recombinant_table(
        pairs_path=str(_pairs(tmp_path)),
        validation_input_path=str(_stage2_binding(tmp_path)),
        proteome_dir=str(tmp_path / "no_such_dir"),
        output_path=str(tmp_path / "recombinant.csv"),
        proteome_id="HBV_ayw_panel4",
    )
    assert list(out.columns) == OUTPUT_COLUMNS
    assert synthetic == len(PEPTIDES)
    assert out["protein_name"].str.startswith("SYNTH_").all()


def test_a_stage2_binding_csv_without_the_keyword_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="proteome_id"):
        build_recombinant_table(
            pairs_path=str(_pairs(tmp_path)),
            validation_input_path=str(_stage2_binding(tmp_path)),
            proteome_dir=str(_proteome_dir(tmp_path)),
            output_path=str(tmp_path / "recombinant.csv"),
        )
    assert not (tmp_path / "recombinant.csv").exists()


def test_a_csv_with_no_peptide_column_is_a_value_error(tmp_path):
    meta = tmp_path / "meta.csv"
    pd.DataFrame({"proteome_id": ["EBV_B95_8_panel8"], "virus": ["EBV"]}).to_csv(
        meta, index=False
    )
    with pytest.raises(ValueError, match="peptide"):
        build_recombinant_table(
            pairs_path=str(_pairs(tmp_path)),
            validation_input_path=str(meta),
            proteome_dir=str(_proteome_dir(tmp_path)),
            output_path=str(tmp_path / "recombinant.csv"),
            proteome_id="EBV_B95_8_panel8",
        )


@pytest.mark.parametrize("keyword", [None, "HPV16_18_panel8"], ids=["no-keyword", "other-keyword"])
def test_a_csv_that_carries_proteome_id_keeps_its_own(tmp_path, keyword):
    """The external-validation route: the CSV's column wins over any keyword.

    With the keyword overriding, every row would look in HPV's FASTA, find nothing,
    and fall through to synthetic context, so `synthetic == 0` is the discriminating
    assertion.
    """
    out, synthetic = build_recombinant_table(
        pairs_path=str(_pairs(tmp_path)),
        validation_input_path=str(_validation_input(tmp_path)),
        proteome_dir=str(_proteome_dir(tmp_path)),
        output_path=str(tmp_path / "recombinant.csv"),
        proteome_id=keyword,
    )
    assert list(out.columns) == OUTPUT_COLUMNS
    assert synthetic == 0
    assert (out["protein_name"] == PARENT_NAME).all()


# ---------------------------------------------------------------------------
# scripts/run_predig_wrapper.py
# ---------------------------------------------------------------------------


def _load_wrapper():
    spec = importlib.util.spec_from_file_location(
        "run_predig_wrapper_input_under_test", REPO_ROOT / "scripts" / "run_predig_wrapper.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_wrapper(tmp_path, monkeypatch, extra_args: list[str]):
    """Run the wrapper with docker detected and `docker run` faked.

    Returns (docker_run_calls, recombinant_input_seen_by_docker, exit_code). The
    recombinant input is read inside the fake, because the wrapper deletes its temp
    files once docker returns.
    """
    wrapper = _load_wrapper()
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
    out = tmp_path / "results" / "EBV_B95_8_panel8_predig_output.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_predig_wrapper.py", "--binding-csv", str(_stage2_binding(tmp_path)),
         "--output", str(out), "--alleles", ALLELE, *extra_args],
    )
    code = None
    try:
        wrapper.main()
    except SystemExit as exc:
        code = exc.code
    return docker_runs, seen.get("input"), code, out


def test_the_wrapper_reaches_docker_when_given_the_proteome_id(tmp_path, monkeypatch, capsys):
    docker_runs, built, code, _out = _run_wrapper(
        tmp_path, monkeypatch, ["--proteome-id", "EBV_B95_8_panel8"]
    )
    assert code == 0
    assert len(docker_runs) == 1, "the PredIG docker command was never invoked"
    assert "/predig/temp_predig_input.csv" in docker_runs[0]
    assert built is not None
    assert list(built.columns) == OUTPUT_COLUMNS
    assert built["epitope"].tolist() == PEPTIDES
    assert "Failed to build recombinant input" not in capsys.readouterr().out


def test_the_wrapper_still_simulates_a_stage2_csv_without_the_proteome_id(
    tmp_path, monkeypatch, capsys
):
    docker_runs, _built, code, out = _run_wrapper(tmp_path, monkeypatch, [])
    assert docker_runs == []
    assert code is None  # the simulated branch returns rather than exiting
    captured = capsys.readouterr()
    assert "Failed to build recombinant input" in captured.out
    assert "proteome_id" in captured.out
    assert "SIMULATED" in captured.err
    rows = pd.read_csv(out)
    assert (rows["simulated"] == 1).all()


# ---------------------------------------------------------------------------
# standardize_outputs.smk
# ---------------------------------------------------------------------------


def _rule_shell(rule: str) -> str:
    """The single string a rule's `shell:` directive hands Snakemake.

    The directive body is implicitly concatenated string literals with comments
    between them; wrapping it in parentheses and evaluating it with `ast` reproduces
    the concatenation and drops the comments.
    """
    text = (REPO_ROOT / "standardize_outputs.smk").read_text(encoding="utf-8")
    assert f"rule {rule}:" in text, f"rule {rule} is no longer in standardize_outputs.smk"
    block = text.split(f"rule {rule}:", 1)[1].split("\nrule ", 1)[0]
    body = block.split("shell:", 1)[1]
    return ast.literal_eval("(" + body + "\n)")


def test_the_rule_passes_its_proteome_id_wildcard_quoted():
    command = _rule_shell("run_predig")
    assert "scripts/run_predig_wrapper.py" in command
    assert "--proteome-id {wildcards.proteome_id:q}" in command
    assert "--binding-csv {input.binding:q}" in command
