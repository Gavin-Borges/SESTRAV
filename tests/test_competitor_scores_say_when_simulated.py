"""The standardized outputs say when a PRIME or PredIG score was simulated.

Why this test exists
--------------------
`rule Results` (pipeline.smk) requires `results/{id}_standardized_outputs.csv`, so
the default Snakemake target runs `rule run_prime` and `rule run_predig`
(standardize_outputs.smk). When PRIME or PredIG (Docker) is not installed, or fails,
their wrappers write seeded noise and still exit 0, and
scripts/standardize_outputs.py merged those scores in with nothing to tell them from
real tool output (to-do PC-9). A benchmark built on that file would compare SESTRAV
against noise. PRIME's simulated score is derived from the binding input; PredIG's
usually is not, because no stage-2 column matches an allele name, so it is noise
around a constant.

Pinned here: each wrapper marks every simulated row `simulated=1`; the standardized
CSV carries `prime_simulated` and `predig_simulated`, each following its own tool's
file; the columns that were already there keep their order. A 0 means the file
carries no simulation mark, not that the scores are known to be real: a file written
before the wrappers marked their output reads 0. Whether to simulate at all, or
refuse, is unchanged and is an owner decision.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLELE = "HLA-A*02:01"
PEPTIDES = ["GILGFVFTL", "SLYNTVATL", "NLVPMVATV"]
EXISTING_COLUMNS = [
    "peptide",
    "hla",
    "binding_ic50",
    "prime_score",
    "predig_score",
    "sestrav_score",
    "assay_type",
    "hard_negative_flag",
]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_simulation_under_test", REPO_ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _binding(tmp_path: Path) -> Path:
    path = tmp_path / "binding.csv"
    pd.DataFrame({"peptide": PEPTIDES, ALLELE: [0.9, 0.4, 0.1]}).to_csv(path, index=False)
    return path


def _no_tool(*args, **kwargs):
    raise OSError("tool not installed (test)")


def _simulate_prime(tmp_path, monkeypatch) -> Path:
    wrapper = _load("run_prime_wrapper")
    real_isfile = wrapper.os.path.isfile
    monkeypatch.setattr(
        wrapper.os.path, "isfile", lambda p: False if "PRIME" in str(p) else real_isfile(p)
    )
    monkeypatch.setattr(wrapper.subprocess, "run", _no_tool)
    out = tmp_path / "prime_output.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_prime_wrapper.py", "--binding-csv", str(_binding(tmp_path)), "--output", str(out),
         "--alleles", ALLELE],
    )
    wrapper.main()
    return out


def _simulate_predig(tmp_path, monkeypatch) -> Path:
    wrapper = _load("run_predig_wrapper")
    monkeypatch.chdir(tmp_path)  # the wrapper writes its temp files under ./results
    monkeypatch.setattr(wrapper.subprocess, "run", _no_tool)
    out = tmp_path / "predig_output.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_predig_wrapper.py", "--binding-csv", str(_binding(tmp_path)), "--output", str(out),
         "--alleles", ALLELE],
    )
    wrapper.main()
    return out


def _standardize(tmp_path, monkeypatch, prime: Path, predig: Path) -> pd.DataFrame:
    sestrav = tmp_path / "ranked.csv"
    pd.DataFrame({"peptide": PEPTIDES, "immunogenicity_score": [0.7, 0.5, 0.2]}).to_csv(
        sestrav, index=False
    )
    out = tmp_path / "standardized_outputs.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["standardize_outputs.py", "--binding", str(_binding(tmp_path)), "--prime", str(prime),
         "--predig", str(predig), "--sestrav", str(sestrav), "--output", str(out)],
    )
    _load("standardize_outputs").main()
    return pd.read_csv(out)


def test_the_prime_wrapper_marks_every_simulated_row(tmp_path, monkeypatch, capsys):
    out = _simulate_prime(tmp_path, monkeypatch)
    rows = pd.read_csv(out, sep="\t")
    assert len(rows) == len(PEPTIDES)
    assert (rows["simulated"] == 1).all()
    assert "SIMULATED" in capsys.readouterr().err


def test_the_predig_wrapper_marks_every_simulated_row(tmp_path, monkeypatch, capsys):
    out = _simulate_predig(tmp_path, monkeypatch)
    rows = pd.read_csv(out)
    assert len(rows) == len(PEPTIDES)
    assert (rows["simulated"] == 1).all()
    assert "SIMULATED" in capsys.readouterr().err


def test_simulated_scores_are_flagged_in_the_standardized_csv(tmp_path, monkeypatch, capsys):
    prime = _simulate_prime(tmp_path, monkeypatch)
    predig = _simulate_predig(tmp_path, monkeypatch)
    capsys.readouterr()
    merged = _standardize(tmp_path, monkeypatch, prime, predig)
    assert list(merged.columns) == EXISTING_COLUMNS + ["prime_simulated", "predig_simulated"]
    assert merged["prime_score"].notna().all() and merged["predig_score"].notna().all()
    assert (merged["prime_simulated"] == 1).all()
    assert (merged["predig_simulated"] == 1).all()
    err = capsys.readouterr().err
    assert "PRIME scores are SIMULATED" in err and "PredIG scores are SIMULATED" in err


def test_each_flag_follows_its_own_tool(tmp_path, monkeypatch, capsys):
    """One tool simulated and the other not: the two flags must not be swapped or shared."""
    prime = _simulate_prime(tmp_path, monkeypatch)
    predig = tmp_path / "predig_real.csv"
    pd.DataFrame(
        {"peptide": PEPTIDES, "allele": [ALLELE] * 3, "predig_score": [0.6, 0.5, 0.4]}
    ).to_csv(predig, index=False)
    capsys.readouterr()
    merged = _standardize(tmp_path, monkeypatch, prime, predig)
    assert (merged["prime_simulated"] == 1).all()
    assert (merged["predig_simulated"] == 0).all()
    err = capsys.readouterr().err
    assert "PRIME scores are SIMULATED" in err
    assert "PredIG scores are SIMULATED" not in err


def test_real_tool_output_is_not_flagged(tmp_path, monkeypatch, capsys):
    prime = tmp_path / "prime_real.txt"
    pd.DataFrame({"Peptide": PEPTIDES, "PRIME_score": [0.3, 0.2, 0.1]}).to_csv(
        prime, sep="\t", index=False
    )
    predig = tmp_path / "predig_real.csv"
    pd.DataFrame({"peptide": PEPTIDES, "allele": [ALLELE] * 3, "predig_score": [0.6, 0.5, 0.4]}).to_csv(
        predig, index=False
    )
    merged = _standardize(tmp_path, monkeypatch, prime, predig)
    assert (merged["prime_simulated"] == 0).all()
    assert (merged["predig_simulated"] == 0).all()
    assert "SIMULATED" not in capsys.readouterr().err
