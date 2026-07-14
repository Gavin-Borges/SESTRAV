"""Tests for the NetMHCpan 4.1 leave-one-out scoring harness.

These tests validate the harness WITHOUT the license-gated NetMHCpan binary by
using a synthetic fixture that mimics NetMHCpan 4.1 ``-p`` output (real column
names, whitespace delimiting, and '#'/'---' comment lines). They cover:

* allele-format conversion (A/B/C, idempotency);
* header-driven column location and score extraction on the fixture;
* a hand-checkable AUC over a tiny labelled set;
* per-(virus, allele) grouping and allele_cli strings from a tiny TSV.

The tests do not require netMHCpan to be installed and do not depend on the full
nine real held-out files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sklearn.metrics import roc_auc_score

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import parse_netmhcpan_loo as parse_mod  # noqa: E402
import prepare_netmhcpan_inputs as prep_mod  # noqa: E402
from _netmhcpan_common import (  # noqa: E402
    convert_allele_to_cli,
    find_column,
    is_header_line,
    parse_header_columns,
)

# A synthetic NetMHCpan-4.1-style -BA output block. Column names mirror the real
# tool: Pos, MHC, Peptide, Core, ..., EL_Score, %Rank_EL, BA_Score, %Rank_BA.
# Peptide/score pairs are chosen so the AUC is hand-checkable (see test below).
SYNTHETIC_NETMHCPAN_OUTPUT = """\
# NetMHCpan version 4.1b
# Input is in PEPTIDE format
# Peptide length 9
-----------------------------------------------------------------------------------
 Pos         MHC        Peptide  Core Of Gp Gl Ip Il        Icore   Identity   Score_EL %Rank_EL  Score_BA %Rank_BA  Aff(nM) BindLevel
-----------------------------------------------------------------------------------
   1  HLA-A*02:01      AAAWYLWEV AAAWYLWEV  0  0  0  0  0    AAAWYLWEV       PEPLIST   0.968540    0.071  0.812300    0.100    12.34 <= SB
   1  HLA-A*02:01      SLYNTVATL SLYNTVATL  0  0  0  0  0    SLYNTVATL       PEPLIST   0.742100    0.510  0.601200    0.640   210.5 <= WB
   1  HLA-A*02:01      GILGFVFTL GILGFVFTL  0  0  0  0  0    GILGFVFTL       PEPLIST   0.501000    2.100  0.400000    3.000  1200.0
   1  HLA-A*02:01      KKKKKKKKK KKKKKKKKK  0  0  0  0  0    KKKKKKKKK       PEPLIST   0.012000   45.000  0.020000   50.000 40000.0
-----------------------------------------------------------------------------------
"""


# ---------------------------------------------------------------------------
# Allele conversion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("HLA-B*07:02", "HLA-B07:02"),
        ("HLA-A*02:01", "HLA-A02:01"),
        ("HLA-C*07:01", "HLA-C07:01"),
        ("  HLA-B*44:02  ", "HLA-B44:02"),
        ("HLA-B53", "HLA-B53"),
    ],
)
def test_convert_allele_basic(raw: str, expected: str) -> None:
    """Asterisk is stripped, colon kept, whitespace trimmed, A/B/C handled."""
    assert convert_allele_to_cli(raw) == expected


def test_convert_allele_idempotent() -> None:
    """Applying the conversion twice equals applying it once."""
    for raw in ("HLA-B*07:02", "HLA-A*02:01", "HLA-C*07:01", "HLA-B53"):
        once = convert_allele_to_cli(raw)
        assert convert_allele_to_cli(once) == once


# ---------------------------------------------------------------------------
# Header location + parsing on the synthetic fixture
# ---------------------------------------------------------------------------
def _write_fixture(tmp_path: Path) -> Path:
    """Write the synthetic NetMHCpan output to a temp file and return its path."""
    out = tmp_path / "HLA-A02_01.txt"
    out.write_text(SYNTHETIC_NETMHCPAN_OUTPUT, encoding="utf-8")
    return out


def test_header_detection_and_column_location() -> None:
    """The header row is detected and named columns resolve to indices."""
    header = (
        " Pos MHC Peptide Core Of Gp Gl Ip Il Icore Identity "
        "Score_EL %Rank_EL Score_BA %Rank_BA Aff(nM) BindLevel"
    )
    assert is_header_line(header)
    assert not is_header_line("# comment")
    assert not is_header_line("-------------")
    cols = parse_header_columns(header)
    assert find_column(cols, ("Peptide",)) is not None
    assert find_column(cols, ("MHC",)) is not None
    # EL score candidates include Score_EL; percentile is %Rank_EL.
    assert find_column(cols, ("Score_EL",)) is not None
    assert find_column(cols, ("%Rank_EL",)) is not None


def test_parse_extracts_el_scores(tmp_path: Path) -> None:
    """Parser prefers the raw EL score and reads the correct value per peptide."""
    fixture = _write_fixture(tmp_path)
    scores = parse_mod.parse_netmhcpan_file(fixture)
    # Keys are (peptide, allele_cli); allele_cli strips the asterisk.
    assert scores[("AAAWYLWEV", "HLA-A02:01")] == pytest.approx(0.968540)
    assert scores[("SLYNTVATL", "HLA-A02:01")] == pytest.approx(0.742100)
    assert scores[("GILGFVFTL", "HLA-A02:01")] == pytest.approx(0.501000)
    assert scores[("KKKKKKKKK", "HLA-A02:01")] == pytest.approx(0.012000)


def test_parse_rank_fallback(tmp_path: Path) -> None:
    """When only %Rank_EL is present, predictor = 1 - rank/100 (higher=better)."""
    rank_only = "\n".join(
        [
            "# rank-only fixture",
            "----------------------------",
            " Pos MHC Peptide %Rank_EL",
            "----------------------------",
            "  1 HLA-A*02:01 AAAWYLWEV 0.100",
            "  1 HLA-A*02:01 KKKKKKKKK 90.000",
            "----------------------------",
        ]
    )
    path = tmp_path / "rank_only.txt"
    path.write_text(rank_only + "\n", encoding="utf-8")
    scores = parse_mod.parse_netmhcpan_file(path)
    assert scores[("AAAWYLWEV", "HLA-A02:01")] == pytest.approx(1 - 0.100 / 100)
    assert scores[("KKKKKKKKK", "HLA-A02:01")] == pytest.approx(1 - 90.0 / 100)
    # Stronger (lower) rank yields the higher predictor.
    assert scores[("AAAWYLWEV", "HLA-A02:01")] > scores[("KKKKKKKKK", "HLA-A02:01")]


def test_hand_checkable_auc(tmp_path: Path) -> None:
    """AUC over a tiny labelled set matches a hand computation.

    Labels: AAAWYLWEV=1, SLYNTVATL=1, GILGFVFTL=0, KKKKKKKKK=0. The two
    positives have the two highest EL scores, so ranking is perfect and the
    expected AUC is 1.0.
    """
    fixture = _write_fixture(tmp_path)
    scores = parse_mod.parse_netmhcpan_file(fixture)
    labels = [1, 1, 0, 0]
    predictors = [
        scores[("AAAWYLWEV", "HLA-A02:01")],
        scores[("SLYNTVATL", "HLA-A02:01")],
        scores[("GILGFVFTL", "HLA-A02:01")],
        scores[("KKKKKKKKK", "HLA-A02:01")],
    ]
    assert roc_auc_score(labels, predictors) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# prepare_netmhcpan_inputs grouping
# ---------------------------------------------------------------------------
TINY_TSV = "\n".join(
    [
        "peptide\thla_allele\tlabel\tnegative_origin\treference_pmid",
        "AAAWYLWEV\tHLA-A*02:01\t1\t\t111",
        "AAAWYLWEV\tHLA-A*02:01\t1\t\t111",  # duplicate -> deduped in the group
        "SLYNTVATL\tHLA-A*02:01\t0\ttested_negative\t222",
        "GILGFVFTL\tHLA-B*07:02\t1\t\t333",
        "CCXCCCCCC\tHLA-B*07:02\t1\t\t444",  # invalid AA 'X' -> dropped
    ]
)


def test_prepare_grouping_and_allele_cli(tmp_path: Path) -> None:
    """Groups form per (virus, allele) with correct CLI strings and dedup."""
    test_dir = tmp_path / "loo"
    test_dir.mkdir()
    (test_dir / "TESTV_held_out.tsv").write_text(TINY_TSV + "\n", encoding="utf-8")
    out_dir = tmp_path / "ext_scores"

    records = prep_mod.prepare_inputs(test_dir, out_dir)

    keyed = {(r["virus"], r["allele_cli"]): r for r in records}
    assert set(keyed) == {("TESTV", "HLA-A02:01"), ("TESTV", "HLA-B07:02")}

    a_group = keyed[("TESTV", "HLA-A02:01")]
    assert a_group["allele_original"] == "HLA-A*02:01"
    assert a_group["n_peptides"] == 2  # duplicate collapsed

    b_group = keyed[("TESTV", "HLA-B07:02")]
    # The invalid-AA peptide is dropped, leaving one clean peptide.
    assert b_group["n_peptides"] == 1
    pep_file = Path(str(b_group["peptide_list_path"]))
    assert pep_file.read_text(encoding="utf-8").split() == ["GILGFVFTL"]

    # Manifest and runner are produced.
    assert (out_dir / "inputs" / "manifest.csv").is_file()
    runner = out_dir / "run_netmhcpan.sh"
    assert runner.is_file()
    runner_text = runner.read_text(encoding="utf-8")
    assert "netMHCpan -p" in runner_text
    assert '-a "HLA-A02:01"' in runner_text
    assert "-BA -xls" in runner_text


def test_prepare_is_deterministic(tmp_path: Path) -> None:
    """Re-running prepare_inputs yields byte-identical peptide lists and manifest."""
    test_dir = tmp_path / "loo"
    test_dir.mkdir()
    (test_dir / "TESTV_held_out.tsv").write_text(TINY_TSV + "\n", encoding="utf-8")

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    prep_mod.prepare_inputs(test_dir, out1)
    prep_mod.prepare_inputs(test_dir, out2)

    man1 = (out1 / "inputs" / "manifest.csv").read_text(encoding="utf-8")
    man2 = (out2 / "inputs" / "manifest.csv").read_text(encoding="utf-8")
    # Paths differ by root, so compare only the row shape (allele/n_peptides).
    def rows(text: str) -> list[str]:
        return [",".join(line.split(",")[:2] + line.split(",")[-1:]) for line in text.splitlines()]

    assert rows(man1) == rows(man2)


def test_unscorable_allele_is_skipped_in_runner(tmp_path: Path) -> None:
    """A bare 'HLA-A' allele is recorded but commented out of the runner."""
    tsv = "\n".join(
        [
            "peptide\thla_allele\tlabel\tnegative_origin\treference_pmid",
            "AAAWYLWEV\tHLA-A\t1\t\t111",
        ]
    )
    test_dir = tmp_path / "loo"
    test_dir.mkdir()
    (test_dir / "TESTV_held_out.tsv").write_text(tsv + "\n", encoding="utf-8")
    out_dir = tmp_path / "ext_scores"

    records = prep_mod.prepare_inputs(test_dir, out_dir)
    assert records[0]["allele_cli"] == "HLA-A"
    runner_text = (out_dir / "run_netmhcpan.sh").read_text(encoding="utf-8")
    assert "SKIPPED (unscorable allele)" in runner_text
    assert "netMHCpan -p" not in runner_text
