"""Regression tests: reference_pmid must be written as a bare integer string.

A PMID column that carries blanks is inferred as float64 by pandas, so any
plain ``.astype(str)`` round trip renders 38923358 as "38923358.0". Every
downstream join against a real PubMed identifier then misses, returning an
empty result that reads exactly like a confident "no matching rows".

These tests pin the writer side of that contract across the v5 ingest chain:
the shared normalizer in scripts/_dataset_utils.py, the two frames that build
it (scripts/ingest_iedb_negatives.py, scripts/merge_iedb_api_negatives.py),
the scripts/build_dataset_v5.py round trip that re-reads those CSVs, and the
scripts/export_loo_test_sets.py re-emission.

Assertions on written files read the raw text with the csv module, never with
pandas, because pandas re-infers the column and would hide the defect.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._dataset_utils import normalize_reference_pmids
from scripts.build_dataset_v5 import main as build_v5_main
from scripts.export_loo_test_sets import build_test_partition, load_dataset
from scripts.ingest_iedb_negatives import ResolvedColumns, build_output, resolve_columns
from scripts.merge_iedb_api_negatives import _normalize_to_schema

LOGGER = logging.getLogger("test_reference_pmid")

SCHEMA_PATH = Path("data/immunogenicity_dataset_v5_schema.json")

PEP_A = "GILGFVFTL"
PEP_B = "NLVPMVATV"
PEP_C = "CINGVCWTV"


def _raw_column(path: Path, column: str, delimiter: str = ",") -> list[str]:
    """Read one column as raw text, bypassing pandas type inference."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return [row[column] for row in csv.DictReader(fh, delimiter=delimiter)]


# ---------------------------------------------------------------------------
# normalize_reference_pmids: the shared contract
# ---------------------------------------------------------------------------


def test_normalize_strips_the_float_suffix_from_inferred_floats() -> None:
    """The exact shape pandas produces: a numeric column holding a blank."""
    series = pd.Series([38923358.0, np.nan, 21918184.0])
    assert list(normalize_reference_pmids(series)) == ["38923358", None, "21918184"]


def test_normalize_strips_the_float_suffix_from_round_trip_text() -> None:
    """Text of the form 38923358.0, read back from a CSV the old path wrote."""
    series = pd.Series(["38923358.0", "21918184.0"])
    assert list(normalize_reference_pmids(series)) == ["38923358", "21918184"]


def test_normalize_keeps_bare_integer_strings_unchanged() -> None:
    series = pd.Series(["38923358", " 21918184 ", 33853928])
    assert list(normalize_reference_pmids(series)) == ["38923358", "21918184", "33853928"]


def test_normalize_renders_every_missing_form_as_none() -> None:
    """Missing stays null, never the text nan and never an empty-looking value."""
    series = pd.Series([None, np.nan, "", "  ", "nan", "NaN", "None", pd.NA], dtype=object)
    assert list(normalize_reference_pmids(series)) == [None] * 8


def test_normalize_passes_free_text_references_through() -> None:
    """LANL exports carry free-text references, not bare PMIDs."""
    series = pd.Series(["Plana2004 PMID:15213562", "Llano2019"])
    assert list(normalize_reference_pmids(series)) == [
        "Plana2004 PMID:15213562",
        "Llano2019",
    ]


# ---------------------------------------------------------------------------
# scripts/ingest_iedb_negatives.py build_output
# ---------------------------------------------------------------------------


def _iedb_frame(pmids: list[object]) -> tuple[pd.DataFrame, ResolvedColumns]:
    n = len(pmids)
    df = pd.DataFrame(
        {
            "Host Organism Name": ["Homo sapiens"] * n,
            "Qualitative Measure": ["Negative"] * n,
            "Epitope Name": [PEP_A, PEP_B, PEP_C][:n],
            "Allele Name": ["HLA-A*02:01"] * n,
            "Assay Group": ["IFN-gamma ELISpot"] * n,
            "Antigen Name": ["Matrix protein 1"] * n,
            "Reference PubMed ID": pmids,
            "Description": ["Influenza A virus"] * n,
            "Response Frequency As Reported": [""] * n,
            "Assay ID": [f"AS-{i}" for i in range(n)],
        }
    )
    return df, resolve_columns(df, LOGGER)


def test_build_output_writes_bare_integer_pmid_from_a_float_column() -> None:
    """A blank in the IEDB export makes pandas hand build_output float64."""
    df, cols = _iedb_frame([38923358.0, np.nan, 21918184.0])
    assert df["Reference PubMed ID"].dtype == np.float64  # the defect precondition

    out = build_output(df, cols, LOGGER)

    assert list(out["reference_pmid"]) == ["38923358", None, "21918184"]


# ---------------------------------------------------------------------------
# scripts/merge_iedb_api_negatives.py _normalize_to_schema
# ---------------------------------------------------------------------------


def test_merge_normalize_to_schema_writes_bare_integer_pmid() -> None:
    raw = pd.DataFrame(
        {
            "peptide": [PEP_A, PEP_B],
            "reference_pmid": [39753970.0, np.nan],
        }
    )
    out = _normalize_to_schema(raw)
    assert list(out["reference_pmid"]) == ["39753970", None]


# ---------------------------------------------------------------------------
# scripts/build_dataset_v5.py: the CSV round trip that re-reads its components
# ---------------------------------------------------------------------------


def _write_base_v4(path: Path) -> None:
    pd.DataFrame(
        {
            "peptide": [PEP_A, PEP_B],
            "label": [1, 1],
            "virus": ["IAV", "CMV"],
            "protein": ["M1", "pp65"],
            "strain": [None, None],
            "hla_allele": ["HLA-A*02:01"] * 2,
            "source_type": ["Virus", "Virus"],
            "database_source": ["IEDB", "IEDB"],
        }
    ).to_csv(path, index=False)


def _write_iedb_negatives(path: Path) -> None:
    """A correct component CSV: bare integer PMIDs, one blank."""
    path.write_text(
        "peptide,label,virus,hla_allele,source_type,database_source,negative_origin,reference_pmid\n"
        f"{PEP_C},0,HCV,HLA-A*02:01,Virus,IEDB,tested_negative,38923358\n"
        "KLGGALQAK,0,CMV,HLA-A*02:01,Virus,IEDB,tested_negative,21918184\n"
        "AAAAAAAAA,0,Self,HLA-A*02:01,Self,UniProt,tested_negative,\n",
        encoding="utf-8",
    )


def test_build_dataset_v5_round_trip_does_not_refloat_pmid(tmp_path: Path) -> None:
    """The component CSV is correct; the build must not re-float it on write.

    build_dataset_v5 reads every component with dtype inference, so a blank in
    the column yields float64 and a plain write would emit 38923358.0 as text.
    """
    base = tmp_path / "v4.csv"
    iedb = tmp_path / "iedb_neg.csv"
    out = tmp_path / "v5.csv"
    _write_base_v4(base)
    _write_iedb_negatives(iedb)

    rc = build_v5_main(
        [
            "--base-dataset",
            str(base),
            "--iedb-negatives",
            str(iedb),
            "--output",
            str(out),
            "--schema",
            str(SCHEMA_PATH),
            "--conflict-audit-path",
            str(tmp_path / "conflicts.csv"),
        ]
    )
    assert rc == 0

    values = [v for v in _raw_column(out, "reference_pmid") if v != ""]
    assert sorted(values) == ["21918184", "38923358"]
    assert all(v.isdigit() for v in values), values


# ---------------------------------------------------------------------------
# scripts/export_loo_test_sets.py re-emission
# ---------------------------------------------------------------------------


def test_export_loo_test_sets_does_not_refloat_pmid(tmp_path: Path) -> None:
    dataset = tmp_path / "v5.csv"
    dataset.write_text(
        "peptide,hla_allele,label,virus,is_quarantined,negative_origin,reference_pmid\n"
        f"{PEP_A},HLA-A*02:01,1,CMV,False,,21918184\n"
        f"{PEP_B},HLA-A*02:01,0,CMV,False,tested_negative,38923358\n"
        f"{PEP_C},HLA-A*02:01,0,CMV,False,iedb_api,\n",
        encoding="utf-8",
    )

    partition = build_test_partition(load_dataset(str(dataset)), "CMV")
    out = tmp_path / "CMV_held_out.tsv"
    partition.to_csv(out, sep="\t", index=False)

    values = [v for v in _raw_column(out, "reference_pmid", delimiter="\t") if v != ""]
    assert sorted(values) == ["21918184", "38923358"]
    assert all(v.isdigit() for v in values), values
