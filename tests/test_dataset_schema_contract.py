"""Bind the v5 dataset schema to the code that writes it, and to the artifact it describes.

Why this file exists: `scripts/build_dataset_v5.py` ships a function named
`validate_output_schema` that checks only that the schema's `required` COLUMN NAMES
are present, then logs "Schema validation passed". No value, type or enum
constraint was ever evaluated on the v5 path, so the schema and the artifact drifted
apart without any gate noticing. Measured at d064dbc before the accompanying schema
correction: 26,329 of 51,185 rows (51.4%) violated the shipped schema, across
`virus_family` (21,551 rows), `negative_origin` (4,778) and a non-nullable `protein`
(123). The v4 path never had this gap because `build_dataset_v4.py` calls the
full `_dataset_utils.validate_against_schema`.

The tests below close it from both directions: the schema must admit everything the
generator can emit, and the tracked artifact must satisfy the schema.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import jsonschema
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "data" / "immunogenicity_dataset_v5_schema.json"
DATASET_PATH = REPO_ROOT / "data" / "immunogenicity_dataset_v5.csv"
BUILDER_PATH = REPO_ROOT / "scripts" / "build_dataset_v5.py"


def _load_schema() -> dict:
    assert SCHEMA_PATH.is_file(), f"tracked schema missing: {SCHEMA_PATH}"
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _properties() -> dict:
    schema = _load_schema()
    return schema.get("items", schema).get("properties", {})


def _enum(column: str) -> set[str]:
    spec = _properties().get(column, {})
    assert "enum" in spec, f"{column} carries no enum in the schema"
    return {value for value in spec["enum"] if value is not None}


def _builder_constant(name: str):
    """Read a module-level constant out of the builder without importing it.

    The builder pulls in the full ingest stack at import time; parsing keeps this
    test independent of that and of any optional dependency.
    """
    tree = ast.parse(BUILDER_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
            if isinstance(target, ast.Name) and target.id == name and node.value is not None:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found as a literal assignment in {BUILDER_PATH.name}")


def _string_columns() -> set[str]:
    """Columns the schema declares as text (possibly nullable), and nothing else."""
    out = set()
    for column, spec in _properties().items():
        declared = spec.get("type")
        types = {declared} if isinstance(declared, str) else set(declared or ())
        if "string" in types and not types & {"integer", "number", "boolean"}:
            out.add(column)
    return out


def _dataset() -> pd.DataFrame:
    """Read the corpus with each declared-string column typed as text.

    A CSV carries no types, so validating one against a JSON Schema requires a
    decision about how each column is typed on read, and dtype inference is the
    wrong instrument for a column the schema calls a string. `reference_pmid` is
    the case that proves it: pandas infers object for it today ONLY because 2,238
    LANL rows carry free-text references such as "Plana2004 PMID:15213562". Drop
    those 2,238 and the remaining 18,671 all-numeric values infer as float64, so
    the column would fail `type: ["string", "null"]` and this file would go red
    without a single defect having been introduced. Measured 2026-09-17: a CSV
    whose PMID column is already CORRECT (bare integers plus a blank) fails that
    same check under inference, and passes when read as text.

    Typing the read from the schema removes that coupling. It does not weaken the
    float-suffix coverage, which is a data-format question rather than a schema
    one and is held by tests/test_reference_pmid_format.py.
    """
    assert DATASET_PATH.is_file(), f"tracked dataset missing: {DATASET_PATH}"
    return _read_csv_typed(DATASET_PATH)


def _read_csv_typed(path: Path) -> pd.DataFrame:
    """Read a CSV with every declared-string column typed as text."""
    dtypes = {column: "string" for column in _string_columns()}
    return pd.read_csv(path, dtype=dtypes, low_memory=False)


# ---------------------------------------------------------------------------
# Direction 1: the schema must admit everything the generator can emit.
# ---------------------------------------------------------------------------


def test_virus_family_enum_covers_every_family_the_builder_can_emit() -> None:
    """VIRUS_FAMILY_MAP is the generator; the enum must be a superset of its range.

    Derived from the map rather than from observed values on purpose: an enum
    rebuilt from whatever a given build happened to contain would re-bless the
    corpus instead of constraining it, and would go stale again the first time a
    mapped virus appeared.
    """
    emitted = set(_builder_constant("VIRUS_FAMILY_MAP").values())
    declared = _enum("virus_family")
    missing = sorted(emitted - declared)
    assert not missing, (
        f"VIRUS_FAMILY_MAP can emit {missing}, which the schema's virus_family enum "
        f"does not list. Add them to the enum or remove them from the map."
    )


def test_negative_origin_enum_covers_the_real_negative_definition() -> None:
    """`_real_neg_origins` decides which rows count as assay-confirmed negatives.

    It is consumed well beyond the builder (scripts/evaluate_per_virus.py mirrors it
    as REAL_NEG_ORIGINS), so a value that is load-bearing for the science but absent
    from the schema is the drift this file exists to catch. `iedb_api` was exactly
    that: 2,540 rows in the tracked artifact and no enum entry.
    """
    declared = _enum("negative_origin")
    real_neg = {"tested_negative", "iedb_api"}
    missing = sorted(real_neg - declared)
    assert not missing, (
        f"negative_origin enum omits {missing}, which build_dataset_v5.py treats as "
        f"real assay-confirmed negatives."
    )


# ---------------------------------------------------------------------------
# Direction 2: the tracked artifact must satisfy the schema.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("column", ["source_type", "virus_family", "negative_origin"])
def test_tracked_dataset_uses_only_declared_enum_values(column: str) -> None:
    """Fast, offender-naming form of the full validation below.

    Kept separate because a jsonschema failure reports one row and one message,
    while this reports every undeclared value with its row count, which is what a
    reader needs to decide whether the schema or the data is wrong.
    """
    df = _dataset()
    declared = _enum(column)
    observed = df[column].dropna().astype(str)
    undeclared = observed[~observed.isin(declared)].value_counts().to_dict()
    assert not undeclared, (
        f"{column} carries values absent from the schema enum: {undeclared}. "
        f"Declared: {sorted(declared)}."
    )


def test_tracked_dataset_validates_against_its_schema() -> None:
    """The whole artifact, every row, against the shipped schema.

    This is the assertion `validate_output_schema` never made. Read exactly the way
    `_dataset_utils.validate_against_schema` reads: dtype-inferred, records, NaN
    rendered as null.
    """
    df = _dataset()
    schema = _load_schema()

    records = df.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None

    try:
        jsonschema.validate(instance=records, schema=schema)
    except jsonschema.ValidationError as exc:
        location = list(exc.absolute_path)
        pytest.fail(
            f"data/immunogenicity_dataset_v5.csv violates its own schema at {location}: "
            f"{exc.message}"
        )


def test_validation_does_not_depend_on_free_text_pmids_being_present(tmp_path: Path) -> None:
    """The corpus must still validate once its free-text PMIDs are gone.

    Guards the coupling described in `_dataset`. The subset is written out and
    RE-READ rather than merely filtered: a dtype is fixed by the whole column at
    read time, so dropping rows from an already-read frame leaves object dtype in
    place and exercises nothing. An earlier version of this test did exactly that
    and passed with dtype inference restored, which is to say it tested nothing.
    """
    df = _dataset()
    pmid = df["reference_pmid"].astype("string")
    # Every non-null value that is not a bare integer (optionally float-suffixed).
    # Filtering on "PMID:" alone is NOT enough: 82 of the 2,238 free-text values are
    # bare author-year tokens such as "Llano2019", and leaving them in keeps the
    # column a string column, which makes this test pass without exercising anything.
    numeric_like = pmid.str.match(r"^\d+(\.0+)?$").fillna(False)
    free_text = pmid.notna() & ~numeric_like
    assert int(free_text.sum()) > 0, "expected free-text PMID rows to exist in the corpus"

    subset_path = tmp_path / "numeric_pmid_only.csv"
    df[~free_text].to_csv(subset_path, index=False)
    reread = _read_csv_typed(subset_path)

    schema = _load_schema()
    records = reread.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None
    try:
        jsonschema.validate(instance=records, schema=schema)
    except jsonschema.ValidationError as exc:
        pytest.fail(
            "the corpus stops validating once free-text PMIDs are removed, which means "
            f"the read is inferring column types rather than taking them from the "
            f"schema: {exc.message}"
        )
