"""Every claims-register section keys its rows, and the keys stay citable.

The register is cited by row id throughout this project's rules and reviews. Sections 1
and 5 carried an ID column; Sections 2, 3 and 4 did not, so a row there could only be
cited by quoting its claim text, which is long and gets reworded. They now carry one.

This guards properties, not contents. Measured while writing it, and the reason the
obvious assertion is not the one made: Section 1 holds 41 rows numbered D1 to D42 with
D41 absent, and its rows are not in id order. So "1..n with no gaps" is true of the three
newly keyed sections and false of the oldest one. Uniqueness is the invariant that holds
everywhere; Section 1's gap is instead bound to the heading that declares it, which turns
that heading into an enforced statement.
"""

from __future__ import annotations

import re
from pathlib import Path

REGISTER = Path(__file__).resolve().parents[1] / "docs" / "claims_register.md"
ID = re.compile(r"^([A-Z]+)(\d+)$")
NEWLY_KEYED = {"HD", "BA", "MON"}


def _section_tables() -> dict[str, list[list[str]]]:
    """Section heading -> the rows of its markdown table, each as a list of cells."""
    tables: dict[str, list[list[str]]] = {}
    section = None
    for line in REGISTER.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line
            continue
        if section is None or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-"):
            continue
        tables.setdefault(section, []).append(cells)
    return tables


def _ids(rows: list[list[str]]) -> list[re.Match[str]]:
    parsed = []
    for row in rows[1:]:
        match = ID.match(row[0])
        assert match is not None, f"id is not a prefix plus a number: {row[0]!r}"
        parsed.append(match)
    return parsed


def test_every_section_table_keys_its_rows() -> None:
    tables = _section_tables()
    assert len(tables) == 5, f"expected five sections, found {sorted(tables)}"
    for section, rows in tables.items():
        assert rows[0][0] == "ID", f"{section} does not key its rows"
        assert len(rows) > 1, f"{section} has a header and no rows"


def test_ids_are_unique_and_single_prefixed_within_each_section() -> None:
    for section, rows in _section_tables().items():
        parsed = _ids(rows)
        values = [match.group(0) for match in parsed]
        assert len(set(values)) == len(values), f"{section} repeats an id"
        prefixes = {match.group(1) for match in parsed}
        assert len(prefixes) == 1, f"{section} mixes id prefixes: {sorted(prefixes)}"


def test_the_section_prefixes_are_distinct() -> None:
    prefixes = [_ids(rows)[0].group(1) for rows in _section_tables().values()]
    assert len(set(prefixes)) == len(prefixes), f"two sections share an id prefix: {prefixes}"
    # D and ES predate this test; HD, BA and MON were chosen to collide with neither them nor
    # any identifier already present in the tracked tree.
    assert set(prefixes) == {"D", "HD", "BA", "MON", "ES"}


def test_the_newly_keyed_sections_number_from_one_without_gaps() -> None:
    seen = set()
    for section, rows in _section_tables().items():
        parsed = _ids(rows)
        prefix = parsed[0].group(1)
        if prefix not in NEWLY_KEYED:
            continue
        seen.add(prefix)
        numbers = [int(match.group(2)) for match in parsed]
        assert numbers == list(range(1, len(numbers) + 1)), f"{section} is not 1..n: {numbers}"
    assert seen == NEWLY_KEYED, f"a newly keyed section vanished: {sorted(seen)}"


def test_section_one_matches_the_range_its_heading_declares() -> None:
    heading, rows = next(
        (head, body) for head, body in _section_tables().items() if head.startswith("## Section 1")
    )
    declared = re.search(r"\(([^)]*)\)", heading)
    assert declared is not None, f"Section 1's heading no longer declares its range: {heading}"
    expected: set[int] = set()
    for part in declared.group(1).split(","):
        part = part.strip().lstrip("D")
        if "-" in part:
            low, high = (int(x.lstrip("D")) for x in part.split("-"))
            expected |= set(range(low, high + 1))
        else:
            expected.add(int(part))
    measured = {int(match.group(2)) for match in _ids(rows)}
    assert measured == expected, "Section 1's rows and its declared range disagree"
    assert 41 not in measured, "D41 is absent by design; the heading says so"
