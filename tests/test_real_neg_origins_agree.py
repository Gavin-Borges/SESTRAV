"""Pin the REAL_NEG_ORIGINS invariant equal across every tracked carrier.

REAL_NEG_ORIGINS names the negative_origin values that are GENUINE assay-confirmed
negatives, as opposed to synthetic decoys (allele_matched_nonbinder,
self_proteome_decoy, published_negative). It is load-bearing for honest-metric
reporting: a carrier that drifts silently reclassifies decoys as real negatives in
whichever analysis reads it, and the resulting AUC-PR is then not the number it
claims to be. The constant is duplicated across several scripts and, until this
file, nothing held the copies in agreement.

WHY AST AND NOT IMPORT
    The carriers are analysis entry-point scripts. Importing them executes
    module-level code and pulls pandas, numpy and sklearn, and several have side
    effects. An ast.literal_eval of the assignment's value node reads the constant
    without running anything, so this test passes in a clone with no model, no data
    and no network.

WHY THE NAME MATCH IS EXACT AND CASE-INSENSITIVE
    Carriers spell the name three ways: REAL_NEG_ORIGINS, the underscore-prefixed
    _REAL_NEG_ORIGINS, and a function-local lowercase _real_neg_origins in
    scripts/build_dataset_v5.py. A case-SENSITIVE search for REAL_NEG_ORIGINS finds
    only six of the seven and reports the lowercase one as absent, so discovery here
    normalizes case and strips leading underscores.

    The match is on the WHOLE normalized identifier, never a substring. Two scripts,
    scripts/audit_cv_leakage.py and scripts/compute_pooled_honest_metric.py, define a
    SINGULAR REAL_NEG_ORIGIN holding the single string "iedb_api". That is a
    different, narrower constant, not a drifted copy of this one; a substring match
    would sweep it in and fail on a legitimate difference.

WHY CONTAINER TYPE IS IGNORED
    The carriers use a set, a tuple and an annotated frozenset. The container is a
    style difference. The VALUES are the invariant, so every comparison here is
    between frozensets.

WHAT THIS TEST CANNOT DO
    No test can discover a carrier introduced under an arbitrary unrelated name. The
    two mitigations are both partial and are named as such: the count FLOOR below
    catches removal of coverage, and KNOWN_ALIAS_CARRIERS pins the one divergent-name
    copy that is known to exist. A new copy under a new unrelated name is invisible
    here and is caught only by review.

No path:NNN line citations appear in this file. scripts/check_doc_line_citations.py
scans .py sources and its exempt ledger is at its ceiling, so files are named by path
and symbols by name only.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# Roots walked with pathlib rather than a git pathspec. Default git pathspec is not
# glob-magic and its "*" crosses "/", so `git ls-files 'scripts/**/*.py'` returns zero
# for a flat directory - a false negative. A pathlib walk has no such failure mode.
SCAN_ROOTS = ("scripts", "src")

# The normalized identifier a carrier must bind. Compared against
# name.lstrip("_").upper(), so REAL_NEG_ORIGINS, _REAL_NEG_ORIGINS and
# _real_neg_origins all match, while the singular REAL_NEG_ORIGIN does not.
CARRIER_NAME = "REAL_NEG_ORIGINS"

# Call wrappers whose single literal argument is unwrapped before literal_eval,
# because ast.literal_eval does not evaluate a frozenset(...) call.
_LITERAL_WRAPPERS = frozenset({"frozenset", "set", "tuple", "list"})

# The definition itself. Equality across carriers cannot detect a change applied to
# ALL carriers at once, which is exactly what a redefinition of "real negative" would
# look like. This pin is the second instrument for that case. If the definition ever
# changes legitimately, update this tuple in the same commit as the carriers.
EXPECTED_VALUES = frozenset({"tested_negative", "iedb_api"})

# Measured floor, not an exact count. An exact count turns every new legitimate
# carrier into a failure, which is a ratchet nobody asked for; a floor still catches
# the case this test exists to catch, which is coverage of the invariant being
# deleted or the discovery quietly ceasing to find the carriers it used to find.
# Measured at repository HEAD by walking SCAN_ROOTS with the discovery below.
EXPECTED_CARRIER_FLOOR = 7

# Carriers holding the same invariant under a DIVERGENT name, which discovery cannot
# find. Each entry carries the note that justifies it, because an allowlist entry
# with no justifying note is itself a defect this repository has shipped before.
#
#   src/ml_utils.py :: _ORIGIN_REAL
#       Builds the stratification composite key. Its own comment states it is "the
#       same pairing" that scripts/build_dataset_v5.py and
#       scripts/analyze_hiv1_binding_bias.py use, so it is a copy of this invariant
#       that merely does not share its name.
KNOWN_ALIAS_CARRIERS = (("src/ml_utils.py", "_ORIGIN_REAL"),)

# Singular-named constants that MUST NOT be swept in. Each holds one string, not the
# pair, and means something narrower; pinning them equal would be a false failure.
#
#   scripts/audit_cv_leakage.py :: REAL_NEG_ORIGIN
#   scripts/compute_pooled_honest_metric.py :: REAL_NEG_ORIGIN
#       Both bind the single value "iedb_api" for a Def-A style pooled metric that
#       deliberately excludes the bulk-export "tested_negative" origin.
NEAR_MISS_NAMES = (
    ("scripts/audit_cv_leakage.py", "REAL_NEG_ORIGIN"),
    ("scripts/compute_pooled_honest_metric.py", "REAL_NEG_ORIGIN"),
)


class Carrier(NamedTuple):
    """One assignment binding the invariant, located without importing its module."""

    path: str
    name: str
    lineno: int
    values: frozenset[str] | None

    def where(self) -> str:
        """Human-readable location. Deliberately not the path:NNN citation form."""
        return f"{self.path} (line {self.lineno}, name {self.name})"


def _normalize(name: str) -> str:
    return name.lstrip("_").upper()


def _literal_values(node: ast.AST) -> frozenset[str] | None:
    """Return the string values of a set/tuple/list literal, unwrapping one call.

    Returns None when the node is not a literal container of strings, which keeps a
    computed or imported value from being silently read as an empty set.
    """
    if isinstance(node, ast.Call):
        func = node.func
        if not isinstance(func, ast.Name) or func.id not in _LITERAL_WRAPPERS:
            return None
        if not node.args:
            return frozenset()
        if len(node.args) != 1:
            return None
        node = node.args[0]
    if not isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        return None
    try:
        raw = ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None
    if not all(isinstance(item, str) for item in raw):
        return None
    return frozenset(raw)


def _assignments_in(path: Path, root: Path) -> list[Carrier]:
    """Every simple-name assignment in one source file, as Carrier records.

    ast.walk is used rather than a scan of the module body because one carrier is a
    function-local assignment, which a module-body-only scan would miss.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(root).as_posix()
    out: list[Carrier] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                out.append(
                    Carrier(rel, target.id, node.lineno, _literal_values(node.value))
                )
    return out


def _iter_sources(root: Path) -> list[Path]:
    paths: list[Path] = []
    for scan_root in SCAN_ROOTS:
        base = root / scan_root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            paths.append(path)
    return paths


def discover_carriers(root: Path = REPO_ROOT) -> list[Carrier]:
    """All assignments under SCAN_ROOTS whose normalized name is CARRIER_NAME."""
    return [
        carrier
        for path in _iter_sources(root)
        for carrier in _assignments_in(path, root)
        if _normalize(carrier.name) == CARRIER_NAME
    ]


def _lookup(root: Path, rel_path: str, symbol: str) -> Carrier:
    path = root / rel_path
    assert path.is_file(), f"expected carrier file is missing: {rel_path}"
    matches = [c for c in _assignments_in(path, root) if c.name == symbol]
    assert matches, f"{rel_path} no longer binds {symbol}"
    assert len(matches) == 1, f"{rel_path} binds {symbol} more than once"
    return matches[0]


def test_carriers_are_discoverable() -> None:
    """Discovery finds carriers at all. Guards every later test against vacuity."""
    carriers = discover_carriers()
    assert carriers, (
        "no REAL_NEG_ORIGINS carrier was found under "
        f"{list(SCAN_ROOTS)}; discovery is broken or the constant was renamed"
    )


def test_every_carrier_holds_a_readable_literal() -> None:
    """A carrier whose value cannot be literal-eval'd is unpinnable, not passing."""
    unreadable = [c.where() for c in discover_carriers() if c.values is None]
    assert not unreadable, (
        "these carriers no longer bind a literal container of strings, so this test "
        "cannot pin them: " + "; ".join(unreadable)
    )


def test_all_carriers_agree() -> None:
    """The invariant: every carrier holds the same VALUES, whatever its container."""
    carriers = discover_carriers()
    distinct = {c.values for c in carriers}
    assert len(distinct) == 1, (
        "REAL_NEG_ORIGINS carriers disagree. Per carrier: "
        + "; ".join(f"{c.where()} -> {sorted(c.values or [])}" for c in carriers)
    )


def test_carrier_values_match_the_pinned_definition() -> None:
    """Catches a redefinition applied to every carrier at once, which equality cannot."""
    for carrier in discover_carriers():
        assert carrier.values == EXPECTED_VALUES, (
            f"{carrier.where()} holds {sorted(carrier.values or [])} but the pinned "
            f"definition is {sorted(EXPECTED_VALUES)}. If the definition of a real "
            "negative genuinely changed, update EXPECTED_VALUES in the same commit."
        )


def test_carrier_count_holds_its_floor() -> None:
    """A floor, not an equality. See EXPECTED_CARRIER_FLOOR for why."""
    carriers = discover_carriers()
    assert len(carriers) >= EXPECTED_CARRIER_FLOOR, (
        f"expected at least {EXPECTED_CARRIER_FLOOR} carriers, found "
        f"{len(carriers)}: " + "; ".join(c.where() for c in carriers)
    )


def test_divergent_name_aliases_agree() -> None:
    """Pin the known same-invariant copies that do not share the carrier name."""
    for rel_path, symbol in KNOWN_ALIAS_CARRIERS:
        alias = _lookup(REPO_ROOT, rel_path, symbol)
        assert alias.values == EXPECTED_VALUES, (
            f"{alias.where()} holds {sorted(alias.values or [])}, which disagrees "
            f"with the pinned definition {sorted(EXPECTED_VALUES)}"
        )


def test_singular_named_constants_are_not_swept_in() -> None:
    """The near-miss guard: a widened match would fail on a legitimate difference."""
    discovered = {(c.path, c.name) for c in discover_carriers()}
    for rel_path, symbol in NEAR_MISS_NAMES:
        assert (rel_path, symbol) not in discovered, (
            f"{rel_path} :: {symbol} is a SINGULAR constant holding one origin, not "
            "the pair; discovery must not treat it as a REAL_NEG_ORIGINS carrier"
        )
        near_miss = _lookup(REPO_ROOT, rel_path, symbol)
        assert near_miss.values is None, (
            f"{near_miss.where()} is expected to bind a single string, not a "
            "container; if it now binds the pair, re-adjudicate whether it is a "
            "carrier rather than leaving it excluded"
        )


def test_comparison_logic_catches_a_mutated_carrier(tmp_path: Path) -> None:
    """Anti-vacuity: prove the comparison can FAIL, not only that it passes.

    Writes synthetic modules outside the repository reproducing all three real
    container shapes, then re-runs the very same discovery and comparison. The clean
    fixture must agree; the mutated one must be detected. A check that cannot fail is
    not evidence, so both directions are asserted here.
    """
    clean_root = tmp_path / "clean"
    (clean_root / "scripts").mkdir(parents=True)
    (clean_root / "scripts" / "as_set.py").write_text(
        'REAL_NEG_ORIGINS = {"tested_negative", "iedb_api"}\n', encoding="utf-8"
    )
    (clean_root / "scripts" / "as_tuple.py").write_text(
        'REAL_NEG_ORIGINS = ("tested_negative", "iedb_api")\n', encoding="utf-8"
    )
    (clean_root / "scripts" / "as_frozenset.py").write_text(
        "REAL_NEG_ORIGINS: frozenset[str] = frozenset("
        '{"tested_negative", "iedb_api"})\n',
        encoding="utf-8",
    )
    (clean_root / "scripts" / "as_local_lowercase.py").write_text(
        "def build():\n"
        '    _real_neg_origins = {"tested_negative", "iedb_api"}\n'
        "    return _real_neg_origins\n",
        encoding="utf-8",
    )

    clean = discover_carriers(clean_root)
    assert len(clean) == 4, f"fixture discovery found {len(clean)} of 4 shapes"
    assert len({c.values for c in clean}) == 1, "clean fixture should agree"
    assert clean[0].values == EXPECTED_VALUES

    # Seed one defect: a decoy origin promoted into the real-negative set.
    mutated_root = tmp_path / "mutated"
    (mutated_root / "scripts").mkdir(parents=True)
    for src in sorted((clean_root / "scripts").glob("*.py")):
        (mutated_root / "scripts" / src.name).write_text(
            src.read_text(encoding="utf-8"), encoding="utf-8"
        )
    (mutated_root / "scripts" / "as_tuple.py").write_text(
        'REAL_NEG_ORIGINS = ("tested_negative", "allele_matched_nonbinder")\n',
        encoding="utf-8",
    )

    mutated = discover_carriers(mutated_root)
    assert len(mutated) == 4, "mutation must not change how many carriers are found"
    assert len({c.values for c in mutated}) == 2, (
        "the seeded mutation was NOT detected; this test would be vacuous. Values: "
        + "; ".join(f"{c.where()} -> {sorted(c.values or [])}" for c in mutated)
    )
    offenders = [c for c in mutated if c.values != EXPECTED_VALUES]
    assert len(offenders) == 1 and offenders[0].path == "scripts/as_tuple.py"
