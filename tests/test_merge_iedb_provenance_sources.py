"""The merged-IEDB generator must emit the sidecar its artifact already carries.

PR #553 hand-corrected `data/iedb_negatives_v5_merged_provenance.json` to drop an
unresolvable `data\\iedb` entry and its Windows separators. It changed the ARTIFACT
only. The generator that writes that artifact was left emitting the old shape, so
the next `merge()` run would have silently restored both halves and nothing would
have caught it: the only tracked reference to that filename is `.gitattributes`.

These tests pin the generator to the corrected artifact, so the two cannot drift
apart again without a red test.

Pinning the SHAPE is not enough. The anchor test feeds the artifact's own last
`sources` entry back into the generator, so it checks how the list is built and
never WHICH file it names: rewriting `sources` to `data/iedb_negatives_v5.csv`,
the value retracted as instance #11 in the third-party-claims incident ledger,
left every test here green, and `tools/check_dataset_provenance.py` does not
read this sidecar at all. The identity tests below bind the named file to the
counts the record carries, and the lineage tests require the record of an
idempotent re-merge to retain the earlier run that actually produced its rows.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "scripts" / "merge_iedb_api_negatives.py"
ARTIFACT = REPO_ROOT / "data" / "iedb_negatives_v5_merged_provenance.json"

# The `sources` value retracted as ledger instance #11. It RESOLVES in every
# clone, which is exactly why a resolvability check cannot catch it.
LEDGER_11_SOURCES = ["data/iedb_negatives_v5.csv"]


def _load_generator():
    """Import the generator without executing its CLI.

    It is a `scripts/` module rather than a package, and it imports `_ssl_fix`
    and `_dataset_utils` from its own directory, so the directory has to be on
    the path before the spec is executed.
    """
    import sys

    scripts_dir = str(GENERATOR.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("merge_iedb_api_negatives", GENERATOR)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        pytest.skip("cannot load the generator module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sources_use_posix_separators_not_the_host_platform_s():
    """A tracked sidecar read from a clone must not carry Windows separators."""
    mod = _load_generator()
    sources = mod.provenance_sources(
        Path("data") / "iedb",
        Path("data") / "iedb_negatives_v5_merged.csv",
        net_new_api_rows=7,
    )
    assert sources == ["data/iedb", "data/iedb_negatives_v5_merged.csv"]
    for entry in sources:
        assert "\\" not in entry, f"non-portable separator in sources entry: {entry!r}"


def test_an_api_dir_that_contributed_nothing_is_not_named_as_a_source():
    """The idempotent re-merge case, which is what the tracked artifact records.

    Naming an untracked directory that supplied zero rows declares an
    unresolvable path as the source of rows it never provided.
    """
    mod = _load_generator()
    sources = mod.provenance_sources(
        Path("data") / "iedb",
        Path("data") / "iedb_negatives_v5_merged.csv",
        net_new_api_rows=0,
    )
    assert sources == ["data/iedb_negatives_v5_merged.csv"]


def test_an_api_dir_that_did_contribute_is_still_named():
    """Non-vacuity partner for the test above.

    Without this, the policy could be implemented as "never record api_dir" and
    the zero-row test would still pass while a real input went undeclared.
    """
    mod = _load_generator()
    sources = mod.provenance_sources(
        Path("data") / "iedb",
        Path("data") / "iedb_negatives_v5_merged.csv",
        net_new_api_rows=1,
    )
    assert "data/iedb" in sources


def test_the_generator_reproduces_the_tracked_artifact_s_sources():
    """The anchor: generator output must equal what the tracked sidecar holds.

    The sidecar records `net_new_api_rows: 0`, so the generator must produce
    exactly its `sources` for that case. This is the assertion that would have
    gone red had the artifact been corrected while the generator was not.

    It checks SHAPE only: the existing-input path is the artifact's own last
    entry, fed back in. Which file that entry names is bound by
    test_the_named_existing_input_has_the_row_count_the_record_carries.
    """
    mod = _load_generator()
    recorded = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    net_new = recorded["net_new_api_rows"]

    regenerated = mod.provenance_sources(
        Path("data") / "iedb",
        Path(recorded["sources"][-1]),
        net_new_api_rows=net_new,
    )
    assert regenerated == recorded["sources"], (
        "the generator no longer emits the sources its own tracked artifact "
        f"carries: generator {regenerated!r} vs artifact {recorded['sources']!r}"
    )


# ---------------------------------------------------------------------------
# Identity: the file `sources` names must be the file the counts describe
# ---------------------------------------------------------------------------


def _recorded() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _output_rel(record: dict) -> str:
    """Repository-relative path of the artifact this record describes."""
    return (ARTIFACT.parent / record["output"]).relative_to(REPO_ROOT).as_posix()


def _count_csv_rows(text_stream) -> int:
    """Data records in a CSV stream, header excluded, quoted newlines honoured."""
    return max(sum(1 for _ in csv.reader(text_stream)) - 1, 0)


def _data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return _count_csv_rows(handle)


def _identity_problems(record: dict) -> list[str]:
    """Every way the record's named files disagree with the counts it carries.

    `provenance_sources()` always places the existing input LAST, so that entry
    is the file `existing_rows` was counted from. Row-count equality is a
    NECESSARY condition of identity, not a sufficient one; it is the one a
    clone can check without the untracked API directory, and it separates the
    two IEDB negatives files this record could plausibly name.
    """
    problems: list[str] = []
    existing_rows = int(record["existing_rows"])
    net_new = int(record["net_new_api_rows"])
    row_count = int(record["row_count"])

    if existing_rows + net_new != row_count:
        problems.append(
            f"existing_rows {existing_rows} + net_new_api_rows {net_new} != row_count {row_count}"
        )

    output_rel = _output_rel(record)
    output = REPO_ROOT / output_rel
    if not output.is_file():
        problems.append(f"output {output_rel} does not resolve in this checkout")
    else:
        output_rows = _data_rows(output)
        if output_rows != row_count:
            problems.append(
                f"output {output_rel} has {output_rows} data rows, "
                f"record says row_count {row_count}"
            )

    existing = str(record["sources"][-1])
    existing_path = REPO_ROOT / existing
    if not existing_path.is_file():
        problems.append(f"existing input {existing} does not resolve in this checkout")
    else:
        rows = _data_rows(existing_path)
        if rows != existing_rows:
            problems.append(
                f"existing input {existing} has {rows} data rows, "
                f"record says existing_rows {existing_rows}"
            )
    return problems


def test_the_named_existing_input_has_the_row_count_the_record_carries():
    """Bind WHICH file `sources` names, not just the shape of the list."""
    problems = _identity_problems(_recorded())
    assert problems == [], "the sidecar's sources disagree with its own counts:\n" + "\n".join(
        problems
    )


def test_the_ledger_11_substitution_is_rejected_although_it_resolves():
    """Non-vacuity partner: the retracted value must turn the identity check red.

    The precondition matters. The ledger #11 path IS tracked and DOES resolve,
    so the failure asserted here is an identity failure, not an absent file.
    A check that only asked "does this path resolve" would pass it.
    """
    assert (REPO_ROOT / LEDGER_11_SOURCES[0]).is_file(), (
        "precondition: the ledger #11 path must resolve, or this test proves nothing"
    )
    doctored = {**_recorded(), "sources": list(LEDGER_11_SOURCES)}
    problems = _identity_problems(doctored)
    assert any(
        p.startswith(f"existing input {LEDGER_11_SOURCES[0]} has ") and "existing_rows" in p
        for p in problems
    ), f"the ledger #11 sources value was not rejected on identity: {problems!r}"


# ---------------------------------------------------------------------------
# Lineage: an idempotent re-merge must not erase the run that made the rows
# ---------------------------------------------------------------------------


def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _retained_run(record: dict) -> dict:
    """The `upstream_generator_run` block an idempotent re-merge must carry.

    A re-merge over its own output records `net_new_api_rows: 0` and names the
    output as its only source, so on its own it says nothing about where any
    row came from. `merge()` rewrites the whole sidecar, so a future re-merge
    would drop this block; this is the check that would then go red.
    """
    is_self_remerge = int(record["net_new_api_rows"]) == 0 and record["sources"] == [
        _output_rel(record)
    ]
    if not is_self_remerge:
        pytest.skip("the tracked record is not an idempotent re-merge over its own output")
    run = record.get("upstream_generator_run")
    assert isinstance(run, dict), (
        "the record is an idempotent re-merge over its own output but retains no "
        "upstream_generator_run, so the lineage of every row it describes is lost"
    )
    return run


def test_an_idempotent_re_merge_retains_the_run_that_produced_its_rows():
    record = _recorded()
    run = _retained_run(record)
    assert int(run["net_new_api_rows"]) > 0, "the retained run must be one that added rows"
    assert int(run["existing_rows"]) + int(run["net_new_api_rows"]) == int(run["row_count"])
    assert int(run["row_count"]) == int(record["row_count"])
    assert run["per_virus_counts"] == record["per_virus_counts"]
    assert run["sources"][-1] != _output_rel(record), (
        "the retained run must name a real upstream input, not this artifact"
    )
    for entry in run["sources"]:
        assert "\\" not in entry, f"non-portable separator in retained sources: {entry!r}"


def test_the_retained_run_re_derives_from_tracked_objects():
    """The retained counts are claims; check them against the Git objects they cite.

    Needs the two cited commits, so a shallow clone skips rather than passes.
    """
    record = _recorded()
    run = _retained_run(record)
    for revision in (run["existing_input_revision"], run["recorded_in_revision"]):
        if _git("cat-file", "-e", f"{revision}^{{commit}}").returncode != 0:
            pytest.skip(f"{revision} is not in this clone (shallow checkout)")

    # 1. The named existing input, as tracked at the cited revision, has the
    #    row count the run recorded.
    shown = _git("show", f"{run['existing_input_revision']}:{run['sources'][-1]}")
    assert shown.returncode == 0, shown.stderr.decode("utf-8", "replace")
    as_text = io.StringIO(shown.stdout.decode("utf-8"), newline="")
    assert _count_csv_rows(as_text) == int(run["existing_rows"])

    # 2. The artifact that run committed is the same blob tracked here now.
    then = _git("rev-parse", f"{run['recorded_in_revision']}:{_output_rel(record)}")
    now = _git("hash-object", "--", _output_rel(record))
    assert then.returncode == 0 and now.returncode == 0
    assert then.stdout.strip() == now.stdout.strip(), (
        "the artifact changed since the retained run produced it, so that run no "
        "longer describes these rows"
    )

    # 3. The block is the sidecar that run committed, with only the documented
    #    edits: separators normalised and the unresolvable git_sha omitted.
    sidecar_rel = ARTIFACT.relative_to(REPO_ROOT).as_posix()
    original_blob = _git("show", f"{run['recorded_in_revision']}:{sidecar_rel}")
    assert original_blob.returncode == 0, original_blob.stderr.decode("utf-8", "replace")
    original = json.loads(original_blob.stdout.decode("utf-8"))
    assert [s.replace("\\", "/") for s in original["sources"]] == run["sources"]
    for key in ("row_count", "generated_utc", "existing_rows", "net_new_api_rows"):
        assert original[key] == run[key], f"retained {key} differs from the committed record"
    assert original["per_virus_counts"] == run["per_virus_counts"]
