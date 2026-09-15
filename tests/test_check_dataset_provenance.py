"""Tests for tools/check_dataset_provenance.py.

Every negative case is built by perturbing ONE field of an otherwise-consistent
fixture, so a failure names the check that fired rather than a tangle of them.
The final test runs against the real repository, which is what makes this suite
able to see live drift: per `.claude/rules/deletion-safety-battery.md`, a check
whose tests only use temp fixtures cannot.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "check_dataset_provenance.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("check_dataset_provenance", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TOOL_MOD = _load_tool()

# A corpus small enough to write inline. Row count and digest are derived from it
# rather than hardcoded, so the fixture cannot drift from its own sidecar.
CORPUS_ROWS = ["peptide,hla_allele,label"] + [
    f"PEPTIDE{i:03d},HLA-A*02:01,{i % 2}" for i in range(12)
]


def _write_fixture(root: pathlib.Path, *, counts=None, pins=None, corpus_lines=None,
                   checksum=None, config_checksum=None, output_file=None):
    """Build a self-consistent fixture repo, then let callers perturb one field."""
    lines = corpus_lines if corpus_lines is not None else CORPUS_ROWS
    body = "\n".join(lines) + "\n"
    (root / "data").mkdir(parents=True, exist_ok=True)
    corpus = root / "data" / "immunogenicity_dataset_v5.csv"
    corpus.write_text(body, encoding="utf-8", newline="")
    rows = len(lines) - 1
    digest = hashlib.sha256(body.replace("\r\n", "\n").encode("utf-8")).hexdigest()

    base_counts = {
        "v4_positives": 4,
        "published_panels": 5,
        "v4_hard_decoys": 3,
        "iedb_negatives": 2,
        "v4_other_negatives": 99,  # counted in provenance, never appended to parts
        "dedup_dropped": 2,
        "merged_total": rows,
    }
    if counts:
        base_counts.update(counts)

    sidecar = {
        "source_counts": base_counts,
        "output_file": output_file if output_file is not None else "data\\immunogenicity_dataset_v5.csv",
        "output_checksum_sha256": checksum if checksum is not None else digest,
    }
    (root / "data" / "immunogenicity_dataset_v5_provenance.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )

    base_pins = {k: base_counts[k] for k in
                 ("v4_positives", "published_panels", "v4_hard_decoys", "iedb_negatives",
                  "dedup_dropped", "merged_total")}
    if pins is not None:
        base_pins = pins
    pin_lines = "".join(f"      {k}: {v}\n" for k, v in base_pins.items())
    cfg = (
        "dataset_governance:\n"
        "  provenance:\n"
        f'    checksum: "{config_checksum if config_checksum is not None else digest}"\n'
        + ("    source_counts:\n" + pin_lines if base_pins else "")
    )
    (root / "config.yaml").write_text(cfg, encoding="utf-8")
    return root, rows, digest


def test_a_consistent_fixture_reports_no_problems(tmp_path):
    root, rows, _ = _write_fixture(tmp_path)
    # 4 + 5 + 3 + 2 = 14, minus dedup_dropped 2 = 12 = rows
    assert rows == 12
    assert TOOL_MOD.check(root) == []


def test_v4_other_negatives_is_deliberately_not_summed(tmp_path):
    """The build counts it into provenance but never appends it to `parts`.

    Including it would make a CORRECT sidecar fail, so this pins the exclusion.
    The fixture carries a deliberately absurd 99 for exactly that reason.
    """
    root, _rows, _ = _write_fixture(tmp_path)
    assert TOOL_MOD.check(root) == [], "v4_other_negatives must not enter the arithmetic"


def test_arithmetic_contradiction_is_caught(tmp_path):
    root, rows, _ = _write_fixture(tmp_path, counts={"dedup_dropped": 3},
                                   pins={"dedup_dropped": 3})
    problems = TOOL_MOD.check(root)
    assert any("contradicts itself" in p for p in problems), problems


def test_row_count_mismatch_is_caught(tmp_path):
    """A sidecar that is internally consistent but describes a different corpus."""
    root, rows, _ = _write_fixture(
        tmp_path, counts={"merged_total": 11, "dedup_dropped": 3}, pins={"merged_total": 11}
    )
    problems = TOOL_MOD.check(root)
    assert any("data rows but the sidecar records" in p for p in problems), problems


def test_stale_sidecar_checksum_is_caught(tmp_path):
    root, _rows, _digest = _write_fixture(tmp_path, checksum="0" * 64)
    problems = TOOL_MOD.check(root)
    assert any("LF-normalised sha256" in p for p in problems), problems


def test_config_checksum_disagreement_is_caught(tmp_path):
    root, _rows, _ = _write_fixture(tmp_path, config_checksum="f" * 64)
    problems = TOOL_MOD.check(root)
    assert any("freeze_mode would reject" in p for p in problems), problems


def test_a_changed_stream_count_is_caught(tmp_path):
    """The failure class the class-ratio gate is structurally blind to."""
    root, _rows, _ = _write_fixture(tmp_path, pins={"v4_hard_decoys": 6})
    problems = TOOL_MOD.check(root)
    assert any("v4_hard_decoys" in p and "pins 6" in p for p in problems), problems


def test_absent_pins_fail_closed(tmp_path):
    """No pinned counts means the gate can see nothing, which must not read as clean."""
    root, _rows, _ = _write_fixture(tmp_path, pins={})
    problems = TOOL_MOD.check(root)
    assert any("source_counts is absent" in p for p in problems), problems


def test_missing_sidecar_fails_closed(tmp_path):
    root, _rows, _ = _write_fixture(tmp_path)
    (root / "data" / "immunogenicity_dataset_v5_provenance.json").unlink()
    problems = TOOL_MOD.check(root)
    assert len(problems) == 1 and "missing (fail closed)" in problems[0], problems


def test_a_crlf_corpus_still_matches_the_lf_digest(tmp_path):
    """The digest must be convention-independent.

    A CRLF working-tree digest is what made config.yaml's pin wrong for two
    months; computing that convention here would reintroduce the same defect on
    Windows.
    """
    root, _rows, digest = _write_fixture(tmp_path)
    corpus = root / "data" / "immunogenicity_dataset_v5.csv"
    corpus.write_bytes(corpus.read_bytes().replace(b"\n", b"\r\n"))
    assert b"\r\n" in corpus.read_bytes()
    assert TOOL_MOD.check(root) == [], "LF normalisation should make CRLF immaterial"


def test_output_file_with_windows_separators_resolves(tmp_path):
    root, _rows, _ = _write_fixture(
        tmp_path, output_file="data\\immunogenicity_dataset_v5.csv"
    )
    assert TOOL_MOD.check(root) == []


def test_unresolvable_output_file_is_caught(tmp_path):
    root, _rows, _ = _write_fixture(tmp_path, output_file="data/does_not_exist.csv")
    problems = TOOL_MOD.check(root)
    assert any("does not resolve to a file" in p for p in problems), problems


def test_the_live_repository_is_consistent():
    """Live coverage, not a fixture. This is what can see real drift."""
    problems = TOOL_MOD.check(REPO_ROOT)
    assert problems == [], "\n".join(problems)
