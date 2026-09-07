from __future__ import annotations

import json

from scripts.check_digest_portability import (
    EXEMPT_DIGESTS,
    EXEMPT_REASON,
    apply_exemption,
    classify,
    extract_records,
    print_human,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def test_extracts_heterogeneous_digest_schemas(tmp_path):
    manifest = tmp_path / "sample.provenance.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact": "results/report.csv",
                "sha256": DIGEST_A,
                "input_file": "data/input.csv",
                "input_sha256": DIGEST_B,
                "nested": {"model_path": "models/rf.joblib", "model_sha256": DIGEST_A},
                "inputs": {"data/source.csv": DIGEST_B},
            }
        ),
        encoding="utf-8",
    )

    records = extract_records(manifest.name, json.loads(manifest.read_text(encoding="utf-8")))

    assert {(record["path"], record["recorded"]) for record in records} == {
        ("results/report.csv", DIGEST_A),
        ("data/input.csv", DIGEST_B),
        ("models/rf.joblib", DIGEST_A),
        ("data/source.csv", DIGEST_B),
    }


def test_extracts_manifest_relative_artifacts_and_unresolved(tmp_path):
    manifest = tmp_path / "model_artifact_checksums.json"
    payload = {
        "artifacts": {"v5/model.joblib": {"sha256": DIGEST_A}},
        "orphan_sha256": DIGEST_B,
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    records = extract_records(manifest.name, json.loads(manifest.read_text(encoding="utf-8")))

    assert records[0]["path"] == "v5/model.joblib"
    assert records[0]["relative"] is True
    assert records[1]["path"] is None
    assert records[1]["reason"] == "no subject path paired with orphan_sha256"


def test_classifies_all_comparison_outcomes(tmp_path):
    (tmp_path / "artifact.txt").write_bytes(b"fixture")

    assert classify(DIGEST_A, DIGEST_A, DIGEST_B) == "PORTABLE"
    assert classify(DIGEST_A, DIGEST_B, DIGEST_A) == "WINDOWS_ONLY"
    assert classify(DIGEST_A, DIGEST_B, None) == "MISMATCH"
    assert classify(DIGEST_A, None, DIGEST_A) == "MISSING"


def test_human_output_stays_below_120_columns(tmp_path, capsys):
    long_path = (tmp_path / ("segment-" * 30)).as_posix()
    row = {
        "verdict": "UNRESOLVED",
        "path": long_path,
        "manifest": "manifest.json",
        "recorded": DIGEST_A,
        "blob": None,
        "worktree": None,
        "eol": "lf",
        "reason": "test reason",
    }

    print_human([row])

    assert max(map(len, capsys.readouterr().out.splitlines())) < 120


def test_exemption_is_scoped_to_the_exact_manifest_and_source_pair():
    manifest, source = next(iter(sorted(EXEMPT_DIGESTS)))

    assert apply_exemption(manifest, source, "MISMATCH") == ("EXEMPT", EXEMPT_REASON)
    assert apply_exemption(manifest, source, "WINDOWS_ONLY") == ("EXEMPT", EXEMPT_REASON)

    # A different digest in the SAME manifest is not swallowed. This is the
    # property that makes the exemption safe: it is keyed on the pair, so the
    # live output_checksum_sha256 - which names the same subject path as the
    # exempt historical one - still fails on a genuine mismatch.
    assert apply_exemption(manifest, "output_checksum_sha256", "MISMATCH") == ("MISMATCH", None)

    # The same source key in a different manifest is not exempt.
    assert apply_exemption("data/other_provenance.json", source, "MISMATCH") == ("MISMATCH", None)

    # Prefix membership is not enough in either direction.
    assert apply_exemption(manifest, f"{source}_extra", "MISMATCH") == ("MISMATCH", None)
    assert apply_exemption(f"{manifest}.bak", source, "MISMATCH") == ("MISMATCH", None)

    # Only the two failing verdicts are downgraded.
    assert apply_exemption(manifest, source, "MISSING") == ("MISSING", None)
    assert apply_exemption(manifest, source, "UNRESOLVED") == ("UNRESOLVED", None)
    assert apply_exemption(manifest, source, "PORTABLE") == ("PORTABLE", None)


def test_exempt_pair_names_a_source_the_extractor_actually_emits():
    """An exemption keyed on a source string extract_records never emits is inert.

    The nested historical block and the live top-level pin record digests for
    the SAME output_file, so this also pins that the two are distinguishable at
    all - the reason the exemption can be keyed on the source rather than on the
    subject path.
    """
    manifest = "data/iedb_negatives_v5_provenance.json"
    payload = {
        "output_file": "data/iedb_negatives_v5.csv",
        "output_checksum_sha256": DIGEST_A,
        "upstream_generator_run": {
            "output_file": "data/iedb_negatives_v5.csv",
            "output_checksum_sha256": DIGEST_B,
        },
    }

    records = extract_records(manifest, payload)
    verdicts = {
        str(record["source"]): apply_exemption(manifest, str(record["source"]), "MISMATCH")[0]
        for record in records
    }

    assert verdicts == {
        "output_checksum_sha256": "MISMATCH",
        "upstream_generator_run.output_checksum_sha256": "EXEMPT",
    }
    assert {record["path"] for record in records} == {"data/iedb_negatives_v5.csv"}
