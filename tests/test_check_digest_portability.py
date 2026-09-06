from __future__ import annotations

import json

from scripts.check_digest_portability import classify, extract_records, print_human

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
