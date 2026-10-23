"""models/v5/conformal_coverage_mode31.json: its source digest reproduces from the Git blob.

Why this test exists
--------------------
The coverage report records `source_oof.sha256`, the digest of the OOF file the
calibrator was fitted on. scripts/fit_conformal_calibrator.py hashed that file's
raw bytes, and the run that wrote the report read a CRLF working copy of
models/v5/rf_oof_predictions.csv, which .gitattributes pins to LF. So the
recorded value was the CRLF rendering's digest and matched no clone's blob.
The same digest pair was corrected for a sibling carrier on 2026-08-13 (claims
register D24-resid), and this carrier was missed because
scripts/check_digest_portability.py did not scan the report. Pinned here:

- the recorded digest is the digest of the source's LF content;
- the fitter's digest does not depend on the checkout's line endings, and it
  writes the report itself with LF line endings;
- the digest gate scans the report and calls its source digest PORTABLE.
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_REL = "models/v5/conformal_coverage_mode31.json"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_recorded_source_digest_is_the_digest_of_the_lf_content() -> None:
    report = json.loads((REPO_ROOT / ARTIFACT_REL).read_text(encoding="utf-8"))
    source = report["source_oof"]
    lf_content = (REPO_ROOT / source["path"]).read_bytes().replace(b"\r\n", b"\n")
    assert source["sha256"] == hashlib.sha256(lf_content).hexdigest()


def test_the_fitter_digest_ignores_the_checkout_line_endings(tmp_path: Path) -> None:
    fitter = _load("fit_conformal_calibrator_digest_under_test", "scripts/fit_conformal_calibrator.py")
    content = b"peptide,score,label\nSLYNTVATL,0.5,1\nGILGFVFTL,0.25,0\n"
    lf, crlf = tmp_path / "lf.csv", tmp_path / "crlf.csv"
    lf.write_bytes(content)
    crlf.write_bytes(content.replace(b"\n", b"\r\n"))
    assert fitter.sha256_lf(lf) == fitter.sha256_lf(crlf) == hashlib.sha256(content).hexdigest()


def test_the_fitter_writes_the_report_with_lf_line_endings(tmp_path: Path) -> None:
    fitter = _load("fit_conformal_calibrator_writer_under_test", "scripts/fit_conformal_calibrator.py")
    report = {"n_samples": 2, "source_oof": {"path": "x.csv", "sha256": None}}
    out = tmp_path / "coverage.json"
    fitter.write_coverage_report(out, report)
    written = out.read_bytes()
    assert b"\r" not in written
    assert json.loads(written) == report


def test_the_digest_gate_scans_the_report_and_calls_its_source_portable() -> None:
    gate = _load("check_digest_portability_under_test", "scripts/check_digest_portability.py")
    assert any(fnmatch.fnmatch(ARTIFACT_REL, pattern) for pattern in gate.PATTERNS)
    rows = [row for row in gate.scan_repository(REPO_ROOT) if row["manifest"] == ARTIFACT_REL]
    verdicts = {row["source"]: row["verdict"] for row in rows}
    assert verdicts.get("source_oof.sha256") == "PORTABLE", verdicts
