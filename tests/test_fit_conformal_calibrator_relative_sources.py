"""scripts/fit_conformal_calibrator.py records its two sources through relative_source.

Why this test exists
--------------------
The coverage report it writes (models/v5/conformal_coverage_mode31.json is
tracked) records `source_model.path` and `source_oof.path`. Both were written as
`str(path)` with backslashes turned into "/", so an absolute `--model` or `--oof`
put a workstation path into the report. They now go through
`src.provenance_paths.relative_source`, as the other provenance writers do:

- an absolute source outside the checkout is recorded as `external:<name>`;
- an absolute source inside the checkout is recorded repo-relative;
- the default relative sources are recorded exactly as the tracked report
  records them, so regenerating it does not change those two strings;
- main(), end to end, writes no workstation path into the report.

The digests are still taken from the real files: only the recorded path changes.
No calibrator is fitted here. `compute_coverage_report` needs only the folds
count and seed of the fitted object, and main() runs with `fit_venn_abers` and
`oof_intervals` stubbed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO = os.path.realpath(REPO_ROOT)
SCRIPT_REL = "scripts/fit_conformal_calibrator.py"
ARTIFACT_REL = "models/v5/conformal_coverage_mode31.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / SCRIPT_REL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs():
    df = pd.DataFrame(
        {"peptide": ["SLYNTVATL", "GILGFVFTL"], "score": [0.25, 0.75], "label": [0, 1]}
    )
    model = SimpleNamespace(n_folds=2, random_state=7)
    return df, model, np.array([0.2, 0.7]), np.array([0.3, 0.8])


def _strings(node):
    """Yield every string in a JSON-shaped value, keys included."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _strings(key)
            yield from _strings(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _strings(value)


def _workstation_strings(document, *directories: Path) -> list[str]:
    """Every string in ``document`` that spells one of ``directories`` (in either
    separator form, resolved or as given) or is itself an absolute path."""
    spellings = set()
    for directory in directories:
        for form in (str(directory), os.path.realpath(directory)):
            spellings.update({form, form.replace("\\", "/")})
    return [
        s
        for s in _strings(document)
        if os.path.isabs(s) or any(spelling in s for spelling in spellings)
    ]


def _outside_the_checkout(path: Path) -> bool:
    try:
        return os.path.commonpath([REPO, os.path.realpath(path)]) != REPO
    except ValueError:  # on Windows, a different drive from the checkout
        return True


def test_an_absolute_source_outside_the_checkout_is_recorded_by_name_only(tmp_path):
    fitter = _load("fit_conformal_calibrator_outside_under_test")
    assert _outside_the_checkout(tmp_path), "tmp_path must lie outside the checkout"
    model_path, oof_path = tmp_path / "rf.joblib", tmp_path / "oof.csv"
    model_path.write_bytes(b"not a model\n")
    oof_path.write_bytes(b"peptide,score,label\r\nSLYNTVATL,0.25,0\r\n")
    df, model, lower, upper = _inputs()

    report = fitter.compute_coverage_report(df, model, lower, upper, model_path, oof_path)

    assert report["source_model"]["path"] == "external:rf.joblib"
    assert report["source_oof"]["path"] == "external:oof.csv"
    assert report["source_model"]["sha256"] == hashlib.sha256(b"not a model\n").hexdigest()
    lf_content = b"peptide,score,label\nSLYNTVATL,0.25,0\n"
    assert report["source_oof"]["sha256"] == hashlib.sha256(lf_content).hexdigest()
    assert _workstation_strings(report, tmp_path) == []


def test_an_absolute_source_inside_the_checkout_is_recorded_repo_relative():
    fitter = _load("fit_conformal_calibrator_inside_under_test")
    model_path = Path(REPO) / fitter.DEFAULT_MODEL_PATH
    oof_path = Path(REPO) / fitter.DEFAULT_OOF_PATH
    df, model, lower, upper = _inputs()

    report = fitter.compute_coverage_report(df, model, lower, upper, model_path, oof_path)

    assert report["source_model"]["path"] == "models/v5/rf_31feature_integrated.joblib"
    assert report["source_oof"]["path"] == "models/v5/rf_oof_predictions.csv"
    assert _workstation_strings(report, REPO_ROOT) == []


def test_the_default_sources_are_recorded_as_the_tracked_report_records_them(tmp_path, monkeypatch):
    """Run from a directory holding neither default, so nothing is hashed: only
    the recorded strings are compared. On Windows the defaults stringify with
    backslashes, which the report has always recorded as "/"."""
    fitter = _load("fit_conformal_calibrator_defaults_under_test")
    tracked = json.loads((REPO_ROOT / ARTIFACT_REL).read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)
    df, model, lower, upper = _inputs()

    report = fitter.compute_coverage_report(
        df, model, lower, upper, fitter.DEFAULT_MODEL_PATH, fitter.DEFAULT_OOF_PATH
    )

    assert report["source_model"]["path"] == tracked["source_model"]["path"]
    assert report["source_oof"]["path"] == tracked["source_oof"]["path"]
    assert tracked["source_model"]["path"] == fitter.DEFAULT_MODEL_PATH.as_posix()
    assert tracked["source_oof"]["path"] == fitter.DEFAULT_OOF_PATH.as_posix()


def test_main_writes_no_workstation_path_into_the_coverage_report(tmp_path, monkeypatch):
    fitter = _load("fit_conformal_calibrator_main_under_test")
    assert _outside_the_checkout(tmp_path), "tmp_path must lie outside the checkout"
    oof_path, model_path = tmp_path / "in" / "oof.csv", tmp_path / "in" / "rf.joblib"
    oof_path.parent.mkdir()
    pd.DataFrame(
        {
            "method": ["RandomForest", "RandomForest", "XGBoost"],
            "peptide": ["SLYNTVATL", "GILGFVFTL", "NLVPMVATV"],
            "score": [0.25, 0.75, 0.5],
            "label": [0, 1, 1],
            "virus": ["CMV", "IAV", "CMV"],
        }
    ).to_csv(oof_path, index=False)
    model_path.write_bytes(b"not a model\n")
    out_dir = tmp_path / "out"
    coverage = out_dir / "coverage.json"

    def fake_fit(scores, labels, groups, n_folds, random_state):
        return SimpleNamespace(n_folds=n_folds, random_state=random_state, n=len(scores))

    def fake_intervals(model):
        return np.full(model.n, 0.2), np.full(model.n, 0.4)

    monkeypatch.setattr(fitter, "fit_venn_abers", fake_fit)
    monkeypatch.setattr(fitter, "oof_intervals", fake_intervals)
    argv = [SCRIPT_REL, "--oof", str(oof_path), "--model", str(model_path)]
    argv += ["--output", str(out_dir / "calibrator.joblib"), "--coverage-output", str(coverage)]
    argv += ["--n-folds", "2"]
    monkeypatch.setattr(sys, "argv", argv)

    fitter.main()

    report = json.loads(coverage.read_text(encoding="utf-8"))
    assert report["n_samples"] == 2
    assert report["source_model"]["path"] == "external:rf.joblib"
    assert report["source_oof"]["path"] == "external:oof.csv"
    assert _workstation_strings(report, tmp_path) == []
    manifest = json.loads((out_dir / "model_artifact_checksums.json").read_text(encoding="utf-8"))
    assert sorted(manifest["artifacts"]) == ["calibrator.joblib", "coverage.json"]
    assert _workstation_strings(manifest, tmp_path) == []
