"""scripts/compute_pooled_honest_metric.py records its source through relative_source.

Why this test exists
--------------------
The row it writes (results/pooled_honest_same_pathogen.csv is tracked, and its
auc_roc cell is bound by the integrity harness) records `source_oof`. It was
written as `str(path)` with backslashes turned into "/", so an absolute `--oof`
put a workstation path into the row. It now goes through
`src.provenance_paths.relative_source`, as the other provenance writers do:

- an absolute source outside the checkout is recorded as `external:<name>`;
- an absolute source inside the checkout is recorded repo-relative;
- the default relative source is recorded exactly as the tracked row records
  it, so regenerating the row does not change that string;
- main(), end to end, writes no workstation path into any cell of the row.

Only the recorded string changes: the metric is still computed from the file
the argument names. The fixtures are synthetic. The committed out-of-fold file
is never read, so these tests neither depend on nor re-certify the published
Def A figures; where a path inside the checkout is needed, the reader is
stubbed and asked which path it was given.
"""

from __future__ import annotations

import csv
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO = os.path.realpath(REPO_ROOT)
SCRIPT_REL = "scripts/compute_pooled_honest_metric.py"
DEFAULT_OOF_REL = "models/v5/rf_oof_predictions_mode31.csv"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / SCRIPT_REL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _oof_frame(module) -> pd.DataFrame:
    """Two positives and two real-IEDB negatives on target viruses, plus one row
    the Def A filter drops (an off-panel virus)."""
    return pd.DataFrame(
        {
            "virus": ["CMV", "CMV", "DENV", "DENV", "Vaccinia"],
            "label": [1, 0, 1, 0, 0],
            "negative_origin": ["", module.REAL_NEG_ORIGIN, "", module.REAL_NEG_ORIGIN,
                                module.REAL_NEG_ORIGIN],
            "score": [0.9, 0.2, 0.7, 0.4, 0.95],
        }
    )


def _workstation_strings(values, *directories: Path) -> list[str]:
    """Every string in ``values`` that spells one of ``directories`` (in either
    separator form, resolved or as given) or is itself an absolute path."""
    spellings = set()
    for directory in directories:
        for form in (str(directory), os.path.realpath(directory)):
            spellings.update({form, form.replace("\\", "/")})
    return [
        s
        for s in values
        if isinstance(s, str)
        and (os.path.isabs(s) or any(spelling in s for spelling in spellings))
    ]


def _outside_the_checkout(path: Path) -> bool:
    try:
        return os.path.commonpath([REPO, os.path.realpath(path)]) != REPO
    except ValueError:  # on Windows, a different drive from the checkout
        return True


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_an_absolute_source_outside_the_checkout_is_recorded_by_name_only(tmp_path):
    metric = _load("compute_pooled_honest_metric_outside_under_test")
    assert _outside_the_checkout(tmp_path), "tmp_path must lie outside the checkout"
    oof_path = tmp_path / "oof.csv"
    _oof_frame(metric).to_csv(oof_path, index=False)

    row = metric.compute(oof_path)

    assert row["source_oof"] == "external:oof.csv"
    assert (row["n_pos"], row["n_neg"]) == (2, 2)
    assert _workstation_strings(row.values(), tmp_path) == []


def test_an_absolute_source_inside_the_checkout_is_recorded_repo_relative(monkeypatch):
    """The reader is stubbed, so the committed out-of-fold file is not read; the
    stub records the path it was handed, which must still be the absolute one."""
    metric = _load("compute_pooled_honest_metric_inside_under_test")
    oof_path = Path(REPO) / DEFAULT_OOF_REL
    frame = _oof_frame(metric)
    read = []

    def fake_read_csv(path, *args, **kwargs):
        read.append(path)
        return frame.copy()

    monkeypatch.setattr(metric, "pd", SimpleNamespace(read_csv=fake_read_csv))

    row = metric.compute(oof_path)

    assert read == [oof_path]
    assert row["source_oof"] == DEFAULT_OOF_REL
    assert _workstation_strings(row.values(), REPO_ROOT) == []


def test_the_default_source_is_recorded_as_the_tracked_row_records_it(tmp_path, monkeypatch):
    """main() with no --oof, run from a scratch directory holding a synthetic file
    at the default relative path. On Windows the default stringifies with
    backslashes, which the row has always recorded as "/"."""
    metric = _load("compute_pooled_honest_metric_defaults_under_test")
    tracked = _read_rows(REPO_ROOT / metric.TRACKED_OUTPUT)
    assert len(tracked) == 1
    work = tmp_path / "work"
    (work / DEFAULT_OOF_REL).parent.mkdir(parents=True)
    _oof_frame(metric).to_csv(work / DEFAULT_OOF_REL, index=False)
    monkeypatch.chdir(work)
    out = tmp_path / "out" / "honest.csv"

    assert metric.main(["--output", str(out)]) == 0

    (written,) = _read_rows(out)
    assert written["source_oof"] == tracked[0]["source_oof"] == DEFAULT_OOF_REL


def test_main_writes_no_workstation_path_into_the_row(tmp_path):
    metric = _load("compute_pooled_honest_metric_main_under_test")
    assert _outside_the_checkout(tmp_path), "tmp_path must lie outside the checkout"
    oof_path = tmp_path / "in" / "oof.csv"
    oof_path.parent.mkdir()
    _oof_frame(metric).to_csv(oof_path, index=False)
    out = tmp_path / "out" / "honest.csv"

    assert metric.main(["--oof", str(oof_path), "--output", str(out)]) == 0

    rows = _read_rows(out)
    assert len(rows) == 1
    assert rows[0]["source_oof"] == "external:oof.csv"
    cells = [cell for row in rows for item in row.items() for cell in item]
    assert _workstation_strings(cells, tmp_path) == []
