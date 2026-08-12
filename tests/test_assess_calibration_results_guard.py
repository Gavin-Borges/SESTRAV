"""Guard and honesty tests for scripts/assess_calibration.py.

This script writes a git-tracked results/ artifact
(results/calibration_assessment_v5_mode31.csv) plus a .provenance.json sidecar,
and the numbers in it are cited by docs/paper.md. It carried the shared
overwrite guard from src/artifact_guard.py from the start, but had no test file
at all, which is the gap this closes. src/artifact_guard.py's docstring states
the reason a shared implementation exists: "so a module cannot quietly be held
to a weaker standard than its siblings". A guarded module with no tests is
exactly that weaker standard, reintroduced by omission rather than by divergence.

Four things are pinned here.

1. BOTH planned paths are load-bearing. `_planned_paths` enumerates the CSV and
   its provenance sidecar; a future edit that forgets the sidecar would leave a
   guard that still looks correct in every whole-directory test while silently
   permitting the sidecar to be rewritten. The sidecar is checked on its own,
   not only alongside the CSV.
2. The escape hatch works, and the guard runs BEFORE any input is read (a guard
   defined but placed after the OOF load would pass a naive collision test and
   still do the work).
3. The honesty invariant of `cross_fitted_calibration`: no row is calibrated by
   a fold that saw its peptide, and every row is calibrated exactly once. The
   whole point of this script over scripts/fit_calibrator.py is that its ECE is
   out-of-sample, so peptide-group disjointness between train and test is the
   property every number in the artifact rests on. Mode-31 features are a pure
   function of the peptide string, so two rows sharing a peptide are
   feature-identical and an ungrouped split would leak.
4. Scope completeness: n(pooled_all) == n(target_viruses) + n(off_panel), and
   off_panel is emitted unconditionally - including when it is empty. The
   pooled figure is an artifact of cancellation between two populations
   miscalibrated in opposite directions (see the script's module docstring), so
   it is only interpretable next to the off_panel row. An edit that dropped
   that scope would leave the pooled number quotable and unfalsifiable.

Everything runs against a small synthetic OOF frame. The real
models/v5/rf_oof_predictions_mode31.csv is 35k+ rows and permission-gated; a
test that depended on it would be slow and would couple guard behaviour to
published science it is not testing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts import assess_calibration as ac
from scripts.fit_calibrator import TARGET_VIRUSES

# Two viruses deliberately outside the nine-virus panel, so `off_panel` is a
# real population in the fixture rather than an empty slice.
OFF_PANEL_VIRUSES = ("Rotavirus", "Zika")


def _synthetic_oof(*, include_off_panel: bool = True, n_peptides: int = 30) -> pd.DataFrame:
    """A small, valid stand-in for models/v5/rf_oof_predictions_mode31.csv.

    One label per peptide (the real feature set is a pure function of the
    peptide, so rows sharing a peptide share a label in practice), two rows per
    peptide so the group constraint has something to bite on, and enough groups
    of each class for StratifiedGroupKFold's five folds.
    """
    viruses = list(TARGET_VIRUSES)
    if include_off_panel:
        viruses += list(OFF_PANEL_VIRUSES)

    rows: list[dict[str, Any]] = []
    for i in range(n_peptides):
        peptide = f"PEPTIDE{i:03d}"
        label = i % 2
        virus = viruses[i % len(viruses)]
        for rep in range(2):
            rows.append(
                {
                    "method": "RandomForest",
                    "peptide": peptide,
                    "virus": virus,
                    "label": label,
                    "score": round(0.05 + 0.9 * (((i * 7) + (rep * 3)) % 20) / 19.0, 4),
                }
            )
    # Rows the loader must drop: a non-RandomForest method, and a NaN score.
    rows.append(
        {
            "method": "XGBoost",
            "peptide": "PEPTIDE000",
            "virus": viruses[0],
            "label": 1,
            "score": 0.9,
        }
    )
    rows.append(
        {
            "method": "RandomForest",
            "peptide": "PEPTIDE001",
            "virus": viruses[0],
            "label": 0,
            "score": np.nan,
        }
    )
    return pd.DataFrame(rows)


def _rf_frame() -> pd.DataFrame:
    """The synthetic fixture after the loader's own filtering, row-index reset.

    Mirrors `ac._load_rf_oof` rather than calling it, so the fold tests below do
    not need a file on disk and are not coupled to the loader's I/O.
    """
    df = _synthetic_oof()
    df = df[df["method"] == "RandomForest"].dropna(subset=["score", "label"])
    return df.reset_index(drop=True)


@pytest.fixture()
def oof_path(tmp_path: Path) -> Path:
    path = tmp_path / "oof.csv"
    _synthetic_oof().to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Planned-path enumeration
# ---------------------------------------------------------------------------


def test_planned_paths_lists_the_csv_and_its_provenance_sidecar(tmp_path: Path) -> None:
    """The sidecar is a written artifact, so it must be a planned one."""
    out = tmp_path / "c.csv"
    names = [Path(p).name for p in ac._planned_paths(out)]
    assert names == ["c.csv", "c.csv.provenance.json"]


def test_planned_paths_are_scoped_to_the_output_parent(tmp_path: Path) -> None:
    out = tmp_path / "c.csv"
    for p in ac._planned_paths(out):
        assert Path(p).parent == tmp_path


def test_planned_paths_are_unique(tmp_path: Path) -> None:
    planned = ac._planned_paths(tmp_path / "c.csv")
    assert len(planned) == len(set(planned))


def test_provenance_path_appends_rather_than_replaces_the_suffix(tmp_path: Path) -> None:
    """`.with_suffix()` on a bare name would turn c.csv into c.provenance.json,
    which is a different file from the one the writer actually creates."""
    assert ac._provenance_path(tmp_path / "c.csv").name == "c.csv.provenance.json"


# ---------------------------------------------------------------------------
# Guard behaviour: BOTH planned paths, one at a time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("existing_name", ["c.csv", "c.csv.provenance.json"])
def test_guard_refuses_when_either_planned_artifact_exists(
    tmp_path: Path, oof_path: Path, existing_name: str
) -> None:
    """Checked one file at a time on purpose.

    A `_planned_paths` that forgot the sidecar would still pass a both-files
    test (the CSV alone trips the guard) while leaving the sidecar - the file
    recording which OOF hash and splitter produced the published numbers -
    silently overwritable.
    """
    (tmp_path / existing_name).write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        ac.main(["--oof", str(oof_path), "--out", str(tmp_path / "c.csv")])
    assert existing_name in str(exc.value), (
        f"{existing_name} is written by this script but does not trip the guard"
    )


def test_guard_names_every_collision_and_the_escape_hatch(
    tmp_path: Path, oof_path: Path
) -> None:
    for name in ("c.csv", "c.csv.provenance.json"):
        (tmp_path / name).write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc:
        ac.main(["--oof", str(oof_path), "--out", str(tmp_path / "c.csv")])
    message = str(exc.value)
    assert "c.csv" in message
    assert "c.csv.provenance.json" in message
    assert "2 existing artifact(s)" in message, (
        f"message under-reports how many artifacts would be replaced: {message}"
    )
    assert "--out" in message, "message does not say which flag to redirect"
    assert "--allow-overwrite" in message, "message does not document its own escape hatch"


def test_guard_is_silent_on_a_fresh_destination(tmp_path: Path, oof_path: Path) -> None:
    """A first-ever run into an empty directory must not be blocked."""
    assert ac.main(["--oof", str(oof_path), "--out", str(tmp_path / "c.csv")]) == 0


def test_guard_runs_before_the_oof_file_is_read(tmp_path: Path) -> None:
    """A guard placed after the load would pass every collision test above and
    still have done the work. Pointing --oof at a file that does not exist makes
    the ordering observable: FileExistsError means the guard won the race."""
    (tmp_path / "c.csv").write_text("published", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ac.main(["--oof", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "c.csv")])


def test_allow_overwrite_disarms_the_guard(tmp_path: Path, oof_path: Path) -> None:
    """The escape hatch must work with every planned artifact already present."""
    out = tmp_path / "c.csv"
    for name in ("c.csv", "c.csv.provenance.json"):
        (tmp_path / name).write_text("published", encoding="utf-8")
    assert ac.main(["--oof", str(oof_path), "--out", str(out), "--allow-overwrite"]) == 0
    assert out.read_text(encoding="utf-8") != "published"
    assert "scope" in pd.read_csv(out).columns


# ---------------------------------------------------------------------------
# The honesty invariant: cross-fitting must not leak a peptide across a fold
# ---------------------------------------------------------------------------


def _record_folds(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture every (train, test, groups) the module's splitter yields.

    Delegates to the real StratifiedGroupKFold rather than reimplementing the
    split, so what is asserted is the fold structure the script actually uses -
    including the fact that it passes the peptide column as `groups`. Recreating
    the split in the test would only prove the test can call sklearn.
    """
    folds: list[dict[str, Any]] = []
    base = ac.StratifiedGroupKFold

    class _Recorder:
        def __init__(self, **kwargs: Any) -> None:
            self._inner = base(**kwargs)

        def split(self, X: Any, y: Any = None, groups: Any = None) -> Any:
            for train_idx, test_idx in self._inner.split(X, y, groups=groups):
                folds.append(
                    {
                        "train": np.asarray(train_idx),
                        "test": np.asarray(test_idx),
                        "groups": np.asarray(groups),
                    }
                )
                yield train_idx, test_idx

    monkeypatch.setattr(ac, "StratifiedGroupKFold", _Recorder)
    return folds


def test_cross_fitting_never_shares_a_peptide_between_train_and_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero peptide-group intersection in every fold.

    Mode-31 features are a pure function of the peptide string, so a peptide
    present on both sides of a split makes the "out-of-sample" ECE in the
    published artifact an in-sample one.
    """
    folds = _record_folds(monkeypatch)
    df = _rf_frame()
    ac.cross_fitted_calibration(df)

    assert len(folds) == ac.N_SPLITS, f"expected {ac.N_SPLITS} folds, recorded {len(folds)}"
    for i, fold in enumerate(folds):
        groups = fold["groups"]
        # Asserted explicitly, not just for length: a split grouped on the row
        # index instead of the peptide would satisfy the disjointness check
        # below vacuously (every row its own group) while leaking every
        # duplicated peptide across the fold boundary.
        assert list(groups) == list(df["peptide"]), (
            "the splitter was grouped on something other than the peptide column"
        )
        overlap = set(groups[fold["train"]]) & set(groups[fold["test"]])
        assert not overlap, f"fold {i} trains and tests on shared peptide(s): {sorted(overlap)}"


def test_every_row_is_calibrated_by_exactly_one_test_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partition, not merely a cover.

    A row in two test folds would be overwritten by whichever fold ran last
    (silently discarding one estimate); a row in none would keep the NaN the
    function raises on. Both are checked here against the fold indices rather
    than against the NaN guard, which only catches the second failure mode.
    """
    folds = _record_folds(monkeypatch)
    df = _rf_frame()
    calibrated = ac.cross_fitted_calibration(df)

    assigned = np.concatenate([fold["test"] for fold in folds])
    assert len(assigned) == len(df), "rows are assigned to more or fewer than one test fold"
    assert sorted(assigned.tolist()) == list(range(len(df))), (
        "the test folds are not a partition of the row index"
    )
    assert len(calibrated) == len(df)
    assert not np.isnan(calibrated).any()


def test_cross_fitted_scores_stay_in_the_unit_interval() -> None:
    calibrated = ac.cross_fitted_calibration(_rf_frame())
    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0


# ---------------------------------------------------------------------------
# Scope completeness
# ---------------------------------------------------------------------------


def test_pooled_scope_is_exactly_target_plus_off_panel(oof_path: Path) -> None:
    """The decomposition has to be exhaustive and non-overlapping, or the
    pooled/target gap the module docstring attributes to cancellation is
    attributable to a dropped subpopulation instead."""
    result = ac.run(oof_path)
    n = {row["scope"]: int(row["n"]) for _, row in result.iterrows()}
    assert n["pooled_all"] == n["target_viruses"] + n["off_panel"], (
        f"scopes do not partition the population: {n}"
    )


def test_off_panel_scope_is_always_emitted(oof_path: Path) -> None:
    """Quoting the pooled ECE without the off_panel row invites the reading that
    the pool is simply easier to calibrate - which the off_panel ece_cal
    refutes. The row is what makes the pooled figure checkable, so its presence
    is a contract, not a formatting choice."""
    scopes = list(ac.run(oof_path)["scope"])
    assert "off_panel" in scopes
    assert "pooled_all" in scopes
    assert "target_viruses" in scopes
    assert scopes[:3] == ["pooled_all", "target_viruses", "off_panel"]


# numpy warns twice on the empty slice (positive_rate is a mean of no rows and
# is legitimately nan here). That is the expected shape of an empty scope row,
# not a defect this test is reporting, so it is silenced rather than left as
# suite noise.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_off_panel_scope_is_emitted_even_when_empty(tmp_path: Path) -> None:
    """The strongest form of the contract above.

    A `run()` that emitted off_panel only when non-empty would look correct on
    the real corpus and silently drop the row on any future all-panel input,
    turning the pooled figure back into an unfalsifiable one.
    """
    path = tmp_path / "targets_only.csv"
    _synthetic_oof(include_off_panel=False).to_csv(path, index=False)
    result = ac.run(path)
    off = result[result["scope"] == "off_panel"]
    assert len(off) == 1, "off_panel was dropped when the population was empty"
    assert int(off.iloc[0]["n"]) == 0


def test_per_virus_rows_cover_the_target_panel(oof_path: Path) -> None:
    scopes = set(ac.run(oof_path)["scope"])
    assert set(TARGET_VIRUSES) <= scopes
    for off_panel_virus in OFF_PANEL_VIRUSES:
        assert off_panel_virus not in scopes, (
            "an off-panel virus got its own scope row; the per-virus block is "
            "meant to be restricted to the nine-virus panel"
        )


def test_loader_drops_non_random_forest_and_nan_score_rows(oof_path: Path) -> None:
    """The assessed population must match the one fit_calibrator fits on."""
    raw = pd.read_csv(oof_path)
    pooled = ac.run(oof_path)
    n_pooled = int(pooled[pooled["scope"] == "pooled_all"].iloc[0]["n"])
    assert n_pooled == len(
        raw[(raw["method"] == "RandomForest") & raw["score"].notna()]
    ), "the loader kept rows fit_calibrator would have dropped, or dropped rows it keeps"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_runs_produce_byte_identical_csvs(tmp_path: Path, oof_path: Path) -> None:
    """RANDOM_STATE is fixed, so the artifact must be reproducible bit for bit.

    Only the CSV is compared: the provenance sidecar carries a generation
    timestamp and is expected to differ between runs.
    """
    first = tmp_path / "first" / "c.csv"
    second = tmp_path / "second" / "c.csv"
    assert ac.main(["--oof", str(oof_path), "--out", str(first)]) == 0
    assert ac.main(["--oof", str(oof_path), "--out", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes(), (
        "two runs over identical input disagree; the assessment is not reproducible"
    )


def test_provenance_sidecar_records_the_out_of_sample_claim(
    tmp_path: Path, oof_path: Path
) -> None:
    """in_sample=False is the one claim separating this artifact from
    fit_calibrator.py's overfit ECE of 0.00000."""
    out = tmp_path / "c.csv"
    assert ac.main(["--oof", str(oof_path), "--out", str(out)]) == 0
    sidecar = ac._provenance_path(out)
    assert sidecar.is_file()
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    assert provenance["in_sample"] is False
    assert provenance["splitter"] == "StratifiedGroupKFold(groups=peptide)"
    assert provenance["n_rows_assessed"] == int(
        ac.run(oof_path).iloc[0]["n"]
    )
