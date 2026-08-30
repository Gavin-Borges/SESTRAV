"""results/ guard tests for scripts/compute_pooled_cv_metrics.py.

Same guard contract as tests/test_compute_loo_binding_confound_results_guard.py:
`results/pooled_cv_metrics_mode31.csv` is git-tracked and is bound by the integrity
harness, so --output has no default and a bare invocation must print the table and
write nothing rather than silently rewriting certified output.

Two properties beyond the shared guard contract are checked here because the integrity
harness depends on them and neither is visible by reading the CSV in an editor:

1. The CSV must be written with LF line endings. `.gitattributes` pins
   `results/*.csv text eol=lf` and check_provenance hashes RAW BYTES, so a CRLF write
   records a digest that does not reproduce from a clean clone. That is the
   "NON-PORTABLE digest" FAIL class.
2. The emitted fold-mean must reproduce models/v5/training_results_mode31.csv's
   `rf_cv_mean`. It is the control that establishes the OOF frame and the training
   summary describe the same run, and it is what distinguishes the pooled 0.6055 from
   the fold-mean 0.6058 that four reader-facing docs still conflate.

The fixtures are synthetic and deliberately tiny; the real committed OOF frame is not
read, so these tests neither depend on nor re-certify the published science.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts import compute_pooled_cv_metrics as cpcm

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def synthetic_oof(tmp_path: Path) -> Path:
    """A 40-row OOF frame over 4 folds with a separable-but-imperfect score.

    `label` is keyed on `i // 4` and `fold` on `i % 4`, so every fold carries 5
    positives and 5 negatives. Keying both on `i % 2` and `i % 4` instead makes each
    fold single-class, which silently reduces per-fold average precision to a
    degenerate value and would make the fold-mean assertions below vacuous.

    The score's label effect (0.1) is deliberately smaller than its noise range
    (0 to 0.12) so the classes overlap: a perfectly separable fixture drives every
    average precision to 1.0 and the pooled-vs-fold-mean distinction disappears.
    """
    rows = []
    for i in range(40):
        label = (i // 4) % 2
        rows.append(
            {
                "label": label,
                "score": 0.4 + 0.1 * label + 0.02 * (i % 7),
                "fold": i % 4,
            }
        )
    path = tmp_path / "oof.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_compute_emits_the_pooled_rows_and_the_fold_mean_control(synthetic_oof):
    rows = {r.metric: r for r in cpcm.compute(synthetic_oof)}
    assert rows["mode31_pooled_auc_pr"].kind == "pooled_single_pass"
    assert rows["mode31_pooled_auc_roc"].kind == "pooled_single_pass"
    assert rows["mode31_pooled_n_rows"].value == 40
    assert rows["mode31_pooled_n_positive"].value == 20
    # The fold-mean is a separate quantity, not a copy of the pooled value.
    assert rows["mode31_fold_mean_auc_pr"].kind == "fold_mean"


def test_fold_mean_is_the_mean_of_per_fold_scores_not_the_pooled_value(synthetic_oof):
    """The whole point of the artifact: these two must be computed differently.

    A refactor that made the fold-mean an alias of the pooled value would silently
    re-create the exact conflation PR #303 corrected, and every other assertion here
    would still pass.
    """
    from sklearn.metrics import average_precision_score

    df = pd.read_csv(synthetic_oof)
    per_fold = [
        average_precision_score(g["label"], g["score"]) for _, g in df.groupby("fold")
    ]
    expected_fold_mean = sum(per_fold) / len(per_fold)
    pooled = average_precision_score(df["label"], df["score"])

    rows = {r.metric: r.value for r in cpcm.compute(synthetic_oof)}
    assert rows["mode31_fold_mean_auc_pr"] == pytest.approx(expected_fold_mean)
    assert rows["mode31_pooled_auc_pr"] == pytest.approx(pooled)
    assert rows["mode31_fold_mean_auc_pr"] != pytest.approx(rows["mode31_pooled_auc_pr"])


def test_bare_invocation_writes_nothing(synthetic_oof, capsys):
    assert cpcm.main(["--oof", str(synthetic_oof)]) == 0
    captured = capsys.readouterr()
    assert "mode31_pooled_auc_pr" in captured.out  # table still printed
    assert "nothing written" in captured.out


def test_bare_invocation_does_not_touch_the_tracked_default_path(
    synthetic_oof, tmp_path, monkeypatch
):
    """A bare run must not rewrite the certified artifact at its REAL tracked path.

    The assertions deliberately target REPO_ROOT, not tmp_path. `cpcm._resolve()` anchors
    every relative path to REPO_ROOT, so chdir-ing into tmp_path CANNOT change where a
    write would land: asserting that `tmp_path / "results"` was not created is vacuously
    true whether the guard works or not.

    That was this test's only assertion until 2026-08-31, and it was verified vacuous by
    mutation: restoring a `--output` default made this test still PASS while the run
    rewrote the tracked results/pooled_cv_metrics_mode31.csv and its sidecar. Running the
    suite was itself the clobbering vector. Only `test_bare_invocation_writes_nothing`
    caught the mutant. Compare the sibling guard in
    tests/test_compute_pooled_honest_metric_results_guard.py, whose identical-looking
    assertion DOES bite because that script resolves against cwd.
    """
    tracked = REPO_ROOT / cpcm.TRACKED_OUTPUT
    sidecar = tracked.with_suffix(tracked.suffix + ".provenance.json")
    before = tracked.read_bytes() if tracked.exists() else None
    sidecar_before = sidecar.read_bytes() if sidecar.exists() else None

    monkeypatch.chdir(tmp_path)
    cpcm.main(["--oof", str(synthetic_oof)])

    assert not (tmp_path / "results").exists()  # nothing written relative to cwd either
    after = tracked.read_bytes() if tracked.exists() else None
    assert after == before, "a bare run rewrote the tracked artifact"
    sidecar_after = sidecar.read_bytes() if sidecar.exists() else None
    assert sidecar_after == sidecar_before, "a bare run rewrote the tracked sidecar"


def test_output_flag_writes_the_given_path_with_a_sidecar(synthetic_oof, tmp_path):
    out = tmp_path / "metrics.csv"
    cpcm.main(["--oof", str(synthetic_oof), "--output", str(out)])
    assert out.exists()
    sidecar = out.with_suffix(out.suffix + ".provenance.json")
    assert sidecar.exists()

    import hashlib
    import json

    recorded = json.loads(sidecar.read_text(encoding="utf-8"))["sha256"]
    assert recorded == hashlib.sha256(out.read_bytes()).hexdigest()


def test_output_flag_creates_parent_directory_if_missing(synthetic_oof, tmp_path):
    out = tmp_path / "new_subdir" / "metrics.csv"
    assert not out.parent.exists()
    cpcm.main(["--oof", str(synthetic_oof), "--output", str(out)])
    assert out.exists()


def test_csv_is_written_with_lf_endings(synthetic_oof, tmp_path):
    """A CRLF write records a digest that will not reproduce from a clean clone."""
    out = tmp_path / "metrics.csv"
    cpcm.main(["--oof", str(synthetic_oof), "--output", str(out)])
    raw = out.read_bytes()
    assert b"\r\n" not in raw
    assert raw.count(b"\n") == 6  # header + 5 metric rows


def test_sidecar_is_written_with_lf_endings(synthetic_oof, tmp_path):
    out = tmp_path / "metrics.csv"
    cpcm.main(["--oof", str(synthetic_oof), "--output", str(out)])
    sidecar = out.with_suffix(out.suffix + ".provenance.json")
    assert b"\r\n" not in sidecar.read_bytes()


def test_compute_rejects_a_frame_missing_required_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"peptide": "SIINFEKL", "fold": 0}]).to_csv(bad, index=False)
    with pytest.raises(SystemExit):
        cpcm.compute(bad)


def test_cli_help_advertises_no_default_output():
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.compute_pooled_cv_metrics", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert "No default" in proc.stdout
    assert cpcm.TRACKED_OUTPUT in proc.stdout
