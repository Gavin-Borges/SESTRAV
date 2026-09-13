"""Smoke tests for the ``sestrav`` CLI (MASTER_STRATEGIC_PLAN.md Part 12, obj #14).

Verifies the dependency-light entry points run and produce parseable output
without requiring a model, FASTA, or heavy stage modules: ``info``, the no-arg
help banner, and ``--help`` for the program and each subcommand.
"""

from unittest.mock import create_autospec

import pandas as pd
import pytest

from src.cli import main


def test_info_returns_zero_and_prints_summary(capsys):
    rc = main(["info"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SESTRAV Environment Info" in out
    assert "torch" in out  # provenance summary lists torch (or 'not installed')


def test_no_subcommand_prints_help_and_returns_zero(capsys):
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Subcommands:" in out
    assert "predict" in out and "validate" in out and "benchmark" in out


def test_top_level_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("sub", ["predict", "validate", "benchmark", "info"])
def test_subcommand_help_exits_zero(sub):
    with pytest.raises(SystemExit) as exc:
        main([sub, "--help"])
    assert exc.value.code == 0


def test_validate_requires_model_dir(capsys):
    """validate retrains, so its destination is explicit - no fallback to models/."""
    with pytest.raises(SystemExit) as exc:
        main(["validate", "--dataset", "does_not_matter.csv"])
    assert exc.value.code != 0
    assert "--model-dir" in capsys.readouterr().err


def test_predict_exposes_virus_and_per_virus_calibration_dir_flags(capsys):
    """A1-B: predict's --help must advertise both new flags."""
    with pytest.raises(SystemExit) as exc:
        main(["predict", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--virus" in out
    assert "--per-virus-calibration-dir" in out


# ---------------------------------------------------------------------------
# cmd_predict, executed for real.
#
# This replaces a substring check against cmd_predict's own SOURCE TEXT. No
# test invoked cmd_predict at all, so nothing verified that `args` actually
# CARRIES the attribute that line reads: main()'s p_predict and cmd_predict are
# joined only by an argparse dest name, and nothing pinned it.
#
# Measured, not argued. Giving --virus an explicit `dest="virus_label"` leaves
# the whole existing suite green (10/10 passed), while a real
# `sestrav predict --virus HPV16` runs Stages 1 to 3 and then dies at Stage 4
# with AttributeError: 'Namespace' object has no attribute 'virus'. The
# source-text line is untouched by that edit, so it could not see it.
#
# Stages are stubbed because the assertion is purely about argument forwarding,
# not about any stage's behaviour - the same shape, and the same both-directions
# rationale, as tests/test_pipeline_forwards_freeze_mode.py.
# ---------------------------------------------------------------------------


@pytest.fixture
def stage_modules():
    """The four stage modules cmd_predict imports lazily, inside its body.

    Patching targets these modules rather than src.cli, because each
    ``from functions.stageN_... import ...`` line runs at CALL time and re-reads
    the attribute off the source module.
    """
    import functions.stage1_peptide_generation as s1
    import functions.stage2_mhc_binding_prediction as s2
    import functions.stage3_tcr_feature_extraction as s3
    import functions.stage4_immunogenicity_scoring as s4

    return s1, s2, s3, s4


def _stub_stages(monkeypatch, stage_modules, captured):
    """Stub stages 1-3 and spy on stage 4, recording the kwargs it was handed."""
    s1, s2, s3, s4 = stage_modules
    frame = pd.DataFrame({"peptide": ["CLGGLLTMV"], "immunogenicity_score": [0.5]})
    monkeypatch.setattr(s1, "generate_peptides", lambda *a, **k: frame)
    monkeypatch.setattr(s2, "predict_binding", lambda *a, **k: frame)
    monkeypatch.setattr(s3, "extract_tcr_features", lambda *a, **k: frame)

    def _score(features_df, proteome_id, **kwargs):
        captured.update(kwargs)
        return frame, None

    # create_autospec, not a bare closure: it binds the REAL signature of
    # score_immunogenicity, so a version that stopped accepting `virus=` at all
    # raises TypeError here instead of being silently absorbed by **kwargs. A
    # bare closure was measured to be blind to exactly that regression.
    monkeypatch.setattr(
        s4, "score_immunogenicity", create_autospec(s4.score_immunogenicity, side_effect=_score)
    )


def _predict_argv(tmp_path, *extra):
    fasta = tmp_path / "hpv16.fasta"
    fasta.write_text(">prot\nMAAAKLLGV\n", encoding="utf-8")
    model = tmp_path / "rf_31feature_integrated.joblib"
    model.write_bytes(b"stub-model")  # only _require_file's isfile() ever looks
    return [
        "predict",
        "--fasta",
        str(fasta),
        "--model",
        str(model),
        "--output",
        str(tmp_path / "out"),
        *extra,
    ]


def test_predict_forwards_the_virus_flags_into_stage_four(
    monkeypatch, tmp_path, capsys, stage_modules
):
    """``--virus HPV16`` must arrive at score_immunogenicity as ``virus='HPV16'``.

    If it does not, every peptide is scored by the global calibrator while the
    run reports itself as a per-virus one.
    """
    captured: dict = {}
    _stub_stages(monkeypatch, stage_modules, captured)

    cal_dir = tmp_path / "per_virus_calibrators"
    rc = main(
        _predict_argv(tmp_path, "--virus", "HPV16", "--per-virus-calibration-dir", str(cal_dir))
    )
    capsys.readouterr()  # cmd_predict prints a stage log and a top-10 preview

    assert rc == 0
    assert captured.get("virus") == "HPV16", (
        f"--virus never reached Stage 4; it got virus={captured.get('virus')!r}, "
        "so the global calibrator would have scored every peptide"
    )
    assert captured.get("per_virus_calibration_dir") == str(cal_dir)


def test_predict_leaves_the_virus_flags_unset_when_omitted(
    monkeypatch, tmp_path, capsys, stage_modules
):
    """Both directions are asserted deliberately.

    Pinning only the populated case would also pass against a cmd_predict that
    hardcoded a virus, which is a different defect in the opposite direction -
    the same reason test_run_pipeline_forwards_configured_freeze_mode
    parametrizes both values.
    """
    captured: dict = {}
    _stub_stages(monkeypatch, stage_modules, captured)

    rc = main(_predict_argv(tmp_path))
    capsys.readouterr()

    assert rc == 0
    assert captured.get("virus") is None
    assert captured.get("per_virus_calibration_dir") is None
