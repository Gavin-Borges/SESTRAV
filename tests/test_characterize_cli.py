"""Characterisation tests for src/cli.py.

src/cli.py carries a __main__ guard, so .coveragerc.library omits it and the
required coverage gate never measures it. Before this file the `validate`
subcommand (cmd_validate and _build_validate_parser) had never executed under
the suite, and neither had cmd_info's torch-missing, config-absent and
mhcflurry-pin-mismatch branches; its other degraded-environment branches
already ran.

Every test here PINS CURRENT BEHAVIOUR. None asserts that the behaviour is
right; several record behaviour an owner may want to change, and say so in
their docstrings. Nothing in src/cli.py is edited, and no stage, model or
network is touched: the training call and the four prediction stages are
replaced with spies that keep the real signatures (create_autospec), so a
forwarding change raises instead of being absorbed by **kwargs.
"""

import argparse
import inspect
import sys
import types
from unittest.mock import create_autospec

import pandas as pd
import pytest

import src.cli as cli
import src.train_classifier as train_classifier

# Captured at import, before any test swaps it for a spy.
_REAL_TRAIN_MODELS = train_classifier.train_models

_RF_AVG = {"auc_pr": 0.5, "auc_roc": 0.75, "issr_10": 0.125}
_XGB_AVG = {"auc_pr": 0.25, "auc_roc": 0.625, "issr_10": 0.0}


def _spy_train_models(monkeypatch, rf_avg=None, xgb_avg=None):
    """Replace train_models on its own module; cmd_validate imports it at call time."""
    spy = create_autospec(
        train_classifier.train_models,
        return_value=(
            "rf-final",
            "xgb-final",
            dict(_RF_AVG if rf_avg is None else rf_avg),
            dict(_XGB_AVG if xgb_avg is None else xgb_avg),
        ),
    )
    monkeypatch.setattr(train_classifier, "train_models", spy)
    return spy


def _validate_args(tmp_path, **overrides):
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("peptide,label\nSLLMWITQV,1\n", encoding="utf-8")
    values = {
        "dataset": str(dataset),
        "binding_matrix": None,
        "model_dir": str(tmp_path / "models_out"),
        "folds": 3,
        "feature_mode": "31",
        "sample_weights": False,
        "allow_overwrite": False,
        "report": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


# ---------------------------------------------------------------------------
# cmd_validate
# ---------------------------------------------------------------------------


def test_validate_forwards_exactly_seven_keywords_and_no_cv_grouping(tmp_path, monkeypatch, capsys):
    """`sestrav validate` never forwards cv_group_by or fold_impute.

    OWNER LOOK: train_models' own defaults therefore apply, which are the
    UNGROUPED splitter (cv_group_by=None, peptide leakage per claims register
    D15) and whole-cache imputation (fold_impute=False). The module entry
    point `python -m src.train_classifier` defaults to the opposite on both,
    so the two entry points report CV figures from different splitters for the
    same flags.
    """
    # pins behaviour, not correctness
    spy = _spy_train_models(monkeypatch)
    binding = tmp_path / "binding.csv"
    binding.write_text("peptide\n", encoding="utf-8")
    args = _validate_args(
        tmp_path, binding_matrix=str(binding), sample_weights=True, allow_overwrite=True
    )

    assert cli.cmd_validate(args) == 0
    capsys.readouterr()

    assert spy.call_count == 1
    call = spy.call_args
    assert call.args == ()
    assert call.kwargs == {
        "data_path": args.dataset,
        "model_dir": args.model_dir,
        "n_cv_folds": 3,
        "feature_mode": 31,
        "binding_matrix_path": str(binding),
        "use_sample_weights": True,
        "allow_overwrite": True,
    }
    defaults = inspect.signature(_REAL_TRAIN_MODELS).parameters
    assert defaults["cv_group_by"].default is None
    assert defaults["fold_impute"].default is False


@pytest.mark.parametrize(
    "config, expected_mode, expected_matrix",
    [
        (
            {"feature_mode": 33, "binding_matrix_path": "cfg_matrix_absent.csv"},
            33,
            "cfg_matrix_absent.csv",
        ),
        ({}, 21, None),
    ],
    ids=["from-config", "no-config"],
)
def test_validate_falls_back_to_config_then_to_mode_21(
    tmp_path, monkeypatch, capsys, config, expected_mode, expected_matrix
):
    """Without --feature-mode the mode comes from config.yaml, else "21".

    The config's binding_matrix_path is forwarded WITHOUT the existence check
    that an explicit --binding-matrix gets, so a stale config path reaches
    train_models unchecked.
    """
    # pins behaviour, not correctness
    spy = _spy_train_models(monkeypatch)
    monkeypatch.setattr(cli, "_read_config", lambda: dict(config))

    assert cli.cmd_validate(_validate_args(tmp_path, feature_mode=None)) == 0
    out = capsys.readouterr().out

    assert spy.call_args.kwargs["feature_mode"] == expected_mode
    assert spy.call_args.kwargs["binding_matrix_path"] == expected_matrix
    assert f"[sestrav validate] Feature mode: {expected_mode}" in out


@pytest.mark.parametrize(
    "given, forwarded",
    [("30_esm", "30_esm"), ("99", 99), ("031", 31)],
    ids=["non-digit-kept-as-str", "unknown-digit-mode-int", "leading-zero-int"],
)
def test_validate_forwards_any_feature_mode_without_validation(
    tmp_path, monkeypatch, capsys, given, forwarded
):
    """Digit strings become ints, anything else stays a string, nothing is rejected HERE.

    OWNER LOOK: unlike `python -m src.train_classifier`, whose --feature-mode
    carries an argparse `choices` list and exits 2 on an unrecognised value,
    `sestrav validate --feature-mode 99` is accepted and handed to train_models
    unchecked. What this test pins is that forwarding, and it is unchanged.

    train_models itself now REJECTS such a mode: the unknown-mode guard raises
    above `os.makedirs`, so the dispatch chain's final `else` is unreachable for
    it and no mode-99 artifact is written. Measured on this file's own base: the
    call raises `Unknown feature_mode 99` and model_dir is never created. Before
    that guard the `else` DID train the 21-feature sequence-only matrix under a
    mode-99 filename, which is the defect it closed. See
    test_characterize_train_classifier's
    test_train_models_names_the_unknown_mode_it_rejects, which now asserts the
    refusal rather than pinning the old behaviour.
    """
    # pins behaviour, not correctness
    spy = _spy_train_models(monkeypatch)
    assert cli.cmd_validate(_validate_args(tmp_path, feature_mode=given)) == 0
    capsys.readouterr()
    assert spy.call_args.kwargs["feature_mode"] == forwarded
    assert type(spy.call_args.kwargs["feature_mode"]) is type(forwarded)


def test_validate_report_is_written_verbatim_into_new_nested_dirs(tmp_path, monkeypatch, capsys):
    # pins behaviour, not correctness
    _spy_train_models(monkeypatch)
    report = tmp_path / "deep" / "er" / "validation.md"
    args = _validate_args(tmp_path, report=str(report))

    assert cli.cmd_validate(args) == 0
    out = capsys.readouterr().out

    expected = (
        "# SESTRAV Validation Report\n\n"
        f"**Dataset:** `{args.dataset}`  \n"
        "**Feature mode:** 31  \n"
        "**CV folds:** 3  \n\n"
        "## Results\n\n"
        "| Model | AUC-PR | AUC-ROC | ISSR@10 |\n"
        "|-------|--------|---------|--------|\n"
        "| RF    | 0.5000 | 0.7500 | 0.1250 |\n"
        "| XGB   | 0.2500 | 0.6250 | 0.0000 |\n"
    )
    assert report.read_text(encoding="utf-8") == expected
    assert f"[sestrav validate] Report -> {args.report}" in out


def test_validate_report_crashes_after_training_when_a_metric_is_absent(
    tmp_path, monkeypatch, capsys
):
    """A metric missing from the CV averages becomes None and the report raises TypeError.

    OWNER LOOK: train_models has already run (and written its artifacts) by the
    time the report is formatted, so the command fails AFTER the expensive work
    and leaves a truncated report file behind.

    Unreachable in practice: train_models averages the per-fold dicts that
    evaluate() returns, and those always carry auc_pr, auc_roc and issr_10 (NaN
    rather than absent when a metric is undefined), so the spy here supplies a
    shape the real trainer does not return.
    """
    # pins behaviour, not correctness
    rf_avg = {"auc_pr": 0.5, "auc_roc": 0.75}
    spy = _spy_train_models(monkeypatch, rf_avg=rf_avg)
    report = tmp_path / "r" / "validation.md"

    with pytest.raises(TypeError):
        cli.cmd_validate(_validate_args(tmp_path, report=str(report)))
    capsys.readouterr()

    assert spy.call_count == 1
    assert report.exists()
    assert "| RF " not in report.read_text(encoding="utf-8")


def test_validate_missing_dataset_exits_2_with_the_validate_usage(tmp_path, capsys):
    # pins behaviour, not correctness
    missing = tmp_path / "absent.csv"
    with pytest.raises(SystemExit) as exc:
        cli.main(["validate", "--dataset", str(missing), "--model-dir", str(tmp_path / "m")])
    err = " ".join(capsys.readouterr().err.split())

    assert exc.value.code == 2
    assert "--dataset not found" in err
    assert "[--binding-matrix BINDING_MATRIX]" in err
    assert "--model-dir MODEL_DIR" in err
    assert "[--report REPORT]" in err


def test_validate_missing_explicit_binding_matrix_exits_2_before_training(
    tmp_path, monkeypatch, capsys
):
    # pins behaviour, not correctness
    spy = _spy_train_models(monkeypatch)
    args = _validate_args(tmp_path, binding_matrix=str(tmp_path / "absent_matrix.csv"))

    with pytest.raises(SystemExit) as exc:
        cli.cmd_validate(args)
    err = capsys.readouterr().err

    assert exc.value.code == 2
    assert "--binding-matrix not found" in err
    assert spy.call_count == 0


# ---------------------------------------------------------------------------
# cmd_predict
# ---------------------------------------------------------------------------


def _stub_stages(monkeypatch):
    """Spy on all four stages once per test; returns the dict the spies append to."""
    import functions.stage1_peptide_generation as s1
    import functions.stage2_mhc_binding_prediction as s2
    import functions.stage3_tcr_feature_extraction as s3
    import functions.stage4_immunogenicity_scoring as s4

    captured: dict = {}
    frame = pd.DataFrame({"peptide": ["CLGGLLTMV"], "immunogenicity_score": [0.5]})

    def _record(stage):
        def _inner(*args, **kwargs):
            captured.setdefault(stage, []).append((args[1:], dict(kwargs)))
            return frame if stage != "s4" else (frame, None)

        return _inner

    monkeypatch.setattr(
        s1, "generate_peptides", create_autospec(s1.generate_peptides, side_effect=_record("s1"))
    )
    monkeypatch.setattr(
        s2, "predict_binding", create_autospec(s2.predict_binding, side_effect=_record("s2"))
    )
    monkeypatch.setattr(
        s3,
        "extract_tcr_features",
        create_autospec(s3.extract_tcr_features, side_effect=_record("s3")),
    )
    monkeypatch.setattr(
        s4,
        "score_immunogenicity",
        create_autospec(s4.score_immunogenicity, side_effect=_record("s4")),
    )
    return captured


def _satisfy_predict_preflight(monkeypatch, tmp_path):
    """Meet cmd_predict's pre-flight refusals where they exist, so the pins stay on behaviour.

    A cmd_predict that checks before Stage 1 refuses without MHCflurry model data,
    which CI never fetches, and, in freeze mode, without a conformal calibrator.
    Every stage is stubbed here, so neither is needed: stub the model-data lookup
    (raising=False, because a cli.py without the check has no such function, and
    the stub is then inert) and put a calibrator beside the model, where Stage 4's
    resolver looks first when no calibrator path is given.
    """
    monkeypatch.setattr(cli, "_mhcflurry_model_data_path", lambda: "stubbed", raising=False)
    (tmp_path / "conformal_calibrator.joblib").write_bytes(b"stub-calibrator")


def _predict_argv(tmp_path, *extra):
    fasta = tmp_path / "proteome.fasta"
    fasta.write_text(">prot\nMAAAKLLGV\n", encoding="utf-8")
    model = tmp_path / "model.joblib"
    model.write_bytes(b"not-a-model")
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


def test_predict_feature_mode_flag_reaches_no_stage(tmp_path, monkeypatch, capsys):
    """`sestrav predict --feature-mode` is parsed and then never read.

    OWNER LOOK: the help text calls it a "Feature mode override", but every
    stage receives identical arguments with and without it; Stage 4 infers the
    layout from the model artifact instead.
    """
    # pins behaviour, not correctness
    captured = _stub_stages(monkeypatch)
    _satisfy_predict_preflight(monkeypatch, tmp_path)
    assert cli.main(_predict_argv(tmp_path, "--feature-mode", "33", "--no-freeze-mode")) == 0
    with_flag = dict(captured)
    captured.clear()

    assert cli.main(_predict_argv(tmp_path, "--no-freeze-mode")) == 0
    without_flag = dict(captured)
    capsys.readouterr()

    assert with_flag == without_flag
    assert set(with_flag) == {"s1", "s2", "s3", "s4"}
    for stage_calls in with_flag.values():
        for _args, kwargs in stage_calls:
            assert "feature_mode" not in kwargs


def test_predict_freeze_mode_falls_to_false_when_config_is_unreachable(
    tmp_path, monkeypatch, capsys
):
    """freeze_mode comes from a config.yaml found relative to the installed module.

    OWNER LOOK: pyproject.toml's package-data ships no config.yaml, so for a
    non-editable install that lookup finds nothing and `sestrav predict` runs
    with freeze_mode False, although the repository's config.yaml sets it true.
    Simulated here by pointing the module's __file__ at an empty tree.
    """
    # pins behaviour, not correctness
    captured = _stub_stages(monkeypatch)
    _satisfy_predict_preflight(monkeypatch, tmp_path)
    assert cli.main(_predict_argv(tmp_path)) == 0
    out_repo = capsys.readouterr().out
    in_repo = dict(captured)
    captured.clear()

    monkeypatch.setattr(cli, "__file__", str(tmp_path / "site-packages" / "src" / "cli.py"))
    assert cli._read_config() == {}
    assert cli.main(_predict_argv(tmp_path)) == 0
    out_installed = capsys.readouterr().out
    installed = dict(captured)

    assert in_repo["s4"][0][1]["freeze_mode"] is True
    assert "[sestrav predict] Freeze mode: True" in out_repo
    assert installed["s4"][0][1]["freeze_mode"] is False
    assert "[sestrav predict] Freeze mode: False" in out_installed


# ---------------------------------------------------------------------------
# _read_config and cmd_info
# ---------------------------------------------------------------------------


def test_read_config_is_empty_without_pyyaml(monkeypatch):
    # pins behaviour, not correctness
    monkeypatch.setitem(sys.modules, "yaml", None)
    assert cli._read_config() == {}


def test_info_reports_missing_mhcflurry_and_torch(monkeypatch, capsys):
    # pins behaviour, not correctness
    monkeypatch.setitem(sys.modules, "mhcflurry", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(cli, "_read_config", lambda: {"mhcflurry_model_version": "2.2.1"})

    assert cli.cmd_info(argparse.Namespace()) == 0
    out = capsys.readouterr().out

    assert "  mhcflurry       : not installed" in out
    assert "  torch           : not installed" in out
    assert "CUDA" not in out
    assert "mismatch" not in out


def _fake_modules(monkeypatch, mhcflurry_version):
    fake_mhc = types.ModuleType("mhcflurry")
    fake_mhc.__version__ = mhcflurry_version
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "0.0-fake"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "mhcflurry", fake_mhc)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_info_warns_on_a_pin_mismatch_and_truncates_the_allele_panel(monkeypatch, capsys):
    # pins behaviour, not correctness
    _fake_modules(monkeypatch, "0.0.0+fake")
    cfg = {
        "mhcflurry_model_version": "9.9.9",
        "alleles": ["HLA-A*01:01", "HLA-A*02:01", "HLA-A*03:01", "HLA-B*07:02"],
        "antigens": ["EBV", "HPV16"],
    }
    monkeypatch.setattr(cli, "_read_config", lambda: cfg)

    assert cli.cmd_info(argparse.Namespace()) == 0
    out = capsys.readouterr().out

    assert "  torch           : 0.0-fake" in out
    assert "  CUDA            : not available" in out
    assert "  feature_mode    : not set" in out
    assert "  allele panel    : 4 alleles (HLA-A*01:01, HLA-A*02:01, HLA-A*03:01...)" in out
    assert "  viruses/panels  : EBV, HPV16" in out
    assert "[WARNING] MHCflurry version mismatch:" in out
    assert "    config pins   : 9.9.9" in out
    assert "    installed     : 0.0.0+fake" in out


def test_info_without_config_skips_the_pin_check(monkeypatch, capsys):
    # pins behaviour, not correctness
    _fake_modules(monkeypatch, "0.0.0+fake")
    monkeypatch.setattr(cli, "_read_config", lambda: {})

    assert cli.cmd_info(argparse.Namespace()) == 0
    out = capsys.readouterr().out

    assert "  config          : config.yaml not found" in out
    assert "feature_mode" not in out
    assert "mismatch" not in out
