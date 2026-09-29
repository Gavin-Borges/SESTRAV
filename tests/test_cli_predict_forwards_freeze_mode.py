"""`sestrav predict` must forward config.freeze_mode into Stage 4.

This is the same defect `tests/test_pipeline_forwards_freeze_mode.py` records for
`pipeline.run_pipeline`, in a second entry point. `cmd_predict` hardcoded
`freeze_mode=False` while `config.yaml` ships `freeze_mode: true`, so the
guardrail was off through the CLI no matter how the repository was configured.

The stages are stubbed because the assertion is purely about argument
forwarding, not about any stage behaviour.
"""

from __future__ import annotations

import argparse

import pandas as pd
import pytest


@pytest.fixture
def cli_module():
    from src import cli

    return cli


def _stub_stages(monkeypatch, captured):
    frame = pd.DataFrame({"peptide": ["CLGGLLTMV"], "immunogenicity_score": [0.5]})

    import functions.stage1_peptide_generation as s1
    import functions.stage2_mhc_binding_prediction as s2
    import functions.stage3_tcr_feature_extraction as s3
    import functions.stage4_immunogenicity_scoring as s4

    monkeypatch.setattr(s1, "generate_peptides", lambda *a, **k: frame)
    monkeypatch.setattr(s2, "predict_binding", lambda *a, **k: frame)
    monkeypatch.setattr(s3, "extract_tcr_features", lambda *a, **k: frame)

    def _score(features_df, proteome_id, **kwargs):
        captured.update(kwargs)
        return frame, None

    monkeypatch.setattr(s4, "score_immunogenicity", _score)


def _args(tmp_path, freeze_mode):
    fasta = tmp_path / "p.fasta"
    fasta.write_text(">x\nCLGGLLTMV\n", encoding="utf-8")
    model = tmp_path / "m.joblib"
    model.write_bytes(b"stub")
    return argparse.Namespace(
        fasta=str(fasta),
        model=str(model),
        output=str(tmp_path / "out"),
        alleles=None,
        lengths=None,
        conformal=True,
        conformal_calibrator=None,
        virus=None,
        per_virus_calibration_dir=None,
        freeze_mode=freeze_mode,
    )


@pytest.mark.parametrize("configured", [True, False])
def test_cmd_predict_forwards_configured_freeze_mode(monkeypatch, cli_module, tmp_path, configured):
    """Both values are asserted deliberately.

    Pinning only True would also pass against a hardcoded `freeze_mode=True`,
    which would be a different defect in the opposite direction.
    """
    captured: dict = {}
    _stub_stages(monkeypatch, captured)
    monkeypatch.setattr(cli_module, "_read_config", lambda: {"freeze_mode": configured})

    cli_module.cmd_predict(_args(tmp_path, freeze_mode=None))

    assert "freeze_mode" in captured, "cmd_predict did not pass freeze_mode to Stage 4 at all"
    assert captured["freeze_mode"] is configured, (
        f"config freeze_mode={configured} was not forwarded; got {captured['freeze_mode']!r}"
    )


@pytest.mark.parametrize("override", [True, False])
def test_cmd_predict_flag_overrides_config(monkeypatch, cli_module, tmp_path, override):
    """An explicit --freeze-mode/--no-freeze-mode beats config.yaml."""
    captured: dict = {}
    _stub_stages(monkeypatch, captured)
    monkeypatch.setattr(cli_module, "_read_config", lambda: {"freeze_mode": not override})

    cli_module.cmd_predict(_args(tmp_path, freeze_mode=override))

    assert captured["freeze_mode"] is override
