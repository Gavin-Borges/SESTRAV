"""run_pipeline must forward config.freeze_mode into Stage 4.

`freeze_mode` is a required field on SestravConfig and `config.yaml` ships it as
`true`. The Snakemake path honours it: `pipeline.smk` reads it and
`scripts/stage4.py` passes it through. The plain-Python orchestrator did not.
`run_pipeline` loads the same config and reads six other fields off it
(`alleles`, `peptide_lengths`, `model_path`, `mc_dropout`, `calibration_path`,
`thresholds_path`) while omitting this one, so `score_immunogenicity` fell back
to its own `freeze_mode=False` default and the guardrail was silently off.

What that guardrail does, from `functions/stage4_immunogenicity_scoring.py`:
when no trained model produced a score, freeze mode raises rather than quietly
substituting a prototype inline RandomForest. So a release-grade run configured
for freeze mode could still emit prototype-classifier scores through this entry
point, and README describes freeze mode as forbidding exactly that.

No test imported `pipeline.py` at all before this file, which is why the drop
survived. The stages are stubbed because the assertion is purely about argument
forwarding, not about any stage's behaviour.
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def pipeline_module():
    import pipeline

    return pipeline


class _Registry:
    def resolve_model(self, name):
        return f"/models/{name}"


class _Config:
    """Minimal stand-in exposing only what run_pipeline reads."""

    def __init__(self, freeze_mode: bool):
        self.alleles = ["HLA-A*02:01"]
        self.peptide_lengths = [9]
        self.model_path = type("P", (), {"name": "rf.joblib"})()
        self.mc_dropout = False
        self.calibration_path = None
        self.thresholds_path = None
        self.freeze_mode = freeze_mode
        self.output_dir = "results"


def _stub_stages(monkeypatch, module, captured):
    frame = pd.DataFrame({"peptide": ["CLGGLLTMV"]})
    monkeypatch.setattr(module, "generate_peptides", lambda *a, **k: frame)
    monkeypatch.setattr(module, "predict_binding", lambda *a, **k: frame)
    monkeypatch.setattr(module, "extract_tcr_features", lambda *a, **k: frame)
    monkeypatch.setattr(module, "plot_immunogenicity_scores", lambda *a, **k: None)

    def _score(features_df, proteome_id, **kwargs):
        captured.update(kwargs)
        return frame, None

    monkeypatch.setattr(module, "score_immunogenicity", _score)


@pytest.mark.parametrize("configured", [True, False])
def test_run_pipeline_forwards_configured_freeze_mode(monkeypatch, pipeline_module, configured):
    """Both values are asserted deliberately.

    Pinning only True would also pass against a hardcoded `freeze_mode=True`,
    which would be a different defect in the opposite direction.
    """
    captured: dict = {}
    _stub_stages(monkeypatch, pipeline_module, captured)

    pipeline_module.run_pipeline(
        "TEST",
        "unused.fasta",
        _Config(freeze_mode=configured),
        _Registry(),
    )

    assert "freeze_mode" in captured, "run_pipeline did not pass freeze_mode to Stage 4 at all"
    assert captured["freeze_mode"] is configured, (
        f"config.freeze_mode={configured} was not forwarded; got {captured['freeze_mode']!r}"
    )
