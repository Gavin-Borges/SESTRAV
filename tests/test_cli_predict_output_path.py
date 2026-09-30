"""Regression coverage for ``sestrav predict --output`` stage artifacts."""

from argparse import Namespace
from pathlib import Path

import pandas as pd

from functions import stage1_peptide_generation as stage1
from functions import stage2_mhc_binding_prediction as stage2
from functions import stage3_tcr_feature_extraction as stage3
from functions import stage4_immunogenicity_scoring as stage4
from src import cli


class _FakePredictor:
    @classmethod
    def load(cls):
        return cls()

    def predict(self, peptides, alleles, verbose=0):
        del verbose
        return pd.DataFrame(
            {
                "peptide": peptides,
                "allele": [alleles[0]] * len(peptides),
                "affinity": [50.0] * len(peptides),
                "presentation_score": [0.9] * len(peptides),
                "presentation_percentile": [1.0] * len(peptides),
            }
        )


def test_stage1_writes_under_requested_output(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">protein\nACDEFGHIK\n", encoding="utf-8")
    output_dir = tmp_path / "requested"

    stage1.generate_peptides(fasta, "panel", peptide_lengths=[8], output_dir=output_dir)

    assert (output_dir / "panel_peptides.csv").is_file()
    assert not (tmp_path / "results").exists()


def test_stage2_writes_under_requested_output(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stage2, "Class1PresentationPredictor", _FakePredictor)
    output_dir = tmp_path / "requested"
    peptides = pd.DataFrame({"peptide": ["ACDEFGHI"], "protein_id": ["protein"]})

    stage2.predict_binding(
        peptides,
        "panel",
        alleles=["HLA-A*02:01"],
        output_dir=output_dir,
    )

    assert (output_dir / "panel_binding.csv").is_file()
    assert not (tmp_path / "results").exists()


def test_stage3_writes_under_requested_output(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "requested"
    features = pd.DataFrame({"peptide": ["ACDEFGHI"], "presentation_score": [0.9]})
    monkeypatch.setattr(stage3, "compute_features_for_dataset", lambda *args, **kwargs: features)

    stage3.extract_tcr_features(features, "panel", output_dir=output_dir)

    assert (output_dir / "panel_features.csv").is_file()
    assert not (tmp_path / "results").exists()


def test_cmd_predict_threads_output_to_all_writing_stages(monkeypatch, tmp_path):
    fasta = tmp_path / "input.fasta"
    model = tmp_path / "model.joblib"
    fasta.write_text(">protein\nACDEFGHIK\n", encoding="utf-8")
    model.write_bytes(b"stub")
    output_dir = tmp_path / "requested"
    seen = []
    frame = pd.DataFrame({"peptide": ["ACDEFGHI"], "immunogenicity_score": [0.8]})
    # If cmd_predict carries a model-data precondition, stub it so this test
    # exercises output threading rather than that precondition, returning a path
    # because the check refuses on None.
    #
    # The name is READ OFF src/cli.py, not guessed. This guard previously named
    # _require_mhcflurry_model_data, which exists nowhere, so hasattr() was always
    # False and the stub never applied. That is invisible while cmd_predict has no
    # precondition, and fails this test wherever model data is absent - which
    # includes CI - once it has one.
    if hasattr(cli, "_mhcflurry_model_data_path"):
        monkeypatch.setattr(cli, "_mhcflurry_model_data_path", lambda: "stubbed-models-dir")

    def fake_stage1(*args, output_dir, **kwargs):
        seen.append(("stage1", Path(output_dir)))
        return frame

    def fake_stage2(*args, output_dir, **kwargs):
        seen.append(("stage2", Path(output_dir)))
        return frame

    def fake_stage3(*args, output_dir, **kwargs):
        seen.append(("stage3", Path(output_dir)))
        return frame

    def fake_stage4(*args, output_dir, **kwargs):
        seen.append(("stage4", Path(output_dir)))
        return frame, object()

    monkeypatch.setattr(stage1, "generate_peptides", fake_stage1)
    monkeypatch.setattr(stage2, "predict_binding", fake_stage2)
    monkeypatch.setattr(stage3, "extract_tcr_features", fake_stage3)
    monkeypatch.setattr(stage4, "score_immunogenicity", fake_stage4)

    args = Namespace(
        fasta=str(fasta),
        model=str(model),
        output=str(output_dir),
        lengths=[8],
        alleles=["HLA-A*02:01"],
        freeze_mode=False,
        conformal=False,
        conformal_calibrator=None,
        virus=None,
        per_virus_calibration_dir=None,
    )

    assert cli.cmd_predict(args) == 0
    assert seen == [(name, output_dir) for name in ("stage1", "stage2", "stage3", "stage4")]
    assert not (tmp_path / "results").exists()
