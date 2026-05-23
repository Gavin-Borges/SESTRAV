import json

import pandas as pd
import pytest

from functions.stage4_immunogenicity_scoring import score_immunogenicity
from src.final_validation_report import run_final_validation
from src.gold_standard import full_validation_report, validate_stage4


def test_stage4_freeze_mode_blocks_prototype_fallback():
    df = pd.DataFrame(
        {
            "peptide": ["CLGGLLTMV", "RAKFKQLL"],
            "binding_score": [0.9, 0.2],
            "presentation_score": [0.92, 0.21],
        }
    )
    with pytest.raises(RuntimeError, match="prototype inline classifier fallback is disabled"):
        score_immunogenicity(
            df,
            proteome_id="TEST",
            model_path=None,
            freeze_mode=True,
            calibrate=False,
        )


def test_full_validation_report_rejects_mixed_stems_in_freeze_mode(tmp_path):
    # Canonical and legacy stems both present for EBV, which freeze mode must reject.
    pd.DataFrame({"peptide": ["CLGGLLTMV"]}).to_csv(
        tmp_path / "EBV_B95_8_panel8_peptides.csv", index=False
    )
    pd.DataFrame({"peptide": ["CLGGLLTMV"]}).to_csv(
        tmp_path / "EBV_8_FASTAs_binding.csv", index=False
    )

    with pytest.raises(RuntimeError, match="Mixed output stems detected"):
        full_validation_report(str(tmp_path), strict_stems=True)


def test_validate_stage4_reports_missing_score_column(tmp_path, capsys):
    ranked_path = tmp_path / "ranked.csv"
    pd.DataFrame({"peptide": ["CLGGLLTMV"], "rank": [1]}).to_csv(ranked_path, index=False)

    df, _ = validate_stage4(str(ranked_path), virus="EBV", top_pct=25, require_score_column=False)
    captured = capsys.readouterr()

    assert df.empty
    assert "Skipping Stage 4" in captured.out


def test_final_validation_publish_is_atomic_on_failure(tmp_path, monkeypatch):
    def _fake_gs(*args, **kwargs):
        return pd.DataFrame(
            {
                "peptide": ["CLGGLLTMV"],
                "virus": ["EBV"],
                "stage1_found": [True],
                "stage2_found": [True],
                "stage2_strong_binder": [True],
            }
        )

    def _fake_baseline(*args, **kwargs):
        return pd.DataFrame(
            {
                "method": ["RF (SESTRAV)"],
                "virus": ["Combined"],
                "n_peptides": [1],
                "gs_found": [1],
                "gs_in_top_10pct": [1],
                "gs_in_top_25pct": [1],
                "gs_recovery_top10": [1.0],
                "gs_recovery_top25": [1.0],
                "mean_rank_pct": [1.0],
                "median_rank_pct": [1.0],
            }
        )

    def _failing_h2(*args, **kwargs):
        raise RuntimeError("simulated h2 failure")

    monkeypatch.setattr("src.final_validation_report.full_validation_report", _fake_gs)
    monkeypatch.setattr("src.final_validation_report.compare_methods", _fake_baseline)
    monkeypatch.setattr("src.final_validation_report.run_h2_tier_a", _failing_h2)

    with pytest.raises(RuntimeError, match="simulated h2 failure"):
        run_final_validation(
            results_dir=str(tmp_path),
            model_dir="models",
            data_path=str(tmp_path / "missing_data.csv"),
            binding_matrix_path=str(tmp_path / "missing_binding.csv"),
            model_path=str(tmp_path / "missing_model.joblib"),
            freeze_mode=False,
        )

    assert not (tmp_path / "gold_standard_validation.csv").exists()
    assert not (tmp_path / "baseline_comparison.csv").exists()
    status_path = tmp_path / "freeze_status.json"
    assert status_path.exists()
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["valid"] is False
