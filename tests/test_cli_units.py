"""Targeted unit tests for src.cli helpers and cmd_benchmark.

Covers _read_config (with existing config.yaml), _require_file (missing/present),
and cmd_benchmark end-to-end with a minimal predictions CSV - exercising
auto-detect score column, gold-standard label building, and report writing.
"""

import argparse
import os
import pytest

from src.cli import _read_config, _require_file, cmd_benchmark
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES


# ---------------------------------------------------------------------------
# _read_config
# ---------------------------------------------------------------------------


def test_read_config_returns_dict():
    cfg = _read_config()
    assert isinstance(cfg, dict)


def test_read_config_contains_known_keys():
    cfg = _read_config()
    assert "feature_mode" in cfg
    assert "alleles" in cfg
    assert cfg["feature_mode"] == 31


# ---------------------------------------------------------------------------
# _require_file
# ---------------------------------------------------------------------------


def test_require_file_raises_on_missing(tmp_path):
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        _require_file(str(tmp_path / "absent.csv"), parser, "--predictions")


def test_require_file_passes_when_file_exists(tmp_path):
    p = tmp_path / "present.csv"
    p.write_text("peptide,score\nCLGGLLTMV,0.9\n")
    parser = argparse.ArgumentParser()
    _require_file(str(p), parser, "--predictions")  # must not raise


# ---------------------------------------------------------------------------
# cmd_benchmark - helpers
# ---------------------------------------------------------------------------


def _write_predictions(tmp_path, n_gs=5, n_neg=5, score_col="immunogenicity_score"):
    """Write a minimal predictions CSV with gold standard and non-GS peptides."""
    gs_peptides = list(GOLD_STANDARD_EPITOPES)[:n_gs]
    neg_peptides = ["AAAAAAAA", "MMMMMMMM", "FFFFFFFF", "GGGGGGGG", "HHHHHHHH"][:n_neg]
    rows = [f"{p},{0.9 - i * 0.05}" for i, p in enumerate(gs_peptides)]
    rows += [f"{p},{0.1 + i * 0.02}" for i, p in enumerate(neg_peptides)]
    p = tmp_path / "preds.csv"
    p.write_text(f"peptide,{score_col}\n" + "\n".join(rows) + "\n")
    return str(p)


def test_cmd_benchmark_returns_zero(tmp_path):
    pred_path = _write_predictions(tmp_path)
    args = argparse.Namespace(
        predictions=pred_path,
        score_column=None,
        output=None,
    )
    rc = cmd_benchmark(args)
    assert rc == 0


def test_cmd_benchmark_auto_detects_score_column(tmp_path, capsys):
    pred_path = _write_predictions(tmp_path, score_col="immunogenicity_score")
    args = argparse.Namespace(
        predictions=pred_path,
        score_column=None,
        output=None,
    )
    cmd_benchmark(args)
    out = capsys.readouterr().out
    assert "immunogenicity_score" in out


def test_cmd_benchmark_explicit_score_column(tmp_path):
    pred_path = _write_predictions(tmp_path, score_col="probability")
    args = argparse.Namespace(
        predictions=pred_path,
        score_column="probability",
        output=None,
    )
    rc = cmd_benchmark(args)
    assert rc == 0


def test_cmd_benchmark_writes_report(tmp_path):
    pred_path = _write_predictions(tmp_path)
    report_path = str(tmp_path / "report.md")
    args = argparse.Namespace(
        predictions=pred_path,
        score_column=None,
        output=report_path,
    )
    rc = cmd_benchmark(args)
    assert rc == 0
    assert os.path.isfile(report_path)
    content = open(report_path).read()
    assert "SESTRAV Benchmark Report" in content
    assert "auc_pr" in content


def test_cmd_benchmark_missing_score_column_returns_one(tmp_path):
    pred_path = _write_predictions(tmp_path, score_col="immunogenicity_score")
    args = argparse.Namespace(
        predictions=pred_path,
        score_column="nonexistent_col",
        output=None,
    )
    rc = cmd_benchmark(args)
    assert rc == 1


def test_cmd_benchmark_prints_metrics(tmp_path, capsys):
    pred_path = _write_predictions(tmp_path)
    args = argparse.Namespace(
        predictions=pred_path,
        score_column=None,
        output=None,
    )
    cmd_benchmark(args)
    out = capsys.readouterr().out
    assert "AUC-PR" in out
    assert "AUC-ROC" in out
