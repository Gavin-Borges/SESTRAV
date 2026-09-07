"""Tri-state release-gate tests for bias/skew finalization."""

import json

import pandas as pd

from src.bias_skew_finalization import _gate_label, _gate_status


def _measured_gate_inputs(tmp_path):
    subgroup_csv = tmp_path / "subgroups.csv"
    pd.DataFrame(
        [{"subgroup_key": "virus", "auc_pr": 0.75}]
    ).to_csv(subgroup_csv, index=False)
    threshold_json = tmp_path / "thresholds.json"
    threshold_json.write_text(
        json.dumps({"overall_precision": 0.70, "overall_recall": 0.70}),
        encoding="utf-8",
    )
    return subgroup_csv, threshold_json


def _gates(tmp_path, sensitivity_delta_csv):
    subgroup_csv, threshold_json = _measured_gate_inputs(tmp_path)
    return _gate_status(
        bias_summary={"n_total": 10, "raw_n_records": 10},
        subgroup_csv=str(subgroup_csv),
        threshold_json=str(threshold_json),
        sensitivity_delta_csv=str(sensitivity_delta_csv),
    )


def test_missing_sensitivity_deltas_are_unmeasured_not_passed(tmp_path):
    gates = _gates(tmp_path, tmp_path / "missing.csv")
    assert gates["gold_standard_not_brittle"] is None
    assert gates["gold_standard_not_brittle"] is not True
    assert gates["all_passed"] is False
    assert _gate_label(gates["gold_standard_not_brittle"]).startswith("Unmeasured")


def test_empty_sensitivity_deltas_are_unmeasured_not_passed(tmp_path):
    deltas_csv = tmp_path / "empty.csv"
    pd.DataFrame(columns=["delta_recovery_top25"]).to_csv(deltas_csv, index=False)
    gates = _gates(tmp_path, deltas_csv)
    assert gates["gold_standard_not_brittle"] is None
    assert gates["all_passed"] is False


def test_measured_sensitivity_deltas_can_pass(tmp_path):
    deltas_csv = tmp_path / "passing.csv"
    pd.DataFrame({"delta_recovery_top25": [-0.05, -0.10]}).to_csv(
        deltas_csv, index=False
    )
    gates = _gates(tmp_path, deltas_csv)
    assert gates["gold_standard_not_brittle"] is True
    assert gates["all_passed"] is True


def test_measured_sensitivity_deltas_can_fail(tmp_path):
    deltas_csv = tmp_path / "failing.csv"
    pd.DataFrame({"delta_recovery_top25": [-0.05, -0.25]}).to_csv(
        deltas_csv, index=False
    )
    gates = _gates(tmp_path, deltas_csv)
    assert gates["gold_standard_not_brittle"] is False
    assert gates["all_passed"] is False
