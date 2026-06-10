import os
import subprocess
import pandas as pd
import pytest
import yaml

@pytest.fixture
def temp_config(tmp_path):
    config_content = """
dataset_governance:
  qc_thresholds:
    min_peptide_yield: 5
    max_conflict_ratio: 0.15
    max_null_allele_fraction: 0.50
    class_ratio_bounds: [1.5, 4.0]
freeze_mode: false
"""
    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        f.write(config_content)
    return path

@pytest.fixture
def valid_df():
    return pd.DataFrame({
        "peptide": ["ACDEFGHIK", "LMNPQRSTV", "WYACDEFGH", "ACDEFGHIKL", "LMNPQRSTVY"],
        "label": [1, 0, 1, 0, 1],
        "allele": ["HLA-A*02:01", "HLA-B*08:01", "HLA-A*02:01", None, "HLA-A*11:01"]
    })

def test_qc_gate_valid(tmp_path, temp_config, valid_df):
    dataset_path = tmp_path / "valid.csv"
    valid_df.to_csv(dataset_path, index=False)
    
    result = subprocess.run(
        ["python", "scripts/data_qc_gate.py", "--dataset", str(dataset_path), "--config", str(temp_config)],
        capture_output=True,
        text=True
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"QC gate failed on valid dataset: {output}"
    assert "All dataset QC gates passed successfully." in output

def test_qc_gate_length_outlier(tmp_path, temp_config, valid_df):
    # Add a 7-mer length outlier
    invalid_df = valid_df.copy()
    invalid_df.loc[0, "peptide"] = "ACDEFGH" # 7-mer
    
    dataset_path = tmp_path / "length_outlier.csv"
    invalid_df.to_csv(dataset_path, index=False)
    
    quarantine_path = tmp_path / "quarantine.csv"
    result = subprocess.run(
        ["python", "scripts/data_qc_gate.py", "--dataset", str(dataset_path), "--config", str(temp_config), "--quarantine", str(quarantine_path)],
        capture_output=True,
        text=True
    )
    
    # It should pass if we drop row-level outliers but fail if length_and_composition_valid fails,
    # wait: let's verify if the length check causes the gate to fail overall.
    # In scripts/data_qc_gate.py:
    # "length_and_composition_valid": len(indices_to_drop) == 0,
    # So if there are outliers, that check is False, and the gate fails (success = False).
    output = result.stdout + result.stderr
    assert result.returncode == 1, f"QC gate did not fail on length outlier: {output}"
    assert "Dataset QC gate FAILED on one or more admissibility checks." in output
    
    # Check quarantine output
    assert os.path.exists(quarantine_path)
    q_df = pd.read_csv(quarantine_path)
    assert len(q_df) == 1
    assert "Peptide length 7 outside valid 8-11mer window" in q_df.loc[0, "qc_failure_reason"]

def test_qc_gate_non_canonical_aa(tmp_path, temp_config, valid_df):
    invalid_df = valid_df.copy()
    invalid_df.loc[0, "peptide"] = "ACDEFGHIX" # 'X' is invalid
    
    dataset_path = tmp_path / "non_canonical.csv"
    invalid_df.to_csv(dataset_path, index=False)
    
    result = subprocess.run(
        ["python", "scripts/data_qc_gate.py", "--dataset", str(dataset_path), "--config", str(temp_config)],
        capture_output=True,
        text=True
    )
    assert result.returncode == 1, "QC gate did not fail on non-canonical AA"

def test_qc_gate_missing_metadata(tmp_path, temp_config, valid_df):
    invalid_df = valid_df.copy()
    invalid_df.loc[0, "peptide"] = None # missing sequence
    
    dataset_path = tmp_path / "missing_meta.csv"
    invalid_df.to_csv(dataset_path, index=False)
    
    result = subprocess.run(
        ["python", "scripts/data_qc_gate.py", "--dataset", str(dataset_path), "--config", str(temp_config)],
        capture_output=True,
        text=True
    )
    assert result.returncode == 1, "QC gate did not fail on missing peptide"

def test_qc_gate_duplicate_conflict(tmp_path, temp_config, valid_df):
    # Add duplicate with conflicting label
    # valid_df has 5 rows. Let's add 2 identical peptides with conflicting labels.
    conflict_df = pd.concat([
        valid_df,
        pd.DataFrame({
            "peptide": ["ACDEFGHIK", "ACDEFGHIK"],
            "label": [1, 0],
            "allele": ["HLA-A*02:01", "HLA-A*02:01"]
        })
    ], ignore_index=True)
    
    dataset_path = tmp_path / "duplicate_conflict.csv"
    conflict_df.to_csv(dataset_path, index=False)
    
    result = subprocess.run(
        ["python", "scripts/data_qc_gate.py", "--dataset", str(dataset_path), "--config", str(temp_config)],
        capture_output=True,
        text=True
    )
    # Conflict ratio threshold is 0.15. We have 1 conflict out of 5 unique groups (20%), which should fail.
    assert result.returncode == 1, f"QC gate did not fail on duplicate conflicts (status={result.returncode}, err={result.stderr})"

def test_qc_gate_null_allele_fraction(tmp_path, temp_config, valid_df):
    # Null allele threshold is 0.50 (50%).
    # valid_df has 5 rows. Let's make 3 of them have null allele.
    invalid_df = valid_df.copy()
    invalid_df.loc[0, "allele"] = None
    invalid_df.loc[1, "allele"] = None
    invalid_df.loc[3, "allele"] = None # 3 out of 5 is 60% null, which exceeds 50%
    
    dataset_path = tmp_path / "high_null_allele.csv"
    invalid_df.to_csv(dataset_path, index=False)
    
    result = subprocess.run(
        ["python", "scripts/data_qc_gate.py", "--dataset", str(dataset_path), "--config", str(temp_config)],
        capture_output=True,
        text=True
    )
    assert result.returncode == 1, "QC gate did not fail on high null allele fraction"

def test_qc_gate_class_ratio(tmp_path, temp_config, valid_df):
    # class ratio bounds: [1.5, 4.0].
    # Let's make ratio out of bounds by having too many negatives (e.g. 4 negatives, 1 positive)
    invalid_df = valid_df.copy()
    invalid_df["label"] = [1, 0, 0, 0, 0] # Ratio: 1/4 = 0.25 (below 1.5)
    
    dataset_path = tmp_path / "bad_class_ratio.csv"
    invalid_df.to_csv(dataset_path, index=False)
    
    result = subprocess.run(
        ["python", "scripts/data_qc_gate.py", "--dataset", str(dataset_path), "--config", str(temp_config)],
        capture_output=True,
        text=True
    )
    assert result.returncode == 1, "QC gate did not fail on out-of-bounds class ratio"

def test_qc_gate_insufficient_yield(tmp_path, temp_config, valid_df):
    # min_peptide_yield is 5. Let's make it have only 4 rows
    invalid_df = valid_df.iloc[:4].copy()
    
    dataset_path = tmp_path / "low_yield.csv"
    invalid_df.to_csv(dataset_path, index=False)
    
    result = subprocess.run(
        ["python", "scripts/data_qc_gate.py", "--dataset", str(dataset_path), "--config", str(temp_config)],
        capture_output=True,
        text=True
    )
    assert result.returncode == 1, "QC gate did not fail on low yield"
