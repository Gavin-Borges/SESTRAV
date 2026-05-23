import os
import subprocess
import pandas as pd
import pytest

@pytest.fixture
def temp_dataset(tmp_path):
    # Create a valid dataset
    df = pd.DataFrame({
        "peptide": ["ACDEFGHIK", "LMNPQRSTV", "WYACDEFGH"],
        "label": [1, 0, 1]
    })
    path = tmp_path / "valid_dataset.csv"
    df.to_csv(path, index=False)
    return path

@pytest.fixture
def temp_config(tmp_path):
    # Create a dummy config
    config_content = """
freeze_mode: false
dataset_governance:
  require_checksum_match_in_freeze_mode: true
  provenance:
    checksum: "pending"
"""
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        f.write(config_content)
    return path

def test_qc_script_valid(temp_dataset, temp_config):
    # Run the QC script as a subprocess
    result = subprocess.run(
        ["python", "src/data_curation_qc.py", "--check-dataset", str(temp_dataset), "--config", str(temp_config)],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"QC script failed unexpectedly: {result.stderr}"
    assert "All strict dataset QC gates passed successfully" in result.stderr or "All strict dataset QC gates passed successfully" in result.stdout

def test_qc_script_invalid_amino_acids(tmp_path, temp_config):
    df = pd.DataFrame({
        "peptide": ["ACDEFGHIK", "XYZPQRSTV"],  # X, Y, Z (X and Z are invalid)
        "label": [1, 0]
    })
    path = tmp_path / "invalid_aa_dataset.csv"
    df.to_csv(path, index=False)
    
    result = subprocess.run(
        ["python", "src/data_curation_qc.py", "--check-dataset", str(path), "--config", str(temp_config)],
        capture_output=True,
        text=True
    )
    assert result.returncode != 0
    assert "Found 1 peptides with non-canonical amino acids" in result.stderr

def test_qc_script_duplicates(tmp_path, temp_config):
    df = pd.DataFrame({
        "peptide": ["ACDEFGHIK", "ACDEFGHIK"],
        "label": [1, 1]
    })
    path = tmp_path / "dup_dataset.csv"
    df.to_csv(path, index=False)
    
    result = subprocess.run(
        ["python", "src/data_curation_qc.py", "--check-dataset", str(path), "--config", str(temp_config)],
        capture_output=True,
        text=True
    )
    assert result.returncode != 0
    assert "Dataset contains identical peptide-label duplicates" in result.stderr

def test_qc_script_conflicting_labels(tmp_path, temp_config):
    df = pd.DataFrame({
        "peptide": ["ACDEFGHIK", "ACDEFGHIK"],
        "label": [1, 0]
    })
    path = tmp_path / "conflict_dataset.csv"
    df.to_csv(path, index=False)
    
    result = subprocess.run(
        ["python", "src/data_curation_qc.py", "--check-dataset", str(path), "--config", str(temp_config)],
        capture_output=True,
        text=True
    )
    assert result.returncode != 0
    assert "Dataset contains conflicting labels" in result.stderr
