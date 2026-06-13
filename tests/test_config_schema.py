import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError
from src.core.config import SestravConfig

def test_config_loads_valid_yaml(tmp_path):
    # Minimal valid config
    valid_yaml = """
    antigens: ["HPV16"]
    proteome_files: {"HPV16": "data.fasta"}
    alleles: ["HLA-A*02:01"]
    peptide_lengths: [9]
    binding_backend: mhcflurry
    feature_mode: 30
    model_path: models/rf_30feature.joblib
    freeze_mode: false
    dataset_governance:
      current_version: "2.0"
      versions: {"2.0": "IEDB"}
      provenance:
        timestamp: "2026-05-21T00:00:00Z"
        source_databases: ["IEDB"]
        negative_sampling_strategy: "decoy"
        checksum: "abc123def"
      qc_thresholds:
        min_peptide_yield: 500
        max_conflict_ratio: 0.15
        max_null_allele_fraction: 1.0
        class_ratio_bounds: [1.5, 4.0]
      require_checksum_match_in_freeze_mode: true
    dataset_mode: alpha
    dataset_version: "2.0"
    output_dir: results
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(valid_yaml)

    config = SestravConfig.load(config_file)
    assert config.feature_mode == 30
    assert len(config.alleles) == 1
    assert config.model_path == Path("models/rf_30feature.joblib")


def test_config_fails_missing_required(tmp_path):
    invalid_yaml = """
    antigens: ["HPV16"]
    # Missing model_path and dataset_governance
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(invalid_yaml)

    with pytest.raises(ValidationError):
        SestravConfig.load(config_file)
